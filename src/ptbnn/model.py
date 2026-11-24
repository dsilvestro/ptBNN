"""BNN posterior (network + prior + likelihood) and MAP initialisation."""

import math

import numpy as np
import torch

from .likelihood import get_likelihood
from .network import BatchedMLP


class BNN:
    """Posterior of a Bayesian neural network: Normal(0, p_scale) prior on all
    weights and the given likelihood.

    X: [N, F] inputs. Y: targets, [N, S] (or [N] labels for 'categorical');
    NaN = missing.
    likelihood: 'negbin', 'poisson', 'gaussian', 'categorical' or a
        ptbnn.likelihood.Likelihood instance (e.g. Gaussian(sd=0.5)).
    offset: [N, 1] or [N, S], added to the log mean (count likelihoods) or the
        mean (Gaussian).
    hidden_bias: biases in the hidden layers (True by default; npBNN's
        use_bias_node=-1 corresponds to hidden_bias=False, output_bias=True).
    agg: optional [N_obs, N] 0/1 matrix for aggregated counts. Then X holds one
        row per time step (e.g. day), the network predicts the rate per row,
        and the expected count of observation t is the sum of the rates of the
        rows with agg[t] == 1 (for the NB, p of an observation is the mean
        logit p of its rows). Y then has N_obs rows.
    """

    def __init__(self, X, Y, likelihood="negbin", n_nodes=(64, 32), p_scale=1.0, activation="relu",
                 offset=None, hidden_bias=True, output_bias=True,
                 device="cpu", dtype=torch.float64, agg=None, **lik_kwargs):
        self.device = torch.device(device)
        self.dtype = dtype
        self.lik = get_likelihood(likelihood, **lik_kwargs)
        X = np.asarray(X, dtype=float)
        n_targets = self.lik.n_targets(np.asarray(Y, dtype=float).reshape(len(Y), -1))
        self._n_targets = n_targets  # number of target columns (outputs or classes)
        self.data = self.lik.prepare(Y, self.device, dtype)
        self.Y, self.mask = self.data["Y"], self.data["mask"]
        self.nz = self.data.get("nz")
        n_obs = self.Y.shape[0]
        self.X = torch.tensor(X, dtype=dtype, device=self.device)
        if offset is not None and not self.lik.supports_offset:
            raise ValueError(f"{self.lik.name} likelihood does not take an offset")
        self.offset = None if offset is None else torch.tensor(
            np.asarray(offset, dtype=float).reshape(n_obs, -1), dtype=dtype, device=self.device)
        self.agg = self.agg_mean = None
        if agg is not None:
            if not self.lik.supports_agg:
                raise ValueError(f"{self.lik.name} likelihood does not support aggregation")
            agg = np.asarray(agg, dtype=float)
            self.agg = torch.tensor(agg, dtype=dtype, device=self.device)
            self.agg_mean = torch.tensor(agg / agg.sum(1, keepdims=True), dtype=dtype, device=self.device)
            if agg.shape[1] != len(X):
                raise ValueError("agg must have one column per row of X")
        elif len(X) != n_obs:
            raise ValueError("Y rows do not match the rows of X")
        if agg is not None and agg.shape[0] != n_obs:
            raise ValueError("Y rows do not match the rows of agg")
        self.net = BatchedMLP(X.shape[1], n_nodes, self.lik.n_output(n_targets), activation=activation,
                              hidden_bias=hidden_bias, output_bias=output_bias)
        self.p_scale = float(p_scale)
        self.n_params = self.net.n_params

    @property
    def n_targets(self):
        return self._n_targets

    def log_prior(self, theta):
        s = self.p_scale
        return (-0.5 * (theta / s).pow(2).sum(-1)
                - theta.shape[-1] * (math.log(s) + 0.5 * math.log(2 * math.pi)))

    def aggregate(self, out):
        """Row outputs [C, N_rows, n_out] -> observation outputs [C, N_obs, n_out]."""
        return self.lik.aggregate(out, self.agg, self.agg_mean)

    def log_lik(self, theta):
        out = self.net.forward(theta, self.X)
        if self.agg is not None:
            out = self.aggregate(out)
        return self.lik.log_lik(out, self.data, self.offset)

    def log_prob(self, theta):
        return self.log_lik(theta) + self.log_prior(theta)

    def log_prob_and_grad(self, theta):
        theta = theta.detach().requires_grad_(True)
        lp = self.log_prob(theta)
        (g,) = torch.autograd.grad(lp.sum(), theta)
        return lp.detach(), g.detach()

    @torch.no_grad()
    def predict(self, theta, X):
        """theta: [K, D] -> numpy network outputs [K, N, n_output]."""
        X = torch.as_tensor(np.asarray(X, dtype=float), dtype=self.dtype, device=self.device)
        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        outs = []
        for i in range(0, theta.shape[0], 64):
            outs.append(self.net.forward(theta[i:i + 64], X).cpu().numpy())
        return np.concatenate(outs, 0)

    @torch.no_grad()
    def predict_obs(self, theta):
        """Outputs at the observation level (aggregated if agg is set) for the
        training data: numpy [K, N_obs, n_output]."""
        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        outs = []
        for i in range(0, theta.shape[0], 64):
            out = self.net.forward(theta[i:i + 64], self.X)
            if self.agg is not None:
                out = self.aggregate(out)
            outs.append(out.cpu().numpy())
        return np.concatenate(outs, 0)


class NegBinBNN(BNN):
    """BNN with negative binomial likelihood (outputs [:S] log mean, [S:] logit p).
    Defaults as npBNN with use_bias_node=-1: no hidden biases, output bias."""

    def __init__(self, X, Y, n_nodes=(64, 32), p_scale=1.0, activation="relu",
                 offset=None, hidden_bias=False, output_bias=True,
                 device="cpu", dtype=torch.float64, agg=None):
        super().__init__(X, Y, "negbin", n_nodes=n_nodes, p_scale=p_scale, activation=activation,
                         offset=offset, hidden_bias=hidden_bias, output_bias=output_bias,
                         device=device, dtype=dtype, agg=agg)


# --------------------------------------------------------------------------
# MAP initialisation
# --------------------------------------------------------------------------

def find_map(model, theta0, n_steps=10000, lr=0.01, print_f=1000, verbose=True):
    """Maximise the log posterior of every chain in parallel with Adam
    (cosine learning-rate decay). Returns the best point found per chain."""
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, n_steps) / n_steps)) + 1e-3)
    n = model.Y.numel()
    best = theta.detach().clone()
    best_lp = torch.full((theta.shape[0],), -float("inf"), dtype=theta.dtype, device=theta.device)
    for it in range(n_steps):
        opt.zero_grad()
        lp = model.log_prob(theta)
        loss = -(torch.nan_to_num(lp, nan=-1e30)).sum() / n
        loss.backward()
        with torch.no_grad():
            better = lp.detach() > best_lp
            best[better] = theta.detach()[better]
            best_lp[better] = lp.detach()[better]
        opt.step()
        sched.step()
        if verbose and (it % print_f == 0 or it == n_steps - 1):
            print(f"  MAP {it:6d}  log post per chain: "
                  + " ".join(f"{v:.1f}" for v in lp.detach().cpu().tolist()))
    return best.detach()
