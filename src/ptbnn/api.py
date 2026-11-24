"""High-level interface: fit a BNN with one call and work with the posterior."""

import numpy as np
import pandas as pd
import torch

from .diagnostics import split_rhat_ess
from .hmc import run_hmc
from .model import BNN, find_map
from .utils import pick_device, pick_dtype


def fit(X, Y, likelihood="gaussian", nodes=(16, 8), chains=4, warmup=1000, samples=1000,
        leapfrog=50, target_accept=0.8, thin=1, p_scale=1.0, activation="relu", hidden_bias=True,
        offset=None, agg=None, map_steps=1000, device="cpu", dtype="auto", seed=0, logger=None,
        print_f=200, verbose=True, **lik_kwargs):
    """Fit a BNN by HMC and return a Posterior.

    X: [N, F] inputs. Y: [N, S] targets ([N] class labels for 'categorical').
    likelihood: 'gaussian', 'poisson', 'negbin', 'categorical' or a Likelihood
        instance; extra keyword arguments go to the likelihood (e.g. sd=0.1 for
        a Gaussian with fixed noise, n_classes=3 for 'categorical').
    nodes: hidden layer sizes. chains run in parallel as one batched computation.
    warmup / samples: HMC iterations per chain (step size and mass matrix are
        tuned during warmup); leapfrog: steps per iteration; thin: keep every
        thin-th draw.
    map_steps: Adam steps for the starting points (0 = start from small random weights).
    device: 'cpu', 'cuda', 'mps' or 'auto'; dtype: 'auto' (float64 on CPU,
        float32 on GPU), 'float32' or 'float64'.
    """
    torch.manual_seed(seed)
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model = BNN(X, Y, likelihood, n_nodes=nodes, p_scale=p_scale, activation=activation,
                hidden_bias=hidden_bias, offset=offset, agg=agg, device=dev, dtype=dt, **lik_kwargs)
    gen = torch.Generator().manual_seed(seed)
    theta0 = model.net.init_params(chains, init_std=0.1, generator=gen, dtype=dt, device=dev)
    if map_steps > 0:
        theta0 = find_map(model, theta0, n_steps=map_steps, print_f=max(1, map_steps // 2),
                          verbose=verbose)
    res = run_hmc(model.log_prob_and_grad, theta0, n_warmup=warmup, n_samples=samples,
                  n_leapfrog=leapfrog, target_accept=target_accept, thin=thin, seed=seed,
                  logger=logger, print_f=print_f, verbose=verbose)
    post = Posterior(model, res["samples"], res)
    if verbose:
        print(post.diagnostics().round(3).to_string())
    return post


class Posterior:
    """Posterior draws of a BNN.

    samples: [chains, draws, n_params]. predict() returns raw network outputs;
    predict_mean() the expected target (mean count, mean value or class
    probabilities); predict_sample() posterior predictive draws.
    """

    def __init__(self, model, samples, result=None):
        self.model = model
        self.lik = model.lik
        self.samples = np.asarray(samples)
        self.result = result or {}

    @property
    def n_chains(self):
        return self.samples.shape[0]

    def draws(self, n=None):
        """Flat draws [K, D]; n evenly spaced draws over all chains if given."""
        flat = self.samples.reshape(-1, self.samples.shape[-1])
        if n is not None and n < len(flat):
            flat = flat[np.linspace(0, len(flat) - 1, n).round().astype(int)]
        return flat

    def predict(self, X, n=None):
        """Network outputs [K, N, n_output] for inputs X (numpy)."""
        return self.model.predict(self.draws(n), X)

    def _to_tensor(self, a):
        return None if a is None else torch.as_tensor(np.asarray(a, dtype=float).reshape(len(a), -1))

    def predict_mean(self, X, offset=None, n=None):
        """Expected target per draw [K, N, S] (class probabilities [K, N, K_cls] for
        'categorical'); offset as in training (count and Gaussian likelihoods)."""
        out = torch.as_tensor(self.predict(X, n))
        return self.lik.mean(out, self._to_tensor(offset)).numpy()

    def predict_sample(self, X, offset=None, n=None, seed=0):
        """One posterior predictive draw of the target per posterior draw [K, N, S]
        ([K, N] labels for 'categorical')."""
        out = torch.as_tensor(self.predict(X, n))
        gen = torch.Generator().manual_seed(seed)
        return self.lik.sample(out, self._to_tensor(offset), generator=gen).numpy()

    def diagnostics(self):
        """R-hat and bulk ESS of the log-likelihood and of the likelihood's
        per-target summaries on the training data (weights are not identifiable)."""
        C, K, D = self.samples.shape
        th = torch.as_tensor(self.samples.reshape(C * K, D), dtype=self.model.dtype,
                             device=self.model.device)
        rows = {}
        with torch.no_grad():
            ll = torch.cat([self.model.log_lik(th[i:i + 64]) for i in range(0, len(th), 64)])
            rows["log_lik"] = ll.cpu().numpy().reshape(C, K)
            out = torch.cat([self.model.net.forward(th[i:i + 64], self.model.X)
                             for i in range(0, len(th), 64)])
            try:
                summaries = self.lik.summaries(out)
            except NotImplementedError:  # custom likelihood without summaries
                summaries = []
            for name, v in summaries:
                v = v.cpu().numpy().reshape(C, K, -1)
                for s in range(v.shape[-1]):
                    rows[f"{name}_{s + 1}"] = v[:, :, s]
        tab = pd.DataFrame({k: split_rhat_ess(v) for k, v in rows.items()}, index=["R-hat", "ESS"]).T
        if "divergences" in self.result:
            tab.attrs["divergences"] = int(np.sum(self.result["divergences"]))
        return tab

    def save(self, path):
        m = self.model
        torch.save({"samples": self.samples, "likelihood": self.lik,
                    "X": m.X.cpu().numpy(), "Y": m.Y.cpu().numpy(),
                    "mask": None if m.mask is None else m.mask.cpu().numpy(),
                    "offset": None if m.offset is None else m.offset.cpu().numpy(),
                    "agg": None if m.agg is None else m.agg.cpu().numpy(),
                    "nodes": m.net.n_nodes, "activation": m.net.activation, "p_scale": m.p_scale,
                    "hidden_bias": m.net.hidden_bias, "output_bias": m.net.output_bias,
                    "result": {k: v for k, v in self.result.items() if k != "samples"}}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        d = torch.load(path, weights_only=False, map_location="cpu")
        Y = d["Y"].astype(float)
        if d["mask"] is not None:
            Y = np.where(d["mask"] > 0, Y, np.nan)
        model = BNN(d["X"], Y, d["likelihood"], n_nodes=d["nodes"], p_scale=d["p_scale"],
                    activation=d["activation"], offset=d["offset"], agg=d["agg"],
                    hidden_bias=d["hidden_bias"], output_bias=d["output_bias"], device=device)
        return cls(model, d["samples"], d["result"])


def partial_dependence(post, X, feature, grid=None, steps=50, n=None, fn=None):
    """Partial dependence of the prediction on one input feature.

    For each grid value, sets column `feature` of X to that value in every row,
    and averages fn(network output) over the rows, separately for each
    posterior draw (fn defaults to the likelihood's expected value).

    Returns (grid [steps], values [K draws, steps, S]); summarise over draws
    for the posterior mean and credible bands.
    """
    X = np.asarray(X, dtype=float)
    if grid is None:
        grid = np.linspace(X[:, feature].min(), X[:, feature].max(), steps)
    vals = []
    for g in grid:
        Xg = X.copy()
        Xg[:, feature] = g
        out = torch.as_tensor(post.predict(Xg, n))
        v = post.lik.mean(out) if fn is None else fn(out)
        vals.append(np.asarray(v).mean(1))  # average over rows -> [K, S]
    return np.asarray(grid), np.stack(vals, 1)
