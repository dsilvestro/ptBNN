"""
Likelihoods. Each one maps the network output [C, N, n_output] (C chains) to
a log-likelihood per chain.

  NegativeBinomial  counts; outputs [:S] log mean, [S:] logit p (size n = p mu / (1-p))
  Poisson           counts; outputs [:S] log mean
  Gaussian          continuous; outputs [:S] mean and, with sd="learned",
                    [S:] log sd (heteroscedastic); otherwise a fixed sd
  Categorical       class labels 0..K-1; outputs K logits

Missing targets (NaN, or label < 0 for Categorical) are ignored. Count
likelihoods accept an offset on the log mean and window aggregation (the
expected count of an observation is the sum of the rates of several rows).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

LOG_R_SWITCH = math.log(100.0)


def lgamma_diff(y, log_r):
    """lgamma(y + r) - lgamma(r) - y * log r, as a function of log r, accurate
    for any r. (The y * log r term is removed so that it can be cancelled
    analytically against log(1-p) in the NB likelihood.)

    For r > 100 a Stirling expansion of the difference is used, which avoids
    subtracting two huge, nearly equal numbers (and never forms r itself).
    Each branch gets safe inputs so torch.where does not leak NaN gradients."""
    large = log_r > LOG_R_SWITCH
    # small/moderate r: lgamma(y+r) - lgamma(1+r) + log r  (= 0 for y = 0);
    # written this way so that r -> 0 (exp underflow) stays finite
    lr_s = torch.where(large, torch.zeros_like(log_r), log_r)
    r_s = torch.exp(lr_s)
    y0 = y == 0
    y_s = torch.where(y0, torch.ones_like(y), y)
    direct = torch.where(y0, torch.zeros_like(r_s),
                         torch.lgamma(y_s + r_s) - torch.lgamma(1 + r_s) + (1 - y_s) * lr_s)

    lr = torch.where(large, log_r, torch.full_like(log_r, LOG_R_SWITCH))
    inv_r = torch.exp(-lr)
    x = y * inv_r  # y / r
    # r * log1p(x) - y = y * (log1p(x)/x - 1), with a series for small x
    small = x < 1e-3
    x_d = torch.where(small, torch.ones_like(x), x)
    h = torch.where(small, -x / 2 + x * x / 3 - x ** 3 / 4, torch.log1p(x_d) / x_d - 1)
    asym = ((y - 0.5) * torch.log1p(x) + y * h
            - inv_r * x / (1 + x) / 12.0
            - inv_r ** 3 * ((1 + x) ** -3 - 1) / 360.0)
    return torch.where(large, asym, direct)


def log_softplus_neg(z):
    """log(softplus(-z)) = log(-log p) for p = sigmoid(z), stable for large z."""
    big = z > 15
    z_s = torch.where(big, torch.zeros_like(z), z)
    z_b = torch.where(big, z, torch.full_like(z, 15.0))
    return torch.where(big, -z_b - 0.5 * torch.exp(-z_b), torch.log(F.softplus(-z_s)))


def negbin_loglik(out, Y, mask, n_targets, offset=None, nz=None):
    """Negative binomial log-likelihood, summed over data, one value per chain.

    out:  [C, N, 2S] network output (log mean, logit p)
    Y:    [N, S] counts (0 where missing)
    mask: [N, S] 1 for observed, 0 for missing (or None)
    offset: [N, 1] or [N, S] added to the log mean (or None)
    nz:   flat indices of observed cells with y > 0 (computed if None)

    pmf = Gamma(y+r) / (Gamma(r) y!) p^r (1-p)^y  with r = p mu / (1-p)
    note: log r = log mu + logit p, so r is large when p -> 1 (Poisson limit).
    All terms are computed from log r so that they stay accurate in that limit.
    """
    eta = out[..., :n_targets]
    z = out[..., n_targets:]
    if offset is not None:
        eta = eta + offset
    log_r = eta + z
    if nz is None:
        obs = (Y > 0) if mask is None else (Y > 0) & (mask > 0)
        nz = torch.nonzero(obs.flatten()).squeeze(1)
    # r * log p = -exp(log r + log(softplus(-z)))  (-> -mu in the Poisson limit)
    # y * log r + y * log(1 - p) = y * (eta + z - softplus(z)) = y * (eta - softplus(-z))
    ll = -torch.exp(log_r + log_softplus_neg(z)) + Y * (eta - F.softplus(-z))
    if mask is not None:
        ll = ll * mask
    # lgamma(y+r) - lgamma(r) - y log r - lgamma(y+1) is 0 for y = 0: only nonzero cells
    y_nz = Y.flatten()[nz]
    lg = lgamma_diff(y_nz, log_r.flatten(-2)[..., nz]) - torch.lgamma(y_nz + 1)
    return ll.sum(dim=(-1, -2)) + lg.sum(-1)


# --------------------------------------------------------------------------
# likelihood classes
# --------------------------------------------------------------------------

class Likelihood:
    """Base class. Subclasses define n_output, prepare, log_lik, mean, sample
    and summaries; count likelihoods also define aggregate."""

    name = "base"
    supports_offset = False
    supports_agg = False

    def n_output(self, n_targets):
        raise NotImplementedError

    def n_targets(self, Y):
        """Number of target columns (outputs or classes)."""
        return Y.shape[1]

    def prepare(self, Y, device, dtype):
        """Numpy targets -> dict of tensors used by log_lik. NaN = missing."""
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        mask = ~np.isnan(Y)
        return {"Y": torch.tensor(np.where(mask, Y, 0.0), dtype=dtype, device=device),
                "mask": None if mask.all() else torch.tensor(mask, dtype=dtype, device=device)}

    def log_lik(self, out, data, offset=None):
        raise NotImplementedError

    def aggregate(self, out, agg, agg_mean):
        raise ValueError(f"{self.name} likelihood does not support window aggregation")

    def mean(self, out, offset=None):
        """Expected value of the target(s) given the network output [..., N, n_output]."""
        raise NotImplementedError

    def sample(self, out, offset=None, generator=None):
        """One draw of the target(s) per network output (posterior predictive)."""
        raise NotImplementedError

    def summaries(self, out):
        """Identifiable per-target summaries over the rows, for logging and
        convergence checks: list of (name, tensor [C, n_targets])."""
        raise NotImplementedError

    def summary_names(self):
        raise NotImplementedError


def _masked_sum(ll, mask):
    if mask is not None:
        ll = ll * mask
    return ll.sum(dim=(-1, -2))


class NegativeBinomial(Likelihood):
    """Negative binomial with mean mu = exp(out[:S] + offset) and p = sigmoid(out[S:]),
    size n = p mu / (1 - p): pmf = Gamma(y+n) / (Gamma(n) y!) p^n (1-p)^y."""

    name = "negbin"
    supports_offset = True
    supports_agg = True

    def n_output(self, n_targets):
        return 2 * n_targets

    def prepare(self, Y, device, dtype):
        d = super().prepare(Y, device, dtype)
        Yz, mask = d["Y"], d["mask"]
        obs = (Yz > 0) if mask is None else (Yz > 0) & (mask > 0)
        # indices of observed nonzero cells (lgamma terms vanish for y = 0);
        # None for chain-specific data Y [C, N, S]
        d["nz"] = torch.nonzero(obs.flatten()).squeeze(1) if Yz.dim() == 2 else None
        return d

    def log_lik(self, out, data, offset=None):
        S = out.shape[-1] // 2
        if data["nz"] is not None:
            return negbin_loglik(out, data["Y"], data["mask"], S, offset, data["nz"])
        # chain-specific data: all cells
        Y = data["Y"]
        eta = out[..., :S] if offset is None else out[..., :S] + offset
        z = out[..., S:]
        log_r = eta + z
        ll = (-torch.exp(log_r + log_softplus_neg(z)) + Y * (eta - F.softplus(-z))
              + lgamma_diff(Y, log_r) - torch.lgamma(Y + 1))
        return _masked_sum(ll, data["mask"])

    def aggregate(self, out, agg, agg_mean):
        """Rows [C, N_rows, 2S] -> observations [C, N_obs, 2S]: log of the summed
        rates (max-shifted for stability), and mean logit p."""
        S = out.shape[-1] // 2
        eta = out[..., :S]
        m = eta.amax(1, keepdim=True)
        tot = torch.matmul(agg, torch.exp(eta - m)).clamp_min(torch.finfo(out.dtype).tiny)
        z = torch.matmul(agg_mean, out[..., S:])
        return torch.cat([torch.log(tot) + m, z], -1)

    def mean(self, out, offset=None):
        S = out.shape[-1] // 2
        eta = out[..., :S] if offset is None else out[..., :S] + offset
        return torch.exp(eta)

    def sample(self, out, offset=None, generator=None):
        S = out.shape[-1] // 2
        mu = self.mean(out, offset)
        p = torch.sigmoid(out[..., S:])
        # Gamma-Poisson mixture: lambda ~ Gamma(n, scale (1-p)/p), y ~ Poisson(lambda)
        n = p * mu / (1 - p)
        lam = torch._standard_gamma(n, generator=generator) * (1 - p) / p
        return torch.poisson(lam, generator=generator)

    def summaries(self, out):
        S = out.shape[-1] // 2
        log_rate = torch.logsumexp(out[..., :S], 1) - math.log(out.shape[1])
        p = torch.sigmoid(out[..., S:]).mean(1)
        return [("log_mean_rate", log_rate), ("mean_p", p)]

    def summary_names(self):
        return ["log_mean_rate", "mean_p"]


class Poisson(Likelihood):
    """Poisson with mean exp(out[:S] + offset)."""

    name = "poisson"
    supports_offset = True
    supports_agg = True

    def n_output(self, n_targets):
        return n_targets

    def log_lik(self, out, data, offset=None):
        eta = out if offset is None else out + offset
        Y = data["Y"]
        return _masked_sum(Y * eta - torch.exp(eta) - torch.lgamma(Y + 1), data["mask"])

    def aggregate(self, out, agg, agg_mean):
        m = out.amax(1, keepdim=True)
        tot = torch.matmul(agg, torch.exp(out - m)).clamp_min(torch.finfo(out.dtype).tiny)
        return torch.log(tot) + m

    def mean(self, out, offset=None):
        return torch.exp(out if offset is None else out + offset)

    def sample(self, out, offset=None, generator=None):
        return torch.poisson(self.mean(out, offset), generator=generator)

    def summaries(self, out):
        return [("log_mean_rate", torch.logsumexp(out, 1) - math.log(out.shape[1]))]

    def summary_names(self):
        return ["log_mean_rate"]


class Gaussian(Likelihood):
    """Normal with mean out[:S] (+ offset). sd="learned": the network also outputs
    log sd in out[S:] (heteroscedastic noise); a number: fixed sd."""

    name = "gaussian"
    supports_offset = True

    def __init__(self, sd="learned"):
        self.learned = isinstance(sd, str)
        if self.learned and sd != "learned":
            raise ValueError("sd must be 'learned' or a positive number")
        self.sd = None if self.learned else float(sd)

    def n_output(self, n_targets):
        return 2 * n_targets if self.learned else n_targets

    def _split(self, out):
        if self.learned:
            S = out.shape[-1] // 2
            return out[..., :S], out[..., S:]
        return out, None

    def log_lik(self, out, data, offset=None):
        mu, log_sd = self._split(out)
        if offset is not None:
            mu = mu + offset
        if log_sd is None:
            log_sd = torch.full_like(mu, math.log(self.sd))
        r = (data["Y"] - mu) * torch.exp(-log_sd)
        return _masked_sum(-0.5 * r * r - log_sd - 0.5 * math.log(2 * math.pi), data["mask"])

    def mean(self, out, offset=None):
        mu = self._split(out)[0]
        return mu if offset is None else mu + offset

    def sd_of(self, out):
        log_sd = self._split(out)[1]
        return torch.exp(log_sd) if log_sd is not None else torch.full_like(out, self.sd)

    def sample(self, out, offset=None, generator=None):
        mu = self.mean(out, offset)
        eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype, device=mu.device)
        return mu + self.sd_of(out) * eps

    def summaries(self, out):
        mu, log_sd = self._split(out)
        res = [("mean", mu.mean(1))]
        if log_sd is not None:
            res.append(("mean_log_sd", log_sd.mean(1)))
        return res

    def summary_names(self):
        return ["mean", "mean_log_sd"] if self.learned else ["mean"]


class Categorical(Likelihood):
    """Classification: Y holds labels 0..K-1 (-1 or NaN = missing); the network
    outputs K logits. Use n_classes when not all classes occur in Y."""

    name = "categorical"

    def __init__(self, n_classes=None):
        self.n_classes = n_classes

    def n_targets(self, Y):
        Y = np.asarray(Y, dtype=float).ravel()
        k = int(np.nanmax(Y)) + 1
        if self.n_classes is None:
            self.n_classes = k
        elif k > self.n_classes:
            raise ValueError(f"labels up to {k - 1} but n_classes = {self.n_classes}")
        return self.n_classes

    def n_output(self, n_targets):
        return n_targets

    def prepare(self, Y, device, dtype):
        Y = np.asarray(Y, dtype=float)
        Y = Y.reshape(Y.shape[0], -1)[:, 0] if Y.ndim > 1 and Y.shape[-1] == 1 else Y
        mask = ~np.isnan(Y) & (np.nan_to_num(Y, nan=-1) >= 0)
        if not np.allclose(Y[mask], np.round(Y[mask])):
            raise ValueError("Categorical targets must be integer class labels")
        labels = torch.tensor(np.where(mask, Y, 0).astype(np.int64), device=device)
        return {"Y": labels,
                "mask": None if mask.all() else torch.tensor(mask, dtype=dtype, device=device)}

    def log_lik(self, out, data, offset=None):
        lp = torch.log_softmax(out, -1)
        y = data["Y"]
        idx = y.expand(lp.shape[:-1]).unsqueeze(-1)
        ll = torch.gather(lp, -1, idx).squeeze(-1)
        if data["mask"] is not None:
            ll = ll * data["mask"]
        return ll.sum(-1)

    def mean(self, out, offset=None):
        """Class probabilities."""
        return torch.softmax(out, -1)

    def sample(self, out, offset=None, generator=None):
        p = self.mean(out).reshape(-1, out.shape[-1])
        return torch.multinomial(p, 1, generator=generator).reshape(out.shape[:-1])

    def summaries(self, out):
        return [("mean_prob", torch.softmax(out, -1).mean(1))]

    def summary_names(self):
        return ["mean_prob"]


LIKELIHOODS = {"negbin": NegativeBinomial, "poisson": Poisson, "gaussian": Gaussian,
               "categorical": Categorical}


def get_likelihood(lik, **kwargs):
    """Name ('negbin', 'poisson', 'gaussian', 'categorical') or instance -> instance."""
    if isinstance(lik, Likelihood):
        return lik
    if lik not in LIKELIHOODS:
        raise ValueError(f"unknown likelihood {lik!r}; choose from {list(LIKELIHOODS)}")
    return LIKELIHOODS[lik](**kwargs)
