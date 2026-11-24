"""Rank-normalised split R-hat and bulk ESS (Vehtari et al. 2021)."""

import numpy as np


def split_rhat_ess(x, rank_normalize=True):
    """x: [C, S] draws. Returns (split R-hat, bulk ESS).

    With rank_normalize=True (default) the draws are rank-normalised first, as
    in Vehtari et al. (2021, Bayesian Analysis), which makes both measures
    robust to heavy tails and invariant to monotone transformations."""
    x = np.asarray(x, dtype=float)
    if rank_normalize and x.size > 1:
        from scipy.stats import norm, rankdata
        r = rankdata(x, method="average").reshape(x.shape)
        x = norm.ppf((r - 3 / 8) / (x.size + 1 / 4))
    C, S = x.shape
    half = S // 2
    if half < 4:
        return np.nan, np.nan
    xs = np.concatenate([x[:, :half], x[:, half:2 * half]], 0)  # [2C, half]
    m, n = xs.shape
    chain_mean = xs.mean(1)
    chain_var = xs.var(1, ddof=1)
    W = chain_var.mean()
    B = n * chain_mean.var(ddof=1)
    var_plus = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_plus / W) if W > 0 else np.nan
    # ESS (Geyer initial monotone sequence on the multi-chain autocorrelation)
    xc = xs - chain_mean[:, None]
    acov = np.array([np.fft.irfft(np.abs(np.fft.rfft(c, 2 * n)) ** 2)[:n] / n for c in xc])
    acov_mean = acov.mean(0)
    rho = 1 - (W - acov_mean) / var_plus
    rho[0] = 1.0
    t = 0
    s = 0.0
    prev = np.inf
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        s += pair
        prev = pair
        t += 2
    tau = -1 + 2 * s
    ess = m * n / max(tau, 1e-10)
    return rhat, ess
