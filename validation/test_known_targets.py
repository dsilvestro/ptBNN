"""
Check the HMC sampler (ptbnn.run_hmc) on targets whose moments are known
exactly, and check its R-hat / ESS against ArviZ on the same draws.

For every coordinate the Monte Carlo error of the posterior mean is
sd / sqrt(ESS), so z = (mean - true) / (sd / sqrt(ESS)) should be ~N(0, 1)
if the sampler is correct and its ESS estimate is honest.

Usage: python validation/test_known_targets.py
"""
import os
import sys

import numpy as np
import torch

from ptbnn import run_hmc, split_rhat_ess  # noqa: E402

torch.set_default_dtype(torch.float64)
C, WARM, SAMP = 4, 1000, 4000


def make_target(kind, D=100, seed=0):
    g = np.random.default_rng(seed)
    if kind == "gauss_scales":
        # independent Normals with sd spanning 0.01 .. 100 (tests mass-matrix adaptation)
        sd = torch.tensor(np.logspace(-2, 2, D))
        mean = torch.tensor(g.normal(0, 1, D))
        f = lambda th: (-0.5 * (((th - mean) / sd) ** 2).sum(-1), -(th - mean) / sd ** 2)
        return f, mean.numpy(), sd.numpy() ** 2, D
    if kind == "gauss_corr":
        # dense covariance, condition number ~1e3 (diagonal mass cannot remove it)
        A = g.normal(size=(D, D))
        Q, _ = np.linalg.qr(A)
        ev = np.logspace(-1.5, 1.5, D)
        cov = Q @ np.diag(ev) @ Q.T
        P = torch.tensor(np.linalg.inv(cov))
        mean = torch.tensor(g.normal(0, 1, D))
        f = lambda th: (-0.5 * (((th - mean) @ P) * (th - mean)).sum(-1), -(th - mean) @ P)
        return f, mean.numpy(), np.diag(cov), D
    if kind == "banana":
        # x1 ~ N(0, 1), x2 | x1 ~ N(b (x1^2 - 1), 1):  E = 0, Var(x2) = 2 b^2 + 1
        b = 1.0
        def f(th):
            x1, x2 = th[:, 0], th[:, 1]
            r = x2 - b * (x1 ** 2 - 1)
            lp = -0.5 * x1 ** 2 - 0.5 * r ** 2
            g1 = -x1 + r * 2 * b * x1
            return lp, torch.stack([g1, -r], 1)
        return f, np.zeros(2), np.array([1.0, 2 * b ** 2 + 1]), 2
    if kind == "student_t":
        # independent Student-t, 5 df: heavy tails, Var = df / (df - 2)
        nu = 5.0
        def f(th):
            lp = -(nu + 1) / 2 * torch.log1p(th ** 2 / nu)
            return lp.sum(-1), -(nu + 1) * th / (nu + th ** 2)
        return f, np.zeros(D), np.full(D, nu / (nu - 2)), D
    raise ValueError(kind)


def check(kind, D=None, n_leapfrog=30):
    f, mean, var, D = make_target(kind, **({} if D is None else {"D": D}))
    theta0 = torch.randn(C, D) * 2
    r = run_hmc(f, theta0, n_warmup=WARM, n_samples=SAMP, n_leapfrog=n_leapfrog,
                target_accept=0.8, seed=1, verbose=False)
    x = r["samples"].astype(float)  # [C, S, D]
    rhat = np.array([split_rhat_ess(x[:, :, d])[0] for d in range(D)])
    ess = np.array([split_rhat_ess(x[:, :, d])[1] for d in range(D)])
    flat = x.reshape(-1, D)
    m, v = flat.mean(0), flat.var(0)
    z_mean = (m - mean) / np.sqrt(var / ess)
    # sd of a variance estimate ~ var * sqrt(2 / ESS) (Gaussian); looser for heavy tails
    z_var = (v - var) / (var * np.sqrt(2 / ess))

    import arviz as az
    idata = az.from_dict({"posterior": {"x": x}})
    # "z_scale" = rank-normalised split R-hat, the bulk version computed by ptbnn
    az_rhat = az.rhat(idata, method="z_scale")["x"].values
    az_ess = az.ess(idata, method="bulk")["x"].values
    print(f"\n== {kind} (D={D}) ==  {r['time_total']:.0f}s, accept {r['accept_rate'].mean():.2f}, "
          f"divergences {r['divergences'].sum()}")
    print(f"  mean  z-scores: sd {z_mean.std():.2f} (expect ~1), max |z| {np.abs(z_mean).max():.2f}, "
          f"share |z|>3: {np.mean(np.abs(z_mean) > 3):.3f}")
    print(f"  var   z-scores: mean {z_var.mean():+.2f}, sd {z_var.std():.2f}, "
          f"var ratio range {np.min(v / var):.3f}-{np.max(v / var):.3f}")
    print(f"  R-hat max {rhat.max():.4f}  (ArviZ {az_rhat.max():.4f}, max abs diff "
          f"{np.abs(rhat - az_rhat).max():.1e})")
    print(f"  bulk ESS median {np.median(ess):.0f}  (ArviZ {np.median(az_ess):.0f}, max rel diff "
          f"{np.max(np.abs(ess - az_ess) / az_ess):.1e})")
    return dict(kind=kind, z_sd=z_mean.std(), z_max=np.abs(z_mean).max(), z_var_mean=z_var.mean(),
                rhat_max=rhat.max(), ess_median=np.median(ess))


if __name__ == "__main__":
    res = [check("gauss_scales"), check("gauss_corr"), check("banana", n_leapfrog=10),
           check("student_t", D=20)]
    ok = all(r["z_max"] < 4.5 and r["rhat_max"] < 1.01 for r in res)
    print("\nPASS" if ok else "\nCHECK RESULTS ABOVE")
