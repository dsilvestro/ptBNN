import numpy as np
import torch

from ptbnn import run_hmc, split_rhat_ess, warmup_windows


def test_warmup_windows():
    first, ends = warmup_windows(1000)
    assert first == 75
    assert ends[-1] == 1000 - 100 - 1           # final window = 10% of warmup
    lengths = np.diff([first - 1] + ends)
    assert (np.diff(lengths[:-1]) > 0).all()    # windows grow


def test_rhat_ess_iid():
    x = np.random.default_rng(0).normal(size=(4, 2000))
    rhat, ess = split_rhat_ess(x)
    assert abs(rhat - 1) < 0.01
    assert 6000 < ess < 10000


def test_rhat_detects_shifted_chain():
    x = np.random.default_rng(0).normal(size=(4, 1000))
    x[0] += 3
    assert split_rhat_ess(x)[0] > 1.3  # ArviZ (z_scale): 1.47


def test_hmc_gaussian_with_different_scales():
    torch.manual_seed(0)
    D = 20
    sd = torch.tensor(np.logspace(-1.5, 1.5, D))
    mean = torch.linspace(-2, 2, D, dtype=torch.float64)

    def f(th):
        return -0.5 * (((th - mean) / sd) ** 2).sum(-1), -(th - mean) / sd ** 2

    r = run_hmc(f, torch.zeros(4, D, dtype=torch.float64), n_warmup=500, n_samples=1500,
                n_leapfrog=20, seed=1, verbose=False)
    x = r["samples"].reshape(-1, D).astype(float)
    ess = np.array([split_rhat_ess(r["samples"][:, :, d])[1] for d in range(D)])
    z = (x.mean(0) - mean.numpy()) / (sd.numpy() / np.sqrt(ess))
    assert np.abs(z).max() < 4.5
    assert np.allclose(x.std(0) / sd.numpy(), 1, atol=0.12)
    assert r["divergences"].sum() == 0
    assert 0.6 < r["accept_rate"].mean() < 0.97
