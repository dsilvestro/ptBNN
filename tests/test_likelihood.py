import math

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

from ptbnn import lgamma_diff, negbin_loglik


def scipy_nb(y, eta, z):
    """Reference: mean mu = exp(eta), p = sigmoid(z), size n = p mu / (1 - p);
    pmf = Gamma(y+n) / (Gamma(n) y!) p^n (1-p)^y  ==  scipy nbinom(n, p)."""
    p = 1 / (1 + np.exp(-z))
    n = p * np.exp(eta) / (1 - p)
    return nbinom.logpmf(y, n, p)


@pytest.mark.parametrize("eta", [-3.0, 0.0, 2.0, 6.0])
@pytest.mark.parametrize("z", [-4.0, -1.0, 0.0, 2.0, 5.0])
def test_negbin_matches_scipy(eta, z):
    y = np.array([0, 1, 3, 10, 50, 400], dtype=float)
    S = len(y)
    out = torch.tensor(np.r_[np.full(S, eta), np.full(S, z)])[None, None, :]  # [C=1, N=1, 2S]
    ll = negbin_loglik(out, torch.tensor(y)[None, :], None, S).item()
    ref = scipy_nb(y, eta, z).sum()
    assert ll == pytest.approx(ref, rel=1e-9, abs=1e-8)


def test_lgamma_diff_switch_is_continuous():
    # the Stirling branch (r > 100) must agree with the direct formula at the switch
    y = torch.tensor([0.0, 1.0, 7.0, 120.0, 5000.0], dtype=torch.float64)
    for log_r in (math.log(100.0) - 1e-9, math.log(100.0) + 1e-9):
        lr = torch.full_like(y, log_r)
        direct = torch.lgamma(y + lr.exp()) - torch.lgamma(lr.exp()) - y * lr
        assert torch.allclose(lgamma_diff(y, lr), direct, rtol=1e-10, atol=1e-8)


def test_poisson_limit_float32_finite():
    # p -> 1 (r -> inf): the NB tends to a Poisson; float32 must stay finite with finite gradients
    y = torch.tensor([[0.0, 3.0, 40.0]])
    out = torch.tensor([[[1.0, 2.0, 3.5, 30.0, 40.0, 60.0]]], requires_grad=True)
    ll = negbin_loglik(out, y, None, 3)
    ll.sum().backward()
    assert torch.isfinite(ll).all() and torch.isfinite(out.grad).all()
    from scipy.stats import poisson
    ref = poisson.logpmf([0, 3, 40], np.exp([1.0, 2.0, 3.5])).sum()
    assert ll.item() == pytest.approx(ref, rel=1e-4)


def test_extreme_inputs_have_finite_gradients():
    y = torch.tensor([[0.0, 1.0, 1e4]], dtype=torch.float32)
    for vals in ([-80.0, 0.0, 80.0, -80.0, 0.0, 80.0], [50.0, -50.0, 10.0, 50.0, -50.0, -10.0]):
        out = torch.tensor([[vals]], requires_grad=True)
        ll = negbin_loglik(out, y, None, 3)
        ll.sum().backward()
        assert torch.isfinite(out.grad).all()


def test_gradcheck():
    torch.manual_seed(0)
    y = torch.tensor([[0.0, 2.0], [5.0, 30.0]], dtype=torch.float64)
    out = (torch.randn(2, 2, 4, dtype=torch.float64) * 2).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda o: negbin_loglik(o, y, None, 2), (out,))


def test_mask_and_offset():
    y = torch.tensor([[3.0, 0.0], [1.0, 7.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    off = torch.tensor([[0.5], [-0.2]], dtype=torch.float64)
    out = torch.tensor([[[0.1, 0.2, 1.0, -1.0], [1.0, 2.0, 0.0, 0.5]]], dtype=torch.float64)
    ll = negbin_loglik(out, y, mask, 2, offset=off).item()
    ref = sum(scipy_nb(y[i, s].item(), out[0, i, s].item() + off[i, 0].item(), out[0, i, 2 + s].item())
              for i in range(2) for s in range(2) if mask[i, s] > 0)
    assert ll == pytest.approx(ref, rel=1e-10)
