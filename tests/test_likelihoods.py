import numpy as np
import pytest
import torch
from scipy.stats import nbinom, norm, poisson

from ptbnn import BNN, Categorical, Gaussian, NegativeBinomial, Poisson

rng = np.random.default_rng(0)


def test_poisson_matches_scipy_with_offset_and_mask():
    Y = np.array([[0, 3], [7, np.nan], [1, 20]], dtype=float)
    out = torch.tensor(rng.normal(size=(2, 3, 2)))
    off = torch.tensor([[0.5], [-1.0], [0.0]])
    lik = Poisson()
    d = lik.prepare(Y, "cpu", torch.float64)
    ll = lik.log_lik(out, d, off).numpy()
    for c in range(2):
        mu = np.exp(out[c].numpy() + off.numpy())
        ref = np.nansum(np.where(np.isnan(Y), np.nan, poisson.logpmf(np.nan_to_num(Y), mu)))
        assert ll[c] == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("sd", ["learned", 0.3])
def test_gaussian_matches_scipy(sd):
    Y = rng.normal(size=(5, 2))
    lik = Gaussian(sd=sd)
    out = torch.tensor(rng.normal(size=(3, 5, lik.n_output(2))))
    ll = lik.log_lik(out, lik.prepare(Y, "cpu", torch.float64)).numpy()
    for c in range(3):
        o = out[c].numpy()
        s = np.exp(o[:, 2:]) if sd == "learned" else 0.3
        assert ll[c] == pytest.approx(norm.logpdf(Y, o[:, :2], s).sum(), rel=1e-12)


def test_categorical_with_missing_labels():
    y = np.array([0, 2, 1, -1, 2])
    lik = Categorical()
    assert lik.n_targets(y) == 3
    out = torch.tensor(rng.normal(size=(2, 5, 3)))
    ll = lik.log_lik(out, lik.prepare(y, "cpu", torch.float64)).numpy()
    lp = torch.log_softmax(out, -1).numpy()
    for c in range(2):
        ref = sum(lp[c, i, y[i]] for i in range(5) if y[i] >= 0)
        assert ll[c] == pytest.approx(ref, rel=1e-12)


def test_negbin_chain_specific_data_matches_shared():
    # Y [C, N, S] (one data set per chain, used for SBC) gives the same values
    Y = rng.poisson(5, size=(4, 3)).astype(float)
    out = torch.tensor(rng.normal(size=(2, 4, 6)))
    lik = NegativeBinomial()
    shared = lik.log_lik(out, lik.prepare(Y, "cpu", torch.float64))
    batched = lik.log_lik(out, lik.prepare(np.stack([Y, Y]), "cpu", torch.float64))
    assert torch.allclose(shared, batched, rtol=1e-12)
    p = 1 / (1 + np.exp(-out[0, :, 3:].numpy()))
    mu = np.exp(out[0, :, :3].numpy())
    assert shared[0].item() == pytest.approx(nbinom.logpmf(Y, p * mu / (1 - p), p).sum(), rel=1e-10)


def test_poisson_aggregation():
    X = rng.normal(size=(10, 2))
    agg = np.zeros((3, 10))
    agg[0, :4] = agg[1, 4:7] = agg[2, 7:] = 1
    Y = rng.poisson(5, size=(3, 1)).astype(float)
    m = BNN(X, Y, "poisson", n_nodes=[4], agg=agg)
    th = torch.randn(2, m.n_params, dtype=torch.float64)
    rate = np.exp(m.net.forward(th, m.X).detach().numpy())  # [2, 10, 1]
    ref = [poisson.logpmf(Y, agg @ rate[c]).sum() for c in range(2)]
    assert np.allclose(m.log_lik(th).detach().numpy(), ref, rtol=1e-12)


def test_samplers_have_the_right_moments():
    gen = torch.Generator().manual_seed(1)
    n = 200000
    out = torch.tensor([[[2.0, 0.5]]], dtype=torch.float64).expand(n, 1, 2)       # NB: mu = e^2, p = sigmoid(0.5)
    y = NegativeBinomial().sample(out, generator=gen).numpy().ravel()
    mu, p = np.exp(2.0), 1 / (1 + np.exp(-0.5))
    assert y.mean() == pytest.approx(mu, rel=0.01)
    assert y.var() == pytest.approx(mu / p, rel=0.03)        # NB variance = mu / p
    g = Gaussian().sample(torch.tensor([[[1.0, np.log(2.0)]]]).expand(n, 1, 2), generator=gen).numpy()
    assert g.mean() == pytest.approx(1.0, abs=0.02) and g.std() == pytest.approx(2.0, rel=0.01)
    c = Categorical(3).sample(torch.log(torch.tensor([[[0.2, 0.3, 0.5]]])).expand(n, 1, 3), generator=gen)
    assert np.allclose(np.bincount(c.numpy().ravel(), minlength=3) / n, [0.2, 0.3, 0.5], atol=0.005)


def test_bad_inputs():
    X = rng.normal(size=(5, 2))
    with pytest.raises(ValueError):
        BNN(X, rng.normal(size=(5, 1)), "gaussian", agg=np.ones((1, 5)))   # no aggregation for Gaussian
    with pytest.raises(ValueError):
        BNN(X, np.array([0, 1, 1, 0, 1]), "categorical", offset=np.zeros(5))
    with pytest.raises(ValueError):
        BNN(X, np.array([0, 1.5, 1, 0, 1]), "categorical")
    with pytest.raises(ValueError):
        BNN(X, rng.normal(size=(4, 1)), "gaussian")                         # rows do not match
