import numpy as np
import pytest
import torch

import ptbnn
from ptbnn import TracerLogger

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(60, 2))
TARGETS = {
    "gaussian": np.sin(3 * X[:, :1]) + 0.1 * rng.normal(size=(60, 1)),
    "poisson": rng.poisson(np.exp(1 + X[:, :1])).astype(float),
    "negbin": rng.poisson(np.exp(1 + X[:, :1]) * rng.gamma(2, 0.5, (60, 1))).astype(float),
    "categorical": (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5),
}
SHORT = dict(nodes=[4], chains=2, warmup=40, samples=20, leapfrog=5, map_steps=50, verbose=False)


@pytest.mark.parametrize("lik", list(TARGETS))
def test_fit_predict_save_load(lik, tmp_path):
    post = ptbnn.fit(X, TARGETS[lik], lik, **SHORT)
    assert post.samples.shape == (2, 20, post.model.n_params)
    m = post.predict_mean(X[:7])
    k = 3 if lik == "categorical" else 1
    assert m.shape == (40, 7, k)
    if lik == "categorical":
        assert np.allclose(m.sum(-1), 1)
    s = post.predict_sample(X[:7], n=10)
    assert s.shape[:2] == (10, 7)
    tab = post.diagnostics()
    assert "log_lik" in tab.index and np.isfinite(tab.loc["log_lik", "R-hat"])
    post.save(tmp_path / "p.pt")
    back = ptbnn.Posterior.load(tmp_path / "p.pt")
    assert np.allclose(back.predict_mean(X[:7]), m)
    th = torch.tensor(post.draws(3), dtype=torch.float64)
    assert np.allclose(back.model.log_lik(th).numpy(), post.model.log_lik(th).numpy())


def test_fixed_sd_gaussian_and_partial_dependence():
    post = ptbnn.fit(X, TARGETS["gaussian"], "gaussian", sd=0.1, **SHORT)
    assert post.model.net.n_output == 1
    grid, vals = ptbnn.partial_dependence(post, X, feature=0, steps=6, n=10)
    assert grid.shape == (6,) and vals.shape == (10, 6, 1)


def test_tracer_logger_columns(tmp_path):
    m = ptbnn.BNN(X, TARGETS["categorical"], "categorical", n_nodes=[4])
    tr = TracerLogger(m, str(tmp_path / "run"), ["a", "b", "c"], n_chains=2)
    ptbnn.run_hmc(m.log_prob_and_grad, m.net.init_params(2, dtype=torch.float64), n_warmup=10,
                  n_samples=5, n_leapfrog=3, verbose=False, logger=tr)
    tr.close()
    head = open(tr.files[0]).readline().strip().split("\t")
    assert head[-3:] == ["mean_prob_a", "mean_prob_b", "mean_prob_c"]
    assert len(open(tr.files[0]).readlines()) == 16


def test_custom_likelihood(tmp_path):
    """A user-defined Likelihood subclass works with fit, predictions, diagnostics and save/load."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "examples"))
    from custom_likelihood import StudentT

    y = TARGETS["gaussian"].copy()
    y[3] = np.nan                                        # missing value
    post = ptbnn.fit(X, y, StudentT(df=4), **SHORT)
    assert post.predict_mean(X[:5]).shape == (40, 5, 1)
    assert post.predict_sample(X[:5], n=10).shape == (10, 5, 1)
    tab = post.diagnostics()
    assert {"log_lik", "mean_1", "mean_log_scale_1"} <= set(tab.index)
    post.save(tmp_path / "t.pt")
    assert np.allclose(ptbnn.Posterior.load(tmp_path / "t.pt").predict_mean(X[:5]), post.predict_mean(X[:5]))
    # matches torch's Student-t log density summed over the observed values
    th = torch.as_tensor(post.samples[:, -1], dtype=post.model.dtype)
    out = post.model.net.forward(th, post.model.X)
    ref = torch.distributions.StudentT(4.0, out[..., :1], torch.exp(out[..., 1:])).log_prob(
        torch.as_tensor(np.nan_to_num(y), dtype=post.model.dtype))
    ref = (ref * torch.as_tensor(~np.isnan(y))).sum((-1, -2))
    assert torch.allclose(post.model.log_lik(th), ref)


def test_minimal_custom_likelihood():
    """The README's minimal subclass (n_output, log_lik, mean only) runs with verbose output."""
    from ptbnn import Likelihood

    class StudentT(Likelihood):
        name = "student_t"

        def __init__(self, df=3.0):
            self.df = df

        def n_output(self, S):
            return 2 * S

        def log_lik(self, out, data, offset=None):
            S = out.shape[-1] // 2
            ll = torch.distributions.StudentT(self.df, out[..., :S], torch.exp(out[..., S:]),
                                              validate_args=False).log_prob(data["Y"])
            if data["mask"] is not None:
                ll = ll * data["mask"]
            return ll.sum((-1, -2))

        def mean(self, out, offset=None):
            return out[..., : out.shape[-1] // 2]

    post = ptbnn.fit(X, TARGETS["gaussian"], StudentT(), **{**SHORT, "verbose": True})
    assert list(post.diagnostics().index) == ["log_lik"]
    assert post.predict_mean(X[:3]).shape == (40, 3, 1)
