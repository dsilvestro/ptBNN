"""A custom likelihood: robust regression with Student-t noise.

The data have a few gross outliers. A Gaussian likelihood is pulled towards
them; a Student-t likelihood treats them as heavy-tail draws. StudentT below is
all it takes to add a likelihood that ptBNN does not have.
"""
import numpy as np
import torch

import ptbnn
from ptbnn import Likelihood


class StudentT(Likelihood):
    """Student-t with location out[:S], log scale out[S:] and fixed degrees of freedom."""

    name = "student_t"

    def __init__(self, df=3.0):
        self.df = df

    def n_output(self, n_targets):
        return 2 * n_targets                  # location and log scale per target

    def log_lik(self, out, data, offset=None):
        S = out.shape[-1] // 2                # out: [chains, rows, 2S]
        dist = torch.distributions.StudentT(self.df, out[..., :S], torch.exp(out[..., S:]),
                                            validate_args=False)
        ll = dist.log_prob(data["Y"])         # data["Y"]: [rows, S]; missing values set to 0
        if data["mask"] is not None:          # mask: 1 = observed, 0 = missing
            ll = ll * data["mask"]
        return ll.sum((-1, -2))               # one value per chain

    def mean(self, out, offset=None):
        return out[..., : out.shape[-1] // 2]

    def sample(self, out, offset=None, generator=None):
        S = out.shape[-1] // 2
        z = torch.randn(out[..., :S].shape, generator=generator, dtype=out.dtype)
        chi2 = 2 * torch._standard_gamma(torch.full_like(z, self.df / 2), generator=generator)
        return out[..., :S] + torch.exp(out[..., S:]) * z / torch.sqrt(chi2 / self.df)

    def summaries(self, out):                 # identifiable quantities for R-hat / ESS: [chains, S]
        S = out.shape[-1] // 2
        return [("mean", out[..., :S].mean(1)), ("mean_log_scale", out[..., S:].mean(1))]

    def summary_names(self):
        return ["mean", "mean_log_scale"]


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # only needed for the figure

    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(-3, 3, 120))
    y = np.sin(x) + rng.normal(0, 0.1, x.size)
    out = rng.choice(x.size, 8, replace=False)
    y[out] += rng.choice([-1, 1], 8) * rng.uniform(2, 4, 8)          # gross outliers
    X, Y = x[:, None], y[:, None]
    xg = np.linspace(-3.5, 3.5, 200)[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lik, title in [(axes[0], "gaussian", "Gaussian likelihood"),
                           (axes[1], StudentT(df=3), "Student-t likelihood (custom)")]:
        post = ptbnn.fit(X, Y, likelihood=lik, nodes=[16, 8], warmup=500, samples=500, verbose=False)
        m = post.predict_mean(xg)[..., 0]
        ax.fill_between(xg[:, 0], *np.percentile(m, [5, 95], 0), alpha=0.4, label="90% CI of mean")
        ax.plot(xg, m.mean(0), "k", label="posterior mean")
        ax.plot(xg, np.sin(xg), "--", color="C3", label="true function")
        ax.plot(x, y, ".", ms=3, color="0.4", label="data")
        ax.set_title(title)
        print(title); print(post.diagnostics().round(3).to_string())
    axes[0].legend(); axes[0].set_ylim(-3, 3)
    fig.tight_layout(); fig.savefig("custom_likelihood.png", dpi=120)
