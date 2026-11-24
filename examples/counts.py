"""Count regression: Poisson and negative binomial with an exposure offset
(e.g. counts collected over sampling periods of different length)."""
import matplotlib.pyplot as plt
import numpy as np

import ptbnn

rng = np.random.default_rng(2)
x = np.sort(rng.uniform(0, 1, 200))
exposure = rng.integers(5, 30, x.size)                  # e.g. days per sample
rate = np.exp(1 + 1.5 * np.sin(6 * x))                  # true rate per day
y = rng.negative_binomial(5, 5 / (5 + rate * exposure))  # overdispersed counts
X, Y, off = x[:, None], y[:, None], np.log(exposure)[:, None]

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
xg = np.linspace(0, 1, 100)[:, None]
for ax, lik in zip(axs, ["poisson", "negbin"]):
    post = ptbnn.fit(X, Y, likelihood=lik, offset=off, nodes=[16, 8], warmup=500, samples=500)
    r = post.predict_mean(xg)[..., 0]                   # rate per unit exposure (offset 0)
    ax.fill_between(xg[:, 0], *np.percentile(r, [5, 95], 0), alpha=0.4, label="90% CI")
    ax.plot(xg, r.mean(0), "k", label="posterior mean")
    ax.plot(xg, np.exp(1 + 1.5 * np.sin(6 * xg)), "r--", label="true rate")
    ax.plot(x, y / exposure, ".", ms=3, color="0.5", label="count / exposure")
    ax.set_title(lik); ax.set_yscale("log")
axs[0].legend(); fig.tight_layout(); fig.savefig("counts.png", dpi=120)
