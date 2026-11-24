"""Gaussian regression with input-dependent noise (the network also predicts the noise sd)."""
import matplotlib.pyplot as plt
import numpy as np

import ptbnn

rng = np.random.default_rng(1)
x = np.sort(rng.uniform(-3, 3, 150))
y = np.sin(x) + rng.normal(0, 0.05 + 0.15 * (x > 0), x.size)   # noisier for x > 0
X, Y = x[:, None], y[:, None]

post = ptbnn.fit(X, Y, likelihood="gaussian", nodes=[16, 8], warmup=500, samples=500)

xg = np.linspace(-4, 4, 200)[:, None]
mean = post.predict_mean(xg)[..., 0]                    # [draws, points]: posterior of the mean
pred = post.predict_sample(xg)[..., 0]                  # posterior predictive (includes noise)
plt.fill_between(xg[:, 0], *np.percentile(pred, [5, 95], 0), alpha=0.2, label="90% predictive")
plt.fill_between(xg[:, 0], *np.percentile(mean, [5, 95], 0), alpha=0.5, label="90% CI of mean")
plt.plot(xg, mean.mean(0), "k", label="posterior mean")
plt.plot(x, y, ".", ms=3, color="0.4", label="data")
plt.legend(); plt.title("Gaussian regression"); plt.savefig("regression.png", dpi=120)
