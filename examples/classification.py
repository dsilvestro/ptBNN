"""Classification of three classes in two dimensions, with predictive uncertainty."""
import matplotlib.pyplot as plt
import numpy as np

import ptbnn

rng = np.random.default_rng(3)
centers = np.array([[-1, 0], [1, 0.8], [0.5, -1]])
labels = rng.integers(0, 3, 300)
X = centers[labels] + rng.normal(0, 0.5, (300, 2))

post = ptbnn.fit(X, labels, likelihood="categorical", nodes=[16, 8], warmup=500, samples=500)

g = np.linspace(-3, 3, 120)
G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
prob = post.predict_mean(G, n=200).mean(0)              # [points, classes], averaged over draws
entropy = -(prob * np.log(prob)).sum(1)                 # high where the model is unsure
acc = (post.predict_mean(X).mean(0).argmax(1) == labels).mean()

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
axs[0].imshow(prob.reshape(120, 120, 3), origin="lower", extent=[-3, 3, -3, 3])
axs[0].set_title(f"class probabilities (training accuracy {acc:.2f})")
im = axs[1].imshow(entropy.reshape(120, 120), origin="lower", extent=[-3, 3, -3, 3], cmap="magma")
axs[1].set_title("predictive entropy"); fig.colorbar(im, ax=axs[1])
for ax in axs:
    ax.scatter(*X.T, c=np.eye(3)[labels], s=8, edgecolors="k", linewidths=0.4)  # class k = RGB channel k
fig.tight_layout(); fig.savefig("classification.png", dpi=120)
