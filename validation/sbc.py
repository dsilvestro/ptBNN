"""
Simulation-based calibration (SBC; Talts et al. 2018) of ptBNN for any likelihood.

For each replicate: weights ~ prior, targets ~ likelihood(network(weights)),
then fit with the standard pipeline (short MAP, then HMC) and record the rank
of the true value of each test quantity among the posterior draws. With a
correct sampler and likelihood, the ranks are uniform.

All replicates are fitted at once (replicate r = chain r of one vectorised
HMC run). Test quantities are identifiable functions of the network:
the expected target at a few inputs, the likelihood's summaries over the
training inputs, the log-likelihood and the squared weight norm (individual
weights are not used: hidden units can be permuted).

Usage:
    python validation/sbc.py --likelihood gaussian --reps 100
    python validation/sbc.py --likelihood categorical --n_classes 3
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import binom, chi2

import ptbnn
from ptbnn import BNN, find_map, run_hmc, split_rhat_ess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--likelihood", default="gaussian", choices=["gaussian", "poisson", "negbin", "categorical"])
    ap.add_argument("--n_classes", type=int, default=3)
    ap.add_argument("--n", type=int, default=80, help="data points per replicate")
    ap.add_argument("--features", type=int, default=2)
    ap.add_argument("--targets", type=int, default=1, help="output columns (not for categorical)")
    ap.add_argument("--nodes", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--p_scale", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--thin", type=int, default=30)
    ap.add_argument("--leapfrog", type=int, default=30)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--map_steps", type=int, default=500)
    ap.add_argument("--max_abs", type=float, default=1e5,
                    help="redraw replicates with |target| above this (keeps SBC valid, see sbc_bnn notes)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tag = args.likelihood

    X = rng.uniform(-1, 1, (args.n, args.features))
    if args.likelihood == "categorical":
        Y0 = rng.integers(0, args.n_classes, args.n)   # placeholder: sets up the model shapes
        lik_kw = {"n_classes": args.n_classes}
    else:
        Y0 = np.ones((args.n, args.targets))
        lik_kw = {}
    template = BNN(X, Y0, args.likelihood, n_nodes=args.nodes, p_scale=args.p_scale, **lik_kw)
    lik, net, D = template.lik, template.net, template.n_params
    Xt = template.X

    # prior draws and simulated targets (replicates with extreme targets are redrawn:
    # (theta, y) then come from the joint restricted to a set of y, whose
    # conditional p(theta | y) is the unchanged posterior)
    thetas, Ys, n_drawn = [], [], 0
    with torch.no_grad():
        while sum(len(t) for t in thetas) < args.reps:
            th = torch.randn(4 * args.reps, D, generator=gen, dtype=torch.float64) * args.p_scale
            n_drawn += len(th)
            y = lik.sample(net.forward(th, Xt), generator=gen).double()
            ok = torch.isfinite(y).flatten(1).all(1) & (y.abs().flatten(1).amax(1) <= args.max_abs)
            thetas.append(th[ok]); Ys.append(y[ok])
    theta_true = torch.cat(thetas)[:args.reps]
    Y = torch.cat(Ys)[:args.reps]
    print(f"SBC {tag}: {args.n} points, {args.features} features, network "
          f"{[args.features] + args.nodes + [net.n_output]} ({D} parameters), {args.reps} replicates; "
          f"kept {sum(len(t) for t in thetas)}/{n_drawn} prior draws")

    # one data set per chain
    model = BNN(X, Y0, args.likelihood, n_nodes=args.nodes, p_scale=args.p_scale, **lik_kw)
    model.data = lik.prepare(Y.numpy(), model.device, model.dtype)
    model.Y, model.mask = model.data["Y"], model.data["mask"]

    t0 = time.time()
    theta0 = net.init_params(args.reps, init_std=0.1, generator=gen)
    theta0 = find_map(model, theta0, n_steps=args.map_steps, verbose=False)
    r = run_hmc(model.log_prob_and_grad, theta0, n_warmup=args.warmup, n_samples=args.samples,
                n_leapfrog=args.leapfrog, target_accept=args.target_accept, seed=args.seed,
                print_f=500)
    print(f"MAP + HMC: {time.time() - t0:.0f}s; divergences {r['divergences'].sum()} in "
          f"{(r['divergences'] > 0).sum()} replicates; accept rate {r['accept_rate'].mean():.2f}")
    draws = torch.tensor(r["samples"][:, ::args.thin].astype(np.float64))  # [C, L, D]
    L = draws.shape[1]

    pts = np.linspace(0, args.n - 1, 3).astype(int)

    def quantities(th):
        C_, K = th.shape[:2]
        out = net.forward(th.reshape(C_ * K, D), Xt)                      # [C*K, N, n_out]
        q = {}
        mean = lik.mean(out).reshape(C_, K, args.n, -1)
        for i in pts:
            for s in range(mean.shape[-1]):
                q[f"mean_x{i}_{s + 1}"] = mean[:, :, i, s]
        for name, v in lik.summaries(out):
            v = v.reshape(C_, K, -1)
            for s in range(v.shape[-1]):
                q[f"{name}_{s + 1}"] = v[:, :, s]
        data_rep = {k: (v[:, None].expand(C_, K, *v.shape[1:]).reshape(C_ * K, *v.shape[1:])
                        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == C_ else v)
                    for k, v in model.data.items()}
        q["log_lik"] = lik.log_lik(out, data_rep).reshape(C_, K)
        q["sq_norm"] = (th ** 2).sum(-1)
        return q

    with torch.no_grad():
        q_true = quantities(theta_true[:, None])
        q_draw = {}
        for i in range(0, L, 20):
            for k, v in quantities(draws[:, i:i + 20]).items():
                q_draw.setdefault(k, []).append(v)
        q_draw = {k: torch.cat(v, 1) for k, v in q_draw.items()}

    n_bins = 10
    ranks, rows = {}, []
    for k in q_true:
        t, dr = q_true[k].numpy(), q_draw[k].numpy()
        less, ties = (dr < t).sum(1), (dr == t).sum(1)
        rk = less + rng.integers(0, ties + 1)
        ranks[k] = rk
        counts = np.bincount((rk * n_bins // (L + 1)).astype(int), minlength=n_bins)
        exp = len(rk) / n_bins
        stat = ((counts - exp) ** 2 / exp).sum()
        rows.append(dict(quantity=k, chi2=stat, p_value=1 - chi2.cdf(stat, n_bins - 1),
                         mean_rank=rk.mean() / L))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, f"sbc_{tag}_summary.csv"), index=False)
    pd.DataFrame(ranks).assign(L=L).to_csv(os.path.join(args.out, f"sbc_{tag}_ranks.csv"), index=False)
    ess = np.array([split_rhat_ess(r["lp_trace"][c:c + 1, args.warmup:])[1] for c in range(args.reps)])
    print(f"{L} thinned draws per replicate; single-chain ESS of the log posterior: median "
          f"{np.median(ess):.0f}, 5th pct {np.percentile(ess, 5):.0f}")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"quantities with p < 0.01: {(res.p_value < 0.01).sum()} of {len(res)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nc = 5
    nr = math.ceil(len(ranks) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 1.9 * nr), squeeze=False)
    lo, hi = binom.ppf([0.005, 0.995], args.reps, 1 / n_bins)
    for ax, k in zip(axs.flat, ranks):
        ax.axhspan(lo, hi, color="0.85")
        ax.bar(range(n_bins), np.bincount((ranks[k] * n_bins // (L + 1)).astype(int), minlength=n_bins),
               width=1, color="#3b6ea5")
        ax.set_title(k, fontsize=8); ax.set_xticks([]); ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[len(ranks):]:
        ax.axis("off")
    fig.suptitle(f"SBC rank histograms, {tag} (grey: 99% band for uniform ranks)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"sbc_{tag}_ranks.png"), dpi=110)


if __name__ == "__main__":
    main()
