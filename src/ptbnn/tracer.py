"""Per-chain Tracer-compatible log files."""

import torch


class TracerLogger:
    """Writes one tab-separated log per chain (readable by Tracer), with
    posterior, likelihood, prior, acceptance probability, step size and, per
    target, the likelihood's summaries of the network output over the
    training rows (negative binomial: log of the mean predicted rate and mean
    p). These summaries are identifiable, unlike the weights, so they can be
    used to assess convergence and mixing.

    Pass it as run_hmc(..., logger=TracerLogger(...)) and call close() at the end."""

    def __init__(self, model, file_stem, target_names=None, n_chains=4, log_f=1):
        self.model = model
        self.log_f = log_f
        self.S = model.n_targets
        self.files = [f"{file_stem}_chain{c + 1}.log" for c in range(n_chains)]
        self.fh = [open(f, "w") for f in self.files]
        head = ["it", "posterior", "likelihood", "prior", "accept_prob", "step_size"]
        if target_names is None:
            target_names = [str(i + 1) for i in range(self.S)]
        for name in model.lik.summary_names():
            head += [f"{name}_{s}" for s in target_names]
        for fh in self.fh:
            fh.write("\t".join(head) + "\n")

    @torch.no_grad()
    def __call__(self, it, warm, theta, lp, accept_prob, eps):
        if it % self.log_f:
            return
        prior = self.model.log_prior(theta)
        out = self.model.net.forward(theta, self.model.X)  # [C, N, n_output]
        summ = [v for _, v in self.model.lik.summaries(out)]
        rows = torch.cat([lp[:, None], (lp - prior)[:, None], prior[:, None],
                          accept_prob[:, None], eps[:, None]] + summ, 1).cpu().numpy()
        for fh, r in zip(self.fh, rows):
            fh.write(f"{it}\t" + "\t".join(f"{v:.6g}" for v in r) + "\n")

    def close(self):
        for fh in self.fh:
            fh.close()
