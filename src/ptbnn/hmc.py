"""Multi-chain HMC with dual-averaging step size and windowed diagonal mass adaptation."""

import math
import time

import numpy as np
import torch

from .diagnostics import split_rhat_ess


def warmup_windows(n_warmup, init_buffer=75, term_buffer=50, base_window=25):
    """Stan-style mass-matrix adaptation schedule.

    Returns (first, ends): the iteration at which the first slow window
    starts and the iterations (0-based, inclusive) at which each slow window
    ends. Window lengths double (25, 50, 100, ...)."""
    if n_warmup < 20:
        return None, []
    # longer warmups get a longer final window for tuning the step size
    term_buffer = max(term_buffer, int(0.1 * n_warmup))
    if init_buffer + term_buffer + base_window > n_warmup:
        init_buffer = int(0.15 * n_warmup)
        term_buffer = int(0.1 * n_warmup)
        base_window = n_warmup - init_buffer - term_buffer
    ends = []
    start = init_buffer
    w = base_window
    last = n_warmup - term_buffer
    while start < last:
        end = start + w
        # if the next window would not fit, stretch this one to the end
        if end + 2 * w > last:
            end = last
        ends.append(end - 1)
        start = end
        w *= 2
    return init_buffer, ends


class DualAveraging:
    def __init__(self, eps0, target=0.8, gamma=0.05, t0=10.0, kappa=0.75):
        self.target, self.gamma, self.t0, self.kappa = target, gamma, t0, kappa
        self.restart(eps0)

    def restart(self, eps0):
        self.mu = torch.log(10 * eps0)
        self.h_bar = torch.zeros_like(eps0)
        self.log_eps_bar = torch.zeros_like(eps0)
        self.m = 0

    def update(self, accept_prob):
        self.m += 1
        m = self.m
        eta = 1.0 / (m + self.t0)
        self.h_bar = (1 - eta) * self.h_bar + eta * (self.target - accept_prob)
        log_eps = self.mu - math.sqrt(m) / self.gamma * self.h_bar
        w = m ** (-self.kappa)
        self.log_eps_bar = w * log_eps + (1 - w) * self.log_eps_bar
        return torch.exp(log_eps)

    def final(self):
        return torch.exp(self.log_eps_bar)


def leapfrog(f, theta, p, lp, grad, eps, inv_mass, n_steps):
    """Vectorised leapfrog; eps: [C, 1]."""
    p = p + 0.5 * eps * grad
    for i in range(n_steps):
        theta = theta + eps * inv_mass * p
        lp, grad = f(theta)
        if i < n_steps - 1:
            p = p + eps * grad
    p = p + 0.5 * eps * grad
    return theta, p, lp, grad


def find_initial_step_size(f, theta, lp, grad, inv_mass, eps=0.01, max_iter=40):
    """Per-chain heuristic of Hoffman & Gelman (2014), Alg. 4."""
    C = theta.shape[0]
    eps = torch.full((C,), eps, dtype=theta.dtype, device=theta.device)
    direction = None
    for _ in range(max_iter):
        p = torch.randn_like(theta) / inv_mass.sqrt()
        k0 = 0.5 * (p * p * inv_mass).sum(-1)
        th1, p1, lp1, _ = leapfrog(f, theta, p, lp, grad, eps[:, None], inv_mass, 1)
        k1 = 0.5 * (p1 * p1 * inv_mass).sum(-1)
        log_a = torch.nan_to_num((lp1 - k1) - (lp - k0), nan=-float("inf"))
        if direction is None:
            direction = torch.where(log_a > math.log(0.5), 1.0, -1.0).to(theta.dtype)
        cont = (direction * log_a) > (-direction * math.log(2.0))
        if not cont.any():
            break
        eps = torch.where(cont, eps * (2.0 ** direction), eps)
    return eps


def run_hmc(f, theta0, n_warmup=1000, n_samples=1000, n_leapfrog=50,
            target_accept=0.8, thin=1, jitter=0.2, adapt_mass=True,
            print_f=100, seed=None, logger=None, verbose=True):
    """Multi-chain HMC.

    f: theta [C, D] -> (log_prob [C], grad [C, D])
    Returns dict with 'samples' [C, K, D] (numpy, float32), traces and stats.
    """
    if seed is not None:
        torch.manual_seed(seed)
    C, D = theta0.shape
    dtype, device = theta0.dtype, theta0.device
    theta = theta0.clone()
    lp, grad = f(theta)
    inv_mass = torch.ones_like(theta)

    eps = find_initial_step_size(f, theta, lp, grad, inv_mass)
    da = DualAveraging(eps, target=target_accept)
    first_window_start, windows = warmup_windows(n_warmup) if adapt_mass else (None, [])
    window_set = set(windows)
    w_mean = torch.zeros_like(theta)
    w_m2 = torch.zeros_like(theta)
    w_n = 0

    n_total = n_warmup + n_samples
    samples = []
    lp_trace = np.zeros((C, n_total))
    acc_trace = np.zeros((C, n_total))
    div_count = np.zeros(C, dtype=int)
    t_start = time.time()
    t_sampling = None

    for it in range(n_total):
        warm = it < n_warmup
        if warm:
            eps_it = eps
        else:
            eps_it = eps * (1 + jitter * (2 * torch.rand(C, dtype=dtype, device=device) - 1))

        p0 = torch.randn_like(theta) / inv_mass.sqrt()
        k0 = 0.5 * (p0 * p0 * inv_mass).sum(-1)
        th1, p1, lp1, g1 = leapfrog(f, theta, p0, lp, grad, eps_it[:, None], inv_mass, n_leapfrog)
        k1 = 0.5 * (p1 * p1 * inv_mass).sum(-1)
        dH = (lp1 - k1) - (lp - k0)
        dH = torch.nan_to_num(dH, nan=-float("inf"))
        div = dH < -1000
        accept_prob = torch.clamp(torch.exp(torch.clamp(dH, max=0.0)), 0, 1)
        accept = torch.rand(C, dtype=dtype, device=device) < accept_prob
        a = accept[:, None]
        theta = torch.where(a, th1, theta)
        lp = torch.where(accept, lp1, lp)
        grad = torch.where(a, g1, grad)
        div_count += div.cpu().numpy().astype(int) * (0 if warm else 1)

        lp_trace[:, it] = lp.cpu().numpy()
        acc_trace[:, it] = accept_prob.cpu().numpy()

        if warm:
            eps = da.update(accept_prob)
            # mass-matrix window accumulation
            if first_window_start is not None and it >= first_window_start and windows and it <= windows[-1]:
                w_n += 1
                delta = theta - w_mean
                w_mean = w_mean + delta / w_n
                w_m2 = w_m2 + delta * (theta - w_mean)
                if it in window_set:
                    var = w_m2 / max(w_n - 1, 1)
                    var = (w_n / (w_n + 5.0)) * var + 1e-3 * (5.0 / (w_n + 5.0))
                    inv_mass = var
                    w_mean.zero_(); w_m2.zero_(); w_n = 0
                    lp, grad = f(theta)
                    eps = find_initial_step_size(f, theta, lp, grad, inv_mass, eps=float(eps.mean()))
                    da.restart(eps)
            if it == n_warmup - 1:
                eps = da.final()
                t_sampling = time.time()
        else:
            if (it - n_warmup) % thin == 0:
                samples.append(theta.detach().cpu().numpy().astype(np.float32))

        if logger is not None:
            logger(it, warm, theta, lp, accept_prob, eps_it)

        if verbose and (it % print_f == 0 or it == n_total - 1):
            el = time.time() - t_start
            rate = (it + 1) / el
            eta = (n_total - it - 1) / rate
            stage = "warmup" if warm else "sample"
            lo = max(0, it - print_f + 1)
            print(f"  {stage} {it:6d}/{n_total}  acc {acc_trace[:, lo:it + 1].mean():.2f}  "
                  f"eps {eps.mean().item():.2e}  logpost "
                  + " ".join(f"{v:.1f}" for v in lp.cpu().tolist())
                  + f"  [{el:.0f}s, ~{eta:.0f}s left]")

    t_end = time.time()
    samples = np.stack(samples, 1) if samples else np.zeros((C, 0, D), np.float32)
    lp_post = lp_trace[:, n_warmup:]
    rhat, ess = split_rhat_ess(lp_post) if n_samples >= 8 else (np.nan, np.nan)
    return {
        "samples": samples,
        "lp_trace": lp_trace,
        "accept_trace": acc_trace,
        "step_size": eps.cpu().numpy(),
        "inv_mass": inv_mass.cpu().numpy(),
        "divergences": div_count,
        "accept_rate": acc_trace[:, n_warmup:].mean(1) if n_samples else np.nan,
        "rhat_logpost": rhat,
        "ess_logpost": ess,
        "time_total": t_end - t_start,
        "time_sampling": (t_end - t_sampling) if t_sampling else np.nan,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "n_leapfrog": n_leapfrog,
        "thin": thin,
        "last_theta": theta.detach().cpu().numpy(),
    }
