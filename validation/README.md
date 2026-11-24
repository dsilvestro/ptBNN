# Validation

These scripts check that the sampler draws from the right posterior. Install the
extra dependencies first: `pip install -e ".[validation]"`.

## 1. Distributions with known answers: `test_known_targets.py`

```bash
python validation/test_known_targets.py
```

Runs `ptbnn.run_hmc` on four distributions whose moments are known exactly:

- **Wide-scale Gaussian:** 100 independent Normals with standard deviations from
  0.01 to 100, which tests the mass-matrix adaptation.
- **Correlated Gaussian:** a 100-dimensional Normal with dense covariance and
  condition number 1000.
- **Banana:** a banana-shaped 2-D distribution.
- **Heavy tails:** 20 independent Student-t with 5 degrees of freedom.

For each coordinate, z = (estimated mean − true mean) / (sd / √ESS) should be
roughly standard normal if the sampler is correct and its ESS is honest. The
script also checks the package's R-hat and ESS against ArviZ on the same
draws.

## 2. Simulation-based calibration: `sbc.py`

```bash
python validation/sbc.py --likelihood gaussian --reps 100
python validation/sbc.py --likelihood categorical --n_classes 3
```

Simulation-based calibration (Talts et al. 2018) of the whole pipeline: prior,
likelihood, MAP initialisation and HMC. For each replicate, the script:

1. draws weights from the prior;
2. simulates targets for random inputs;
3. fits the simulated data;
4. records the rank of the true value among the posterior draws.

The quantities ranked are the expected target at a few inputs, the
likelihood's summaries, the log-likelihood and the squared weight norm. If the
posterior is sampled correctly, the ranks are uniform.

All replicates are fitted at once, one chain each, as one vectorised run. This
is stricter than a real analysis with several chains: a chain stuck in one mode
shows up as non-uniform ranks. Results go to `results/`: rank histograms and a
chi-square test per quantity.

## Results (ptBNN 0.1.0)

**Distributions with known answers.** All four pass:

- Errors in the estimated means are consistent with their Monte Carlo error
  (z-score SD 0.91–0.99, max |z| < 3.6).
- Variances are within 5–10% of the truth in runs of 4 × 4000 draws.
- On the banana-shaped distribution, longer runs give variances of 1.002 and
  3.005 (true values 1 and 3). NumPyro's NUTS gives 0.986 and 2.937 on the
  same target.
- R-hat and bulk ESS agree with ArviZ (`method="z_scale"`, `"bulk"`) to
  rounding error.

**Simulation-based calibration.** Settings: 50 data sets per likelihood, 80
points, 2 inputs, a [8, 4] network, one chain per data set, 1000 warmup +
3000 sampling iterations, 30 leapfrog steps. The table counts quantities
whose rank histogram deviates from uniform at p < 0.01:

| likelihood | quantities off | single-chain ESS (median) |
|---|---|---|
| Poisson | 0 of 6 | 278 |
| negative binomial | 0 of 7 | 407 |
| categorical (3 classes) | 0 of 14 | 783 |
| Gaussian, learned sd, 30 leapfrog steps | 2 of 7 (log sd, log-likelihood) | 28 |
| Gaussian, learned sd, 150 leapfrog steps, 2000 warmup | 0 of 7 | 199 |
| Gaussian, learned sd, 30 steps, 3 × longer run | 2 of 7 | 32 |

The Gaussian model with learned noise needs longer trajectories: a longer run
with short trajectories does not fix it, longer trajectories do. When chains
mix poorly, SBC shows it as true values outside the range of the posterior
draws.

Practical advice:

- Check `Posterior.diagnostics()`.
- If R-hat > 1.05 or ESS is low, increase `leapfrog`, not only `samples`.
