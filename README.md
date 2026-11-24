# ptBNN

Bayesian neural networks in PyTorch, sampled with Hamiltonian Monte Carlo (HMC).

ptBNN samples the full posterior of the weights of a fully connected network
with HMC. Several chains run as one batched computation on the CPU, a CUDA GPU
or an Apple GPU. Likelihoods cover regression, count data and classification:

| likelihood | targets | network output |
|---|---|---|
| `gaussian` | continuous | mean, and log sd (heteroscedastic) unless a fixed `sd` is given |
| `poisson` | counts | log mean |
| `negbin` | overdispersed counts | log mean and logit p (size n = p·μ/(1−p)) |
| `categorical` | class labels 0..K−1 | K logits |

Count likelihoods also accept an exposure offset and aggregation, where each
observation is the sum of a rate over several rows (for example, a count
accumulated over several time steps). Other likelihoods can be added as a short
class: see [Custom likelihoods](#custom-likelihoods). ptBNN is the successor of
[npBNN](https://github.com/dsilvestro/npBNN), whose Metropolis-Hastings sampler
it replaces.

## Installation

```bash
git clone https://github.com/dsilvestro/ptBNN.git
cd ptBNN
pip install -e .            # extras: [examples] (matplotlib), [dev] (pytest), [validation] (JAX/NumPyro)
```

On a cluster without GPUs, install the CPU build of PyTorch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Quick start

```python
import numpy as np
import ptbnn

x = np.random.uniform(-3, 3, (150, 1))
y = np.sin(x) + 0.1 * np.random.normal(size=x.shape)

post = ptbnn.fit(x, y, likelihood="gaussian", nodes=[16, 8], chains=4, warmup=500, samples=500)

xg = np.linspace(-4, 4, 200)[:, None]
mean = post.predict_mean(xg)        # [draws, points, 1]: posterior of the expected value
pred = post.predict_sample(xg)      # posterior predictive draws (include the noise)
post.diagnostics()                  # R-hat and ESS of identifiable quantities
post.save("fit.pt")                 # reload with ptbnn.Posterior.load("fit.pt")
```

Examples (each runs in about a minute on a laptop CPU and saves a figure):

- `examples/regression.py`: Gaussian regression with input-dependent noise
- `examples/counts.py`: Poisson vs negative binomial counts with an exposure offset
- `examples/classification.py`: three classes in two dimensions, with predictive uncertainty
- `examples/custom_likelihood.py`: robust regression with a user-defined Student-t likelihood

## Options of `ptbnn.fit`

| argument | default | meaning |
|---|---|---|
| `likelihood` | `"gaussian"` | `gaussian`, `poisson`, `negbin`, `categorical`, or a `Likelihood` instance |
| `nodes` | `(16, 8)` | hidden layer sizes |
| `activation` | `"relu"` | also `tanh`, `swish`, `softplus`, `leaky_relu` |
| `hidden_bias` | `True` | biases in the hidden layers (`False` = npBNN's `use_bias_node=-1`) |
| `p_scale` | `1` | SD of the Normal(0, p_scale) prior on every weight |
| `chains` | `4` | chains, run in parallel as one batched computation |
| `warmup`, `samples` | `1000`, `1000` | HMC iterations per chain; step size and mass matrix are tuned during warmup |
| `leapfrog` | `50` | leapfrog steps per iteration |
| `target_accept` | `0.8` | target acceptance rate; raise to 0.9 if there are many divergences |
| `thin` | `1` | keep every thin-th draw |
| `map_steps` | `1000` | Adam steps to find starting points (0 = start from small random weights) |
| `offset` | `None` | added to the log mean (counts) or the mean (Gaussian) |
| `agg` | `None` | `[n_obs, n_rows]` 0/1 matrix: each observation is the sum of the rates of its rows (counts) |
| `device`, `dtype` | `"cpu"`, `"auto"` | `cuda`, `mps` or `auto`; float64 on CPU, float32 on GPU |
| `sd`, `n_classes` | | passed to the likelihood: fixed Gaussian sd; number of classes |

Missing targets can be given as NaN, or as −1 for class labels.

## Custom likelihoods

Any likelihood can be used by subclassing `ptbnn.Likelihood` and passing an
instance to `ptbnn.fit` or `BNN`. The network output `out` has shape
`[chains, rows, n_output]`; the targets arrive in `data["Y"]` as `[rows, S]`,
with missing values set to 0 and flagged in `data["mask"]`.

| method | returns |
|---|---|
| `n_output(S)` | number of network outputs for S target columns |
| `log_lik(out, data, offset)` | log-likelihood per chain, shape `[chains]` (differentiable) |
| `mean(out, offset)` | expected target, used by `predict_mean` |
| `sample(out, offset, generator)` | one draw of the target, used by `predict_sample` |
| `summaries(out)`, `summary_names()` | per-target summaries over the rows, used for R-hat and ESS |

Only `n_output` and `log_lik` are needed to sample; the others enable
predictions and diagnostics. Optional: `prepare` (convert the targets, e.g.
class labels), `supports_offset` and `aggregate` (exposure offsets and summed
rates, as in the count likelihoods). For example, Student-t noise for robust
regression:

```python
import torch
from ptbnn import Likelihood

class StudentT(Likelihood):
    name = "student_t"

    def __init__(self, df=3.0):
        self.df = df

    def n_output(self, S):
        return 2 * S                                    # location and log scale

    def log_lik(self, out, data, offset=None):
        S = out.shape[-1] // 2
        ll = torch.distributions.StudentT(self.df, out[..., :S], torch.exp(out[..., S:]),
                                          validate_args=False).log_prob(data["Y"])
        if data["mask"] is not None:
            ll = ll * data["mask"]
        return ll.sum((-1, -2))

    def mean(self, out, offset=None):
        return out[..., : out.shape[-1] // 2]

post = ptbnn.fit(X, Y, likelihood=StudentT(df=3))
```

`examples/custom_likelihood.py` has the full class, with `sample` and
`summaries`, and compares it with a Gaussian likelihood on data with outliers.

## Lower-level interface

`ptbnn.fit` wraps these pieces, which can be used directly. For example, to
write per-chain logs readable by Tracer, or to run your own sampling loop:

```python
from ptbnn import BNN, TracerLogger, find_map, run_hmc

model = BNN(X, Y, "negbin", n_nodes=[32, 16], offset=log_exposure)
theta0 = find_map(model, model.net.init_params(4, dtype=model.dtype), n_steps=1000)
log = TracerLogger(model, "run", target_names=["y1", "y2"], n_chains=4, log_f=10)
res = run_hmc(model.log_prob_and_grad, theta0, n_warmup=2000, n_samples=1000, logger=log)
log.close()
```

`run_hmc` samples any differentiable target: it takes a function mapping
`theta [chains, D]` to `(log_prob [chains], grad [chains, D])`.

## Sampler

- **Algorithm:** multi-chain HMC with a fixed number of leapfrog steps, vectorised over chains.
- **Step size:** tuned per chain by dual averaging (Hoffman & Gelman 2014), and jittered
  by ±20% during sampling.
- **Mass matrix:** diagonal, one per chain, estimated in Stan-style warmup windows.
- **Starting points:** from a short MAP fit with Adam. A full MAP fit lies far above
  the typical posterior density and slows down warmup.
- **Diagnostics:** rank-normalised split R-hat and bulk ESS (Vehtari et al. 2021),
  identical to ArviZ. `Posterior.diagnostics()` reports them for the
  log-likelihood and for summaries of the network output, not for the weights:
  hidden units can be permuted, so individual weights are not identifiable.

BNN posteriors have many modes, and chains often settle in different ones. An
R-hat somewhat above 1.01 on predictions therefore usually reflects
differences between modes rather than a failure of the sampler. Pooling the
chains is still the right way to make predictions.

**Known limitation.** The number of leapfrog steps is fixed, not tuned. Some
posteriors need long trajectories to mix. In the validation, the Gaussian
likelihood with learned noise was under-mixed with 30 steps, and its posterior
intervals were too narrow (`validation/README.md`). If ESS is low, increase
`leapfrog`; a longer run alone helps little. Automatic trajectory-length tuning
(e.g. ChEES-HMC) is planned.

## Numerics

The negative binomial log-likelihood is computed from log n (the log of the
size), so it stays accurate in float32 over the whole range, from strongly
overdispersed counts to the Poisson limit, with finite gradients everywhere.
Rates aggregated over rows use a max-shift, so large log rates do not overflow.

## Validation

`validation/` contains:

- **Known distributions:** the sampler recovers exact moments of distributions
  with known answers (badly scaled and correlated Gaussians, a banana-shaped
  distribution, Student-t), and its R-hat and ESS match ArviZ.
- **Simulation-based calibration** for each likelihood.

See `validation/README.md`. `tests/` holds the unit tests (`pytest`), which
check every likelihood against scipy, the gradients, numerical edge cases, the
sampler, the high-level interface and saving and loading.

## License

GNU Lesser General Public License v2.1 (see `LICENSE`).
