"""
ptBNN: Bayesian neural networks in PyTorch, sampled with Hamiltonian Monte Carlo.

Model
  - fully connected network (ReLU by default), with or without biases in the
    hidden layers (npBNN's use_bias_node=-1 = hidden_bias=False)
  - Normal(0, p_scale) prior on every weight
  - likelihoods: Gaussian (regression, fixed or learned noise), Poisson and
    negative binomial (counts, with optional offset and aggregation of rates
    over windows of rows), Categorical (classification)

Sampler
  - several chains run as one batched computation, on CPU, CUDA or Apple MPS
  - per-chain step size tuned by dual averaging (Hoffman & Gelman 2014)
  - per-chain diagonal mass matrix estimated in Stan-style warmup windows
  - step-size jitter during sampling; optional short MAP (Adam) initialisation
  - rank-normalised split R-hat and bulk ESS (Vehtari et al. 2021)
"""

from .api import Posterior, fit, partial_dependence
from .diagnostics import split_rhat_ess
from .hmc import DualAveraging, find_initial_step_size, leapfrog, run_hmc, warmup_windows
from .likelihood import (Categorical, Gaussian, Likelihood, NegativeBinomial, Poisson, get_likelihood,
                         lgamma_diff, log_softplus_neg, negbin_loglik)
from .model import BNN, NegBinBNN, find_map
from .network import ACTIVATIONS, BatchedMLP
from .tracer import TracerLogger
from .utils import pick_device, pick_dtype

__version__ = "0.1.0"

__all__ = [
    "ACTIVATIONS", "BNN", "BatchedMLP", "Categorical", "DualAveraging", "Gaussian", "Likelihood",
    "NegBinBNN", "NegativeBinomial", "Poisson", "Posterior", "TracerLogger", "find_initial_step_size",
    "find_map", "fit", "get_likelihood", "leapfrog", "lgamma_diff", "log_softplus_neg",
    "negbin_loglik", "partial_dependence", "pick_device", "pick_dtype", "run_hmc", "split_rhat_ess",
    "warmup_windows",
]
