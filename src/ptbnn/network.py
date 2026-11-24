"""Batched multilayer perceptron: one flat weight vector per chain."""

import numpy as np
import torch
import torch.nn.functional as F

ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "swish": F.silu,
    "silu": F.silu,
    "softplus": F.softplus,
    "leaky_relu": lambda z: F.leaky_relu(z, 0.1),
}


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

class BatchedMLP:
    """MLP whose parameters are one flat vector per chain: theta has shape [C, D].

    forward(theta, X) -> [C, N, n_output]
    """

    def __init__(self, n_features, n_nodes, n_output, activation="relu",
                 hidden_bias=False, output_bias=True):
        self.n_features = n_features
        self.n_nodes = list(n_nodes)
        self.n_output = n_output
        self.activation = activation
        self.act = ACTIVATIONS[activation]
        self.hidden_bias = hidden_bias
        self.output_bias = output_bias

        sizes = [n_features] + self.n_nodes + [n_output]
        self.layers = []  # (w_slice, w_shape, b_slice or None)
        i = 0
        for k in range(len(sizes) - 1):
            n_in, n_out = sizes[k], sizes[k + 1]
            w_sl = slice(i, i + n_in * n_out)
            i += n_in * n_out
            last = k == len(sizes) - 2
            has_b = output_bias if last else hidden_bias
            b_sl = None
            if has_b:
                b_sl = slice(i, i + n_out)
                i += n_out
            self.layers.append((w_sl, (n_out, n_in), b_sl))
        self.n_params = i

    def init_params(self, n_chains, init_std=0.1, generator=None, dtype=torch.float64, device="cpu"):
        theta = torch.randn(n_chains, self.n_params, generator=generator, dtype=dtype) * init_std
        return theta.to(device)

    def forward(self, theta, X):
        C = theta.shape[0]
        h = X  # [N, F], broadcast over chains in the first matmul
        n_layers = len(self.layers)
        for k, (w_sl, (n_out, n_in), b_sl) in enumerate(self.layers):
            W = theta[:, w_sl].reshape(C, n_out, n_in)
            h = torch.matmul(h, W.transpose(1, 2))  # [C, N, n_out]
            if b_sl is not None:
                h = h + theta[:, b_sl].unsqueeze(1)
            if k < n_layers - 1:
                h = self.act(h)
        return h

    def unflatten(self, theta_1d):
        """Return weights as a list of numpy arrays in npBNN layout
        (n_out, n_in [+1]) with the bias, if any, as the first column."""
        theta_1d = np.asarray(theta_1d)
        out = []
        for w_sl, shape, b_sl in self.layers:
            W = theta_1d[w_sl].reshape(shape)
            if b_sl is not None:
                W = np.hstack([theta_1d[b_sl][:, None], W])
            out.append(W)
        return out
