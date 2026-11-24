"""Device and precision helpers."""

import torch


def pick_device(device="auto"):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple GPU: float32 only, but ~6x faster than the CPU on these data
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def pick_dtype(dtype, device):
    if dtype == "auto":
        return torch.float64 if device.type == "cpu" else torch.float32
    dt = {"float32": torch.float32, "float64": torch.float64}[dtype]
    if device.type == "mps" and dt == torch.float64:
        raise ValueError("MPS does not support float64: use --dtype float32")
    return dt
