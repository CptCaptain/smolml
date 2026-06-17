"""Device auto-detection: cuda > mps > cpu.

FLOPs (the metric) are hardware-agnostic, so the device only affects wall-clock,
never the score. Pick the fastest backend available, allow an explicit override.
"""

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or an explicit `prefer` override.

    Priority: cuda > mps > cpu. `prefer` (e.g. "cpu") forces a choice — useful
    for deterministic tests and CI on CPU-only hosts.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
