"""Model interface + registry — the plug-in contract for candidates.

A candidate mechanism becomes runnable under the harness with **zero harness
changes** by doing three things:

1. subclass :class:`LanguageModel` and implement ``forward``, ``flops``, and the
   ``from_config`` classmethod;
2. report its compute through the shared :mod:`smolml.flops` primitives (so the
   referee is identical for every entrant);
3. decorate the class with ``@register_model("name")``.

The harness only ever speaks to a model through this interface, so it never needs
to know what mechanism sits behind the name.
"""

import abc

import torch
from torch import nn

from smolml.flops import FlopBreakdown


class LanguageModel(nn.Module, abc.ABC):
    """Byte-level next-token predictor with an honest, self-reported FLOP cost.

    Concrete models keep their hyperparameters on ``self.config`` (any dataclass).
    """

    @abc.abstractmethod
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Map token ids ``(batch, seq_len)`` (int64) to next-byte **logits**
        ``(batch, seq_len, 256)`` (float). One distribution over the 256 byte
        values per position; logits, not probabilities (the loss applies softmax).
        """

    @abc.abstractmethod
    def flops(self, seq_len: int) -> FlopBreakdown:
        """Analytic matmul FLOPs to process **one** sequence of ``seq_len`` tokens.

        Returns a :class:`~smolml.flops.FlopBreakdown` (forward + backward). The
        harness multiplies by batch size and accumulates. Forward-only callers
        (inference, the future prequential mode) read ``.forward``.
        """

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict) -> "LanguageModel":
        """Build the model from a plain config dict (as stored in run logs)."""

    def num_params(self) -> int:
        """Total trainable parameters (shared/tied tensors counted once)."""
        return sum(p.numel() for p in self.parameters())


_REGISTRY: dict[str, type[LanguageModel]] = {}


def register_model(name: str):
    """Class decorator: register a :class:`LanguageModel` under ``name``."""

    def decorator(cls: type[LanguageModel]) -> type[LanguageModel]:
        if not issubclass(cls, LanguageModel):
            raise TypeError(f"{cls.__name__} must subclass LanguageModel")
        if name in _REGISTRY:
            raise ValueError(f"model name already registered: {name!r}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str) -> type[LanguageModel]:
    """Look up a registered model class by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Names of all registered models."""
    return sorted(_REGISTRY)


def build_model(name: str, config: dict) -> LanguageModel:
    """Construct a registered model from its name and config dict."""
    return get_model(name).from_config(config)
