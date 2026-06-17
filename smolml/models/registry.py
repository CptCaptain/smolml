"""Model interface + registry — the plug-in contract for candidates.

A candidate mechanism becomes runnable under the harness with **zero harness
changes** by doing three things:

1. subclass :class:`LanguageModel` and implement ``forward``, ``flops``, and the
   ``from_config`` classmethod;
2. report its compute through the shared :mod:`smolml.flops` primitives (so the
   referee is identical for every entrant);
3. decorate the class with ``@register_model("name")``.

A backprop model implements only those three. A **non-backprop** candidate (the
point of the project — ADR 0003) additionally overrides :meth:`LanguageModel.train_step`
(and optionally :meth:`LanguageModel.configure_optimizer`) to express its own
learning rule and its own honest FLOP cost, instead of being charged the default
2x backprop tax. Either way the harness only speaks to a model through this
interface, so it never needs to know the mechanism behind the name.
"""

import abc

import torch
import torch.nn.functional as F
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
        """Analytic FLOPs to process **one** sequence of ``seq_len`` tokens.

        Returns a :class:`~smolml.flops.FlopBreakdown`: ``forward`` is the
        forward-pass cost; ``backward`` is *this model's* own learning/update
        cost per sequence (for backprop models, 2x forward). The default
        :meth:`train_step` charges ``flops(seq_len).scale(batch)``; the harness
        accumulates the value :meth:`train_step` returns, so accounting follows
        the real mechanism rather than any hardcoded multiplier. Forward-only
        callers read ``.forward``.
        """

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict[str, object]) -> "LanguageModel":
        """Build the model from a plain config dict (as stored in run logs)."""

    def num_params(self) -> int:
        """Total trainable parameters (shared/tied tensors counted once)."""
        return sum(p.numel() for p in self.parameters())

    def configure_optimizer(
        self, *, lr: float, weight_decay: float, betas: tuple[float, float]
    ) -> torch.optim.Optimizer:
        """Build the optimizer for this model. Default: AdamW over all params.

        Override for a mechanism that updates differently (a non-backprop
        candidate may return a trivial optimizer it does not use).
        """
        return torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        *,
        grad_clip: float = 1.0,
    ) -> tuple[torch.Tensor, FlopBreakdown]:
        """Run ONE learning step on ``batch`` and report the FLOPs it spent.

        ``batch`` is ``(x, y)`` of shape ``(B, T)``. Returns ``(loss, flops)``:
        ``loss`` is the mini-batch cross-entropy in **nats** (the harness converts
        to bits/byte for logging) and ``flops`` is the compute actually spent. The
        harness accumulates ``flops`` against the budget, so a candidate's
        accounting follows its real learning rule.

        Default = standard backprop: forward, cross-entropy, backward, grad-clip,
        optimizer step, charging ``flops(T).scale(B)``. This is the seam a
        non-backprop candidate (ADR 0003) overrides to express its own learning
        and its own honest cost instead of being charged the 2x backprop tax.
        """
        x, y = batch
        b, t = x.shape
        optimizer.zero_grad(set_to_none=True)
        logits = self(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        return loss, self.flops(t).scale(b)


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


def build_model(name: str, config: dict[str, object]) -> LanguageModel:
    """Construct a registered model from its name and config dict."""
    return get_model(name).from_config(config)
