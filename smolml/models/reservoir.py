"""Echo-state reservoir + trained linear readout (Task C.A.1, minimal-organism control).

The bio thesis (ADR 0007 + ADR 0003): a minimal organism with a **fixed** recurrent
substrate and a cheap, locally-trained readout can do in-context control. A transformer
recomputes ``O(n_layers·d² + context·d)`` per decode step; a reservoir (echo-state
network) rolls its state in ``O(d_res²)`` per step — **independent of context length** —
and only a **linear readout** is trained. The loss-per-FLOP edge is structural: the
expensive recurrent dynamics are **never trained** (0 backward), so the learning cost is
just the readout's ``dW_out`` outer product.

Per token (the input is the token id):

    x_t    = W_in[:, token]                          # fixed-random column = a gather
    pre    = x_t + W_res @ h_{t-1}                    # the dominant matvec, O(d_res²)
    h_t    = (1 - leak) * h_{t-1} + leak * tanh(pre)  # leaky-integrator state
    logits = W_out @ h_t + b_out                      # the TRAINED readout, full vocab

``W_in`` and ``W_res`` are ``nn.Parameter(requires_grad=False)`` (seeded random, ``W_res``
rescaled to spectral radius ``rho`` for the echo-state property). They are **counted in
``num_params``** (memory parity with the transformer bar) but **excluded from the
optimizer** and **charged 0 backward**. ``W_out``/``b_out`` are the only trainable params.

FLOP accounting (honest, via :mod:`smolml.flops` — the product)
---------------------------------------------------------------
- Per-token forward = ``matmul_flops(1, d_res, d_res)`` (``W_res·h``) +
  ``matmul_flops(1, vocab, d_res)`` (readout) + the gather/add/leak+tanh/bias pointwise
  work (``O(d_res)``). ``flops(T).forward = T ×`` that.
- ``flops(T).backward = T × matmul_flops(1, vocab, d_res)`` — the readout ``dW_out`` outer
  product **only**. The recurrence builds **no autograd graph** (frozen params + a constant
  ``h_0`` ⇒ ``h`` does not require grad), so backprop reaches ``W_out``/``b_out`` and stops;
  there is no ``dh`` matmul. This is why :meth:`flops` must NOT use
  ``FlopBreakdown.from_forward`` (that would charge a full 2× backward, which is wrong here).
- ``step``/``decode_step_flops`` = per-token forward, backward 0 (the readout is frozen
  after distillation), ``O(d_res²)`` and context-independent (``h`` is the bounded memory).
"""

from dataclasses import dataclass

import torch
from torch import nn

from smolml.data.corpus import VOCAB_SIZE
from smolml.flops import FlopBreakdown, gather_flops, matmul_flops, pointwise_flops
from smolml.models.registry import DecodeState, LanguageModel, register_model


@dataclass
class ReservoirConfig:
    """Reservoir hyperparameters. ``vocab_size``/``max_seq_len`` are injected by
    ``distill_train_run``; the rest size and seed the frozen echo-state core.

    ``d_res=374`` lands ``num_params`` just under the transformer bar's 148,608 at the
    control vocab of 11 (8 concentration levels + 3 actions): 374² + 2·11·374 + 11 =
    148,115 params (so the regret-per-FLOP comparison is fair on the param axis).
    """

    d_res: int = 374
    leak: float = 0.6
    spectral_radius: float = 0.9
    seed: int = 0
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 256

    def __post_init__(self):
        if not 0.0 < self.leak <= 1.0:
            raise ValueError(f"leak must be in (0, 1], got {self.leak}")
        if self.spectral_radius <= 0.0:
            raise ValueError(f"spectral_radius must be positive, got {self.spectral_radius}")


class _ReservoirCore(nn.Module):
    """The shared **frozen** echo-state substrate: a fixed input embedding ``W_in`` and a
    recurrent matrix ``W_res`` rescaled to spectral radius ``rho`` (the echo-state
    property — past inputs fade, so ``h`` is a bounded, fading summary of the trajectory).

    Both matrices are ``nn.Parameter(requires_grad=False)``: counted in ``num_params`` for
    memory parity, but never trained. The recurrence runs under ``no_grad`` so it builds no
    autograd graph — a downstream readout's backward reaches only the readout's weights.
    Reused unchanged by the (C.A.1b) online-readout sibling.
    """

    def __init__(self, d_res: int, vocab_size: int, spectral_radius: float, seed: int):
        super().__init__()
        self.d_res = d_res
        gen = torch.Generator().manual_seed(seed)
        w_in = torch.randn(d_res, vocab_size, generator=gen)
        w_res = torch.randn(d_res, d_res, generator=gen)
        # Rescale to spectral radius rho: divide by the largest |eigenvalue|, times rho.
        radius = torch.linalg.eigvals(w_res).abs().max()
        w_res = w_res * (spectral_radius / radius)
        self.W_in = nn.Parameter(w_in, requires_grad=False)
        self.W_res = nn.Parameter(w_res, requires_grad=False)

    def initial_state(self) -> torch.Tensor:
        """The constant ``h_0 = 0`` (a ``d_res`` vector) that seeds every rollout."""
        return torch.zeros(self.d_res, device=self.W_in.device, dtype=self.W_in.dtype)

    @torch.no_grad()
    def run(self, idx: torch.Tensor, leak: float) -> torch.Tensor:
        """Sequential recurrence over ``idx`` ``(B, T)`` from ``h_0 = 0`` → states
        ``(B, T, d_res)``. Runs under ``no_grad`` so the returned states carry no graph."""
        b, t = idx.shape
        h = torch.zeros(b, self.d_res, device=self.W_in.device, dtype=self.W_in.dtype)
        states = []
        for pos in range(t):
            x = self.W_in.index_select(1, idx[:, pos]).t()  # (B, d_res) gather of columns
            h = (1.0 - leak) * h + leak * torch.tanh(x + h @ self.W_res.t())
            states.append(h)
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def fold(self, h: torch.Tensor, token: int, leak: float) -> torch.Tensor:
        """Advance one token: fold ``token`` into state ``h`` ``(d_res,)`` → ``h_new``."""
        x = self.W_in[:, token]  # (d_res,) fixed-embedding column
        return (1.0 - leak) * h + leak * torch.tanh(x + self.W_res @ h)

    def recurrence_flops(self) -> int:
        """Forward FLOPs of one recurrence step: the ``W_res·h`` matvec plus the column
        gather, the residual add, and the leaky-integrator ``tanh`` update."""
        d = self.d_res
        return (
            matmul_flops(1, d, d)  # W_res @ h : the dominant O(d_res²) matvec
            + gather_flops(d)  # x = W_in[:, token] : a fixed-embedding column gather
            + pointwise_flops(d)  # pre = x + W_res@h : the residual add
            + pointwise_flops(d, per_elem=4)  # (1-leak)*h + leak*tanh(pre): tanh + 2 scales + add
        )


@register_model("reservoir")
class Reservoir(LanguageModel):
    """Frozen echo-state core + a distilled-frozen linear readout (C.A.1).

    Trained by ``distill_train_run(model="reservoir", …)`` exactly like the transformer
    bar; the only trained tensors are ``readout.weight``/``bias``. In-context adaptation at
    eval is the reservoir **state** echoing the recent trajectory — the readout is frozen
    after distillation (no weight change in :meth:`step`).
    """

    def __init__(self, config: ReservoirConfig):
        super().__init__()
        self.config = config
        self.core = _ReservoirCore(
            config.d_res, config.vocab_size, config.spectral_radius, config.seed
        )
        self.readout = nn.Linear(config.d_res, config.vocab_size)
        # Deterministic readout init from a stream distinct from the core's (so a fixed
        # seed pins the whole model's forward output for the determinism test).
        gen = torch.Generator().manual_seed(config.seed + 1)
        with torch.no_grad():
            self.readout.weight.normal_(0.0, 0.02, generator=gen)
            self.readout.bias.zero_()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        states = self.core.run(idx, self.config.leak)  # (B,T,d_res), no grad through core
        return self.readout(states)  # (B,T,vocab); backward flows only into the readout

    def _per_token_forward(self) -> int:
        """Per-token forward FLOPs: the recurrence + the readout matvec + the bias add."""
        d, v = self.config.d_res, self.config.vocab_size
        return self.core.recurrence_flops() + matmul_flops(1, v, d) + pointwise_flops(v)

    def _readout_backward(self) -> int:
        """Per-token backward FLOPs: the ``dW_out = dlogits ⊗ h`` outer product plus the
        elementwise ``db_out`` bias gradient.

        ``h`` does not require grad (frozen core + constant ``h_0``), so backprop computes
        ``dW_out``/``db_out`` and stops — no ``dh`` matmul, no gradient into the recurrence.
        Hence backward ≠ 2× forward."""
        v = self.config.vocab_size
        return matmul_flops(1, v, self.config.d_res) + pointwise_flops(v)

    def flops(self, seq_len: int) -> FlopBreakdown:
        return FlopBreakdown(
            forward=seq_len * self._per_token_forward(),
            backward=seq_len * self._readout_backward(),
        )

    def configure_optimizer(
        self, *, lr: float, weight_decay: float, betas: tuple[float, float]
    ) -> torch.optim.Optimizer:
        """AdamW over the trainable readout only; the frozen core is excluded."""
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=lr, betas=betas, weight_decay=weight_decay)

    # --- Prequential / online decode seam: h IS the bounded memory --------------

    def init_prequential_state(self) -> DecodeState:
        return DecodeState(cache=self.core.initial_state())

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        """Fold one token into the reservoir state and read out the next distribution.

        ``state.cache`` is the ``d_res`` reservoir state (the entire bounded memory);
        ``state.tokens`` is ignored (no token window is needed). Forward-only: the readout
        is frozen, so backward is 0 and the cost is ``O(d_res²)``, context-independent."""
        h_new = self.core.fold(state.cache, revealed_byte, self.config.leak)
        with torch.no_grad():
            logits = self.readout(h_new).detach()
        flops = FlopBreakdown(forward=self._per_token_forward(), backward=0)
        return DecodeState(cache=h_new, length=state.length + 1), logits, flops

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        """Forward-only per-step cost: ``O(d_res²)``, independent of ``context_len``."""
        return FlopBreakdown(forward=self._per_token_forward(), backward=0)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "Reservoir":
        return cls(ReservoirConfig(**config))
