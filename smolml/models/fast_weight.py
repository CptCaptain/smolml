"""Fast-weight associative memory over a frozen slow transformer core (Task A.1).

The first Source-(iv) candidate. The bet: rote memorization is expensive for
gradient descent (many steps to push a literal pattern into weights) but nearly
free for a **fast-weight associative memory** — a single gradient-free outer-product
write per item. So pair a *slow* gradient-trained core (which learns the
regularities and generalizes) with a *fast* associative store (which soaks up the
rote bits the instant it sees them). If memorization moves from "expensive
gradient steps" to "one cheap write," the gradient FLOPs all buy generalization →
more loss-reduction per FLOP. See ``docs/learning/concepts/fast-weight-memory.md``.

Division of labor
-----------------
- **Slow core** — the existing :class:`~smolml.models.transformer.Transformer`,
  pretrained by the default backprop ``train_step`` on the prior corpus and
  **frozen** at eval. ``forward``/``flops`` delegate to it, so amortized
  pretraining and its FLOP accounting are exactly the baseline's (the memory plays
  no part in training).
- **Fast memory** — a fixed-size linear associative store ``M`` of shape
  ``(d_model, vocab)``, gradient-free, **reset per stream** and written **online**
  during prequential eval. A write is a rank-1 Hebbian outer product
  ``M <- decay*M + gain*(key (X) e_byte)``; a read is a matvec ``key @ M``. Both are
  matmuls, charged honestly through :mod:`smolml.flops` and returned by ``step``.

Addressing (design decision: soft recall on *centered* hidden-state keys)
-------------------------------------------------------------------------
Soft / linear associative recall keyed on the frozen core's hidden state, not exact
byte-match — so an exact context repeat recalls strongly while a near-repeat recalls
gracefully (the division of labor: the core generalizes, the memory does rote).
**But raw transformer hidden states are severely anisotropic** — measured adjacent
cosine ~1.0, i.e. every context maps to nearly the same direction, which gives an
associative store no addressing discrimination and lets crosstalk dominate. The fix
is to key on the **centered** residual ``h - mu`` (``mu`` a per-stream running mean),
which strips the shared common-mode component and leaves what is *distinctive* about
this context; the residual is then L2-normalized so recall is a bounded cosine
similarity. Recall logits are ``recall[b] = sum_{i: y_i=b} (k . k_i)`` — the summed
similarity to every past context that was followed by byte ``b``.

Capacity & eviction (design decision: superposition + exponential decay)
------------------------------------------------------------------------
``M`` is a *superposition* (a sum of rank-1 writes), so it is fixed-size by
construction — every write lands in the same ``d_model x vocab`` matrix regardless
of how many items were stored. FIFO/LRU slot eviction does not apply; the natural
forgetting policy for a superposition is **exponential decay** ``M <- decay*M`` each
write (``memory_decay`` in (0, 1]). Decay bounds the store's norm on a long stream
(geometric sum) and recency-weights it, so stale crosstalk fades instead of
accumulating without limit.

Combining the two predictors (design decision: bounded probability mixing)
--------------------------------------------------------------------------
Adding ``gamma * recall`` straight into the core logits is **unbounded** and blows
up a confident core when recall is noisy (additive recall took a 6.0-bpb core to
25+ bpb in testing). Instead we mix the two as *distributions*, which bounds the
worst-case damage:

    p = (1 - a) * softmax(core_logits) + a * softmax(beta * recall),  logits = log p

with a **confidence gate** ``a = memory_alpha * max_b softmax(beta*recall)_b``: the
memory speaks loudly only when its recall is peaked (a true match) and stays silent
on diffuse crosstalk, so it helps on repeats without corrupting novel predictions.

FLOP honesty (ADR 0004 — the whole point)
------------------------------------------
The memory's dominant compute is matmuls, charged at their **true performed cost**
(charge == reality, asserted by tests), never undercounted:
- read ``key @ M``: dense matvec ``(1,d)@(d,V)`` -> ``matmul_flops(1, V, d) = 2dV``.
- write ``M += gain*(key (X) e_byte)``: a **dense** outer product (``torch.outer``
  materializes all ``d*V`` products; the accumulate adds ``d*V``) ->
  ``matmul_flops(d, V, 1) = 2dV``. The one-hot value is *not* exploited to fake a
  cheap write — that would be the exact "elementwise work scored as free" cheat the
  instrument guards against.
- decay ``M *= decay``: a dense elementwise multiply over ``d*V`` ->
  ``pointwise_flops(d*V)`` (charged though dominated).
- key centering/normalization and the softmax mixing are small ``O(d)``/``O(V)``
  elementwise ops, dominated by the ``O(dV)`` matmuls; charged nominally in good
  faith so the read/write matmuls remain the honest dominant cost.

Forward FLOPs carry prediction (core decode + read + mix); backward FLOPs carry the
online adaptation (decay + write), so ``backward > 0`` marks counted continual
learning — the convention the prequential guard tests use.
"""

from dataclasses import dataclass

import torch

from smolml.data.corpus import VOCAB_SIZE
from smolml.flops import FlopBreakdown, matmul_flops, pointwise_flops
from smolml.models.registry import DecodeState, LanguageModel, register_model
from smolml.models.transformer import Transformer, TransformerConfig


@dataclass
class FastWeightConfig:
    """Slow-core (transformer) hyperparameters plus fast-memory hyperparameters.

    The core fields mirror :class:`~smolml.models.transformer.TransformerConfig`
    (so the slow core is the baseline architecture and the comparison isolates the
    memory). The memory fields are gradient-free scalars, not trained parameters.
    """

    # slow-core (transformer) hyperparameters
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int | None = None
    max_seq_len: int = 256
    vocab_size: int = VOCAB_SIZE
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    # fast associative-memory hyperparameters (gradient-free)
    memory_alpha: float = 0.6  # max mixture weight of recall, gated by its confidence
    memory_beta: float = 2.0  # recall softmax temperature
    memory_decay: float = 0.999  # forgetting factor in (0, 1] (1.0 = no forgetting)
    memory_write_gain: float = 1.0  # outer-product write magnitude
    memory_center_ema: float = 0.01  # running-mean rate for key centering (anisotropy fix)

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if not 0.0 < self.memory_decay <= 1.0:
            raise ValueError(f"memory_decay must be in (0, 1], got {self.memory_decay}")
        if not 0.0 <= self.memory_alpha <= 1.0:
            raise ValueError(f"memory_alpha must be in [0, 1], got {self.memory_alpha}")
        if self.memory_beta < 0.0:
            raise ValueError(f"memory_beta must be non-negative, got {self.memory_beta}")
        if not 0.0 < self.memory_center_ema <= 1.0:
            raise ValueError(f"memory_center_ema must be in (0, 1], got {self.memory_center_ema}")

    def core_config(self) -> TransformerConfig:
        return TransformerConfig(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            vocab_size=self.vocab_size,
            rope_base=self.rope_base,
            dropout=self.dropout,
            tie_embeddings=self.tie_embeddings,
        )


@dataclass
class _FastWeightCache:
    """Per-stream fast-weight state threaded through :meth:`FastWeightMemory.step`.

    ``memory`` is the ``(d_model, vocab)`` associative store; ``center_mean`` is the
    running hidden-state mean used to center keys; ``pending_key`` is the key that
    produced the *previous* prediction (written against the byte revealed next step);
    ``kv`` is the slow core's per-layer KV cache while in the growing regime (``None``
    once the stream exceeds the context window and decode switches to a bounded
    windowed recompute).
    """

    memory: torch.Tensor
    center_mean: torch.Tensor | None
    pending_key: torch.Tensor | None
    kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None


@register_model("fast_weight")
class FastWeightMemory(LanguageModel):
    """Frozen slow transformer core + an online fast-weight associative memory."""

    def __init__(self, config: FastWeightConfig):
        super().__init__()
        self.config = config
        # The only trained parameters are the slow core's; the memory is a
        # gradient-free runtime tensor (lives in DecodeState, never an nn.Parameter).
        self.core = Transformer(config.core_config())

    # --- Amortized path: the slow core only (memory is eval-only) ---------------

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.core(idx)

    def flops(self, seq_len: int) -> FlopBreakdown:
        return self.core.flops(seq_len)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "FastWeightMemory":
        return cls(FastWeightConfig(**config))

    # --- Memory FLOP accounting (charge == reality; see module docstring) -------

    def _memory_read_flops(self) -> int:
        """Forward FLOPs of a read + combine: the matvec ``key @ M`` plus the small
        (matmul-dominated) key centering/normalization and softmax mixing."""
        d, v = self.config.d_model, self.config.vocab_size
        read = matmul_flops(1, v, d)  # key @ M : (1,d) @ (d,V) -> (V,)
        keybuild = pointwise_flops(d, per_elem=6)  # EMA-center (2) + subtract (1) + L2-norm (3)
        mix = pointwise_flops(v, per_elem=8)  # 2 softmaxes + confidence gate + convex mix + log
        return read + keybuild + mix

    def _memory_write_flops(self) -> int:
        """Backward (adaptation) FLOPs of one write: dense outer-product update plus
        the dense decay multiply (when forgetting is enabled)."""
        d, v = self.config.d_model, self.config.vocab_size
        write = matmul_flops(d, v, 1)  # key (X) e_byte then accumulate: 2dV
        decay = pointwise_flops(d * v) if self.config.memory_decay < 1.0 else 0
        return write + decay

    # --- Prequential / online decode seam ---------------------------------------

    def init_prequential_state(self) -> DecodeState:
        device = self.core.rope_cos.device
        cache = _FastWeightCache(
            memory=torch.zeros(self.config.d_model, self.config.vocab_size, device=device),
            center_mean=None,
            pending_key=None,
            kv=[None] * self.config.n_layers,
        )
        return DecodeState(cache=cache)

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        cfg = self.config
        cache: _FastWeightCache = state.cache
        new_len = state.length + 1
        window_cap = cfg.max_seq_len
        window = [*state.tokens, revealed_byte][-window_cap:]

        memory = cache.memory
        backward = 0
        with torch.no_grad():
            # (1) Adapt: write the PREVIOUS prediction's key against the byte it was
            # predicting, now revealed. Uses only past/present bytes (no leakage).
            if cache.pending_key is not None:
                memory = self._write(memory, cache.pending_key, revealed_byte)
                backward = self._memory_write_flops()

            # (2) Frozen slow core: decode the revealed byte to a hidden state.
            if cache.kv is not None and new_len <= window_cap:
                if pos != new_len - 1:
                    raise ValueError(
                        f"step expects consecutive positions: pos={pos}, length={new_len - 1}"
                    )
                hidden, core_logits, new_kv = self._decode_incremental(revealed_byte, pos, cache.kv)
                core_fwd = self.core.decode_step_flops(new_len).forward
            else:
                hidden, core_logits = self._decode_window(window)
                new_kv = None
                core_fwd = self.core.flops(len(window)).forward

            # (3) Build a centered key, read the memory, mix recall into the logits.
            key, center_mean = self._make_key(hidden, cache.center_mean)
            recall = key @ memory  # (V,)
            next_logits = self._combine(core_logits, recall)

        forward = core_fwd + self._memory_read_flops()
        flops = FlopBreakdown(forward=forward, backward=backward)
        new_cache = _FastWeightCache(
            memory=memory, center_mean=center_mean, pending_key=key, kv=new_kv
        )
        new_state = DecodeState(tokens=window, cache=new_cache, length=new_len)
        return new_state, next_logits.detach(), flops

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        """Forward-only per-byte decode cost: the core's incremental decode plus the
        memory read (the write is adaptation, reported in ``step``'s backward)."""
        core = self.core.decode_step_flops(context_len).forward
        return FlopBreakdown(forward=core + self._memory_read_flops(), backward=0)

    # --- internals ---------------------------------------------------------------

    def _make_key(
        self, hidden: torch.Tensor, center_mean: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Center the hidden state by a running mean (anisotropy fix), then
        L2-normalize so recall is a bounded cosine similarity. Returns the key and
        the updated running mean (uses past/present bytes only — no leakage)."""
        ema = self.config.memory_center_ema
        new_mean = hidden.clone() if center_mean is None else center_mean.lerp(hidden, ema)
        residual = hidden - new_mean
        return residual / (residual.norm() + 1e-8), new_mean

    def _combine(self, core_logits: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
        """Confidence-gated probability mixture of the core and recall distributions.

        ``log((1-a)*softmax(core) + a*softmax(beta*recall))`` with the mixture weight
        ``a = memory_alpha * max(softmax(beta*recall))`` gated by recall confidence,
        so the bounded mixture can never blow up a confident core."""
        cfg = self.config
        p_core = torch.softmax(core_logits, dim=-1)
        p_mem = torch.softmax(cfg.memory_beta * recall, dim=-1)
        a = cfg.memory_alpha * p_mem.max()
        return torch.log((1.0 - a) * p_core + a * p_mem)

    def _write(self, memory: torch.Tensor, key: torch.Tensor, byte: int) -> torch.Tensor:
        """``M <- decay*M + gain*(key (X) e_byte)`` — a dense rank-1 outer product.

        The value vector is a one-hot of the revealed byte, but the outer product is
        materialized densely so the performed FLOPs equal the charged matmul cost
        (no sparsity is exploited to fake a cheap write)."""
        cfg = self.config
        if cfg.memory_decay < 1.0:
            memory = memory * cfg.memory_decay
        value = torch.zeros(cfg.vocab_size, device=memory.device)
        value[byte] = cfg.memory_write_gain
        return memory + torch.outer(key, value)

    def _decode_incremental(
        self,
        revealed_byte: int,
        pos: int,
        kv: list[tuple[torch.Tensor, torch.Tensor] | None],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Growing-regime KV-cache decode of one byte; returns (hidden, logits, kv).

        Replays the core's incremental decode (reusing its blocks/norm/head/rope) so
        the cost equals ``core.decode_step_flops`` exactly, while also exposing the
        final hidden state the memory keys on."""
        core = self.core
        device = core.rope_cos.device
        x = core.tok_emb(torch.tensor([[revealed_byte]], dtype=torch.long, device=device))
        cos, sin = core.rope_cos[pos : pos + 1], core.rope_sin[pos : pos + 1]
        new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block, layer_kv in zip(core.blocks, kv, strict=True):
            x, nkv = block.decode_step(x, cos, sin, layer_kv)
            new_kv.append(nkv)
        hidden = core.norm_f(x)[0, -1]
        return hidden, core.head(hidden), new_kv

    def _decode_window(self, window: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Sliding-regime full recompute over the last ``window`` bytes; returns the
        final-position (hidden, logits). Bounded memory, length-matched, exact."""
        core = self.core
        device = core.rope_cos.device
        idx = torch.tensor([window], dtype=torch.long, device=device)
        t = idx.shape[1]
        x = core.tok_emb(idx)
        cos, sin = core.rope_cos[:t], core.rope_sin[:t]
        for block in core.blocks:
            x = block(x, cos, sin)
        hidden = core.norm_f(x)[0, -1]
        return hidden, core.head(hidden)
