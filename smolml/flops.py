"""The shared, honest FLOP counter — the critical correctness surface.

Every model reports its compute through *these* primitives so comparisons are
fair across mechanisms, machines, and frameworks. The counter is **analytic**
(computed from shapes, not profiled): deterministic, hardware-independent, and
exactly hand-checkable. A bug here silently invalidates every comparison, so the
accounting is spelled out and unit-tested against hand-computed values.

Conventions (assumptions made explicit)
---------------------------------------
- **MAC = 2 FLOPs.** A multiply-accumulate is 1 multiply + 1 add. We count both.
- **Matmul (m,k)·(k,n) -> (m,n) costs ``2*m*n*k`` FLOPs.** Each of the ``m*n``
  outputs is a length-``k`` dot product = ``k`` MACs = ``2*k`` FLOPs.
- **What we count:** the compute that *dominates* a mechanism. For the transformer
  (and Phase-A fast-weight memory, whose reads/writes are outer-products/matvecs =
  matmuls) that is the matmuls — linear/projection layers (``O(tokens * d^2)``) and
  the attention score/value matmuls (``O(tokens^2 * d)``).
- **What we omit, and the condition for omitting it:** elementwise ops
  (activations, RMSNorm, residual adds, softmax normalization, RoPE rotations,
  dropout) and embedding gathers — omitted **only because they are dominated by
  the matmuls** here (``O(tokens*d)`` vs. ``O(tokens*d^2)``), and counting them
  exactly is framework-dependent without moving the metric. This omission is
  **conditional, not universal**: a mechanism whose dominant compute is *not*
  matmuls (e.g. the Task 0.3 online context-mixer — table lookups + logistic
  mixing) MUST charge that work via ``pointwise_flops``/``gather_flops`` below,
  or the instrument would silently score it as nearly free.
- **Backward = 2x forward (matmul FLOPs).** For ``Y = A·B`` where both operands
  feed gradients, backprop computes ``dA = dY·Bᵀ`` and ``dB = Aᵀ·dY`` — two
  matmuls of the same FLOP magnitude as the forward one. So a training step costs
  ``forward + 2*forward = 3*forward``. This is exactly the textbook ``C ≈ 6*N*D``
  rule (2 FLOPs/param/token forward, 4 backward) generalized to also charge the
  attention activation matmuls.

Extensibility toward Task 0.2 (do NOT build 0.2 here)
-----------------------------------------------------
``FlopBreakdown`` keeps ``forward`` and ``backward`` separate, and these
primitives are reusable, so the prequential/total-FLOP mode (ADR 0004) builds on
this foundation rather than replacing it:

- **training** step  -> ``forward + backward`` (``.total``),
- **inference** step -> ``forward`` only (no ``.backward``).

But it is an interface **extension**, not free: prequential prediction and
test-time adaptation are *context-length dependent*, so Task 0.2 will ADD methods
such as ``decode_step_flops(context_len)`` and ``adapt_step_flops(context_len)``
to the model interface (and generalize :meth:`LanguageModel.train_step` to an
``adapt`` path). Those methods are deliberately **not** implemented here.
"""

from dataclasses import dataclass

MAC_FLOPS: int = 2
"""FLOPs per multiply-accumulate (1 multiply + 1 add)."""

BACKWARD_MULTIPLIER: int = 2
"""Backward-pass matmul FLOPs as a multiple of forward-pass matmul FLOPs."""


def matmul_flops(m: int, n: int, k: int) -> int:
    """FLOPs for a dense matmul ``(m, k) @ (k, n) -> (m, n)``.

    ``m*n`` outputs, each a length-``k`` dot product (``k`` MACs). So
    ``MAC_FLOPS * m * n * k`` = ``2*m*n*k``.
    """
    return MAC_FLOPS * m * n * k


def linear_flops(tokens: int, in_features: int, out_features: int) -> int:
    """Forward FLOPs for a ``nn.Linear(in_features, out_features)`` applied to
    ``tokens`` rows (bias add is elementwise and ignored)."""
    return matmul_flops(tokens, out_features, in_features)


def causal_attention_flops(seq_len: int, d_model: int) -> int:
    """Forward matmul FLOPs for one causal self-attention layer over one sequence.

    Covers the two activation matmuls — scores ``Q·Kᵀ`` and value mixing
    ``softmax(scores)·V`` — summed over all query positions.

    With ``h`` heads of width ``d_head = d_model/h``, query position ``i`` attends
    keys ``0..i`` (causal), so the number of (query, key) pairs over the sequence
    is ``P = T*(T+1)/2``. Each pair contributes one length-``d_head`` dot product
    per head for ``Q·Kᵀ`` and one for value mixing. Summed over ``h`` heads the
    ``d_head`` factors recombine to ``d_model`` (so the count is **independent of
    head count**):

        Q·Kᵀ          = MAC_FLOPS * d_model * P
        softmax·V     = MAC_FLOPS * d_model * P
        attention     = 2 * MAC_FLOPS * d_model * P   (= 4 * d_model * P)
    """
    pairs = seq_len * (seq_len + 1) // 2
    return 2 * MAC_FLOPS * d_model * pairs


def pointwise_flops(n_elems: int, per_elem: int = 1) -> int:
    """FLOPs for an elementwise op over ``n_elems`` elements.

    ``per_elem`` is the arithmetic ops charged per element (e.g. 1 for an add or
    multiply; more for a fused expression). For matmul-dominated models this work
    is omitted as negligible; mechanisms whose dominant compute is elementwise
    (e.g. logistic mixing in a context-mixer) MUST charge it here.
    """
    return n_elems * per_elem


def gather_flops(n: int, cost_per_lookup: int = 1) -> int:
    """Nominal cost of ``n`` table lookups / gathers.

    A gather is a memory op with no multiply-add, so matmul-dominated models
    charge 0 for embedding lookups. But a lookup-*dominated* mechanism (e.g. the
    Task 0.3 context-mixer's order-k byte tables) would otherwise be scored free;
    this primitive charges a documented nominal ``cost_per_lookup`` (default 1) so
    such mechanisms pay an honest, non-zero price for their dominant work.
    """
    return n * cost_per_lookup


@dataclass(frozen=True)
class FlopBreakdown:
    """Matmul FLOPs split into forward and backward passes.

    ``forward`` alone is the inference cost; ``total`` is one training step
    (forward + backward). Instances add and scale so a model can compose its
    layers and the harness can multiply by batch size.
    """

    forward: int = 0
    backward: int = 0

    @property
    def total(self) -> int:
        """Training-step FLOPs: forward + backward."""
        return self.forward + self.backward

    def __add__(self, other: "FlopBreakdown") -> "FlopBreakdown":
        return FlopBreakdown(self.forward + other.forward, self.backward + other.backward)

    def scale(self, factor: int) -> "FlopBreakdown":
        """Replicate this cost ``factor`` times (e.g. across a batch)."""
        return FlopBreakdown(self.forward * factor, self.backward * factor)

    @classmethod
    def from_forward(cls, forward: int) -> "FlopBreakdown":
        """Build a breakdown from forward FLOPs, charging the standard backward."""
        return cls(forward=forward, backward=BACKWARD_MULTIPLIER * forward)
