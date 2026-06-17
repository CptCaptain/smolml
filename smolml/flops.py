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
- **What we count:** the matmuls that dominate compute — linear/projection layers
  (``O(tokens * d^2)``) and the attention score/value matmuls (``O(tokens^2 * d)``).
- **What we ignore, and why:** elementwise ops (activations, RMSNorm, residual
  adds, softmax normalization, RoPE rotations, dropout). These are ``O(tokens*d)``
  vs. ``O(tokens*d^2)`` for the matmuls — asymptotically negligible, and counting
  them precisely is framework-dependent without moving the metric. Embedding
  lookups are gathers (no multiply-add) and cost 0.
- **Backward = 2x forward (matmul FLOPs).** For ``Y = A·B`` where both operands
  feed gradients, backprop computes ``dA = dY·Bᵀ`` and ``dB = Aᵀ·dY`` — two
  matmuls of the same FLOP magnitude as the forward one. So a training step costs
  ``forward + 2*forward = 3*forward``. This is exactly the textbook ``C ≈ 6*N*D``
  rule (2 FLOPs/param/token forward, 4 backward) generalized to also charge the
  attention activation matmuls.

Extensibility (Task 0.2 hook — do not build 0.2 here)
-----------------------------------------------------
``FlopBreakdown`` keeps ``forward`` and ``backward`` separate on purpose:

- **training** step  -> ``forward + backward`` (``.total``),
- **inference** step -> ``forward`` only (no ``.backward``),
- **online adaptation** step -> ``forward + backward`` of the adapted submodule.

A future prequential/total-FLOP mode reuses these primitives and the same
``FlopBreakdown`` to add inference and adaptation paths with no redesign.
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
