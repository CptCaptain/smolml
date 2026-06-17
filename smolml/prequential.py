"""Prequential (online) evaluation — predict each byte *before* it is revealed.

The protocol (ADR 0004): the model predicts the next byte, pays −log2 p(true)
bits, then *may* adapt on the revealed byte. Cumulative bits / bytes = bpb. Every
FLOP is counted — prediction (decode) and any adaptation — so a model that learns
at test time pays for it in the same budget.

**Leakage is structural, not checked at runtime.** Each iteration calls
``predict_logits`` (which sees only already-``observe``-d bytes) and scores the
true byte *before* the next ``observe`` reveals it. The model never receives byte
``t`` while predicting byte ``t``. ``test_prequential.py`` proves it: perturbing
the stream at/after position ``t`` leaves the prediction at ``t`` bit-identical.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from smolml.flops import FlopBreakdown
from smolml.models.registry import LanguageModel

_LN2 = math.log(2.0)


def score_bits(logits: torch.Tensor, byte: int) -> float:
    """Bits to encode ``byte`` under ``logits``: −log2 softmax(logits)[byte]."""
    log_probs = F.log_softmax(logits, dim=-1)
    return float(-log_probs[byte].item() / _LN2)


@dataclass
class PrequentialResult:
    """Outcome of a prequential pass over one eval stream."""

    total_bits: float
    n_bytes: int
    decode_flops: FlopBreakdown
    adapt_flops: FlopBreakdown
    # (bytes_seen, cumulative_eval_flops, cumulative_bpb) trajectory checkpoints.
    checkpoints: list[tuple[int, int, float]] = field(default_factory=list)
    # Per-position predicted logits, only when collect_logits=True (tests).
    predicted_logits: list[torch.Tensor] | None = None

    @property
    def bpb(self) -> float:
        return self.total_bits / self.n_bytes

    @property
    def eval_flops(self) -> int:
        """Total inference + adaptation FLOPs spent over the stream."""
        return (self.decode_flops + self.adapt_flops).total


def prequential_bpb(
    model: LanguageModel,
    stream: np.ndarray,
    *,
    device: torch.device,
    adapt_interval: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    checkpoint_interval: int = 0,
    collect_logits: bool = False,
) -> PrequentialResult:
    """Score ``stream`` prequentially; accumulate decode (and adapt) FLOPs.

    ``adapt_interval`` > 0 calls ``model.adapt`` every that-many bytes (0 = frozen).
    ``checkpoint_interval`` > 0 records a (bytes, eval_flops, bpb) trajectory point
    every that-many bytes (the final point is always recorded). The returned FLOPs
    are exactly what the model reported computing — the budget honestly includes
    inference and any test-time adaptation.
    """
    if len(stream) < 1:
        raise ValueError("eval stream must be non-empty")
    model.eval()
    bytes_ = [int(b) for b in stream]
    n = len(bytes_)

    state = model.init_prequential_state()
    total_bits = 0.0
    decode = FlopBreakdown()
    adapt = FlopBreakdown()
    checkpoints: list[tuple[int, int, float]] = []
    logits_log: list[torch.Tensor] = []

    def record_checkpoint(bytes_seen: int) -> None:
        checkpoints.append((bytes_seen, (decode + adapt).total, total_bits / bytes_seen))

    # Byte 0 is predicted from the prior (no observations yet).
    logits = model.predict_logits(state)
    if collect_logits:
        logits_log.append(logits)
    total_bits += score_bits(logits, bytes_[0])

    for pos in range(n - 1):
        # Reveal byte `pos` (already scored), fold it into the state.
        state, step_decode = model.observe(state, bytes_[pos], pos)
        decode += step_decode
        if adapt_interval and (pos + 1) % adapt_interval == 0:
            state, step_adapt = model.adapt(state, optimizer, grad_clip=grad_clip)
            adapt += step_adapt
        # Predict byte `pos+1` BEFORE it is revealed (next iteration observes it).
        logits = model.predict_logits(state)
        if collect_logits:
            logits_log.append(logits)
        total_bits += score_bits(logits, bytes_[pos + 1])

        bytes_seen = pos + 2
        if checkpoint_interval and bytes_seen % checkpoint_interval == 0:
            record_checkpoint(bytes_seen)

    if not checkpoints or checkpoints[-1][0] != n:
        record_checkpoint(n)

    return PrequentialResult(
        total_bits=total_bits,
        n_bytes=n,
        decode_flops=decode,
        adapt_flops=adapt,
        checkpoints=checkpoints,
        predicted_logits=logits_log if collect_logits else None,
    )
