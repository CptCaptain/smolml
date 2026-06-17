"""Bits-per-byte evaluation.

bpb is cross-entropy measured in **bits** and normalized **per byte**:

    bpb = (1 / n_bytes) * Σ −log₂ p(byte_i | context_i)

It is the comparison currency (tokenizer-independent). The conversions are
exact: ``F.cross_entropy`` returns mean negative log-likelihood in *nats*
(natural log); dividing by ``ln 2`` converts nats to bits. A model that knows
nothing (uniform over 256 bytes) scores exactly 8.0 bpb.
"""

import math

import torch
import torch.nn.functional as F

from smolml.data.corpus import get_batch
from smolml.models.registry import LanguageModel

_LN2 = math.log(2.0)


def cross_entropy_bits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, int]:
    """Total cross-entropy in **bits** and the number of target bytes.

    ``logits`` is ``(..., vocab)``; ``targets`` is the matching ``(...)`` int64.
    Returns ``(total_bits, n_bytes)`` so callers can aggregate across batches
    before dividing.
    """
    vocab = logits.size(-1)
    loss_nats = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="sum")
    return loss_nats.item() / _LN2, int(targets.numel())


def bits_per_byte(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean bits-per-byte for a single batch of logits/targets."""
    total_bits, n_bytes = cross_entropy_bits(logits, targets)
    return total_bits / n_bytes


@torch.no_grad()
def evaluate_bpb(
    model: LanguageModel,
    val_data,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    n_batches: int,
    seed: int = 0,
) -> float:
    """Average validation bpb over ``n_batches`` seeded batches.

    Uses a fixed ``seed`` so the validation set sampled is identical across runs,
    making bpb comparable. Restores the model's train/eval mode afterward. This
    measurement (a forward pass) is **not** charged to the training-FLOP budget
    under the amortized protocol; see ``docs/harness.md``.
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_bits = 0.0
    total_bytes = 0
    for _ in range(n_batches):
        x, y = get_batch(val_data, batch_size, seq_len, device, generator)
        bits, n = cross_entropy_bits(model(x), y)
        total_bits += bits
        total_bytes += n
    if was_training:
        model.train()
    return total_bits / total_bytes
