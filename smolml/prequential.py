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

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from smolml.data.corpus import VOCAB_SIZE, get_batch
from smolml.device import get_device
from smolml.flops import FlopBreakdown
from smolml.models.registry import LanguageModel, build_model

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
    flops: FlopBreakdown  # ALL per-byte compute (prediction + any adaptation), one channel
    # (bytes_seen, cumulative_eval_flops, cumulative_bpb) trajectory checkpoints.
    checkpoints: list[tuple[int, int, float]] = field(default_factory=list)
    # Per-position predicted logits, only when collect_logits=True (tests).
    predicted_logits: list[torch.Tensor] | None = None

    @property
    def bpb(self) -> float:
        return self.total_bits / self.n_bytes

    @property
    def eval_flops(self) -> int:
        """Total per-byte FLOPs (inference + any test-time adaptation)."""
        return self.flops.total


def prequential_bpb(
    model: LanguageModel,
    stream: np.ndarray,
    *,
    device: torch.device,
    checkpoint_interval: int = 0,
    collect_logits: bool = False,
) -> PrequentialResult:
    """Score ``stream`` prequentially through the single ``step`` channel.

    Per byte: ``model.step(state, revealed_byte, pos)`` folds the byte, runs any
    online adaptation, and returns the next-byte distribution plus **all** FLOPs
    it spent — accumulated here. Byte 0 has no context, so it is scored against a
    uniform prior (8 bits, no model compute); every model-computed prediction
    flows through ``step``, so compute cannot hide at eval time. Whether the model
    adapts is the model's business (the loop just measures).

    ``checkpoint_interval`` > 0 records a (bytes, eval_flops, bpb) trajectory point
    every that-many bytes (the final point is always recorded).
    """
    if len(stream) < 1:
        raise ValueError("eval stream must be non-empty")
    model.eval()
    bytes_ = [int(b) for b in stream]
    n = len(bytes_)

    state = model.init_prequential_state()
    total_bits = 0.0
    flops = FlopBreakdown()
    checkpoints: list[tuple[int, int, float]] = []
    logits_log: list[torch.Tensor] = []
    uniform = torch.zeros(VOCAB_SIZE)

    def record_checkpoint(bytes_seen: int) -> None:
        checkpoints.append((bytes_seen, flops.total, total_bits / bytes_seen))

    # Byte 0: no context -> uniform prior (8 bits), zero model FLOPs.
    if collect_logits:
        logits_log.append(uniform)
    total_bits += score_bits(uniform, bytes_[0])

    for pos in range(n - 1):
        state, next_logits, step_flops = model.step(state, bytes_[pos], pos)
        flops += step_flops
        if collect_logits:
            logits_log.append(next_logits)
        total_bits += score_bits(next_logits, bytes_[pos + 1])
        bytes_seen = pos + 2
        if checkpoint_interval and bytes_seen % checkpoint_interval == 0:
            record_checkpoint(bytes_seen)

    if not checkpoints or checkpoints[-1][0] != n:
        record_checkpoint(n)

    return PrequentialResult(
        total_bits=total_bits,
        n_bytes=n,
        flops=flops,
        checkpoints=checkpoints,
        predicted_logits=logits_log if collect_logits else None,
    )


def pretrain(
    model: LanguageModel,
    train_data: np.ndarray,
    *,
    flop_budget: float,
    batch_size: int,
    seq_len: int,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    grad_clip: float,
    seed: int,
    device: torch.device,
) -> int:
    """Amortized pretraining on the prior corpus to a FLOP ceiling.

    Mirrors the amortized training loop (budget is a ceiling) but writes no log
    and returns the **training** FLOPs spent — the pretraining share of the total
    budget. ``flop_budget`` below one step trains nothing (a valid zero-pretrain,
    fully-transductive point on the curve).
    """
    if flop_budget < 0:
        raise ValueError(f"flop_budget must be non-negative, got {flop_budget}")
    model.train()
    gen = torch.Generator().manual_seed(seed)
    optimizer = model.configure_optimizer(lr=lr, weight_decay=weight_decay, betas=betas)
    step_flops = model.flops(seq_len).scale(batch_size).total
    spent = 0
    while spent + step_flops <= flop_budget:
        x, y = get_batch(train_data, batch_size, seq_len, device, gen)
        _, step = model.train_step((x, y), optimizer, grad_clip=grad_clip)
        spent += step.total
    return spent


@dataclass
class PrequentialConfig:
    """Configuration for a prequential run (pretrain budget + eval protocol)."""

    model: str = "transformer"
    model_config: dict[str, object] = field(default_factory=dict)
    pretrain_flop_budget: float = 1e10
    batch_size: int = 16
    seq_len: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 0
    checkpoint_interval: int = 0
    device: str | None = None
    run_name: str | None = None


@dataclass
class PrequentialSummary:
    """Result of a prequential run (the leaderboard re-derives this from the log)."""

    run: str
    model: str
    params: int
    device: str
    seed: int
    pretrain_flops: int
    eval_flops: int
    total_flops: int
    eval_bytes: int
    bpb: float
    log_path: str


def prequential_run(
    prior_corpus: np.ndarray,
    eval_stream: np.ndarray,
    cfg: PrequentialConfig,
    runs_dir: str | Path = "runs",
) -> PrequentialSummary:
    """Pretrain on the prior corpus, then score the eval stream prequentially.

    The pretrain FLOP budget is an enforced ceiling; total FLOPs = pretrain +
    Σ per-byte step FLOPs (inference + any adaptation), which is *reported*, not
    capped. Streams longer than the model context are handled by the bounded
    sliding-window decode. Writes a ``protocol="prequential"`` JSONL and returns a
    :class:`PrequentialSummary`.
    """
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    model = build_model(cfg.model, cfg.model_config).to(device)
    start_perf = time.perf_counter()
    pretrain_flops = pretrain(
        model,
        prior_corpus,
        flop_budget=cfg.pretrain_flop_budget,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        grad_clip=cfg.grad_clip,
        seed=cfg.seed,
        device=device,
    )
    result = prequential_bpb(
        model,
        eval_stream,
        device=device,
        checkpoint_interval=cfg.checkpoint_interval,
    )
    total_flops = pretrain_flops + result.eval_flops

    run_name = cfg.run_name or f"{cfg.model}-preq-{int(time.time())}"
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    log_path = runs_path / f"{run_name}.jsonl"
    resolved_config = (
        model.config.__dict__ if hasattr(model.config, "__dict__") else dict(cfg.model_config)
    )
    with log_path.open("w") as log:
        meta = {
            "type": "meta",
            "protocol": "prequential",
            "run": run_name,
            "model": cfg.model,
            "config": resolved_config,
            "params": model.num_params(),
            "device": device.type,
            "seed": cfg.seed,
            "pretrain_flop_budget": cfg.pretrain_flop_budget,
            "pretrain_flops": pretrain_flops,
            "eval_bytes": result.n_bytes,
            "started_at": time.time(),
        }
        log.write(json.dumps(meta) + "\n")
        # Trajectory: cumulative bpb vs cumulative TOTAL FLOPs (pretrain + eval-so-far).
        for bytes_seen, eval_flops_so_far, cum_bpb in result.checkpoints:
            log.write(
                json.dumps(
                    {
                        "type": "step",
                        "wallclock": time.perf_counter() - start_perf,
                        "step": bytes_seen,
                        "cumulative_flops": pretrain_flops + eval_flops_so_far,
                        "train_loss": None,
                        "val_bpb": cum_bpb,
                    }
                )
                + "\n"
            )

    return PrequentialSummary(
        run=run_name,
        model=cfg.model,
        params=model.num_params(),
        device=device.type,
        seed=cfg.seed,
        pretrain_flops=pretrain_flops,
        eval_flops=result.eval_flops,
        total_flops=total_flops,
        eval_bytes=result.n_bytes,
        bpb=result.bpb,
        log_path=str(log_path),
    )
