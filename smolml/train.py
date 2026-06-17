"""Training loop to a fixed FLOP budget, with per-run JSONL logging.

The budget is a **training-FLOP** allowance (forward + backward of the training
steps), counted by the shared :mod:`smolml.flops` accounting — never wall-clock.
Training stops as soon as cumulative training FLOPs reach the budget, so two
models are compared at equal compute regardless of how fast they run.

Run log (one JSON object per line, ``runs/<run>.jsonl``)
-------------------------------------------------------
- line 1 — ``{"type": "meta", ...}``: run identity and hyperparameters
  (run, model, config, params, device, seed, flop_budget, batch_size, seq_len,
  lr, started_at).
- each later line — ``{"type": "step", "wallclock", "step", "cumulative_flops",
  "train_loss", "val_bpb"}``:
    - ``wallclock`` — seconds elapsed since training started,
    - ``step`` — optimizer steps taken,
    - ``cumulative_flops`` — training FLOPs spent so far,
    - ``train_loss`` — mini-batch training loss in **bits/byte** (same unit as
      ``val_bpb``, so the two curves are directly comparable),
    - ``val_bpb`` — validation bits-per-byte at this step.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from smolml.data.corpus import ByteCorpus, get_batch
from smolml.device import get_device
from smolml.eval import evaluate_bpb
from smolml.models.registry import build_model

_LN2 = math.log(2.0)


@dataclass
class TrainConfig:
    """Everything needed to reproduce a run (fixed seed makes it deterministic)."""

    model: str = "transformer"
    model_config: dict = field(default_factory=dict)
    flop_budget: float = 1e12
    batch_size: int = 16
    seq_len: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 0
    eval_interval: int = 50
    eval_batches: int = 8
    val_fraction: float = 0.1
    device: str | None = None
    run_name: str | None = None


@dataclass
class RunSummary:
    """Result of a finished run (the leaderboard re-derives this from the log)."""

    run: str
    model: str
    params: int
    seed: int
    device: str
    flop_budget: float
    total_flops: int
    steps: int
    final_val_bpb: float
    elapsed_sec: float
    log_path: str


def train_run(corpus: ByteCorpus, cfg: TrainConfig, runs_dir: str | Path = "runs") -> RunSummary:
    """Train ``cfg.model`` on ``corpus`` until the FLOP budget is spent.

    Writes a JSONL log under ``runs_dir`` and returns a :class:`RunSummary`.
    Deterministic given ``cfg.seed`` and the corpus.
    """
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    batch_gen = torch.Generator().manual_seed(cfg.seed)

    train_data, val_data = corpus.split(cfg.val_fraction)
    model = build_model(cfg.model, cfg.model_config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )

    # Training FLOPs per optimizer step: one (forward + backward) per sequence,
    # times the batch size. Constant across steps for a fixed shape.
    step_flops = model.flops(cfg.seq_len).total * cfg.batch_size

    run_name = cfg.run_name or f"{cfg.model}-{int(time.time())}"
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    log_path = runs_path / f"{run_name}.jsonl"

    started_at = time.time()
    start_perf = time.perf_counter()

    def evaluate() -> float:
        return evaluate_bpb(
            model,
            val_data,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=device,
            n_batches=cfg.eval_batches,
        )

    with log_path.open("w") as log:
        meta = {
            "type": "meta",
            "run": run_name,
            "model": cfg.model,
            "config": cfg.model_config,
            "params": model.num_params(),
            "device": device.type,
            "seed": cfg.seed,
            "flop_budget": cfg.flop_budget,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "lr": cfg.lr,
            "started_at": started_at,
        }
        log.write(json.dumps(meta) + "\n")

        step = 0
        cumulative_flops = 0
        last_logged_step = -1
        final_bpb = math.nan

        def log_step(train_bpb: float, val_bpb: float) -> None:
            nonlocal last_logged_step
            record = {
                "type": "step",
                "wallclock": time.perf_counter() - start_perf,
                "step": step,
                "cumulative_flops": cumulative_flops,
                "train_loss": train_bpb,
                "val_bpb": val_bpb,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            last_logged_step = step

        while step == 0 or cumulative_flops < cfg.flop_budget:
            x, y = get_batch(train_data, cfg.batch_size, cfg.seq_len, device, batch_gen)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            cumulative_flops += step_flops
            step += 1
            if step % cfg.eval_interval == 0:
                final_bpb = evaluate()
                log_step(loss.item() / _LN2, final_bpb)

        if last_logged_step != step:
            final_bpb = evaluate()
            log_step(loss.item() / _LN2, final_bpb)

    return RunSummary(
        run=run_name,
        model=cfg.model,
        params=model.num_params(),
        seed=cfg.seed,
        device=device.type,
        flop_budget=cfg.flop_budget,
        total_flops=cumulative_flops,
        steps=step,
        final_val_bpb=final_bpb,
        elapsed_sec=time.perf_counter() - start_perf,
        log_path=str(log_path),
    )


def train_config_to_dict(cfg: TrainConfig) -> dict:
    """Serialize a TrainConfig (handy for logging/CLI)."""
    return asdict(cfg)
