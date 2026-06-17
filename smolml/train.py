"""Training loop to a fixed FLOP budget, with per-run JSONL logging.

The budget is a **training-FLOP** allowance (forward + the model's own update
per step), counted by the shared :mod:`smolml.flops` accounting — never
wall-clock. The budget is a **ceiling**: a step runs only if it still fits, so a
run never overspends and every run's endpoint sits at ``<= budget`` FLOPs — equal
compute, comparable endpoints, regardless of per-step cost.

Run log (one JSON object per line, ``runs/<run>.jsonl``)
-------------------------------------------------------
- line 1 — ``{"type": "meta", ...}``: run identity, the **resolved** model config
  (defaults filled in), and every training hyperparameter (run, model, config,
  params, device, seed, flop_budget, batch_size, seq_len, eval_seq_len,
  eval_batches, eval_interval, val_fraction, lr, weight_decay, betas, grad_clip,
  started_at) — so a run is reproducible from its log alone.
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
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import torch

from smolml.data.corpus import ByteCorpus, get_batch
from smolml.device import get_device
from smolml.eval import evaluate_bpb
from smolml.models.registry import build_model

_LN2 = math.log(2.0)


@dataclass
class TrainConfig:
    """Everything needed to reproduce a run (fixed seed makes it deterministic)."""

    model: str = "transformer"
    model_config: dict[str, object] = field(default_factory=dict)
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
    # Validation uses a FIXED context length and window count for EVERY run,
    # independent of training seq_len, so bpb (which depends on conditioning
    # length) is comparable across runs. Must be <= the model's max context.
    eval_seq_len: int = 128
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
    """Train ``cfg.model`` on ``corpus`` until the FLOP budget would be exceeded.

    The budget is a ceiling: the final ``total_flops`` is ``<= cfg.flop_budget``.
    If the budget is too small for even one step, the run still logs a step-0
    point (0 FLOPs, init losses). Writes a JSONL log under ``runs_dir`` and
    returns a :class:`RunSummary`. Deterministic given ``cfg.seed`` and the
    corpus. Raises ``ValueError`` if ``flop_budget <= 0``.
    """
    if cfg.flop_budget <= 0:
        raise ValueError(f"flop_budget must be positive, got {cfg.flop_budget}")
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    batch_gen = torch.Generator().manual_seed(cfg.seed)

    train_data, val_data = corpus.split(cfg.val_fraction)
    model = build_model(cfg.model, cfg.model_config).to(device)
    model.train()
    max_ctx = getattr(model.config, "max_seq_len", None)
    if max_ctx is not None and cfg.eval_seq_len > max_ctx:
        raise ValueError(f"eval_seq_len {cfg.eval_seq_len} exceeds model max context {max_ctx}")
    optimizer = model.configure_optimizer(lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)

    # Look-ahead estimate of one step's cost, used as the budget *ceiling* gate.
    # (Accumulation below uses the FLOPs train_step actually reports; for the
    # constant-cost transformer the two coincide exactly.)
    step_flops = model.flops(cfg.seq_len).scale(cfg.batch_size).total
    if step_flops <= 0:
        raise ValueError(f"model reports non-positive step cost: {step_flops}")
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
            seq_len=cfg.eval_seq_len,
            device=device,
            n_batches=cfg.eval_batches,
        )

    def measure_train_bpb() -> float:
        # Used only for the step-0 log line (no optimizer step has run yet);
        # a no-grad forward over train batches, not charged to the budget.
        return evaluate_bpb(
            model,
            train_data,
            batch_size=cfg.batch_size,
            seq_len=cfg.eval_seq_len,
            device=device,
            n_batches=cfg.eval_batches,
        )

    with log_path.open("w") as log:
        # Log the RESOLVED model config (defaults filled in, e.g. d_ff), not the
        # partial request dict, plus every TrainConfig hyperparameter — so a run
        # is reproducible from its log alone.
        resolved_config = (
            asdict(model.config) if is_dataclass(model.config) else dict(cfg.model_config)
        )
        meta = {
            "type": "meta",
            "protocol": "amortized",
            "run": run_name,
            "model": cfg.model,
            "config": resolved_config,
            "params": model.num_params(),
            "device": device.type,
            "seed": cfg.seed,
            "flop_budget": cfg.flop_budget,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "eval_seq_len": cfg.eval_seq_len,
            "eval_batches": cfg.eval_batches,
            "eval_interval": cfg.eval_interval,
            "val_fraction": cfg.val_fraction,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "betas": list(cfg.betas),
            "grad_clip": cfg.grad_clip,
            "started_at": started_at,
        }
        log.write(json.dumps(meta) + "\n")

        step = 0
        cumulative_flops = 0
        last_logged_step = -1
        final_bpb = math.nan
        last_train_bpb = math.nan

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

        # Budget is a CEILING: take a step only if it still fits, so a run never
        # overspends and every run's endpoint sits at <= budget FLOPs (endpoints
        # are comparable across models regardless of per-step cost).
        while cumulative_flops + step_flops <= cfg.flop_budget:
            x, y = get_batch(train_data, cfg.batch_size, cfg.seq_len, device, batch_gen)
            loss, spent = model.train_step((x, y), optimizer, grad_clip=cfg.grad_clip)
            cumulative_flops += spent.total
            step += 1
            last_train_bpb = loss.item() / _LN2
            if step % cfg.eval_interval == 0:
                final_bpb = evaluate()
                log_step(last_train_bpb, final_bpb)

        # Always log a final point — including the degenerate case where the
        # budget was too small for even one step (step 0, 0 FLOPs).
        if last_logged_step != step:
            if step == 0:
                last_train_bpb = measure_train_bpb()
            final_bpb = evaluate()
            log_step(last_train_bpb, final_bpb)

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
