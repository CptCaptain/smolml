"""Offline end-to-end smoke run: train -> JSONL log -> leaderboard row + PNG.

Tiny by design: a few steps of the transformer baseline under a tiny FLOP budget
on the bundled sample, fully offline, finishing in seconds on CPU. This is the
proof the whole harness wires together.
"""

import json

from smolml.data import load_sample
from smolml.leaderboard import collect_runs, regenerate
from smolml.models import list_models
from smolml.train import TrainConfig, train_run

TINY_MODEL = {"d_model": 32, "n_layers": 2, "n_heads": 4, "max_seq_len": 32}


def _tiny_config(budget: float, run_name: str) -> TrainConfig:
    return TrainConfig(
        model="transformer",
        model_config=TINY_MODEL,
        flop_budget=budget,
        batch_size=8,
        seq_len=32,
        lr=3e-3,
        seed=0,
        eval_interval=4,
        eval_batches=4,
        val_fraction=0.1,
        device="cpu",
        run_name=run_name,
    )


def test_transformer_is_registered():
    assert "transformer" in list_models()


def test_end_to_end_smoke(tmp_path):
    runs_dir = tmp_path / "runs"
    from smolml.models import build_model

    step_flops = build_model("transformer", TINY_MODEL).flops(32).total * 8
    cfg = _tiny_config(budget=step_flops * 12, run_name="smoke")

    summary = train_run(load_sample(), cfg, runs_dir=runs_dir)

    # Budget honored: stopped at/after budget, took a small number of steps.
    assert summary.steps >= 1
    assert summary.total_flops >= cfg.flop_budget
    assert summary.total_flops == step_flops * summary.steps
    assert summary.device == "cpu"

    # JSONL log exists, with a meta line and well-formed step lines.
    log_path = runs_dir / "smoke.jsonl"
    assert log_path.exists()
    lines = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    meta = lines[0]
    steps = [ln for ln in lines[1:] if ln["type"] == "step"]
    assert meta["type"] == "meta"
    assert meta["model"] == "transformer"
    assert meta["params"] == summary.params > 0
    assert len(steps) >= 1
    for rec in steps:
        assert set(rec) == {
            "type",
            "wallclock",
            "step",
            "cumulative_flops",
            "train_loss",
            "val_bpb",
        }
        assert rec["cumulative_flops"] > 0
        assert rec["wallclock"] >= 0.0
        assert 0.0 < rec["val_bpb"] <= 8.5  # bounded; uniform baseline is 8.0

    # cumulative_flops is monotonic non-decreasing across logged steps.
    flops = [rec["cumulative_flops"] for rec in steps]
    assert flops == sorted(flops)

    # The model actually learned: final val bpb beat the uninformed 8.0 bpb.
    assert summary.final_val_bpb < 8.0

    # Leaderboard: a table row + a PNG file are produced.
    table, png = regenerate(
        runs_dir, table_path=tmp_path / "leaderboard.md", plot_path=tmp_path / "leaderboard.png"
    )
    assert png.exists()
    assert png.stat().st_size > 0
    assert "transformer" in table
    assert "final val bpb" in table

    records = collect_runs(runs_dir)
    assert len(records) == 1
    rec = records[0]
    assert rec.run == "smoke"
    assert rec.model == "transformer"
    assert rec.params == summary.params
    assert rec.final_val_bpb == summary.final_val_bpb
    assert rec.steps == len(steps)
