"""Regression tests for the properties that protect the metric.

These guard the *comparison*, not just arithmetic: no eval/train leakage, runs are
bit-reproducible on CPU for a fixed seed, the model's self-reported FLOPs match an
independent measurement, and the budget is an honest ceiling.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from smolml.data import load_sample
from smolml.data.corpus import ByteCorpus, get_batch
from smolml.flops import causal_attention_flops
from smolml.models import Transformer, TransformerConfig, build_model
from smolml.train import TrainConfig, train_run

TINY = {"d_model": 16, "n_layers": 2, "n_heads": 2, "max_seq_len": 32}
BATCH, SEQ, EVAL = 4, 16, 16


def _cfg(budget: float, run_name: str) -> TrainConfig:
    return TrainConfig(
        model="transformer",
        model_config=TINY,
        flop_budget=budget,
        batch_size=BATCH,
        seq_len=SEQ,
        eval_seq_len=EVAL,
        eval_interval=2,
        eval_batches=2,
        seed=0,
        device="cpu",
        run_name=run_name,
    )


def _step_flops() -> int:
    return build_model("transformer", TINY).flops(SEQ).total * BATCH


def _step_records(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "step":
            out.append(obj)
    return out


# (a) No eval-stream leakage into the prior corpus.
def test_split_is_disjoint_and_train_batches_never_read_val():
    # Train region is all 0x00, val region all 0xFF: any 0xFF in a *train* batch
    # would prove a window reached into the held-out tail.
    data = np.concatenate([np.zeros(900, np.uint8), np.full(100, 255, np.uint8)])
    train, val = ByteCorpus(data).split(0.1)
    assert (len(train), len(val)) == (900, 100)
    assert np.unique(train).tolist() == [0]
    assert np.unique(val).tolist() == [255]
    gen = torch.Generator().manual_seed(0)
    for _ in range(300):
        x, y = get_batch(train, BATCH, SEQ, torch.device("cpu"), gen)
        assert bool((x != 255).all()) and bool((y != 255).all())


# (b) Determinism: two seed=0 CPU runs are bit-identical (sans wallclock).
def test_two_seeded_cpu_runs_are_identical(tmp_path):
    sf = _step_flops()
    sa = train_run(load_sample(), _cfg(sf * 6, "a"), runs_dir=tmp_path)
    sb = train_run(load_sample(), _cfg(sf * 6, "b"), runs_dir=tmp_path)
    steps_a = _step_records(tmp_path / "a.jsonl")
    steps_b = _step_records(tmp_path / "b.jsonl")
    assert len(steps_a) == len(steps_b) >= 1
    for ra, rb in zip(steps_a, steps_b, strict=True):
        assert ra["step"] == rb["step"]
        assert ra["cumulative_flops"] == rb["cumulative_flops"]
        assert ra["train_loss"] == rb["train_loss"]
        assert ra["val_bpb"] == rb["val_bpb"]
    assert sa.total_flops == sb.total_flops
    assert sa.final_val_bpb == sb.final_val_bpb


# (c) The model's declared forward matmul FLOPs match an independent measurement.
def test_declared_forward_matmul_matches_flopcountermode():
    cfg = TransformerConfig(d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=8)
    model = Transformer(cfg)
    model.eval()
    t = 8
    x = torch.randint(0, 256, (1, t))  # batch 1 -> compare per-sequence FLOPs
    with FlopCounterMode(display=False) as fc:
        model(x)
    counts = fc.get_flop_counts()["Global"]
    measured_mm = sum(
        v for k, v in counts.items() if str(k) in ("aten.mm", "aten.addmm", "aten.bmm")
    )
    # Our dense (projection + head) forward = total forward minus the attention
    # activation term. SDPA is fused (FlopCounterMode reports it separately / as 0
    # on CPU) and torch counts the full T^2 square rather than the causal half, so
    # we assert the dense-matmul subtotal only — see docs/harness.md.
    attn = cfg.n_layers * causal_attention_flops(t, cfg.d_model)
    declared_dense = model.flops(t).forward - attn
    assert measured_mm == declared_dense


# (d) The budget is a ceiling, including non-multiple and impossible budgets.
def test_non_multiple_budget_never_exceeded(tmp_path):
    sf = _step_flops()
    summary = train_run(load_sample(), _cfg(sf * 3.5, "frac"), runs_dir=tmp_path)
    assert summary.steps == 3  # only 3 whole steps fit under 3.5x
    assert summary.total_flops == 3 * sf
    assert summary.total_flops <= sf * 3.5


def test_zero_and_negative_budget_raise(tmp_path):
    with pytest.raises(ValueError):
        train_run(load_sample(), _cfg(0.0, "zero"), runs_dir=tmp_path)
    with pytest.raises(ValueError):
        train_run(load_sample(), _cfg(-5.0, "neg"), runs_dir=tmp_path)


def test_impossible_budget_logs_step_zero(tmp_path):
    sf = _step_flops()
    summary = train_run(load_sample(), _cfg(sf * 0.5, "imp"), runs_dir=tmp_path)
    assert summary.steps == 0
    assert summary.total_flops == 0
    steps = _step_records(tmp_path / "imp.jsonl")
    assert len(steps) == 1
    assert steps[0]["step"] == 0
    assert steps[0]["cumulative_flops"] == 0
    assert 0.0 < steps[0]["val_bpb"] <= 8.5
