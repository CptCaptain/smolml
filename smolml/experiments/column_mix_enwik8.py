"""column_mix kill-test on REAL enwik8 — Task B.5 (the gate before any full-corpus claim).

``column_mix`` replaces ``delta_mix``'s single delta stream with ``C`` routed columns + a per-arm
bandit gate. The bet (see ``docs/tasks/B.5-column-mix.md``): route-conditional SELECTION — a
context×feature interaction one linear-in-φ ``W`` cannot represent — buys more loss-reduction/FLOP
than the bar — NOT capacity (a single ``W`` at ``delta_dim=C·d`` matches capacity for free). A.1 /
B.1 / gated_mix all show "refine an already-good cheap learner" usually FAILS, so this matched-FLOP
run decides it BEFORE the multi-hour full run, with a control denying the capacity confound:

- (a) ``delta``          — ``delta_mix`` (the bar, ``C = 1``) warmed at budget ``P``;
- (b) ``column_learned`` — ``column_mix`` (``C > 1``, learned gate) at the SAME budget;
- (c) ``column_gateoff`` — ``column_mix`` (``C > 1``, frozen fixed-hash route) at the SAME budget;
- (d) ``delta_bigdim``   — ``delta_mix`` with ``delta_dim = C·d`` (matched CAPACITY, no router).

The router is ``O(C)`` (≪ 0.2% of the bar's per-byte cost), so the four land at matched total FLOPs.
**column_mix PASSES iff max((b), (c)) beats BOTH (a) and (d)** — routing must earn its keep as
*selection*, not table size. (c) vs (a)/(d) is the selection verdict; (b) vs (c) the learned-gate
verdict. Diagnostics make it mechanistic: per-column load (collapse → ≡ bar, an accepted outcome),
per-column conditional bpb, the delta-row mixer weight, and gate drift off the hash prior.

Bounded-but-real scale (pure-Python mixing): a real-enwik8 slice, the warm budget caps folded bytes
to a few-hundred-k, eval a few-ten-k. NOT the full 5 MB ADR endpoint (that is ``full_corpus.py``,
gated on this PASS). Run it (CPU; downloads enwik8 once to ``data/cache/``, then minutes)::

    uv run python -m smolml.experiments.column_mix_enwik8
"""

import numpy as np
import torch

from smolml.data.corpus import prepare_enwik8
from smolml.models import build_model
from smolml.prequential import (
    PrequentialConfig,
    PrequentialSummary,
    prequential_run,
    pretrain,
)

CPU = torch.device("cpu")
RUNS_DIR = "runs/b5"
N_BYTES = 4_000_000  # first 4 MB of real enwik8 (prior ~= N_BYTES - EVAL_BYTES)
EVAL_BYTES = 32_768  # real-enwik8 eval window
SEED = 0
SEQ_LEN = 128
PRETRAIN_BUDGET = 1.0e10  # P: the shared pretrain budget for every config
MAX_ORDER = 6
N_COLUMNS = 4
DELTA_DIM = 1 << 16

HASHED_CFG: dict[str, object] = {"max_order": MAX_ORDER, "hash_min_order": 4, "table_bits": 18}
DELTA_CFG: dict[str, object] = {
    **HASHED_CFG,
    "delta_dim": DELTA_DIM,
    "delta_orders": (3, 4, 5, 6, 7, 8),
    "delta_eta": 0.2,
}
ROUTE_CFG: dict[str, object] = {
    **DELTA_CFG,
    "n_columns": N_COLUMNS,
    "route_buckets": 1 << 12,
    "route_order": 4,
    "delta_eta": 0.2,
}


def _run(model: str, cfg_extra: dict[str, object], run_name: str) -> PrequentialSummary:
    corpus = prepare_enwik8(n_bytes=N_BYTES)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=EVAL_BYTES)
    cfg = PrequentialConfig(
        model=model,
        model_config=cfg_extra,
        pretrain_flop_budget=PRETRAIN_BUDGET,
        seq_len=SEQ_LEN,
        seed=SEED,
        run_name=run_name,
    )
    return prequential_run(prior, eval_stream, cfg, runs_dir=RUNS_DIR)


def _report(label: str, s: PrequentialSummary) -> None:
    print(
        f"{label:22s} bpb={s.bpb:.4f}  total={s.total_flops:.3e}  "
        f"(pretrain={s.pretrain_flops:.3e}  eval={s.eval_flops:.3e})",
        flush=True,
    )


def _diagnose() -> None:
    """Diagnostics on a freshly warmed learned column_mix: per-column load (does the gate
    collapse onto one column == the bar?), per-column conditional bpb (do columns specialize?), the
    delta-row mixer weight (→0 == dead, the A.1 tell), and gate drift off the hash prior."""
    corpus = prepare_enwik8(n_bytes=N_BYTES)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=EVAL_BYTES)
    cfg = {**ROUTE_CFG, "gate_lr": 0.2, "route_epsilon": 0.05, "seed": SEED}
    model = build_model("column_mix", cfg)
    pretrain(
        model,
        prior,
        flop_budget=PRETRAIN_BUDGET,
        batch_size=16,
        seq_len=SEQ_LEN,
        lr=0.02,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        seed=SEED,
        device=CPU,
    )
    k = model.num_predictors  # delta/column row is mixer slot K
    c = model.config.n_columns
    b = model.config.route_buckets
    # Gate drift: fraction of buckets whose argmax moved off the hash prior (b mod C).
    prior_arg = np.arange(b) % c
    drifted = int(np.sum(np.argmax(model._warm.gate, axis=1) != prior_arg))
    # Walk the eval stream; track per-column load + conditional bits (route is chosen each step).
    state = model.init_prequential_state()
    ln2 = float(np.log(2.0))
    load = np.zeros(c, dtype=np.int64)
    bits = np.zeros(c, dtype=np.float64)
    for pos in range(len(eval_stream) - 1):
        bn = int(eval_stream[pos])
        state, logits, _ = model.step(state, bn, pos)
        col = state.cache.last_route
        load[col] += 1
        lp = torch.log_softmax(logits, dim=-1)
        bits[col] += -float(lp[int(eval_stream[pos + 1])].item()) / ln2
    w_delta = float(model._warm.weights[k])
    print("\n--- diagnostics (warmed learned column_mix) ---", flush=True)
    print(f"final mixer weight on delta/column row: {w_delta:+.4f} (near 0 => dead)", flush=True)
    print(f"gate drift off hash prior: {drifted}/{b} buckets reassigned", flush=True)
    tot = int(load.sum())
    for ci in range(c):
        frac = load[ci] / tot if tot else 0.0
        cbpb = bits[ci] / load[ci] if load[ci] else float("nan")
        print(f"  column {ci}: load={frac:6.1%}  conditional_bpb={cbpb:.4f}", flush=True)
    if (load == 0).any() or (load.max() / max(tot, 1)) > 0.95:
        print(
            "  NOTE: gate near-collapse onto one column -> ~= the bar (an accepted outcome)",
            flush=True,
        )


def main() -> None:
    print(
        f"column_mix kill-test (real enwik8, {N_BYTES / 1e6:.0f}MB pool, eval {EVAL_BYTES}, "
        f"budget {PRETRAIN_BUDGET:.0e}, C={N_COLUMNS})\n",
        flush=True,
    )
    a = _run("delta_mix", dict(DELTA_CFG), "b5_a_delta")
    b = _run("column_mix", {**ROUTE_CFG, "gate_lr": 0.2, "route_epsilon": 0.05}, "b5_b_learned")
    c = _run("column_mix", {**ROUTE_CFG, "gate_lr": 0.0, "route_epsilon": 0.0}, "b5_c_gateoff")
    d = _run("delta_mix", {**DELTA_CFG, "delta_dim": N_COLUMNS * DELTA_DIM}, "b5_d_bigdim")
    _report("(a) delta [bar, C=1]", a)
    _report("(b) column_learned", b)
    _report("(c) column_gateoff", c)
    _report("(d) delta_bigdim [C*d]", d)
    best_route = min(b.bpb, c.bpb)
    passed = best_route < a.bpb and best_route < d.bpb
    print(
        f"\nVERDICT: best routed {best_route:.4f} vs bar(a) {a.bpb:.4f} / capacity(d) {d.bpb:.4f}"
        f"  -> {'PASS (routing earns its keep)' if passed else 'KILL (Pareto-hollow)'}",
        flush=True,
    )
    print(
        f"  selection (c vs a/d): {c.bpb:.4f} vs {a.bpb:.4f}/{d.bpb:.4f}   "
        f"learned gate (b vs c): {b.bpb:.4f} vs {c.bpb:.4f}",
        flush=True,
    )
    _diagnose()


if __name__ == "__main__":
    main()
