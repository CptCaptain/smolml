"""gated_mix vs the fixed-order warm_mix curve on REAL enwik8 — Task B.2, Phase 2 verdict.

The Phase-1 order curve showed deep context pays to ~order-6 on real enwik8, but fixed-order
``warm_mix`` evaluates *all* orders on *every* byte. ``gated_mix`` escalates orders cheapest-first
and stops on a pre-reveal ``1 - max p`` gate, paying for only the orders it evaluates. The (iv)
question: does gated escalation reach the deep-order bpb at a *fraction* of the FLOPs — i.e. sit
below/left of the fixed-order frontier on bpb-vs-total-FLOP?

Same real-enwik8 carve as Phase 1 (4 MB slice, 32 k eval). All entrants warmed at the same budget
so only the prediction-time mechanism differs.

Run it (CPU; enwik8 cached from Phase 1; a few minutes)::

    uv run python -m smolml.experiments.gated_mix_enwik8

Writes ``runs/b2gated/*.jsonl`` + ``runs/b2gated/leaderboard.{md,png}`` and prints the verdict.
"""

from smolml.data.corpus import prepare_enwik8
from smolml.leaderboard import regenerate
from smolml.prequential import PrequentialConfig, PrequentialSummary, prequential_run

RUNS_DIR = "runs/b2gated"
N_BYTES = 4_000_000
EVAL_BYTES = 32_768
SEED = 0
SEQ_LEN = 128
WARMUP_BUDGET = 1e9
MAX_ORDER = 6  # the depth the order curve says pays on real enwik8


def _run(model: str, cfg_extra: dict[str, object], run_name: str) -> PrequentialSummary:
    corpus = prepare_enwik8(n_bytes=N_BYTES)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=EVAL_BYTES)
    cfg = PrequentialConfig(
        model=model,
        model_config=cfg_extra,
        pretrain_flop_budget=WARMUP_BUDGET,
        seq_len=SEQ_LEN,
        seed=SEED,
        run_name=run_name,
    )
    return prequential_run(prior, eval_stream, cfg, runs_dir=RUNS_DIR)


def _row(label: str, s: PrequentialSummary) -> None:
    print(f"{label:24} {s.bpb:8.4f} {s.total_flops:11.3e} {s.eval_flops:11.3e}")


def main() -> None:
    # The bar: fixed-order warm_mix curve (every order evaluated every byte).
    fixed: list[PrequentialSummary] = [
        _run("warm_mix", {"max_order": k}, f"warm_mix_o{k}") for k in (2, 3, 4, 5, 6)
    ]
    # gated_mix: hold orders 0..MAX_ORDER, escalate on a pre-reveal gate. Higher threshold =>
    # stops escalating sooner => cheaper. Sweep from near-full (0.1) to aggressive (0.7).
    gated: list[tuple[float, PrequentialSummary]] = [
        (
            t,
            _run(
                "gated_mix",
                {"max_order": MAX_ORDER, "min_order": 1, "gate_threshold": t},
                f"gated_t{t}",
            ),
        )
        for t in (0.1, 0.3, 0.5, 0.7)
    ]

    print("\n=== Task B.2 Phase 2 — gated_mix vs fixed-order warm_mix on REAL enwik8 ===")
    print(f"(enwik8 {N_BYTES // 1_000_000}MB; eval={EVAL_BYTES}B; warmup={WARMUP_BUDGET:.0e})\n")
    print(f"{'entrant':24} {'bpb':>8} {'total':>11} {'eval':>11}")
    for k, s in zip((2, 3, 4, 5, 6), fixed, strict=True):
        _row(f"fixed warm_mix order-{k}", s)
    for t, s in gated:
        _row(f"gated_mix thr={t}", s)

    # Pareto verdict: gated WINS if no fixed-order point has both <= bpb AND <= FLOPs.
    print("\n[verdict] gated points vs the fixed-order frontier (total FLOPs):")
    any_win = False
    for t, g in gated:
        dominated_by = [
            f"order-{k}"
            for k, s in zip((2, 3, 4, 5, 6), fixed, strict=True)
            if s.bpb <= g.bpb + 1e-9 and s.total_flops <= g.total_flops
        ]
        verdict = (
            "DOMINATED by " + ",".join(dominated_by)
            if dominated_by
            else "on/extends the frontier (no fixed point dominates it)"
        )
        if not dominated_by:
            any_win = True
        print(f"  thr={t}: bpb={g.bpb:.4f} @ {g.total_flops:.3e}  ->  {verdict}")
    print(
        "\n  => gated escalation",
        "EXTENDS the per-FLOP frontier."
        if any_win
        else "does NOT beat fixed-order; report honestly.",
    )

    table, png = regenerate(
        RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md", plot_path=f"{RUNS_DIR}/leaderboard.png"
    )
    print("\n" + table)
    print(f"\nplot: {png}")


if __name__ == "__main__":
    main()
