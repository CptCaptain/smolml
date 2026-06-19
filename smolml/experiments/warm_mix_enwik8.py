"""warm_mix vs the transformer on the REAL enwik8 corpus — Task B.2, Phase 1.

The B.1 design fan-out's critics surfaced (and an in-session probe confirmed on the synthetic
clone) that the context-mixing reference loses to the transformer only from its *transductive
handicap* (cold start), not its structure. ``warm_mix`` removes that handicap cheaply — it folds
the freely-usable prior corpus into its count tables (warmup, **all FLOPs counted**) and hands the
warmed state to the eval stream. This runner tests the claim on **real enwik8** text.

Bounded-but-real scale (pure-Python context-mixing is ~53 us/byte, so the full 95MB/5MB ADR carve
is intractable here): a real-enwik8 slice, a few-hundred-k eval window, warmup capped at ~1M bytes.
Real text, prior/eval structurally disjoint (tail carve), every FLOP on the curve. NOT the full-5MB
ADR endpoint (still GPU/opt-in for the transformer) — but a faithful real-text per-FLOP ordering.

Run it (CPU; downloads enwik8 once to data/cache/, then a few minutes)::

    uv run python -m smolml.experiments.warm_mix_enwik8

Writes ``runs/b2/*.jsonl`` + a matched ``runs/b2/leaderboard.{md,png}`` and prints the curve.
"""

from smolml.data.corpus import prepare_enwik8
from smolml.leaderboard import regenerate
from smolml.prequential import PrequentialConfig, PrequentialSummary, prequential_run

RUNS_DIR = "runs/b2"
N_BYTES = 4_000_000  # first 4 MB of real enwik8 (prior ~= N_BYTES - EVAL_BYTES)
EVAL_BYTES = 32_768  # real-enwik8 eval window (bounded so the transformer recompute is tractable)
SEED = 0
SEQ_LEN = 128
MAX_ORDER = 3
# Transformer anchor: the same small core used throughout the project (B.1 baseline).
CORE_CONFIG: dict[str, object] = {"d_model": 48, "n_layers": 3, "n_heads": 4, "max_seq_len": 128}


def _run(
    model: str, cfg_extra: dict[str, object], budget: float, run_name: str
) -> PrequentialSummary:
    corpus = prepare_enwik8(n_bytes=N_BYTES)  # cached after first download
    prior, eval_stream = corpus.prequential_carve(eval_bytes=EVAL_BYTES)
    cfg = PrequentialConfig(
        model=model,
        model_config=cfg_extra,
        pretrain_flop_budget=budget,
        seq_len=SEQ_LEN,
        seed=SEED,
        run_name=run_name,
    )
    return prequential_run(prior, eval_stream, cfg, runs_dir=RUNS_DIR)


def _report(label: str, s: PrequentialSummary) -> None:
    print(
        f"{label:30s} bpb={s.bpb:.4f}  total={s.total_flops:.3e}  "
        f"(pretrain={s.pretrain_flops:.3e}  eval={s.eval_flops:.3e})  params={s.params}"
    )


def main() -> None:
    mix = {"max_order": MAX_ORDER}
    # warm_mix swept over warmup budget: 0 == the cold context-mixing reference point.
    cold = _run("warm_mix", mix, 0.0, "warm_mix_cold")
    warm1 = _run("warm_mix", mix, 1e9, "warm_mix_b1e9")
    warm2 = _run("warm_mix", mix, 1e10, "warm_mix_b1e10")
    # bare transformer anchor (the expensive baseline; eval is windowed recompute).
    tf = _run("transformer", dict(CORE_CONFIG), 2e10, "transformer_b2e10")

    print("\n=== Task B.2 Phase 1 — warm_mix vs transformer on REAL enwik8 ===")
    print(f"(real enwik8 first {N_BYTES / 1e6:.0f}MB; eval = final {EVAL_BYTES} bytes, disjoint)\n")
    _report("warm_mix cold (== reference)", cold)
    _report("warm_mix warmed @1e9", warm1)
    _report("warm_mix warmed @1e10", warm2)
    _report("transformer (anchor)", tf)

    best = min((cold, warm1, warm2), key=lambda s: s.bpb)
    print(
        f"\n[per-FLOP] best warm_mix: bpb={best.bpb:.4f} @ {best.total_flops:.3e} FLOPs  vs  "
        f"transformer bpb={tf.bpb:.4f} @ {tf.total_flops:.3e}"
    )
    if best.bpb <= tf.bpb and best.total_flops < tf.total_flops:
        print("  -> warm_mix STRICTLY DOMINATES the transformer (lower bpb AND fewer total FLOPs).")
    else:
        print("  -> NOT a strict domination; read the curve honestly.")

    table, png = regenerate(
        RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md", plot_path=f"{RUNS_DIR}/leaderboard.png"
    )
    print("\n" + table)
    print(f"\nplot: {png}")


if __name__ == "__main__":
    main()
