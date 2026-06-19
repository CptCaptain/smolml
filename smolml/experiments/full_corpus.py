"""The full enwik8 ADR carve — Task B.3 (the run B.2 could not do).

ADR-0004 carve on the REAL full corpus: final **5 MB = fixed prequential eval stream**, first
**~95 MB = freely-usable prior** (warmup). B.2 ran a 4 MB slice because the order-6 win used
unbounded dict tables that OOM (~58 GB) on a full-corpus warmup. `hashed_mix` (B.3) bounds the
high-order tables to fixed memory, so the full carve becomes feasible — at the cost of collision
noise. The open question: **does the order-6 per-FLOP win survive bounded-memory hashing at full
corpus, and does it still beat the transformer?**

Cast (cheapest first, so the answer lands before the slow transformer; each result is written +
the leaderboard regenerated incrementally, so a box recycle keeps the early/headline points):

1. `context_mixing` reference — cold, order-3 (the cheap online ceiling).
2. `hashed_mix` cold order-6 — no warmup (the transductive-handicap point).
3. `hashed_mix` warmed order-6 @1e11 — a quick (~few-MB) warmup (early warmed signal).
4. `hashed_mix` warmed order-6 @1.4e12 — full ~95 MB warmup (THE headline).
5. `transformer` anchor — the expensive baseline (windowed recompute over 5 MB ~ hours).

`gated_mix` is excluded: it is dict-based (would OOM at full corpus) and its Pareto-hollow verdict
is already established at 4 MB (B.2). `table_bits=20` keeps peak RAM ~4 GB (orders 4-6 hashed).

Multi-hour on CPU — run it DETACHED::

    nohup uv run python -m smolml.experiments.full_corpus > runs/full/run.log 2>&1 &

Monitor ``runs/full/run.log`` and ``runs/full/leaderboard.md``.
"""

import resource
import time

from smolml.data.corpus import prepare_enwik8
from smolml.leaderboard import regenerate
from smolml.prequential import PrequentialConfig, PrequentialSummary, prequential_run

RUNS_DIR = "runs/full"
EVAL_BYTES = 5_000_000  # the ADR-0004 fixed eval stream (final 5 MB)
SEED = 0
SEQ_LEN = 128
# hashed order-6: orders 4-6 in fixed 2**20-slot tables (~0.5 GiB/order, ~1.5 GiB; eval doubles it).
HASHED_CFG: dict[str, object] = {"max_order": 6, "hash_min_order": 4, "table_bits": 20}
CORE_CFG: dict[str, object] = {"d_model": 48, "n_layers": 3, "n_heads": 4, "max_seq_len": 128}


def _peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # ru_maxrss is KiB on Linux


def _run(model: str, cfg_extra: dict[str, object], budget: float, run_name: str) -> None:
    """Run one entrant on the full ADR carve, write its log + regenerate the leaderboard, so
    each result is durable the moment it finishes (cheapest entrants first)."""
    t0 = time.time()
    corpus = prepare_enwik8(n_bytes=None)  # full 100 MB (cached)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=EVAL_BYTES)
    cfg = PrequentialConfig(
        model=model,
        model_config=cfg_extra,
        pretrain_flop_budget=budget,
        seq_len=SEQ_LEN,
        seed=SEED,
        run_name=run_name,
    )
    try:
        s: PrequentialSummary = prequential_run(prior, eval_stream, cfg, runs_dir=RUNS_DIR)
        print(
            f"[{run_name}] bpb={s.bpb:.4f} total={s.total_flops:.3e} "
            f"(pretrain={s.pretrain_flops:.3e} eval={s.eval_flops:.3e}) "
            f"| {time.time() - t0:.0f}s peak_rss={_peak_gib():.1f}GiB",
            flush=True,
        )
        regenerate(
            RUNS_DIR,
            table_path=f"{RUNS_DIR}/leaderboard.md",
            plot_path=f"{RUNS_DIR}/leaderboard.png",
        )
    except Exception as exc:  # keep earlier results; report and continue the cast
        print(f"[{run_name}] FAILED after {time.time() - t0:.0f}s: {exc!r}", flush=True)


def main() -> None:
    print(
        f"=== Full enwik8 ADR carve (eval={EVAL_BYTES / 1e6:.0f}MB, prior=full ~95MB) ===",
        flush=True,
    )
    _run("context_mixing", {"max_order": 3}, 0.0, "reference_cold")
    _run("hashed_mix", dict(HASHED_CFG), 0.0, "hashed_o6_cold")
    _run("hashed_mix", dict(HASHED_CFG), 1e11, "hashed_o6_warm1e11")
    _run("hashed_mix", dict(HASHED_CFG), 1.4e12, "hashed_o6_warmfull")
    _run("transformer", dict(CORE_CFG), 2e10, "transformer")
    table, _ = regenerate(
        RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md", plot_path=f"{RUNS_DIR}/leaderboard.png"
    )
    print("\n=== final leaderboard ===\n" + table, flush=True)


if __name__ == "__main__":
    main()
