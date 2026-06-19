"""Surprise-gated vs uniform predictive-coding refinement — the Task B.1 (iv) controlled run.

Produces a **matched, apples-to-apples** bpb-vs-total-FLOP comparison on ONE identical
CI-scale ``synthetic_text8`` carve (same prior + same eval stream for every entrant):

- ``transformer`` — the frozen slow core alone (the "does PC refinement help at all?" baseline),
- ``context_mixing`` — the online context-mixing reference (the per-FLOP ceiling, ADR/§0.3),
- ``pc_refine`` (``gate="uniform"``) — PC refinement with constant settling depth,
- ``pc_refine`` (``gate="surprise"``) — PC refinement with surprise-gated settling depth.

The headline (iv) test is the **uniform vs gated** pair: the two ``pc_refine`` runs differ *only*
in the settling-depth ``gate`` at matched ``k``, so realized mean K matches and total FLOPs are
(near-)identical — the only lever is *how* a matched settling budget is allocated across bytes.
Falsifiable claim: at matched total FLOPs, the surprise-gated run reaches lower bpb because
settling concentrates on hard bytes. Honest by construction: ``K`` is data-dependent, so the
harness sums the per-byte ``step`` FLOPs — read ``total_flops``, never an assumption.

NB (reported, not hidden): on this order-0 synthetic corpus the per-byte difficulty structure is
weak, so the gate has little to act on and the effect is small + budget-fragile; below ~2e11
pretrain the core is barely trained and the gate is degenerate (gated == uniform). This is a
CI-scale *demonstration of the lever*, not the enwik8 verdict.

Run it (CPU; ~90s, dominated by the three amortized core pretrains)::

    uv run python -m smolml.experiments.pc_refine_sweep

Writes ``runs/b1/{transformer,context_mixing,pc_refine_uniform,pc_refine_gated}.jsonl`` plus a
matched ``runs/b1/leaderboard.{md,png}`` curve, and prints the table + the matched-FLOP delta.
"""

from smolml.data import synthetic_text8
from smolml.leaderboard import regenerate
from smolml.prequential import PrequentialConfig, PrequentialSummary, prequential_run

RUNS_DIR = "runs/b1"
CORPUS_BYTES = 8000
EVAL_BYTES = 1200
PRETRAIN_FLOP_BUDGET = 2e11  # smallest scale where the trained core exposes per-byte surprise
SEED = 0
SEQ_LEN = 64

# Shared slow-core hyperparameters (the transformer baseline uses exactly these core keys).
CORE_CONFIG: dict[str, object] = {"d_model": 48, "n_layers": 3, "n_heads": 4, "max_seq_len": 128}
# The two pc_refine runs override ONLY ``gate`` (and share every ``k``), so realized mean K
# matches and the only difference is the per-byte allocation of a matched settling budget.
PC_CONFIG: dict[str, object] = {
    **CORE_CONFIG,
    "m": 32,
    "eta": 0.1,
    "k_min": 1,
    "k_max": 7,
    "k_uniform": 4,
    "gate_sensitivity": 1.5,
    "gate_eps": 1e-3,
    "surprise_ema": 0.05,
    "update_surprise_threshold": 0.5,
    "lr_readout": 0.2,
    "lr_gen": 0.05,
    "weight_decay_fast": 0.01,
}


def _run(
    model: str,
    model_config: dict[str, object],
    pretrain_budget: float,
    run_name: str,
) -> PrequentialSummary:
    """Pretrain + prequentially score one entrant on the shared carve (fresh per call so the
    eval stream is byte-identical and the comparison is fair)."""
    corpus = synthetic_text8(CORPUS_BYTES, seed=SEED)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=EVAL_BYTES)
    cfg = PrequentialConfig(
        model=model,
        model_config=model_config,
        pretrain_flop_budget=pretrain_budget,
        seq_len=SEQ_LEN,
        seed=SEED,
        run_name=run_name,
    )
    return prequential_run(prior, eval_stream, cfg, runs_dir=RUNS_DIR)


def _report(label: str, s: PrequentialSummary) -> None:
    print(
        f"{label:22s} bpb={s.bpb:.4f}  total_flops={s.total_flops:.3e}  "
        f"(pretrain={s.pretrain_flops:.3e}  eval={s.eval_flops:.3e})  "
        f"eval_bytes={s.eval_bytes}  params={s.params}"
    )


def main() -> None:
    core = _run("transformer", dict(CORE_CONFIG), PRETRAIN_FLOP_BUDGET, "transformer")
    mixer = _run("context_mixing", {}, 0.0, "context_mixing")
    uniform = _run(
        "pc_refine", {**PC_CONFIG, "gate": "uniform"}, PRETRAIN_FLOP_BUDGET, "pc_refine_uniform"
    )
    gated = _run(
        "pc_refine", {**PC_CONFIG, "gate": "surprise"}, PRETRAIN_FLOP_BUDGET, "pc_refine_gated"
    )

    print("\n=== Task B.1 — surprise-gated PC refinement (matched-stream comparison) ===")
    _report("transformer (core)", core)
    _report("context_mixing (ceiling)", mixer)
    _report("pc_refine uniform", uniform)
    _report("pc_refine gated", gated)

    bpb_delta = gated.bpb - uniform.bpb
    flop_ratio = gated.total_flops / uniform.total_flops
    verdict = "BEAT" if bpb_delta < 0 else "did NOT beat"
    print(
        f"\n[iv] gated − uniform: Δbpb={bpb_delta:+.4f} bits/byte at "
        f"{flop_ratio:.3f}× uniform total FLOPs → gating {verdict} uniform."
    )
    print(f"pc_refine vs bare core: Δbpb={uniform.bpb - core.bpb:+.4f} (uniform − transformer)")

    table, png = regenerate(
        RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md", plot_path=f"{RUNS_DIR}/leaderboard.png"
    )
    print("\n" + table)
    print(f"\nplot: {png}")


if __name__ == "__main__":
    main()
