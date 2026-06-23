"""delta_mix kill-test on REAL enwik8 — Task B.4 (the gate before any full-corpus claim).

``delta_mix`` adds one online error-correcting fast-weight stream to the warmed count ensemble. The
bet (``docs/tasks/B.4-delta-mix.md``): its per-feature generalization buys more loss-reduction per
FLOP than spending the same FLOPs warming the cheap count ladder on more bytes. A.1/B.1/B.2-gated
all show "refine an already-good cheap learner" usually FAILS, so this matched-FLOP 3-way decides it
cheaply BEFORE the multi-hour full-corpus run:

- (a) ``counts_only``      — ``hashed_mix`` warmed at pretrain budget ``P`` (the cheap ladder);
- (b) ``delta``            — ``delta_mix`` at the SAME budget ``P`` (so it warms on fewer bytes,
      the delta stream stealing FLOPs from warmup);
- (c) ``counts_more_warm`` — ``hashed_mix`` warmed to (b)'s TOTAL FLOPs (the delta's FLOPs
      reallocated to MORE warm bytes — the binding per-FLOP comparison).

**VERDICT: delta_mix PASSES iff (b) beats BOTH (a) and (c) in bpb.** Beating (a) shows the stream
helps at a fixed warm level; beating (c) shows it beats just warming the counts more — the real
per-FLOP win. If (b) loses to (c), the delta stream is Pareto-hollow (kill it).

Two diagnostics make the verdict mechanistic: the delta stream's final learned mixer weight (->0 ==
dead weight, the A.1 tell) and delta-only vs top-count-order next-byte bpb on contexts unseen during
warm (the generalization claim is true iff delta wins there).

Bounded-but-real scale (pure-Python mixing ~50 us/byte): a real-enwik8 slice, a few-hundred-k eval
window, warmup capped in the low MB. NOT the full 5 MB ADR endpoint (that is ``full_corpus.py``,
gated on this PASS) — but a faithful real-text per-FLOP ordering. Run it (CPU; downloads enwik8 once
to ``data/cache/``, then minutes)::

    uv run python -m smolml.experiments.delta_mix_enwik8
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
RUNS_DIR = "runs/b4"
N_BYTES = 4_000_000  # first 4 MB of real enwik8 (prior ~= N_BYTES - EVAL_BYTES)
EVAL_BYTES = 32_768  # real-enwik8 eval window
SEED = 0
SEQ_LEN = 128
PRETRAIN_BUDGET = 1.0e10  # P: the shared pretrain budget for (a) and (b)
MAX_ORDER = 6
HASHED_CFG: dict[str, object] = {"max_order": MAX_ORDER, "hash_min_order": 4, "table_bits": 18}
DELTA_CFG: dict[str, object] = {
    **HASHED_CFG,
    "delta_dim": 1 << 18,
    "delta_orders": (3, 4, 5, 6, 7, 8),
    "delta_eta": 0.2,
}


def _run(
    model: str, cfg_extra: dict[str, object], budget: float, run_name: str
) -> PrequentialSummary:
    corpus = prepare_enwik8(n_bytes=N_BYTES)
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
        f"{label:28s} bpb={s.bpb:.4f}  total={s.total_flops:.3e}  "
        f"(pretrain={s.pretrain_flops:.3e}  eval={s.eval_flops:.3e})",
        flush=True,
    )


def _diagnose() -> None:
    """Mechanistic diagnostics on a freshly warmed delta_mix: (1) the delta stream's final mixer
    weight (a near-zero weight == the mixer found it useless, the A.1 'gate weight identical' tell);
    (2) delta-only vs top-count-order next-byte bpb on contexts UNSEEN during warm (the
    generalization claim is true iff delta wins there)."""
    corpus = prepare_enwik8(n_bytes=N_BYTES)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=EVAL_BYTES)
    model = build_model("delta_mix", dict(DELTA_CFG))
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
    k = model.num_predictors  # the delta stream is mixer slot K
    # Walk the eval stream; accumulate delta-only vs top-order bits on UNSEEN top n-grams.
    state = model.init_prequential_state()
    ms = state.cache
    seen_top: set[bytes] = set()
    top = MAX_ORDER
    d_bits = c_bits = 0.0
    n_unseen = 0
    ln2 = float(np.log(2.0))
    for pos in range(len(eval_stream) - 1):
        b = int(eval_stream[pos])
        window = state.tokens
        if len(window) >= top:
            key = bytes(window[-top:])
            if key not in seen_top:  # the top-order count abstains here (novel) — the delta test
                # delta-only distribution from the live W (pre-update prediction for byte pos+1):
                new_window = [*window, b][-model._window_cap :]
                idxs, signs = model._build_phi(new_window)
                z = (ms.W[:, idxs] * signs[None, :]).sum(axis=1) if idxs.shape[0] else np.zeros(256)
                pd = np.exp(z - z.max())
                pd /= pd.sum()
                nb = int(eval_stream[pos + 1])
                d_bits += -float(np.log(max(pd[nb], 1e-12))) / ln2
                c_bits += 8.0  # the abstaining top-order count == uniform == 8 bits/byte
                n_unseen += 1
            seen_top.add(key)
        state, _, _ = model.step(state, b, pos)
    w_delta = float(ms.weights[k])
    print("\n--- diagnostics (warmed delta_mix) ---", flush=True)
    print(
        f"final mixer weight on the delta stream: {w_delta:+.4f}  (near 0 => dead weight)",
        flush=True,
    )
    if n_unseen:
        print(
            f"on {n_unseen} top-order-UNSEEN contexts: delta-only {d_bits / n_unseen:.4f} bpb  vs  "
            f"order-{top} count (abstains) 8.0000 bpb  -> "
            f"{'delta carries signal' if d_bits / n_unseen < 8.0 else 'delta abstains too'}",
            flush=True,
        )
    else:
        print("no top-order-unseen contexts in the eval window (raise EVAL_BYTES)", flush=True)


def main() -> None:
    print(
        f"=== Task B.4 delta_mix kill-test (real enwik8 first {N_BYTES / 1e6:.0f}MB; "
        f"eval = final {EVAL_BYTES} bytes, disjoint) ===\n",
        flush=True,
    )
    a = _run("hashed_mix", dict(HASHED_CFG), PRETRAIN_BUDGET, "counts_only")
    b = _run("delta_mix", dict(DELTA_CFG), PRETRAIN_BUDGET, "delta")
    # (c) hashed_mix warmed to (b)'s TOTAL: give the counts the FLOPs delta spent on its stream.
    pretrain_c = max(0.0, b.total_flops - a.eval_flops)
    c = _run("hashed_mix", dict(HASHED_CFG), pretrain_c, "counts_more_warm")

    print("\n=== matched-FLOP 3-way (lower bpb at equal-or-lower total FLOPs wins) ===")
    _report("(a) counts_only", a)
    _report("(b) delta", b)
    _report("(c) counts_more_warm", c)

    beats_a = b.bpb < a.bpb
    beats_c = b.bpb < c.bpb
    verdict = "PASS" if (beats_a and beats_c) else "FAIL (Pareto-hollow)"
    print(
        f"\n[verdict] delta beats (a)? {beats_a}  beats (c)? {beats_c}  ->  {verdict}\n"
        f"  (c) total {c.total_flops:.3e} vs (b) total {b.total_flops:.3e} "
        f"(matched-total quality)",
        flush=True,
    )
    _diagnose()


if __name__ == "__main__":
    main()
