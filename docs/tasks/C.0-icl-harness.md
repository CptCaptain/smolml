# Task C.0 — the ICL measuring spine (graded in-context-learning harness)

- Status: SPEC — design approved (pivot to ADR 0007). Phase 0 of the new direction; no candidate
  yet — this builds the metric, the eval, and the baseline.
- Depends on: nothing new (reuses the model seam, FLOP counter, transformer, training loop).
- Branch: `task/C.0-icl-harness` off `main`. Own PR; do not merge (human merges).

## Why (ADR 0007)

The enwik8 bpb-per-FLOP metric rewarded **memorization** (count tables, hashing, B.4's 268 MB delta
store all won by being better lookup machines). We are pivoting the north star to **held-out
in-context-learning (ICL) loss per training FLOP, at a fixed parameter/memory budget** — a cheap,
falsifiable proxy that memorization *cannot* win (fresh tasks, held-out rules) and that tracks the
chatbot/agent capability (learn a new task from examples). This task builds the measuring spine, as
0.1 did for the old metric. The transformer is the honest baseline (attention does ICL via induction
heads). The first deliverable is **infrastructure + baseline, not a candidate.**

## The metric

For a trained model at a fixed parameter budget `P`: **held-out ICL loss** = mean cross-entropy in
**bits, on the query/answer positions only**, over a fixed seeded set of **held-out-task** sequences,
reported **per rung** and as an aggregate, plotted vs **total training FLOPs**. Lower at equal
`(FLOPs, P)` wins. Training FLOPs are counted by the shared `smolml.flops` accounting (forward +
the model's own update); the eval forward is **not** charged (amortized, as `evaluate_bpb` is).
Report `P` (params) and peak state bytes — the memory constraint is first-class.

## The graded suite (3 rungs) — `smolml/data/icl.py`

Small configurable symbol vocab (NOT raw bytes): `n_symbols` data symbols `[0, n_symbols)` plus
control tokens `MAP` and `SEP` (and `PAD`); the model's `vocab_size` is set to match. Every sequence
is `x1 MAP y1 SEP x2 MAP y2 SEP … xk MAP yk SEP xq MAP yq`; the model predicts the answer token(s)
`yq`. A boolean **query mask** marks exactly the `yq` positions — loss is computed there only. The
context never contains `yq` before it is predicted (no leakage — a test asserts this).

The generator is `make_icl_batch(rung, split, *, batch_size, n_shots, seed, ...) -> (tokens,
targets, query_mask)` with `split ∈ {"train","eval"}` drawing from **disjoint rule pools**, so eval
measures generalization to **unseen rules**, not recall of trained ones.

1. **Recall** (induction-head floor). A fresh random injective map of `k` distinct random keys →
   random values (length-`L` symbol strings). Present the `k` pairs, then `xq` = one of the keys;
   `yq` = its value. Every sequence is fresh random, so it is memorization-proof by construction;
   `split` only changes the seed stream. Capability: retrieve a value from context by key.
2. **Rule-induction.** A fresh deterministic function `f` drawn from a family `F` (v1:
   **random permutations of the symbol alphabet** = substitution ciphers; also support modular
   shifts). Present `k` demo pairs `x_i MAP f(x_i)` for random `x_i`, then `xq MAP ?` with `xq` a
   symbol **not shown in the demos**; `yq = f(xq)`. **Split `F` into disjoint train/eval permutation
   pools** (held-out ciphers). The model must induce `f` from the shots and apply it to a new input.
3. **Composition** (systematic generalization). The per-symbol map `f` is demonstrated on individual
   symbols/short strings; the query is a **novel arrangement** — a longer/recombined string whose
   every symbol's mapping was demonstrated but whose specific combination/length was **not** (a
   miniaturized SCAN-style held-out split). `yq = f` applied element-wise to `xq`. Tests applying a
   learned rule systematically to held-out combinations. (Highest design-risk rung — pin the
   held-out split precisely in code + a test; an alternative `g∘h` two-rule-composition flavor is a
   documented extension, not v1.)

## The scorer — `smolml/icl_eval.py` (mirrors `eval.py`)

`evaluate_icl(model, rung, *, split="eval", n_batches, batch_size, n_shots, seed, device) -> float`:
generate held-out-task sequences, forward `model(tokens)`, compute `cross_entropy_bits` on the
**query-mask positions only**, average over the seeded set → held-out ICL bits. Reuse
`cross_entropy_bits`. Deterministic seed so the eval set is identical across runs.

## Training — reuse the loop

Reuse `train.py`'s FLOP-budgeted loop and JSONL logging, sourcing **ICL train-batches** instead of
`get_batch` over a corpus (v1: train on an even **mixture** of the three rungs; eval **per rung** to
expose where ICL breaks — per-rung training is a documented option). Fixed seed, fixed params. A
thin driver `smolml/experiments/icl_baseline.py` trains the transformer across a small FLOP sweep,
evaluates per rung, and writes the leaderboard. The model seam is untouched: `model(tokens)->logits`
works on ICL sequences as-is, so every future Space-C candidate plugs in with no harness change.

## Leaderboard

Extend the leaderboard to the ICL metric: per-rung held-out ICL bits + aggregate vs total training
FLOPs at fixed `P`; the transformer traces the bar. Reuse the plotting where possible.

## FLOP honesty & memory

- Training FLOPs via the shared counter (the transformer's `flops`/`train_step`); eval forward not
  charged. Every future candidate charges its own train+update honestly (the project invariant).
- **Memory is a hard constraint:** the metric compares at fixed `P`; report params + peak state
  bytes. No unbounded lookup tables (the whole point of the pivot).

## Acceptance criteria

- Gates green: `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest` (paste outputs).
- `tests/test_icl.py`:
  - **format & mask**: tokens follow `x MAP y SEP …`; the query mask marks exactly the `yq`
    positions; targets are the next-token shift of those positions; and the prediction is **causal**
    — `yq` is scored from the tokens strictly preceding it and is never fed as input before being
    scored (**no leakage**). NB: for recall, `yq`'s *value* legitimately appears earlier in the
    context (it is demonstrated) — that is the retrieval task, not leakage; the invariant is only
    that the answer at the query position is not given to the model before it predicts it;
  - **held-out disjointness**: rule-induction/composition train vs eval rule pools are disjoint;
    recall is fresh-random;
  - **recall is learnable in principle / chance baseline**: an oracle that copies the matching
    value scores ~0 bits; a model with no context scores ≈ `log2(n_symbols)` bits on the answer
    (sanity bounds the metric);
  - **composition held-out split** is genuinely novel (the query arrangement is absent from demos);
  - **scorer** computes bits only on query positions; deterministic under fixed seed;
  - tiny **end-to-end smoke**: train a small transformer a few FLOP-budgeted steps on the mixture,
    `evaluate_icl` per rung returns finite bits that **improve over the no-context chance baseline**,
    and a leaderboard row is written.
- `docs/harness.md`: how to run an ICL baseline, add a Space-C candidate, regenerate the ICL
  leaderboard.
- Per AGENTS.md: a researcher note + an `in-context-learning` concept handed to the docs-builder
  (the Phase-0 baseline finding once it lands).

## Out of scope (later tasks / candidates)

- Any non-transformer ICL mechanism (Space C: linear attention / DeltaNet, SSMs, RWKV, TTT /
  meta-plasticity) — those are candidates scored on this spine, not part of Phase 0.
- The `g∘h` two-rule composition flavor; multi-token answers beyond length-`L`; curriculum/per-rung
  training schedules (documented options, not v1).
- Real downstream/agent evals (ADR 0007: the synthetic graded suite is the cheap, falsifiable
  stand-in).
