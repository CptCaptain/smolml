# Experiment B.4 — online delta-rule fast-weight memory (the first non-hollow Space-B win)

> **Researcher note for the docs-builder.** Plain-md source for the MDX page. Reuse
> `BpbFlopChart`, `Callout`, `ConceptMap`, and the role colors in `global.css`. The full-carve has
> LANDED — `delta_o6_warmfull` = **1.8485 bpb @ 1.322e12 FLOPs** (`runs/full/leaderboard.md`).

**Status:** mechanism sound, honestly metered, **cross-vendor reviewed**, and — for the first time
in a Space-B (learning-rule) candidate — **NOT Pareto-hollow**. Both verdicts are in and BOTH are
wins: the CI matched-FLOP kill-test PASSES, and the **full ADR carve beats the bar outright** —
`delta_o6_warmfull` reaches **1.8485 bpb @ 1.322e12 total FLOPs**, **strictly dominating** the
previous bar `hashed_o6_warmfull` (2.0157 @ 1.478e12) on BOTH axes (−0.167 bpb AND fewer FLOPs,
while warming on fewer bytes). The first candidate to beat the warmed-mixing bar.

## The bet (Source-(iv))

Every prior fast-weight/PC attempt died the same way: it *refined an already-good cheap learner*,
and a free/cheap online baseline dominated it per FLOP. So this candidate attacks a **structural
weakness of the count mixer itself**, not its compute budget.

A count table is a *degenerate associative memory*: a **one-hot-key Hebbian** store — each exact
context owns a cell, bumped by `+1`. Two hard limits follow:

1. **Zero generalization.** A never-seen k-gram makes that order *abstain*; at scale most
   high-order contexts are novel.
2. **Only `K` global mixer weights** combine the orders — "order-3 is worth `w_3` *everywhere*." It
   cannot learn that *some* trigrams are far more predictive than others.

`delta_mix` adds **one** online predictor whose learning rule is an **error-correcting delta (LMS)
update** on a *distributed* key, so it holds `d×V` **per-feature, per-byte** affinities: it learns
"`tio`→`n` is a strong rule" once and shares it across every context containing `tio`, weighted by
its *measured residual utility*. It also reaches **beyond the order-`max_order` ladder** (orders
7–8 in the key at ~free marginal cost).

**Why it's a real (iv) claim — honestly.** It *is* online gradient descent, but the model is
**linear in a fixed feature map** and the loss is **convex**, so the exact gradient is `(p−target)
⊗ φ` — a single rank-1 outer product with **zero backward pass, no chain rule, no 2× tax, no bad
minima**. That is the legitimate "more loss-reduction per FLOP than SGD-on-a-net" edge; the win
over the *count mixer* is the per-feature generalization above.

## Mechanism (one extra stream on the warmed hashed ladder)

- **Key** `φ(ctx)`: for each n-gram order in `delta_orders` (default 3–8), hash the suffix n-gram
  (the same Fibonacci hash the count tables use) to a bucket in `[0, d)`, with a **±1 sign** from a
  second hash (signed feature-hashing — collisions cancel in expectation). `φ` is `s`-sparse.
- **Predict** `z = W·φ` (touches only the `s` active columns) → one more **raw-logit** row in the
  existing logistic mixer.
- **Learn** (deferred to the next byte; no leakage): `W[:,j] −= η·φ[j]·(softmax(z) − onehot(byte))`
  over the `s` touched columns. Error-correcting — *not* plain Hebbian (which collapses to the byte
  marginal on a superposed key, the A.1 death; an ablation test confirms delta beats it).

**Feasibility crux (FLOP honesty):** sparsity makes both the read and the rank-1 write `O(sV)`, not
`O(dV)` — a dense key would be ~5600× the bar and dead. `d` (default `2¹⁸`, ~268 MiB) costs only RAM
and collision noise, never FLOPs. Add-on ≈ **8.2k FLOPs/byte** on the bar's ~15.7k; every op charged
via `smolml.flops` (the cross-vendor review caught two real discrepancies — a sign-hash overcharge
on the unsigned path and a colliding-column write that didn't accumulate; both fixed).

## Result — the matched-FLOP kill-test (real enwik8, 4 MB slice)

The reflex that killed every false win: *plot the cheap baselines first.* This kill-test does that
**at matched total FLOPs** (≈1.07e10) — the candidate must beat both, or it's hollow:

| run | bpb | total FLOPs | what it isolates |
| --- | ---: | ---: | --- |
| (a) `counts_only` (hashed order-6) | 2.4353 | 1.050e10 | the cheap ladder at budget |
| **(b) `delta` (counts + delta)** | **2.4181** | 1.074e10 | the candidate |
| (c) `counts_more_warm` (hashed order-6) | 2.4327 | 1.072e10 | the SAME FLOPs, all on more warm counts |

**Verdict: PASS** — `delta` beats **both** (a) and (c). The binding comparison is **(b) vs (c)** at
matched total FLOPs: spending the delta stream's FLOPs on the learning rule beats spending them on
more warm count bytes by **−0.0146 bpb**. Small, but on the right side of the line A.1/B.1/B.2-gated
all fell on.

### Two mechanistic diagnostics (why it's real, not noise)

- **The mixer trusts it.** Final learned mixer weight on the delta stream = **+0.8595** — strongly
  load-bearing, the opposite of A.1's tell (where the gate weight was ~identical trained-vs-untrained,
  i.e. dead). The ensemble *chose* to lean on the delta stream.
- **It generalizes where counts abstain.** On **20,051 eval contexts whose exact order-6 n-gram was
  never seen during warm** (so the order-6 count abstains → 8.0 bpb), the **delta stream alone scores
  3.73 bpb**. That is the (iv) mechanism made visible: a distributed key carries signal from seen
  sub-n-grams exactly where an exact-match table has nothing.

## Honest framing (what is and isn't established)

- The CI-scale per-FLOP win was **real but modest** (−0.0146 bpb at matched FLOPs) — enough to clear
  the cheap-baseline bar that killed every prior Space-B candidate.
- At **full corpus the edge GREW, not shrank**: `delta_o6_warmfull` reaches **1.8485 bpb @ 1.322e12**
  vs the bar's 2.0157 @ 1.478e12 — **strictly dominating** (−0.167 bpb AND fewer total FLOPs), and it
  did so warming on *fewer* bytes (1.2e12 budget vs 1.4e12). The concern that saturating counts would
  erase the delta edge was wrong — the per-feature generalization compounds as more contexts go novel
  at high order. The first candidate to beat the warmed-mixing bar; `delta_mix` is the new bar.
- The downside is bounded: error-correction prevents A.1's marginal-collapse, and the online mixer
  drives a useless stream's weight → 0 with no bpb blow-up.

## What's worth visualizing (docs-builder)

1. **The kill-test as a `BpbFlopChart`** — plot (a)/(b)/(c) (and ideally the full-carve points + the
   2.0157 bar) on bpb-vs-total-FLOP; the story is "(b) sits below the counts-only frontier." Mark the
   matched-total pair (b,c) explicitly.
2. **A `Callout`** for the binding insight: *beating an internal variant is not a per-FLOP win — beat
   the cheap baseline at matched FLOPs.* This is the project's recurring reflex, finally cleared.
3. **The generalization diagnostic** — a tiny two-bar comparison (delta-only 3.73 vs abstaining count
   8.0 on unseen contexts) is the clearest single picture of *why* it works.
4. **A `ConceptMap` edge** from `online-warmup`/`context-mixing` to a new/extended
   `fast-weight-memory` concept (delta-rule / DeltaNet / signed feature-hashing lineage), contrasted
   with A.1's transformer-bolt-on fast weights.

## Reproduce

```bash
# CI-fast kill-test (downloads enwik8 once; minutes):
uv run python -m smolml.experiments.delta_mix_enwik8
# full ADR carve headline (multi-hour, detached), the delta entrant in:
uv run python -m smolml.experiments.full_corpus   # runs the cast incl. delta_o6_warmfull
```

Tests: `tests/test_delta_mix.py` (degenerate identity, exact FLOP charge, no-leakage,
error-correction-beats-Hebbian, generalization-beyond-cap, colliding-column accumulation, leak-free
warm handoff). Model: `smolml/models/delta_mix.py`. Spec: `docs/tasks/B.4-delta-mix.md`.
