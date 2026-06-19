# Experiment log

One entry per experiment. Failures are first-class data — log them as carefully as wins. Newest
on top. Each entry should link the concept pages it touches and embed/link the bpb-vs-FLOP plot.

## Template

```
## YYYY-MM-DD — <short title>  [status: planned | running | done]
- Concepts: [link], [link]
- Hypothesis: what we expect to move, and why (the Source-(iv) story).
- Setup: model/config, prior-corpus usage, total-FLOP budget(s), seed.
- Result: bpb-vs-FLOP curve (link to plot) + the headline number vs. baseline.
- Verdict: beat-baseline / lost / inconclusive.
- What we learned: the actual insight, including anything that surprised us.
```

## Entries

## 2026-06-19 — Warmed online mixing (B.2): first real win + an honest negative  [status: done]

- **Concepts:** [Online warm-start](../concepts/online-warmup.md),
  [Online context mixing](../concepts/context-mixing.md),
  [Prequential evaluation](../concepts/prequential-evaluation.md),
  [Loss per FLOP](../concepts/loss-per-flop-and-scaling-laws.md)
- **Hypothesis:** the context-mixing reference loses to the transformer only from its *cold-start*
  handicap; a FLOP-counted prior warm-start (Phase 1, `warm_mix`) removes it, and gated order
  escalation (Phase 2, `gated_mix`, the A∩C fusion) then beats fixed-order mixing per FLOP.
- **Setup:** REAL enwik8 (first 4 MB; eval = final 32 k bytes, disjoint); `warm_mix` = context-mixing
  + a stateful prior→eval handoff; `gated_mix` escalates orders on a pre-reveal gate. First run off
  the synthetic clone.
- **Result:** `warm_mix` **2.7700 bpb @ 1.03e10** vs the transformer's **5.5453 @ 9.71e11** — strictly
  dominates per FLOP (~94× fewer FLOPs); cold == reference. `gated_mix` is **Pareto-dominated** by
  fixed-order `warm_mix` (gate overhead exceeds the savings on already-cheap mixing).
  [Full note + curves.](B.2-warmed-mixing.md)
- **Verdict:** Phase 1 = the project's **first genuine per-FLOP win** (new bar); Phase 2 = honest
  Pareto-hollow.
- **What we learned:** the *handicap*, not the mechanism, was the reference's weakness; real enwik8
  rewards deep context (synthetic punished it); gating an already-cheap learner doesn't pay; the bar
  is now `warm_mix`, not the transformer.

## 2026-06-19 — Surprise-gated predictive-coding refinement (B.1)  [status: done]

- **Concepts:** [Predictive coding](../concepts/predictive-coding.md),
  [Source-(iv) advantage](../concepts/source-iv-advantage.md),
  [Prequential evaluation](../concepts/prequential-evaluation.md),
  [Loss per FLOP](../concepts/loss-per-flop-and-scaling-laws.md)
- **Hypothesis:** gating predictive-coding *settling depth* by per-byte surprise concentrates
  loss-reducing compute on hard bytes → lower bpb at matched total FLOPs (pure Source-(iv)).
- **Setup:** frozen `transformer` core (2e11 pretrain) + a gradient-free logit-correction PC
  module (variant α); `uniform` vs `surprise` settling differ in *only* the gate (matched mean
  K). Baselines on the same 1200-byte synthetic carve: the bare core and the context-mixing
  reference.
- **Result:** gated **4.2288** vs uniform **4.2333** bpb at identical 2.312e11 FLOPs (−0.0045,
  allocation only); both *worse* than the bare core (**4.1992**) and dominated per-FLOP by
  context-mixing (**4.4637** @ 1.0e7). [Full note + curve.](B.1-surprise-gated-pc-refinement.md)
- **Verdict:** the (iv) *gating lever* is real and directional, but the PC-refinement *mechanism*
  is Pareto-hollow on this data (lost per FLOP).
- **What we learned:** the gate is degenerate until the core is trained enough to be
  *differentially* confident, and order-0 synthetic data starves the lever — a real-enwik8
  control is the needed next test. The A.1 reflex holds: beating the uniform variant ≠ good per
  FLOP.

## 2026-06-18 — Context-mixing reference ceiling  [status: done]

- **Concepts:** [Online context mixing](../concepts/context-mixing.md),
  [Prequential evaluation](../concepts/prequential-evaluation.md),
  [Source-(iv) advantage](../concepts/source-iv-advantage.md),
  [Compression = prediction](../concepts/compression-equals-prediction.md)
- **This is a reference, not a candidate.** It does not "enter" the search; it marks the
  bpb-per-FLOP ceiling that single-pass online learning reaches, so a candidate has a target to
  approach. "Beat-baseline / lost" does not apply to a yardstick.
- **Hypothesis:** a handful of order-k byte models mixed by online-learned logistic weights
  (zero pretraining) should reach far lower bpb than an *untrained* transformer at a tiny
  fraction of the FLOPs — the Hutter-Prize lineage, reproduced at toy scale.
- **Setup:** `model=context_mixing`, prequential / total-FLOP protocol, **zero pretrain**
  (transductive). Corpus: bundled English `sample` (offline), final 800 bytes as the eval
  stream. Sweep `max_order ∈ {0,1,2,3}`; contrast = one untrained (`pretrain_budget=0`)
  `transformer` (d=32, L=2, ctx=64) on the same stream. Seed 0, CPU. All per-byte compute
  charged via the non-matmul `pointwise`/`gather` primitives.
- **Result:** bpb falls as orders (and per-byte FLOPs) rise; the untrained transformer is far
  worse at ~500× the compute. (FLOPs are charged exactly per byte for the branches each step
  runs — see below — so these totals are executed work, not a constant estimate.)

  | run | bpb | total FLOPs |
  | --- | ---: | ---: |
  | context_mixing order 0..3 (reference) | **4.1733** | 7.02e6 |
  | context_mixing order 0..2 (reference) | 4.2284 | 5.74e6 |
  | context_mixing order 0..1 (reference) | 4.3922 | 4.28e6 |
  | context_mixing order 0..0 (reference) | 4.7066 | 2.66e6 |
  | transformer (untrained, contrast) | 7.9786 | 3.48e9 |

  ![Reference bpb-vs-FLOP curve: context mixing (reference, not a candidate) vs an untrained
  transformer](context-mixing-reference.png)

  _Curve: the blue dashed line is the **context-mixing reference (not a candidate)** order
  sweep; the orange point is the untrained transformer contrast. Lower-left is better._
- **Verdict:** reference ceiling established. Single-pass online mixing reaches ≈4.17 bpb for
  <1e7 FLOPs; the untrained transformer needs 3.5e9 FLOPs to manage ≈8.0 bpb.
- **What we learned:** (1) The non-matmul FLOP primitives matter — the reference's per-byte cost
  (order-3 averages ~8.8k FLOPs/byte, steady-state ~9.5k) is real, charged work, not free, and is
  charged *exactly* for the branches each `step` runs (no constant over-estimate). (2) On this
  *real-English* corpus higher orders
  genuinely help (bigram/trigram structure); on the synthetic `text8` clone, whose letters are
  i.i.d. within words, orders >0 add ~nothing — an honest reminder that the curve's shape is a
  property of the corpus, not just the model. (3) A trained transformer would beat this bpb, but
  only by spending orders of magnitude more pretraining FLOPs — which is exactly the per-FLOP
  gap a Source-(iv) candidate is meant to close.

_Pending: the transformer baseline curve and the Phase A fast-weight-memory candidate._
