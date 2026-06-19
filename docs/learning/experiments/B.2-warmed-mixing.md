# Experiment B.2 — warmed online mixing (the first real win) + gated escalation (honest negative)

**Status:** the project's **first genuine per-FLOP win**, on the **real enwik8 corpus**. Two phases:
**`warm_mix`** (Phase 1) strictly dominates the transformer per FLOP; **`gated_mix`** (Phase 2 —
the bold A∩C fusion) is honestly **Pareto-hollow** against `warm_mix`. This is also our first move
off the synthetic CI clone onto real text (ADR-0004 enwik8 carve), at a bounded-but-real scale.

## Where it came from

The B.1 design fan-out's adversarial critics noticed (and a verification probe confirmed) that the
[context-mixing reference](context-mixing-reference.md) loses to the transformer **only because it
cold-starts** — it is purely transductive and throws away everything it could learn from the
freely-usable prior corpus. That is a *handicap*, not a property of the mechanism. Remove it cheaply
and the cheap learner wins.

## Phase 1 — `warm_mix`: remove the transductive handicap

`warm_mix` is the context-mixer with **one** new idea: a **stateful prior→eval handoff**. During
pretraining it folds the prior corpus into its count tables (warmup, **every FLOP counted**), then
**deep-copies** that warmed state into each eval stream so prediction starts from warmed statistics
instead of cold uniform priors. At warmup-budget 0 it is **bit-identical** to the cold reference (a
clean baseline). No leakage: warmup sees only the carve-disjoint prior; each eval folds into its own
copy.

**Real enwik8** (first 4 MB; eval = final 32,768 bytes, disjoint; all FLOPs counted):

| run | bpb | total FLOPs |
| --- | ---: | ---: |
| transformer (anchor) | 5.5453 | 9.71e11 |
| warm_mix cold (= reference) | 3.2106 | 3.05e8 |
| warm_mix warmed @1e9 | 2.8805 | 1.30e9 |
| **warm_mix warmed @1e10** | **2.7700** | **1.03e10** |

`warm_mix` **strictly dominates the transformer**: lower bpb at **~94× fewer total FLOPs**, with
warmup monotonic (−0.44 bpb cold→warm). Honest caveats: the tiny transformer is badly undertrained
at this budget (real enwik8 ≫ the synthetic clone) and its windowed-recompute *eval alone* is 9.5e11
(≈3000× warm_mix's eval); at much larger scale it would eventually win absolute bpb, but **per FLOP
at every tractable budget, warmed mixing wins**. This is the Hutter/PAQ thesis confirmed on real
text: a transformer is a FLOP-inefficient *learner* at tiny scale.

## The order curve (what motivated Phase 2)

On real enwik8, deeper context keeps paying — unlike the order-0 synthetic clone where it *hurt*:

| max_order (warmed @1e9) | 2 | 3 | 4 | 5 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bpb | 3.248 | 2.881 | 2.710 | 2.667 | **2.655** | 2.664 |

But the fixed-order mixer evaluates **all** orders on **every** byte, while the high orders earn
bits only on the minority with deep recurring context. That looked like a Source-(iv) lever.

## Phase 2 — `gated_mix`: the A∩C fusion (gated order escalation)

`gated_mix` holds orders `0..K_max` (warmed) but predicts by escalating cheapest-first and
**stopping** once a **pre-reveal `1 − max p` gate** says the partial mix is confident enough —
charging FLOPs for only the orders evaluated. This is simultaneously **A** (the expensive
high-order compute fires only on high-surprise bytes) and **C** (an order is "grown into" the mix
only where it pays). At `min_order == max_order` it is bit-identical to fixed-order `warm_mix`.

**Verdict — Pareto-dominated** (real enwik8, warmed @1e9, vs the fixed-order curve):

| entrant | bpb | total FLOPs | eval FLOPs |
| --- | ---: | ---: | ---: |
| fixed warm_mix order-6 | **2.6552** | 1.477e9 | 4.87e8 |
| gated_mix thr=0.1 (near-full) | 2.6698 | 1.570e9 | 6.11e8 |
| gated_mix thr=0.5 | 2.8274 | 1.463e9 | 4.90e8 |
| gated_mix thr=0.7 (aggressive) | 3.1179 | 1.335e9 | 3.75e8 |

Every gated point is dominated by a fixed-order point (≤ bpb **and** ≤ FLOPs). The mechanism is
clear: the gate **recomputes a confidence softmax at each escalation level** (`O(depth·V)`
overhead), but the underlying mix is *already* a single cheap `O(K·V)` matmul — so escalating deep
costs **more** than fixed (thr=0.1: worse bpb *and* more FLOPs), and gating hard degrades bpb faster
than FLOPs drop (the aggressive points lose even with a hypothetically free gate). The B.1 critics
predicted exactly this: the mixer's online weights already do soft marginal-loss allocation, so hard
gating cannot beat it — and the gate is not free.

## FLOP honesty

Both candidates charge every op via `smolml.flops` and return it from `step`/`train_step` (warmup
folds, mixing, Laplace, mixer-SGD, and — for `gated_mix` — the per-escalation gate arithmetic). A
cross-vendor review caught and fixed an uncharged warmup-CE recomputation in `warm_mix`; it verified
`gated_mix`'s per-byte charge equals exactly the evaluated orders, the budget guard is a safe upper
bound, the gate is pre-reveal, and `min_order==max_order` is bit-identical to `warm_mix`.

## Verdict

**`warm_mix` is the first real per-FLOP win** and the new bar (the transformer is decisively beaten;
the cold reference is strictly extended). **`gated_mix` is an honest Pareto-hollow** against it —
**online mixing is already near its per-FLOP frontier, and adaptive-depth gating doesn't pay because
the thing being gated is already cheap and the gate is not free.**

## What we learned

1. **The handicap, not the mechanism.** The context-mixing reference lost to the transformer purely
   from its cold-start; a cheap, FLOP-counted warm-start flips it to a decisive per-FLOP win. The
   cheap online learner was never the problem.
2. **The corpus matters.** Real enwik8 rewards deep context (to ~order-6); the order-0 synthetic
   clone punished it. Levers that need genuine difficulty/structure (B.1's gate, B.2's escalation)
   can only be judged on real text — which is why we moved here.
3. **Gating a cheap thing rarely pays.** When the base computation is already a single cheap matmul
   with learned soft allocation, a per-byte gate adds overhead comparable to what it saves. Gating
   wins only when the gated work is *expensive* relative to the gate.
4. **The bar moved.** Future candidates must beat `warm_mix` (≈2.66 bpb @ ~1.5e9 on this carve), not
   the transformer. That likely needs a mechanism capturing structure the mixer **cannot** (long-
   range / positional / semantic) at genuinely low FLOP — not a re-allocation of the mixer's own
   already-efficient compute.

## Scale & reproduce

Bounded-but-real (pure-Python mixing is ~53 µs/byte, so the full 95 MB/5 MB ADR carve stays
GPU/opt-in): 4 MB real-enwik8 slice, 32 k eval, warmup ≤ ~1 M bytes. Real text, prior/eval disjoint,
all FLOPs counted — not the full-5 MB endpoint, but a faithful real-text per-FLOP ordering.

    uv run python -m smolml.experiments.warm_mix_enwik8     # Phase 1: warm_mix vs transformer
    uv run python -m smolml.experiments.gated_mix_enwik8    # Phase 2: gated_mix vs fixed-order
