# Online warm-start & the transductive handicap

## Intuition

Two ways to spend FLOPs before predicting the eval stream:

- **Amortized** (the transformer): pour compute into *pretraining* weights on the prior corpus, then
  predict with a frozen model.
- **Transductive** (the context-mixing reference): pretrain *nothing*; learn entirely online, from a
  **cold start**, on the eval stream itself.

[Prequential evaluation](prequential-evaluation.md) (ADR-0004) puts both on one curve — and it
permits a third thing the cold reference forgoes: **online warm-start**. A transductive learner can
stream the freely-usable prior corpus through its *own* online update first (warming its state),
then carry that state into the eval stream. Its FLOPs are counted like any other, but they are
absurdly cheap compared to gradient pretraining.

The **transductive handicap** is what the cold reference suffers by *not* doing this: it begins the
eval stream knowing nothing, paying ~8 bits on byte 1 and climbing the learning curve from scratch.
That handicap — not the mechanism — is why it loses to the transformer. Remove it and the cheap
learner wins (see [B.2](../experiments/B.2-warmed-mixing.md)).

## The math (why warming is nearly free and very effective)

A context model's prediction is a smoothed frequency, `p(b | ctx) = (count[ctx][b] + α) / (Σ + αV)`.
Cold, every `count` is zero, so early predictions are ~uniform. Warming on `N` prior bytes fills the
tables with real frequencies, so the *first* eval byte is already predicted from thousands of prior
observations. The cost is just `N` online updates — for byte-level mixing, ~`O(K·V)` FLOPs each, no
backprop, no matmul-heavy forward/backward. On real enwik8, warming a few hundred-k bytes turns a
3.2-bpb cold start into ~2.8 bpb at a tiny FLOP cost — while a transformer needs **orders of
magnitude** more compute to reach comparable bpb.

## Why it's a legitimate Source-(iv) move

Warm-start is *amortized pretraining for a transductive learner*. It is allowed (ADR-0004 lists
"online warmup" as in-scope; the prior corpus is freely usable) and it is honestly metered (every
warmup byte's FLOPs are on the curve). It qualifies under the [Source-(iv)](source-iv-advantage.md)
filter because the *dynamics* — cheap online frequency estimation — extract far more loss-reduction
per FLOP than gradient descent does at this scale. It is the cleanest demonstration that **how** you
spend FLOPs matters more than **how many**.

## See also

- [Prequential evaluation](prequential-evaluation.md) — the protocol that unifies amortized,
  transductive, and warm-started on one curve.
- [Online context mixing](context-mixing.md) — the learner being warmed.
- [Loss per FLOP](loss-per-flop-and-scaling-laws.md) — the axis warm-start moves so cheaply.
- [Experiment B.2](../experiments/B.2-warmed-mixing.md) — warm_mix beats the transformer per FLOP.
