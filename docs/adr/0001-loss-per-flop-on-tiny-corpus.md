# 0001 — Loss-per-FLOP on a tiny corpus is the only metric

- Status: accepted
- Date: 2026-06-17

## Context

The motivating ambition ("replicate GPT-3 cheaply, get Fable at home") bundles three
different problems: (a) training a GPT-3-class model from scratch, (b) running inference of
one, (c) finding a more compute-efficient *learning mechanism*. We are doing (c) only.

(a) is bounded by scaling-law arithmetic, not by a missing idea: training FLOPs ≈ 6·N·D, so
GPT-3 (N≈175B, D≈300B) is ~3×10²³ FLOPs — centuries on a laptop GPU, and ~TBs of optimizer
state. The claim that "needs billions" is merely psychological ignores the empirical Kaplan
(2020) and Chinchilla (2022) curves. (b) already exists (llama.cpp et al.).

For (c), "find things that are fast" is not a research program until "fast" is a single,
fixed, falsifiable number. "Fast" otherwise splits into sample efficiency, compute
efficiency, parameter efficiency, wall-clock, and inference speed — which trade off against
each other.

## Decision

The north-star metric is **validation bits-per-byte at a fixed training-FLOP budget on a
fixed tiny corpus** (`enwik8`/`text8`). A candidate "wins" only by achieving lower bpb than
the transformer baseline at equal FLOPs. Capability demonstrations are deferred until after
a candidate wins on this metric.

## Consequences

- Every candidate shares one harness and one honest FLOP counter; comparisons are fair
  across machines and frameworks.
- Hundreds of cheap experiments per week become possible — the search has a compass.
- We will sometimes discard ideas that *feel* profound but lose on bpb-per-FLOP. That is the
  point.

## Alternatives considered

- **Sample efficiency (loss per token):** matters only if data, not compute, is the bottleneck. Not our framing.
- **Inference speed / "Fable at home":** already solved by existing runtimes; not a learning-algorithm question.
- **"Can it answer questions?":** not reproducible, not cheap, not falsifiable at tiny scale.
