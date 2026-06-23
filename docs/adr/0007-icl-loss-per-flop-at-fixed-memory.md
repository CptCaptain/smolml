# 0007 — In-context-learning loss-per-FLOP at fixed memory (amends 0001/0004)

- Status: accepted
- Date: 2026-06-23
- Amends: 0001 (the metric's *operationalization*), 0004 (the eval protocol). Preserves 0001's
  *intent* and 0002 (PyTorch/uv) and 0003 (the Source-(iv) filter, re-pointed below).

## Context

ADR 0001 made the north star **validation bits-per-byte at a fixed training-FLOP budget on tiny
`enwik8`**, as a cheap, falsifiable proxy for "find a learning mechanism that learns more per FLOP."
The Space-A/B search (0.x, A.1, B.1–B.4) then *worked* — and exposed that the proxy decoupled from
the intent:

- At tiny scale, on one corpus, **minimizing bpb ≡ memorizing n-gram statistics.** The global
  optimum is the best *compressor* (PAQ/cmix), which is a lookup machine, not a mind.
- Every "win" was a better lookup table: count tables, fixed-memory hashing (B.3), and B.4's
  delta-rule fast weights — which beat the bar by being a *more general lookup* (a distributed hashed
  key interpolating between memorized n-grams) on top of a **268 MB** count store. Memory was
  unbounded and was the real carrier of capability; the "learning" was incidental.
- bpb-per-FLOP on tiny enwik8 therefore measures "compress Wikipedia bytes per FLOP," not "acquire
  generalizable capability per FLOP." A winner on it (the cmix frontier) is a known dead end for the
  stated goal: *a mechanism feasibly useful as a chatbot/agent.*

ADR 0001 had rejected "can it answer questions?" as *not reproducible, not cheap, not falsifiable at
tiny scale* — correctly, for open-ended capability evals. The resolution is a capability proxy that
**is** cheap and falsifiable yet **cannot be won by memorization**: in-context learning of fresh
synthetic tasks at bounded memory.

## Decision

The north-star metric becomes **held-out in-context-learning (ICL) loss per training FLOP, at a
fixed parameter/memory budget.**

- **Train** a mechanism on a *distribution of tasks*; **score** its few-shot loss on **held-out
  tasks** (rules disjoint from training) measured **only on the query/answer positions**, versus
  total training FLOPs, with **parameters/memory capped**. Lower held-out ICL loss at equal
  `(FLOPs, memory)` wins.
- The eval is a **graded synthetic suite** (recall → rule-induction → composition), scored per rung
  so we see *where* a mechanism's ICL breaks, not just a scalar.
- The **transformer is the honest baseline** (attention is built for ICL via induction heads), not a
  strawman — beating transformer-ICL-per-FLOP-at-fixed-memory is the real question.

This keeps everything that made 0001 good — one shared harness, one honest FLOP counter (0002),
cheap falsifiable numbers — while fixing what it measured. Three properties hold by construction:
**memorization is dead** (tasks are fresh and the eval rules are held out), **memory is bounded**
(no lookup-table wins), and the score **tracks the chatbot/agent capability** (learn a new task from
examples — the substrate of instruction-following, few-shot, and tool-use).

**Source-(iv) filter (0003) carries over, re-pointed:** a non-backprop candidate qualifies only if
its *learning dynamics* extract more **held-out-ICL-loss reduction per FLOP** (not bpb). "Avoids
backprop" / "is parallel" still do not qualify.

The Space-A/B enwik8 work is **retained in the compendium** as the lesson on the memorization failure
mode (B.4's `delta_mix` notably *is* a fast-weight programmer — a real ICL mechanism measured on the
wrong task; it re-enters Space C as a candidate on honest ground).

## Consequences

- New eval infrastructure: ICL task generators + a query-masked scorer + an ICL leaderboard
  (Phase 0, `docs/tasks/C.0-icl-harness.md`). The model seam (`LanguageModel.forward`), the FLOP
  counter, and the transformer baseline are reused unchanged.
- The candidate space shifts to **ICL mechanisms (Space C):** softmax attention (baseline), linear
  attention / DeltaNet fast-weights, state-space models, RWKV, test-time-training / meta-plasticity,
  hybrids — judged on held-out ICL loss per FLOP at fixed params.
- We will discard mechanisms that *compress* well but do not *generalize in-context*. That is now the
  point.

## Alternatives considered

- **Keep bpb, just cap memory.** Bounded-memory bpb still rewards in-distribution memorization (a
  small model that overfits the corpus tail); it does not require *learning a new task*. Necessary
  (we keep the memory cap) but not sufficient — hence the task-distribution + held-out-rule design.
- **Generalization gap on held-out text.** Cheap to bolt on, but a small gap does not imply
  agent-usefulness and it is still next-token text. Weaker proxy than ICL.
- **Real downstream/agent evals at tiny scale.** Not reproducible or cheap (0001's original, valid
  objection). The synthetic graded suite is the cheap, falsifiable stand-in.
