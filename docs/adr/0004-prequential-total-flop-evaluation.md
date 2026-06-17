# 0004 — Prequential (online) bpb vs. total FLOPs is the evaluation protocol

- Status: accepted (supersedes the evaluation-protocol portion of ADR 0001)
- Date: 2026-06-17

## Context

ADR 0001 fixed the metric as loss-per-FLOP via a train/val split (amortized). We have since
admitted continual/hybrid learning as a goal (in-scope) and required inference/test-time
FLOPs to be counted. A train/val split cannot cleanly score a model that keeps learning,
and forcing an amortized-vs-transductive choice is artificial.

## Decision

Evaluate **prequentially**: the model sees the stream one unit at a time, **predicts each
unit before it is revealed** (incurring log-loss in bits), then **may adapt** on the revealed
unit. Score = cumulative bits / total units = **bpb**, against a **total-FLOP budget** =
pretraining + online-update + prediction FLOPs. Report bpb as a curve over a small set of
fixed total-FLOP budgets.

This is the prequential / online-MDL principle. The metric *spirit* from ADR 0001
(loss-per-FLOP on tiny enwik8) is unchanged; only the protocol sharpens.

## Why

- **Honest with no held-out split:** one-step-ahead prediction means adapting online cannot
  leak the future, so generalization is measured for free.
- **Unifies the spectrum:** amortized (all FLOPs up front, then frozen), transductive (zero
  pretraining), and hybrid are all points on one curve, compared fairly. The winner reveals
  the best mix rather than us choosing it.
- **Continual learning + inference FLOPs come for free:** test-time adaptation is just more
  FLOPs on the same budget; it cannot hide compute at eval.
- **It is compression:** cumulative log-loss = compressed length, so this directly chases the
  context-mixing (Hutter) ceiling, now generalized to allow counted pretraining.

## Consequences

- The first harness (train/val split) remains valid foundation; a follow-up task adds the
  prequential/online eval mode and inference-FLOP accounting on top.
- Open sub-decision: how to carve enwik8 into a freely-usable prior corpus vs. the fixed
  evaluation stream (see CONTEXT.md / candidates.md).
