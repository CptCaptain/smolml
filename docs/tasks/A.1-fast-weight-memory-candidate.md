# Task A.1 — Fast-weight associative memory + slow core (maiden candidate)

- Status: BLOCKED on 0.2 (needs prequential mode)
- Depends on: 0.2
- Branch from: `master` after 0.2 merges

## Context
Read `docs/learning/concepts/fast-weight-memory.md` and `source-iv-advantage.md`. First real
Source-(iv) candidate: make rote memorization ~free so gradient FLOPs buy generalization.

## Design questions to resolve in the PR (grill before/while building)
- Addressing: exact key match vs. soft/modern-Hopfield retrieval.
- Capacity & eviction: fixed-size memory → forgetting policy.
- **Fair FLOP-counting** of memory reads/writes vs. the gradient core (must be honest, see ADR 0004).

## Scope
- A "slow" gradient-trained core + an O(1) fast associative memory (Hebbian/Hopfield write & read),
  combined into a next-byte distribution; memory writes online during eval (FLOPs counted).
- Register it via the model interface; no harness changes.

## Acceptance criteria
- Gates green; offline smoke run.
- **bpb-vs-total-FLOP curve vs. the transformer baseline** on the leaderboard — the headline result.
- `docs/learning/experiments/` entry: hypothesis, setup, curve, verdict (beat/lost), what we learned.

## Deliverable
Own branch + own PR (do not merge); cross-reviewed by a different vendor. First point in the hunt.
