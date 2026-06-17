# PRD — smolml

- Status: living document
- Owner: nils (human) · orchestrated by polly
- Related: `README.md`, `AGENTS.md`, `docs/plan.md`, `docs/adr/*`, `docs/candidates.md`

## Problem

"Replicate GPT-3 on a laptop" conflates three problems. Training a GPT-3-class model from
scratch is bounded by scaling-law arithmetic (~3e23 FLOPs), not a missing idea; running
inference already exists. The *real* open question is whether a **learning algorithm exists
that extracts more capability per FLOP** than today's transformer-on-a-cluster recipe.

## Goal / hypothesis

Find a mechanism that achieves **lower bits-per-byte at an equal total-FLOP budget** than a
clean transformer baseline, on a tiny corpus — and ideally one that is a **reusable, continually
adapting** model, not a static blob.

## Non-goals

- Training or matching GPT-3 / any large model.
- Hand-written C/CUDA/Bend during the search (wall-clock is not the metric).
- Optimizing parallelism/locality (Source-(ii); out of scope under the metric).
- Capability demos ("it chats") before a candidate wins on the metric.

## Success metric

Prequential **bits-per-byte vs. total FLOPs** on byte-level enwik8 (final 5 MB = fixed eval
stream; first ~95 MB = free prior corpus). All FLOPs counted (pretrain + inference + online
adaptation). Reported as a curve over several budgets. See ADR 0001 + ADR 0004.

## Success criteria

1. A trustworthy, reproducible measuring spine (honest FLOP counter, prequential eval, leaderboard).
2. A transformer baseline curve, and ≥1 Space-A calibration point matching known results
   (proves the instrument).
3. At least one Source-(iv) candidate evaluated honestly against baseline on the curve —
   win or lose, logged in `docs/learning/experiments/`.

## Scope (phases)

See `docs/plan.md`. Phase 0 = measuring spine; Phase A = fast-weight-memory candidate; Phase B
= weirder (iv) backlog. Each numbered item is a ticket in `docs/tasks/`.

## Key risks / assumptions

- **Instrument risk:** a FLOP-counter or eval-leak bug makes all comparisons meaningless →
  mitigated by Space-A calibration against known numbers before trusting Space-B results.
- **Strong baseline:** transformers are brutally good per-FLOP at tiny scale; beating them is
  real work, and many Space-B ideas will lose. That's expected; failures are logged.
- **Metric honesty:** anything done at inference (adaptation) must be FLOP-counted or it games
  the score.

## Definition of done (per candidate)

A merged-by-human PR with: the mechanism behind the shared interface, a bpb-vs-FLOP curve vs.
baseline, passing gates, a `docs/learning/` concept page + experiment entry, and a cross-vendor
review approval.
