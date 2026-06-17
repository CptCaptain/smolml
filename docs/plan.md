# Plan & roadmap

The measurement spine is locked (see `AGENTS.md` + ADRs). Work proceeds in phases; each numbered
item is a scoped task → its own PR → cross-vendor review → human merge.

## Phase 0 — the measuring & tracking spine

0.1 **Baseline harness** *(in progress)* — data pipeline, honest FLOP counter, train loop to a
    fixed FLOP budget, bpb eval, model registry, transformer baseline, leaderboard. Amortized
    (train/val) to start.
0.2 **Prequential eval mode** — add the online one-step-ahead protocol + inference/adaptation
    FLOP accounting and the enwik8 data carve (final 5 MB eval stream). (ADR 0004)
0.3 **Context-mixing reference ceiling** *(optional, cheap)* — a small online context-mixer as a
    **non-candidate** yardstick: tells us the bpb-per-FLOP that single-pass online learning can
    reach, i.e. the target to approach.

## Phase A — maiden candidate: fast-weight associative memory + slow core

A.1 A gradient-trained "slow" core (for generalization) + an O(1) Hebbian/Hopfield-style
    "fast" memory (for rote memorization via instant writes, no gradient). A hybrid + genuine
    continual learner; Source-(iv) story = make memorization ~free so gradient FLOPs buy
    generalization. First real point on the leaderboard.

## Phase B — "more out there" methods (backlog, later)

Deliberately weirder Source-(iv) ideas, to be sharpened before building. Seed list lives in
`docs/candidates.md`. Examples to grill later: learning rules whose *update* is information-
theoretically targeted, growth/morphing driven by prediction error, predictor mixtures that
amortize into a reusable model, etc. Each must pass the Source-(iv) screen before it earns a
GPU-hour.

## Always, every phase

- Grow `docs/learning/` (concepts + experiments) alongside the code — see the standing directive
  in `AGENTS.md`.
- Report results as a **bpb-vs-total-FLOP curve**, never a single cherry-picked point.
- Discard candidates that lose to baseline on the curve. Failures get logged in
  `docs/learning/experiments/` — they are data, not waste.
