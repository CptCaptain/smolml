# smolml

A scout project: **search for learning algorithms with better loss-per-FLOP than the transformer baseline, at tiny scale.**

## What this is (and is not)

This is **not** an attempt to "replicate GPT-3 on a laptop" by training a 175B-parameter
model from scratch — that runs into scaling-law arithmetic (~3×10²³ training FLOPs for
GPT-3), not into a missing idea. See [`docs/adr/0001`](docs/adr/0001-loss-per-flop-on-tiny-corpus.md).

This **is** the legitimate research question hiding inside that ambition: *are there
mechanisms capable of learning that are fundamentally more compute-efficient than what we
use today?* We answer it the only way a small team can — by building a fast, falsifiable
local leaderboard and scouting the idea space against it.

## The one metric

**Validation bits-per-byte (bpb) at a fixed FLOP budget, on a fixed tiny corpus.**

- Corpus: `enwik8` / `text8` (byte- or char-level; no tokenizer politics).
- Budget: a small, fixed training-FLOP budget (minutes–hours on one machine).
- "Faster" means **lower bpb at equal FLOPs.** Nothing else counts as a win — not
  wall-clock, not param count, not "it answered a question." Those come later, only after
  something beats the transformer baseline on this curve.

Why loss-per-FLOP and not the alternatives: see [`docs/adr/0001`](docs/adr/0001-loss-per-flop-on-tiny-corpus.md).

## Method

1. Build an honest **transformer baseline** and a **FLOP counter** shared by all candidates.
2. Implement each candidate mechanism behind the same training/eval harness.
3. Plot bpb-vs-FLOPs for each. Keep what beats baseline; discard what doesn't.
4. Only *then* consider scaling a winner up (and only *then* consider hand-written kernels).

## Stack

- **PyTorch + MPS** on the M4 Max for the search (unified memory is the right shape for tiny-scale work).
- Portability is deliberate: the same PyTorch code runs on Mac (MPS) → AMD (ROCm) → cloud (CUDA) with no rewrite when it's time to scale. See [`docs/adr/0002`](docs/adr/0002-pytorch-mps-defer-cuda.md).
- **No hand-written C/CUDA/Bend during the search.** At a fixed FLOP budget, kernel language doesn't move the metric — it only moves wall-clock and iteration speed, and iteration speed is the scout's lifeblood.

## Layout

- `README.md` — this file.
- `CONTEXT.md` — glossary of the project's load-bearing terms.
- `docs/adr/` — the decisions we'll regret not writing down.
- `docs/candidates.md` — the running list of mechanisms to scout, with status.
