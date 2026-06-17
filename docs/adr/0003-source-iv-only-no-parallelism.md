# 0003 — A Space-B candidate qualifies only via Source (iv); parallelism is out of scope

- Status: accepted
- Date: 2026-06-17

## Context

Space-B candidates (alternatives to backprop) are usually justified by one of four sources
of "advantage". Under our metric (ADR 0001, loss-per-FLOP), they are not equal:

- (i) Cheaper credit assignment — fewer FLOPs/update. Backprop is already ~2× the forward
  pass, so the ceiling is ~3×, and methods like forward-forward (two forwards) or evolution
  strategies (many forwards) usually spend it back. Modest.
- (ii) Locality → parallelism / async — buys **wall-clock and scalability**, scores **zero**
  on a fixed-FLOP metric.
- (iii) No activation storage — buys **memory**, barely touches FLOPs.
- (iv) The learning **dynamics** extract more loss-reduction per FLOP — independent of the
  backward-pass question. The only source that actually moves our scoreboard.

## Decision

A Space-B candidate **qualifies only if its advantage is Source (iv)**: a plausible reason
the update reduces loss *faster per FLOP*. Candidates whose only story is (ii)
parallelism/locality are **parked** under the current metric, not pursued. "Avoids the
backward pass" is, by itself, not a qualifying reason.

Screening question for every candidate: *"Does this plausibly reduce loss faster per FLOP —
or is it just avoiding a backward pass that was already cheap, or buying parallelism our
metric does not reward?"*

## Consequences

- Many famous backprop-free methods (forward-forward, equilibrium propagation, feedback
  alignment) are parked unless a Source-(iv) story is articulated for them.
- The hunt aims squarely at learning *dynamics* that are more compute-efficient, which is
  the rarest and most interesting region.
- If we ever decide wall-clock/parallelism matters, that is a deliberate revision of ADR
  0001 (the metric), not a quiet exception here.
