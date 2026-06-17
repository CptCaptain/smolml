# 0002 — PyTorch + MPS for the search; defer C/CUDA

- Status: accepted
- Date: 2026-06-17

## Context

The motivating post demands hand-written C/CUDA (or Bend) and "no Python, zero excuses."
But the chosen metric (ADR 0001) is **loss-per-FLOP**, and FLOPs are language-agnostic: a
matmul costs the same FLOPs from Python or from a hand kernel. Implementation language moves
*wall-clock* and *iteration speed*, not the scoreboard.

The project is a search, whose throughput is ideas-tested-per-day. Hand-writing a kernel per
hunch slows iteration 10–100×, shrinking idea coverage — fatal for a scout.

Primary dev machine is an Apple M4 Max. Framework options there:
- JAX: weak on Apple Silicon (`jax-metal` is experimental, partial op coverage, version lag).
- MLX: fast and Apple-native, but Mac-only — a winner would need porting to scale.
- PyTorch + MPS: mature Metal backend, and the *only* option that runs on all our targets
  (Mac MPS → AMD ROCm → cloud CUDA) from one codebase.

## Decision

Use **PyTorch with the MPS backend** for the entire search phase. Write a shared, honest
FLOP counter so the metric is fair. **Defer all hand-written C/CUDA/Bend** until *after* a
candidate wins the loss-per-FLOP leaderboard and we deliberately choose to scale it up.

## Consequences

- "Prove local → scale up" needs zero framework migration.
- Iteration speed stays high; idea coverage stays wide.
- We knowingly leave wall-clock performance on the table during the search. Accepted — it
  does not affect the metric.

## Alternatives considered

- **MLX:** kept in reserve if local iteration speed becomes the binding constraint.
- **JAX:** strong on cloud TPU/CUDA, but rough on the M4 Max *now*; reconsider only if the project moves primarily to cloud.
- **Obey the no-Python mandate:** rejected — optimizes an axis (kernel wall-clock) that does not move the metric, at the cost of the axis (iteration count) that decides success.
