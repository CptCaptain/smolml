# Task C.A.2 — chemotaxis-minimal (minimal-organism control candidate)

- Status: SPEC — design approved in brainstorm (purest source-(iv) bet: a hand-structured
  run-and-tumble controller with a few learnable scalars, adapting almost entirely **online in
  `step`**, ≈ zero distillation).
- Depends on: the **C.A.0 control rung** (merged to `main`): `smolml/envs/chemotaxis.py`,
  `smolml/control_eval.py::evaluate_control`, `smolml/control_train.py::distill_train_run`,
  `smolml/leaderboard.py::regenerate_control`. **Zero rung changes** — a candidate is just a
  registered `LanguageModel` that also implements the per-step seam.
- Branch: `task/C.A.2-chemotaxis-minimal` off `main` (worktree
  `../smolml.worktrees/C.A.2-chemotaxis-minimal`). Own PR; do not merge (human merges).

## Why (ADR 0007 + ADR 0003)

A bacterium does chemotaxis with almost no machinery: integrate the sensed concentration, ask "am I
improving?", and bias the next move accordingly. The **in-context adaptation is the integrator state**
(a leaky baseline tracking the gradient), not weight learning — so the mechanism costs ≈0
distillation and a few pointwise ops per `step`. Its **total FLOPs is dominated by the (cheap) eval
rollout**, counted honestly through `step` (ADR 0004). If it competes with a distilled transformer on
**regret per total FLOP**, that is the cleanest possible source-(iv) result; if it cannot, that is a
documented finding. This is the FLOP-floor reference for the control rung.

## The bar to beat (transformer baseline, held-out drifting eval)

regret **0.229 → 0.171 → 0.141** at total FLOPs **2.97e11 → 1.19e12 → 2.96e12**, at **148,608
params**. Headline = **regret vs oracle per total FLOP at fixed params** (lower-left wins). This
candidate's bet is the **FLOP axis**: tiny params, tiny total FLOPs.

## The mechanism — `@register_model("chemotaxis_min")` (`smolml/models/chemotaxis_min.py`)

State (in `DecodeState.cache`): leaky concentration baseline `b`, the `last_action`, the last sensed
level. A **handful of learnable scalars** (`nn.Parameter`): leak `λ` (via a sigmoid of a logit),
policy gain `g`, `stay_bias`, and a small world-model term. Every `step` emits **full-vocab logits**
(both the `conc_slice` world-model head and the `action_slice` policy head — the scorer reads the
slice for the position's parity).

- **Policy (action positions).** After folding the just-sensed level `c_t`, surprise `s = c_t − b`
  (rising?); update `b ← (1−λ)·b + λ·c_t`. Action logits: `keep` = `g·s` on `last_action`, `reverse`
  = `−g·s` on the opposite direction, `STAY` = `stay_bias`. (Run-and-tumble: keep climbing while
  improving, tumble on a drop.) This is differentiable in `λ, g, stay_bias` so a short distillation
  can tune them; the dominant adaptation is `b` updating each step (no weight change).
- **World model (action positions → predict next concentration).** A minimal learnable predictor over
  levels (e.g. peak at the current level shifted by a learnable climb term, learnable sharpness). It
  is the secondary diagnostic; keep it small and honest (charge its FLOPs).
- `forward(idx)`: vectorize the recurrence over `T` (sequential in `b`/`last_action`) to produce
  per-position full-vocab logits, so `distill_train_run` can next-token-train the few scalars.
- `init_prequential_state` / `step`: fold the token, update `b`/`last_action`, emit next logits;
  the integrator update is the in-context adaptation. ≈ zero distillation: run with a tiny FLOP budget
  (0–few train steps); total FLOPs ≈ the eval rollout.

## FLOP accounting (honest, via `smolml.flops` — the product)

All compute is **pointwise** (no matmuls dominate), so it MUST be charged via `pointwise_flops` —
NOT omitted (the flops module's conditional-omission rule: a non-matmul mechanism charges its real
work or the instrument scores it as free). `flops(seq_len)`: forward = `seq_len ×` (the per-step
pointwise op count: baseline update, surprise, the action-logit assembly, the world-model softmax
terms); backward = the standard 2× for the few distilled scalars (or 0 when run with no training).
`step` returns the same per-step forward (and `backward = 0` — no online weight update; the adaptation
is integrator state, which is forward compute). `decode_step_flops(context_len)` = the per-step
forward (context-independent). `configure_optimizer`: AdamW over the few scalars. `step_flops` must be
`> 0` (it is — pointwise) so `distill_train_run`'s guard passes even at a 0-step budget.

## Files

- `smolml/models/chemotaxis_min.py` — `ChemoMinConfig`, `@register_model("chemotaxis_min")`
  `ChemotaxisMin`.
- `smolml/models/__init__.py` — import + register.
- `smolml/experiments/chemotaxis_min_control.py` — driver mirroring `control_baseline.py`: build the
  model, eval held-out at ≈0 distillation (and a tiny sweep), write a `runs/control/*.jsonl` row,
  print the comparison to the bar (emphasize total FLOPs).
- `tests/test_chemotaxis_min.py` — acceptance tests below.
- `docs/harness.md` — a short note under §6 that `chemotaxis_min` is a registered control candidate.

## Acceptance criteria

- Gates green: `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest` (paste outputs).
- `tests/test_chemotaxis_min.py`:
  - **forward shape & full-vocab logits**: `forward((B,T))` → `(B,T,vocab)`; both slices populated.
  - **run-and-tumble semantics**: with `g>0`, on a rising concentration the policy favors
    `last_action`; on a drop it favors the reversed direction (assert argmax on the action slice).
  - **integrator is the memory**: changing recent sensed concentrations changes the action
    distribution (the leaky baseline carries history) — in-context, no weight change.
  - **step ≡ forward**: summed `step` last-position logits match `forward` on the same token sequence;
    summed `step` FLOPs match `T · decode_step_flops`.
  - **FLOP honesty**: `flops(T).forward > 0` and equals the hand-derived pointwise op count (the
    mechanism is NOT charged as free); `step` returns nonzero forward FLOPs.
  - **end-to-end smoke**: `evaluate_control` on held-out envs (≈0 distillation) returns finite **mean
    reward strictly above the random-policy floor** and `second_half_reward > first_half_reward`; a
    `runs/control` row is written with a **total FLOPs far below** the transformer bar, and
    `regenerate_control` ranks it next to the bar.

## Out of scope

- The reservoir candidate (C.A.1 — its own task/PR).
- Any change to `ChemoEnv`, `evaluate_control`, `distill_train_run`, the leaderboard, or the renderer.
- 2-D navigation, multi-peak/adversarial drift, a separate reward channel (C.A.0's documented options).
