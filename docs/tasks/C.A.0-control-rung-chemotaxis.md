# Task C.A.0 — the control rung (in-context-RL measuring spine + chemotaxis env)

- Status: SPEC — design approved in brainstorm (drifting-gradient env "B" + minimal-organism
  candidates "C"). Phase 0 of the **action** sub-track of Space C; no candidate yet — this builds
  the interactive eval, the metric, and the transformer baseline.
- Depends on: nothing new (reuses the model seam, FLOP counter, transformer, the `step` decode
  channel, the FLOP-budgeted train loop). Sibling of `C.0-icl-harness.md`.
- Branch: `task/C.A.0-control-rung` off `main`. Own PR; do not merge (human merges).

## Why (ADR 0007, extended)

ADR 0007's graded suite is *recall → rule-induction → composition*. This task adds the top rung —
*… → **control*** — an **interactive** rung: the model must **act in an environment and learn from
the consequences in-context**, not just predict a static scored sequence. Same philosophy (score
per rung, see where a mechanism breaks); new eval mode (autoregressive rollout vs a live env).

Motivation (the bio thesis): even minimal organisms *learn, predict, and act* with tiny machinery.
A transformer does in-context control by recomputing `O(window·d)` attention every step; a
minimal-organism learner (fixed reservoir + plastic readout; or run-and-tumble chemotaxis) adapts
online at `O(d)`/step. This rung is where that structural FLOP edge can show. The rung scores **two**
things at once: how well the model **acts** (reward/regret) and how well it **predicts its
environment** (world-model bits) — both halves of the steer.

The candidates that get scored here are later PRs: **C.A.1** (reservoir + plastic readout) and
**C.A.2** (chemotaxis-minimal). This task is **infrastructure + baseline, not a candidate** (0.1's
order for the new rung).

## The environment — `ChemoEnv` (`smolml/envs/chemotaxis.py`)

A 1-D ring of `W` cells (default 16). A unimodal concentration bump peaks at `μ_t`:
`conc(x) = exp(−ringdist(x, μ_t)² / 2σ²)`, `σ` ≈ 2–3 cells, **quantized to `L` levels** (default 8)
→ a concentration symbol in `[0, L)`. The agent holds a position `p_t` and **senses only**
`conc(p_t)` (local, like a bacterium). Actions are `{LEFT(−1), STAY(0), RIGHT(+1)}`.

- **Drift (non-stationarity).** `μ_{t+1} = μ_t + drift_t`, where the per-episode drift
  speed/direction is drawn from a **held-out pool**: train and eval draw drift parameters (and
  seeds) from **disjoint** sets, so eval episodes are unseen dynamics, not memorized ones. Drift
  forces *continual* in-context adaptation.
- **obs ≡ reward ≡ sensed concentration.** The only feedback is the concentration the agent now
  smells; reward at step `t` = `conc(p_{t+1})` (normalized to `[0, 1]`). Pure chemotaxis: no
  separate reward channel, no global state revealed.
- **Determinism.** A `ChemoEnv(seed)` is fully reproducible: `reset() → c0`; `step(action) →
  (c_next, reward)`. Episode horizon `H` (default 64).
- **References to bound the metric:** `oracle_policy` (observes `μ_t`, moves to maximize sensed
  concentration → ≈ max reward, regret ≈ 0); `random_policy` (uniform action → the floor);
  `run_and_tumble_policy` (keep direction if concentration rose else reverse — a scripted adaptive
  reference between the two).

## Generalization — the `Environment` seam

`ChemoEnv` implements a minimal `Environment` protocol (typing-only, no runtime cost):
`reset(seed) -> obs`, `step(action) -> (obs, reward)`, `oracle_action(state) -> action`, plus
metadata (action set, obs/vocab sizes). `evaluate_control`, the distillation training, the renderer,
and **every candidate depend only on this protocol** — so they are environment-agnostic. Chemotaxis
is the minimal instance of "online optimization of a feedback signal under non-stationarity from
local information"; other feedback tasks (non-stationary bandit, homeostatic setpoint control,
derivative-free objective) become **sibling rungs** that reuse the entire spine and all candidates
with zero scorer/candidate changes. C.A.0 builds **only `ChemoEnv`**; the seam makes the rest free.
The deeper arc (out of scope here): reframing next-token surprise as the feedback signal lets the
*same* learner span the static rungs and control — chemotaxis in belief space = free-energy
minimization.

## Tape format, vocab, and the policy

Symbol vocab = `L` concentration symbols `[0, L)` + 3 action symbols (`LEFT/STAY/RIGHT`) → the model
`vocab_size = L + 3`; `max_seq_len ≥ 2·H + 1`. The episode tape interleaves sensed concentration and
moves:

```
c0 · a0 · c1 · a1 · c2 · a2 · …          (cᵢ ∈ concentration symbols, aᵢ ∈ action symbols)
```

The model's **policy** at a post-concentration position = `softmax(logits[ACTION_SLICE])` (the action
sub-vocab). Its **world model** at a post-action position = `softmax(logits[CONC_SLICE])` (predicting
the next sensed concentration). A position-parity convention pins which slice is scored where; a test
asserts action and concentration sub-vocabs are disjoint and that scoring uses the right slice.

## The rollout scorer — `smolml/control_eval.py` (mirrors `eval.py`/`icl_eval.py`)

`evaluate_control(model, *, split="eval", n_episodes, horizon, seed, device, greedy=False, record=False) ->
ControlResult`. For each held-out episode, run an **autoregressive rollout reusing the existing
`model.step` channel** (the one `prequential_bpb` uses): `step(state, tok, pos) → (state, logits,
all_flops)`. The only change from prequential scoring: at **action positions** the next token is
**sampled (or argmax if `greedy`) from `logits[ACTION_SLICE]`** (the policy) instead of read from a
fixed stream; at **concentration positions** the env supplies the token *after* applying the sampled
action, and we score **world-model bits** on `logits[CONC_SLICE]` via `cross_entropy_bits`/`score_bits`.

`ControlResult` aggregates over the seeded held-out episode set: mean reward, **regret vs the
oracle** (`oracle_reward − agent_reward`, normalized), world-model bits, and the `FlopBreakdown` of
the whole rollout. Deterministic given `seed`. **Every rollout FLOP is counted** (folding, online
adaptation, prediction) and is part of the **total-FLOP** budget (ADR 0004) — eval compute is never
free here. Bounded memory: the tape is windowed by `context_window` exactly as in `step`.

With `record=True` the returned `ControlResult` also carries a per-step `Trajectory` (peak `μ_t`,
agent `p_t`, sensed concentration, action, reward, predicted-concentration distribution) — consumed
by the renderer and by the determinism tests.

## Training the baseline — Algorithm Distillation

A transformer can't do in-context control untrained. Baseline training = **Algorithm Distillation**
(Laskin 2022): roll an **adaptive source policy whose return improves within an episode** across many
**fresh** envs, logging trajectory tapes; train the transformer with plain **next-token CE over the
whole tape** (it learns to predict both actions and concentrations — policy + world model jointly);
roll out on **held-out** envs. In-context improvement emerges; memorization-proof (held-out dynamics);
FLOP-honest.

- `make_distillation_batch(split, *, batch_size, horizon, seed) -> (tokens, targets)` rolls the
  source policy in `ChemoEnv` and returns **static** tapes (training is batched next-token, not
  interactive — only *eval* rolls out). Training reuses the FLOP-budgeted loop with this batch
  source (generalize `train_run`'s loop to take a batch-source + eval callable, or a thin
  `distill_train_run`; avoid a third near-duplicate loop).
- **Design-risk to pin in the spec + a test:** the source must show genuine *within-episode*
  improvement (return rises with in-episode experience), else there is no learning algorithm to
  distill. A test asserts the source's 2nd-half reward > 1st-half reward on average. Documented
  alternatives (not v1): behavior-clone a noisy-greedy heuristic (simpler, less adaptive);
  return-conditioned Decision-Transformer (heavier).

## Leaderboard

Extend the leaderboard to the control metric: **regret vs oracle** (headline) and **world-model
bits** (secondary) vs **total** training+rollout FLOPs at fixed params `P`; the transformer traces
the bar. Report `P` and peak state bytes (the memory constraint is first-class). Reuse the plotting.

## Visualization — see the rollout

The rollout records a `Trajectory`; `smolml/envs/render.py` turns it into pictures so a run is
inspectable ("is it working or not"):
- `render_rollout(trajectory) -> PNG` — a **spacetime raster** (default, plain matplotlib, no new
  deps): rows = timesteps, cols = ring cells, color = concentration, with the **peak path** and the
  **agent path** overlaid, plus a cumulative reward-vs-oracle panel. One static image per episode.
- `animate_rollout(trajectory, out_gif) -> GIF` — **opt-in** animated playback (`FuncAnimation` +
  pillow), behind an availability guard (skipped if the writer is unavailable).
The baseline driver emits a raster (and optional GIF) for a sample held-out episode next to the
leaderboard. A richer **interactive, scrubbable** rollout viz is a docs-builder deliverable on the
MDX learning site (the `in-context-control` page), built post-baseline.

## FLOP honesty & memory

- All rollout compute flows through `model.step` and is counted; eval-rollout FLOPs are part of the
  total budget (ADR 0004). Training FLOPs via the shared counter. Every future candidate (C.A.1/2)
  charges its own train + online-adaptation honestly (the project invariant).
- **Memory is a hard constraint:** compare at fixed `P`; the tape is bounded by `context_window`.
  No unbounded state.

## Files

- `smolml/envs/chemotaxis.py` — `ChemoEnv`, the three reference policies, vocab/slice constants,
  `make_distillation_batch`.
- `smolml/control_eval.py` — `evaluate_control` + `ControlResult`.
- `smolml/envs/render.py` — `render_rollout` (spacetime-raster PNG) + `animate_rollout` (opt-in GIF).
- `smolml/control_train.py` — `distill_train_run`, reusing `train_run`'s FLOP-budget + JSONL shape
  (a future shared loop factored across corpus/icl/control training is welcome, not required here).
- `smolml/experiments/control_baseline.py` — thin driver: distill-train the transformer across a
  small FLOP sweep, eval on held-out episodes, write the leaderboard, print the bar.
- `smolml/leaderboard.py` — control leaderboard functions (table + plot).
- `tests/test_control.py` — the acceptance tests below.
- `docs/harness.md` — how to run a control baseline, add a control candidate, regenerate the board.

## Acceptance criteria

- Gates green: `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest` (paste outputs).
- `tests/test_control.py`:
  - **env determinism & held-out disjointness**: same seed → identical trajectory; train vs eval
    drift pools are disjoint.
  - **metric bounds**: `oracle_policy` ≈ max reward (regret ≈ 0); `random_policy` ≈ floor;
    `run_and_tumble_policy` strictly between — sanity-bounds regret.
  - **tape & slices**: alternating concentration/action tokens; action and concentration sub-vocabs
    disjoint; targets are the next-token shift; world-model bits scored **only** on concentration
    positions.
  - **causal / honest interaction**: the sampled action depends only on tokens ≤ its position
    (causal); the env feedback is computed **from** the sampled action (not pre-determined) — i.e.
    changing the policy changes the trajectory.
  - **rollout FLOP accounting**: summed `step` FLOPs match the analytic transformer decode cost for
    the rollout length.
  - **source is a learner**: the distillation source's mean 2nd-half reward > 1st-half (within-
    episode improvement exists to distill).
  - **end-to-end smoke**: distill-train a small transformer a few FLOP-budgeted steps;
    `evaluate_control` on held-out envs returns finite **mean reward strictly above the
    random-policy baseline** (beats chance) and shows within-episode improvement (2nd-half reward >
    1st-half); a leaderboard row is written.
  - **visualization**: a recorded `Trajectory` is complete and deterministic under fixed seed;
    `render_rollout` writes a non-empty PNG; the GIF path is exercised behind an availability guard.
- `docs/harness.md` updated.
- Per AGENTS.md: a researcher note + an `in-context-control` concept (incl. an interactive,
  scrubbable rollout visualization) handed to the docs-builder once the baseline lands (confirmed
  accurate by a researcher).

## Out of scope (later tasks / candidates)

- The candidates themselves: **C.A.1** reservoir + plastic readout, **C.A.2** chemotaxis-minimal
  (scored on this spine, not part of Phase 0).
- The static rungs 1–3 (`C.0-icl-harness.md`) — independent; the candidate is graded on both.
- 2-D / gridworld navigation; multi-peak or adversarial drift; return-conditioned (DT) training;
  reward-channel separate from sensing (documented options, not v1).
- **Sibling environments** (non-stationary bandit, homeostatic setpoint control, derivative-free
  objective) — they reuse the `Environment` seam, scorer, and candidates, but are future rungs, not
  C.A.0.
