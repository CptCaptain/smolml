# Task C.A.3 — contingency-forage control rung (realize the Environment seam + a reflex-proof rung)

- Status: SPEC — design approved in brainstorm. Phase 2 of the embodied-control arc: a second control
  rung that — unlike chemotaxis (C.A.0) — **demands genuine in-context learning**, so regret separates
  a real learner from a fixed reflex. Infrastructure + transformer baseline, **not a candidate** (the
  brain-style local-learning candidate is the follow-on PR, scored on this rung — mirrors C.A.0 →
  C.A.1/C.A.2).
- Branch: `task/C.A.3-contingency-forage` off `main`. Own PR; **do not merge** (human merges).
- Metric (ADR 0007): held-out **regret-per-FLOP at fixed params `P`** (+ world-model bits). The
  transformer is the **honest baseline, not a strawman** — so its bar is the **best of a
  training-hyperparameter sweep**, not one arbitrary config (see "Fair baseline").

## Why this rung (the C.A.0 gap, fixed)

`ChemoEnv` *has* drift, but its optimal policy is "climb the local gradient" — a **fixed reflex**
(`chemotaxis_min`, a weightless leaky integrator) does that without ever *inferring* the held-out
drift, so it floored the FLOPs and "won" without learning. The rung never **required** in-context
inference. ADR 0007's whole point is a capability proxy that **cannot be won without learning**.

`ForageEnv` makes the optimal policy depend on a **per-episode latent the agent can only learn from
its own eat-outcomes**, so no fixed policy can be near-optimal: in-context learning is *required*, and
this is the scalar-reward / no-derivative setting where a local learner's credit-assignment edge over
the transformer's `O(window·d)` attention can show.

## The environment — `ForageEnv` (`smolml/envs/forage.py`)

A ring of `W` cells (default 16). **Every cell holds a cue of one of `K` types** (default `K=3`), laid
out fresh per episode. The agent at position `p_t` **senses only the cue type at its own cell**.
Actions reuse the three deltas `{LEFT(−1), STAY=EAT(0), RIGHT(+1)}`.

- **Latent reward-contingency.** Exactly one cue type `g ∈ [0,K)` is *good* this episode, drawn
  **uniformly per episode**; `EAT` of type `g` → reward `+1`, any other → `0`. `g` AND the layout are
  **fresh every episode** (train/eval draw from **disjoint seed pools** — the held-out split), so there
  is **no fixed mapping to memorize**: only the *inference algorithm* ("eat types, keep the one that
  pays") generalizes. Memorization-proof and reflex-proof by construction.
- **Dynamics.** `EAT` (STAY) consumes the current cell's cue (reward by contingency) and the cell
  **refreshes to a fresh random type** so the horizon-`H` episode keeps going; `LEFT`/`RIGHT` move one
  cell (no eat) to sample a neighbor. Explore (move to sense other cells) vs exploit (EAT the
  inferred-good type). Deterministic given `(split, seed)`: `reset() → obs0`; `step(action) →
  (obs_next, reward)`.
- **References (bound the metric, mirror ChemoEnv's three).** `oracle` (knows `g` → EATs only when the
  current cell is `g`, else moves toward... — see oracle note; reward ≈ 1, regret ≈ 0); `random`
  (floor); **`win_stay_lose_shift`** (a scripted *adaptive* reference and the distillation source: EAT
  while the current best-believed type pays, switch on a miss, ε-explore — strictly between).
  - *Oracle note.* With purely-local sensing the oracle cannot teleport to a `g` cell; the **regret
    reference** is the **best achievable known-`g` policy** under local sensing (EAT when on `g`, else
    move one step; with cells dense and `g` ≈ `1/K` of cells, expected reward is well-defined and
    bounded above any contingency-blind policy). This keeps regret ≥ 0 and reflex-proof; pin the
    oracle's expected reward in a test against a Monte-Carlo estimate.

## Phase A — realize the `Environment` seam (the C.A.0 doc designed it; the impl hardcoded ChemoEnv)

`evaluate_control`, `distill_train_run`, and `make_distillation_batch` are currently **hardcoded** to
`ChemoEnv`/`ChemoConfig`/`conc_slice`/`action_slice`/`RunAndTumble`. To add a sibling rung **without a
second scorer** (the no-duplicate-convention rule), generalize them to consume the protocol + a tiny
tape spec, with **chemotaxis as instance #1 and forage as instance #2**:

- **`Environment` protocol** (extend minimally, typing-only): `reset() -> int`, `step(action) ->
  (obs, reward)`, `oracle_action() -> int`, `n_actions`, and an optional `record_state()` hook for the
  per-env trajectory (chemotaxis: `mu/p/field`; forage: layout/eaten-type) so the core scorer stays
  env-agnostic.
- **`TapeSpec`** (new small dataclass): `vocab_size`, `obs_slice`, `action_slice`, `action_token(idx)`
  — the obs/action sub-vocab layout. Chemotaxis and forage each construct one.
- **`evaluate_control(model, env_factory, tape_spec, *, n_episodes, seed, device, greedy, record)`** —
  the reward/regret/world-model-bits/FLOP loop becomes protocol-only (env via `env_factory(split,
  seed)`, slices via `tape_spec`, oracle via `env.oracle_action()`). The generic trace records
  `(obs_token, action, reward)`; per-env renderers interpret it.
- **`make_distillation_batch(env_factory, source_factory, tape_spec, ...)`** — rolls a
  `source_factory(seed)` (any `reset()/act(obs)->action` policy) into next-token tapes.
- **`distill_train_run`** — takes the env_factory + source_factory + tape_spec (a thin
  `EnvSpec` bundle) instead of building `ChemoConfig` internally.
- **Hard invariant + regression test:** the chemotaxis baseline is **bit-identical** after the
  refactor — same regret/reward/world-model-bits/FLOPs on a fixed seed (assert against pre-refactor
  values, mirroring the degenerate-identity discipline). Chemotaxis and forage both register an
  `EnvSpec`; nothing else in the scorer/leaderboard branches on the env.

## Tape, vocab, slices (forage)

For in-context contingency learning the tape **must carry `(eaten-type, reward)` pairs** — that is the
entire signal. Minimal extension of the 2-token `obs·action·obs·action` cycle:

- **Obs vocab = `K` cue-type symbols + `2` reward symbols** (`0`/`+1`). The observed symbol after a
  **MOVE** is the new cell's cue type; after an **EAT** it is the **reward symbol**. Tape reads
  `type · EAT · reward · type · MOVE · type · …`, so each eat writes `(type just sensed, action,
  reward)` into the context the policy conditions on.
- `vocab_size = K + 2 + 3`. Policy = `softmax(logits[action_slice])` at post-obs positions;
  world-model = `softmax(logits[obs_slice])` at post-action positions (predicting reward after an EAT
  IS the contingency belief — a direct "does it know `g` yet?" readout). A parity convention + a
  disjoint-slice test, exactly as chemotaxis.

## Training the baseline — Algorithm Distillation + a within-episode-learner source

Reuse the generalized `distill_train_run`: roll the adaptive source over fresh episodes → next-token CE
over the whole tape → roll out on held-out episodes.

- **Source = `win_stay_lose_shift`** generalized to a per-type reward tracker: EAT the best-believed
  type when on it (else move to sample), `+1` reinforces, `0` switches, ε-explores; its return rises
  within-episode as it pins `g`.
- **Source-is-a-learner guard (the C.A.0 pinned design-risk, as a test):** source mean **2nd-half
  reward > 1st-half** on fresh episodes — else there is no learning algorithm to distill.

## Fair baseline (the bar is the best of a sweep, not one config)

A transformer distill-train is **fast** (15 s–2 min on the chemotaxis proxy: 150/600/1500/3000 steps →
15/29/66/120 s), and regret **plateaus on FLOPs alone** while sitting far from oracle — so the
hyperparameters, not compute, are the lever. An un-swept bar would be the strawman ADR 0007 forbids.

- `smolml/experiments/forage_baseline.py` sweeps **training** hyperparameters **at fixed params `P`**
  (so the fixed-memory comparison is honest): `lr ∈ {1e-3, 3e-3, 1e-2}`, `weight_decay ∈ {0.0, 0.1}`,
  `batch_size ∈ {16, 32}`, `grad_clip`, and the **AD-source `epsilon`** — at a fast budget; pick the
  **min-regret** config; then trace the FLOP-budget curve (3–4 points) with that winner for the
  leaderboard. (`d_model`/`n_layers` are **not** swept — they change `P`; a second `P` may be reported
  as a separate point, not a tuning knob.) The chosen config + the sweep table are logged and pasted in
  the PR.

## Scorer / leaderboard / viz

- After Phase A, `evaluate_control` + `ControlResult` + the control leaderboard (`regenerate_control`)
  serve forage with no further change (regret vs **total** FLOPs at fixed `P`; peak state bytes
  reported). Every rollout FLOP counted (ADR 0004).
- `render_rollout` generalizes via the per-env trace: a spacetime raster (rows=timesteps, cols=ring
  cells, color=**cue type**) with the agent path + eaten-type marks + cumulative reward-vs-oracle. The
  interactive scrubbable viz on the `in-context-control` MDX page is a **docs-builder** follow-on
  (ADR 0006 — researcher note handed over, not hand-written here).

## Acceptance

- **Gates green:** `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest` (paste outputs).
- **Chemotaxis regression (Phase A safety net):** post-refactor chemotaxis baseline is **bit-identical**
  — same regret/reward/world-model-bits/FLOPs on a fixed seed; existing `tests/test_control.py` stays
  green unchanged.
- **`tests/test_forage.py`:**
  - env determinism (same `(split,seed)` → identical trajectory); **train/eval seed pools disjoint**;
    fresh `g`+layout per episode (no fixed mapping).
  - tape & slices: alternating obs/action; obs (cue+reward) and action sub-vocabs disjoint; world-model
    bits scored only on obs positions; reward symbol appears iff the prior action was EAT.
  - **metric bounds:** `oracle` ≈ its (Monte-Carlo-pinned) max, regret ≈ 0; `random` ≈ floor;
    `win_stay_lose_shift` strictly between.
  - **REFLEX-PROOF (the headline, closing C.A.0's gap):** the best *fixed* policy (eat-always / eat-a-
    fixed-type) has regret bounded **well above** `win_stay_lose_shift` — the rung *requires* in-context
    adaptation.
  - **causal / honest interaction:** sampled action depends only on tokens ≤ its position; env feedback
    computed **from** the sampled action (changing the policy changes the trajectory).
  - **rollout FLOP accounting:** summed `step` FLOPs == analytic transformer decode for the rollout.
  - **source-is-a-learner:** distillation source mean 2nd-half reward > 1st-half.
  - **end-to-end smoke:** distill a small transformer a few FLOP-budgeted steps; held-out mean reward
    strictly above random AND 2nd-half > 1st-half AND regret **below the best-fixed-policy regret** (it
    learned to adapt); a leaderboard row written; `render_rollout` writes a non-empty PNG.
- **Fair baseline run:** the hyperparameter sweep table + the chosen config + the FLOP-budget regret
  curve, pasted in the PR.
- `docs/harness.md` updated (run a forage baseline, add a forage candidate, regen the board); a
  researcher note + `in-context-control` concept update handed to the docs-builder.

## Out of scope (later tasks / candidates)

- **The brain-style local-learning candidate** scored on this rung (the real prize) — its own PR.
- Documented variants, not v1: 2-D grid / growing body / self-collision; **drifting** within-episode
  contingency (`g` flips mid-episode); poison (`−1`) rewards; multi-good-type; richer egocentric
  sensing (a local window). v1 = one fixed-per-episode `g`, reward `{0,+1}`, single-cell sensing.
- The static ICL rungs (`C.0-icl-harness.md`) — independent.
