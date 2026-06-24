# Task C.A.3 — contingency-forage control rung (realize the Environment seam + a reflex-proof rung)

- Status: SPEC — design approved in brainstorm; **revised after a 4-lens adversarial review that
  (Monte-Carlo) proved the first dynamics were NOT reflex-proof** and surfaced the seam/migration
  surface. Phase 2 of the embodied-control arc: a control rung that — unlike chemotaxis (C.A.0) —
  **demands genuine in-context learning**, so regret separates a real learner from a fixed reflex.
  Infrastructure + transformer baseline, **not a candidate** (the brain-style local-learning candidate
  is the follow-on PR, mirroring C.A.0 → C.A.1/C.A.2).
- Branch: `task/C.A.3-contingency-forage` off `main`. Own PR; **do not merge** (human merges).
- Metric (ADR 0007): held-out **regret-per-FLOP at fixed params `P`** (+ world-model bits). The
  transformer is the **honest baseline, not a strawman** — its bar is the **best of a
  training-hyperparameter sweep** (see "Fair baseline"), not one arbitrary config.

## Why this rung (the C.A.0 gap, fixed)

`ChemoEnv`'s optimal policy is "climb the local gradient" — a **fixed reflex** does that without
inferring the held-out drift, so `chemotaxis_min` (a weightless leaky integrator) floored the FLOPs and
"won" without learning. `ForageEnv` makes the optimal policy depend on a **per-episode latent the agent
can only learn from its own eat-outcomes**: no fixed policy is near-optimal, in-context learning is
*required*, and it is the scalar-reward / no-derivative setting where a local learner's
credit-assignment edge over the transformer's `O(window·d)` attention can show.

## The environment — `ForageEnv` (`smolml/envs/forage.py`)

A ring of `W` cells (default 16). **Every cell holds a cue of one of `K` types** (default `K=3`),
sampled i.i.d. uniform at episode start and **stationary thereafter** (cells never change type). The
agent at position `p_t` senses the cue type at its own cell. Actions: `{LEFT(−1), EAT, RIGHT(+1)}`.

- **Latent reward-contingency.** Exactly one type `g ∈ [0,K)` is *good* this episode, drawn **uniform
  per episode**. `EAT` of the current cell → reward `+1` if its type `== g` else **`−1` (poison)**,
  then the agent **advances one cell** (`p += 1`). `LEFT`/`RIGHT` move without eating (reward `0`).
  `g` AND the layout are **fresh every episode** (train/eval = disjoint seed bands), so there is **no
  fixed mapping to memorize**; only the inference algorithm ("eat the type that pays, skip the type
  that poisons") generalizes.
- **Why these exact dynamics (all three properties are load-bearing, MC-verified):**
  - **Poison (`−1`)** ⇒ blind eating is net-negative ⇒ **reflex-proof** (a weightless `always-EAT`
    cannot win).
  - **Stationary cells + `EAT`-advances (no camping)** ⇒ the agent re-encounters each `g` cell every
    lap, so the food **never depletes** (the oracle's reward is flat, not decaying) ⇒ there is a stable
    target to learn ⇒ the rung is **distillable** (a learner improves within-episode). Eating cannot
    camp one cell, so recognizing the rewarding **type** (to eat fresh `g` cells on sight) is the
    efficient strategy — cell-by-cell memory also works but is slower (higher regret). Honest framing:
    the rung *requires in-context adaptation*; type-generalization is the FLOP-efficient solution.
- **Determinism.** `ForageEnv(split, seed)` fully reproducible: `reset() -> obs0`; `step(action) ->
  (obs_next, reward)`; `oracle_action()`; exposes `horizon`, `n_actions`.
- **MC-pinned references (W=16, K=3, H=64; pin in tests to these ± tol):** `oracle` (knows `g`; EAT
  when on `g` else move) **≈ +0.335**, regret ≈ 0; `always_eat` (the reflex) **≈ −0.329**;
  `always_right` (best contingency-blind fixed policy) **≈ 0.000**; `random` ≈ −0.112; the distillation
  source `win_stay_lose_shift` **≈ +0.283** with within-episode improvement (1st ≈ +0.25, 2nd ≈ +0.31).
  - **Oracle = the regret reference**, defined under the SAME local sensing as the agent (it knows `g`
    but not cell positions), so regret is purely the cost of not knowing `g` (learning-attributable).
    Pinned by a Monte-Carlo estimate in a test; it **upper-bounds every contingency-blind policy**
    (assert `oracle_reward ≥ best_fixed_reward + margin` and `regret(always_eat) > margin > 0`).

## Tape, vocab, slices (forage) — combined obs avoids post-eat blindness, keeps the 2-token scorer

The obs symbol encodes **both** the current cell type **and** the reward of the last action, so the
policy always sees the current type (informed decisions) AND the reward signal stays in-context — with
the **same 2-token `obs·action·obs·action` cycle chemotaxis uses** (no scorer schedule change):

- **Obs vocab = `3·K` combined `(type, last_reward)` symbols.** `obs = type · 3 + (last_reward + 1)`,
  `last_reward ∈ {−1, 0, +1} → {0, 1, 2}` (at `reset`, `last_reward = 0`). The world-model (post-action
  position) predicts the next combined obs (its reward component IS the contingency belief).
- **Exact id map (pin in a test; the scorer indexes within-slice by the raw token, so obs MUST occupy
  `[0, obs_len)`):** combined obs `0..3K−1`; actions `LEFT, EAT, RIGHT → 3K, 3K+1, 3K+2`.
  `obs_slice = slice(0, 3K)`, `action_slice = slice(3K, 3K+3)`. `vocab_size = 3K + 3` (= 12 at K=3).
- Policy = `softmax(logits[action_slice])` at post-obs positions; world-model = `softmax(logits[
  obs_slice])` at post-action positions. A disjoint-slice + obs-at-0 invariant test.

## Phase A — realize the `Environment` seam (the C.A.0 doc designed it; the impl hardcoded ChemoEnv)

`evaluate_control(model, cfg: ChemoConfig, …)`, `distill_train_run`, and `make_distillation_batch` are
**hardcoded** to `ChemoEnv`/`conc_slice`/`action_slice`/`RunAndTumble`. Generalize them to the protocol
+ a small spec bundle, **chemotaxis = instance #1, forage = instance #2, chemotaxis kept
bit-identical** (no second scorer — the no-duplicate-convention rule):

- **`Environment` protocol** (extend, typing-only): `reset()`, `step(action)`, `oracle_action()`,
  `n_actions: int`, `horizon: int`, and `record_state() -> dict` (per-env trajectory payload).
- **`EnvSpec`** (new dataclass): `env_factory(split, seed) -> Environment`, `source_factory(seed) ->
  Policy` (any `reset()/act(obs)->action`), `tape_spec` (`vocab_size, obs_slice, action_slice,
  action_token(idx)`). Chemotaxis and forage each build one; a one-call `chemo_env_spec(chem)` /
  `forage_env_spec(cfg)` helper makes call-site migration one line.
- **`evaluate_control(model, env_spec, *, n_episodes, seed, device, greedy, record)`** — the
  reward/regret/world-model-bits/FLOP loop becomes protocol-only. The generic trace records
  `(obs_token, action, reward, predicted_obs_dist)` (all generic); `record_state()` adds the per-env
  payload, sampled **once after `reset()` and once after each `step()`** (`H+1` rows, matching today's
  `len(mu)==len(pos)==H+1`). Per-env renderers consume the payload.
- **`make_distillation_batch(env_spec, *, batch_size, seed, device, **source_kwargs)`** and
  **`distill_train_run(cfg)`** take the `EnvSpec` (cfg carries which env + its config).
- **RNG bit-identity (the load-bearing constraint — drift is invisible to FLOPs):** preserve the THREE
  seed formulas **verbatim** (centralize in named helpers, do **not** unify): env `seed*100003+b` /
  eval `seed*100003+ep`, source `seed*7919+b`, distill outer `cfg.seed*1009+step`; and
  `torch.manual_seed(cfg.seed)` before `build_model`; `env_factory` draws **no torch RNG** at
  construction. **A NEW pin test runs `distill_train_run` end-to-end on chemotaxis and asserts the
  numeric regret/reward/world-model-bits/FLOPs equal pre-refactor values** (not just FLOPs).
- **Migration (hard cutover, budget it):** the signature change breaks ~11 call sites
  (`tests/test_control_eval.py`, `test_control_train.py`, `test_control_baseline.py`,
  `test_chemotaxis_min.py`, `test_reservoir.py`, `test_reservoir_plastic.py`) **and 4 candidate drivers**
  (`experiments/chemotaxis_min_control.py`, `reservoir_control.py`, `reservoir_plastic_control.py`,
  `export_demo_fixtures.py`). Migrate each to pass `chemo_env_spec(chem)` (one line); assertions
  unchanged. Existing structural tests stay green **after migration** (not "unchanged").
- **Leaderboard:** add an **env identifier** to the control meta record (replace the chemo-only
  `levels/width/sigma` with the `EnvSpec` config dict + env name); `forage_baseline` writes its **own
  `runs/forage` dir** so chemo-regret and forage-regret are never ranked together.

## Training the baseline — Algorithm Distillation + a within-episode-learner source

Reuse the generalized `distill_train_run`: roll the source over fresh episodes → next-token CE over the
whole tape → roll out on held-out episodes.

- **Source = `win_stay_lose_shift` (elimination tracker):** per-type belief `k[type] ∈ {unknown, good,
  bad}`; act on the sensed type — EAT if good or unknown (explore by eating to learn), skip (move) if
  bad; `+1` marks the type good, `−1` marks it bad; `ε`-explore (random action). MC-verified to improve
  within-episode (explore early → exploit `g` late). Pin the exact tracker (init `unknown`; switch
  rule; the move direction is `RIGHT`; `ε` grid `{0.05, 0.1, 0.2}`).
- **Source-is-a-learner guard (test):** source mean **2nd-half reward > 1st-half by a pinned delta**
  (≈ +0.06 at the defaults) AND 2nd-half > `always_eat` rate — a real, distillable learning curve.

## Fair baseline (the bar is the best of a sweep, not one config)

A transformer distill-train is **fast** (15 s–2 min on the chemotaxis proxy; regret plateaus on FLOPs
alone) — so hyperparameters, not compute, are the lever; an un-swept bar is the strawman ADR 0007
forbids. `smolml/experiments/forage_baseline.py`:

- Sweep **training** hyperparameters at **fixed params `P`** (so the fixed-memory comparison is honest):
  `lr ∈ {1e-3, 3e-3, 1e-2}`, `weight_decay ∈ {0.0, 0.1}`, `batch_size ∈ {16, 32}`, `grad_clip`, AD-source
  `ε ∈ {0.05, 0.1, 0.2}`, at a fast budget. (`d_model`/`n_layers` are **not** swept — they change `P`.)
- **Re-rank the top 2–3 fast-budget configs at the actual leaderboard budget** before committing the
  winner (the small-budget ranking can differ); **log/verify the FLOP-plateau empirically for forage**
  (don't inherit the chemotaxis claim). Trace the FLOP-budget curve (3–4 points) with the winner.
- The sweep table + chosen config + FLOP-budget regret curve are logged and pasted in the PR.

## Visualization

`render_rollout` generalizes via the per-env `record_state()` payload: a spacetime raster (rows=steps,
cols=ring cells, color=**cue type**, agent path + eaten-type marks + cumulative reward-vs-oracle). The
interactive scrubbable viz on the `in-context-control` MDX page is a **docs-builder** follow-on
(ADR 0006 — researcher note handed over, not hand-written here).

## Acceptance

- **Gates green:** `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest` (paste outputs).
- **Chemotaxis bit-identity (Phase A safety net):** the new end-to-end pin test asserts chemotaxis
  regret/reward/world-model-bits/FLOPs are unchanged on a fixed seed; all migrated control tests green.
- **`tests/test_forage.py`:**
  - env determinism (same `(split,seed)` → identical trajectory); **train/eval seed bands disjoint**
    (assert non-overlap); fresh `g`+layout per episode.
  - tape & slices: alternating obs/action; **obs sub-vocab occupies `[0, 3K)`, actions above**;
    `vocab_size == 3K+3`; world-model bits scored only on obs positions; the obs symbol encodes
    `(type, last_reward)` per the pinned id map.
  - **metric bounds (at production H=64):** `oracle ≈ +0.335` (MC-pinned, regret ≈ 0); `random ≈
    −0.112`; `always_eat ≈ −0.329`; `win_stay_lose_shift` strictly between `always_right` and oracle.
  - **REFLEX-PROOF (the headline, closing C.A.0's gap):** over the fixed-policy family
    (`always_eat`, `always_right`, `always_left`, `eat_fixed_type_k` for each `k`), `max regret >
    win_stay_lose_shift regret` by a pinned delta — AND `oracle_reward ≥ best_fixed_reward + margin`
    AND `regret(always_eat) > margin > 0`. The rung *requires* in-context adaptation.
  - **causal / honest interaction:** sampled action depends only on tokens ≤ its position; env feedback
    computed **from** the sampled action (changing the policy changes the trajectory).
  - **rollout FLOP accounting:** summed `step` FLOPs == analytic transformer decode for the rollout.
  - **source-is-a-learner:** source 2nd-half reward > 1st-half by the pinned delta.
  - **end-to-end smoke:** distill a small transformer a few FLOP-budgeted steps; held-out mean reward
    strictly above random AND 2nd-half > 1st-half AND **regret below the best-fixed-policy regret** (it
    learned to adapt); a leaderboard row written; `render_rollout` writes a non-empty PNG.
- **Fair baseline run:** the sweep table + chosen config + the FLOP-budget regret curve, in the PR.
- `docs/harness.md` updated; researcher note + `in-context-control` concept update handed to the
  docs-builder.

## Out of scope (later tasks / candidates)

- **The brain-style local-learning candidate** scored on this rung (the real prize) — its own PR.
- Documented variants, not v1: 2-D grid / growing body / self-collision; **drifting** within-episode
  contingency (`g` flips mid-episode); multi-good-type; a finer egocentric sensing window; refreshing
  (non-stationary) layouts that keep `g`-density constant (a richer non-depleting variant).
- The static ICL rungs (`C.0-icl-harness.md`) — independent.
