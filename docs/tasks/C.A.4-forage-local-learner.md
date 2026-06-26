# Task C.A.4 — gradient-free local-learning candidates on the forage rung (the prize)

- Status: SPEC — design approved in brainstorm (mechanism: tracker headline + reservoir control;
  ~0-distillation headline + regret-vs-FLOP curve; comparison at production **H=64**). Phase 3 of the
  embodied-control arc: the first **candidate** on the reflex-proof `forage` rung (C.A.3), mirroring
  C.A.0 → C.A.1/C.A.1b/C.A.2. This is the direct test of the project thesis: local learning's
  credit-assignment edge in the **scalar-reward / no-derivative** setting, where `column_mix` (B.5)
  could *not* win on dense byte prediction.
- Branch: `task/C.A.4-forage-local-learner` off `main`. Own PR; **do not merge** (human merges).
  Cross-review by a **different vendor** (codex FLOP audit before landing).
- Metric (ADR 0007): held-out **regret-per-FLOP at fixed params `P`** on the forage rung, reported as a
  **regret-vs-total-FLOP curve** (pretraining + inference + online adaptation all counted, ADR 0004),
  never a point. The transformer is the **honest bar** — best of a hyperparameter sweep at fixed `P`.

## The bar to beat (re-established on this machine at H=64)

`forage_baseline.py` is **re-run at H=64** (production) — full training-hyperparameter sweep at fixed
`P` + re-rank top configs + a FLOP-budget regret curve with the winner. Reference rung means at H=64
(MC-pinned in `tests/test_forage.py`): `oracle ≈ +0.96` (regret 0), `win_stay_lose_shift ≈ +0.85`
(regret ≈ 0.11, the distillation source), `always_eat ≈ −0.333`, `always_right/left = 0.0`,
`random ≈ −0.112`. The handoff's working bar was ≈ 0.16 regret / +0.77 reward @ ~3e11 FLOPs at H=32;
the H=64 bar is re-measured and pasted in the PR.

## The two candidates (one PR; the C.A.1/C.A.1b/C.A.2 sibling pattern)

### `forage_min` — per-type contingency tracker (headline; the FLOP-floor reference)

The bandit analogue of `chemotaxis_min`: swap its scalar leaky-integrator baseline `b` for a **per-type
value vector** `v[K]`, the in-context memory (reset per episode, **not** an `nn.Parameter`). Mirrors
`chemotaxis_min` structurally — a shared `_emit_logits` head with **identical arithmetic** in the
scalar `step` path and the batched `forward` path, an in-context state updated each token, and the
distilled scalars differentiable through the recurrence so a short distill can tune them.

- **In-context state** (`init_prequential_state`): `v[K]` initialized to `v_init` (slightly optimistic
  ⇒ unknown types get eaten = explore-by-eating), plus `last_action` and `last_type`. O(K) memory; no
  token window retained.
- **Decode** (the seam id-map): `type = obs // 3`, `last_reward = obs % 3 − 1`.
- **Local update rule** — the credit assignment is exact, local, O(1): at an obs (EVEN) position, if the
  preceding action was `EAT`, the agent ate the cell it stood on, whose type equals the current type
  (EAT does not move). So `v[t] += lr · (r − v[t])` — a delta rule = an EMA of observed reward for type
  `t`. This **is** the contingency estimate. Gated on `last_action == EAT` (moves give `r = 0`).
- **Policy head** (action slice), distilled scalars `g, b_eat, b_left, b_right`:
  `logit(EAT) = g · v[t] + b_eat`; `logit(LEFT) = b_left`; `logit(RIGHT) = b_right`. Eat high-value
  (good) types, search past poison toward a good cell, then camp it — win-stay-lose-shift expressed as a
  **learned, differentiable, FLOP-counted softmax policy**, not a hardcoded if/else.
- **World-model head** (obs slice, `3K` logits): the reward component *is* the contingency belief. After
  `EAT`: next type = current type (sticky), reward-level peaked by a world-model gain `g_wm · v[t]`;
  after a move: type uniform over `K`, reward = 0. Built as a (type ⊗ reward) outer-product, O(K)
  pointwise — mirrors `chemotaxis_min`'s peaked concentration head. A couple of distilled world-model
  scalars (`g_wm`, type stickiness).
- **Params**: ~8 scalar `nn.Parameter`s — `lr_logit` (rate `lr = sigmoid(lr_logit)`, learnable like
  `chemotaxis_min`'s leak), `v_init`, the policy `g, b_eat, b_left, b_right`, and the world-model
  `g_wm` + type-stickiness scalar. All differentiable through the recurrence so distillation tunes
  them. `num_params` ≈ 8, like `chemotaxis_min`'s 5 — dominates the bar on memory.
- **FLOP accounting** (honest, via `smolml.flops`): all compute is **pointwise** (no matmul dominates),
  charged via `pointwise_flops` per the flops-module conditional-omission rule. `flops(seq_len)` =
  `FlopBreakdown.from_forward(seq_len × per_step_ops)` (the standard 3×: the distilled scalars
  participate at every position). `step` and `decode_step_flops` charge exactly one per-step forward,
  **backward = 0** — the `v` delta-rule update is forward compute, not a weight change (identical
  treatment to `chemotaxis_min`'s integrator and `reservoir_plastic`'s rule, except that rule's outer
  products land in `step.backward`; here the update is O(K) pointwise and is forward compute). The
  per-step op count is fixed and context-independent; hand-checkable named constants like
  `chemotaxis_min`'s `_LEAK_OPS`/`_SENSE_OPS`/….

### `forage_reservoir` — frozen reservoir core + plastic readout (control; generic capacity)

The `reservoir_plastic` (C.A.1b) mechanism — `_ReservoirCore` reused **unchanged** (frozen echo-state
substrate, ~148k params at `d_res=374` for memory-parity with the transformer bar) + a reward-modulated
plastic readout adapted by a local rule in `step`, all adaptation FLOPs charged to `step.backward`,
~0 distillation headline. **One change** from `reservoir_plastic`: decode the forage reward correctly.
`reservoir_plastic`'s reward proxy `r = revealed_byte/(levels−1)` treats the obs token as a monotone
concentration; for forage the obs is a **combined** `(type, reward)` symbol, so the reward proxy must be
`r = obs % 3 − 1` (the decoded last-reward), reward-modulating only when an eat-reward occurred. The
world-model delta rule targets the full `obs_slice` one-hot (the `3K` combined symbols) as-is.

- This exact shape **already lost on chemotaxis** (regret 0.49/0.37/0.28 per memory). Running it on
  forage isolates the claim that **per-type credit-assignment structure, not raw capacity**, is the
  lever: the reservoir state conflates types across the trajectory, so per-type credit assignment
  through one plastic readout is muddy, while `forage_min` knows exactly which type was eaten.
- Reuse `_ReservoirCore` and the `_PlasticCache`/`step` machinery; factor only what is genuinely shared
  (KISS, no premature abstraction). New registered class with a forage reward decode.

## The (iv) bet + the honest risk (the skeptic's mandate)

- **The per-FLOP edge**: `forage_min` runs an O(K)/step local update with **perfect** local credit
  assignment and ~0 distillation, so its *total* FLOPs are ~1e5–1e6 (eval rollout only) vs the
  transformer bar's ~3e11 — a ~6-OOM FLOP-floor, the same shape `chemotaxis_min` showed on C.A.0. If it
  also matches `win_stay_lose_shift`'s regret (~0.11 at H=64) it **beats** the bar's regret on *both*
  axes (memory and FLOPs).
- **The trap, stated up front**: the tracker is well-matched to the bandit, so a win partly re-proves
  "forage's optimum is cheaply learnable by a local rule," as `chemotaxis_min` re-proved for chemotaxis.
  But forage is **reflex-proof** (C.A.3): no *fixed* policy is near-optimal, and the tracker is a genuine
  *online learner* that adapts within-episode (its 2nd-half reward must exceed its 1st-half). So a
  tracker win is exactly the (iv) thesis, not a strawman. Defensibility vs "WSLS smuggled as a
  hand-policy": it is a learned differentiable softmax policy emitting an action **distribution** through
  `model.step`, every FLOP charged — identical standing to the **accepted** `chemotaxis_min`. The
  `win_stay_lose_shift` source is itself a cheap local learner and the transformer bar is **distilled
  from it**, so the candidate must beat the **distilled transformer** per-FLOP, not just WSLS.
- **Failures are data.** Honest outcomes: (a) `forage_min` wins per-FLOP → local credit-assignment edge
  demonstrated (the prize); (b) `forage_reservoir` loses but `forage_min` wins → it is *structure*, not
  capacity, that matters (a sharp contrast); (c) even `forage_min` cannot match the bar's regret → the
  distilled transformer's policy is genuinely better, a real data point.

## Comparison protocol (the deliverable)

- **Same `forage_env_spec(ForageConfig(horizon=64))` and same horizon for candidate and bar** — apples
  to apples. Re-run the transformer bar on this machine at H=64.
- **Regret-vs-total-FLOP curve, never a point**, for `forage_min`, `forage_reservoir`, and the
  transformer bar, on held-out (eval-band) episodes. The candidates sweep the distillation budget
  (including a ~0-distillation point); the bar sweeps its FLOP budget with the swept-winning config.
- **`distill_train_run` with `flop_budget → 0` steps** drives the ~0-distillation headline (set a small
  positive `flop_budget` so the loop runs 0 train steps — the eval rollout is the whole cost), matching
  `reservoir_plastic`'s headline path. The differentiable scalars mean a few FLOP-budgeted steps tune
  the inits where it helps; the curve shows the tradeoff.
- Both candidates register with **zero harness changes** (subclass `LanguageModel`, `@register_model`),
  run through `evaluate_control(model, forage_env_spec(cfg), …)`, and write their own `runs/forage`
  leaderboard rows via `regenerate_control`.

## Testing (mirror `tests/test_reservoir_plastic.py` + `tests/test_forage.py`)

- **`forage_min` unit tests**: `step`/`forward` parity (identical logits on the same tape, the
  `chemotaxis_min` invariant); determinism (fixed seed ⇒ fixed logits); the delta-rule update moves
  `v[t]` toward the observed reward and **only** on a post-`EAT` obs fold; optimistic init eats unknowns;
  after learning, the policy eats the good type and moves past poison; `num_params` is the expected small
  count; full-vocab logits with obs in `[0, 3K)` and actions above; FLOP honesty — summed `step` FLOPs
  equal the analytic `decode_step_flops × steps`, `step.backward == 0`, `flops()` is `from_forward`.
- **`forage_reservoir` unit tests**: reward is decoded as `obs % 3 − 1` (not the raw token); the plastic
  readout adapts in `step` (eval mutates no `nn.Parameter`); `num_params` matches memory-parity sizing;
  adaptation FLOPs land in `step.backward`; `_ReservoirCore` is reused unchanged.
- **Behavioral / end-to-end (both)**: through `evaluate_control` on held-out episodes — mean reward
  strictly above `random`, **2nd-half reward > 1st-half** (genuine within-episode learning, the
  reflex-proof requirement), and **regret below the best-fixed-policy regret**; a leaderboard row
  written; `render_rollout` writes a non-empty PNG.
- **No mocks; real seam.** Behavior, not plumbing.

## Acceptance

- **Gates green** (pasted in the PR): `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest`,
  plus the regret-vs-total-FLOP curve for both candidates vs the re-run transformer bar at H=64.
  (Pre-existing ruff drift in `smolml/experiments/export_demo_fixtures.py` is OUT OF SCOPE.)
- **`forage_min` headline**: a ~0-distillation point on the curve at total FLOPs ≥ ~6 OOM below the bar,
  with regret reported (target ≈ `win_stay_lose_shift`'s ~0.11; a regret win over the bar's H=64 number
  is the headline if achieved).
- **`forage_reservoir` control**: the curve showing generic capacity loses on regret-per-FLOP (or, if it
  surprises us, that is data — report honestly).
- **Cross-vendor codex FLOP audit** before landing (`codex exec -s read-only -C "$PWD" "<focused FLOP
  prompt>"`, gpt-5.5) — it caught a real undercharge on every prior candidate. Run `uv run pytest`
  yourself (codex's sandbox pytest fails — environmental).
- **`docs/harness.md`** gets a short forage-candidate note; a **researcher note** for both candidates is
  handed to the docs-builder (ADR 0006 — researchers do not hand-write `docs/learning/`), then the
  resulting page is confirmed accurate.

## Out of scope

- Hand-written `docs/learning/` MDX (docs-builder's job; hand over a researcher note).
- The harder forage variants documented out-of-scope in the C.A.3 spec (drifting within-episode
  contingency, 2-D grid, multi-good-type, refreshing layouts).
- Re-litigating the C.A.3 rung dynamics or the seam (locked; `fix/cax3-spec-metric-bounds` already
  corrects the stale spec metric-bound line).
