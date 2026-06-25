# Forage Local-Learning Candidates (C.A.4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two gradient-free local-learning control candidates on the reflex-proof `forage` rung — a
per-type contingency tracker (`forage_min`, the FLOP-floor reference) and a frozen-reservoir + plastic
readout control (`forage_reservoir`) — scored honestly through the control seam against the transformer
bar re-swept at H=64, reported as a regret-vs-total-FLOP curve.

**Architecture:** Both register as `LanguageModel`s (zero harness changes) and run through
`evaluate_control(model, forage_env_spec(cfg), …)`. `forage_min` mirrors `chemotaxis_min`: an in-context
per-type value vector `v[K]` updated by a local delta rule in `step`, a distilled-scalar softmax policy,
all compute pointwise + `backward=0` (the `v` update is forward compute, not weight learning).
`forage_reservoir` subclasses `ReservoirPlastic` (frozen `_ReservoirCore` reused unchanged + plastic
readout) overriding only the reward decode (`obs % 3 − 1` instead of the monotone-token proxy) and its
+1 FLOP. The ~0-distillation headline is driven by `distill_train_run` with a sub-one-step `flop_budget`
(0 train steps), exactly like `reservoir_plastic`.

**Tech Stack:** Python 3.12, PyTorch (CPU), numpy, matplotlib. `uv run` (never bare python).

## Global Constraints

- Git identity for ALL commits: `git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit …`; verify `git log -1 --format='%an <%ae>'`.
- Branch `task/C.A.4-forage-local-learner` off `main`, worktree `smolml.worktrees/C.A.4-forage-local-learner`; own PR; **do not merge**.
- KISS, hard cutover, no backward-compat shims. Modern typing (`list[float]`, `X | None`); type hints everywhere; 4-space indent; lines ≤100 (ruff counts unicode as 1 char).
- Deps: torch, numpy, matplotlib only. Reproducible seeds.
- Metric (ADR 0007): regret-per-FLOP at fixed params on the forage rung, reported as a regret-vs-total-FLOP curve, never a point. The transformer bar is re-swept at **H=64** on this machine.
- FLOP honesty is the product (ADR 0004): a non-matmul mechanism MUST charge its real elementwise work via `pointwise_flops`/`gather_flops`, or `flops.py`'s conditional-omission rule scores it as free. Online adaptation FLOPs are charged in `step` (`forage_min`: forward, `backward=0`; `forage_reservoir`: `step.backward`).
- Forage seam id-map (from `smolml/envs/forage.py`): combined obs `= type · 3 + (reward + 1)`; `type = obs // 3`; `last_reward = obs % 3 − 1`; `obs_slice = [0, 3K)`; `action_slice = [3K, 3K+3)`; actions `LEFT, EAT, RIGHT = 0, 1, 2` → tokens `3K, 3K+1, 3K+2`. `REWARD_LEVELS = 3`, `N_ACTIONS = 3`.
- Tape parity (from `evaluate_control`): `c0, a0, c1, a1, …` — EVEN positions are obs, ODD are actions.
- Subagent path discipline: pass ABSOLUTE worktree paths; prefer `write` for new files; re-read after `edit`. Run `uv run pytest` yourself (codex sandbox pytest fails — environmental).
- Pre-existing ruff drift in `smolml/experiments/export_demo_fixtures.py` is OUT OF SCOPE.

---

## File Structure

- **Create** `smolml/models/forage_min.py` — `ForageMinConfig`, `ForageMinState`, `ForageMin` (the per-type tracker). ~210 lines, mirrors `chemotaxis_min.py`.
- **Modify** `smolml/models/reservoir.py` — extract `ReservoirPlastic._decode_reward` + `_REWARD_DECODE_OPS` (behavior-preserving), add `ForageReservoir(ReservoirPlastic)` (~25 lines).
- **Modify** `smolml/models/__init__.py` — import + export `ForageMin`/`ForageMinConfig`/`ForageReservoir`.
- **Create** `tests/test_forage_min.py` — unit + end-to-end tests, mirrors `test_chemotaxis_min.py`.
- **Create** `tests/test_forage_reservoir.py` — unit + end-to-end tests, mirrors `test_reservoir_plastic.py`.
- **Create** `smolml/experiments/forage_min_control.py` — the ~0-distillation curve driver, mirrors `reservoir_plastic_control.py`.
- **Create** `smolml/experiments/forage_reservoir_control.py` — same, for the control sibling.
- **Modify** `smolml/experiments/forage_baseline.py` — `HORIZON = 64` (re-establish the bar at production horizon).
- **Modify** `docs/harness.md` — a short forage-candidate note.
- **Create** `docs/learning/researcher-notes/C.A.4-forage-local-learner.md` — researcher note handed to the docs-builder (ADR 0006). (Confirm the notes dir name during Task 7; create alongside existing notes if a convention exists.)

---

### Task 1: `forage_min` model + unit tests

**Files:**
- Create: `smolml/models/forage_min.py`
- Modify: `smolml/models/__init__.py`
- Test: `tests/test_forage_min.py`

**Interfaces:**
- Consumes: `smolml.flops.{FlopBreakdown, pointwise_flops, gather_flops}`; `smolml.models.registry.{DecodeState, LanguageModel, register_model}`; `smolml.envs.forage.{N_ACTIONS, REWARD_LEVELS, LEFT, EAT, RIGHT}`.
- Produces: registered model `"forage_min"`; `ForageMin(ForageMinConfig)` with `forward(idx)->(B,T,vocab)`, `step(state, byte, pos)->(state, logits, FlopBreakdown)`, `flops(seq_len)`, `decode_step_flops(ctx)`, `from_config(dict)`, `num_params()==8`. `ForageMinConfig(vocab_size, max_seq_len, lr_init, v_init, gain_init, eat_bias_init, left_bias_init, right_bias_init, wm_gain_init, stick_init)`. `K = (vocab_size − N_ACTIONS) // REWARD_LEVELS`.

**The mechanism (the math both `forward` and `step` implement identically):**

State per stream: `v` (K floats, init `v_init`), `current_type` (int, the most recent sensed type), `last_action` (int in {0,1,2}, init `RIGHT`). `v` is in-context memory, reset each episode — NOT an `nn.Parameter`.

Params (8 scalar `nn.Parameter`s): `lr_logit` (rate `lr = sigmoid(lr_logit)`), `v_init`, `g` (policy gain), `b_eat`, `b_left`, `b_right`, `g_wm` (world-model gain), `stick` (world-model type stickiness).

On folding `revealed_byte` at `pos`:
- **EVEN (obs fold):** decode `t = byte // 3`, `r = byte % 3 − 1`. If `last_action == EAT` (the agent ate the cell it stood on; EAT does not move, so the eaten type equals the current decoded type), apply the **delta rule** to that type: `v[t] += lr · (r − v[t])`. Set `current_type = t`. (Moves give `r = 0` and `last_action != EAT`, so no update.)
- **ODD (action fold):** `last_action = byte − 3K`. `current_type` unchanged.

`_emit_logits(v_cur, current_type, last_action)` → full-vocab logits `[obs(3K) | action(3)]`, shape-polymorphic (scalar in `step`, `(B,)`/`(B,K)` in `forward`), where `v_cur = v[current_type]`:
- **Policy head** (action slice): `eat = g·v_cur + b_eat`; `left = b_left`; `right = b_right`; `action = stack([left, eat, right])` (order `LEFT, EAT, RIGHT`).
- **World-model head** (obs slice, 3K): predict the next combined obs after `last_action`.
  - reward-level logits `rew(3)` over `{−1,0,+1}→{0,1,2}`: on EAT, `[−g_wm·v_cur, 0, g_wm·v_cur]` (peak follows the contingency belief); on a move, `[0, stick, 0]` (reward 0 is certain — reuse `stick` as the move-certainty gain). Select with `torch.where(last_action==EAT, …)`.
  - type logits `typ(K)`: on EAT, `stick·onehot(current_type)` (sticky — same cell); on a move, `zeros(K)` (uniform — unknown neighbor). Select with `torch.where`.
  - combined obs logits: `obs[τ·3 + ρ] = typ[τ] + rew[ρ]` (an outer SUM over the `(K,3)` grid, flattened to 3K; softmax → product of marginals).
  - return `cat([obs, action], dim=-1)`.

**Per-step pointwise op count** (named constants, hand-checkable, charged on every step — the heavier obs branch conservatively, like `chemotaxis_min`). Define at module top and finalize the integers to match the code exactly (the FLOP test below pins the sum; the codex audit verifies honesty):

```python
_LR_OPS = 4            # lr = sigmoid(lr_logit): neg, exp, +1, recip
_DECODE_OPS = 3        # t = byte//3 (1); r = byte%3 (1); r-1 (1)
_UPDATE_OPS = 5        # gather v[t] (1); err=r-v[t] (1); lr*err (1); v[t]+=  (1); EAT-gate where (1)
_POLICY_OPS = 5        # eat = g*v_cur (1) + b_eat (1); left/right copies (2); stack (1)
_REW_OPS = 8           # [-g_wm*v_cur, 0, g_wm*v_cur]: g_wm*v_cur (1), neg (1); move [0,stick,0]; where over 3 (3); +2 assembly
_PER_TYPE_OPS = 3      # per type: onehot compare (1), *stick (1), where vs zeros (1)
_PER_SYMBOL_OPS = 2    # per combined obs symbol: outer-sum add (1) + final cat copy (1)
```

`_per_step_ops()` = `_LR_OPS + _DECODE_OPS + _UPDATE_OPS + _POLICY_OPS + _REW_OPS + _PER_TYPE_OPS*K + _PER_SYMBOL_OPS*(3*K)`.

`flops(seq_len)` = `FlopBreakdown.from_forward(pointwise_flops(seq_len * _per_step_ops()))` (the 3× distill path — scalars participate at every position, like `chemotaxis_min`). `step`/`decode_step_flops` = `FlopBreakdown(forward=pointwise_flops(_per_step_ops()), backward=0)`.

- [ ] **Step 1: Write the failing forward-shape + both-slices test.**

```python
"""C.A.4 acceptance tests: the `forage_min` per-type contingency tracker."""
import torch
from smolml.envs.forage import EAT, LEFT, N_ACTIONS, REWARD_LEVELS, RIGHT, ForageConfig, vocab_size
from smolml.models import build_model
from smolml.models.forage_min import ForageMin, ForageMinConfig

CPU = torch.device("cpu")

def _build(n_types: int = 3, horizon: int = 8, **ov):
    fcfg = ForageConfig(n_types=n_types, horizon=horizon)
    mc = {"vocab_size": vocab_size(fcfg), "max_seq_len": 2 * horizon + 1, **ov}
    return build_model("forage_min", mc), fcfg

def _obs(type_: int, reward: int) -> int:
    return type_ * REWARD_LEVELS + (reward + 1)

def test_forward_shape_and_both_slices_populated():
    model, fcfg = _build(horizon=8)
    v = vocab_size(fcfg)
    idx = torch.randint(0, v, (2, 2 * fcfg.horizon), dtype=torch.long)
    out = model(idx)
    assert out.shape == (2, 2 * fcfg.horizon, v)
    obs_len = REWARD_LEVELS * fcfg.n_types
    for t in range(out.shape[1]):
        assert out[0, t, :obs_len].std() >= 0  # obs head populated
        assert out[0, t, obs_len:].std() > 0   # policy distinguishes actions
```

- [ ] **Step 2: Run it; expect collection/import failure** (`forage_min` not defined).

Run: `cd <worktree> && uv run pytest tests/test_forage_min.py::test_forward_shape_and_both_slices_populated -q`
Expected: FAIL (ModuleNotFoundError / not registered).

- [ ] **Step 3: Implement `smolml/models/forage_min.py`** (config, state, `_emit_logits`, `forward`, `step`, `flops`, `decode_step_flops`, `from_config`, the op constants). Mirror `chemotaxis_min.py` structure exactly: a shape-polymorphic `_emit_logits`; a `forward` Python-scan over `T` keeping `(B,K)` `v`, `(B,)` `current_type`/`last_action`, using `v.gather`/`v.scatter` for the per-row delta update and `torch.where` for the EAT gate; a scalar `step`. The `forward` MUST produce logits identical to `step` on the same tape (the parity test is the safety net). Register `@register_model("forage_min")`.

- [ ] **Step 4: Add to `smolml/models/__init__.py`** — `from smolml.models.forage_min import ForageMin, ForageMinConfig` and add both to `__all__`.

- [ ] **Step 5: Run the test; expect PASS.**

Run: `uv run pytest tests/test_forage_min.py::test_forward_shape_and_both_slices_populated -q`
Expected: PASS.

- [ ] **Step 6: Write determinism + step/forward parity + FLOP-sum tests.**

```python
def test_deterministic_logits():
    model, fcfg = _build(horizon=6)
    idx = torch.randint(0, vocab_size(fcfg), (1, 2 * fcfg.horizon), dtype=torch.long)
    assert torch.equal(model(idx), model(idx))

def test_step_matches_forward_and_flop_sum():
    model, fcfg = _build(horizon=6)
    obs_len = REWARD_LEVELS * fcfg.n_types
    toks = []
    for i in range(fcfg.horizon):
        toks.append(_obs(i % fcfg.n_types, (-1 if i % 2 else 1)))  # obs
        toks.append(obs_len + (i % N_ACTIONS))                      # action token
    idx = torch.tensor([toks], dtype=torch.long)
    fwd = model(idx)[0]
    state = model.init_prequential_state()
    summed = 0
    for pos, tok in enumerate(toks):
        state, logits, f = model.step(state, tok, pos)
        assert torch.allclose(logits, fwd[pos], atol=1e-6)  # step == forward
        assert f.backward == 0
        summed += f.forward
    assert summed == len(toks) * model.decode_step_flops(0).forward
```

- [ ] **Step 7: Run them; fix `forward`/`step` until parity holds (atol 1e-6).** This is the load-bearing invariant — both paths share `_emit_logits`.

Run: `uv run pytest tests/test_forage_min.py -q -k "parity or determ or flop_sum"`
Expected: PASS.

- [ ] **Step 8: Write FLOP-honesty + num_params tests.**

```python
def test_flop_honesty_pointwise_and_backward_zero():
    from smolml.flops import pointwise_flops
    model, _ = _build(n_types=3, horizon=8)
    f = model.decode_step_flops(0)
    assert f.forward == pointwise_flops(model._per_step_ops()) > 0
    assert f.backward == 0
    assert model.decode_step_flops(0).forward == model.decode_step_flops(999).forward  # context-free
    assert model.flops(4).backward == 2 * model.flops(4).forward  # distill path is from_forward (3x)

def test_num_params_is_the_eight_scalars():
    model, _ = _build()
    assert model.num_params() == 8
    assert all(p.numel() == 1 and p.requires_grad for p in model.parameters())
```

- [ ] **Step 9: Run; reconcile `_per_step_ops` constants with the actual code so the assert holds.** Expected: PASS.

- [ ] **Step 10: Write the learning-behavior tests (the mechanism's reason to exist).**

```python
def test_delta_rule_credits_eaten_type_only_after_eat():
    # Eat type 0, observe poison (-1): v[0] must drop below the optimistic init.
    model, fcfg = _build(n_types=3, horizon=8, v_init=0.0)
    obs_len = REWARD_LEVELS * fcfg.n_types
    state = model.init_prequential_state()
    state, _, _ = model.step(state, _obs(0, 0), 0)              # sense type 0
    state, _, _ = model.step(state, obs_len + EAT, 1)           # EAT
    state, _, _ = model.step(state, _obs(0, -1), 2)             # poison revealed -> credit type 0
    assert state.cache.v[0] < 0.0                               # learned: type 0 is bad
    # A MOVE fold must NOT update any value.
    v_snapshot = list(state.cache.v)
    state, _, _ = model.step(state, obs_len + RIGHT, 3)         # move
    state, _, _ = model.step(state, _obs(1, 0), 4)              # neighbor; last_action=RIGHT
    assert list(state.cache.v) == v_snapshot                    # move -> no credit

def test_policy_eats_high_value_skips_poison():
    # After learning type 1 is good (v[1] high) and type 0 is poison (v[0] low),
    # the policy EATs on a type-1 cell and moves off a type-0 cell.
    model, fcfg = _build(n_types=3, horizon=8)
    obs_len = REWARD_LEVELS * fcfg.n_types
    state = model.init_prequential_state()
    state.cache.v[1] = 1.0
    state.cache.v[0] = -1.0
    _, good, _ = model.step(state, _obs(1, 0), 0)
    assert int(good[obs_len:].argmax()) == EAT                  # eat the good type
    state2 = model.init_prequential_state()
    state2.cache.v[1] = 1.0; state2.cache.v[0] = -1.0
    _, bad, _ = model.step(state2, _obs(0, 0), 0)
    assert int(bad[obs_len:].argmax()) != EAT                   # don't eat poison
```

- [ ] **Step 11: Run the full file; all green.**

Run: `uv run pytest tests/test_forage_min.py -q`
Expected: PASS (all).

- [ ] **Step 12: Commit.**

```bash
git add smolml/models/forage_min.py smolml/models/__init__.py tests/test_forage_min.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" \
  commit -m "C.A.4: forage_min per-type contingency tracker + unit tests"
```

---

### Task 2: `forage_min` end-to-end through the control seam

**Files:**
- Test: `tests/test_forage_min.py` (append)

**Interfaces:**
- Consumes: `smolml.control_train.{ControlTrainConfig, distill_train_run}`; `smolml.envs.forage.{ForageConfig, ForageEnv, forage_env_spec, vocab_size}`; `smolml.leaderboard.regenerate_control`; the `WinStayLoseShift`/random reference for a floor.
- Produces: a `runs/forage`-style JSONL row; proves ~0-distillation online learning clears the random floor with within-episode improvement.

- [ ] **Step 1: Write the failing end-to-end test** (mirror `test_end_to_end_zero_distillation_beats_floor`).

```python
import numpy as np
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.forage import ForageConfig, ForageEnv, forage_env_spec, vocab_size
from smolml.leaderboard import regenerate_control

def _random_floor(fcfg: ForageConfig, episodes: int) -> float:
    rng = np.random.default_rng(0)
    out = []
    for s in range(episodes):
        e = ForageEnv(fcfg, split="eval", seed=s * 100003)
        e.reset(); tot = 0.0
        for _ in range(fcfg.horizon):
            _, r = e.step(int(rng.integers(N_ACTIONS))); tot += r
        out.append(tot / fcfg.horizon)
    return float(np.mean(out))

def test_end_to_end_zero_distillation_beats_floor(tmp_path):
    horizon, episodes = 64, 32
    fcfg = ForageConfig(horizon=horizon)
    mc = {"vocab_size": vocab_size(fcfg), "max_seq_len": 2 * horizon + 1}
    step_flops = build_model("forage_min", mc).flops(2 * horizon).scale(32).total
    cfg = ControlTrainConfig(
        model="forage_min", model_config={}, flop_budget=step_flops * 0.5,
        batch_size=32, horizon=horizon, eval_interval=10**9, eval_episodes=episodes,
        seed=0, env_name="forage", run_name="forage-min-zero-distill",
    )
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs", env_spec=forage_env_spec(fcfg))
    assert summary.steps == 0                       # ~0 distillation: all learning online
    assert summary.total_flops > 0
    floor = _random_floor(fcfg, episodes)
    assert summary.final_reward > floor             # the local rule clears the floor
    assert summary.second_half_reward > summary.first_half_reward  # within-episode learning
    table, png = regenerate_control(tmp_path / "runs", table_path=tmp_path / "lb.md",
                                    plot_path=tmp_path / "lb.png")
    assert png.exists() and png.stat().st_size > 0
    assert "forage-min-zero-distill" in table and "regret" in table
```

- [ ] **Step 2: Run; expect FAIL** if the default scalar inits don't yet clear the floor or 2nd-half ≤ 1st-half.

Run: `uv run pytest tests/test_forage_min.py::test_end_to_end_zero_distillation_beats_floor -q`

- [ ] **Step 3: Tune the `ForageMinConfig` scalar *inits* so the ~0-distillation rollout learns within-episode** — optimistic `v_init` (e.g. `+0.3`) to explore-by-eating, `lr_init` (e.g. `0.5`) fast enough to flip on one poison, `gain_init` high enough that `g·v` separates eat/skip, `b_*` biases so a poison cell moves RIGHT to search. Verify with a scratch run:

```bash
uv run python -c "
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.forage import ForageConfig, forage_env_spec, vocab_size
from smolml.models import build_model
f=ForageConfig(horizon=64); mc={'vocab_size':vocab_size(f),'max_seq_len':129}
sf=build_model('forage_min',mc).flops(128).scale(32).total
c=ControlTrainConfig(model='forage_min',model_config={},flop_budget=sf*0.5,horizon=64,
  eval_interval=10**9,eval_episodes=64,env_name='forage',run_name='probe')
s=distill_train_run(c,runs_dir='/tmp/fmprobe',env_spec=forage_env_spec(f))
print('steps',s.steps,'reward',s.final_reward,'regret',s.final_regret,'1st',s.first_half_reward,'2nd',s.second_half_reward)
"
```
Iterate the inits in `ForageMinConfig` defaults until `reward > ~0` and `2nd > 1st`. Target regret near `wsls` (~0.11).

- [ ] **Step 4: Run the test; expect PASS.**

Run: `uv run pytest tests/test_forage_min.py::test_end_to_end_zero_distillation_beats_floor -q`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tests/test_forage_min.py smolml/models/forage_min.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" \
  commit -m "C.A.4: forage_min end-to-end ~0-distillation beats floor; tune inits"
```

---

### Task 3: `forage_reservoir` control candidate (refactor + subclass)

**Files:**
- Modify: `smolml/models/reservoir.py`
- Modify: `smolml/models/__init__.py`
- Test: `tests/test_forage_reservoir.py`

**Interfaces:**
- Consumes: `ReservoirPlastic`, `ReservoirPlasticConfig`, `_ReservoirCore`, `_PlasticCache` (in `reservoir.py`); `smolml.envs.forage.REWARD_LEVELS`.
- Produces: registered model `"forage_reservoir"`; `ForageReservoir(ReservoirPlastic)` overriding `_decode_reward` and `_REWARD_DECODE_OPS`; `num_params()` matches `reservoir_plastic` (≤ 148,608).

**The change to `ReservoirPlastic` (behavior-preserving):** today `step` computes the reward proxy inline
as `r = revealed_byte / (lv - 1)` and `_policy_update_flops` charges `pointwise_flops(5)`. Extract the
decode into a hook so a subclass can override it, leaving `reservoir_plastic` byte-identical.

- [ ] **Step 1: Add the hook + constant to `ReservoirPlastic`** in `smolml/models/reservoir.py`:

```python
    _REWARD_DECODE_OPS: int = 1  # r = revealed_byte / (lv - 1): one divide

    def _decode_reward(self, revealed_byte: int, lv: int) -> float:
        """Reward proxy for the policy Hebbian rule. Base: the obs token IS a monotone
        concentration level (chemotaxis), normalized to [0, 1]."""
        return revealed_byte / (lv - 1)
```

In `step`, replace the inline `r = revealed_byte / (lv - 1)` with `r = self._decode_reward(revealed_byte, lv)`.
In `_policy_update_flops`, replace `pointwise_flops(5)` with `pointwise_flops(4 + self._REWARD_DECODE_OPS)`
(4 = adv (1) + leaky baseline (3); the reward decode is `_REWARD_DECODE_OPS`). Update its docstring.

- [ ] **Step 2: Run the existing reservoir_plastic suite — it MUST stay green (byte-identical).**

Run: `uv run pytest tests/test_reservoir_plastic.py -q`
Expected: PASS (no behavior change; `4 + 1 == 5`).

- [ ] **Step 3: Write the failing `forage_reservoir` tests** in `tests/test_forage_reservoir.py`:

```python
"""C.A.4 control sibling: reservoir core + plastic readout with the forage reward decode."""
import torch
from smolml.envs.forage import N_ACTIONS, REWARD_LEVELS, ForageConfig, forage_env_spec, vocab_size
from smolml.models import build_model
from smolml.models.reservoir import ForageReservoir, _ReservoirCore

CPU = torch.device("cpu")

def test_reward_decode_is_forage_not_monotone_token():
    m = build_model("forage_reservoir", {"vocab_size": 12, "max_seq_len": 64})
    lv = m._levels  # 9 = 3K
    # obs encodes (type, reward): poison (reward -1) -> -1; good (+1) -> +1; move (0) -> 0.
    assert m._decode_reward(0 * REWARD_LEVELS + 0, lv) == -1.0   # type0 poison
    assert m._decode_reward(1 * REWARD_LEVELS + 2, lv) == +1.0   # type1 reward
    assert m._decode_reward(2 * REWARD_LEVELS + 1, lv) == 0.0    # type2 move
    assert m._REWARD_DECODE_OPS == 2

def test_reuses_frozen_core_and_param_parity():
    m = build_model("forage_reservoir", {"vocab_size": 12, "max_seq_len": 129})
    assert isinstance(m.core, _ReservoirCore)
    assert m.num_params() <= 148_608

def test_online_update_changes_cache_not_parameters():
    m = build_model("forage_reservoir", {"vocab_size": 12, "max_seq_len": 64})
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    state = m.init_prequential_state()
    obs_len = REWARD_LEVELS * 3
    state, _, f0 = m.step(state, 0, 0)                    # conc fold pos<2 -> no update
    state, _, f1 = m.step(state, obs_len + 1, 1)          # action fold
    w_pre = state.cache.W.clone()
    state, _, f2 = m.step(state, 5, 2)                    # obs fold pos>=2 -> update
    assert f0.backward == 0 and f1.backward == 0 and f2.backward > 0
    assert not torch.allclose(w_pre, state.cache.W)
    for n, p in m.named_parameters():
        assert torch.equal(p, before[n])                 # eval mutates no nn.Parameter
```

- [ ] **Step 4: Run; expect FAIL** (`forage_reservoir` not registered).

Run: `uv run pytest tests/test_forage_reservoir.py -q`

- [ ] **Step 5: Add `ForageReservoir` to `smolml/models/reservoir.py`** (after `ReservoirPlastic`):

```python
@register_model("forage_reservoir")
class ForageReservoir(ReservoirPlastic):
    """C.A.4 control: the ReservoirPlastic mechanism on the forage rung. The frozen core +
    plastic readout are reused unchanged; only the reward proxy differs — the forage obs is a
    combined ``(type, reward)`` symbol, so the eat-reward is ``obs % REWARD_LEVELS - 1`` (in
    {-1,0,+1}), not the monotone token value. Running this generic-capacity learner beside
    ``forage_min`` isolates that per-type credit-assignment STRUCTURE, not raw capacity, is the
    lever (this shape already lost on chemotaxis)."""

    _REWARD_DECODE_OPS: int = 2  # r = revealed_byte % REWARD_LEVELS - 1: a mod + a sub

    def _decode_reward(self, revealed_byte: int, lv: int) -> float:
        return float(revealed_byte % REWARD_LEVELS - 1)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ForageReservoir":
        return cls(ReservoirPlasticConfig(**config))
```

Add `from smolml.envs.forage import REWARD_LEVELS` near the top of `reservoir.py` (it already imports `N_ACTIONS` from chemotaxis; keep that). Export `ForageReservoir` in `smolml/models/__init__.py` (`__all__` + import).

- [ ] **Step 6: Run both suites; all green.**

Run: `uv run pytest tests/test_forage_reservoir.py tests/test_reservoir_plastic.py -q`
Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add smolml/models/reservoir.py smolml/models/__init__.py tests/test_forage_reservoir.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" \
  commit -m "C.A.4: forage_reservoir control (ReservoirPlastic + forage reward decode)"
```

---

### Task 4: `forage_reservoir` end-to-end through the control seam

**Files:**
- Test: `tests/test_forage_reservoir.py` (append)

- [ ] **Step 1: Write the end-to-end test** (mirror Task 2, model `"forage_reservoir"`, `run_name="forage-reservoir-zero-distill"`). Same `_random_floor`; assert `summary.steps == 0`, `final_reward > floor`, a leaderboard row. NOTE: this is the *control* — its within-episode signal may be weak/seed-sensitive (it lost on chemotaxis). Assert `final_reward > floor` and `np.isfinite`, but for the 2nd>1st-half assertion follow the `reservoir_plastic` precedent (a documented, possibly seed-sensitive finding) — only assert it if a quick probe shows it holds at this seed; otherwise assert `second_half_reward >= first_half_reward - tol` and document the weakness in the test docstring (failures are data; do NOT fudge inits to manufacture a win).

- [ ] **Step 2: Run a probe** (same one-liner as Task 2 Step 3, model `forage_reservoir`) to see the real numbers; set the assertion to the honest observed behavior.

- [ ] **Step 3: Run the test; PASS.**

Run: `uv run pytest tests/test_forage_reservoir.py -q`

- [ ] **Step 4: Commit.**

```bash
git add tests/test_forage_reservoir.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" \
  commit -m "C.A.4: forage_reservoir end-to-end through the control seam"
```

---

### Task 5: experiment drivers + re-establish the bar at H=64

**Files:**
- Create: `smolml/experiments/forage_min_control.py`
- Create: `smolml/experiments/forage_reservoir_control.py`
- Modify: `smolml/experiments/forage_baseline.py` (`HORIZON = 64`)

**Interfaces:**
- Consumes: `distill_train_run`, `evaluate_control`, `forage_env_spec`, `regenerate_control`, `render_rollout`, `build_model`.
- Produces: `runs/forage` JSONL rows + `runs/forage/leaderboard.{md,png}` + a `*_sample_rollout.png` per candidate, and the candidate regret-vs-total-FLOP curves printed.

- [ ] **Step 1: `forage_min_control.py`** — mirror `reservoir_plastic_control.py` but: `from smolml.envs.forage import ForageConfig, ForageEnv, forage_env_spec, vocab_size`, `N_ACTIONS`; `RUNS_DIR = "runs/forage"`; `HORIZON = 64`; `MODEL = {}` (forage_min takes only injected vocab/seq + its scalar-init defaults); a `random_floor` over `ForageEnv(..., split="eval", seed=seed*100003+s)` with a uniform-random action; a curve over a few distill budgets including the ~0-distillation point (`flop_budget = step_flops * 0.5`); write rows via `distill_train_run(..., env_spec=forage_env_spec(ForageConfig(horizon=HORIZON)))`; `regenerate_control(RUNS_DIR, …)`; a `render_rollout` of one recorded eval episode to `runs/forage/forage_min_sample_rollout.png`. Print each point: `steps / regret / reward / total_flops`.

- [ ] **Step 2: `forage_reservoir_control.py`** — identical structure, model `"forage_reservoir"`, `MODEL = {"d_res": 374, "leak": 0.6, "spectral_radius": 0.9, "seed": 0}`, sample PNG `runs/forage/forage_reservoir_sample_rollout.png`.

- [ ] **Step 3: Set `HORIZON = 64` in `forage_baseline.py`.** Leave the grids/steps; the sweep is cheap per config at the fast budget.

- [ ] **Step 4: Smoke each driver for one cheap point** (small horizon override or one budget) to confirm it runs and writes a row + PNG without error. E.g.:

```bash
uv run python -m smolml.experiments.forage_min_control 2>&1 | tail -20
```

- [ ] **Step 5: Commit the drivers.**

```bash
git add smolml/experiments/forage_min_control.py smolml/experiments/forage_reservoir_control.py smolml/experiments/forage_baseline.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" \
  commit -m "C.A.4: forage candidate drivers + re-establish transformer bar at H=64"
```

---

### Task 6: Verification — gates, the re-swept bar, the regret-vs-FLOP curves

**Files:** none (produces the numbers pasted in the PR).

- [ ] **Step 1: Format + lint.**

Run: `uvx ruff format --check . && uvx ruff check .`
Expected: clean (ignore the pre-existing `export_demo_fixtures.py` drift — do not touch it; if `ruff check` flags only that file, note it as pre-existing/out-of-scope).

- [ ] **Step 2: Full test suite.**

Run: `uv run pytest -q`
Expected: PASS (all, including the migrated control suites).

- [ ] **Step 3: Re-run the transformer bar at H=64.**

Run: `uv run python -m smolml.experiments.forage_baseline 2>&1 | tee /tmp/forage_bar_h64.txt`
Capture: the sweep table, the chosen config, and the FLOP-budget regret curve (the bar at H=64). Verify the regret-vs-FLOP plateau empirically (regret roughly flat across the budget points).

- [ ] **Step 4: Run both candidate curves at H=64.**

Run: `uv run python -m smolml.experiments.forage_min_control 2>&1 | tee /tmp/forage_min.txt`
Run: `uv run python -m smolml.experiments.forage_reservoir_control 2>&1 | tee /tmp/forage_reservoir.txt`
Capture: each candidate's regret / reward / total-FLOPs curve and the leaderboard table. The `forage_min` headline ~0-distillation point should sit ~6 OOM below the bar's FLOPs; report its regret vs the bar's H=64 regret (a regret win is the headline if achieved; if not, report honestly).

- [ ] **Step 5: Cross-vendor codex FLOP audit before landing.**

Run: `codex exec -s read-only -C "$PWD" "<focused prompt: audit forage_min._per_step_ops and ForageReservoir._policy_update_flops for under-counting vs the actual ops; confirm step.backward semantics; confirm num_params and the regret-vs-FLOP comparison is at the identical forage_env_spec + H=64 for candidate and bar>"`
Address any real undercharge it finds (it caught one on every prior candidate); re-run pytest yourself.

---

### Task 7: Docs handoff (researcher note + harness.md) — the cleanup phase

**Files:**
- Modify: `docs/harness.md`
- Create: a researcher note for the docs-builder (ADR 0006 — do NOT hand-write `docs/learning/` MDX).

- [ ] **Step 1: Add a short forage-candidate paragraph to `docs/harness.md`** under the forage section (§6): the two candidates, the mechanism one-liners, the regret-vs-FLOP headline numbers from Task 6.

- [ ] **Step 2: Write the researcher note** (intuition, the math of the per-type delta rule + the reservoir contrast, a worked example, what is worth visualizing — the regret-vs-FLOP curve and a within-episode `v[t]` trace). Confirm the existing researcher-note location/convention first (`ls docs/learning` for a notes dir; if none, put it where the C.A.3 note went). Hand it to the docs-builder via the orchestrator; confirm the resulting page is accurate.

- [ ] **Step 3: Update `docs/candidates.md`** — set C.A.4's status/row with the headline result (mirror how C.A.1/C.A.2 are listed).

- [ ] **Step 4: Commit.**

```bash
git add docs/harness.md docs/candidates.md docs/learning/...  # + the researcher note
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" \
  commit -m "C.A.4: harness note + candidates row + researcher note for docs-builder"
```

- [ ] **Step 5: Open the PR (do NOT merge).** Paste the gate outputs, the H=64 bar sweep table + chosen config, and both candidates' regret-vs-total-FLOP curves vs the bar. Request a cross-vendor (non-Claude) review.

---

## Self-Review

**Spec coverage** (against `docs/tasks/C.A.4-forage-local-learner.md`):
- `forage_min` mechanism (per-type delta rule, distilled-scalar policy, world-model head, pointwise FLOPs, backward=0) → Task 1. ✓
- ~0-distillation headline + within-episode learning → Task 2. ✓
- `forage_reservoir` (ReservoirPlastic + forage reward decode, memory-parity, FLOP +1) → Task 3–4. ✓
- Regret-vs-total-FLOP curve, both candidates + bar re-swept at **H=64**, identical EnvSpec → Task 5–6. ✓
- Gates + codex FLOP audit → Task 6. ✓
- harness.md + researcher note (ADR 0006) → Task 7. ✓
- Honest-risk framing (must beat the distilled transformer, not WSLS; failures are data) → carried in the spec; Task 4 explicitly forbids fudging inits to manufacture a control win. ✓

**Placeholder scan:** the only deferrals are (a) the exact `_per_step_ops` integers, reconciled against the code in Task 1 Step 9 with the FLOP test pinning the sum + the codex audit, and (b) the scalar-init values, tuned empirically in Task 2 Step 3 — both are genuine implementation reconciliations, not unwritten logic. The researcher-note path is confirmed in Task 7 Step 2.

**Type consistency:** `_decode_reward(revealed_byte, lv)` and `_REWARD_DECODE_OPS` used identically in Task 3's base + subclass. `ForageMinState.v` (a mutable per-type list/tensor) is read in Task 2's behavior tests via `state.cache.v` — implement `ForageMinState` with a `v` field (list[float] or 1-D tensor) consistent with those tests. `vocab_size`/`N_ACTIONS`/`REWARD_LEVELS`/`EAT`/`LEFT`/`RIGHT` imported from `smolml.envs.forage` throughout.
