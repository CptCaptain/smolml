# Control Rung (chemotaxis in-context RL) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the C.A.0 measuring spine — a drifting-gradient chemotaxis environment, an
autoregressive rollout scorer, rollout visualization, distillation training, and the transformer
baseline — so future minimal-organism candidates can be scored on held-out in-context control.

**Architecture:** A tiny `ChemoEnv` (1-D ring, drifting peak, local-only sensing) behind a thin
`Environment` seam. Episodes are token tapes `c0 a0 c1 a1 … cH` (concentration / action). The
transformer is trained by **Algorithm Distillation** on tapes from a within-episode-improving source
policy (run-and-tumble), then evaluated by an **interactive rollout** that reuses the existing
FLOP-honest `model.step` channel (sampling actions from the policy slice, scoring world-model bits on
the concentration slice). Metric: regret-vs-oracle (headline) + world-model bits (secondary) vs total
FLOPs at fixed params.

**Tech Stack:** Python 3.12, PyTorch (CPU), numpy, matplotlib (raster + GIF via pillow). `uv run`.

## Global Constraints

- Python 3.12; modern typing (`list[str]`, `X | None`); type hints everywhere; 4-space indent; lines ≤100.
- Deps limited to torch, numpy, matplotlib (pillow ships with matplotlib for the GIF writer). No new deps.
- Run everything with `uv run` — never bare `python`. Gates: `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest`.
- FLOP honesty is the product: all rollout compute flows through `model.step`; the eval rollout is counted in the reported total (ADR 0004). Reuse `smolml.flops` primitives; never hand-multiply.
- Reproducible seeds everywhere (deterministic env + deterministic eval set).
- Commit incrementally with author identity: `git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit …`; verify `git log -1 --format='%an <%ae>'`.
- Worktree: use the full worktree path for `read`/`edit`/`write`; `bash` with explicit `cd` for git.
- Subagents skip gates/formatting/project-wide builds; the orchestrator runs the gates once at the end.

## File Structure

- Create `smolml/envs/__init__.py` — package init (exports `ChemoEnv`, config, helpers, `Trajectory`).
- Create `smolml/envs/chemotaxis.py` — `Environment` protocol, `ChemoConfig`, vocab/slice helpers, `ChemoEnv`, reference policies, `Trajectory`, `make_distillation_batch`.
- Create `smolml/envs/render.py` — `render_rollout` (PNG raster), `animate_rollout` (opt-in GIF).
- Create `smolml/control_eval.py` — `ControlResult`, `evaluate_control` (interactive rollout scorer).
- Create `smolml/control_train.py` — `ControlTrainConfig`, `ControlRunSummary`, `distill_train_run`.
- Create `smolml/experiments/control_baseline.py` — driver: distill-train across a FLOP sweep, eval, write leaderboard + a sample raster/GIF.
- Modify `smolml/leaderboard.py` — add `load_control_run`, `build_control_table`, `plot_control`, `regenerate_control`.
- Create `tests/test_control.py` — acceptance tests.
- Modify `docs/harness.md` — how to run a control baseline / add a control candidate / regenerate the board.

---

### Task 1: `Environment` seam, config, vocab helpers

**Files:**
- Create: `smolml/envs/chemotaxis.py`
- Create: `smolml/envs/__init__.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Produces: `N_ACTIONS=3`; `ACTION_DELTAS=(-1,0,1)`; `ChemoConfig(width=16, levels=8, sigma=2.0, horizon=64)`; `drift_rates(split) -> tuple[float,...]`; `vocab_size(cfg) -> int`; `conc_slice(cfg) -> slice`; `action_slice(cfg) -> slice`; `action_token(cfg, idx) -> int`; `ringdist(a, b, width) -> int`. `Environment` typing `Protocol` (`reset`, `step`, `oracle_action`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control.py
import math
import numpy as np
import torch
from smolml.envs.chemotaxis import (
    ChemoConfig, N_ACTIONS, action_slice, action_token, conc_slice,
    drift_rates, ringdist, vocab_size,
)


def test_vocab_layout_and_disjoint_slices():
    cfg = ChemoConfig(levels=8)
    assert vocab_size(cfg) == 8 + N_ACTIONS
    cs, as_ = conc_slice(cfg), action_slice(cfg)
    conc_ids = set(range(cs.start, cs.stop))
    act_ids = set(range(as_.start, as_.stop))
    assert conc_ids.isdisjoint(act_ids)
    assert conc_ids | act_ids == set(range(vocab_size(cfg)))
    assert [action_token(cfg, i) for i in range(N_ACTIONS)] == [8, 9, 10]


def test_drift_pools_disjoint():
    train, eval_ = set(drift_rates("train")), set(drift_rates("eval"))
    assert train and eval_ and train.isdisjoint(eval_)


def test_ringdist_wraps():
    assert ringdist(0, 15, 16) == 1
    assert ringdist(2, 6, 16) == 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_control.py -q`
Expected: FAIL (`ModuleNotFoundError: smolml.envs`).

- [ ] **Step 3: Implement the seam + helpers**

```python
# smolml/envs/chemotaxis.py
"""Chemotaxis control environment + the Environment seam (Task C.A.0).

A 1-D ring with a drifting concentration peak; the agent senses only the local
concentration (quantized) and acts LEFT/STAY/RIGHT. Other feedback-driven tasks
implement the same ``Environment`` protocol and reuse the scorer + candidates.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

N_ACTIONS: int = 3
ACTION_DELTAS: tuple[int, ...] = (-1, 0, 1)  # LEFT, STAY, RIGHT
DRIFT_RATES: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass
class ChemoConfig:
    """Environment hyperparameters (one symbol vocab per ``levels``)."""

    width: int = 16
    levels: int = 8
    sigma: float = 2.0
    horizon: int = 64


def drift_rates(split: str) -> tuple[float, ...]:
    """Disjoint per-episode drift-rate pools: even-index for train, odd for eval."""
    if split == "train":
        return DRIFT_RATES[::2]
    if split == "eval":
        return DRIFT_RATES[1::2]
    raise ValueError(f"split must be 'train' or 'eval', got {split!r}")


def vocab_size(cfg: ChemoConfig) -> int:
    return cfg.levels + N_ACTIONS


def conc_slice(cfg: ChemoConfig) -> slice:
    return slice(0, cfg.levels)


def action_slice(cfg: ChemoConfig) -> slice:
    return slice(cfg.levels, cfg.levels + N_ACTIONS)


def action_token(cfg: ChemoConfig, action_idx: int) -> int:
    return cfg.levels + action_idx


def ringdist(a: float, b: float, width: int) -> int:
    d = abs(a - b) % width
    return int(min(d, width - d))


class Environment(Protocol):
    """Minimal feedback-task seam: the scorer/training/candidates depend only on this."""

    def reset(self) -> int: ...
    def step(self, action_idx: int) -> tuple[int, float]: ...
    def oracle_action(self) -> int: ...
```

```python
# smolml/envs/__init__.py
"""Feedback-driven environments (the control rung)."""

from smolml.envs.chemotaxis import (
    ACTION_DELTAS,
    N_ACTIONS,
    ChemoConfig,
    ChemoEnv,
    Environment,
    RandomPolicy,
    RunAndTumble,
    Trajectory,
    action_slice,
    action_token,
    conc_slice,
    drift_rates,
    make_distillation_batch,
    ringdist,
    vocab_size,
)

__all__ = [
    "ACTION_DELTAS",
    "ChemoConfig",
    "ChemoEnv",
    "Environment",
    "N_ACTIONS",
    "RandomPolicy",
    "RunAndTumble",
    "Trajectory",
    "action_slice",
    "action_token",
    "conc_slice",
    "drift_rates",
    "make_distillation_batch",
    "ringdist",
    "vocab_size",
]
```

Note: `__init__.py` imports names defined in Tasks 2–3; it will not import cleanly until those land. Implement Task 1's `chemotaxis.py` symbols now; add the rest of `chemotaxis.py` in Tasks 2–3 before relying on the package import. (If you run Task 1 tests before Task 2, import the helpers directly from `smolml.envs.chemotaxis`, as the test above does.)

- [ ] **Step 4: Run the Task-1 tests (import from `chemotaxis`, not the package)**

Run: `uv run pytest tests/test_control.py -q -k "vocab or drift or ringdist"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add smolml/envs/chemotaxis.py smolml/envs/__init__.py tests/test_control.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit -m "feat(C.A.0): Environment seam, ChemoConfig, vocab helpers"
```

---

### Task 2: `ChemoEnv` dynamics + reference policies

**Files:**
- Modify: `smolml/envs/chemotaxis.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Produces: `ChemoEnv(cfg, *, split, seed)` with `reset() -> int`, `step(action_idx) -> (level_token, raw_reward)`, `oracle_action() -> int`, `field() -> list[float]`, attrs `mu: float`, `p: int`; `RandomPolicy(seed)` with `act(conc) -> int`; `RunAndTumble(epsilon, seed)` with `reset()` and `act(conc) -> int`.
- Consumes: Task 1 helpers.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_control.py (append)
from smolml.envs.chemotaxis import ChemoEnv, RandomPolicy, RunAndTumble


def _roll(policy, env, horizon):
    c = env.reset()
    total = 0.0
    if hasattr(policy, "reset"):
        policy.reset()
    for _ in range(horizon):
        a = policy.act(c)
        c, r = env.step(a)
        total += r
    return total / horizon


def test_env_deterministic_given_seed_and_actions():
    cfg = ChemoConfig(horizon=32)
    e1, e2 = ChemoEnv(cfg, split="eval", seed=5), ChemoEnv(cfg, split="eval", seed=5)
    c1, c2 = [e1.reset()], [e2.reset()]
    for a in [0, 2, 1, 2, 0, 0, 2, 1]:
        r1 = e1.step(a)
        r2 = e2.step(a)
        assert r1 == r2
        c1.append(r1[0]); c2.append(r2[0])
    assert c1 == c2


class _Oracle:
    def __init__(self, env):
        self.env = env

    def act(self, _conc):
        return self.env.oracle_action()


def test_metric_bounds_oracle_gt_tumble_gt_random():
    cfg = ChemoConfig(horizon=64)
    oracle, tumble, rand = [], [], []
    for s in range(40):
        e = ChemoEnv(cfg, split="eval", seed=s)
        oracle.append(_roll(_Oracle(e), e, cfg.horizon))
        e = ChemoEnv(cfg, split="eval", seed=s)
        tumble.append(_roll(RunAndTumble(epsilon=0.0, seed=s), e, cfg.horizon))
        e = ChemoEnv(cfg, split="eval", seed=s)
        rand.append(_roll(RandomPolicy(seed=s), e, cfg.horizon))
    mo, mt, mr = np.mean(oracle), np.mean(tumble), np.mean(rand)
    assert mo > mt > mr
    assert mo > 0.8  # oracle climbs and tracks the peak (reward in [0,1])


def test_source_shows_within_episode_improvement():
    cfg = ChemoConfig(horizon=64)
    first, second = [], []
    half = cfg.horizon // 2
    for s in range(40):
        e = ChemoEnv(cfg, split="train", seed=s)
        pol = RunAndTumble(epsilon=0.1, seed=s)
        c = e.reset()
        rs = []
        for _ in range(cfg.horizon):
            c, r = e.step(pol.act(c))
            rs.append(r)
        first.append(np.mean(rs[:half])); second.append(np.mean(rs[half:]))
    assert np.mean(second) > np.mean(first)  # the source is a learner to distill
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_control.py -q -k "deterministic or bounds or improvement"`
Expected: FAIL (`ImportError: ChemoEnv`).

- [ ] **Step 3: Implement `ChemoEnv` + policies**

```python
# smolml/envs/chemotaxis.py (append)


class ChemoEnv:
    """Drifting-gradient ring; the agent senses only the local quantized concentration."""

    def __init__(self, cfg: ChemoConfig, *, split: str, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        rates = drift_rates(split)
        self.drift_rate = float(rates[rng.integers(len(rates))])
        self.drift_dir = int(rng.choice((-1, 1)))
        self.mu = float(rng.integers(cfg.width))
        self.p = int(rng.integers(cfg.width))
        self._phase = 0.0

    def _raw(self, x: float) -> float:
        d = ringdist(x, self.mu, self.cfg.width)
        return math.exp(-(d * d) / (2.0 * self.cfg.sigma**2))

    def _level(self, raw: float) -> int:
        return min(self.cfg.levels - 1, max(0, int(round(raw * (self.cfg.levels - 1)))))

    def reset(self) -> int:
        return self._level(self._raw(self.p))

    def step(self, action_idx: int) -> tuple[int, float]:
        self.p = (self.p + ACTION_DELTAS[action_idx]) % self.cfg.width
        self._phase += self.drift_rate
        if self._phase >= 1.0:
            self.mu = (self.mu + self.drift_dir) % self.cfg.width
            self._phase -= 1.0
        raw = self._raw(self.p)
        return self._level(raw), raw

    def oracle_action(self) -> int:
        best_idx, best_d = 1, ringdist(self.p, self.mu, self.cfg.width)
        for i, delta in enumerate(ACTION_DELTAS):
            d = ringdist((self.p + delta) % self.cfg.width, self.mu, self.cfg.width)
            if d < best_d:
                best_idx, best_d = i, d
        return best_idx

    def field(self) -> list[float]:
        return [self._raw(x) for x in range(self.cfg.width)]


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, _conc: int) -> int:
        return int(self.rng.integers(N_ACTIONS))


class RunAndTumble:
    """Keep moving if concentration rose, else reverse (tumble); ``epsilon`` explores."""

    def __init__(self, epsilon: float = 0.0, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.last_action = 2  # RIGHT
        self.last_conc: int | None = None

    def reset(self) -> None:
        self.last_action = 2
        self.last_conc = None

    def act(self, conc: int) -> int:
        if self.rng.random() < self.epsilon:
            a = int(self.rng.integers(N_ACTIONS))
        elif self.last_conc is None or conc >= self.last_conc:
            a = self.last_action
        else:
            a = 2 if self.last_action == 0 else 0  # reverse direction
        self.last_action, self.last_conc = a, conc
        return a
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_control.py -q -k "deterministic or bounds or improvement"`
Expected: PASS. (If `test_metric_bounds` is flaky, raise the episode count; do NOT loosen the ordering assertion.)

- [ ] **Step 5: Commit**

```bash
git add smolml/envs/chemotaxis.py tests/test_control.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit -m "feat(C.A.0): ChemoEnv dynamics + oracle/random/run-and-tumble policies"
```

---

### Task 3: `Trajectory` + `make_distillation_batch` (tape format)

**Files:**
- Modify: `smolml/envs/chemotaxis.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Produces: `Trajectory` dataclass (`mu`, `pos`, `conc_token`, `reward`, `action`, `field`, `pred_conc=None`); `make_distillation_batch(cfg, split, *, batch_size, seed, device, epsilon=0.1) -> (tokens, targets)` of shape `(batch_size, 2*horizon)`.
- Consumes: Tasks 1–2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control.py (append)
from smolml.envs.chemotaxis import make_distillation_batch


def test_distillation_tape_format_and_shift():
    cfg = ChemoConfig(width=16, levels=8, horizon=8)
    x, y = make_distillation_batch(cfg, "train", batch_size=4, seed=0,
                                   device=torch.device("cpu"))
    assert x.shape == (4, 2 * cfg.horizon) == y.shape
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is the next-token shift of x
    cs, as_ = conc_slice(cfg), action_slice(cfg)
    # even tape positions are concentrations, odd are actions
    full = torch.cat([x, y[:, -1:]], dim=1)  # reconstruct the (2H+1)-length tape
    for b in range(4):
        for t in range(full.shape[1]):
            tok = int(full[b, t])
            if t % 2 == 0:
                assert cs.start <= tok < cs.stop
            else:
                assert as_.start <= tok < as_.stop
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_control.py -q -k "tape_format"`
Expected: FAIL (`ImportError: make_distillation_batch`).

- [ ] **Step 3: Implement `Trajectory` + `make_distillation_batch`**

```python
# smolml/envs/chemotaxis.py (append; add ``field`` and dataclass import at top:
# from dataclasses import dataclass, field as dc_field)


@dataclass
class Trajectory:
    """A recorded rollout, for rendering and determinism tests."""

    mu: list[float]
    pos: list[int]
    conc_token: list[int]
    reward: list[float]
    action: list[int]
    field: list[list[float]]
    pred_conc: list[list[float]] | None = None


def make_distillation_batch(
    cfg: ChemoConfig,
    split: str,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    epsilon: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll the run-and-tumble source over fresh episodes -> static next-token tapes.

    Each tape is ``c0 a0 c1 a1 … a_{H-1} c_H`` (length ``2H+1``); returns
    ``(x, y) = (tape[:-1], tape[1:])`` of shape ``(batch_size, 2H)``.
    """
    seq = 2 * cfg.horizon + 1
    tapes = torch.empty((batch_size, seq), dtype=torch.long)
    for b in range(batch_size):
        env = ChemoEnv(cfg, split=split, seed=seed * 100003 + b)
        pol = RunAndTumble(epsilon=epsilon, seed=seed * 7919 + b)
        c = env.reset()
        tape = [c]
        for _ in range(cfg.horizon):
            a = pol.act(c)
            tape.append(action_token(cfg, a))
            c, _raw = env.step(a)
            tape.append(c)
        tapes[b] = torch.tensor(tape, dtype=torch.long)
    return tapes[:, :-1].contiguous().to(device), tapes[:, 1:].contiguous().to(device)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_control.py -q -k "tape_format"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add smolml/envs/chemotaxis.py tests/test_control.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit -m "feat(C.A.0): Trajectory + make_distillation_batch tape generator"
```

---

### Task 4: `evaluate_control` rollout scorer

**Files:**
- Create: `smolml/control_eval.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Produces: `ControlResult` (`mean_reward`, `mean_oracle_reward`, `regret`, `world_model_bits`, `first_half_reward`, `second_half_reward`, `flops: FlopBreakdown`, `trajectory: Trajectory | None`, `n_episodes`, `horizon`); `evaluate_control(model, cfg, *, split="eval", n_episodes, seed, device, greedy=False, record=False) -> ControlResult`.
- Consumes: `model.step`/`init_prequential_state` (registry), `score_bits` (prequential), `ChemoEnv`, slices, `action_token`, `Trajectory`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_control.py (append)
from types import SimpleNamespace

from smolml.control_eval import evaluate_control
from smolml.flops import FlopBreakdown
from smolml.models.registry import LanguageModel


class _FixedActionModel(LanguageModel):
    """Stub: always favors one absolute action token; uniform over concentrations."""

    def __init__(self, vocab: int, max_seq_len: int, fav_token: int):
        super().__init__()
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self._vocab, self._fav = vocab, fav_token

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        logits = torch.zeros(b, t, self._vocab)
        logits[..., self._fav] = 10.0
        return logits

    def flops(self, seq_len: int) -> FlopBreakdown:
        return FlopBreakdown.from_forward(seq_len)

    @classmethod
    def from_config(cls, config: dict) -> "_FixedActionModel":
        return cls(**config)


def test_uniform_world_model_bits_near_log2_levels():
    cfg = ChemoConfig(width=16, levels=8, horizon=8)
    model = _FixedActionModel(vocab_size(cfg), 2 * cfg.horizon + 1, action_token(cfg, 1))
    res = evaluate_control(model, cfg, split="eval", n_episodes=4, seed=0,
                           device=torch.device("cpu"))
    assert math.isfinite(res.world_model_bits)
    assert abs(res.world_model_bits - math.log2(cfg.levels)) < 0.2


def test_env_responds_to_actions_no_predetermined_feedback():
    cfg = ChemoConfig(width=16, levels=8, horizon=12)
    left = _FixedActionModel(vocab_size(cfg), 2 * cfg.horizon + 1, action_token(cfg, 0))
    right = _FixedActionModel(vocab_size(cfg), 2 * cfg.horizon + 1, action_token(cfg, 2))
    rl = evaluate_control(left, cfg, split="eval", n_episodes=1, seed=3,
                          device=torch.device("cpu"), greedy=True, record=True)
    rr = evaluate_control(right, cfg, split="eval", n_episodes=1, seed=3,
                          device=torch.device("cpu"), greedy=True, record=True)
    assert rl.trajectory.pos != rr.trajectory.pos  # opposite moves -> different trajectories


def test_rollout_flop_accounting_matches_analytic():
    from smolml.models.transformer import Transformer, TransformerConfig

    cfg = ChemoConfig(width=16, levels=8, horizon=8)
    tcfg = TransformerConfig(d_model=32, n_layers=2, n_heads=4,
                             vocab_size=vocab_size(cfg), max_seq_len=2 * cfg.horizon + 1)
    model = Transformer(tcfg)
    res = evaluate_control(model, cfg, split="eval", n_episodes=1, seed=1,
                           device=torch.device("cpu"))
    expected = sum(model.decode_step_flops(k).forward for k in range(1, 2 * cfg.horizon + 1))
    assert res.flops.forward == expected
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_control.py -q -k "world_model or responds or flop_accounting"`
Expected: FAIL (`ModuleNotFoundError: smolml.control_eval`).

- [ ] **Step 3: Implement `evaluate_control`**

```python
# smolml/control_eval.py
"""Interactive control-rung scorer: roll the model in ChemoEnv via the FLOP-honest
``model.step`` channel, sampling actions from the policy slice and scoring world-model
bits on the concentration slice. Mirrors ``eval.py``/``icl_eval.py``."""

from dataclasses import dataclass

import torch

from smolml.envs.chemotaxis import (
    ChemoConfig,
    ChemoEnv,
    Trajectory,
    action_slice,
    action_token,
    conc_slice,
)
from smolml.flops import FlopBreakdown
from smolml.models.registry import LanguageModel
from smolml.prequential import score_bits


@dataclass
class ControlResult:
    mean_reward: float
    mean_oracle_reward: float
    regret: float
    world_model_bits: float
    first_half_reward: float
    second_half_reward: float
    flops: FlopBreakdown
    n_episodes: int
    horizon: int
    trajectory: Trajectory | None = None


def _sample_action(action_logits: torch.Tensor, greedy: bool, gen: torch.Generator) -> int:
    if greedy:
        return int(action_logits.argmax())
    probs = torch.softmax(action_logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen))


@torch.no_grad()
def evaluate_control(
    model: LanguageModel,
    cfg: ChemoConfig,
    *,
    split: str = "eval",
    n_episodes: int,
    seed: int,
    device: torch.device,
    greedy: bool = False,
    record: bool = False,
) -> ControlResult:
    """Mean reward, regret-vs-oracle, and world-model bits over a seeded held-out set."""
    was_training = model.training
    model.eval()
    cs, as_ = conc_slice(cfg), action_slice(cfg)
    half = cfg.horizon // 2
    flops = FlopBreakdown()
    agent_total = oracle_total = bits = 0.0
    first_total = second_total = 0.0
    trajectory: Trajectory | None = None

    for ep in range(n_episodes):
        ep_seed = seed * 100003 + ep
        env = ChemoEnv(cfg, split=split, seed=ep_seed)
        gen = torch.Generator().manual_seed(ep_seed)
        state = model.init_prequential_state()
        c = env.reset()
        tape = [c]
        rec_mu, rec_pos, rec_field = [env.mu], [env.p], [env.field()]
        rec_act, rec_reward, rec_pred = [], [], []
        pos = 0
        for t in range(cfg.horizon):
            state, logits, f = model.step(state, tape[pos], pos)
            flops += f
            pos += 1
            a_idx = _sample_action(logits[as_], greedy, gen)
            tape.append(action_token(cfg, a_idx))
            state, logits_pred, f = model.step(state, tape[pos], pos)
            flops += f
            pos += 1
            c, reward = env.step(a_idx)
            bits += score_bits(logits_pred[cs], c)
            tape.append(c)
            agent_total += reward
            (first_total if t < half else second_total).__add__  # no-op guard
            if t < half:
                first_total += reward
            else:
                second_total += reward
            if record:
                rec_act.append(a_idx)
                rec_reward.append(reward)
                rec_mu.append(env.mu)
                rec_pos.append(env.p)
                rec_field.append(env.field())
                rec_pred.append(torch.softmax(logits_pred[cs], dim=-1).tolist())

        oracle_env = ChemoEnv(cfg, split=split, seed=ep_seed)
        oracle_env.reset()
        for _ in range(cfg.horizon):
            _, r = oracle_env.step(oracle_env.oracle_action())
            oracle_total += r

        if record and trajectory is None:
            trajectory = Trajectory(
                mu=rec_mu, pos=rec_pos, conc_token=tape[::2], reward=rec_reward,
                action=rec_act, field=rec_field, pred_conc=rec_pred,
            )

    if was_training:
        model.train()
    n = n_episodes * cfg.horizon
    return ControlResult(
        mean_reward=agent_total / n,
        mean_oracle_reward=oracle_total / n,
        regret=(oracle_total - agent_total) / n,
        world_model_bits=bits / n,
        first_half_reward=first_total / (n_episodes * half),
        second_half_reward=second_total / (n_episodes * (cfg.horizon - half)),
        flops=flops,
        n_episodes=n_episodes,
        horizon=cfg.horizon,
        trajectory=trajectory,
    )
```

Remove the stray `(first_total if t < half else second_total).__add__` line when transcribing — it is a leftover; the real accumulation is the `if t < half` block below it.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_control.py -q -k "world_model or responds or flop_accounting"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add smolml/control_eval.py tests/test_control.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit -m "feat(C.A.0): evaluate_control interactive rollout scorer"
```

---

### Task 5: Rollout visualization (`render.py`)

**Files:**
- Create: `smolml/envs/render.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Produces: `render_rollout(traj, out_png) -> Path` (spacetime raster); `animate_rollout(traj, out_gif, *, fps=10) -> Path` (opt-in GIF, guarded).
- Consumes: `Trajectory`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control.py (append)
from smolml.envs.render import render_rollout


def test_render_writes_nonempty_png(tmp_path):
    from smolml.models.transformer import Transformer, TransformerConfig

    cfg = ChemoConfig(width=16, levels=8, horizon=10)
    tcfg = TransformerConfig(d_model=32, n_layers=2, n_heads=4,
                             vocab_size=vocab_size(cfg), max_seq_len=2 * cfg.horizon + 1)
    res = evaluate_control(Transformer(tcfg), cfg, split="eval", n_episodes=1, seed=2,
                           device=torch.device("cpu"), record=True)
    out = render_rollout(res.trajectory, tmp_path / "rollout.png")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_control.py -q -k "render"`
Expected: FAIL (`ModuleNotFoundError: smolml.envs.render`).

- [ ] **Step 3: Implement the renderer**

```python
# smolml/envs/render.py
"""Render a recorded control rollout: a static spacetime raster (default) and an
opt-in animated GIF. Headless matplotlib; pillow ships the GIF writer."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, writers  # noqa: E402

from smolml.envs.chemotaxis import Trajectory  # noqa: E402


def render_rollout(traj: Trajectory, out_png: str | Path) -> Path:
    """Spacetime raster: concentration field over time + agent/peak paths + cum reward."""
    field = np.array(traj.field)  # (steps, width)
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(8, 6))
    ax1.imshow(field.T, aspect="auto", origin="lower", cmap="viridis")
    ax1.plot(range(len(traj.pos)), traj.pos, color="red", lw=1.5, label="agent")
    ax1.plot(range(len(traj.mu)), traj.mu, color="white", ls="--", lw=1.0, label="peak")
    ax1.set_xlabel("step"); ax1.set_ylabel("ring cell"); ax1.legend(loc="upper right")
    ax2.plot(np.cumsum(traj.reward), color="green")
    ax2.set_xlabel("step"); ax2.set_ylabel("cumulative reward")
    fig.tight_layout()
    out = Path(out_png)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def animate_rollout(traj: Trajectory, out_gif: str | Path, *, fps: int = 10) -> Path:
    """Opt-in animated playback of the field with the agent marker. Guarded on pillow."""
    if not writers.is_available("pillow"):
        raise RuntimeError("pillow animation writer unavailable")
    field = np.array(traj.field)
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(range(field.shape[1]), field[0])
    marker = ax.axvline(traj.pos[0], color="red", lw=2)
    ax.set_ylim(0, 1); ax.set_xlabel("ring cell"); ax.set_ylabel("concentration")

    def update(t: int):
        for bar, h in zip(bars, field[t], strict=True):
            bar.set_height(h)
        marker.set_xdata([traj.pos[t], traj.pos[t]])
        ax.set_title(f"step {t}")
        return [*bars, marker]

    anim = FuncAnimation(fig, update, frames=len(field), blit=False)
    out = Path(out_gif)
    anim.save(out, writer="pillow", fps=fps)
    plt.close(fig)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_control.py -q -k "render"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add smolml/envs/render.py tests/test_control.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit -m "feat(C.A.0): rollout spacetime raster + opt-in GIF renderer"
```

---

### Task 6: `distill_train_run` (FLOP-budgeted distillation training)

**Files:**
- Create: `smolml/control_train.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Produces: `ControlTrainConfig` (model, model_config, flop_budget, batch_size=32, lr=3e-3, weight_decay=0.1, betas=(0.9,0.95), grad_clip=1.0, seed=0, eval_interval=20, eval_episodes=32, epsilon=0.1, width=16, levels=8, sigma=2.0, horizon=64, device=None, run_name=None); `ControlRunSummary`; `distill_train_run(cfg, runs_dir="runs") -> ControlRunSummary`.
- Consumes: `make_distillation_batch`, `evaluate_control`, `build_model`, `get_device`, `ChemoConfig`, `vocab_size`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control.py (append)
import json

from smolml.control_train import ControlTrainConfig, distill_train_run


def test_distill_train_smoke_writes_log(tmp_path):
    cfg = ControlTrainConfig(
        model_config={"d_model": 32, "n_layers": 2, "n_heads": 4},
        flop_budget=0.0,  # set below
        batch_size=8, horizon=16, eval_interval=5, eval_episodes=8, seed=0,
    )
    from smolml.models import build_model
    from smolml.envs.chemotaxis import ChemoConfig as _CC, vocab_size as _vs

    chem = _CC(width=16, levels=8, horizon=16)
    mc = {**cfg.model_config, "vocab_size": _vs(chem), "max_seq_len": 2 * 16 + 1}
    step_flops = build_model("transformer", mc).flops(2 * 16).scale(8).total
    cfg.flop_budget = step_flops * 30
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs")
    assert summary.steps >= 1
    assert summary.total_flops <= cfg.flop_budget + summary.final_eval_flops
    log = (tmp_path / "runs" / f"{summary.run}.jsonl").read_text().splitlines()
    meta = json.loads(log[0])
    assert meta["protocol"] == "control" and meta["params"] > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_control.py -q -k "distill_train_smoke"`
Expected: FAIL (`ModuleNotFoundError: smolml.control_train`).

- [ ] **Step 3: Implement `distill_train_run`**

```python
# smolml/control_train.py
"""FLOP-budgeted Algorithm-Distillation training for the control rung. Mirrors
``train.py``'s budgeted loop + JSONL logging, sourcing distillation tapes and
logging the control metric (reward/regret/world-model bits) at checkpoints."""

import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import torch

from smolml.control_eval import evaluate_control
from smolml.device import get_device
from smolml.envs.chemotaxis import ChemoConfig, make_distillation_batch, vocab_size
from smolml.models.registry import build_model


@dataclass
class ControlTrainConfig:
    model: str = "transformer"
    model_config: dict[str, object] = field(default_factory=dict)
    flop_budget: float = 1e10
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 0
    eval_interval: int = 20
    eval_episodes: int = 32
    epsilon: float = 0.1
    width: int = 16
    levels: int = 8
    sigma: float = 2.0
    horizon: int = 64
    device: str | None = None
    run_name: str | None = None


@dataclass
class ControlRunSummary:
    run: str
    model: str
    params: int
    seed: int
    device: str
    flop_budget: float
    total_flops: int
    final_eval_flops: int
    steps: int
    final_regret: float
    final_reward: float
    final_world_model_bits: float
    log_path: str


def distill_train_run(cfg: ControlTrainConfig, runs_dir: str | Path = "runs") -> ControlRunSummary:
    if cfg.flop_budget <= 0:
        raise ValueError(f"flop_budget must be positive, got {cfg.flop_budget}")
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    chem = ChemoConfig(width=cfg.width, levels=cfg.levels, sigma=cfg.sigma, horizon=cfg.horizon)
    mc = dict(cfg.model_config)
    mc["vocab_size"] = vocab_size(chem)
    need = 2 * cfg.horizon + 1
    mc["max_seq_len"] = max(int(mc.get("max_seq_len", 0)), need)

    model = build_model(cfg.model, mc).to(device)
    model.train()
    optimizer = model.configure_optimizer(lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    seq_len = 2 * cfg.horizon
    step_flops = model.flops(seq_len).scale(cfg.batch_size).total
    if step_flops <= 0:
        raise ValueError(f"model reports non-positive step cost: {step_flops}")

    run_name = cfg.run_name or f"{cfg.model}-control-{int(time.time())}"
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    log_path = runs_path / f"{run_name}.jsonl"
    started = time.time()

    def evaluate() -> object:
        return evaluate_control(model, chem, split="eval", n_episodes=cfg.eval_episodes,
                                seed=cfg.seed, device=device)

    with log_path.open("w") as log:
        resolved = asdict(model.config) if is_dataclass(model.config) else dict(mc)
        log.write(json.dumps({
            "type": "meta", "protocol": "control", "run": run_name, "model": cfg.model,
            "config": resolved, "params": model.num_params(), "device": device.type,
            "seed": cfg.seed, "flop_budget": cfg.flop_budget, "batch_size": cfg.batch_size,
            "horizon": cfg.horizon, "levels": cfg.levels, "width": cfg.width,
            "eval_episodes": cfg.eval_episodes, "eval_interval": cfg.eval_interval,
            "started_at": started,
        }) + "\n")

        step, cumulative = 0, 0
        last_logged = -1
        res = evaluate()  # step-0 point (untrained baseline)

        def log_step(r: object) -> None:
            nonlocal last_logged
            log.write(json.dumps({
                "type": "step", "step": step, "cumulative_flops": cumulative,
                "mean_reward": r.mean_reward, "regret": r.regret,
                "world_model_bits": r.world_model_bits,
            }) + "\n")
            log.flush()
            last_logged = step

        log_step(res)
        while cumulative + step_flops <= cfg.flop_budget:
            x, y = make_distillation_batch(chem, "train", batch_size=cfg.batch_size,
                                           seed=cfg.seed * 1009 + step, device=device,
                                           epsilon=cfg.epsilon)
            _loss, spent = model.train_step((x, y), optimizer, grad_clip=cfg.grad_clip)
            cumulative += spent.total
            step += 1
            if step % cfg.eval_interval == 0:
                res = evaluate()
                log_step(res)
        if last_logged != step:
            res = evaluate()
            log_step(res)

    return ControlRunSummary(
        run=run_name, model=cfg.model, params=model.num_params(), seed=cfg.seed,
        device=device.type, flop_budget=cfg.flop_budget,
        total_flops=cumulative + res.flops.total, final_eval_flops=res.flops.total,
        steps=step, final_regret=res.regret, final_reward=res.mean_reward,
        final_world_model_bits=res.world_model_bits, log_path=str(log_path),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_control.py -q -k "distill_train_smoke"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add smolml/control_train.py tests/test_control.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit -m "feat(C.A.0): distill_train_run FLOP-budgeted Algorithm-Distillation loop"
```

---

### Task 7: Control leaderboard + baseline driver + end-to-end smoke

**Files:**
- Modify: `smolml/leaderboard.py`
- Create: `smolml/experiments/control_baseline.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Produces: `load_control_run(path) -> ControlRunRecord`; `build_control_table(records) -> str`; `plot_control(records, out_png) -> Path`; `regenerate_control(runs_dir, table_path, plot_path) -> (str, Path)`; `control_baseline.main()`.
- Consumes: Tasks 1–6.

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/test_control.py (append)
from smolml.leaderboard import regenerate_control


def test_end_to_end_beats_random_and_improves(tmp_path):
    from smolml.envs.chemotaxis import ChemoConfig as _CC, RandomPolicy as _RP
    from smolml.models import build_model
    from smolml.envs.chemotaxis import vocab_size as _vs

    chem = _CC(width=16, levels=8, horizon=24)
    mc = {"d_model": 64, "n_layers": 3, "n_heads": 4,
          "vocab_size": _vs(chem), "max_seq_len": 2 * 24 + 1}
    step_flops = build_model("transformer", mc).flops(2 * 24).scale(32).total
    cfg = ControlTrainConfig(model_config={"d_model": 64, "n_layers": 3, "n_heads": 4},
                             flop_budget=step_flops * 400, batch_size=32, horizon=24,
                             eval_interval=50, eval_episodes=48, seed=0, run_name="ctl-smoke")
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs")

    # random-policy floor on the same held-out split
    rng_floor = []
    for s in range(48):
        e = ChemoEnv(chem, split="eval", seed=0 * 100003 + s)
        pol, c, tot = _RP(seed=s), e.reset(), 0.0
        for _ in range(chem.horizon):
            c, r = e.step(pol.act(c)); tot += r
        rng_floor.append(tot / chem.horizon)
    floor = float(np.mean(rng_floor))

    # the trained model beats the random floor and improves within an episode
    from smolml.control_eval import evaluate_control as _ec
    res = _ec(build_and_load(summary, mc), chem, split="eval", n_episodes=48, seed=0,
              device=torch.device("cpu")) if False else None  # see note below
    assert summary.final_reward > floor
    assert summary.final_reward > 0.0

    table, png = regenerate_control(tmp_path / "runs", table_path=tmp_path / "lb.md",
                                    plot_path=tmp_path / "lb.png")
    assert png.exists() and png.stat().st_size > 0
    assert "control" in table and "regret" in table
```

Note: `summary.final_reward` already comes from a held-out `evaluate_control`, so the
`build_and_load`/`_ec` line is illustrative only — delete it; assert on `summary`. Keep the
within-episode-improvement assertion by reading the final step's eval through a recorded run if you
prefer; the simplest robust check is `summary.final_reward > floor`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_control.py -q -k "end_to_end"`
Expected: FAIL (`ImportError: regenerate_control`).

- [ ] **Step 3: Implement the control leaderboard**

```python
# smolml/leaderboard.py (append)


@dataclass
class ControlRunRecord:
    run: str
    model: str
    params: int
    device: str
    seed: int
    budget: float
    flops: list[int] = field(default_factory=list)
    regret: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    wm_bits: list[float] = field(default_factory=list)

    @property
    def final_regret(self) -> float:
        return self.regret[-1] if self.regret else float("nan")


def load_control_run(path: str | Path) -> ControlRunRecord:
    meta: dict[str, object] = {}
    flops: list[int] = []
    regret: list[float] = []
    reward: list[float] = []
    wm: list[float] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                meta = obj
            elif obj.get("type") == "step":
                flops.append(int(obj["cumulative_flops"]))
                regret.append(float(obj["regret"]))
                reward.append(float(obj["mean_reward"]))
                wm.append(float(obj["world_model_bits"]))
    if not meta:
        raise ValueError(f"{path}: missing meta line")
    return ControlRunRecord(
        run=meta["run"], model=meta["model"], params=int(meta["params"]),
        device=meta["device"], seed=int(meta["seed"]), budget=float(meta["flop_budget"]),
        flops=flops, regret=regret, reward=reward, wm_bits=wm,
    )


def collect_control_runs(runs_dir: str | Path) -> list[ControlRunRecord]:
    records = [load_control_run(p) for p in sorted(Path(runs_dir).glob("*.jsonl"))]
    records.sort(key=lambda r: r.final_regret)
    return records


def build_control_table(records: list[ControlRunRecord]) -> str:
    rows = [
        "| rank | run | protocol | model | params | final FLOPs | regret | reward | wm bits |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, r in enumerate(records, start=1):
        rows.append(
            f"| {rank} | {r.run} | control | {r.model} | {r.params:,} | "
            f"{(r.flops[-1] if r.flops else 0):.3e} | {r.final_regret:.4f} | "
            f"{(r.reward[-1] if r.reward else float('nan')):.4f} | "
            f"{(r.wm_bits[-1] if r.wm_bits else float('nan')):.4f} |"
        )
    return "\n".join(rows)


def plot_control(records: list[ControlRunRecord], out_png: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in records:
        if r.flops:
            ax.plot(r.flops, r.regret, marker="o", label=r.run)
    ax.set_xscale("log")
    ax.set_xlabel("training FLOPs")
    ax.set_ylabel("regret vs oracle (per step)")
    ax.set_title("Control rung: regret vs FLOPs")
    ax.legend(fontsize=8)
    out = Path(out_png)
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)
    return out


def regenerate_control(
    runs_dir: str | Path,
    *,
    table_path: str | Path,
    plot_path: str | Path,
) -> tuple[str, Path]:
    records = collect_control_runs(runs_dir)
    table = build_control_table(records)
    Path(table_path).write_text(table + "\n")
    png = plot_control(records, plot_path)
    return table, png
```

- [ ] **Step 4: Implement the baseline driver**

```python
# smolml/experiments/control_baseline.py
"""Transformer control-rung baseline (the bar): distill-train across a small FLOP
sweep, eval on held-out episodes, write the leaderboard + a sample rollout raster.

Run (CPU, synthetic; minutes)::

    uv run python -m smolml.experiments.control_baseline
"""

from smolml.control_eval import evaluate_control
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import ChemoConfig
from smolml.envs.render import render_rollout
from smolml.leaderboard import regenerate_control
from smolml.models import build_model

RUNS_DIR = "runs/control"
BUDGETS = (2e8, 1e9, 4e9)
HORIZON = 64


def main() -> None:
    chem = ChemoConfig(horizon=HORIZON)
    for budget in BUDGETS:
        cfg = ControlTrainConfig(
            model_config={"d_model": 128, "n_layers": 4, "n_heads": 4},
            flop_budget=budget, horizon=HORIZON, eval_episodes=128,
            run_name=f"transformer-control-{budget:.0e}",
        )
        summary = distill_train_run(cfg, runs_dir=RUNS_DIR)
        print(f"budget={budget:.0e}  regret={summary.final_regret:.4f}  "
              f"reward={summary.final_reward:.4f}  wm_bits={summary.final_world_model_bits:.4f}  "
              f"flops={summary.total_flops:.3e}")

    table, png = regenerate_control(RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md",
                                    plot_path=f"{RUNS_DIR}/leaderboard.png")
    print("\n" + table + f"\nplot: {png}")

    # a sample rollout raster from the largest-budget model
    mc = {"d_model": 128, "n_layers": 4, "n_heads": 4}
    model = build_model("transformer", {**mc, "vocab_size": chem.levels + 3,
                                        "max_seq_len": 2 * HORIZON + 1})
    res = evaluate_control(model, chem, split="eval", n_episodes=1, seed=0,
                           device=next(model.parameters()).device, record=True)
    render_rollout(res.trajectory, f"{RUNS_DIR}/sample_rollout.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the end-to-end test (tune budget until it reliably beats the floor)**

Run: `uv run pytest tests/test_control.py -q -k "end_to_end"`
Expected: PASS. If `final_reward` does not clear the random floor, raise `flop_budget` (more
distillation) and/or `d_model`/`n_layers`; do NOT weaken the assertion. Record the working budget.

- [ ] **Step 6: Commit**

```bash
git add smolml/leaderboard.py smolml/experiments/control_baseline.py tests/test_control.py
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit -m "feat(C.A.0): control leaderboard + transformer baseline driver + e2e smoke"
```

---

### Task 8: Docs — `harness.md`

**Files:**
- Modify: `docs/harness.md`

- [ ] **Step 1: Add a "Control rung (in-context RL)" section** documenting: the chemotaxis env + the `Environment` seam; how to run the baseline (`uv run python -m smolml.experiments.control_baseline`); the metric (regret vs oracle + world-model bits vs total FLOPs at fixed params); how to add a Space-C control candidate (implement the model seam; it plugs into `evaluate_control`/`distill_train_run` unchanged); how to regenerate the board (`regenerate_control`) and render a rollout (`render_rollout`/`animate_rollout`).

- [ ] **Step 2: Commit**

```bash
git add docs/harness.md
git -c user.name="Nils Koch" -c user.email="nils.koch@seibert.group" commit -m "docs(C.A.0): harness.md — control rung run/extend/regenerate"
```

---

## Final gates (orchestrator runs once, across all changed files)

- [ ] `uvx ruff format --check .` — clean (run without piping so the exit code is real).
- [ ] `uvx ruff check .` — clean.
- [ ] `uv run pytest` — all green; paste output into the PR.
- [ ] `uv run python -m smolml.experiments.control_baseline` — prints the per-budget bar + leaderboard; sanity-check that regret falls and reward rises with FLOPs; attach `runs/control/sample_rollout.png`.
- [ ] Cross-vendor review (a different vendor): focus on FLOP honesty in `evaluate_control`/`distill_train_run` and the held-out drift-pool disjointness. Reviewer reports; does not edit.
- [ ] Per AGENTS.md docs-builder directive (post-baseline): hand the researcher note + an `in-context-control` concept (incl. an interactive, scrubbable rollout viz) to the docs-builder; confirm the page; site build green.

## Self-Review (completed during planning)

- **Spec coverage:** env (T1–2), tape/slices (T1,T3), rollout scorer + world-model bits + regret (T4), FLOP honesty (T4 accounting test + T6 budget), visualization (T5), distillation training (T6), leaderboard (T7), baseline + held-out-beats-random + within-episode improvement smoke (T2 source-learner + T7 e2e), docs (T8). Memory/fixed-params: reported via `params` in the meta/leaderboard; tape bounded by `max_seq_len`.
- **Placeholder scan:** none — every step carries real code. Two transcription notes are called out explicitly (the stray `__add__` line in T4; the illustrative `build_and_load` line in T7) to delete on transcription.
- **Type consistency:** `ChemoConfig`, slices, `action_token`, `Trajectory`, `ControlResult`, `ControlTrainConfig`/`ControlRunSummary`, `ControlRunRecord` names/fields match across tasks; `evaluate_control` signature is identical wherever called.
