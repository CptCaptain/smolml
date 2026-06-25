"""Acceptance tests for the contingency-forage control rung (Task C.A.3).

Mirrors ``test_control_eval.py`` + ``test_chemotaxis.py`` and pins the spec's
Monte-Carlo references. The headline guard is REFLEX-PROOF: no contingency-blind
fixed policy competes with the oracle, so regret measures in-context learning.
"""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smolml.control_eval import evaluate_control
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import Trajectory
from smolml.envs.forage import (
    EAT,
    LEFT,
    N_ACTIONS,
    REWARD_LEVELS,
    RIGHT,
    ForageConfig,
    ForageEnv,
    WinStayLoseShift,
    band_seed,
    forage_env_spec,
    vocab_size,
)
from smolml.envs.render import render_rollout
from smolml.envs.spec import make_distillation_batch
from smolml.experiments.forage_baseline import ref_step_flops
from smolml.flops import FlopBreakdown
from smolml.leaderboard import regenerate_control
from smolml.models.registry import LanguageModel

CPU = torch.device("cpu")
FIXED_FAMILY = ("always_eat", "always_right", "always_left", "eat_0", "eat_1", "eat_2")


# --- reference policies + Monte-Carlo harness ----------------------------------


def _const(action: int):
    return lambda _seed: lambda _obs, _env: action


def _oracle_pol():
    return lambda _seed: lambda _obs, env: env.oracle_action()


def _eat_type_pol(k: int):
    return lambda _seed: lambda obs, _env: EAT if obs // REWARD_LEVELS == k else RIGHT


def _random_pol():
    def make(seed: int):
        rng = np.random.default_rng(seed)
        return lambda _obs, _env: int(rng.integers(N_ACTIONS))

    return make


def _wsls_pol(epsilon: float, n_types: int = 3):
    def make(seed: int):
        src = WinStayLoseShift(n_types, epsilon=epsilon, seed=seed)
        return lambda obs, _env: src.act(obs)

    return make


def _mc(make_pol, n: int, fcfg: ForageConfig, *, base_seed: int = 0, split: str = "eval"):
    """Mean / first-half / second-half reward of a policy over ``n`` fresh episodes."""
    horizon = fcfg.horizon
    half = horizon // 2
    total = first = second = 0.0
    for ep in range(n):
        env = ForageEnv(fcfg, split=split, seed=base_seed + ep)
        act = make_pol(base_seed + ep)
        obs = env.reset()
        for t in range(horizon):
            obs, reward = env.step(act(obs, env))
            total += reward
            if t < half:
                first += reward
            else:
                second += reward
    return total / (n * horizon), first / (n * half), second / (n * (horizon - half))


@functools.cache
def _reference_means(n: int = 2000, horizon: int = 64):
    fcfg = ForageConfig(horizon=horizon)
    pols = {
        "oracle": _oracle_pol(),
        "always_eat": _const(EAT),
        "always_right": _const(RIGHT),
        "always_left": _const(LEFT),
        "random": _random_pol(),
        "eat_0": _eat_type_pol(0),
        "eat_1": _eat_type_pol(1),
        "eat_2": _eat_type_pol(2),
        "wsls": _wsls_pol(0.05),
    }
    return {name: _mc(mk, n, fcfg) for name, mk in pols.items()}


# --- stub model (mirrors test_control_eval._FixedActionModel) -------------------


class _FixedActionModel(LanguageModel):
    """Always favors one absolute action token; uniform over the obs sub-vocab."""

    def __init__(self, vocab: int, max_seq_len: int, fav_token: int):
        super().__init__()
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self._vocab, self._fav = vocab, fav_token
        self._anchor = torch.nn.Parameter(torch.zeros(1))

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


# --- determinism + held-out seed bands -----------------------------------------


def test_env_deterministic_and_fresh_per_episode():
    fcfg = ForageConfig(horizon=32)
    actions = [EAT, RIGHT, LEFT, EAT, EAT, RIGHT] * 6

    def rollout(seed: int):
        env = ForageEnv(fcfg, split="eval", seed=seed)
        trace = [(env.reset(), 0.0, env.record_state())]
        for a in actions[: fcfg.horizon]:
            obs, reward = env.step(a)
            trace.append((obs, reward, env.record_state()))
        return trace

    assert rollout(7) == rollout(7)  # same (split, seed) -> identical trajectory
    e7, e8 = ForageEnv(fcfg, split="eval", seed=7), ForageEnv(fcfg, split="eval", seed=8)
    assert (e7.cells, e7.g, e7.p) != (e8.cells, e8.g, e8.p)  # fresh layout/g per episode
    assert len({ForageEnv(fcfg, split="eval", seed=s).g for s in range(40)}) > 1


def test_train_eval_seed_bands_disjoint():
    seeds = list(range(3000)) + [100003 * i + j for i in range(60) for j in range(5)]
    train = {band_seed("train", s) for s in seeds}
    held_out = {band_seed("eval", s) for s in seeds}
    assert train and held_out and train.isdisjoint(held_out)
    assert band_seed("eval", 5) == band_seed("eval", 5)  # deterministic
    with pytest.raises(ValueError):
        band_seed("bogus", 0)
    fcfg = ForageConfig()
    et, ee = ForageEnv(fcfg, split="train", seed=3), ForageEnv(fcfg, split="eval", seed=3)
    assert (et.cells, et.g) != (ee.cells, ee.g)  # eval layout held out from train


# --- tape, vocab, slices, combined-obs id map ----------------------------------


def test_tape_layout_and_obs_encoding():
    fcfg = ForageConfig(n_types=3)
    spec = forage_env_spec(fcfg)
    ts = spec.tape_spec
    k = fcfg.n_types
    assert vocab_size(fcfg) == ts.vocab_size == 3 * k + 3
    assert ts.obs_slice == slice(0, 3 * k) and ts.action_slice == slice(3 * k, 3 * k + 3)
    assert [ts.action_token(i) for i in range(N_ACTIONS)] == [3 * k, 3 * k + 1, 3 * k + 2]

    env = ForageEnv(fcfg, split="eval", seed=0)
    obs0 = env.reset()
    assert obs0 // REWARD_LEVELS == env.cells[env.p]  # type component
    assert obs0 % REWARD_LEVELS - 1 == 0  # reset reward component is 0
    for a in (EAT, RIGHT, LEFT, EAT, EAT, RIGHT):
        obs, reward = env.step(a)
        assert ts.obs_slice.start <= obs < ts.obs_slice.stop  # obs occupies [0, 3K)
        assert obs // REWARD_LEVELS == env.cells[env.p]  # current (post-move) type
        assert obs % REWARD_LEVELS - 1 == reward  # last-reward component matches grade


def test_distillation_tape_format_and_shift():
    fcfg = ForageConfig(horizon=8)
    spec = forage_env_spec(fcfg)
    x, y = make_distillation_batch(spec, batch_size=4, seed=0, device=CPU, epsilon=0.1)
    assert x.shape == (4, 2 * fcfg.horizon) == y.shape
    assert torch.equal(x[:, 1:], y[:, :-1])  # next-token shift
    obs_sl, act_sl = spec.tape_spec.obs_slice, spec.tape_spec.action_slice
    full = torch.cat([x, y[:, -1:]], dim=1)  # reconstruct the (2H+1) tape
    for b in range(4):
        for t in range(full.shape[1]):
            tok = int(full[b, t])
            if t % 2 == 0:
                assert obs_sl.start <= tok < obs_sl.stop
            else:
                assert act_sl.start <= tok < act_sl.stop


# --- MC-pinned metric bounds (production H=64) ----------------------------------


def test_metric_bounds_mc_pinned():
    m = _reference_means()
    assert abs(m["oracle"][0] - 0.96) < 0.05  # camp g => ~+1/step (the true optimum)
    assert abs(m["always_eat"][0] - (-0.333)) < 0.04  # blind camp = +1 w.p 1/K else poison
    assert m["always_right"][0] == 0.0 and m["always_left"][0] == 0.0
    assert abs(m["random"][0] - (-0.112)) < 0.03
    assert m["always_right"][0] < m["wsls"][0] < m["oracle"][0]  # source strictly between
    assert abs(m["wsls"][0] - 0.85) < 0.06  # search early, camp g late


# --- REFLEX-PROOF: no fixed policy competes; learning is required ---------------


def test_reflex_proof_no_fixed_policy_competes():
    m = _reference_means()
    margin = 0.1
    oracle = m["oracle"][0]
    best_fixed = max(m[p][0] for p in FIXED_FAMILY)
    assert oracle >= best_fixed + margin  # oracle beats every blind fixed policy
    assert oracle - m["always_eat"][0] > margin > 0  # the reflex carries real regret
    max_family_regret = max(oracle - m[p][0] for p in FIXED_FAMILY)
    wsls_regret = oracle - m["wsls"][0]
    assert max_family_regret > wsls_regret + 0.3  # a learner separates from the reflex


def test_source_is_a_learner():
    _mean, first, second = _reference_means()["wsls"]
    assert second > first + 0.03  # within-episode improvement (explore early, exploit late)
    assert second > _reference_means()["always_eat"][0]  # 2nd half beats the blind reflex


# --- causal / honest interaction -----------------------------------------------


def test_interaction_is_causal_and_honest():
    fcfg = ForageConfig(horizon=16)
    spec = forage_env_spec(fcfg)
    vocab, msl = spec.tape_spec.vocab_size, 2 * fcfg.horizon + 1
    tok = spec.tape_spec.action_token

    def run(action_idx: int):
        model = _FixedActionModel(vocab, msl, tok(action_idx))
        return evaluate_control(
            model, spec, split="eval", n_episodes=1, seed=3, device=CPU, greedy=True, record=True
        )

    rl, rr, re = run(LEFT), run(RIGHT), run(EAT)
    lpos = [s["p"] for s in rl.trajectory.states]
    rpos = [s["p"] for s in rr.trajectory.states]
    assert lpos != rpos  # changing the policy changes the trajectory (causal)
    assert rr.mean_reward == 0.0  # moving is never rewarded
    assert re.mean_reward != 0.0  # eating is graded by the env
    # reward is computed FROM the sampled action: every EAT is graded +1/-1
    graded = zip(re.trajectory.action, re.trajectory.reward, strict=True)
    assert all(a == EAT and r in (1.0, -1.0) for a, r in graded)


# --- rollout FLOP accounting ----------------------------------------------------


def test_rollout_flop_accounting_matches_analytic():
    from smolml.models.transformer import Transformer, TransformerConfig

    fcfg = ForageConfig(horizon=8)
    spec = forage_env_spec(fcfg)
    tcfg = TransformerConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        vocab_size=spec.tape_spec.vocab_size,
        max_seq_len=2 * fcfg.horizon + 1,
    )
    model = Transformer(tcfg)
    res = evaluate_control(model, spec, split="eval", n_episodes=1, seed=1, device=CPU)
    expected = sum(model.decode_step_flops(k).forward for k in range(1, 2 * fcfg.horizon + 1))
    assert res.flops.forward == expected


# --- end-to-end distillation smoke ---------------------------------------------


def test_end_to_end_distill_smoke(tmp_path):
    fcfg = ForageConfig(horizon=32)
    spec = forage_env_spec(fcfg)
    budget = ref_step_flops(fcfg, 32) * 500
    cfg = ControlTrainConfig(
        model_config={"d_model": 64, "n_layers": 3, "n_heads": 4},
        flop_budget=budget,
        batch_size=32,
        horizon=32,
        eval_episodes=64,
        eval_interval=10**9,
        epsilon=0.05,
        env_name="forage",
        run_name="forage-smoke",
        seed=0,
    )
    summary, model = distill_train_run(cfg, runs_dir=tmp_path, return_model=True, env_spec=spec)
    res = evaluate_control(model, spec, split="eval", n_episodes=128, seed=999, device=CPU)

    refs = _reference_means(512, 32)
    best_fixed_regret = refs["oracle"][0] - max(refs[p][0] for p in FIXED_FAMILY)
    assert res.mean_reward > refs["random"][0]  # beats random
    assert res.second_half_reward > res.first_half_reward  # within-episode adaptation
    assert res.regret < best_fixed_regret  # beats the best fixed policy -> it learned to adapt

    table, _png = regenerate_control(
        tmp_path, table_path=tmp_path / "lb.md", plot_path=tmp_path / "lb.png"
    )
    assert "forage-smoke" in table  # a leaderboard row is written

    rec = evaluate_control(model, spec, split="eval", n_episodes=1, seed=0, device=CPU, record=True)
    out = render_rollout(rec.trajectory, tmp_path / "rollout.png")
    assert out.exists() and out.stat().st_size > 0


# --- renderer (isolated, no training) ------------------------------------------


def test_render_forage_writes_nonempty_png(tmp_path):
    width, steps = 16, 9
    cells = [(x * 7) % 3 for x in range(width)]
    pos = [t % width for t in range(steps)]
    states = [{"cells": cells, "g": 1, "p": pos[t]} for t in range(steps)]
    action = [EAT if t % 2 == 0 else RIGHT for t in range(steps - 1)]
    reward = [
        (1.0 if cells[pos[t]] == 1 else -1.0) if action[t] == EAT else 0.0 for t in range(steps - 1)
    ]
    traj = Trajectory(obs_token=[0] * steps, action=action, reward=reward, states=states)
    out = render_rollout(traj, tmp_path / "forage.png")
    assert out.exists() and out.stat().st_size > 0
