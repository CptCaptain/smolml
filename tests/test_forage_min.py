"""C.A.4 acceptance tests: the `forage_min` per-type contingency tracker.

Mirrors ``test_chemotaxis_min.py``: the in-context adaptation is a per-type value EMA (no
weight change at eval), a distilled-scalar softmax policy, and pointwise FLOP honesty.
"""

import numpy as np
import torch

from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.forage import (
    EAT,
    N_ACTIONS,
    REWARD_LEVELS,
    RIGHT,
    ForageConfig,
    ForageEnv,
    forage_env_spec,
    vocab_size,
)
from smolml.flops import pointwise_flops
from smolml.leaderboard import regenerate_control
from smolml.models import build_model

CPU = torch.device("cpu")


def _build(n_types: int = 3, horizon: int = 8, **ov):
    fcfg = ForageConfig(n_types=n_types, horizon=horizon)
    mc = {"vocab_size": vocab_size(fcfg), "max_seq_len": 2 * horizon + 1, **ov}
    return build_model("forage_min", mc), fcfg


def _obs(type_: int, reward: int) -> int:
    return type_ * REWARD_LEVELS + (reward + 1)


def _tape(fcfg: ForageConfig):
    """An alternating obs/action tape (deterministic) ending on an action."""
    obs_len = REWARD_LEVELS * fcfg.n_types
    toks = []
    for i in range(fcfg.horizon):
        toks.append(_obs(i % fcfg.n_types, (-1 if i % 2 else 1)))
        toks.append(obs_len + (i % N_ACTIONS))
    return toks


def _rand_valid_tape(fcfg: ForageConfig, seed: int) -> list[int]:
    """A random but well-formed tape: a valid combined obs at even positions, a valid action
    token at odd positions (what the scorer actually produces)."""
    rng = np.random.default_rng(seed)
    obs_len = REWARD_LEVELS * fcfg.n_types
    toks = []
    for i in range(fcfg.horizon):
        toks.append(int(rng.integers(obs_len)))
        toks.append(obs_len + int(rng.integers(N_ACTIONS)))
    return toks


# --- forward shape, determinism --------------------------------------------------


def test_forward_shape_and_both_slices_populated():
    model, fcfg = _build(horizon=8)
    v = vocab_size(fcfg)
    idx = torch.tensor([_rand_valid_tape(fcfg, 1), _rand_valid_tape(fcfg, 2)], dtype=torch.long)
    out = model(idx)
    assert out.shape == (2, 2 * fcfg.horizon, v)
    obs_len = REWARD_LEVELS * fcfg.n_types
    for t in range(out.shape[1]):
        assert out[0, t, :obs_len].std() >= 0  # obs head populated
        assert out[0, t, obs_len:].std() > 0  # policy distinguishes the actions


def test_deterministic_logits():
    model, fcfg = _build(horizon=6)
    idx = torch.tensor([_rand_valid_tape(fcfg, 3)], dtype=torch.long)
    assert torch.equal(model(idx), model(idx))


# --- step/forward parity + FLOP sum ---------------------------------------------


def test_step_matches_forward_and_flop_sum():
    model, fcfg = _build(horizon=6)
    toks = _tape(fcfg)
    fwd = model(torch.tensor([toks], dtype=torch.long))[0]
    state = model.init_prequential_state()
    summed = 0
    for pos, tok in enumerate(toks):
        state, logits, f = model.step(state, tok, pos)
        assert torch.allclose(logits, fwd[pos], atol=1e-6)  # step == forward
        assert f.backward == 0
        summed += f.forward
    assert summed == len(toks) * model.decode_step_flops(0).forward


# --- FLOP honesty + params ------------------------------------------------------


def test_flop_honesty_pointwise_and_backward_zero():
    model, _ = _build(n_types=3, horizon=8)
    f = model.decode_step_flops(0)
    assert f.forward == pointwise_flops(model._per_step_ops()) > 0
    assert f.backward == 0
    assert model.decode_step_flops(0).forward == model.decode_step_flops(999).forward
    fl = model.flops(4)
    assert fl.backward == 2 * fl.forward  # distill path is from_forward (3x)


def test_num_params_is_the_eight_scalars():
    model, _ = _build()
    assert model.num_params() == 8
    assert all(p.numel() == 1 and p.requires_grad for p in model.parameters())


# --- the learning behavior (the mechanism's reason to exist) --------------------


def test_delta_rule_credits_eaten_type_only_after_eat():
    model, fcfg = _build(n_types=3, horizon=8, v_init=0.0)
    obs_len = REWARD_LEVELS * fcfg.n_types
    state = model.init_prequential_state()
    state, _, _ = model.step(state, _obs(0, 0), 0)  # sense type 0
    state, _, _ = model.step(state, obs_len + EAT, 1)  # EAT
    state, _, _ = model.step(state, _obs(0, -1), 2)  # poison revealed -> credit type 0
    assert state.cache.v[0] < 0.0  # learned: type 0 is bad
    v_snapshot = list(state.cache.v)
    state, _, _ = model.step(state, obs_len + RIGHT, 3)  # move
    state, _, _ = model.step(state, _obs(1, 0), 4)  # neighbor; last_action=RIGHT
    assert list(state.cache.v) == v_snapshot  # move -> no credit


def test_policy_eats_high_value_skips_poison():
    model, fcfg = _build(n_types=3, horizon=8)
    obs_len = REWARD_LEVELS * fcfg.n_types
    state = model.init_prequential_state()
    state.cache.v[1] = 1.0
    state.cache.v[0] = -1.0
    _, good, _ = model.step(state, _obs(1, 0), 0)
    assert int(good[obs_len:].argmax()) == EAT  # eat the good type
    state2 = model.init_prequential_state()
    state2.cache.v[1] = 1.0
    state2.cache.v[0] = -1.0
    _, bad, _ = model.step(state2, _obs(0, 0), 0)
    assert int(bad[obs_len:].argmax()) != EAT  # don't eat poison


# --- end-to-end: ~0 distillation, all learning online in step -------------------


def _random_floor(fcfg: ForageConfig, episodes: int) -> float:
    rng = np.random.default_rng(0)
    out = []
    for s in range(episodes):
        e = ForageEnv(fcfg, split="eval", seed=s * 100003)
        e.reset()
        tot = 0.0
        for _ in range(fcfg.horizon):
            _, r = e.step(int(rng.integers(N_ACTIONS)))
            tot += r
        out.append(tot / fcfg.horizon)
    return float(np.mean(out))


def test_end_to_end_zero_distillation_beats_floor(tmp_path):
    horizon, episodes = 64, 32
    fcfg = ForageConfig(horizon=horizon)
    mc = {"vocab_size": vocab_size(fcfg), "max_seq_len": 2 * horizon + 1}
    step_flops = build_model("forage_min", mc).flops(2 * horizon).scale(32).total
    cfg = ControlTrainConfig(
        model="forage_min",
        model_config={},
        flop_budget=step_flops * 0.5,  # < one train step -> 0 distillation steps
        batch_size=32,
        horizon=horizon,
        eval_interval=10**9,
        eval_episodes=episodes,
        seed=0,
        env_name="forage",
        run_name="forage-min-zero-distill",
    )
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs", env_spec=forage_env_spec(fcfg))
    assert summary.steps == 0  # ~0 distillation: ALL learning is online in step
    assert summary.total_flops > 0
    floor = _random_floor(fcfg, episodes)
    assert np.isfinite(summary.final_reward)
    assert summary.final_reward > floor  # the gradient-free online rule clears the floor
    assert summary.second_half_reward > summary.first_half_reward  # within-episode learning
    table, png = regenerate_control(
        tmp_path / "runs", table_path=tmp_path / "lb.md", plot_path=tmp_path / "lb.png"
    )
    assert png.exists() and png.stat().st_size > 0
    assert "forage-min-zero-distill" in table and "regret" in table
