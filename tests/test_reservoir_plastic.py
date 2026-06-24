"""Reservoir + ONLINE reward-modulated plastic readout (Task C.A.1b).

The frozen :class:`_ReservoirCore` is reused unchanged; the readout is no longer
distilled-and-frozen — a working copy of ``(W, b)`` lives in the decode cache and is adapted
by a gradient-free LOCAL rule inside ``step`` (``evaluate_control`` is ``@torch.no_grad()``).
The headline is the ~0-distillation point: 0 train steps, all learning online in ``step``,
every adaptation FLOP charged to the returned breakdown's ``backward`` (ADR 0004).
"""

import numpy as np
import torch

from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import (
    N_ACTIONS,
    ChemoConfig,
    ChemoEnv,
    RandomPolicy,
    action_token,
    vocab_size,
)
from smolml.flops import gather_flops, matmul_flops, pointwise_flops
from smolml.leaderboard import regenerate_control
from smolml.models import build_model
from smolml.models.reservoir import ReservoirPlastic, ReservoirPlasticConfig

CPU = torch.device("cpu")
TINY = {"d_res": 24, "vocab_size": 11, "max_seq_len": 64}


def _model(**overrides) -> ReservoirPlastic:
    return ReservoirPlastic(ReservoirPlasticConfig(**{**TINY, **overrides}))


def _per_token_forward(d: int, v: int) -> int:
    """Hand-derived per-token forward (recurrence + plastic readout), independent of code."""
    return (
        matmul_flops(1, d, d)  # W_res @ h
        + gather_flops(d)  # W_in[:, token]
        + pointwise_flops(d)  # residual add
        + pointwise_flops(d, per_elem=4)  # leaky tanh update
        + matmul_flops(1, v, d)  # readout matvec
        + pointwise_flops(v)  # readout bias add
    )


def _update_flops(d: int, lv: int, na: int) -> int:
    """Hand-derived online-update cost (world-model delta rule + Hebbian policy update)."""
    wm = (
        pointwise_flops(lv, per_elem=3)  # softmax(prev conc-pred logits)
        + pointwise_flops(lv, per_elem=2)  # onehot target + (target - pred)
        + matmul_flops(d, lv, 1)  # outer(err, h_action)
        + pointwise_flops(d * lv, per_elem=2)  # W[conc] += lr_wm * outer
        + pointwise_flops(lv, per_elem=2)  # b[conc] += lr_wm * err
    )
    pol = (
        pointwise_flops(5)  # r (1) + adv (1) + leaky baseline b+decay·(r-b) (3)
        + pointwise_flops(na)  # onehot(a_taken)
        + matmul_flops(d, na, 1)  # outer(onehot_a, h_conc)
        + pointwise_flops(d * na, per_elem=2)  # W[action] += lr_pol * adv * outer
        + pointwise_flops(na, per_elem=2)  # b[action] += lr_pol * adv * onehot_a
    )
    return wm + pol


# --- forward shape, determinism, and the reused frozen core ------------------
def test_forward_shape_and_determinism():
    m = _model()
    idx = torch.randint(0, 11, (2, 7))
    out = m(idx)
    assert out.shape == (2, 7, 11)
    assert torch.equal(out, _model()(idx))  # fixed seed -> identical
    assert not torch.equal(out, _model(seed=1)(idx))


def test_reuses_frozen_core():
    m = _model(d_res=64)
    # The shared echo-state core is frozen (never trained) and spectral-radius-scaled.
    assert m.core.W_in.requires_grad is False
    assert m.core.W_res.requires_grad is False
    radius = torch.linalg.eigvals(m.core.W_res).abs().max().item()
    assert abs(radius - 0.9) < 1e-4
    # Only the seed readout is trainable (the optimizer excludes the frozen core).
    opt = m.configure_optimizer(lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95))
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.core.W_in) not in opt_ids and id(m.core.W_res) not in opt_ids
    assert id(m.readout.weight) in opt_ids and id(m.readout.bias) in opt_ids


# --- ONLINE LEARNING: the cache (W,b) adapt; the nn.Parameters never change ---
def test_online_update_changes_cache_not_parameters():
    m = _model()
    lv = m._levels
    # Snapshot every nn.Parameter; eval must not train any of them.
    params_before = {n: p.detach().clone() for n, p in m.named_parameters()}

    state = m.init_prequential_state()
    w_seed = state.cache.W.clone()
    # Tape parity: c0 (pos0), a0 (pos1), c1 (pos2). The update fires only at pos2.
    state, _, f0 = m.step(state, 3, 0)  # conc fold, pos<2 -> no update
    assert f0.backward == 0
    state, _, f1 = m.step(state, action_token(ChemoConfig(levels=lv), 1), 1)  # action fold
    assert f1.backward == 0
    w_pre = state.cache.W.clone()
    state, _, f2 = m.step(state, 5, 2)  # conc fold at pos>=2 -> ONLINE UPDATE
    w_post = state.cache.W.clone()

    # The plastic readout in the cache CHANGED across the conc-fold update.
    assert not torch.allclose(w_pre, w_post)
    assert torch.allclose(w_seed, w_pre)  # unchanged before the first eligible update
    # Both blocks moved: world-model (conc) columns and policy (action) columns.
    assert not torch.allclose(w_pre[m._conc], w_post[m._conc])
    assert not torch.allclose(w_pre[m._action], w_post[m._action])
    # The update charged a positive, exact backward; no model weight was trained.
    assert f2.backward == _update_flops(m.config.d_res, lv, N_ACTIONS) > 0
    for n, p in m.named_parameters():
        assert torch.equal(p, params_before[n]), f"eval mutated parameter {n}"


# --- FLOP honesty: forward = recurrence+readout; backward only on conc folds --
def test_flop_accounting_forward_and_online_backward():
    d, v = 24, 11
    m = _model(d_res=d)
    lv = m._levels
    expected_fwd = _per_token_forward(d, v)
    assert m.decode_step_flops(0).forward == expected_fwd
    assert m.decode_step_flops(0).forward == m.decode_step_flops(999).forward  # context-free

    # Roll a realistic c,a,c,a,... tape; sum forward, check per-parity backward.
    chem = ChemoConfig(levels=lv)
    state = m.init_prequential_state()
    total_forward = 0
    t = 8
    for pos in range(t):
        tok = (pos % 4) + 1 if pos % 2 == 0 else action_token(chem, pos % N_ACTIONS)
        state, _, f = m.step(state, tok, pos)
        assert f.forward == expected_fwd
        if pos % 2 == 0 and pos >= 2:
            assert f.backward == _update_flops(d, lv, N_ACTIONS) > 0  # conc fold -> update
        else:
            assert f.backward == 0  # action fold (or pos<2) -> no update
        total_forward += f.forward
    # Summed step forward == T x the context-independent per-step decode cost.
    assert total_forward == t * m.decode_step_flops(0).forward

    # The distill path's backward is readout-only (NOT the 2x-forward backprop tax).
    fb = m.flops(t)
    assert fb.forward == t * expected_fwd
    assert fb.backward == t * (matmul_flops(1, v, d) + pointwise_flops(v))
    assert fb.backward != 2 * fb.forward


# --- param budget: counts the frozen core, fits under the transformer bar -----
def test_num_params_counts_frozen_core_under_bar():
    m = build_model("reservoir_plastic", {"vocab_size": 11, "max_seq_len": 129})
    assert m.config.d_res == 374
    core = m.core.W_in.numel() + m.core.W_res.numel()
    readout = m.readout.weight.numel() + m.readout.bias.numel()
    assert m.num_params() == core + readout == 148_115  # frozen core IS counted
    assert m.num_params() <= 148_608


# --- end-to-end smoke: ~0 distillation, all learning online in step ----------
def _random_floor(chem: ChemoConfig, episodes: int, seed: int = 0) -> float:
    floor = []
    for s in range(episodes):
        e = ChemoEnv(chem, split="eval", seed=seed * 100003 + s)
        pol, c, tot = RandomPolicy(seed=s), e.reset(), 0.0
        for _ in range(chem.horizon):
            c, r = e.step(pol.act(c))
            tot += r
        floor.append(tot / chem.horizon)
    return float(np.mean(floor))


def test_end_to_end_zero_distillation_beats_floor(tmp_path):
    """Headline: flop_budget below one train step -> 0 train steps; the online rule alone
    must clear the random floor. (Observed: a healthy +0.05..+0.09 reward margin across
    seeds/horizons; the within-episode 2nd>1st signal is positive at this seed but
    seed-sensitive under env drift — a documented source-(iv) finding, not a stub.)"""
    horizon, episodes = 16, 24
    chem = ChemoConfig(width=16, levels=8, horizon=horizon)
    mc = {"vocab_size": vocab_size(chem), "max_seq_len": 2 * horizon + 1}
    # flop_budget strictly below one train step => the distill loop runs 0 steps.
    step_flops = build_model("reservoir_plastic", mc).flops(2 * horizon).scale(32).total
    cfg = ControlTrainConfig(
        model="reservoir_plastic",
        model_config={},
        flop_budget=step_flops * 0.5,  # < one step -> 0 distillation steps
        batch_size=32,
        horizon=horizon,
        eval_interval=10**9,
        eval_episodes=episodes,
        seed=0,
        run_name="reservoir-plastic-zero-distill",
    )
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs")
    assert summary.steps == 0  # ~0 distillation: ALL learning is online in step
    assert summary.total_flops > 0  # online adaptation FLOPs were charged at eval

    floor = _random_floor(chem, episodes, seed=0)
    assert np.isfinite(summary.final_reward)
    # The gradient-free online rule clears the random floor (the documented headline).
    assert summary.final_reward > floor
    # Within-episode improvement (the learning signal) is positive at this seed.
    assert summary.second_half_reward > summary.first_half_reward

    assert (tmp_path / "runs" / f"{summary.run}.jsonl").exists()
    table, png = regenerate_control(
        tmp_path / "runs", table_path=tmp_path / "lb.md", plot_path=tmp_path / "lb.png"
    )
    assert png.exists() and png.stat().st_size > 0
    assert "reservoir-plastic-zero-distill" in table and "regret" in table
