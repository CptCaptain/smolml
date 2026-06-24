"""Task C.A.2 acceptance tests: the `chemotaxis_min` minimal-organism candidate.

Mirrors the control-rung test patterns (env determinism, metric bounds,
step ≡ forward, analytic FLOP honesty) for a hand-structured run-and-tumble
controller whose in-context adaptation is a leaky integrator (no weight change)."""

import math

import numpy as np
import torch

from smolml.control_eval import evaluate_control
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import (
    ChemoConfig,
    ChemoEnv,
    RandomPolicy,
    action_slice,
    action_token,
    conc_slice,
    vocab_size,
)
from smolml.leaderboard import regenerate_control
from smolml.models import build_model
from smolml.models.chemotaxis_min import LEFT, RIGHT

# The transformer bar's CHEAPEST total-FLOP point (the FLOP axis this candidate bets on).
TRANSFORMER_BAR_MIN_FLOPS = 2.97e11
DEVICE = torch.device("cpu")


def _build(levels: int = 8, horizon: int = 32, **overrides):
    chem = ChemoConfig(width=16, levels=levels, horizon=horizon)
    mc = {"vocab_size": vocab_size(chem), "max_seq_len": 2 * horizon + 1, **overrides}
    return build_model("chemotaxis_min", mc), chem


def _roll_levels(model, levels: int, sensed: list[int], heading: int):
    """Replay an alternating tape (sense c, act `heading`, ...) ending on a sense,
    and return the just-emitted full-vocab logits (the action prediction)."""
    state = model.init_prequential_state()
    logits = None
    pos = 0
    for i, c in enumerate(sensed):
        state, logits, _ = model.step(state, c, pos)
        pos += 1
        if i < len(sensed) - 1:  # interleave an action token, except after the last sense
            state, logits, _ = model.step(state, action_token_for(levels, heading), pos)
            pos += 1
    return logits


def action_token_for(levels: int, idx: int) -> int:
    return levels + idx


def test_forward_shape_and_both_slices_populated():
    model, chem = _build(horizon=8)
    v = vocab_size(chem)
    # A valid tape: even positions concentrations, odd positions action tokens.
    toks = [3, action_token(chem, RIGHT), 5, action_token(chem, RIGHT), 6, action_token(chem, LEFT)]
    x = torch.tensor([toks, toks])  # batch of 2
    out = model(x)
    assert out.shape == (2, len(toks), v)
    cs, as_ = conc_slice(chem), action_slice(chem)
    # Both heads carry signal at every position (not a degenerate constant slice).
    for t in range(len(toks)):
        assert out[0, t, cs].std() > 0  # world-model peak varies across levels
        assert out[0, t, as_].std() > 0  # policy distinguishes the actions


def test_run_and_tumble_keep_on_rise_reverse_on_drop():
    model, chem = _build()
    L, cs_a = chem.levels, action_slice(chem)
    with torch.no_grad():  # g>0 by default; argmax over the policy slice
        for heading, reverse in ((RIGHT, LEFT), (LEFT, RIGHT)):
            rise = _roll_levels(model, L, [2, 6], heading)  # concentration rose
            drop = _roll_levels(model, L, [6, 2], heading)  # concentration fell
            assert int(rise[cs_a].argmax()) == heading  # keep heading while improving
            assert int(drop[cs_a].argmax()) == reverse  # tumble (reverse) on a drop


def test_integrator_is_the_memory_no_weight_change():
    # Same just-sensed level, SAME last action — but a different recent history
    # (carried only by the leaky baseline) flips the action distribution. No weights
    # move: the adaptation is integrator state, not learning.
    model, chem = _build()
    L, cs_a = chem.levels, action_slice(chem)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    # Both streams fold the identical final level 4 after heading RIGHT; only the
    # preceding concentrations differ, so the baseline (memory) differs.
    high_hist = _roll_levels(model, L, [7, 7, 7, 4], RIGHT)  # baseline high -> 4 is a DROP
    low_hist = _roll_levels(model, L, [0, 0, 0, 4], RIGHT)  # baseline low  -> 4 is a RISE
    assert int(high_hist[cs_a].argmax()) == LEFT  # remembered-high -> reverse
    assert int(low_hist[cs_a].argmax()) == RIGHT  # remembered-low  -> keep
    assert not torch.allclose(high_hist[cs_a], low_hist[cs_a])
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n])  # no weight change at decode time


def test_step_matches_forward_and_flop_sum():
    model, chem = _build(horizon=8)
    model.eval()
    toks = [
        3,
        action_token(chem, RIGHT),
        5,
        action_token(chem, RIGHT),
        6,
        action_token(chem, LEFT),
        2,
    ]
    forward = model(torch.tensor([toks]))[0]
    state = model.init_prequential_state()
    summed_forward = 0
    for pos, tok in enumerate(toks):
        state, logits, f = model.step(state, tok, pos)
        assert torch.allclose(logits, forward[pos], atol=1e-5)  # step ≡ forward per position
        summed_forward += f.forward
    # Summed step FLOPs == T · decode_step_flops (context-independent per step).
    assert summed_forward == len(toks) * model.decode_step_flops(0).forward


def test_flop_honesty_pointwise_charged_not_free():
    model, chem = _build(levels=8, horizon=16)
    levels = chem.levels
    # Hand-derived per-step pointwise op count (charge == code; NOT scored as free).
    # Conservative even-branch charge incl. every elementwise op: leak 5 + sense 4 +
    # action 8 (g·s, -keep, compare, 2 where, stack-3) + center 6 + cat-action 3 + 5·levels.
    per_step = 5 + 4 + 8 + 6 + 3 + 5 * levels
    assert per_step == 26 + 5 * levels
    for seq_len in (1, 7, 48):
        fb = model.flops(seq_len)
        assert fb.forward == seq_len * per_step > 0  # pointwise work charged, never omitted
        assert fb.backward == 2 * seq_len * per_step  # distilled scalars -> standard 2x
    # The online step charges nonzero forward and ZERO backward (no weight update).
    state = model.init_prequential_state()
    _, _, f = model.step(state, 3, 0)
    assert f.forward == per_step > 0
    assert f.backward == 0


def test_num_params_is_the_five_scalars():
    model, _ = _build()
    assert model.num_params() == 5  # leak_logit, g, stay_bias, climb, sharpness
    assert all(p.numel() == 1 and p.requires_grad for p in model.parameters())


def test_end_to_end_beats_random_and_improves(tmp_path):
    chem = ChemoConfig(width=16, levels=8, horizon=24)
    model, _ = _build(levels=8, horizon=24)
    res = evaluate_control(model, chem, split="eval", n_episodes=48, seed=0, device=DEVICE)

    # random-policy floor on the same held-out split (~0.37).
    floor = []
    for s in range(48):
        e = ChemoEnv(chem, split="eval", seed=0 * 100003 + s)
        pol, c, tot = RandomPolicy(seed=s), e.reset(), 0.0
        for _ in range(chem.horizon):
            c, r = e.step(pol.act(c))
            tot += r
        floor.append(tot / chem.horizon)
    random_floor = float(np.mean(floor))

    assert math.isfinite(res.mean_reward) and math.isfinite(res.world_model_bits)
    assert res.mean_reward > random_floor  # untrained mechanism clears the floor
    assert res.second_half_reward > res.first_half_reward  # climbs onto the peak in-context

    # A runs/control row at ≈0 distillation (budget below one train step -> 0 steps),
    # whose HONEST total FLOPs (eval rollout only) is far below the transformer bar.
    cfg = ControlTrainConfig(
        model="chemotaxis_min",
        model_config={},
        flop_budget=1.0,  # < one train step -> pure eval, ≈0 distillation
        batch_size=32,
        horizon=24,
        eval_episodes=48,
        seed=0,
        run_name="chemotaxis_min-smoke",
    )
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs")
    assert summary.steps == 0
    assert summary.total_flops > 0
    assert summary.total_flops < TRANSFORMER_BAR_MIN_FLOPS / 1000  # FAR below the bar

    table, png = regenerate_control(
        tmp_path / "runs", table_path=tmp_path / "lb.md", plot_path=tmp_path / "lb.png"
    )
    assert png.exists() and png.stat().st_size > 0
    assert "control" in table and "regret" in table
    assert "chemotaxis_min" in table
