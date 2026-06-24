"""Reservoir + distilled-frozen linear readout (Task C.A.1).

The headline the harness measures is regret-per-total-FLOP at fixed params; these
tests pin the mechanism that makes the claim honest: a FROZEN echo-state core
(counted in params, 0 backward) + a trained linear readout (the only backward),
``step`` ≡ ``forward``, and an end-to-end control rollout that clears the random floor.
"""

import numpy as np
import torch

from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import ChemoConfig, ChemoEnv, RandomPolicy, vocab_size
from smolml.flops import gather_flops, matmul_flops, pointwise_flops
from smolml.leaderboard import regenerate_control
from smolml.models import build_model
from smolml.models.reservoir import Reservoir, ReservoirConfig

CPU = torch.device("cpu")
TINY = {"d_res": 24, "vocab_size": 11, "max_seq_len": 64}


def _model(**overrides) -> Reservoir:
    return Reservoir(ReservoirConfig(**{**TINY, **overrides}))


def _per_token_forward(d: int, v: int) -> int:
    """The hand-derived per-token forward sum (recurrence + readout), independent of
    the model's own helper, so the test pins the FLOP charge to first principles."""
    return (
        matmul_flops(1, d, d)  # W_res @ h
        + gather_flops(d)  # x = W_in[:, token]
        + pointwise_flops(d)  # pre = x + W_res@h
        + pointwise_flops(d, per_elem=4)  # (1-leak)*h + leak*tanh(pre)
        + matmul_flops(1, v, d)  # W_out @ h
        + pointwise_flops(v)  # + b_out
    )


# --- forward shape & determinism ---------------------------------------------
def test_forward_shape_and_determinism():
    m = _model()
    idx = torch.randint(0, 11, (3, 9))
    out = m(idx)
    assert out.shape == (3, 9, 11)
    # A fixed seed pins the whole forward output (frozen core + seeded readout init).
    assert torch.equal(out, _model()(idx))
    # A different seed gives a different reservoir (the seed actually wires the core).
    assert not torch.equal(out, _model(seed=1)(idx))


def test_spectral_radius_is_rescaled_to_rho():
    # The echo-state property hinges on rho<1; the init must actually rescale W_res.
    m = _model(d_res=64, spectral_radius=0.9)
    radius = torch.linalg.eigvals(m.core.W_res).abs().max().item()
    assert abs(radius - 0.9) < 1e-4


# --- frozen core: counted, excluded from the optimizer, 0 backward in training -
def test_core_is_frozen_counted_and_excluded_from_optimizer():
    m = _model(d_res=374)
    assert m.core.W_in.requires_grad is False
    assert m.core.W_res.requires_grad is False
    # Counted in num_params for memory parity with the bar (frozen params included).
    d, v = 374, 11
    assert m.num_params() == d * v + d * d + v * d + v  # W_in + W_res + W_out + b_out
    # ... but NOT handed to the optimizer (only the readout is trained).
    opt = m.configure_optimizer(lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95))
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.core.W_in) not in opt_ids and id(m.core.W_res) not in opt_ids
    assert id(m.readout.weight) in opt_ids and id(m.readout.bias) in opt_ids


def test_train_step_changes_only_the_readout():
    m = _model()
    opt = m.configure_optimizer(lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95))
    win0, wres0 = m.core.W_in.clone(), m.core.W_res.clone()
    wout0, bout0 = m.readout.weight.clone(), m.readout.bias.clone()
    x = torch.randint(0, 11, (4, 16))
    y = torch.randint(0, 11, (4, 16))
    m.train_step((x, y), opt)
    assert torch.equal(win0, m.core.W_in) and torch.equal(wres0, m.core.W_res)
    assert not torch.equal(wout0, m.readout.weight) and not torch.equal(bout0, m.readout.bias)


# --- FLOP analytic match: backward is readout-only, NOT 2x forward ------------
def test_flops_match_hand_derivation_and_charge_readout_only_backward():
    d, v, t = 374, 11, 20
    m = _model(d_res=d)
    fb = m.flops(t)
    assert fb.forward == t * _per_token_forward(d, v)
    # The frozen recurrence gets 0 backward; only dW_out (outer product) + db_out (bias) charged.
    assert fb.backward == t * (matmul_flops(1, v, d) + pointwise_flops(v))
    # Explicitly NOT the default 2x-forward backprop tax.
    assert fb.backward != 2 * fb.forward


# --- step == forward, and step FLOPs are context-independent ------------------
def test_step_matches_forward_and_flop_accounting():
    m = _model()
    toks = torch.randint(0, 11, (1, 11))
    state = m.init_prequential_state()
    total_forward = 0
    last_logits = None
    for i in range(toks.shape[1]):
        state, last_logits, f = m.step(state, int(toks[0, i]), i)
        total_forward += f.forward
        assert f.backward == 0  # the readout is frozen at decode
    # Iterated step's final-position logits == forward()'s last position.
    assert torch.allclose(last_logits, m(toks)[0, -1], atol=1e-5)
    # Summed step FLOPs == T x the (context-independent) per-step decode cost.
    t = toks.shape[1]
    assert total_forward == t * m.decode_step_flops(0).forward
    assert m.decode_step_flops(0).forward == m.decode_step_flops(999).forward


# --- param budget: fits under the transformer bar ------------------------------
def test_num_params_under_the_bar():
    m = build_model("reservoir", {"vocab_size": 11, "max_seq_len": 129})
    assert m.config.d_res == 374
    assert m.num_params() == 148_115
    assert m.num_params() <= 148_608


# --- end-to-end smoke: clears the random floor and improves within episodes ----
def test_end_to_end_beats_random_and_improves(tmp_path):
    horizon, episodes = 16, 24
    chem = ChemoConfig(width=16, levels=8, horizon=horizon)
    mc = {"vocab_size": vocab_size(chem), "max_seq_len": 2 * horizon + 1}
    step_flops = build_model("reservoir", mc).flops(2 * horizon).scale(32).total
    cfg = ControlTrainConfig(
        model="reservoir",
        model_config={},
        flop_budget=step_flops * 400,
        batch_size=32,
        horizon=horizon,
        eval_interval=10**9,  # final point only
        eval_episodes=episodes,
        seed=0,
        run_name="reservoir-ctl-smoke",
    )
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs")

    # random-policy floor on the same held-out split
    rng_floor = []
    for s in range(episodes):
        e = ChemoEnv(chem, split="eval", seed=0 * 100003 + s)
        pol, c, tot = RandomPolicy(seed=s), e.reset(), 0.0
        for _ in range(horizon):
            c, r = e.step(pol.act(c))
            tot += r
        rng_floor.append(tot / horizon)
    floor = float(np.mean(rng_floor))

    assert np.isfinite(summary.final_reward)
    assert summary.final_reward > floor  # the distilled readout beats random
    assert summary.second_half_reward > summary.first_half_reward  # in-context improvement
    assert (tmp_path / "runs" / f"{summary.run}.jsonl").exists()

    # the leaderboard regenerates a table + plot that ranks the run.
    table, png = regenerate_control(
        tmp_path / "runs", table_path=tmp_path / "lb.md", plot_path=tmp_path / "lb.png"
    )
    assert png.exists() and png.stat().st_size > 0
    assert "reservoir-ctl-smoke" in table and "regret" in table
