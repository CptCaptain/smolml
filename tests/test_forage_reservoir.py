"""C.A.4 control sibling: reservoir core + plastic readout with the forage reward decode.

``forage_reservoir`` is ``reservoir_plastic`` (frozen ``_ReservoirCore`` + online plastic
readout) with the forage reward proxy (``obs % 3 - 1``). It is the generic-capacity contrast to
``forage_min``: same online local rule, no per-type credit-assignment structure.
"""

import numpy as np
import torch

from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.forage import (
    EAT,
    N_ACTIONS,
    REWARD_LEVELS,
    ForageConfig,
    ForageEnv,
    forage_env_spec,
    vocab_size,
)
from smolml.leaderboard import regenerate_control
from smolml.models import build_model
from smolml.models.reservoir import ForageReservoir, _ReservoirCore

CPU = torch.device("cpu")


def _tiny(**ov):
    return build_model("forage_reservoir", {"vocab_size": 12, "max_seq_len": 64, **ov})


def test_reward_decode_is_forage_not_monotone_token():
    m = _tiny()
    lv = m._levels  # 9 = 3K
    assert m._decode_reward(0 * REWARD_LEVELS + 0, lv) == -1.0  # type0 poison
    assert m._decode_reward(1 * REWARD_LEVELS + 2, lv) == +1.0  # type1 reward
    assert m._decode_reward(2 * REWARD_LEVELS + 1, lv) == 0.0  # type2 move
    assert m._REWARD_DECODE_OPS == 2


def test_reuses_frozen_core_and_param_parity():
    m = build_model("forage_reservoir", {"vocab_size": 12, "max_seq_len": 129})
    assert isinstance(m.core, _ReservoirCore)
    assert isinstance(m, ForageReservoir)
    assert m.num_params() <= 148_608  # memory-parity with the transformer bar


def test_online_update_changes_cache_not_parameters():
    m = _tiny()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    obs_len = REWARD_LEVELS * 3
    state = m.init_prequential_state()
    state, _, f0 = m.step(state, 1 * REWARD_LEVELS + 1, 0)  # obs (type1, r0) pos<2 -> no update
    state, _, f1 = m.step(state, obs_len + EAT, 1)  # action fold (EAT)
    w_pre = state.cache.W.clone()
    state, _, f2 = m.step(state, 1 * REWARD_LEVELS + 2, 2)  # obs (type1, r+1) pos>=2 -> update
    assert f0.backward == 0 and f1.backward == 0 and f2.backward > 0
    assert not torch.allclose(w_pre, state.cache.W)  # plastic readout adapted
    for n, p in m.named_parameters():
        assert torch.equal(p, before[n]), f"eval mutated parameter {n}"


def test_policy_update_flop_charges_the_extra_decode_op():
    # forage decode (mod + sub) costs one more pointwise op than the base divide.
    base = build_model("reservoir_plastic", {"vocab_size": 12, "max_seq_len": 64, "d_res": 373})
    forage = _tiny()
    assert forage._policy_update_flops() == base._policy_update_flops() + 1


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
    """Headline of the CONTROL sibling: a purely-online local rule on a generic reservoir clears
    the random floor. Its within-episode signal may be weaker/seed-sensitive than ``forage_min``
    (it conflates types; this shape lost on chemotaxis) — failures are data; inits are NOT fudged
    to manufacture a win."""
    horizon, episodes = 64, 32
    fcfg = ForageConfig(horizon=horizon)
    mc = {"vocab_size": vocab_size(fcfg), "max_seq_len": 2 * horizon + 1}
    step_flops = build_model("forage_reservoir", mc).flops(2 * horizon).scale(32).total
    cfg = ControlTrainConfig(
        model="forage_reservoir",
        model_config={},
        flop_budget=step_flops * 0.5,  # < one train step -> 0 distillation steps
        batch_size=32,
        horizon=horizon,
        eval_interval=10**9,
        eval_episodes=episodes,
        seed=0,
        env_name="forage",
        run_name="forage-reservoir-zero-distill",
    )
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs", env_spec=forage_env_spec(fcfg))
    assert summary.steps == 0  # ~0 distillation: ALL learning is online in step
    assert summary.total_flops > 0
    floor = _random_floor(fcfg, episodes)
    assert np.isfinite(summary.final_reward)
    assert summary.final_reward > floor  # the gradient-free online rule clears the floor
    table, png = regenerate_control(
        tmp_path / "runs", table_path=tmp_path / "lb.md", plot_path=tmp_path / "lb.png"
    )
    assert png.exists() and png.stat().st_size > 0
    assert "forage-reservoir-zero-distill" in table and "regret" in table
