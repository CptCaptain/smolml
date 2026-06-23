"""Task 7 end-to-end smoke: the distilled transformer baseline's held-out mean
reward clears the random-policy floor, and the control leaderboard regenerates a
table + plot from the run log."""

import numpy as np

from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import ChemoConfig, ChemoEnv, RandomPolicy, vocab_size
from smolml.leaderboard import regenerate_control
from smolml.models import build_model


def test_end_to_end_beats_random_and_improves(tmp_path):
    chem = ChemoConfig(width=16, levels=8, horizon=24)
    mc = {
        "d_model": 64,
        "n_layers": 3,
        "n_heads": 4,
        "vocab_size": vocab_size(chem),
        "max_seq_len": 2 * 24 + 1,
    }
    step_flops = build_model("transformer", mc).flops(2 * 24).scale(32).total
    cfg = ControlTrainConfig(
        model_config={"d_model": 64, "n_layers": 3, "n_heads": 4},
        flop_budget=step_flops * 400,
        batch_size=32,
        horizon=24,
        eval_interval=50,
        eval_episodes=48,
        seed=0,
        run_name="ctl-smoke",
    )
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs")

    # random-policy floor on the same held-out split
    rng_floor = []
    for s in range(48):
        e = ChemoEnv(chem, split="eval", seed=0 * 100003 + s)
        pol, c, tot = RandomPolicy(seed=s), e.reset(), 0.0
        for _ in range(chem.horizon):
            c, r = e.step(pol.act(c))
            tot += r
        rng_floor.append(tot / chem.horizon)
    floor = float(np.mean(rng_floor))

    # the trained model beats the random floor AND improves within an episode
    # (the headline in-context signal: mean 2nd-half reward > 1st-half)
    assert summary.final_reward > floor
    assert summary.second_half_reward > summary.first_half_reward

    table, png = regenerate_control(
        tmp_path / "runs", table_path=tmp_path / "lb.md", plot_path=tmp_path / "lb.png"
    )
    assert png.exists() and png.stat().st_size > 0
    assert "control" in table and "regret" in table
