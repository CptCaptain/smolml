"""Phase-A seam pin test: the generalized control spine must keep chemotaxis
BIT-IDENTICAL. Runs ``distill_train_run`` end-to-end on chemotaxis at a tiny fixed
budget+seed and asserts regret/reward/world-model-bits/total-FLOPs equal the values
captured on the pre-refactor (hardcoded-ChemoEnv) spine. FLOPs alone miss seed drift,
so reward/regret/bits are pinned too (a drifted seed shifts them by >> tol)."""

import math

from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import ChemoConfig, vocab_size
from smolml.models import build_model

# Captured on the pre-refactor spine (hardcoded ChemoEnv / conc_slice / RunAndTumble):
# transformer d_model=32/n_layers=2/n_heads=4, batch=8, horizon=16, seed=0, CPU,
# OMP_NUM_THREADS=4, flop_budget = step_flops * 30 (-> 30 distillation steps).
PIN_STEPS = 30
PIN_REGRET = 0.5292334495293068
PIN_REWARD = 0.37559575097604936
PIN_WM_BITS = 2.873976560642574
PIN_TOTAL_FLOPS = 1259847680


def test_chemotaxis_bit_identical_after_seam_refactor(tmp_path):
    chem = ChemoConfig(width=16, levels=8, sigma=2.0, horizon=16)
    mc = {
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "vocab_size": vocab_size(chem),
        "max_seq_len": 2 * chem.horizon + 1,
    }
    step_flops = build_model("transformer", mc).flops(2 * chem.horizon).scale(8).total
    cfg = ControlTrainConfig(
        model="transformer",
        model_config={"d_model": 32, "n_layers": 2, "n_heads": 4},
        flop_budget=step_flops * 30,
        batch_size=8,
        horizon=16,
        eval_interval=5,
        eval_episodes=8,
        seed=0,
        device="cpu",
        run_name="pin",
    )
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs")

    assert summary.steps == PIN_STEPS
    assert summary.total_flops == PIN_TOTAL_FLOPS
    assert math.isclose(summary.final_regret, PIN_REGRET, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(summary.final_reward, PIN_REWARD, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(summary.final_world_model_bits, PIN_WM_BITS, rel_tol=1e-9, abs_tol=1e-12)
