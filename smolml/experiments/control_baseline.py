"""Transformer control-rung baseline (the bar): distill-train across a small FLOP
sweep, eval on held-out episodes, write the leaderboard + a sample rollout raster.

Budgets are expressed as multiples of one training step (robust to model size), so
the sweep always trains a meaningful number of steps regardless of d_model/horizon.

Run (CPU, synthetic; minutes)::

    uv run python -m smolml.experiments.control_baseline
"""

from smolml.control_eval import evaluate_control
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import ChemoConfig, chemo_env_spec, vocab_size
from smolml.envs.render import render_rollout
from smolml.leaderboard import regenerate_control
from smolml.models import build_model

RUNS_DIR = "runs/control"
HORIZON = 32
MODEL = {"d_model": 64, "n_layers": 3, "n_heads": 4}
STEP_SWEEP = (150, 600, 1500)  # FLOP budgets as multiples of one train step
BATCH_SIZE = 32


def main() -> None:
    chem = ChemoConfig(horizon=HORIZON)
    mc = {**MODEL, "vocab_size": vocab_size(chem), "max_seq_len": 2 * HORIZON + 1}
    step_flops = build_model("transformer", mc).flops(2 * HORIZON).scale(BATCH_SIZE).total

    trained = None
    for steps in STEP_SWEEP:
        cfg = ControlTrainConfig(
            model_config=dict(MODEL),
            flop_budget=step_flops * steps,
            batch_size=BATCH_SIZE,
            horizon=HORIZON,
            eval_episodes=64,
            eval_interval=10**9,  # cross-run bar needs the final point only
            run_name=f"transformer-control-{steps}steps",
        )
        summary, trained = distill_train_run(cfg, runs_dir=RUNS_DIR, return_model=True)
        print(
            f"steps={steps:5d}  regret={summary.final_regret:.4f}  "
            f"reward={summary.final_reward:.4f}  wm_bits={summary.final_world_model_bits:.4f}  "
            f"flops={summary.total_flops:.3e}"
        )

    table, png = regenerate_control(
        RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md", plot_path=f"{RUNS_DIR}/leaderboard.png"
    )
    print("\n" + table + f"\nplot: {png}")

    # a sample rollout raster from the largest-budget (best) trained model
    res = evaluate_control(
        trained,
        chemo_env_spec(chem),
        split="eval",
        n_episodes=1,
        seed=0,
        device=next(trained.parameters()).device,
        record=True,
    )
    render_rollout(res.trajectory, f"{RUNS_DIR}/sample_rollout.png")
    print(f"sample rollout: {RUNS_DIR}/sample_rollout.png")


if __name__ == "__main__":
    main()
