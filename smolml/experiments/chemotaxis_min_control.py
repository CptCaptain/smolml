"""Minimal-organism control candidate (`chemotaxis_min`) on the chemotaxis rung.

The bet is the **FLOP axis**: five learnable scalars and a leaky-integrator memory,
adapting almost entirely **online in `step`** at ≈0 distillation, so total FLOPs is
dominated by the (cheap) eval rollout — orders of magnitude below the distilled
transformer bar. This driver mirrors ``control_baseline.py``: build the model, eval
held-out across a tiny distillation sweep (a ≈0-distillation point plus a couple of
short ones to show the scalars *can* be tuned), write ``runs/control/*.jsonl`` rows,
regenerate the board, and print the comparison to the bar — emphasizing total FLOPs.

Run (CPU, synthetic; seconds)::

    uv run python -m smolml.experiments.chemotaxis_min_control
"""

from smolml.control_eval import evaluate_control
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import ChemoConfig, vocab_size
from smolml.envs.render import render_rollout
from smolml.leaderboard import regenerate_control
from smolml.models import build_model

RUNS_DIR = "runs/control"
HORIZON = 32
BATCH_SIZE = 32
# Distillation budgets as multiples of one train step; 0 == the ≈0-distillation point
# (total FLOPs == the eval rollout), the cleanest source-(iv) reading.
STEP_SWEEP = (0, 100, 400)

# The transformer baseline bar (held-out drifting eval, 148,608 params).
BAR = "transformer bar: regret 0.229->0.171->0.141 at total FLOPs 2.97e11->1.19e12->2.96e12"


def main() -> None:
    chem = ChemoConfig(horizon=HORIZON)
    mc = {"vocab_size": vocab_size(chem), "max_seq_len": 2 * HORIZON + 1}
    step_flops = build_model("chemotaxis_min", mc).flops(2 * HORIZON).scale(BATCH_SIZE).total

    trained = None
    print(BAR)
    for steps in STEP_SWEEP:
        cfg = ControlTrainConfig(
            model="chemotaxis_min",
            model_config={},
            # 0 steps -> a budget below one train step (pure eval); else `steps` steps.
            flop_budget=step_flops * steps if steps else step_flops * 0.5,
            batch_size=BATCH_SIZE,
            horizon=HORIZON,
            eval_episodes=64,
            eval_interval=10**9,  # cross-run bar needs the final point only
            run_name=f"chemotaxis_min-control-{steps}steps",
        )
        summary, trained = distill_train_run(cfg, runs_dir=RUNS_DIR, return_model=True)
        print(
            f"steps={steps:5d}  regret={summary.final_regret:.4f}  "
            f"reward={summary.final_reward:.4f}  wm_bits={summary.final_world_model_bits:.4f}  "
            f"total_flops={summary.total_flops:.3e}  (params={summary.params})"
        )

    table, png = regenerate_control(
        RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md", plot_path=f"{RUNS_DIR}/leaderboard.png"
    )
    print("\n" + table + f"\nplot: {png}")

    res = evaluate_control(
        trained,
        chem,
        split="eval",
        n_episodes=1,
        seed=0,
        device=next(trained.parameters()).device,
        record=True,
    )
    render_rollout(res.trajectory, f"{RUNS_DIR}/chemotaxis_min_rollout.png")
    print(f"sample rollout: {RUNS_DIR}/chemotaxis_min_rollout.png")


if __name__ == "__main__":
    main()
