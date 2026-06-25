"""Reservoir control-rung candidate (C.A.1): distill-train the frozen echo-state core
+ linear readout across a small FLOP sweep, eval on held-out episodes, write the control
leaderboard + a sample rollout raster, and print the comparison to the transformer bar.

The headline is regret-vs-oracle per total FLOP at fixed params (148,115 ≤ the bar's
148,608): the expensive recurrence is never trained (0 backward), so only the readout's
``dW_out`` outer product is paid for. Budgets are multiples of one training step.

Run (CPU, synthetic; minutes)::

    uv run python -m smolml.experiments.reservoir_control
"""

from smolml.control_eval import evaluate_control
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import ChemoConfig, chemo_env_spec, vocab_size
from smolml.envs.render import render_rollout
from smolml.leaderboard import regenerate_control
from smolml.models import build_model

RUNS_DIR = "runs/control"
HORIZON = 32
MODEL = {"d_res": 374, "leak": 0.6, "spectral_radius": 0.9, "seed": 0}
STEP_SWEEP = (150, 600, 1500)  # FLOP budgets as multiples of one train step
BATCH_SIZE = 32
# transformer bar (held-out drifting eval, 148,608 params): regret @ total FLOPs.
BAR = ((0.229, 2.97e11), (0.171, 1.19e12), (0.141, 2.96e12))


def main() -> None:
    chem = ChemoConfig(horizon=HORIZON)
    mc = {**MODEL, "vocab_size": vocab_size(chem), "max_seq_len": 2 * HORIZON + 1}
    model = build_model("reservoir", mc)
    step_flops = model.flops(2 * HORIZON).scale(BATCH_SIZE).total
    print(f"reservoir params={model.num_params()} (bar: 148,608); d_res={MODEL['d_res']}")

    trained = None
    for steps in STEP_SWEEP:
        cfg = ControlTrainConfig(
            model="reservoir",
            model_config=dict(MODEL),
            flop_budget=step_flops * steps,
            batch_size=BATCH_SIZE,
            horizon=HORIZON,
            eval_episodes=64,
            eval_interval=10**9,  # cross-run bar needs the final point only
            run_name=f"reservoir-control-{steps}steps",
        )
        summary, trained = distill_train_run(cfg, runs_dir=RUNS_DIR, return_model=True)
        print(
            f"steps={steps:5d}  regret={summary.final_regret:.4f}  "
            f"reward={summary.final_reward:.4f}  wm_bits={summary.final_world_model_bits:.4f}  "
            f"flops={summary.total_flops:.3e}"
        )

    print("\ntransformer bar (regret @ total FLOPs):")
    for regret, flops in BAR:
        print(f"  regret={regret:.4f}  flops={flops:.3e}")

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
    render_rollout(res.trajectory, f"{RUNS_DIR}/reservoir_sample_rollout.png")
    print(f"sample rollout: {RUNS_DIR}/reservoir_sample_rollout.png")


if __name__ == "__main__":
    main()
