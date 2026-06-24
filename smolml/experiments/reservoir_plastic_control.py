"""Reservoir + ONLINE plastic-readout control candidate (C.A.1b): the ~0-distillation point.

The headline (vs the distilled ``reservoir`` of C.A.1): set the FLOP budget BELOW one
training step, so the distillation loop runs **0 steps** and ALL learning happens online in
``step`` — a gradient-free local rule (a softmax delta rule on the world-model ``conc_slice``
+ a reward-modulated Hebbian rule with a leaky baseline on the policy ``action_slice``).
Every adaptation FLOP is charged to ``step``'s ``backward`` (ADR 0004), summed into the eval
total by ``evaluate_control``, and written as a ``runs/control`` row alongside the bar.

The frozen echo-state core is reused unchanged from ``reservoir`` (148,115 params ≤ the bar's
148,608); the expensive recurrence is never trained (0 distillation, 0 backprop). This is a
source-(iv) probe: does a purely-online local rule extract regret-reduction per total FLOP?

Run (CPU, synthetic; minutes)::

    uv run python -m smolml.experiments.reservoir_plastic_control
"""

import numpy as np

from smolml.control_eval import evaluate_control
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.chemotaxis import ChemoConfig, ChemoEnv, RandomPolicy, chemo_env_spec, vocab_size
from smolml.envs.render import render_rollout
from smolml.leaderboard import regenerate_control
from smolml.models import build_model

RUNS_DIR = "runs/control"
HORIZON = 32
EPISODES = 64
SEED = 0
MODEL = {"d_res": 374, "leak": 0.6, "spectral_radius": 0.9, "seed": 0}
# transformer bar (held-out drifting eval, 148,608 params): regret @ total FLOPs.
BAR = ((0.229, 2.97e11), (0.171, 1.19e12), (0.141, 2.96e12))


def random_floor(chem: ChemoConfig, episodes: int, seed: int) -> float:
    """Mean reward of a uniform-random policy on the same held-out split (the floor)."""
    floor = []
    for s in range(episodes):
        env = ChemoEnv(chem, split="eval", seed=seed * 100003 + s)
        pol, c, tot = RandomPolicy(seed=s), env.reset(), 0.0
        for _ in range(chem.horizon):
            c, r = env.step(pol.act(c))
            tot += r
        floor.append(tot / chem.horizon)
    return float(np.mean(floor))


def main() -> None:
    chem = ChemoConfig(horizon=HORIZON)
    mc = {**MODEL, "vocab_size": vocab_size(chem), "max_seq_len": 2 * HORIZON + 1}
    model = build_model("reservoir_plastic", mc)
    step_flops = model.flops(2 * HORIZON).scale(32).total
    print(
        f"reservoir_plastic params={model.num_params()} (bar: 148,608); "
        f"d_res={MODEL['d_res']}  one-train-step={step_flops:.3e} FLOPs"
    )

    # Headline: budget BELOW one train step => 0 distillation steps; all learning online.
    cfg = ControlTrainConfig(
        model="reservoir_plastic",
        model_config=dict(MODEL),
        flop_budget=step_flops * 0.5,
        batch_size=32,
        horizon=HORIZON,
        eval_episodes=EPISODES,
        eval_interval=10**9,
        seed=SEED,
        run_name="reservoir-plastic-zero-distill",
    )
    summary, trained = distill_train_run(cfg, runs_dir=RUNS_DIR, return_model=True)

    floor = random_floor(chem, EPISODES, SEED)
    beats = summary.final_reward > floor
    improved = summary.second_half_reward - summary.first_half_reward
    print(
        f"\n~0-distillation (steps={summary.steps}): "
        f"reward={summary.final_reward:.4f}  regret={summary.final_regret:.4f}  "
        f"wm_bits={summary.final_world_model_bits:.4f}  total_flops={summary.total_flops:.3e}"
    )
    print(
        f"random floor={floor:.4f}  ->  {'BEATS' if beats else 'BELOW'} the floor "
        f"(margin {summary.final_reward - floor:+.4f})"
    )
    print(
        f"within-episode learning: 2nd-half={summary.second_half_reward:.4f} vs "
        f"1st-half={summary.first_half_reward:.4f} ({improved:+.4f})"
    )

    print("\ntransformer bar (regret @ total FLOPs):")
    for regret, flops in BAR:
        print(f"  regret={regret:.4f}  flops={flops:.3e}")

    table, png = regenerate_control(
        RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md", plot_path=f"{RUNS_DIR}/leaderboard.png"
    )
    print("\n" + table + f"\nplot: {png}")

    res = evaluate_control(
        trained,
        chemo_env_spec(chem),
        split="eval",
        n_episodes=1,
        seed=SEED,
        device=next(trained.parameters()).device,
        record=True,
    )
    render_rollout(res.trajectory, f"{RUNS_DIR}/reservoir_plastic_sample_rollout.png")
    print(f"sample rollout: {RUNS_DIR}/reservoir_plastic_sample_rollout.png")


if __name__ == "__main__":
    main()
