"""Reservoir core + plastic readout control sibling (C.A.4) — the generic-capacity contrast.

``forage_reservoir`` is the ``reservoir_plastic`` mechanism (frozen echo-state core + online
plastic readout) on the forage rung, decoding the eat-reward from the combined obs. It has the
SAME online local rule as ``reservoir_plastic`` but no per-type credit-assignment structure: the
reservoir state conflates cue types across the trajectory. Running it beside ``forage_min``
isolates that STRUCTURE, not raw capacity, is the regret-per-FLOP lever (this shape lost on
chemotaxis). ~148k params (memory-parity with the bar); the per-step recurrence is O(d_res^2), so
its FLOP floor is far above ``forage_min``'s pointwise tracker.

Run (CPU, synthetic; minutes)::

    uv run python -m smolml.experiments.forage_reservoir_control
"""

import numpy as np

from smolml.control_eval import evaluate_control
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.forage import N_ACTIONS, ForageConfig, ForageEnv, forage_env_spec, vocab_size
from smolml.envs.render import render_rollout
from smolml.leaderboard import regenerate_control
from smolml.models import build_model

RUNS_DIR = "runs/forage"
HORIZON = 64
EPISODES = 32  # held-out eval; matches forage_baseline EVAL_EPISODES for an apples-to-apples curve
SEED = 0
MODEL: dict[str, object] = {"d_res": 373, "leak": 0.6, "spectral_radius": 0.9, "seed": 0}
# Distillation curve points, in reference training steps (0 => the ~0-distillation headline).
# Fewer/cheaper points than forage_min: the O(d_res^2) recurrence makes each step costly.
CURVE_STEPS = (0, 50, 200)


def random_floor(fcfg: ForageConfig, episodes: int, seed: int) -> float:
    """Mean reward of a uniform-random policy on the held-out split (the floor)."""
    rng = np.random.default_rng(seed)
    floor = []
    for s in range(episodes):
        env = ForageEnv(fcfg, split="eval", seed=seed * 100003 + s)
        env.reset()
        tot = 0.0
        for _ in range(fcfg.horizon):
            _, r = env.step(int(rng.integers(N_ACTIONS)))
            tot += r
        floor.append(tot / fcfg.horizon)
    return float(np.mean(floor))


def main() -> None:
    fcfg = ForageConfig(horizon=HORIZON)
    spec = forage_env_spec(fcfg)
    mc = {**MODEL, "vocab_size": vocab_size(fcfg), "max_seq_len": 2 * HORIZON + 1}
    model = build_model("forage_reservoir", mc)
    ref_step = model.flops(2 * HORIZON).scale(32).total
    print(
        f"forage_reservoir params={model.num_params()} (bar: 148,608); "
        f"d_res={MODEL['d_res']}  one-train-step={ref_step:.3e} FLOPs"
    )

    floor = random_floor(fcfg, EPISODES, SEED)
    print(f"\nrandom floor (held-out): {floor:+.4f}")
    print("\nregret-vs-total-FLOP curve (forage_reservoir):")
    trained = None
    for steps in CURVE_STEPS:
        cfg = ControlTrainConfig(
            model="forage_reservoir",
            model_config=dict(MODEL),
            flop_budget=ref_step * (steps + 0.5),
            batch_size=32,
            horizon=HORIZON,
            eval_episodes=EPISODES,
            eval_interval=10**9,
            seed=SEED,
            env_name="forage",
            run_name=f"forage-reservoir-{steps}steps",
        )
        summary, trained = distill_train_run(
            cfg, runs_dir=RUNS_DIR, return_model=True, env_spec=spec
        )
        tag = "~0-distill" if steps == 0 else f"{summary.steps:4d} steps"
        print(
            f"  {tag:>11s}  regret={summary.final_regret:.4f}  reward={summary.final_reward:+.4f}  "
            f"2nd-1st={summary.second_half_reward - summary.first_half_reward:+.4f}  "
            f"total_flops={summary.total_flops:.3e}"
        )

    table, png = regenerate_control(
        RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md", plot_path=f"{RUNS_DIR}/leaderboard.png"
    )
    print("\n" + table + f"\nplot: {png}")

    res = evaluate_control(
        trained,
        spec,
        split="eval",
        n_episodes=1,
        seed=SEED,
        device=next(trained.parameters()).device,
        record=True,
    )
    render_rollout(res.trajectory, f"{RUNS_DIR}/forage_reservoir_sample_rollout.png")
    print(f"sample rollout: {RUNS_DIR}/forage_reservoir_sample_rollout.png")


if __name__ == "__main__":
    main()
