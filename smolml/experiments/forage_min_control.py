"""Per-type contingency-tracker control candidate (C.A.4) — the ~0-distillation headline +
a regret-vs-total-FLOP curve on the forage rung.

``forage_min`` is the bandit analogue of ``chemotaxis_min``: an in-context per-type value vector
updated by a local delta rule in ``step`` (no weight change), a distilled-scalar softmax policy.
The headline sets the FLOP budget BELOW one training step, so the distillation loop runs **0
steps** and ALL learning happens online — every adaptation FLOP charged to ``step`` (ADR 0004),
summed into the eval total by ``evaluate_control``, written as a ``runs/forage`` row beside the
transformer bar. The curve adds a few distilled points to show the regret-vs-FLOP shape.

Run (CPU, synthetic; seconds)::

    uv run python -m smolml.experiments.forage_min_control
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
EPISODES = 64
SEED = 0
MODEL: dict[str, object] = {}  # forage_min takes only injected vocab/seq + its scalar-init defaults
# Distillation curve points, in reference training steps (0 => the ~0-distillation headline).
CURVE_STEPS = (0, 100, 400)
# transformer bar (held-out forage eval @ H=64, 148,608 params): regret @ total FLOPs.
# Re-measured by `python -m smolml.experiments.forage_baseline`; filled in the PR.


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
    model = build_model("forage_min", mc)
    ref_step = model.flops(2 * HORIZON).scale(32).total
    print(
        f"forage_min params={model.num_params()} (bar: 148,608); "
        f"one-train-step={ref_step:.3e} FLOPs"
    )

    floor = random_floor(fcfg, EPISODES, SEED)
    print(f"\nrandom floor (held-out): {floor:+.4f}")
    print("\nregret-vs-total-FLOP curve (forage_min):")
    trained = None
    for steps in CURVE_STEPS:
        # budget = (steps + 0.5) ref-steps: steps==0 -> below one step -> 0 distillation steps.
        cfg = ControlTrainConfig(
            model="forage_min",
            model_config=dict(MODEL),
            flop_budget=ref_step * (steps + 0.5),
            batch_size=32,
            horizon=HORIZON,
            eval_episodes=EPISODES,
            eval_interval=10**9,
            seed=SEED,
            env_name="forage",
            run_name=f"forage-min-{steps}steps",
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
    render_rollout(res.trajectory, f"{RUNS_DIR}/forage_min_sample_rollout.png")
    print(f"sample rollout: {RUNS_DIR}/forage_min_sample_rollout.png")


if __name__ == "__main__":
    main()
