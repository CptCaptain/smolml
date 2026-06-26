"""Transformer contingency-forage baseline (the honest bar): sweep TRAINING
hyperparameters at fixed params ``P``, re-rank the top configs at the leaderboard
budget, then trace a FLOP-budget regret curve with the winner.

The bar is the best of a sweep, not one arbitrary config (ADR 0007): ``d_model`` /
``n_layers`` are NOT swept (they change ``P``); only ``lr``/``weight_decay``/
``batch_size``/AD-source ``epsilon`` are. Budgets are fixed FLOP amounts (a
reference step count x one step's cost at batch 32), so configs with different
batch sizes are compared at equal compute.

Run (CPU, synthetic; a few minutes)::

    uv run python -m smolml.experiments.forage_baseline
"""

import tempfile
from pathlib import Path

from smolml.control_eval import evaluate_control
from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.envs.forage import ForageConfig, forage_env_spec
from smolml.envs.render import render_rollout
from smolml.leaderboard import regenerate_control
from smolml.models import build_model

RUNS_DIR = "runs/forage"
HORIZON = 64  # production horizon (matches the C.A.4 candidates + the test-pin metric bounds)
MODEL = {"d_model": 64, "n_layers": 3, "n_heads": 4}  # fixed P (not swept)
LR_GRID = (1e-3, 3e-3, 1e-2)
WD_GRID = (0.0, 0.1)
BS_GRID = (16, 32)
EPS_GRID = (0.05, 0.1, 0.2)
FAST_STEPS = 150  # reference-step budget for the broad sweep
RERANK_STEPS = 400  # leaderboard budget for re-ranking the top configs
RERANK_K = 3
CURVE_STEPS = (150, 400, 900)  # FLOP-budget curve points (winner)
EVAL_EPISODES = 32  # held-out regret; SE ~0.01 << candidate gap (H=64 ~2x the H=32 cost)


def ref_step_flops(fcfg: ForageConfig, batch_size: int) -> int:
    """One training step's FLOPs at ``batch_size`` — the unit budgets are multiples of."""
    spec = forage_env_spec(fcfg)
    mc = {**MODEL, "vocab_size": spec.tape_spec.vocab_size, "max_seq_len": 2 * fcfg.horizon + 1}
    return build_model("transformer", mc).flops(2 * fcfg.horizon).scale(batch_size).total


def distill_at(fcfg, hp, budget, runs_dir, run_name, *, seed=0, return_model=False):
    """One distill run at hyperparameters ``hp`` (lr/weight_decay/batch_size/epsilon)."""
    cfg = ControlTrainConfig(
        model_config=dict(MODEL),
        flop_budget=budget,
        batch_size=hp["batch_size"],
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        epsilon=hp["epsilon"],
        horizon=fcfg.horizon,
        eval_episodes=EVAL_EPISODES,
        eval_interval=10**9,  # cross-run bar: only the final point matters
        env_name="forage",
        run_name=run_name,
        seed=seed,
    )
    return distill_train_run(
        cfg, runs_dir=runs_dir, return_model=return_model, env_spec=forage_env_spec(fcfg)
    )


def _run_name(prefix: str, hp: dict) -> str:
    return f"{prefix}-lr{hp['lr']}-wd{hp['weight_decay']}-bs{hp['batch_size']}-eps{hp['epsilon']}"


def sweep(fcfg, budget, *, grids=None, seed=0):
    """Train every hyperparameter combo to ``budget`` in a scratch dir; rank by regret."""
    lr_g, wd_g, bs_g, eps_g = grids or (LR_GRID, WD_GRID, BS_GRID, EPS_GRID)
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        for lr in lr_g:
            for wd in wd_g:
                for bs in bs_g:
                    for eps in eps_g:
                        hp = {"lr": lr, "weight_decay": wd, "batch_size": bs, "epsilon": eps}
                        name = _run_name("sweep", hp)
                        summ = distill_at(fcfg, hp, budget, scratch, name, seed=seed)
                        results.append((hp, summ.final_regret, summ.final_reward))
    results.sort(key=lambda r: r[1])
    return results


def sweep_table(results) -> str:
    rows = [
        "| rank | lr | weight_decay | batch_size | epsilon | regret | reward |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, (hp, regret, reward) in enumerate(results, start=1):
        rows.append(
            f"| {rank} | {hp['lr']} | {hp['weight_decay']} | {hp['batch_size']} | "
            f"{hp['epsilon']} | {regret:.4f} | {reward:+.4f} |"
        )
    return "\n".join(rows)


def main() -> None:
    fcfg = ForageConfig(horizon=HORIZON)
    ref = ref_step_flops(fcfg, 32)

    results = sweep(fcfg, ref * FAST_STEPS)
    print(f"Fast-budget sweep ({FAST_STEPS} ref-steps, {len(results)} configs):")
    print(sweep_table(results))

    reranked = []
    with tempfile.TemporaryDirectory() as scratch:
        for hp, _, _ in results[:RERANK_K]:
            name = _run_name("rerank", hp)
            summ = distill_at(fcfg, hp, ref * RERANK_STEPS, scratch, name)
            reranked.append((hp, summ.final_regret, summ.final_reward))
    reranked.sort(key=lambda r: r[1])
    print(f"\nRe-rank top {RERANK_K} at the leaderboard budget ({RERANK_STEPS} ref-steps):")
    print(sweep_table(reranked))
    winner = reranked[0][0]
    print(f"\nchosen config: {winner}")

    Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)
    trained = None
    print("\nFLOP-budget curve (winner):")
    for steps in CURVE_STEPS:
        summ, trained = distill_at(
            fcfg,
            winner,
            ref * steps,
            RUNS_DIR,
            f"transformer-forage-{steps}steps",
            return_model=True,
        )
        print(
            f"  steps={steps:5d}  regret={summ.final_regret:.4f}  "
            f"reward={summ.final_reward:+.4f}  flops={summ.total_flops:.3e}"
        )

    table, png = regenerate_control(
        RUNS_DIR, table_path=f"{RUNS_DIR}/leaderboard.md", plot_path=f"{RUNS_DIR}/leaderboard.png"
    )
    print("\n" + table + f"\nplot: {png}")

    res = evaluate_control(
        trained,
        forage_env_spec(fcfg),
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
