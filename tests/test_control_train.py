import json

from smolml.control_train import ControlTrainConfig, distill_train_run


def test_distill_train_smoke_writes_log(tmp_path):
    cfg = ControlTrainConfig(
        model_config={"d_model": 32, "n_layers": 2, "n_heads": 4},
        flop_budget=0.0,  # set below
        batch_size=8,
        horizon=16,
        eval_interval=5,
        eval_episodes=8,
        seed=0,
    )
    from smolml.envs.chemotaxis import ChemoConfig as _CC
    from smolml.envs.chemotaxis import vocab_size as _vs
    from smolml.models import build_model

    chem = _CC(width=16, levels=8, horizon=16)
    mc = {**cfg.model_config, "vocab_size": _vs(chem), "max_seq_len": 2 * 16 + 1}
    step_flops = build_model("transformer", mc).flops(2 * 16).scale(8).total
    cfg.flop_budget = step_flops * 30
    summary = distill_train_run(cfg, runs_dir=tmp_path / "runs")
    assert summary.steps >= 1
    assert summary.total_flops <= cfg.flop_budget + summary.final_eval_flops
    log = (tmp_path / "runs" / f"{summary.run}.jsonl").read_text().splitlines()
    meta = json.loads(log[0])
    assert meta["protocol"] == "control" and meta["params"] > 0
    # FLOP honesty (ADR 0004): the FLOPs the leaderboard reads (final cumulative_flops)
    # must be the HONEST TOTAL — training + the eval rollout — not training-only.
    final = json.loads(log[-1])
    assert final["eval_flops"] > 0
    assert final["cumulative_flops"] == final["train_flops"] + final["eval_flops"]
    assert final["cumulative_flops"] == summary.total_flops
