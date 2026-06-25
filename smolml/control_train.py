"""FLOP-budgeted Algorithm-Distillation training for the control rung. Mirrors
``train.py``'s budgeted loop + JSONL logging, sourcing distillation tapes and
logging the control metric (reward/regret/world-model bits) at checkpoints."""

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import torch

from smolml.control_eval import evaluate_control
from smolml.device import get_device
from smolml.envs.chemotaxis import ChemoConfig, chemo_env_spec
from smolml.envs.spec import EnvSpec, distill_seed, make_distillation_batch
from smolml.models.registry import LanguageModel, build_model


@dataclass
class ControlTrainConfig:
    model: str = "transformer"
    model_config: dict[str, object] = field(default_factory=dict)
    flop_budget: float = 1e10
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 0
    eval_interval: int = 20
    eval_episodes: int = 32
    epsilon: float = 0.1
    width: int = 16
    levels: int = 8
    sigma: float = 2.0
    horizon: int = 64
    device: str | None = None
    run_name: str | None = None
    env_name: str = "chemotaxis"


@dataclass
class ControlRunSummary:
    run: str
    model: str
    params: int
    seed: int
    device: str
    flop_budget: float
    total_flops: int
    final_eval_flops: int
    steps: int
    final_regret: float
    final_reward: float
    final_world_model_bits: float
    first_half_reward: float
    second_half_reward: float
    log_path: str


def distill_train_run(
    cfg: ControlTrainConfig,
    runs_dir: str | Path = "runs",
    *,
    return_model: bool = False,
    env_spec: EnvSpec | None = None,
) -> ControlRunSummary | tuple[ControlRunSummary, LanguageModel]:
    if cfg.flop_budget <= 0:
        raise ValueError(f"flop_budget must be positive, got {cfg.flop_budget}")
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    if env_spec is None:
        chem = ChemoConfig(width=cfg.width, levels=cfg.levels, sigma=cfg.sigma, horizon=cfg.horizon)
        env_spec = chemo_env_spec(chem)
    mc = dict(cfg.model_config)
    mc["vocab_size"] = env_spec.tape_spec.vocab_size
    need = 2 * cfg.horizon + 1
    mc["max_seq_len"] = max(int(mc.get("max_seq_len", 0)), need)

    model = build_model(cfg.model, mc).to(device)
    model.train()
    optimizer = model.configure_optimizer(lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    seq_len = 2 * cfg.horizon
    step_flops = model.flops(seq_len).scale(cfg.batch_size).total
    if step_flops <= 0:
        raise ValueError(f"model reports non-positive step cost: {step_flops}")

    run_name = cfg.run_name or f"{cfg.model}-control-{int(time.time())}"
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    log_path = runs_path / f"{run_name}.jsonl"
    started = time.time()

    def evaluate() -> object:
        return evaluate_control(
            model, env_spec, n_episodes=cfg.eval_episodes, seed=cfg.seed, device=device
        )

    with log_path.open("w") as log:
        resolved = asdict(model.config) if is_dataclass(model.config) else dict(mc)
        log.write(
            json.dumps(
                {
                    "type": "meta",
                    "protocol": "control",
                    "env": cfg.env_name,
                    "run": run_name,
                    "model": cfg.model,
                    "config": resolved,
                    "params": model.num_params(),
                    "device": device.type,
                    "seed": cfg.seed,
                    "flop_budget": cfg.flop_budget,
                    "batch_size": cfg.batch_size,
                    "horizon": cfg.horizon,
                    "levels": cfg.levels,
                    "width": cfg.width,
                    "eval_episodes": cfg.eval_episodes,
                    "eval_interval": cfg.eval_interval,
                    "started_at": started,
                }
            )
            + "\n"
        )

        step, cumulative = 0, 0
        last_logged = -1
        res = evaluate()  # step-0 point (untrained baseline)

        def log_step(r: object) -> None:
            nonlocal last_logged
            log.write(
                json.dumps(
                    {
                        "type": "step",
                        "step": step,
                        "cumulative_flops": cumulative + r.flops.total,
                        "train_flops": cumulative,
                        "eval_flops": r.flops.total,
                        "mean_reward": r.mean_reward,
                        "regret": r.regret,
                        "world_model_bits": r.world_model_bits,
                        "first_half_reward": r.first_half_reward,
                        "second_half_reward": r.second_half_reward,
                    }
                )
                + "\n"
            )
            log.flush()
            last_logged = step

        log_step(res)
        while cumulative + step_flops <= cfg.flop_budget:
            x, y = make_distillation_batch(
                env_spec,
                batch_size=cfg.batch_size,
                seed=distill_seed(cfg.seed, step),
                device=device,
                epsilon=cfg.epsilon,
            )
            _loss, spent = model.train_step((x, y), optimizer, grad_clip=cfg.grad_clip)
            cumulative += spent.total
            step += 1
            if step % cfg.eval_interval == 0:
                res = evaluate()
                log_step(res)
        if last_logged != step:
            res = evaluate()
            log_step(res)

    summary = ControlRunSummary(
        run=run_name,
        model=cfg.model,
        params=model.num_params(),
        seed=cfg.seed,
        device=device.type,
        flop_budget=cfg.flop_budget,
        total_flops=cumulative + res.flops.total,
        final_eval_flops=res.flops.total,
        steps=step,
        final_regret=res.regret,
        final_reward=res.mean_reward,
        final_world_model_bits=res.world_model_bits,
        first_half_reward=res.first_half_reward,
        second_half_reward=res.second_half_reward,
        log_path=str(log_path),
    )
    return (summary, model) if return_model else summary
