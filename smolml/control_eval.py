"""Interactive control-rung scorer: roll the model in ChemoEnv via the FLOP-honest
``model.step`` channel, sampling actions from the policy slice and scoring world-model
bits on the concentration slice. Mirrors ``eval.py``/``icl_eval.py``."""

from dataclasses import dataclass

import torch

from smolml.envs.chemotaxis import (
    ChemoConfig,
    ChemoEnv,
    Trajectory,
    action_slice,
    action_token,
    conc_slice,
)
from smolml.flops import FlopBreakdown
from smolml.models.registry import LanguageModel
from smolml.prequential import score_bits


@dataclass
class ControlResult:
    mean_reward: float
    mean_oracle_reward: float
    regret: float
    world_model_bits: float
    first_half_reward: float
    second_half_reward: float
    flops: FlopBreakdown
    n_episodes: int
    horizon: int
    trajectory: Trajectory | None = None


def _sample_action(action_logits: torch.Tensor, greedy: bool, gen: torch.Generator) -> int:
    if greedy:
        return int(action_logits.argmax())
    probs = torch.softmax(action_logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen))


@torch.no_grad()
def evaluate_control(
    model: LanguageModel,
    cfg: ChemoConfig,
    *,
    split: str = "eval",
    n_episodes: int,
    seed: int,
    device: torch.device,
    greedy: bool = False,
    record: bool = False,
) -> ControlResult:
    """Mean reward, regret-vs-oracle, and world-model bits over a seeded held-out set."""
    was_training = model.training
    model.eval()
    cs, as_ = conc_slice(cfg), action_slice(cfg)
    half = cfg.horizon // 2
    flops = FlopBreakdown()
    agent_total = oracle_total = bits = 0.0
    first_total = second_total = 0.0
    trajectory: Trajectory | None = None

    for ep in range(n_episodes):
        ep_seed = seed * 100003 + ep
        env = ChemoEnv(cfg, split=split, seed=ep_seed)
        gen = torch.Generator().manual_seed(ep_seed)
        state = model.init_prequential_state()
        c = env.reset()
        tape = [c]
        rec_mu, rec_pos, rec_field = [env.mu], [env.p], [env.field()]
        rec_act, rec_reward, rec_pred = [], [], []
        pos = 0
        for t in range(cfg.horizon):
            state, logits, f = model.step(state, tape[pos], pos)
            flops += f
            pos += 1
            a_idx = _sample_action(logits[as_], greedy, gen)
            tape.append(action_token(cfg, a_idx))
            state, logits_pred, f = model.step(state, tape[pos], pos)
            flops += f
            pos += 1
            c, reward = env.step(a_idx)
            bits += score_bits(logits_pred[cs], c)
            tape.append(c)
            agent_total += reward
            if t < half:
                first_total += reward
            else:
                second_total += reward
            if record:
                rec_act.append(a_idx)
                rec_reward.append(reward)
                rec_mu.append(env.mu)
                rec_pos.append(env.p)
                rec_field.append(env.field())
                rec_pred.append(torch.softmax(logits_pred[cs], dim=-1).tolist())

        oracle_env = ChemoEnv(cfg, split=split, seed=ep_seed)
        oracle_env.reset()
        for _ in range(cfg.horizon):
            _, r = oracle_env.step(oracle_env.oracle_action())
            oracle_total += r

        if record and trajectory is None:
            trajectory = Trajectory(
                mu=rec_mu,
                pos=rec_pos,
                conc_token=tape[::2],
                reward=rec_reward,
                action=rec_act,
                field=rec_field,
                pred_conc=rec_pred,
            )

    if was_training:
        model.train()
    n = n_episodes * cfg.horizon
    return ControlResult(
        mean_reward=agent_total / n,
        mean_oracle_reward=oracle_total / n,
        regret=(oracle_total - agent_total) / n,
        world_model_bits=bits / n,
        first_half_reward=first_total / (n_episodes * half),
        second_half_reward=second_total / (n_episodes * (cfg.horizon - half)),
        flops=flops,
        n_episodes=n_episodes,
        horizon=cfg.horizon,
        trajectory=trajectory,
    )
