"""Interactive control-rung scorer: roll the model through any ``Environment`` (via an
``EnvSpec``) on the FLOP-honest ``model.step`` channel, sampling actions from the policy
slice and scoring world-model bits on the obs slice. Mirrors ``eval.py``/``icl_eval.py``."""

from dataclasses import dataclass

import torch

from smolml.envs.chemotaxis import Trajectory
from smolml.envs.spec import EnvSpec, env_seed
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
    env_spec: EnvSpec,
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
    ts = env_spec.tape_spec
    obs_sl, act_sl = ts.obs_slice, ts.action_slice
    flops = FlopBreakdown()
    agent_total = oracle_total = bits = 0.0
    first_total = second_total = 0.0
    trajectory: Trajectory | None = None
    horizon = half = 0

    for ep in range(n_episodes):
        ep_seed = env_seed(seed, ep)
        env = env_spec.env_factory(split, ep_seed)
        horizon = env.horizon
        half = horizon // 2
        gen = torch.Generator().manual_seed(ep_seed)
        state = model.init_prequential_state()
        obs = env.reset()
        tape = [obs]
        rec_states = [env.record_state()] if record else []
        rec_act, rec_reward, rec_pred = [], [], []
        pos = 0
        for t in range(horizon):
            state, logits, f = model.step(state, tape[pos], pos)
            flops += f
            pos += 1
            a_idx = _sample_action(logits[act_sl], greedy, gen)
            tape.append(ts.action_token(a_idx))
            state, logits_pred, f = model.step(state, tape[pos], pos)
            flops += f
            pos += 1
            obs, reward = env.step(a_idx)
            bits += score_bits(logits_pred[obs_sl], obs)
            tape.append(obs)
            agent_total += reward
            if t < half:
                first_total += reward
            else:
                second_total += reward
            if record:
                rec_act.append(a_idx)
                rec_reward.append(reward)
                rec_states.append(env.record_state())
                rec_pred.append(torch.softmax(logits_pred[obs_sl], dim=-1).tolist())

        oracle_env = env_spec.env_factory(split, ep_seed)
        oracle_env.reset()
        for _ in range(horizon):
            _, r = oracle_env.step(oracle_env.oracle_action())
            oracle_total += r

        if record and trajectory is None:
            trajectory = Trajectory(
                obs_token=tape[::2],
                action=rec_act,
                reward=rec_reward,
                states=rec_states,
                pred_obs=rec_pred,
            )

    if was_training:
        model.train()
    n = n_episodes * horizon
    return ControlResult(
        mean_reward=agent_total / n,
        mean_oracle_reward=oracle_total / n,
        regret=(oracle_total - agent_total) / n,
        world_model_bits=bits / n,
        first_half_reward=first_total / (n_episodes * half),
        second_half_reward=second_total / (n_episodes * (horizon - half)),
        flops=flops,
        n_episodes=n_episodes,
        horizon=horizon,
        trajectory=trajectory,
    )
