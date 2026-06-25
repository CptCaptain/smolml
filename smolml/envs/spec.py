"""The ``Environment`` seam bundle (Task C.A.3): an :class:`EnvSpec` ties an
``env_factory`` + a distillation ``source_factory`` + a :class:`TapeSpec` so the
control scorer/trainer are env-agnostic. Chemotaxis is instance #1, forage #2;
each env ships a one-call ``*_env_spec`` helper so call-site migration is one line.

The three RNG seed formulas are centralized here (named helpers, **not** unified):
seed drift is invisible to FLOPs, the one bit-identity hazard the pin test guards.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from smolml.envs.chemotaxis import Environment


class Policy(Protocol):
    """A distillation source: stateful, sees one obs token, emits an action index."""

    def reset(self) -> None: ...
    def act(self, obs: int) -> int: ...


@dataclass(frozen=True)
class TapeSpec:
    """Symbol layout of a control tape: obs occupy ``[0, obs_len)``, actions above.

    ``action_token(idx)`` indexes within the action sub-vocab by raw offset, so the
    scorer reads policy logits at ``action_slice`` and world-model logits at
    ``obs_slice`` (a disjoint, obs-at-0 layout shared by every env).
    """

    vocab_size: int
    obs_slice: slice
    action_slice: slice

    def action_token(self, action_idx: int) -> int:
        return self.action_slice.start + action_idx


@dataclass(frozen=True)
class EnvSpec:
    """Everything env-specific the control spine needs, behind one bundle.

    ``env_factory(split, seed) -> Environment`` and ``source_factory(seed,
    **source_kwargs) -> Policy`` draw **no torch RNG** at construction (only their
    own numpy streams), so torch determinism is owned solely by the caller.
    """

    env_factory: Callable[[str, int], "Environment"]
    source_factory: Callable[..., Policy]
    tape_spec: TapeSpec


def env_seed(seed: int, index: int) -> int:
    """Per-episode (eval ``ep``) / per-batch-row (distill ``b``) environment seed."""
    return seed * 100003 + index


def source_seed(seed: int, index: int) -> int:
    """Per-batch-row distillation-source seed."""
    return seed * 7919 + index


def distill_seed(seed: int, step: int) -> int:
    """Per-step distillation outer-loop seed."""
    return seed * 1009 + step


def make_distillation_batch(
    env_spec: EnvSpec,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    **source_kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll the distillation source over fresh ``train`` episodes -> static next-token
    tapes. Each tape is ``o0 a0 o1 a1 … a_{H-1} o_H`` (length ``2H+1``); returns
    ``(x, y) = (tape[:-1], tape[1:])`` of shape ``(batch_size, 2H)``.
    """
    ts = env_spec.tape_spec
    rows: list[list[int]] = []
    for b in range(batch_size):
        env = env_spec.env_factory("train", env_seed(seed, b))
        pol = env_spec.source_factory(source_seed(seed, b), **source_kwargs)
        obs = env.reset()
        tape = [obs]
        for _ in range(env.horizon):
            a = pol.act(obs)
            tape.append(ts.action_token(a))
            obs, _reward = env.step(a)
            tape.append(obs)
        rows.append(tape)
    tapes = torch.tensor(rows, dtype=torch.long)
    return tapes[:, :-1].contiguous().to(device), tapes[:, 1:].contiguous().to(device)
