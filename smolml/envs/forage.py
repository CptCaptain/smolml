"""Contingency-forage control environment + its EnvSpec (Task C.A.3).

A stationary ring of cue cells with a per-episode latent rewarding type ``g`` the
agent can only learn from its own eat-outcomes. Unlike chemotaxis (a fixed
gradient-climbing reflex wins), no contingency-blind fixed policy is near-optimal,
so regret separates a genuine in-context learner from a reflex. Forage is the
``Environment`` seam's instance #2; the scorer/trainer stay env-agnostic.
"""

from dataclasses import dataclass

import numpy as np

from smolml.envs.spec import EnvSpec, TapeSpec

N_ACTIONS: int = 3
LEFT, EAT, RIGHT = 0, 1, 2
REWARD_LEVELS: int = 3  # reward in {-1, 0, +1} -> {0, 1, 2}
SEED_BAND: int = 1 << 40
"""Width of each split's seed band; train and eval occupy disjoint bands."""


def band_seed(split: str, seed: int) -> int:
    """Map ``(split, seed)`` into a per-split seed band so train/eval never collide.

    train -> ``[0, SEED_BAND)``, eval -> ``[SEED_BAND, 2*SEED_BAND)``; the bands are
    disjoint for every input, so eval layouts are genuinely held out from training.
    """
    bands = {"train": 0, "eval": 1}
    if split not in bands:
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    return bands[split] * SEED_BAND + seed % SEED_BAND


@dataclass
class ForageConfig:
    """Environment hyperparameters: a ``width``-cell ring of ``n_types`` cue types."""

    width: int = 16
    n_types: int = 3
    horizon: int = 64


def obs_len(cfg: ForageConfig) -> int:
    """Length of the combined-obs sub-vocab: ``REWARD_LEVELS`` per cue type."""
    return REWARD_LEVELS * cfg.n_types


def vocab_size(cfg: ForageConfig) -> int:
    return obs_len(cfg) + N_ACTIONS


class ForageEnv:
    """Stationary cue ring; the agent senses only its current cell's combined obs.

    Cells are i.i.d. uniform cue types fixed for the whole episode; exactly one type
    ``g`` pays ``+1`` on EAT (others ``-1``: poison). EAT advances (no camping), so
    food never depletes and recognizing the rewarding *type* is the efficient policy.
    The combined obs symbol packs the cell type after the move with the last action's
    reward, keeping both the current cue and the reward signal in a single token.
    """

    n_actions: int = N_ACTIONS

    def __init__(self, cfg: ForageConfig, *, split: str, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(band_seed(split, seed))
        self.cells: list[int] = rng.integers(cfg.n_types, size=cfg.width).tolist()
        self.g = int(rng.integers(cfg.n_types))
        self.p = int(rng.integers(cfg.width))

    def _obs(self, reward: int) -> int:
        """Combined obs: ``cell_type * REWARD_LEVELS + (reward + 1)`` at the current cell."""
        return self.cells[self.p] * REWARD_LEVELS + (reward + 1)

    def reset(self) -> int:
        return self._obs(0)

    def step(self, action_idx: int) -> tuple[int, float]:
        if action_idx == EAT:
            reward = 1 if self.cells[self.p] == self.g else -1
            self.p = (self.p + 1) % self.cfg.width
        elif action_idx == LEFT:
            reward = 0
            self.p = (self.p - 1) % self.cfg.width
        elif action_idx == RIGHT:
            reward = 0
            self.p = (self.p + 1) % self.cfg.width
        else:
            raise ValueError(f"action_idx must be in [0, {N_ACTIONS}), got {action_idx}")
        return self._obs(reward), float(reward)

    def oracle_action(self) -> int:
        """Knows ``g`` (but not positions): EAT on a good cell, else move RIGHT."""
        return EAT if self.cells[self.p] == self.g else RIGHT

    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    def record_state(self) -> dict:
        return {"cells": list(self.cells), "g": self.g, "p": self.p}


class WinStayLoseShift:
    """Elimination tracker source: eat unknown/known-good types, skip known-bad.

    Decodes each combined obs into ``(type, last_reward)``; a remembered eat is graded
    by the next obs's reward component (win -> good, lose -> bad). It eats unknowns to
    explore and the good type to exploit, skipping known-bad cells; ``epsilon`` adds
    uniform exploration. This explores early and exploits ``g`` late, the within-episode
    learning curve Algorithm Distillation copies into the model.
    """

    UNKNOWN, GOOD, BAD = 0, 1, 2

    def __init__(self, n_types: int, *, epsilon: float = 0.1, seed: int = 0):
        self.n_types = n_types
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.belief = [self.UNKNOWN] * n_types
        self.last_eaten_type: int | None = None

    def reset(self) -> None:
        self.belief = [self.UNKNOWN] * self.n_types
        self.last_eaten_type = None

    def act(self, obs: int) -> int:
        cur_type, last_reward = obs // REWARD_LEVELS, obs % REWARD_LEVELS - 1
        if self.last_eaten_type is not None:
            self.belief[self.last_eaten_type] = self.GOOD if last_reward > 0 else self.BAD
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(N_ACTIONS))
        elif self.belief[cur_type] == self.BAD:
            action = RIGHT
        else:
            action = EAT
        self.last_eaten_type = cur_type if action == EAT else None
        return action


def forage_env_spec(cfg: ForageConfig) -> EnvSpec:
    """Build the forage :class:`EnvSpec` — the control spine's env instance #2."""
    length = obs_len(cfg)
    return EnvSpec(
        env_factory=lambda split, seed: ForageEnv(cfg, split=split, seed=seed),
        source_factory=lambda seed, epsilon=0.1: WinStayLoseShift(
            cfg.n_types, epsilon=epsilon, seed=seed
        ),
        tape_spec=TapeSpec(
            vocab_size=length + N_ACTIONS,
            obs_slice=slice(0, length),
            action_slice=slice(length, length + N_ACTIONS),
        ),
    )
