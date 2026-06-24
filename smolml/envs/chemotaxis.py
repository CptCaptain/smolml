"""Chemotaxis control environment + the Environment seam (Task C.A.0).

A 1-D ring with a drifting concentration peak; the agent senses only the local
concentration (quantized) and acts LEFT/STAY/RIGHT. Other feedback-driven tasks
implement the same ``Environment`` protocol and reuse the scorer + candidates.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from smolml.envs.spec import EnvSpec, TapeSpec

N_ACTIONS: int = 3
ACTION_DELTAS: tuple[int, ...] = (-1, 0, 1)  # LEFT, STAY, RIGHT
DRIFT_RATES: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass
class ChemoConfig:
    """Environment hyperparameters (one symbol vocab per ``levels``)."""

    width: int = 16
    levels: int = 8
    sigma: float = 2.0
    horizon: int = 64


def drift_rates(split: str) -> tuple[float, ...]:
    """Disjoint per-episode drift-rate pools: even-index for train, odd for eval."""
    if split == "train":
        return DRIFT_RATES[::2]
    if split == "eval":
        return DRIFT_RATES[1::2]
    raise ValueError(f"split must be 'train' or 'eval', got {split!r}")


def vocab_size(cfg: ChemoConfig) -> int:
    return cfg.levels + N_ACTIONS


def conc_slice(cfg: ChemoConfig) -> slice:
    return slice(0, cfg.levels)


def action_slice(cfg: ChemoConfig) -> slice:
    return slice(cfg.levels, cfg.levels + N_ACTIONS)


def action_token(cfg: ChemoConfig, action_idx: int) -> int:
    return cfg.levels + action_idx


def ringdist(a: float, b: float, width: int) -> int:
    d = abs(a - b) % width
    return int(min(d, width - d))


class Environment(Protocol):
    """Minimal feedback-task seam: the scorer/training/candidates depend only on this."""

    n_actions: int
    horizon: int

    def reset(self) -> int: ...
    def step(self, action_idx: int) -> tuple[int, float]: ...
    def oracle_action(self) -> int: ...
    def record_state(self) -> dict: ...


class ChemoEnv:
    """Drifting-gradient ring; the agent senses only the local quantized concentration."""

    n_actions: int = N_ACTIONS

    def __init__(self, cfg: ChemoConfig, *, split: str, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        rates = drift_rates(split)
        self.drift_rate = float(rates[rng.integers(len(rates))])
        self.drift_dir = int(rng.choice((-1, 1)))
        self.mu = float(rng.integers(cfg.width))
        self.p = int(rng.integers(cfg.width))
        self._phase = 0.0

    def _raw(self, x: float) -> float:
        d = ringdist(x, self.mu, self.cfg.width)
        return math.exp(-(d * d) / (2.0 * self.cfg.sigma**2))

    def _level(self, raw: float) -> int:
        return min(self.cfg.levels - 1, max(0, int(round(raw * (self.cfg.levels - 1)))))

    def reset(self) -> int:
        return self._level(self._raw(self.p))

    def step(self, action_idx: int) -> tuple[int, float]:
        self.p = (self.p + ACTION_DELTAS[action_idx]) % self.cfg.width
        self._phase += self.drift_rate
        if self._phase >= 1.0:
            self.mu = (self.mu + self.drift_dir) % self.cfg.width
            self._phase -= 1.0
        raw = self._raw(self.p)
        return self._level(raw), raw

    def oracle_action(self) -> int:
        best_idx, best_d = 1, ringdist(self.p, self.mu, self.cfg.width)
        for i, delta in enumerate(ACTION_DELTAS):
            d = ringdist((self.p + delta) % self.cfg.width, self.mu, self.cfg.width)
            if d < best_d:
                best_idx, best_d = i, d
        return best_idx

    def field(self) -> list[float]:
        return [self._raw(x) for x in range(self.cfg.width)]

    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    def record_state(self) -> dict:
        return {"mu": self.mu, "p": self.p, "field": self.field()}


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, _conc: int) -> int:
        return int(self.rng.integers(N_ACTIONS))


class RunAndTumble:
    """Keep moving if concentration rose, else reverse (tumble); ``epsilon`` explores."""

    def __init__(self, epsilon: float = 0.0, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.last_action = 2  # RIGHT
        self.last_conc: int | None = None

    def reset(self) -> None:
        self.last_action = 2
        self.last_conc = None

    def act(self, conc: int) -> int:
        if self.rng.random() < self.epsilon:
            a = int(self.rng.integers(N_ACTIONS))
        elif self.last_conc is None or conc >= self.last_conc:
            a = self.last_action
        else:
            a = 2 if self.last_action == 0 else 0  # reverse direction
        self.last_action, self.last_conc = a, conc
        return a


@dataclass
class Trajectory:
    """A recorded rollout: generic trace + per-env ``record_state`` payloads."""

    obs_token: list[int]
    action: list[int]
    reward: list[float]
    states: list[dict]
    pred_obs: list[list[float]] | None = None


def chemo_env_spec(chem: ChemoConfig) -> EnvSpec:
    """Build the chemotaxis :class:`EnvSpec` — the control spine's env instance #1."""
    return EnvSpec(
        env_factory=lambda split, seed: ChemoEnv(chem, split=split, seed=seed),
        source_factory=lambda seed, epsilon=0.1: RunAndTumble(epsilon=epsilon, seed=seed),
        tape_spec=TapeSpec(
            vocab_size=vocab_size(chem),
            obs_slice=conc_slice(chem),
            action_slice=action_slice(chem),
        ),
    )
