import math

import matplotlib
import pytest

from smolml.envs.chemotaxis import Trajectory
from smolml.envs.render import animate_rollout, render_rollout


def _synthetic_traj() -> Trajectory:
    """A hand-built rollout: width=8, 7 record-state rows (mu/p/field), reward/action length 6.

    Avoids importing ``control_eval`` or training a model — exercises the renderer alone.
    """
    width, steps = 8, 7
    field = [
        [0.5 + 0.5 * math.sin((x - t) / width * 2 * math.pi) for x in range(width)]
        for t in range(steps)
    ]
    # field values are guaranteed within [0, 1] by the 0.5 + 0.5*sin construction.
    pos = [t % width for t in range(steps)]
    mu = [(t * 1.3) % width for t in range(steps)]
    obs_token = [t % width for t in range(steps)]
    reward = [0.1 * (i + 1) for i in range(steps - 1)]
    action = [i % 3 for i in range(steps - 1)]
    states = [{"mu": mu[t], "p": pos[t], "field": field[t]} for t in range(steps)]
    return Trajectory(
        obs_token=obs_token,
        action=action,
        reward=reward,
        states=states,
    )


def test_render_writes_nonempty_png(tmp_path):
    traj = _synthetic_traj()
    out = render_rollout(traj, tmp_path / "rollout.png")
    assert out.exists() and out.stat().st_size > 0


def test_animate_writes_nonempty_gif(tmp_path):
    if not matplotlib.animation.writers.is_available("pillow"):
        pytest.skip("pillow animation writer unavailable")
    traj = _synthetic_traj()
    out = animate_rollout(traj, tmp_path / "rollout.gif", fps=10)
    assert out.exists() and out.stat().st_size > 0
