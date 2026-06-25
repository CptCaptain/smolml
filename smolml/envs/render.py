"""Render a recorded control rollout: a static spacetime raster (default) and an
opt-in animated GIF. Headless matplotlib; pillow ships the GIF writer.

``render_rollout`` dispatches on the per-step ``record_state`` payload off the
generic ``Trajectory`` (chemotaxis: ``mu``/``p``/``field``; forage:
``cells``/``g``/``p``), so each env renders itself without a scorer change."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, writers  # noqa: E402

from smolml.envs.chemotaxis import Trajectory  # noqa: E402
from smolml.envs.forage import EAT  # noqa: E402


def render_rollout(traj: Trajectory, out_png: str | Path) -> Path:
    """Spacetime raster for a rollout, dispatched on the ``record_state`` payload."""
    first = traj.states[0]
    if "field" in first:
        return _render_chemo(traj, out_png)
    if "cells" in first:
        return _render_forage(traj, out_png)
    raise ValueError(f"unrecognized record_state payload keys: {sorted(first)}")


def _render_chemo(traj: Trajectory, out_png: str | Path) -> Path:
    """Concentration field over time + agent/peak paths + cumulative reward."""
    field = np.array([s["field"] for s in traj.states])  # (steps, width)
    pos = [s["p"] for s in traj.states]
    mu = [s["mu"] for s in traj.states]
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(8, 6))
    ax1.imshow(field.T, aspect="auto", origin="lower", cmap="viridis")
    ax1.plot(range(len(pos)), pos, color="red", lw=1.5, label="agent")
    ax1.plot(range(len(mu)), mu, color="white", ls="--", lw=1.0, label="peak")
    ax1.set_xlabel("step")
    ax1.set_ylabel("ring cell")
    ax1.legend(loc="upper right")
    ax2.plot(np.cumsum(traj.reward), color="green")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cumulative reward")
    fig.tight_layout()
    out = Path(out_png)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def _render_forage(traj: Trajectory, out_png: str | Path) -> Path:
    """Cue-type ring over time (rows=steps, cols=cells) + agent path, eaten cells, reward."""
    states = traj.states
    cells = np.array([s["cells"] for s in states])  # (steps, width); rows are constant
    pos = [s["p"] for s in states]
    g = states[0]["g"]
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(8, 6))
    ax1.imshow(
        cells, aspect="auto", origin="lower", interpolation="nearest", cmap="tab10", vmin=0, vmax=9
    )
    ax1.plot(pos, range(len(pos)), color="black", lw=1.5, label="agent")
    eaten_p, eaten_t, eaten_c = [], [], []
    for t, a in enumerate(traj.action):
        if a == EAT:  # the graded cell is the pre-step position states[t]["p"]
            eaten_p.append(states[t]["p"])
            eaten_t.append(t)
            eaten_c.append("lime" if traj.reward[t] > 0 else "red")
    ax1.scatter(
        eaten_p,
        eaten_t,
        c=eaten_c,
        marker="s",
        s=18,
        edgecolors="white",
        linewidths=0.4,
        label="eat (green=+1, red=-1)",
    )
    ax1.set_xlabel("ring cell (color = cue type)")
    ax1.set_ylabel("step")
    ax1.set_title(f"forage rollout (good type g={g})")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.plot(np.cumsum(traj.reward), color="green")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cumulative reward")
    fig.tight_layout()
    out = Path(out_png)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def animate_rollout(traj: Trajectory, out_gif: str | Path, *, fps: int = 10) -> Path:
    """Opt-in animated playback of the field with the agent marker. Guarded on pillow."""
    if not writers.is_available("pillow"):
        raise RuntimeError("pillow animation writer unavailable")
    field = np.array([s["field"] for s in traj.states])
    pos = [s["p"] for s in traj.states]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(range(field.shape[1]), field[0])
    marker = ax.axvline(pos[0], color="red", lw=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("ring cell")
    ax.set_ylabel("concentration")

    def update(t: int):
        for bar, h in zip(bars, field[t], strict=True):
            bar.set_height(h)
        marker.set_xdata([pos[t], pos[t]])
        ax.set_title(f"step {t}")
        return [*bars, marker]

    anim = FuncAnimation(fig, update, frames=len(field), blit=False)
    out = Path(out_gif)
    anim.save(out, writer="pillow", fps=fps)
    plt.close(fig)
    return out
