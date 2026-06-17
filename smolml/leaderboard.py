"""Leaderboard: aggregate run logs into a table + a bpb-vs-FLOPs plot.

Reads every ``runs/*.jsonl`` produced by :mod:`smolml.train`, sorts by final
validation bpb (lower is better — the one metric), renders a markdown table, and
draws each run's bpb-vs-training-FLOPs trajectory on a log-x plot saved as PNG.
Re-running it after new runs land regenerates both, so the board is reproducible
and never hand-edited.
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class RunRecord:
    """One run parsed from its JSONL log: identity, hyperparameters, and curve."""

    run: str
    model: str
    params: int
    device: str
    seed: int
    flop_budget: float
    flops: list[int] = field(default_factory=list)
    val_bpb: list[float] = field(default_factory=list)
    opt_steps: list[int] = field(default_factory=list)

    @property
    def final_flops(self) -> int:
        return self.flops[-1] if self.flops else 0

    @property
    def final_val_bpb(self) -> float:
        return self.val_bpb[-1] if self.val_bpb else float("nan")

    @property
    def steps(self) -> int:
        """Optimizer steps taken (final logged ``step``)."""
        return self.opt_steps[-1] if self.opt_steps else 0

    @property
    def n_points(self) -> int:
        """Number of logged points on the bpb-vs-FLOPs curve."""
        return len(self.flops)


def load_run(path: str | Path) -> RunRecord:
    """Parse a single run log into a :class:`RunRecord`."""
    meta: dict = {}
    flops: list[int] = []
    bpb: list[float] = []
    opt_steps: list[int] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                meta = obj
            elif obj.get("type") == "step":
                flops.append(int(obj["cumulative_flops"]))
                bpb.append(float(obj["val_bpb"]))
                opt_steps.append(int(obj["step"]))
    if not meta:
        raise ValueError(f"{path}: missing meta line")
    return RunRecord(
        run=meta["run"],
        model=meta["model"],
        params=int(meta["params"]),
        device=meta["device"],
        seed=int(meta["seed"]),
        flop_budget=float(meta["flop_budget"]),
        flops=flops,
        val_bpb=bpb,
        opt_steps=opt_steps,
    )


def collect_runs(runs_dir: str | Path) -> list[RunRecord]:
    """Load all run logs in ``runs_dir``, sorted best (lowest final bpb) first."""
    records = [load_run(p) for p in sorted(Path(runs_dir).glob("*.jsonl"))]
    records.sort(key=lambda r: r.final_val_bpb)
    return records


def build_table(records: list[RunRecord]) -> str:
    """Render the leaderboard as a markdown table (best run first)."""
    header = (
        "| rank | run | model | params | train FLOPs | steps | final val bpb |\n"
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |"
    )
    rows = [header]
    for rank, r in enumerate(records, start=1):
        rows.append(
            f"| {rank} | {r.run} | {r.model} | {r.params:,} | "
            f"{r.final_flops:.3e} | {r.steps} | {r.final_val_bpb:.4f} |"
        )
    return "\n".join(rows)


def plot_bpb_vs_flops(records: list[RunRecord], out_png: str | Path) -> Path:
    """Draw bpb-vs-training-FLOPs (log-x) for every run and save a PNG."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in records:
        if not r.flops:
            continue
        ax.plot(r.flops, r.val_bpb, marker="o", markersize=4, label=f"{r.run} ({r.model})")
    ax.set_xscale("log")
    ax.set_xlabel("cumulative training FLOPs")
    ax.set_ylabel("validation bits-per-byte")
    ax.set_title("smolml leaderboard — bpb vs FLOPs (lower is better)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    if records:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def regenerate(
    runs_dir: str | Path = "runs",
    table_path: str | Path | None = None,
    plot_path: str | Path = "runs/leaderboard.png",
) -> tuple[str, Path]:
    """Rebuild the leaderboard table and plot from all logs in ``runs_dir``.

    Returns ``(table_markdown, plot_path)``; also writes the table to
    ``table_path`` when given.
    """
    records = collect_runs(runs_dir)
    table = build_table(records)
    png = plot_bpb_vs_flops(records, plot_path)
    if table_path is not None:
        Path(table_path).write_text(table + "\n")
    return table, png
