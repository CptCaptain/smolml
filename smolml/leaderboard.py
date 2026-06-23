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
    """One run parsed from its JSONL log: identity, protocol, and bpb-vs-FLOPs curve.

    ``flops``/``val_bpb`` are the plotted trajectory; for **amortized** runs the
    x-axis is cumulative *training* FLOPs, for **prequential** runs it is
    cumulative *total* FLOPs (pretrain + inference + adaptation).
    """

    run: str
    model: str
    protocol: str
    params: int
    device: str
    seed: int
    detail: str
    budget: float
    flops: list[int] = field(default_factory=list)
    val_bpb: list[float] = field(default_factory=list)
    x_steps: list[int] = field(default_factory=list)
    eval_seq_len: int | None = None
    val_fraction: float | None = None

    @property
    def final_flops(self) -> int:
        return self.flops[-1] if self.flops else 0

    @property
    def final_val_bpb(self) -> float:
        return self.val_bpb[-1] if self.val_bpb else float("nan")

    @property
    def steps(self) -> int:
        """Final logged ``step`` (optimizer steps for amortized; bytes for prequential)."""
        return self.x_steps[-1] if self.x_steps else 0

    @property
    def n_points(self) -> int:
        """Number of logged points on the curve."""
        return len(self.flops)


def load_run(path: str | Path) -> RunRecord:
    """Parse a single run log (either protocol) into a :class:`RunRecord`."""
    meta: dict[str, object] = {}
    flops: list[int] = []
    bpb: list[float] = []
    x_steps: list[int] = []
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
                x_steps.append(int(obj["step"]))
    if not meta:
        raise ValueError(f"{path}: missing meta line")
    protocol = str(meta.get("protocol", "amortized"))
    common = {
        "run": meta["run"],
        "model": meta["model"],
        "protocol": protocol,
        "params": int(meta["params"]),
        "device": meta["device"],
        "seed": int(meta["seed"]),
        "flops": flops,
        "val_bpb": bpb,
        "x_steps": x_steps,
    }
    if protocol == "prequential":
        budget = float(meta["pretrain_flop_budget"])
        detail = f"stream={meta['eval_bytes']}B, pretrain={float(meta['pretrain_flops']):.2e}"
        return RunRecord(detail=detail, budget=budget, **common)
    budget = float(meta["flop_budget"])
    eval_seq_len = int(meta["eval_seq_len"])
    val_fraction = float(meta["val_fraction"])
    detail = f"ctx={eval_seq_len}, val={val_fraction:.2f}, budget={budget:.1e}"
    return RunRecord(
        detail=detail,
        budget=budget,
        eval_seq_len=eval_seq_len,
        val_fraction=val_fraction,
        **common,
    )


def collect_runs(runs_dir: str | Path) -> list[RunRecord]:
    """Load all run logs in ``runs_dir``, sorted best (lowest final bpb) first."""
    records = [load_run(p) for p in sorted(Path(runs_dir).glob("*.jsonl"))]
    records.sort(key=lambda r: r.final_val_bpb)
    return records


def protocol_warnings(records: list[RunRecord]) -> list[str]:
    """Comparability warnings: ranking by final bpb is only fair within one
    protocol, one eval protocol, and one FLOP budget. The bpb-vs-FLOP *plot* spans
    budgets (and shows both protocols) on purpose; the *table* ranking does not."""
    out: list[str] = []
    protocols = {r.protocol for r in records}
    if len(protocols) > 1:
        out.append(
            f"runs span multiple protocols {sorted(protocols)} -- final-bpb ranking is "
            "comparable only within a protocol (amortized val bpb vs prequential bpb differ)."
        )
    amortized = [r for r in records if r.protocol == "amortized"]
    eval_protocols = {(r.eval_seq_len, r.val_fraction) for r in amortized}
    if len(eval_protocols) > 1:
        out.append(
            "amortized runs span multiple eval protocols (eval_seq_len, val_fraction): "
            f"{sorted(eval_protocols)} -- not comparable."
        )
    budgets = {r.budget for r in records}
    if len(budgets) > 1:
        out.append(
            f"runs span multiple FLOP budgets {sorted(budgets)} -- rank by final bpb only "
            "within an equal budget; the plot shows the curves across budgets."
        )
    return out


def build_table(records: list[RunRecord]) -> str:
    """Render the leaderboard as a markdown table (best run first), protocol-aware."""
    rows: list[str] = [f"> WARNING: {w}" for w in protocol_warnings(records)]
    if rows:
        rows.append("")
    rows.append(
        "| rank | run | protocol | model | params | final FLOPs | final bpb | detail |\n"
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |"
    )
    for rank, r in enumerate(records, start=1):
        rows.append(
            f"| {rank} | {r.run} | {r.protocol} | {r.model} | {r.params:,} | "
            f"{r.final_flops:.3e} | {r.final_val_bpb:.4f} | {r.detail} |"
        )
    return "\n".join(rows)


def plot_bpb_vs_flops(records: list[RunRecord], out_png: str | Path) -> Path:
    """Draw the budget-sweep curve: ONE final point per run (full-stream bpb vs
    final FLOPs), connected within each (model, protocol) group in FLOP order.

    This is the comparison curve the ADR means — each point is a finished run at a
    budget. (Within-run prefix trajectories are a separate diagnostic; for a frozen
    run they slope down only because the running average smooths and context grows,
    which is easy to misread as compute efficiency.)
    """
    style = {"amortized": ("-", "o"), "prequential": ("--", "s")}
    fig, ax = plt.subplots(figsize=(7, 5))
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for r in records:
        if r.flops:
            groups.setdefault((r.model, r.protocol), []).append(r)
    for (model, protocol), group in sorted(groups.items()):
        group.sort(key=lambda r: r.final_flops)
        xs = [r.final_flops for r in group]
        ys = [r.final_val_bpb for r in group]
        linestyle, marker = style.get(protocol, ("-", "o"))
        ax.plot(
            xs, ys, linestyle=linestyle, marker=marker, markersize=6, label=f"{model} [{protocol}]"
        )
    ax.set_xscale("log")
    ax.set_xlabel("total FLOPs (amortized: training | prequential: pretrain + eval)")
    ax.set_ylabel("final bits-per-byte")
    ax.set_title("smolml leaderboard — final bpb vs FLOPs budget sweep (lower is better)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    if groups:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_run_trajectories(records: list[RunRecord], out_png: str | Path) -> Path:
    """Diagnostic: each run's WITHIN-run cumulative-bpb-vs-FLOPs trajectory.

    Not the comparison curve (see :func:`plot_bpb_vs_flops`) — a per-run view of
    how cumulative bpb evolves as the stream / training progresses.
    """
    style = {"amortized": ("-", "o"), "prequential": ("--", "s")}
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in records:
        if not r.flops:
            continue
        linestyle, marker = style.get(r.protocol, ("-", "o"))
        ax.plot(
            r.flops,
            r.val_bpb,
            linestyle=linestyle,
            marker=marker,
            markersize=3,
            label=f"{r.run} [{r.protocol}]",
        )
    ax.set_xscale("log")
    ax.set_xlabel("cumulative FLOPs")
    ax.set_ylabel("cumulative bits-per-byte")
    ax.set_title("smolml — per-run trajectories (diagnostic, NOT the budget-sweep curve)")
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


@dataclass
class ControlRunRecord:
    run: str
    model: str
    params: int
    device: str
    seed: int
    budget: float
    flops: list[int] = field(default_factory=list)
    regret: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    wm_bits: list[float] = field(default_factory=list)

    @property
    def final_regret(self) -> float:
        return self.regret[-1] if self.regret else float("nan")


def load_control_run(path: str | Path) -> ControlRunRecord:
    meta: dict[str, object] = {}
    flops: list[int] = []
    regret: list[float] = []
    reward: list[float] = []
    wm: list[float] = []
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
                regret.append(float(obj["regret"]))
                reward.append(float(obj["mean_reward"]))
                wm.append(float(obj["world_model_bits"]))
    if not meta:
        raise ValueError(f"{path}: missing meta line")
    return ControlRunRecord(
        run=meta["run"],
        model=meta["model"],
        params=int(meta["params"]),
        device=meta["device"],
        seed=int(meta["seed"]),
        budget=float(meta["flop_budget"]),
        flops=flops,
        regret=regret,
        reward=reward,
        wm_bits=wm,
    )


def collect_control_runs(runs_dir: str | Path) -> list[ControlRunRecord]:
    records = [load_control_run(p) for p in sorted(Path(runs_dir).glob("*.jsonl"))]
    records.sort(key=lambda r: r.final_regret)
    return records


def build_control_table(records: list[ControlRunRecord]) -> str:
    rows = [
        "| rank | run | protocol | model | params | final FLOPs | regret | reward | wm bits |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, r in enumerate(records, start=1):
        rows.append(
            f"| {rank} | {r.run} | control | {r.model} | {r.params:,} | "
            f"{(r.flops[-1] if r.flops else 0):.3e} | {r.final_regret:.4f} | "
            f"{(r.reward[-1] if r.reward else float('nan')):.4f} | "
            f"{(r.wm_bits[-1] if r.wm_bits else float('nan')):.4f} |"
        )
    return "\n".join(rows)


def plot_control(records: list[ControlRunRecord], out_png: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in records:
        if r.flops:
            ax.plot(r.flops, r.regret, marker="o", label=r.run)
    ax.set_xscale("log")
    ax.set_xlabel("training FLOPs")
    ax.set_ylabel("regret vs oracle (per step)")
    ax.set_title("Control rung: regret vs FLOPs")
    ax.legend(fontsize=8)
    out = Path(out_png)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def regenerate_control(
    runs_dir: str | Path,
    *,
    table_path: str | Path,
    plot_path: str | Path,
) -> tuple[str, Path]:
    records = collect_control_runs(runs_dir)
    table = build_control_table(records)
    Path(table_path).write_text(table + "\n")
    png = plot_control(records, plot_path)
    return table, png
