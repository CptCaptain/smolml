"""Command-line entry point: train a run, or regenerate the leaderboard.

Examples
--------
    uv run smolml train --data sample --budget 5e9 --d-model 128 --layers 4
    uv run smolml train --data enwik8 --enwik8-bytes 5000000 --budget 1e13
    uv run smolml leaderboard --runs-dir runs
"""

import argparse

from smolml.data.corpus import ByteCorpus, load_sample, prepare_enwik8, synthetic_text8
from smolml.leaderboard import regenerate
from smolml.train import TrainConfig, train_run


def _load_corpus(args: argparse.Namespace) -> ByteCorpus:
    if args.data == "sample":
        return load_sample()
    if args.data == "synthetic":
        return synthetic_text8(args.synthetic_bytes, seed=args.seed)
    if args.data == "enwik8":
        return prepare_enwik8(n_bytes=args.enwik8_bytes)
    raise ValueError(f"unknown data source {args.data!r}")


def _cmd_train(args: argparse.Namespace) -> None:
    corpus = _load_corpus(args)
    model_config = {
        "d_model": args.d_model,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "max_seq_len": args.seq_len,
    }
    cfg = TrainConfig(
        model=args.model,
        model_config=model_config,
        flop_budget=args.budget,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        seed=args.seed,
        eval_interval=args.eval_interval,
        run_name=args.run_name,
        device=args.device,
    )
    summary = train_run(corpus, cfg, runs_dir=args.runs_dir)
    print(
        f"run={summary.run} model={summary.model} params={summary.params:,} "
        f"device={summary.device} steps={summary.steps} "
        f"flops={summary.total_flops:.3e} val_bpb={summary.final_val_bpb:.4f} "
        f"({summary.elapsed_sec:.1f}s)"
    )
    print(f"log: {summary.log_path}")


def _cmd_leaderboard(args: argparse.Namespace) -> None:
    table, png = regenerate(args.runs_dir, table_path=args.table, plot_path=args.plot)
    print(table)
    print(f"\nplot: {png}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="smolml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a run to a fixed FLOP budget")
    t.add_argument("--model", default="transformer")
    t.add_argument("--data", default="sample", choices=["sample", "synthetic", "enwik8"])
    t.add_argument("--synthetic-bytes", type=int, default=1_000_000)
    t.add_argument("--enwik8-bytes", type=int, default=None)
    t.add_argument("--budget", type=float, default=5e9, help="training-FLOP budget")
    t.add_argument("--d-model", type=int, default=128)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--seq-len", type=int, default=128)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--eval-interval", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--run-name", default=None)
    t.add_argument("--device", default=None, help="override cuda>mps>cpu auto-detect")
    t.add_argument("--runs-dir", default="runs")
    t.set_defaults(func=_cmd_train)

    lb = sub.add_parser("leaderboard", help="regenerate table + plot from run logs")
    lb.add_argument("--runs-dir", default="runs")
    lb.add_argument("--table", default="runs/leaderboard.md")
    lb.add_argument("--plot", default="runs/leaderboard.png")
    lb.set_defaults(func=_cmd_leaderboard)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
