"""Command-line entry point: train (amortized), prequential eval, or leaderboard.

Examples
--------
    uv run smolml train --data sample --budget 5e9 --d-model 128 --layers 4
    uv run smolml prequential --data synthetic --synthetic-bytes 200000 \\
        --eval-bytes 400 --pretrain-budget 1e10 --d-model 32 --layers 2
    uv run smolml prequential --data enwik8 --eval-bytes 5000000 --pretrain-budget 1e13
    uv run smolml leaderboard --runs-dir runs
"""

import argparse

from smolml.data.corpus import ByteCorpus, load_sample, prepare_enwik8, synthetic_text8
from smolml.leaderboard import regenerate
from smolml.prequential import PrequentialConfig, prequential_run
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
        # context must hold both training and (fixed) eval windows
        "max_seq_len": max(args.seq_len, args.eval_seq_len),
    }
    cfg = TrainConfig(
        model=args.model,
        model_config=model_config,
        flop_budget=args.budget,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_seq_len=args.eval_seq_len,
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


def _cmd_prequential(args: argparse.Namespace) -> None:
    corpus = _load_corpus(args)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=args.eval_bytes)
    model_config = {
        "d_model": args.d_model,
        "n_layers": args.layers,
        "n_heads": args.heads,
        # KV-cache decode needs the context to hold the whole eval stream
        "max_seq_len": max(args.seq_len, len(eval_stream)),
    }
    cfg = PrequentialConfig(
        model=args.model,
        model_config=model_config,
        pretrain_flop_budget=args.pretrain_budget,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        adapt_interval=args.adapt_interval,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        run_name=args.run_name,
        device=args.device,
    )
    summary = prequential_run(prior, eval_stream, cfg, runs_dir=args.runs_dir)
    print(
        f"run={summary.run} model={summary.model} params={summary.params:,} "
        f"device={summary.device} eval_bytes={summary.eval_bytes} "
        f"pretrain={summary.pretrain_flops:.3e} eval={summary.eval_flops:.3e} "
        f"total={summary.total_flops:.3e} bpb={summary.bpb:.4f}"
    )
    print(f"log: {summary.log_path}")


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
    t.add_argument(
        "--eval-seq-len", type=int, default=128, help="fixed eval context (identical per run)"
    )
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--eval-interval", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--run-name", default=None)
    t.add_argument("--device", default=None, help="override cuda>mps>cpu auto-detect")
    t.add_argument("--runs-dir", default="runs")
    t.set_defaults(func=_cmd_train)

    p = sub.add_parser("prequential", help="prequential/online eval at a total-FLOP budget")
    p.add_argument("--model", default="transformer")
    p.add_argument("--data", default="synthetic", choices=["sample", "synthetic", "enwik8"])
    p.add_argument("--synthetic-bytes", type=int, default=200_000)
    p.add_argument("--enwik8-bytes", type=int, default=None)
    p.add_argument("--eval-bytes", type=int, default=2000, help="final-bytes prequential stream")
    p.add_argument("--pretrain-budget", type=float, default=1e10, help="pretraining-FLOP budget")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--adapt-interval", type=int, default=0, help="0=frozen; k=adapt every k bytes")
    p.add_argument("--checkpoint-interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None, help="override cuda>mps>cpu auto-detect")
    p.add_argument("--runs-dir", default="runs")
    p.set_defaults(func=_cmd_prequential)

    lb = sub.add_parser("leaderboard", help="regenerate table + plot from run logs")
    lb.add_argument("--runs-dir", default="runs")
    lb.add_argument("--table", default="runs/leaderboard.md")
    lb.add_argument("--plot", default="runs/leaderboard.png")
    lb.set_defaults(func=_cmd_leaderboard)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
