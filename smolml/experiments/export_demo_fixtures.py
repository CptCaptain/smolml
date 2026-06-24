"""Export the interactive-demo model layer: tiny trained / distilled weights and
per-model parity fixtures consumed by the browser JS ports + the parity harness.

Outputs (all under ``docs/learning/public/data/demos/``, each < ~1 MB):

- ``<name>.weights.json`` — float32 weights, base64 little-endian (compact). Only
  the models that need exported weights: ``transformer`` (tiny trained),
  ``reservoir`` (frozen core + short-distilled readout), ``reservoir_plastic``
  (frozen core + seed readout the online rule then adapts).
- ``<name>.fixture.json`` — the parity fixture: the fixed stream/rollout plus,
  per position, the model's argmax and scored value (bits for byte models,
  rewards for control), and the cumulative metric, produced by running the REAL
  Python ``step`` (byte: ``prequential_bpb`` collect_logits; control: a fixed
  seeded greedy rollout with baked env initial conditions).

PRECISION CONTRACT: weights are stored as float32 (compact) but the fixtures are
generated from those exact float32 values upcast to float64 (``model.double()``),
so the float64 JS ports match Python to ~1e-12 and the gate (bpb/reward within
1e-3, argmax identical) holds with wide margin.

DEPENDENCY: the three CONTROL models live on their candidate branches
(``reservoir``/``reservoir_plastic`` on ``task/C.A.1b-reservoir-plastic``,
``chemotaxis_min`` on ``task/C.A.2-chemotaxis-minimal``). This script imports them
normally; it runs as-is on ``main`` only once those candidate PRs land. The three
BYTE models (``context_mixing``/``delta_mix``/``transformer``) are on ``main`` —
no extra dependency. Run: ``uv run python -m smolml.experiments.export_demo_fixtures``.
"""

import base64
import json
import math
from pathlib import Path

import numpy as np
import torch

from smolml.control_train import ControlTrainConfig, distill_train_run
from smolml.data.corpus import VOCAB_SIZE, prepare_enwik8
from smolml.envs.chemotaxis import (
    ChemoConfig,
    ChemoEnv,
    action_slice,
    action_token,
    conc_slice,
    vocab_size,
)
from smolml.models.registry import build_model
from smolml.prequential import prequential_bpb, score_bits

OUT_DIR = Path("docs/learning/public/data/demos")

# ── demo configs (single source of truth; embedded into the JSON) ──────────────
# Byte race: one shared enwik8 snippet, warmed/decoded live by every byte model.
BYTE_STREAM_OFFSET = 2_000_000  # a mid-article slice, disjoint from the train prefix
BYTE_STREAM_LEN = 2048
TRANSFORMER_CFG = {
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "d_ff": 128,
    "max_seq_len": 64,
    "vocab_size": VOCAB_SIZE,
    "tie_embeddings": True,
}
TRANSFORMER_TRAIN = {
    "train_bytes": 1_000_000,  # enwik8 prefix used to train (disjoint from the stream)
    "steps": 3000,
    "batch_size": 32,
    "seq_len": 64,
    "lr": 3e-3,
    "weight_decay": 0.1,
    "betas": (0.9, 0.95),
    "grad_clip": 1.0,
    "seed": 0,
}
CONTEXT_MIXING_CFG = {"max_order": 3, "alpha": 0.5, "lr": 0.02, "vocab_size": VOCAB_SIZE}
DELTA_MIX_CFG = {
    "max_order": 3,
    "alpha": 0.5,
    "lr": 0.02,
    "vocab_size": VOCAB_SIZE,
    "delta_dim": 1 << 14,  # demo-sized table (sparse in JS; light at any size)
    "delta_eta": 0.1,
    "delta_orders": [3, 4, 5, 6, 7, 8],
    "delta_signed": True,
}

# Control: shared ChemoEnv + a fixed seeded greedy rollout for parity.
CHEMO = ChemoConfig(width=16, levels=8, sigma=2.0, horizon=64)
CONTROL_VOCAB = vocab_size(CHEMO)
CONTROL_MAX_SEQ = 2 * CHEMO.horizon + 1
CONTROL_EPISODES = 4
CONTROL_SEED = 0
RESERVOIR_CFG = {"d_res": 64, "leak": 0.6, "spectral_radius": 0.9, "seed": 0}
RESERVOIR_DISTILL_BUDGET = 6e9  # short distillation of the readout (a few hundred steps)
RESERVOIR_PLASTIC_CFG = {
    "d_res": 64,
    "leak": 0.6,
    "spectral_radius": 0.9,
    "seed": 0,
    "lr_wm": 0.5,
    "lr_pol": 0.03,
    "reward_decay": 0.7,
}


def f32_b64(arr: np.ndarray) -> str:
    """Serialize an array as little-endian float32 base64 (the JS decodeF32 format)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def _tensor_b64(t: torch.Tensor) -> str:
    return f32_b64(t.detach().cpu().contiguous().numpy().ravel())


# ── byte models ────────────────────────────────────────────────────────────────
def load_byte_stream() -> tuple[np.ndarray, str]:
    corpus = prepare_enwik8()
    data = corpus.data
    sl = data[BYTE_STREAM_OFFSET : BYTE_STREAM_OFFSET + BYTE_STREAM_LEN]
    stream = np.asarray(sl, dtype=np.int64)
    text = bytes(int(b) for b in stream).decode("latin-1")
    return stream, text


def train_tiny_transformer() -> torch.nn.Module:
    """Train the tiny transformer briefly on a fixed enwik8 prefix (float32)."""
    tt = TRANSFORMER_TRAIN
    torch.manual_seed(tt["seed"])
    device = torch.device("cpu")
    corpus = prepare_enwik8()
    train_data = corpus.data[: tt["train_bytes"]]
    model = build_model("transformer", dict(TRANSFORMER_CFG)).to(device)
    model.train()
    opt = model.configure_optimizer(lr=tt["lr"], weight_decay=tt["weight_decay"], betas=tt["betas"])
    gen = torch.Generator().manual_seed(tt["seed"])
    bs, sl = tt["batch_size"], tt["seq_len"]
    max_start = len(train_data) - sl - 1
    for step in range(tt["steps"]):
        ix = torch.randint(max_start + 1, (bs,), generator=gen)
        x = torch.empty((bs, sl), dtype=torch.long)
        y = torch.empty((bs, sl), dtype=torch.long)
        for b, i in enumerate(ix.tolist()):
            chunk = torch.from_numpy(train_data[i : i + sl + 1].astype(np.int64))
            x[b], y[b] = chunk[:-1], chunk[1:]
        _loss, _ = model.train_step((x.to(device), y.to(device)), opt, grad_clip=tt["grad_clip"])
    model.eval()
    return model


def export_transformer_weights(model: torch.nn.Module) -> dict:
    cfg = model.config
    blocks = []
    for blk in model.blocks:
        blocks.append(
            {
                "norm1": _tensor_b64(blk.norm1.weight),
                "qkv": _tensor_b64(blk.attn.qkv.weight),
                "proj": _tensor_b64(blk.attn.proj.weight),
                "norm2": _tensor_b64(blk.norm2.weight),
                "fc1": _tensor_b64(blk.mlp.fc1.weight),
                "fc2": _tensor_b64(blk.mlp.fc2.weight),
            }
        )
    return {
        "model": "transformer",
        "config": {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "max_seq_len": cfg.max_seq_len,
            "vocab_size": cfg.vocab_size,
            "rope_base": cfg.rope_base,
        },
        "tok_emb": _tensor_b64(model.tok_emb.weight),
        "norm_f": _tensor_b64(model.norm_f.weight),
        "blocks": blocks,
    }


def byte_fixture(model: torch.nn.Module, name: str, cfg: dict, stream: np.ndarray, text: str) -> dict:
    """Run the REAL prequential step (float64) and dump per-position argmax + bits."""
    device = torch.device("cpu")
    res = prequential_bpb(model, stream, device=device, collect_logits=True)
    logits = res.predicted_logits
    n = len(stream)
    argmax = [0] * n          # index 0 is the uniform prior (argmax 0)
    scored = [0.0] * n
    scored[0] = score_bits(torch.zeros(VOCAB_SIZE), int(stream[0]))  # 8.0 bits, uniform
    for p in range(1, n):
        lg = logits[p]
        argmax[p] = int(torch.argmax(lg))
        scored[p] = score_bits(lg, int(stream[p]))
    return {
        "model": name,
        "kind": "byte",
        "config": cfg,
        "stream": [int(b) for b in stream],
        "seed_text": text,
        "argmax": argmax,
        "scored_bits": scored,
        "total_bits": float(res.total_bits),
        "n_bytes": n,
        "cumulative_bpb": float(res.bpb),
    }


# ── control models ───────────────────────────────────────────────────────────
def greedy_action(logits: torch.Tensor, asl: slice) -> int:
    return int(torch.argmax(logits[asl]))


def control_rollout(model: torch.nn.Module, name: str, model_cfg: dict) -> dict:
    """Fixed seeded GREEDY rollout mirroring control_eval.evaluate_control, recording
    baked env initial conditions + per-position argmax/reward so the JS port can
    reproduce the trajectory deterministically (no RNG port)."""
    csl, asl = conc_slice(CHEMO), action_slice(CHEMO)
    levels = CHEMO.levels
    episodes = []
    agent_total = 0.0
    n_steps = 0
    for ep in range(CONTROL_EPISODES):
        ep_seed = CONTROL_SEED * 100003 + ep
        env = ChemoEnv(CHEMO, split="eval", seed=ep_seed)
        env_init = {
            "drift_rate": float(env.drift_rate),
            "drift_dir": int(env.drift_dir),
            "mu": int(env.mu),
            "p": int(env.p),
            "phase": 0.0,
        }
        state = model.init_prequential_state()
        c = env.reset()
        tape = [c]
        pos = 0
        actions, rewards, conc_tokens = [], [], [c]
        action_argmax, conc_argmax, wm_bits = [], [], []
        for _t in range(CHEMO.horizon):
            state, logits, _f = model.step(state, tape[pos], pos)
            a = greedy_action(logits, asl)
            action_argmax.append(a)
            actions.append(a)
            pos += 1
            tape.append(action_token(CHEMO, a))
            state, logits_pred, _f = model.step(state, tape[pos], pos)
            conc_argmax.append(int(torch.argmax(logits_pred[csl])))
            pos += 1
            c, reward = env.step(a)
            wm_bits.append(score_bits(logits_pred[csl], c))
            tape.append(c)
            conc_tokens.append(c)
            rewards.append(float(reward))
            agent_total += reward
            n_steps += 1
        episodes.append(
            {
                "env_init": env_init,
                "tape": [int(t) for t in tape],
                "actions": [int(a) for a in actions],
                "conc_tokens": [int(c) for c in conc_tokens],
                "rewards": rewards,
                "action_argmax": action_argmax,
                "conc_argmax": conc_argmax,
                "wm_bits": wm_bits,
            }
        )
    return {
        "model": name,
        "kind": "control",
        "config": dict(model_cfg, levels=levels),
        "env": {
            "width": CHEMO.width,
            "levels": CHEMO.levels,
            "sigma": CHEMO.sigma,
            "horizon": CHEMO.horizon,
            "vocab_size": CONTROL_VOCAB,
        },
        "episodes": episodes,
        "cumulative_reward": float(agent_total),
        "mean_reward": float(agent_total / n_steps),
        "n": n_steps,
    }


def export_reservoir_weights(model: torch.nn.Module, name: str, cfg: dict) -> dict:
    return {
        "model": name,
        "config": {
            "d_res": model.config.d_res,
            "leak": model.config.leak,
            "vocab_size": model.config.vocab_size,
            "lr_wm": getattr(model.config, "lr_wm", None),
            "lr_pol": getattr(model.config, "lr_pol", None),
            "reward_decay": getattr(model.config, "reward_decay", None),
        },
        "W_in": _tensor_b64(model.core.W_in),
        "W_res": _tensor_b64(model.core.W_res),
        "W_out": _tensor_b64(model.readout.weight),
        "b_out": _tensor_b64(model.readout.bias),
    }


def build_control_model(name: str, extra: dict) -> torch.nn.Module:
    mc = dict(extra)
    mc["vocab_size"] = CONTROL_VOCAB
    mc["max_seq_len"] = CONTROL_MAX_SEQ
    return build_model(name, mc).to(torch.device("cpu"))


def write_json(path: Path, obj: dict) -> int:
    path.write_text(json.dumps(obj, separators=(",", ":")))
    return path.stat().st_size


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sizes: list[tuple[str, int]] = []

    # ── byte race ──
    stream, text = load_byte_stream()
    print(f"byte stream: {len(stream)} bytes @ enwik8[{BYTE_STREAM_OFFSET}]")

    cm = build_model("context_mixing", dict(CONTEXT_MIXING_CFG)).to(torch.device("cpu"))
    fx = byte_fixture(cm, "context_mixing", CONTEXT_MIXING_CFG, stream, text)
    sizes.append(("context_mixing.fixture.json", write_json(OUT_DIR / "context_mixing.fixture.json", fx)))
    print(f"  context_mixing bpb={fx['cumulative_bpb']:.4f}")

    dm = build_model("delta_mix", dict(DELTA_MIX_CFG)).to(torch.device("cpu"))
    fx = byte_fixture(dm, "delta_mix", DELTA_MIX_CFG, stream, text)
    sizes.append(("delta_mix.fixture.json", write_json(OUT_DIR / "delta_mix.fixture.json", fx)))
    print(f"  delta_mix bpb={fx['cumulative_bpb']:.4f}")

    print("training tiny transformer ...")
    tf = train_tiny_transformer()
    w = export_transformer_weights(tf)
    sizes.append(("transformer.weights.json", write_json(OUT_DIR / "transformer.weights.json", w)))
    tf.double()  # fixture in float64 over the float32-valued (upcast) weights
    fx = byte_fixture(tf, "transformer", w["config"], stream, text)
    sizes.append(("transformer.fixture.json", write_json(OUT_DIR / "transformer.fixture.json", fx)))
    print(f"  transformer bpb={fx['cumulative_bpb']:.4f} params={tf.num_params()}")

    # ── control trio ──
    print("distilling reservoir readout ...")
    ctc = ControlTrainConfig(
        model="reservoir",
        model_config=dict(RESERVOIR_CFG),
        flop_budget=RESERVOIR_DISTILL_BUDGET,
        width=CHEMO.width,
        levels=CHEMO.levels,
        sigma=CHEMO.sigma,
        horizon=CHEMO.horizon,
        seed=CONTROL_SEED,
        device="cpu",
        run_name="reservoir-demo-distill",
    )
    _summary, res_model = distill_train_run(ctc, runs_dir="runs", return_model=True)
    rw = export_reservoir_weights(res_model, "reservoir", RESERVOIR_CFG)
    sizes.append(("reservoir.weights.json", write_json(OUT_DIR / "reservoir.weights.json", rw)))
    res_model.double()
    fx = control_rollout(res_model, "reservoir", rw["config"])
    sizes.append(("reservoir.fixture.json", write_json(OUT_DIR / "reservoir.fixture.json", fx)))
    print(f"  reservoir mean_reward={fx['mean_reward']:.4f}")

    print("building reservoir_plastic (seed readout, online adaptation) ...")
    rp = build_control_model("reservoir_plastic", dict(RESERVOIR_PLASTIC_CFG))
    rpw = export_reservoir_weights(rp, "reservoir_plastic", RESERVOIR_PLASTIC_CFG)
    sizes.append(("reservoir_plastic.weights.json", write_json(OUT_DIR / "reservoir_plastic.weights.json", rpw)))
    rp.double()
    fx = control_rollout(rp, "reservoir_plastic", rpw["config"])
    sizes.append(("reservoir_plastic.fixture.json", write_json(OUT_DIR / "reservoir_plastic.fixture.json", fx)))
    print(f"  reservoir_plastic mean_reward={fx['mean_reward']:.4f}")

    print("building chemotaxis_min (untrained, config defaults) ...")
    cmin = build_control_model("chemotaxis_min", {})
    cmin.double()
    cmin_cfg = {
        "vocab_size": CONTROL_VOCAB,
        "leak_init": cmin.config.leak_init,
        "gain_init": cmin.config.gain_init,
        "stay_bias_init": cmin.config.stay_bias_init,
        "climb_init": cmin.config.climb_init,
        "sharpness_init": cmin.config.sharpness_init,
        "baseline_init": cmin.config.baseline_init,
    }
    fx = control_rollout(cmin, "chemotaxis_min", cmin_cfg)
    sizes.append(("chemotaxis_min.fixture.json", write_json(OUT_DIR / "chemotaxis_min.fixture.json", fx)))
    print(f"  chemotaxis_min mean_reward={fx['mean_reward']:.4f}")

    print("\nwrote:")
    for fn, sz in sizes:
        print(f"  {fn:<34} {sz / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
