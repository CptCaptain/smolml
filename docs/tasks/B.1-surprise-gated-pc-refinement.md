# Task B.1 — Surprise-gated predictive-coding refinement (`pc_refine`)

- Status: IN PROGRESS (config **c** / variant **α** this task; config **b** is the follow-up)
- Branch: `task/B.1-surprise-gated-pc` (own PR, do not merge; human merges)
- Depends on: 0.2 (prequential), 0.3 (context-mixing reference = the bar). Builds on A.1's
  hybrid template (`smolml/models/fast_weight.py`).

## Thesis (the Source-(iv) claim on trial)

A frozen, backprop-pretrained transformer core proposes a next-byte distribution; a small
**predictive-coding (PC) module refines it by iterative error-minimization ("settling")**, and
**settling depth + the online weight update are gated by per-byte surprise** so loss-reducing
compute concentrates on hard bytes.

> **(iv) claim, falsifiable:** *at equal total FLOPs, surprise-gated settling reaches lower
> bpb than uniform settling, because we don't waste settling/learning on easy bytes.*

This is the under-measured knob: PC normally settles *uniformly* and tends to lose per-FLOP;
adaptive-compute work (ACT/PonderNet) optimizes latency, almost never **bpb-per-FLOP under an
honest prequential total-FLOP count**. The gate is *intrinsic to the algorithm* (settling
iterations), not bolted on.

## Locked brainstorm decisions (do not relitigate)

- **Fusion shape (i):** one fused mechanism — a frontier learning rule (PC) whose only honest
  path past the per-FLOP ceiling is firing selectively (surprise gate).
- **Core rule:** predictive coding with surprise-gated settling.
- **Config order:** **(c) hybrid first** (backprop-pretrained core + gated PC test-time
  refinement+adaptation) → **(b) pure-PC follow-up** (PC-pretrained core, no backprop anywhere).
- **Variant:** **(α) logit-correction PC** this task — settling produces a *correction added to*
  the core logits, initialized so the correction is zero (= identity to the core). Bounds the
  worst case and directly answers the **A.1 lesson** (a confident-wrong module starving the
  truth — `fast_weight._combine`'s unbounded loss). **(β)** full PC readout head is the natural
  shape for the (b) follow-up, not now.

## Mechanism (variant α, concrete)

Slow core = `Transformer` (reuse `FastWeightConfig.core_config()` pattern), pretrained by the
default backprop `train_step`, frozen at eval. Per position it yields hidden `h ∈ ℝ^d` and base
logits `ℓ_core = head(h) ∈ ℝ^V` (`V = 256`).

PC module = gradient-free runtime state in `DecodeState.cache` (an `nn.Parameter`-free tensor
set, exactly like the fast-weight memory). Latent `z ∈ ℝ^m`. Two runtime weight matrices:
- generative `W ∈ ℝ^{d×m}` (predicts the hidden: `ĥ = W z`),
- readout `Vmat ∈ ℝ^{m×V}` (emits the logit correction: `c = Vmatᵀ z`), initialized to **0**.

**Inference = settling (free-energy descent on the latent).** Minimize
`F(z) = ½‖h − Wz‖²/σ_h² + ½‖z‖²/σ_z²` by `K` gradient steps
`z ← z − η[ −Wᵀ(h − Wz)/σ_h² + z/σ_z² ]`, `z` initialized from the previous step (warm start) or
0. After settling, refined logits `ℓ = ℓ_core + Vmatᵀ z`; predict `softmax(ℓ)`.

**Online learning (after the byte is revealed)** — gradient-free, local, charged honestly:
- readout: `ΔVmat ∝ −η_r · z (p − e_byte)ᵀ` where `p = softmax(ℓ)` (exact CE gradient w.r.t. a
  linear readout — computed locally, no autograd through the core),
- generative: `ΔW ∝ η_g · (h − Wz) zᵀ` (the PC prediction-error rule),
- optional decay toward 0 for stability.

Use the **pending-prediction pattern** from `fast_weight.step`: stash `(z, p)` of the prediction
just made; on the next step, when its target byte is revealed, apply the update. **No leakage** —
the update for the prediction at `pos` uses only the byte revealed at `pos`.

**Surprise gate.**
- *Settling depth* `K` from a **pre-reveal** proxy (entropy or `1 − max p` of the current
  distribution; never peeks at the future byte): `uniform` mode → `K = k_uniform` constant;
  `surprise` mode → `K ∈ [k_min, k_max]` increasing in surprise, thresholds calibrated so the
  realized **mean K ≈ k_uniform** (⇒ matched total settling FLOPs; the win must come from
  *allocation*).
- *Update gating* — apply the weight update only when **post-reveal** surprise `−log p(byte) > θ`;
  charge update FLOPs only when applied.

## Data flow per `step(state, revealed_byte, pos)` (mirrors `fast_weight.step`)

1. **Adapt** (post-reveal of the *previous* byte): if a pending `(z, p)` exists, apply the gated
   `ΔVmat`/`ΔW` against `revealed_byte` → charged to `backward`.
2. **Core decode**: incremental KV decode of `revealed_byte` → `h`, `ℓ_core` (growing regime), or
   sliding-window recompute (regime switch identical to `fast_weight`); charge via the core's
   `decode_step_flops`/`flops`.
3. **Gated settling**: pick `K` from the pre-reveal proxy; settle `z`; readout `ℓ`. Charge every
   iteration's matvecs.
4. Return `(new_state, ℓ, flops)`; `forward = core + settling + readout + gate-arith`,
   `backward = applied-update`. Stash `(z, p)` for the next step's adapt.

## FLOP honesty (the critical surface — see `smolml/flops.py`, the 0.3 finding)

Charge **all** compute via the shared primitives. Per step:
- settling, per iteration: `matmul_flops(1, d, m)` (`Wz`) + `matmul_flops(1, m, d)` (`Wᵀr`) +
  `pointwise_flops` for residual (`d`) and the `z` update (`m`);
- readout: `matmul_flops(1, V, m)` (`Vmatᵀz`) + `pointwise_flops(V)` for the add;
- gate arithmetic: `pointwise_flops(V, ...)` for entropy/threshold;
- update (when applied): `matmul_flops(m, V, 1)` (`ΔVmat` outer) + `matmul_flops(d, m, 1)`
  (`ΔW` outer) + decay `pointwise_flops` → **backward**.
`K` is data-dependent, so total eval FLOPs differ per policy — the harness sums returned `step`
FLOPs, so this is honest by construction. **No compute may be free by omission.**

## Config (`PCRefineConfig`)

Core fields (mirror `FastWeightConfig`): `d_model, n_layers, n_heads, d_ff, max_seq_len,
vocab_size, rope_base, dropout, tie_embeddings`. PC fields: `m` (latent dim), `eta` (settling
step), `sigma_h`, `sigma_z` (or a single prior weight), `k_min`, `k_max`, `k_uniform`,
`gate ∈ {"uniform","surprise"}`, surprise→K mapping params, `update_surprise_threshold`,
`lr_readout`, `lr_gen`, `weight_decay_fast`. Defaults pick `gate="surprise"`. Validate in
`__post_init__`.

## Deliverable — the controlled experiment

Drive `prequential_run` directly (no CLI change) with two `model_config` dicts differing **only**
in `gate` (and matched `k`), distinct `run_name`s `pc_refine_uniform` / `pc_refine_gated`, on the
CI-scale clone (`synthetic`/`text8`). The existing `runs/` already hold the transformer baseline
and `context_mixing_reference`. Regenerate the leaderboard and read the **bpb-vs-total-FLOP**
curve.

A committed, reproducible runner (`smolml/experiments/pc_refine_sweep.py` or equivalent) produces
both runs.

## Acceptance criteria

- Gates green: `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest`.
- `tests/test_pc_refine.py`: FLOP accounting matches hand-computed K-dependent cost; gate
  monotonic in surprise; **settling reduces loss** on a constructed case (more iters → lower
  loss); **no-leakage** (prediction at `pos` independent of bytes `> pos`); reproducible seed;
  `gate="surprise"` reaches **≤** `gate="uniform"` bpb at matched total FLOPs on the clone.
- bpb-vs-total-FLOP curve with **uniform, gated, transformer baseline, and the context-mixing
  ceiling** all plotted. **Pareto-hollow check** (A.1 reflex): state whether any gated win is
  confined to the regime a free online unigram already dominates.
- `docs/learning/experiments/` plain-md note (researcher-authored): hypothesis, setup, curve,
  verdict (beat/lost), what we learned. Docs-builder turns it into the site page; researcher
  confirms accuracy.
- If gated ≯ uniform, **report the negative result honestly** (as A.1 did).

## Follow-up (config b, separate task)

Swap the backprop core for a **PC-pretrained core** (no backprop anywhere) + variant **β** (full
PC readout). Gated to running only if (c)/α shows the gating lever buys loss-per-FLOP.
