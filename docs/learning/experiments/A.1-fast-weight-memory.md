# Experiment A.1 — fast-weight associative memory: the maiden Source-(iv) candidate

**Status:** the first real Source-(iv) entrant on the prequential curve. Frozen slow
transformer core + an online, gradient-free fast-weight memory. Offline tiny clone,
not enwik8 — the *shape* of the curve and the FLOP honesty matter, the absolute bpb
does not.

## Hypothesis

Gradient descent is an expensive way to **memorize** literal patterns (many steps to
push a fact into weights). A [fast-weight associative
memory](../concepts/fast-weight-memory.md) memorizes with a single gradient-free
outer-product write — O(1), no backward pass. If we hand rote memorization to the
cheap memory and let the gradient core do only *generalization*, the gradient FLOPs
should buy more loss-reduction per FLOP — a genuine
[Source-(iv)](../concepts/source-iv-advantage.md) claim. The empirical question:
**does the hybrid beat the bare transformer on the bpb-vs-total-FLOP curve?**

## Setup

- **Slow core:** the transformer baseline (`d_model=48, n_layers=3`, 95,568 params),
  pretrained by the default backprop `train_step` on the prior corpus and **frozen**
  during eval. The hybrid shares the baseline's architecture, seed, and pretraining,
  so at each budget its core weights are *bit-identical* to the baseline's — the
  comparison isolates the memory and nothing else.
- **Fast memory:** a fixed `d_model × 256` linear associative store `M`, reset per
  stream, written online during prequential eval. Write = rank-1 Hebbian outer
  product `M ← decay·M + key⊗e_byte`; read = matvec `key @ M`.
- **Data:** the offline `text8`-style clone, carved per ADR 0004 — the final **512
  bytes** are the fixed prequential eval stream (never trained on); the prefix is the
  prior. No network, fully deterministic, seed 0, CPU.
- **Four total-FLOP budgets** driven by the pretraining allowance: 0, 2×10⁹, 10¹⁰,
  4×10¹⁰. Total FLOPs = pretrain + Σ per-byte `step` (decode + memory read/write).

### Design questions resolved

- **Addressing — soft recall on *centered* hidden-state keys.** The key is the
  frozen core's final hidden state, so an exact context repeat reproduces the key
  exactly (exact recall) and a near-repeat recalls gracefully — the core
  generalizes, the memory does rote. The catch found in testing: **raw transformer
  hidden states are severely anisotropic** (measured adjacent-key cosine ≈ **1.000** —
  every context points the same way), which gives an associative store no addressing
  discrimination and lets crosstalk dominate. Fix: key on the **centered** residual
  `h − μ` (μ a per-stream running mean), which strips the shared common-mode
  component; adjacent cosine drops to ≈ 0.06. Then L2-normalize so recall is a
  bounded cosine similarity.
- **Capacity & eviction — superposition + exponential decay.** `M` is a superposition
  (a sum of rank-1 writes), so it is fixed-size by construction; FIFO/LRU slots do
  not apply. The natural forgetting policy is exponential decay `M ← decay·M` per
  write (`memory_decay = 0.999`): it bounds the store's norm on a long stream and
  recency-weights it, so stale crosstalk fades.
- **Combine — bounded, confidence-gated probability mixing.** Adding `γ·recall`
  straight into the core logits is *unbounded* and catastrophic: it took the 10¹⁰-budget
  core from 6.0 bpb to **25+ bpb** when recall was noisy. We instead mix the two as
  *distributions*, `p = (1−a)·softmax(core) + a·softmax(β·recall)`, with the weight
  `a = α·max(softmax(β·recall))` **gated by recall confidence** — the memory speaks
  only when its recall is peaked (a true match) and stays silent on diffuse crosstalk.
  This bounds the worst-case damage and is what makes the candidate robust.
- **Fair FLOP-counting (ADR 0004).** The memory's dominant compute is matmuls,
  charged at their **true performed cost** through `smolml.flops`, never undercounted:
  read `key @ M` = `matmul_flops(1, V, d) = 2dV`; write = a *dense* outer product
  (`torch.outer` materializes all `d·V` products; the accumulate adds `d·V`) =
  `matmul_flops(d, V, 1) = 2dV`. The one-hot value is **not** exploited to fake a
  cheap write — that would be the exact "elementwise work scored as free" cheat the
  instrument guards against. Decay (`M *= decay`) is a dense `d·V` multiply charged
  via `pointwise_flops`; key centering/normalization and softmax mixing are O(d)/O(V),
  dominated by the O(dV) matmuls but charged nominally in good faith. A test asserts
  `step`'s returned breakdown equals `core_decode + memory` exactly (charge == reality),
  with the read in **forward** and the write/decay in **backward** (so `backward > 0`
  marks counted continual learning). Per byte the memory costs ≈ `5dV` ≈ 6.1×10⁴ FLOPs
  (d=48), about **+19 %** on top of the core's decode — cheap, but emphatically not free.

## Result — the headline curve (offline clone)

Cumulative prequential bpb vs total FLOPs, hybrid vs the bare transformer at four
budgets. Cores are identical at each budget; the only difference is the fast memory.

| pretrain budget | transformer total FLOPs / bpb | fast-weight total FLOPs / bpb | Δ bpb |
| --- | ---: | ---: | ---: |
| 0 (untrained, frozen) | 1.73×10⁸ / 8.0003 | 2.05×10⁸ / **7.4095** | **−0.59** ✅ |
| 2×10⁹ | 1.57×10⁹ / 7.6914 | 1.60×10⁹ / **7.3953** | **−0.30** ✅ |
| 10¹⁰ | 9.96×10⁹ / 6.0125 | 1.00×10¹⁰ / **5.9130** | **−0.10** ✅ |
| 4×10¹⁰ | 3.93×10¹⁰ / **4.2059** | 3.94×10¹⁰ / 4.4017 | **+0.20** ❌ |

**Verdict: mixed, and honestly so.** The fast memory **wins at the three cheaper
budgets** — when the gradient core is under-trained, the online memory adds real
loss-reduction at a ~19 % FLOP premium it more than pays back (e.g. at budget 0 it is
0.59 bpb lower for only 1.19× the FLOPs — a clearly better point on the curve). But at
the **largest budget it loses**: once the core has learned the clone well (4.21 bpb),
the memory's recall on this *non-repetitive* corpus is mostly crosstalk, and even the
gated mixture's small residual leak plus the FLOP overhead make the hybrid a net 0.20
bpb worse.

## Does the mechanism actually work? (memorization-friendly stream)

The high-budget loss is a property of the **corpus**, not the mechanism: the random
clone has almost no literal long-range repetition for the memory to monetize. Re-run
the same two models (same clone pretraining) on an offline eval stream **with** literal
repeats — an in-distribution clone passage tiled verbatim — and the memory wins
decisively at *both* budgets, including the well-trained core that lost above:

| pretrain budget | transformer bpb | fast-weight bpb | Δ bpb |
| --- | ---: | ---: | ---: |
| 0 | 8.0130 | **6.3113** | **−1.70** ✅ |
| 10¹⁰ | 5.9565 | **5.1501** | **−0.81** ✅ |

This is the Source-(iv) story made concrete: when bytes *repeat*, the hybrid writes
"context→next byte" on first sight and recalls it for ~free on every repeat, so its
prequential bill drops far below the frozen core's — exactly the bet, and it holds even
once the core is trained.

## What we learned

- **Free memorization is real but data-gated.** The memory's loss-per-FLOP win scales
  with how much the stream *repeats*. On repetitive data it wins across the board; on a
  near-random corpus it helps only while the core is under-trained, then the crosstalk
  + FLOP overhead flip it to a (small) loss. The clone is an adversarially poor case for
  rote memory; the win on the repeat stream is the mechanism's true signal.
- **Anisotropy is the silent killer.** A naive read of the raw hidden state as a key
  fails completely (adjacent cosine 1.0 → no addressing). Centering the key is not a
  detail — it is the difference between working and not.
- **Combine in probability space, never in logit space.** Additive recall is unbounded
  and blew a 6-bpb core to 25+ bpb. A confidence-gated *distribution* mixture bounds the
  downside and is what lets the memory help without ever exploding — the single most
  important robustness decision.
- **Honesty has a price, and we paid it.** Charging the dense outer-product write (not a
  cheap one-hot scatter) costs the candidate ~19 % eval FLOPs it could have hidden. That
  premium is exactly why the win must come from *better recall*, not from a cheaper
  arithmetic trick — the whole point of the metric.

## Reproduce

```bash
# headline curve on the offline clone (both models, four budgets)
for b in 0 2e9 1e10 4e10; do
  for m in transformer fast_weight; do
    uv run smolml prequential --model $m --data synthetic --synthetic-bytes 200000 \
        --eval-bytes 512 --context-window 512 --pretrain-budget $b --d-model 48 \
        --layers 3 --seq-len 128 --run-name ${m}-preq-b$b
  done
done
uv run smolml leaderboard --runs-dir runs
```

The memorization-friendly comparison (clone pretraining, verbatim-repeat eval stream)
is driven directly from `smolml.prequential.prequential_run` with a tiled eval stream;
see the behavioral test `tests/test_fast_weight.py::test_memory_lowers_bpb_on_repeated_substrings`,
which pins the same effect (the hybrid beats its own frozen core by >1 bpb on repeats).

## What a reviewer should scrutinize

- **Memory FLOP honesty.** Confirm the write is charged as a full `2dV` matmul (dense
  `torch.outer`, no one-hot shortcut) and the decay as a `dV` pointwise multiply —
  `test_step_flops_equal_core_plus_memory` asserts `step`'s breakdown equals
  `core_decode + memory` exactly. The risk to look for is any path where memory compute
  reaches a scored prediction without flowing through `step`'s returned breakdown
  (there is none — read, write, and combine all happen inside `step`).
- **No leakage.** The write at step `pos` uses only the already-revealed byte and a past
  key; the read conditions on bytes `0..pos`. `test_prediction_at_t_cannot_see_byte_t`
  perturbs the future and asserts past predictions are unchanged.
- **Single-seed caveat (per the harness).** These are single-seed point estimates on
  512 bytes; the small clone deltas (especially the −0.10 at 10¹⁰ and +0.20 at 4×10¹⁰)
  are within plausible seed noise. The repeat-stream gaps (−0.8 to −1.7) are the
  load-bearing evidence, not the clone hairlines.
