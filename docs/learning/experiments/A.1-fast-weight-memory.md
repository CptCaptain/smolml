# Experiment A.1 — fast-weight associative memory: the maiden Source-(iv) candidate

**Status:** mechanism sound and honestly metered; **the Source-(iv) thesis is NOT
supported on this data.** The per-total-FLOP win over the transformer is real and
reproducible but *Pareto-hollow* — a free online unigram dominates both models at
exactly the budgets where the memory helps, and the memory *hurts* once the core is
good (confirmed on a real enwik8 carve). Offline tiny clone + one real-data control;
the lesson is the honest accounting, not the absolute bpb.

## The bet (and why it fails here)

Gradient descent is an expensive way to **memorize** literal patterns. A
[fast-weight associative memory](../concepts/fast-weight-memory.md) memorizes with a
single gradient-free outer-product write — O(1), no backward pass. The bet: hand rote
memorization to the cheap memory, let the gradient core do generalization, and the
gradient FLOPs buy more loss-reduction per FLOP — a real
[Source-(iv)](../concepts/source-iv-advantage.md) claim.

**Measured, the bet does not hold on this corpus.** Three facts kill it:

1. **No literal repetition to monetize.** The 512-byte clone eval stream has **0
   repeated 6-grams and 0 repeated 8-grams** (only 4 repeated 4-grams). There is
   nothing for a rote memorizer to recall.
2. **No division of labor.** Gate weight and recall accuracy are essentially
   identical whether the core is trained (10¹⁰) or untrained: gate mean `a` =
   **0.254 vs 0.255**, recall@1 = **10.8 % vs 11.2 %**. The memory does the *same
   weak thing* regardless of core quality — it is not specializing in the bits the
   core fails on.
3. **At budget 0 there are zero gradient FLOPs to "free."** Yet the memory still
   "wins" there — because it is acting as a fixed weak always-on predictor, not as a
   memory freeing the gradient budget.

So the observed crossover is fully explained by **"a weak always-on frequency-like
predictor helps a bad core and hurts a good one"** — not by free memorization.

## Setup

- **Slow core:** the transformer baseline (`d_model=48, n_layers=3`, 95,568 params),
  pretrained by the default backprop `train_step` on the prior corpus and **frozen**
  during eval. The hybrid shares the baseline's architecture, seed, and pretraining,
  so its core is *bit-identical* to the baseline at each budget — the delta isolates
  the memory.
- **Fast memory:** a fixed `d_model × 256` linear associative store, reset per
  stream, written online. Write = rank-1 outer product `M ← decay·M + key⊗e_byte`;
  read = matvec `key @ M`.
- **Free baseline (A1F1):** a trivial adaptive **online unigram** — predict ∝ counts,
  then increment (~1.3×10⁵ total FLOPs, fully prequential). This is the honest floor.
- **Data:** the offline `text8`-style clone, ADR-0004 carve, final **512 bytes** =
  fixed eval stream; **plus a real enwik8 control** (below). Seed 0, CPU.
- **Budgets:** 0, 2×10⁹, 10¹⁰, 4×10¹⁰ total FLOPs (driven by the pretrain allowance).

### Design decisions resolved

- **Addressing — soft recall on *centered* hidden-state keys.** Raw transformer
  hidden states are severely anisotropic (measured adjacent-key cosine ≈ **1.000** —
  every context points the same way → no addressing). Keying on the centered residual
  `h − μ` (μ = per-stream running mean) drops adjacent cosine to ≈ 0.06; then
  L2-normalize for a bounded cosine recall. Centering is load-bearing, not a detail.
- **Capacity & eviction — superposition + exponential decay** (`memory_decay=0.999`):
  the store is fixed-size by construction (a sum of rank-1 writes), so decay is the
  forgetting policy that bounds its norm and recency-weights it.
- **Combine — confidence-gated probability mixing** (NOT additive logits). Adding
  `γ·recall` into the logits is unbounded and took the 10¹⁰ core from 6.0 to **25+
  bpb**. We mix distributions: `p = (1−a)·softmax(core) + a·softmax(β·recall)`,
  `a = α·max(softmax(β·recall))`. **This does NOT make the mixture safe in general** —
  a convex mix keeps `p` valid but can still starve the truth: at `α=1, β=4` it drives
  the 6.0-bpb core to **9.2 bpb** (worse than uniform). The honest bound at the
  default `α=0.6` is worst-case **+log₂(1/0.4) = +1.32 bits/byte** over the pure core;
  the gate makes that rare, it does not forbid it.
- **Fair FLOP-counting (ADR 0004).** Memory compute is matmuls charged at true cost
  through `smolml.flops`, never undercounted: read `key @ M` = `matmul_flops(1,V,d) =
  2dV`; write = a **dense** `torch.outer` (all `d·V` products materialized) + accumulate
  = `matmul_flops(d,V,1) = 2dV` (the one-hot value is *not* exploited to fake a cheap
  write); decay = `pointwise_flops(d·V)`; key centering/norm + softmax mixing are
  O(d)/O(V), dominated, charged nominally. Read → forward, write/decay → backward. A
  test asserts `step`'s breakdown equals `core_decode + memory` exactly. Per-byte the
  memory adds **~13–33 % over the core's decode depending on context length** (33 % at
  the first byte, 19 % at the 256-byte midpoint, 13 % at full 512 context) — **~19 %
  averaged over the 512-byte stream**. Cheap, but emphatically not free.

## Result — the curve, against the Pareto frontier (offline clone)

Cumulative prequential bpb vs total FLOPs. Cores are identical at each budget; the
only difference vs the transformer is the memory. **The free online unigram is on the
plot** (≈1.3×10⁵ FLOPs, 5.33 bpb):

| pretrain budget | transformer / bpb | fast-weight / bpb | Δ bpb | beats free unigram (5.33)? |
| --- | ---: | ---: | ---: | --- |
| 0 | 1.73×10⁸ / 8.0003 | 2.05×10⁸ / 7.4095 | −0.59 (fw) | **no** (both ≫ 5.33) |
| 2×10⁹ | 1.57×10⁹ / 7.6914 | 1.60×10⁹ / 7.3953 | −0.30 (fw) | **no** (both > 5.33) |
| 10¹⁰ | 9.96×10⁹ / 6.0125 | 1.00×10¹⁰ / 5.9130 | −0.10 (fw) | **no** (both > 5.33) |
| 4×10¹⁰ | 3.93×10¹⁰ / 4.2059 | 3.94×10¹⁰ / 4.4017 | +0.20 (tr) | yes — and **memory loses here** |

**Reframed verdict.** The free online unigram (5.33 bpb at ~10⁵ FLOPs)
**Pareto-dominates both the transformer and the hybrid** at budgets 0/2×10⁹/10¹⁰ —
exactly the regime where fast-weight "beats the transformer." So that headline is a
race between two runners both behind a free pedestrian. The models only pass the
unigram at 4×10¹⁰ — and that is precisely the regime where the **memory subtracts
value** (+0.20 bpb). Net: the memory adds value only where a trivial free baseline
already dominates, and subtracts value in the only regime where the core is worth
running.

## The crossover is a real effect, not seed noise (A1F3)

Paired-difference Δ = (fast-weight − transformer) over **24 i.i.d. 512-byte streams**
with the core fixed per budget:

| budget | Δ (mean ± sd) | SE | wins (fw<tr) | |mean|/SE |
| --- | ---: | ---: | :---: | ---: |
| 0 | −0.5397 ± 0.0830 | 0.0169 | 24/24 | ~32 |
| 10¹⁰ | −0.0806 ± 0.0404 | 0.0082 | 22/24 | ~9.8 |
| 4×10¹⁰ | +0.2416 ± 0.0394 | 0.0081 | 0/24 | ~30 |

Both the low-budget win and the high-budget loss are **~10–30 SE from zero and
reproduce across streams** — a small but robust effect. (This supersedes the earlier
draft's "within seed noise" caveat, which was wrong and unmeasured.)

## External validity (A1F6)

The clone is an adversarially repetition-free corpus. Two controls bracket where the
crossover lands on data with real structure.

**(a) Real enwik8 carve — the project's first real-data point.** Pretrain on a
~1 MB real prior slice; prequential-eval the final **2 KB** of real text (which has
**294 repeated 6-grams** of 2043 positions, vs 0 in the clone). Free unigram on this
tail = 5.17 bpb.

| budget | transformer | fast-weight | Δ |
| --- | ---: | ---: | ---: |
| 0 | 7.9996 | 7.0321 | −0.97 (fw) |
| 10¹⁰ | 6.3864 | 6.1150 | −0.27 (fw) |
| 4×10¹⁰ | **4.9137** | 5.5014 | **+0.59 (tr)** |

The same pattern transfers — and the high-budget loss is **worse on real data**: at
4×10¹⁰ the memory makes a competent real-text core (4.91) *lose to the free unigram*
(5.17) by dragging it to 5.50. A trained transformer already memorizes enwik8's repeats
better than the crude associative store, which then only injects crosstalk.

**(b) Repeat-density sweep (synthetic, the *good* 4×10¹⁰ core).** Tile an
in-distribution motif at decreasing period (denser repetition) and measure Δ:

| motif period | repeated 6-grams | transformer | fast-weight | Δ |
| ---: | ---: | ---: | ---: | ---: |
| 512 (none) | 0 | 4.196 | 4.438 | +0.24 |
| 256 | 251 | 4.121 | 4.307 | +0.19 |
| 128 | 379 | 4.075 | 4.128 | +0.05 |
| 96 | 411 | 4.093 | 3.976 | **−0.12** |
| 64 | 443 | 3.986 | 3.711 | −0.28 |
| 32 | 475 | 3.974 | 3.342 | −0.63 |

A good core flips from net-negative to net-positive only once repetition is **very
dense** — around motif period ~96–128, i.e. ~75–80 % of 6-grams recurring. Real
corpora rarely repeat that densely at the byte level, which is why the memory loses on
enwik8 at the budget where the core is good.

## Hyperparameter fragility (A1F7)

The low-budget win is contingent on **moderate** `alpha`/`beta`. At 10¹⁰ (Δ vs the
pure core): default `α=0.6, β=2` → −0.10 (win), but `α=0.8, β=2` → +0.03 (loss),
`α=0.6, β=4` → +0.10 (loss), `α=0.8, β=4` → +0.47 (loss); best is `α≈0.3` → −0.17.
**Methodological caveat:** the defaults were chosen on the clone eval tail itself (a
mild tune-on-test); the 24 held-out i.i.d. streams above confirm the default-config win
*generalizes* (22/24 at 10¹⁰), but the effect is fragile and a clean protocol would tune
on a disjoint stream.

## What we learned

- **"Free memorization" needs literal repetition the data must actually contain.** On a
  non-repetitive corpus there is nothing to memorize; the memory degenerates to a weak
  always-on predictor. The gate/recall stats being identical trained-vs-untrained is the
  tell: no division of labor occurred.
- **The honest competitor is a free online unigram, not the transformer.** Beating a
  badly-under-trained transformer is not a Source-(iv) result when a ~10⁵-FLOP unigram
  beats both. Every candidate from here should be plotted against that floor.
- **A good core is the memory's enemy, on real data too.** Once the transformer learns
  the structure, the crude associative store only adds crosstalk — the enwik8 loss
  (+0.59) is larger than the clone loss (+0.24).
- **Anisotropy and bounded mixing are real engineering, but they buy robustness, not a
  win.** Centering makes recall possible; probability mixing keeps a wrong recall from
  exploding the bpb — but neither manufactures loss-reduction the data doesn't afford.

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

**Variance table + enwik8 control (exact, bit-reproducible).** Live models, public
API only; a fixed pretrained core is shared by the hybrid and the bare transformer
(`fw.core`), so the delta isolates the memory. The 24 clone streams are
`synthetic_text8(512, seed=1000 + i)` for `i in range(24)` (seeds **1000..1023**); the
enwik8 carve is the final 2048 bytes of `prepare_enwik8(n_bytes=1_000_000)`. The
repeat-density sweep tiles `synthetic_text8(2000, seed=7).data[500:1012]` at periods
{512, 256, 128, 96, 64, 32} through the same 4×10¹⁰ core.

```python
import math, numpy as np, torch
from smolml.data import synthetic_text8
from smolml.data.corpus import prepare_enwik8
from smolml.models import build_model
from smolml.prequential import prequential_bpb, pretrain

CPU = torch.device("cpu")
PT = dict(batch_size=16, seq_len=128, lr=3e-3, weight_decay=0.1,
          betas=(0.9, 0.95), grad_clip=1.0, seed=0, device=CPU)

def fixed_core(cfg, prior, budget):
    torch.manual_seed(0)
    fw = build_model("fast_weight", cfg)
    if budget > 0:
        pretrain(fw, prior, flop_budget=budget, **PT)
    return fw.eval()

# --- 24-stream paired variance (clone), seeds 1000..1023, fixed core ---
CFG = {"d_model": 48, "n_layers": 3, "n_heads": 4, "max_seq_len": 512}
prior, _ = synthetic_text8(200000, seed=0).prequential_carve(eval_bytes=512)
streams = [synthetic_text8(512, seed=1000 + i).data for i in range(24)]
for budget in (0.0, 1e10, 4e10):
    fw = fixed_core(CFG, prior, budget)
    d = np.array([prequential_bpb(fw, s, device=CPU).bpb
                  - prequential_bpb(fw.core, s, device=CPU).bpb for s in streams])
    print(budget, round(d.mean(), 4), round(d.std(ddof=1), 4),
          round(d.std(ddof=1) / math.sqrt(24), 4), int((d < 0).sum()))

# --- real enwik8 control: 1 MB prior, final 2 KB tail (downloads ~36 MB once) ---
ECFG = {"d_model": 48, "n_layers": 3, "n_heads": 4, "max_seq_len": 2048}
ep, ev = prepare_enwik8(n_bytes=1_000_000).prequential_carve(eval_bytes=2048)
for budget in (0.0, 1e10, 4e10):
    fw = fixed_core(ECFG, ep, budget)
    print(budget, round(prequential_bpb(fw.core, ev, device=CPU).bpb, 4),
          round(prequential_bpb(fw, ev, device=CPU).bpb, 4))
```

Prints (seed 0, CPU, torch float32): variance `(−0.5397, 0.0830, 0.0169, 24)`,
`(−0.0806, 0.0404, 0.0082, 22)`, `(+0.2416, 0.0394, 0.0081, 0)`; enwik8
`(7.9996, 7.0321)`, `(6.3864, 6.1150)`, `(4.9137, 5.5014)`.

- The free online unigram (5.33 bpb): predict ∝ Laplace counts, then increment, scored
  prequentially on the same 512-byte eval — see
  `tests/test_fast_weight.py::_online_unigram_bpb` and
  `::test_low_budget_win_is_real_and_pareto_hollow` (asserts both the equal-total-FLOP
  win over the transformer *and* that the free unigram dominates both).
- **Existence proof (not the headline):** on a verbatim-repeated stream the memory does
  recall — `b"the quick brown fox jumps over the lazy dog. " * 8` beats the identical
  frozen core by >1 bpb (`::test_memory_lowers_bpb_on_repeated_substrings`). This only
  re-confirms the unit tests; the verbatim tile is the *least* realistic stream and is
  not load-bearing evidence.
- The enwik8 control downloads ~36 MB once (opt-in, cached, network-bound); all tests
  and the smoke run stay fully offline.

## What a reviewer should scrutinize

- **Memory FLOP honesty:** write charged as a full `2dV` dense outer product (no
  one-hot scatter shortcut), decay as `dV` pointwise — `test_step_flops_equal_core_plus_memory`
  asserts `step`'s breakdown equals `core_decode + memory` exactly. No memory compute
  reaches a scored prediction outside `step`'s returned breakdown.
- **No leakage:** the write at step `pos` uses only the revealed byte and a past key;
  the read conditions on bytes `0..pos`. `test_prediction_at_t_cannot_see_byte_t`
  perturbs the future and asserts past predictions are unchanged.
- **The honest framing:** the win is Pareto-hollow (free unigram dominates the win
  regime) and the effect is data-gated (0 repeated n-grams here). Do not read the
  low-budget transformer win as a Source-(iv) result.
