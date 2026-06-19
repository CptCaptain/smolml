# First finding — the fast-weight win is Pareto-hollow (three-way leaderboard)

- Status: done
- Concepts: [loss-per-flop-and-scaling-laws], [prequential-evaluation], [source-iv-advantage], [fast-weight-memory], [context-mixing]
- Plot: `unified-leaderboard.png` (regenerated on `master` from all three merged models)

## What was run

All three models on the **same** prequential protocol — synthetic text8 clone (200000 bytes,
seed 0), final 512 bytes as the eval stream, `context_window=512`, slow-core `d_model=48,
n_layers=3` — swept across pretraining budgets {0, 2e9, 1e10, 4e10} FLOPs, scored as validation
bits-per-byte (bpb) vs **total** FLOPs (pretrain + inference).

| model | budget | total FLOPs | bpb |
| --- | ---: | ---: | ---: |
| context_mixing (reference) | — | 4.28e6 | **4.78** |
| transformer | 0 | 1.73e8 | 8.00 |
| fast_weight | 0 | 2.05e8 | 7.41 |
| transformer | 2e9 | 2.10e9 | 7.06 |
| fast_weight | 2e9 | 2.13e9 | 6.82 |
| transformer | 1e10 | 9.81e9 | 4.61 |
| fast_weight | 1e10 | 9.84e9 | 4.75 |
| transformer | 4e10 | 4.00e10 | 4.16 |
| fast_weight | 4e10 | 4.01e10 | 4.36 |

## The finding (honest)

`fast_weight` (frozen transformer core + online associative memory) beats the transformer
**baseline** at low budgets (−0.59 at b0, −0.23 at b2e9) and **loses** once the core is
well-trained (+0.14 at b1e10, +0.20 at b4e10) — the memory helps a weak core and hurts a good one.

But the win is **Pareto-hollow**: the free online **context-mixing reference** reaches 4.78 bpb
for ~4.3e6 FLOPs — ~1000× cheaper than either neural model and *lower* bpb than both until
~1e10 FLOPs. So `fast_weight`'s wins over the transformer all happen in the regime a free
frequency model already dominates, and it gives up its edge exactly where neural models finally
beat the reference. A real-enwik8 control reproduced the pattern (memory helps weak cores, hurts
good ones). **Verdict: the Source-(iv) "free memorization" thesis is not supported on this data** —
the memory behaves as a weak always-on frequency model, not rote recall.

## What to visualize (for the docs-builder)

- The bpb-vs-total-FLOP curve (log-x): three series (transformer, fast_weight, context_mixing
  reference) — the embedded `unified-leaderboard.png`, ideally re-rendered as an interactive
  chart where the lone reference point visibly Pareto-dominates the low/mid-budget neural runs.
- The crossover: fast_weight below transformer at low budget, above it at high budget.
- A callout that "beats the baseline" ≠ "good per FLOP" once the free reference is on the axis —
  this is the methodological lesson worth making click.
