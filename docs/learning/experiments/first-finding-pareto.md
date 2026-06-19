# First finding — the fast-weight win is Pareto-hollow (three-way leaderboard)

- Status: done
- Concepts: [loss-per-flop-and-scaling-laws], [prequential-evaluation], [source-iv-advantage], [fast-weight-memory], [context-mixing]
- Plot: `unified-leaderboard.png` (regenerated on `master` from all three merged models)

## What was run

All three models on the **same** prequential protocol, identical to the canonical A.1 / 0.2
runs — synthetic text8 clone (200000 bytes, seed 0), final 512 bytes as the eval stream,
`context_window=512`, slow-core `d_model=48, n_layers=3, seq_len=128` — swept across pretraining
budgets {0, 2e9, 1e10, 4e10} FLOPs, scored as validation bits-per-byte (bpb) vs **total** FLOPs
(pretrain + inference).

| model | budget | total FLOPs | bpb |
| --- | ---: | ---: | ---: |
| context_mixing (reference) | — | 4.28e6 | **4.78** |
| transformer | 0 | 1.73e8 | 8.00 |
| fast_weight | 0 | 2.05e8 | 7.41 |
| transformer | 2e9 | 1.57e9 | 7.69 |
| fast_weight | 2e9 | 1.60e9 | 7.40 |
| transformer | 1e10 | 9.96e9 | 6.01 |
| fast_weight | 1e10 | 1.00e10 | 5.91 |
| transformer | 4e10 | 3.93e10 | 4.21 |
| fast_weight | 4e10 | 3.94e10 | 4.40 |

## The finding (honest)

`fast_weight` (frozen transformer core + online associative memory) beats the transformer
**baseline** at every budget where the core is weak — −0.59 at b0, −0.30 at b2e9, −0.10 at b1e10 —
and **loses** once the core is well-trained: +0.19 at b4e10. The memory helps a weak core and
hurts a good one.

But the win is **Pareto-hollow**: the free online **context-mixing reference** reaches 4.78 bpb
for ~4.3e6 FLOPs — ~1000× cheaper than either neural model, and *lower bpb than both* at every
budget through 1e10 (where the neural models are still at 5.9–6.0). The neural models only overtake
the free reference at 4e10 — and that is exactly the budget where `fast_weight` gives up its edge
and loses to the plain transformer. So every `fast_weight` win over the transformer happens inside
the region a free frequency model already dominates, and the memory's value vanishes precisely
where neural compute finally pays off. A real-enwik8 control reproduced the pattern (memory helps
weak cores, hurts good ones). **Verdict: the Source-(iv) "free memorization" thesis is not
supported on this data** — the memory behaves as a weak always-on frequency model, not rote recall.

## What to visualize (for the docs-builder)

- The bpb-vs-total-FLOP curve (log-x): three series (transformer, fast_weight, context_mixing
  reference) — the embedded `unified-leaderboard.png`, ideally re-rendered as an interactive
  chart where the lone reference point visibly sits below both neural curves until ~1e10.
- The crossover: fast_weight below transformer at low/mid budget, above it at 4e10.
- A callout that "beats the baseline" ≠ "good per FLOP" once the free reference is on the axis —
  this is the methodological lesson worth making click.
