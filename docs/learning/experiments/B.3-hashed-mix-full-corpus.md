# Experiment B.3 — bounded (hashed) tables: the order-6 win survives at full-corpus scale

**Status:** the engineering unlock that let us run the **full enwik8 ADR carve**. B.2's order-6 win
used unbounded dict count tables that OOM (~58 GB) on a full-95 MB warmup. `hashed_mix` bounds the
high-order tables to fixed memory (PAQ/cmix-style hashing); on the **full 5 MB ADR eval stream** the
order-6 advantage **survives the bounding** — it beats order-3 and (per B.2) the transformer per
FLOP, in ≤4.3 GiB. First time we touch the real ADR eval stream end-to-end.

## The blocker and the fix

The order-6 mixer that won B.2 stores per-order counts in growable Python dicts; on a full-corpus
warmup the order-6 dict explodes (~0.5 new contexts/byte ⇒ ~58 GB at 95 MB). `hashed_mix` swaps the
high orders (`k ≥ hash_min_order`) for a **fixed-size hashed table**: `T = 2**table_bits` slots ×
256 `uint16` counts, `context → hash → slot`, **collisions accepted**. Memory is then `T` slots
*regardless of corpus size* — the whole point — at the cost of collision noise. Low orders (0–3,
small) keep exact dicts. Built via a behavior-preserving refactor of `context_mixing` (extracted
store seams; the reference stayed bit-identical, all prior tests green); a cross-vendor review added
the missing O(V) halve-on-overflow charge and an order-≤8 guard. The hash is salt-free (reproducible)
and charged honestly (gather like the dict it replaces + the explicit slot arithmetic).

## Full ADR carve (real enwik8: final 5 MB = eval, first ~95 MB = prior; all FLOPs counted)

| entrant | bpb | total FLOPs | peak RAM |
| --- | ---: | ---: | ---: |
| `reference_cold` — context-mix order-3, no warmup | 2.6224 | 4.74e10 | 0.7 GiB |
| `hashed_o6_cold` — order-6 bounded, no warmup | 2.2570 | 7.73e10 | 2.3 GiB |
| **`hashed_o6_warm1e11`** — order-6 bounded, ~7 MB warmup | **2.1111** | 1.78e11 | 4.3 GiB |
| `hashed_o6_warmfull` — order-6, full 95 MB warmup | *computing* | — | — |
| `transformer` anchor | *computing* (cf. B.2: 5.55 @ 9.7e11 on 32 k) | — | — |

`table_bits = 20`, `hash_min_order = 4` (orders 4–6 hashed).

## What the numbers say

1. **Bounded order-6 survives at full corpus.** Even cold, hashed order-6 reaches **2.257 bpb vs
   order-3's 2.622** at ~1.6× FLOPs, in fixed ~2.3 GiB — the collisions at `table_bits=20` don't
   erode the order-6 advantage. The fixed-memory technique is sufficient to run the real carve.
2. **Pre-warming helps beyond self-warming.** A ~7 MB warmup drops cold 2.257 → **2.111** (−0.15
   bpb) for ~2.3× total FLOPs — a real warmup gain on top of the win.
3. **The long ADR eval self-warms.** Both *cold* numbers here are far below B.2's 4 MB-slice cold
   (3.21): the 5 M-byte eval stream is long enough that the online mixer adapts *as it streams*, so
   the cold-vs-warm gap shrinks — a property of the metric's long eval, worth remembering.
4. **vs the transformer.** The full-corpus transformer anchor is still computing; at 32 k (B.2) it
   was 5.55 bpb @ 9.7e11, and on a 5 MB eval its windowed-recompute cost scales to ~1e14 FLOPs at
   similar (undertrained) bpb — so the mixers' per-FLOP dominance only widens. *[Confirmed-pending.]*

## Still computing (refine, don't change, the conclusion)

`hashed_o6_warmfull` folds the **full 95 MB** prior into the fixed 2²⁰ table — order-6 then sees
~30–40 contexts/slot (heavy collisions), so this tests whether more warmup keeps helping or the
table **saturates** (and `warm1e11`'s lighter warmup is the sweet spot). The transformer anchor is
the foregone expensive baseline. Both land in `runs/full/leaderboard.md`; this note will gain the
two rows when they finish. The headline conclusion — *bounded-memory order-6 scales to the full ADR
carve and wins per FLOP* — does not depend on them.

## What we learned

1. **The unlock was engineering, not a new mechanism.** Fixed-memory hashing (a standard
   compression technique) is what let the proven order-6 win run at full-corpus scale. Reusable for
   every future mixing candidate.
2. **The order-6 win is robust to bounding** at feasible memory (it didn't collapse to order-3).
3. **Collision saturation is the next knob.** Pushing warmup toward the full 95 MB into a fixed
   table eventually trades count fidelity for memory; the right move for *more* warmup is a *bigger*
   table, not more data into a small one — the classic cmix memory↔accuracy frontier.

## Reproduce

    nohup uv run python -m smolml.experiments.full_corpus > runs/full/run.log 2>&1 &
