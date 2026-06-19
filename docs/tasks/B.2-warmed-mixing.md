# Task B.2 — warmed online mixing (foundation) + bold layer on the warmed backbone

- Status: IN PROGRESS — Phase 1 (warm_mix foundation) first; bold layer (cascade/growth) staged after.
- Branch: `task/B.2-warmed-mixing` (own PR; human merges).
- Moves to the **real enwik8 corpus** (ADR-0004 carve), bounded scale (see §Scale).
- Origin: the B.1 design fan-out's adversarial critics surfaced — and a verification probe
  **confirmed** — that the context-mixing reference loses to the transformer only from its
  *transductive handicap* (cold start), not its structure. Remove it cheaply (prior warmup, FLOPs
  counted) and the cheap learner **dominates the transformer per FLOP**.

## Verified finding (the reason this task exists)

Same 1200-byte synthetic carve, all FLOPs counted (reproduced in-session, cold order-3 matches the
reference 4.4637 exactly):

| candidate | bpb | total FLOPs |
| --- | ---: | ---: |
| transformer (core) | 4.1992 | 2.311e11 |
| cold context-mix (reference, order-3) | 4.4637 | 1.04e7 |
| **warmed mix, order-2** | **4.1059** | **5.9e7** |
| warmed mix, order-3 | 4.1074 | 7.3e7 |

Warmed mixing **strictly dominates** the transformer (lower bpb AND ~3000–5000× fewer total FLOPs),
and the −0.29 bpb gain over cold is **far above the ~0.01 seed-noise floor**. First non-Pareto-hollow
result in the project. This is both a candidate and the **Tier-0 backbone** the bold layer needs.

## Phase 1 — `warm_mix` (the candidate)

`@register_model("warm_mix") class WarmMix(ContextMixing)` — reuses all the mixing + honest FLOP
machinery; the only new mechanism is the **stateful prior→eval handoff** (harness §5 explicitly
defers this "until a stateful candidate needs it" — warm_mix is that candidate). Zero harness
changes (subclass + register + `models/__init__.py` import).

- **Persistent warmed state** `self._warm: _MixerState` (count tables + mixer weights), created lazily.
- **`train_step((x, y), opt)`** — override: for each row in the batch, fold its bytes **in order**
  (context reset per row) into `self._warm` — reuse `ContextMixing`'s fold + mixer-SGD logic — and
  charge the **honest per-byte step cost** (the budget loop warms on real-enwik8 windows until the
  FLOP ceiling). Return `(warmup_CE, FlopBreakdown)`; loss is logging-only.
- **`flops(seq_len)`** — honest per-row warmup cost (`_steady_step_flops().scale(seq_len)`), so the
  pretrain budget loop terminates correctly and warmup FLOPs are counted in the total.
- **`init_prequential_state()`** — return a **deep copy** of `self._warm` (the warmed tables +
  weights) so eval continues from the warmed statistics. If never warmed (budget 0) → fresh state ⇒
  **bit-identical to the cold reference** (the clean baseline; assert this in a test).
- Everything else (`step`, `_flop_breakdown`, `from_config`) inherited unchanged.
- **No leakage:** warmup folds only prior bytes (carve-disjoint from eval); each eval stream folds
  its bytes into its **own copy**, never the persistent warm state.

### Acceptance (Phase 1)

- Gates green: `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest`.
- `tests/test_warm_mix.py`: warm_mix at budget 0 == cold `context_mixing` (bit-identical predictions
  + identical FLOPs); warmup strictly lowers eval bpb on a tiny fixture; FLOP accounting of warmup ==
  hand-computed per-byte cost × bytes folded; **no-leakage** (eval prediction invariant to future
  bytes; warm state untouched by eval); reproducible seed; registration.
- Real-enwik8 dual-baseline curve (see §Scale, §Experiment): warm_mix **strictly dominates** the
  bare transformer on bpb-vs-total-FLOP, and pushes the cheap frontier below the cold reference. If
  it does not, report honestly.

## Scale (real enwik8, bounded)

Pure-Python context-mixing is ~53 µs/byte (measured on real enwik8 order-3), so the full 95 MB/5 MB
ADR carve is intractable in this runtime. **Bounded-but-real** scale for Phase 1 (real enwik8 bytes,
prior/eval structurally disjoint, ALL FLOPs counted): warmup ~2–4 MB of real prior, eval ~256 k–512 k
bytes (~2–4 min/run). Explicitly **not** the full-5 MB ADR endpoint (that stays GPU/opt-in for the
transformer and slow-pure-Python for the mixer); the bounded run is a faithful real-text result, and
the relative per-FLOP ordering is what the metric asks for. Cache: `prepare_enwik8` (already
downloaded to `data/cache/`).

## Experiment (dual baseline, same eval stream)

A committed runner producing, on **one identical real-enwik8 carve**:
1. `warm_mix` swept over a few pretrain budgets (0 = cold ref point, plus 2–3 warmup sizes),
2. the **bare transformer** trained on the same real prior at a comparable total-FLOP point,
3. the **cold context-mixing reference** (warm_mix @ budget 0).
Plot bpb-vs-**total**-FLOP; a win = warm_mix strictly below/left of the transformer and extending the
reference frontier. Regenerate the leaderboard into a dedicated `runs/b2/`.

## Phase 1 outcome (landed) + the reframed bar

`warm_mix` on real enwik8 (4 MB slice, 32 k eval) **strictly dominates the transformer per FLOP**:
2.7700 bpb @ 1.03e10 vs the transformer's 5.5453 @ 9.71e11 (~94× fewer FLOPs), warmup monotonic
(cold 3.21 → warm@1e10 2.77). So the bar for the bold layer is **no longer the transformer — it is
fixed-order `warm_mix` itself**.

Order-curve probe (warmed @1e9, real enwik8): bpb falls with depth to ~order-6 then plateaus —
order-2 3.248, order-3 2.881, order-4 2.710, order-5 2.667, **order-6 2.655**, order-8 2.664. Deep
context pays on real text (it *hurt* on synthetic). But the mixer pays **all K orders on every
byte**, while the high orders earn bits only on the minority with deep recurring context. That
inefficiency is the (iv) lever — and it **fuses A and C**.

## Phase 2 — `gated_mix`: gated order escalation on the warmed backbone (the A∩C fusion)

`@register_model("gated_mix") class GatedMix(WarmMix)`. Holds orders `0..K_max` (warmed), but per
byte evaluates them **cheapest-first and stops escalating** once the running mix is confident
enough (entropy / `1 − max p` below a threshold) or no higher *active* context exists — charging
FLOPs for **only the orders actually evaluated**. This is simultaneously **A** (the expensive
compute = high-order models, fired only on high-surprise bytes) and **C** (an order is "grown into"
the mix only where its residual-entropy contribution justifies the spend).

- **(iv) dynamic:** mixing one more order costs a fixed ~`O(V)` FLOPs/byte and buys `ΔH_d` bits (the
  conditional-entropy drop at depth `d`); escalate only while `ΔH_d` is likely > 0 (proxied by base
  surprise + context activity). Mean evaluated depth ≪ `K_max` on easy bytes ⇒ near-order-`K_max`
  bpb at a fraction of fixed-order FLOPs.
- **Bar to beat:** the fixed-order `warm_mix` bpb-vs-FLOP curve above. A win = `gated_mix` sits
  strictly below/left of it (e.g. order-6 bpb ≈ 2.655 at ~order-2/3 FLOP cost). "Same bpb at fewer
  FLOPs" is a genuine per-FLOP win here.
- **Gate is pre-reveal** (uses the partial-mix prediction's entropy, never the true byte); the gate
  arithmetic and every evaluated order's stretch/mix/Laplace/update are charged via `smolml.flops`;
  the per-byte charge is dynamic (orders evaluated), summed by the harness — no compute hides.
- **Harness:** subclass `WarmMix`, override `step` (escalation loop + per-order charge) and
  `_flop_breakdown` for the evaluated-orders cost; inherit warmup + handoff. Zero harness changes.
- **Honest risk (critics' warning):** the mixer's soft weights already down-weight dead orders, so
  the gain may be mostly **FLOP savings at ~equal bpb** rather than lower bpb. That is still a
  per-FLOP win; report it as such, and report the realized mean evaluated depth.

**A-as-neural-cascade is deprioritized:** on real enwik8 the transformer is so FLOP-inefficient that
`warm_mix` dominates it, so escalating to a *neural* tier loses per-FLOP — the high-order context
models are the better "expensive tier." (Revisit only if a cheap learned tier captures structure the
mixer cannot, e.g. long-range/positional.)

### Acceptance (Phase 2)

Gates green; `tests/test_gated_mix.py` (gate pre-reveal/no-leakage; per-byte charge == evaluated
orders, hand-checked; a degenerate threshold ⇒ identical to fixed-order `warm_mix`; reproducible);
real-enwik8 run plotting `gated_mix` against the fixed-order `warm_mix` curve + reporting mean
evaluated depth and the per-FLOP verdict (honest, even if it's only FLOP savings).
