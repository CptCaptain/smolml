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

## Phase 2 — bold layer on the warmed backbone (staged; design after Phase 1 lands)

With warm_mix as the verified Tier-0 backbone and the real-corpus bar established, build the bold
(iv) layer the user asked for — **both** directions, designed against the *real* bar:
- **A — surprise cascade:** warmed mixer carries the predictable mass; an expensive *learned* tier
  fires only on residual-surprise bytes and learns from them. The dominant cost stays the cheap
  warmed tier; expensive FLOPs concentrate on the high-bit minority (the structural fix for B.1).
- **C — entropy-gated growth:** grow context orders / experts on the warmed mixer only where
  residual conditional-entropy stays high.
Real enwik8 is where these levers can actually bite (rich per-byte difficulty variation), unlike the
order-0 synthetic clone that starved B.1. Detailed design + acceptance to be written when Phase 1
results are in (do not pre-build).
