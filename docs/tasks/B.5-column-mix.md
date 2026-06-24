# Task B.5 — routed sheet of delta columns (context-conditional selection over context mixing)

- Status: SPEC — design approved (brainstorm) and revised after a 4-lens adversarial spec review
  (FLOP honesty / seam fidelity / Pareto-hollow skepticism / test coverage). Phase 1a of the
  "local-learning cortical-column" candidate: a **flat sheet of routed delta-rule columns**, mixed
  online. Phase 1b (lateral predictive columns) is the back-pocket depth move, gated on 1a earning
  its place. Phase 2 (embodied snake rung) is a separate brainstorm.
- The next Space-B **learning-rule** candidate under the Source-(iv) filter (ADR 0003): a brain-style
  *sheet of local learners* whose only structural addition over the bar is **context-conditional
  selection** of which local predictor fires — many tiny delta predictors, exactly one active per
  byte, routed by a cheap online contextual-bandit gate. 100% local learning; no global backward
  pass (inherited from `delta_mix`).
- **Bar to beat** (real 5 MB enwik8 ADR eval, all FLOPs counted): `delta_mix` order-6 warmfull
  **1.8485 bpb @ 1.322e12 total FLOPs** (the current leader). `column_mix` must beat **`delta_mix`
  itself**, not the transformer.

## The bet (Source-(iv)) — read this honestly

`delta_mix` is already a **sheet of one**: a single online delta-rule fast-weight predictor
`W ∈ ℝ^(V×d)` over a sparse signed hashed bag of byte n-grams, mixed with the count ladder. Its
prediction `z = W @ φ` is **linear in `φ`**.

**What `C` columns do NOT buy.** Naively "C columns = C× capacity at constant FLOPs" is *false*. A
sparse read is `O(sV)` regardless of `d` (the bar's own feasibility crux), so `delta_mix` can simply
set `delta_dim = C·d` for **zero** extra FLOPs and get the *same* `V×(C·d)` capacity in **one shared
`W`** — updated by **every** byte (no data split), with fewer collisions and full convergence. So a
sheet adds no FLOP-free capacity the bar cannot buy more cheaply. **The matched-capacity control in
the kill-test exists precisely to deny this confound.**

**What `C` columns DO buy (the only real lever).** A single linear `W` computes one fixed linear
function of `φ`: the same `φ` always maps to the same logits. A routed sheet computes
`z = W_{route(ctx)} @ φ` — it **switches weight matrices on a context bucket**, so the *same* n-gram
features can map to *different* logits depending on a separate route context. That is a
**multiplicative context×feature interaction** a linear-in-`φ` predictor provably cannot represent
at any `d`. The (iv) bet: when the next byte depends on an *interaction* between a route context and
the n-gram features (not an additive function of either alone), routed columns extract that
loss-reduction at ~the bar's per-byte FLOPs (one delta read/write + an `O(C)` router), where one
global `W` is structurally blind to it.

**Why this is high-risk, stated up front (do not paper over).** This is the "refine an already-good
cheap learner" pattern that A.1 / B.1 / `gated_mix` all show **usually loses** per-FLOP. Three
specific reasons it may be Pareto-hollow here, pre-registered as the expected failure modes:
1. **Redundancy.** The order-0..`max_order` count tables already condition on recent context, and the
   route context is a suffix that the order-≥3 delta features already hash. So the route's *marginal*
   interaction beyond (counts ∪ `φ`) may be tiny. Mitigation: route on context **outside** the count
   ladder and `delta_orders` — but that worsens (2).
2. **Data starvation.** At matched total FLOPs both models see ~`N` bytes, but each `W_c` gets only
   ~`N/C` delta updates, so every column is ~`C`× less converged. The interaction's loss-reduction
   must beat that penalty.
3. **Gate collapse.** A learning gate with no load balancing can route all buckets to one column —
   which is exactly `delta_mix`, after paying `C`× RAM and the router FLOPs. Detected by the
   per-column-load diagnostic; **collapse → "≡ bar" is an accepted (logged) kill outcome**, not a bug
   to patch with a balance knob in 1a.

Three things keep it auditable rather than romantic: the router is `O(C)` (≪ 0.2% of the bar's ~24k
FLOPs/byte at `C ≤ 16`); **`C = 1` is bit-identical to `delta_mix`** and **gate-off** (frozen at the
hash-prior route) is an exact static-routing baseline; and the kill-test (with the matched-capacity
control) decides it before any full-corpus run. **Failures are data.**

## Mechanism — `column_mix`

`@register_model("column_mix") class ColumnMix(DeltaMix)`. Inherits the warm prior→eval handoff, the
hashed count store, the logistic mixing, the online mixer-SGD, the delta feature map `φ` / delta
update, and **all** count- and delta-side FLOP accounting. Adds the column array + the router.

The column path is active **only** when `n_columns > 1 and delta_orders`; otherwise every override
delegates to `super()` (the degenerate identities below), so the added state stays `None`.

**State** (extends `_MixerState`; warm-started then deep-copied per stream, like `W`/`tables`):
- **Per-column fast weights** `Wcols ∈ ℝ^(C×V×d)`, init `0` (an unwritten column ⇒ `z = 0` ⇒ uniform
  abstention, like the bar's unwritten `W`). At `C = 1` the inherited 2-D `W` is used (delegate to
  super) — never `Wcols` — so the identity is exact.
- **Gate value table** `gate ∈ ℝ^(B×C)` (`B = route_buckets`): `gate[b, c]` is the running estimate
  of column `c`'s **reward** (= `−bits`, higher is better) when chosen at bucket `b`. Init
  `gate[b, b mod C] = 0.0` (the hash-prior arm, optimistic), all other entries `= gate_init_other`
  (default `−10.0`, below the worst possible reward `−8`), so `argmax gate[b,:] = b mod C` cold and
  an unvisited sibling never wins the argmax without evidence (it is reached only via ε-exploration).
- **Per-stream RNG** `rng`: a fresh `np.random.default_rng(config.seed)` built **per stream** (never
  copied) — drives ε-exploration deterministically and isolated across streams.
- Deferred scratch: `last_bucket` (int), `last_route` (the chosen column id `c*`), and the inherited
  `last_phi` / `last_p_delta` (`last_p_delta` = the chosen column's `softmax(z_{c*})`, reused for its
  delta update **and** the gate reward).

**Router (picks the column BEFORE computing it — the sparse win survives).**
- Bucket `b = _route_slot(window[-route_order:])`: `int.from_bytes(bytes(window[-route_order:]),
  "little")`, one Fibonacci multiply by `_KNUTH`, mask, and a shift to the **top `log2 B` bits**
  (`B` a power of two, so no `mod`). A short or empty window (`pos < route_order`) hashes the
  available bytes — deterministic, and `b = 0` for the empty prefix at `pos 0`.
- Route: with prob `route_epsilon` draw a uniform column from `rng`; else `c* = argmax gate[b,:]`.

**Prediction.** Build `φ` once (shared key). `z = W_{c*} @ φ` (touches `s` columns of one matrix).
`z` is fed as the **single** `(K+1)`-th raw-logit row into the existing mixer — **mixer width and
the whole count-side path are identical to `delta_mix`** (the chosen column simply *is* the bar's one
delta stream). Mix → softmax → `next_logits`.

**Updates — two local rules, both deferred to the next byte and BOTH gated on `did_update`
(`last_probs is not None`), so a warm-row boundary cleanly skips them (no cross-row leakage; no
`train_step` override). No global backward pass:**
1. **Column delta** (on the previously chosen column, previous key): `W_{c*}[:, j] −= η·φ_prev[j]·
   (softmax(z_{c*}^prev) − onehot(revealed))` over `j ∈ supp(φ_prev)` — the bar's **exact**
   `_apply_delta_update`, applied to one column. Each column learns its routed sub-stream's residual.
2. **Gate** (per-arm contextual bandit; no differentiation through the discrete route). Chosen column
   `c*`, bucket `b_prev`, observed reward `r = log2 softmax(z_{c*}^prev)[revealed]` (`= −bits`, ≤ 0):
   ```
   gate[b_prev, c*] += gate_lr · (r − gate[b_prev, c*])     # EMA of the chosen arm toward its reward
   ```
   Only the chosen arm updates (the credit-assignment cost of staying sparse). `route_epsilon`
   probes siblings so a genuinely-better column can overtake the prior; with `gate_lr = 0` **and**
   `route_epsilon = 0` the gate is frozen at the hash-prior route (the gate-off baseline). There is
   **no** separate baseline EMA — `gate[b,c]` *is* the per-arm value, so the advantage is never
   mean-zero (the reviewed self-referential-baseline failure is removed by construction).

### FLOP honesty (code-reuse identity; every op via `smolml.flops`; `d`/`B` are memory only)

`V = 256`, `s = |active delta_orders|`, `C = n_columns`. The chosen column **is** the bar's single
delta stream, so the count- and delta-side charge is the parent's, reused verbatim:

```
step FLOPs = super()._delta_flop_breakdown(did_update, n_fold, n_active, n_laplace, nd, nd_prev)
             + _route_increment(C, did_update)
```

`_route_increment` carries **only** the router and the gate update:
- *Forward (always):* route hash (`3` ops) + `gather(1)` gate row of `C` + `pointwise(C)` argmax over
  the row ⇒ `gather(1) + pointwise(3 + C)`.
- *Backward (only when `did_update`):* the bandit update — reward `r = log2 p[revealed]` (index + log
  ≈ `2`) + `(r − gate)` (`1`) + `·gate_lr` (`1`) + `+=` (`1`) ⇒ `pointwise(5)` exactly (no `C` term:
  the m=1 bandit touches one cell; the argmax was already charged in forward).
- *ε-exploration:* the PRNG draw + branch is treated as **non-arithmetic and omitted**, matching the
  bar's convention (it charges no RNG). Documented here so the omission is a decision, not a gap.

**Correctness anchor (exact by construction):** `_route_increment` contains no per-column-multiplicity
term (exactly one column is read, charged inside the parent breakdown), so at `C > 1` the only net
charge over `delta_mix` is `gather(1) + pointwise(3 + C)` forward and `pointwise(5)` backward. At
`C = 1` (and `delta_orders = ()`) **every** override — including `_steady_step_flops` / `flops` /
`decode_step_flops` — delegates to `super()` behind the `not (n_columns > 1 and delta_orders)` guard,
so the charge is **bit-identical** to `delta_mix` (zero router). Cross-vendor codex review targets the
router gather/argmax + the bandit-update scalar count specifically (a FLOP undercharge has slipped
through on *every* prior candidate).

### Memory

`Wcols = C·V·d·8 B`. At `C = 8`, `d = 2**18`, ≈ `2.1 GB` per stream (eval deep-copies one more) —
within the bar's 3–5 GiB envelope for modest `C`; `d` can be reduced per-column to hold memory while
sweeping `C` (memory is **not** the metric). The gate table `B·C·8 B` is negligible.

## Config

`ColumnMixConfig(DeltaMixConfig)` adds:
- `n_columns: int = 1` — `C ≥ 1`. `C = 1` ⇒ degenerate identity to `delta_mix`.
- `route_buckets: int = 1 << 12` — `B`, power of two ≥ 1 (Fibonacci top-bits route).
- `route_order: int = 2` — `r`, recent bytes hashed for the bucket; `1 ≤ r ≤ 8`.
- `gate_lr: float = 0.1` — EMA step toward the observed reward; `≥ 0`; `0` ⇒ frozen gate values.
- `route_epsilon: float = 0.05` — ε-greedy exploration prob; `0 ≤ ε < 1`. `gate_lr = 0` **and**
  `ε = 0` ⇒ the static fixed-hash route (gate-off baseline).
- `gate_init_other: float = -10.0` — init reward for non-prior arms (below the worst reward `−8`).
- `seed: int = 0` — seeds the per-stream route RNG (reproducibility).

`route_top_m` is **fixed at 1** in this spec — exactly one column active per byte. `m > 1` soft
routing (a gate-weighted blend, its extra per-column reads, the blend cost, and the soft-mixture gate
gradient) is deferred to a future sweep with its own FLOP charge + hand-formula test; it is **not**
admitted here.

Reuse every `DeltaMix`/`HashedMix`/`ContextMixing` knob. Validate in `__post_init__`: `route_buckets`
a power of two; `n_columns ≥ 1`; `1 ≤ route_order ≤ 8`; `gate_lr ≥ 0`; `0 ≤ route_epsilon < 1`.

**Degenerate identities (assert in tests):**
- `n_columns == 1` ⇒ **bit-identical** to `delta_mix` — same predictions, same per-step *and*
  analytic (`flops`/`decode_step_flops`) `FlopBreakdown`. Every override is behind the column-path
  guard, so it delegates to `super()`.
- `delta_orders == ()` ⇒ no key to route ⇒ delegates to `super()` (⇒ `hashed_mix`), for any `C`.
- `gate_lr == 0 and route_epsilon == 0` ⇒ the route is the deterministic hash partition (`b mod C`)
  every byte — a reproducible static-routing baseline (not an identity to `delta_mix` unless `C = 1`).

## Seam (zero harness changes)

- **`ColumnMixConfig(DeltaMixConfig)`** — the knobs above; `route_top_m` not a knob (fixed 1).
- **`__init__`** — store config; `Wcols`/`gate` are per-stream online state, **not** `nn.Parameter`
  ⇒ `num_params() == 0`, no AdamW. `_window_cap = max(max_order, max(delta_orders, default=0))` when
  the column path is off (mirror `delta_mix` exactly, preserving `context_window`); include
  `route_order` in the cap **only** when the column path is on.
- **`_MixerState`** — extend with `Wcols`, `gate`, `rng`, `last_bucket`, `last_route` (`None` for
  non-column models / the `C = 1` path).
- **`_fresh_cache` / `_ensure_warm`** — when the column path is on, allocate `Wcols` (zeroed), `gate`
  (hash-prior init), and a fresh `default_rng(seed)`; else delegate to `super()`.
- **`init_prequential_state`** — deep-copy the warm state **including `Wcols.copy()`, `gate.copy()`**;
  build a **fresh** `default_rng(seed)` per stream (never copied — deterministic, isolated). Leak-free.
- **`step`** (override; reuse `mix_logits`/`softmax`/`mixer_gradient`, inherited `_build_phi` /
  `_apply_delta_update` / count seams): delegate to `super().step` when the column path is off; else
  (1) deferred mixer update (`did_update`); (2) deferred column-delta update on `last_route`/
  `last_phi` (`did_update`); (3) deferred gate update on `last_bucket`/`last_route` (`did_update`);
  (4) fold counts (parent loop); (5) route → read the active column → `(K+1)`-th row → mix → softmax;
  (6) stash `last_*`; (7) return `super()._delta_flop_breakdown(...) + _route_increment(...)`.
- **`_steady_step_flops` / `flops`** — **guarded**: `if not (n_columns > 1 and delta_orders): return
  super()._steady_step_flops()`; else parent value `+ _route_increment(C, did_update=True)` at full
  support. `flops` / `decode_step_flops` inherit the guard (this is what keeps the `C = 1` analytic
  identity exact *and* the pretrain budget look-ahead at parity).
- **`train_step`** — inherited **unchanged** from `WarmMix` (folds prior windows through `step`; the
  `did_update` gating makes the row-boundary reset clean, so no override is needed).
- **`from_config`** — keep only `ColumnMixConfig` fields; coerce `delta_orders` list → tuple.
- **Register** `ColumnMix`, `ColumnMixConfig` in `smolml/models/__init__.py`.

## Acceptance

- **Gates green:** `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest`.
- **`tests/test_column_mix.py`** (mirror `tests/test_delta_mix.py`):
  - registration + buildable + `num_params() == 0`; `from_config` admits fields / ignores
    transformer keys / coerces list `delta_orders`; config validation (each guard above).
  - **`n_columns=1` ⇒ bit-identical predictions AND per-step `FlopBreakdown` to `delta_mix`**
    (warmed, every eval step), **AND** `flops(seq_len)` / `decode_step_flops` bit-identical (the
    analytic-path identity the `_steady_step_flops` guard protects) — the headline identity.
  - **`delta_orders=()` ⇒ bit-identical to `hashed_mix`** for `C > 1` (inherited path).
  - route map: `_route_slot` deterministic, in `[0, B)`, defined for windows shorter than
    `route_order` (incl. empty ⇒ 0); `gate_lr=0, route_epsilon=0` ⇒ chosen column == `bucket mod C`
    every step (fixed-hash route), reproducible.
  - prequential smoke (finite, `0 < bpb ≤ 8`); **no leakage** (perturbing a future byte leaves all
    past predictions unchanged); **warmup no-leakage** (the deferred column/gate updates are skipped
    at warm-row boundaries — assert a warmed `Wcols`/`gate` is unchanged by a boundary).
  - **FLOP charge matches the code** — `step`'s breakdown == `super()._delta_flop_breakdown(...)` +
    the analytic `_route_increment`, with a hand-formula pin: forward `gather(1) + pointwise(3 + C)`,
    backward (did_update) `pointwise(5)`.
  - **selection existence proof (the (iv) claim, isolated from capacity)** — a deterministic
    **interaction source**: next byte `= f(route_ctx, ngram)` where the *same* ngram maps to
    *different* bytes under different route buckets (an interaction a single linear-in-`φ` `W` cannot
    represent at any `delta_dim`). Assert `C > 1` beats `C = 1` next-byte bpb by a numeric margin,
    **and** that `C = 1` plateaus *even with `delta_dim` raised to `C·d`* (the matched-capacity
    control — proves the win is selection, not table size). Fixed interleaving + seed.
  - **learned-routing existence proof** — on a source where the hash prior misassigns buckets but a
    better column assignment exists, `gate-on` (`gate_lr>0, route_epsilon>0`) reaches lower bpb than
    `gate-off` (`gate_lr=0, route_epsilon=0`). If it cannot be shown cheaply, document why and lean
    on the kill-test.
  - **dead-column abstention** — an unrouted column keeps its zeroed `Wcols` slice ⇒ uniform
    contribution (no spurious signal).
  - deep-copy isolation — an eval stream's `Wcols`/`gate` write never touches the warm state **or**
    another stream (assert both `Wcols` and `gate` independence); reproducible under a fixed seed.
- **Kill-test experiment (CI-fast, a few-MB real-enwik8 slice — `smolml/experiments/
  column_mix_enwik8.py`, mirroring `delta_mix_enwik8.py`). MUST run and be reported BEFORE any
  full-corpus claim.** Matched **total** FLOPs:
  - (a) `delta_mix` (`C = 1`, the bar) warmed to budget `P`;
  - (b) `column_mix` (`C > 1`, **learned** gate `gate_lr>0, ε>0`) at the same total FLOPs;
  - (c) `column_mix` (`C > 1`, **gate-off** fixed-hash route) at matched FLOPs;
  - (d) **matched-capacity control:** `delta_mix` with `delta_dim = C · d_col` (the sheet's total
    width in one shared `W`) at matched FLOPs — denies the "C columns are just a bigger table"
    confound.
  **Kill unless `max((b), (c))` beats BOTH (a) and (d)** in bpb (routing earns its keep as
  *selection*, not capacity). Report (c) vs (a)/(d) = the selection-lever verdict, (b) vs (c) = the
  learned-gate verdict. Diagnostics, logged: per-column load (collapse → "≡ bar", an accepted
  outcome), per-column conditional bpb **against the bar's bpb on the same routed bytes** (so
  under-convergence is visible), the delta-row mixer weight, and gate drift off the hash prior.
- **Full ADR carve** (extend `smolml/experiments/full_corpus.py` with a `column_mix` entrant on the
  full 95 MB prior / 5 MB eval; detached, multi-hour) — run **only if the kill-test passes**. Plot
  bpb-vs-total-FLOP over `C ∈ {2,4,8,16}` against the `1.8485 @ 1.322e12` bar; report peak RAM and the
  gate diagnostics. Honest either way.

## Out of scope (this spec)

- **Phase 1b — lateral predictive columns** (the brain-like depth move): columns predict neighbors'
  `z_c`; each contributes only its *residual* (lateral inhibition / sparse coding). Its own spec
  **only if the flat routed sheet beats the bar** — honors "flat first, then depth" and B.1's
  Pareto-hollow warning.
- **Phase 2 — embodied snake rung** (sensorimotor / scalar-reward control): its own brainstorm after
  phase 1 lands.
- **`route_top_m > 1` soft routing** — deferred (needs its own FLOP charge for the blend + per-column
  reads + soft-mixture gate gradient, and a hand-formula test). Fixed at 1 here.
- **A load-balancing term** — not in 1a; gate collapse → "≡ bar" is an accepted, logged kill outcome.
- **A learned dense/projected route encoder** — the cheap `(B×C)` gate table over a fixed hash bucket
  is the whole point (`O(C)` routing); a learned encoder would be a different, heavier candidate.
