# Task B.5 — routed sheet of delta columns (sparse conditional capacity over context mixing)

- Status: SPEC — design approved (brainstorm). Implementation pending. Phase 1a of the
  "local-learning cortical-column" candidate: a **flat sheet of routed delta-rule columns**, mixed
  online. Phase 1b (lateral predictive columns) is the back-pocket depth move, gated on 1a earning
  its place. Phase 2 (embodied snake rung) is a separate brainstorm.
- The next Space-B **learning-rule** candidate under the Source-(iv) filter (ADR 0003): a brain-style
  *sheet of local learners* whose only structural addition over the bar is **sparse conditional
  capacity** — many tiny delta predictors, one (or top-`m`) active per byte, routed by a cheap
  learned gate. 100% local learning; no global backward pass (inherited from `delta_mix`).
- **Bar to beat** (real 5 MB enwik8 ADR eval, all FLOPs counted): `delta_mix` order-6 warmfull
  **1.8485 bpb @ 1.322e12 total FLOPs** (the current leader; strictly dominates the 2.0157 @ 1.478e12
  hashed bar). `column_mix` must beat **`delta_mix` itself**, not the transformer.

## The bet (Source-(iv))

`delta_mix` is already a **sheet of one**: a single online error-correcting fast-weight predictor
`W ∈ ℝ^(V×d)` over a sparse signed hashed bag of byte n-grams, mixed with the count ladder. That one
`W` must fit the residual structure of the **entire** byte stream with a single weight matrix.

`column_mix` replaces the one `W` with **`C` columns** `W_0..W_{C-1}`, each the *same* delta
predictor on the *same* key `φ`, and adds a **router** that activates only the top-`m` columns per
byte (default `m = 1`). The (iv) bet: a route partitions the stream into sub-streams, and each
column `W_c` then fits only its routed sub-distribution — **more homogeneous, lower-entropy, sharper
to predict** — than one global `W` fitting the mixture. Capacity scales with `C`; **active FLOPs stay
~constant** (one delta read/write + an `O(C)` router), so more loss-reduction per FLOP *iff routing
specializes*.

This is **sparse conditional capacity** (MoE-of-deltas) — the cleanest per-FLOP lever distinct from
the bar, and the most literal reading of "`delta_mix` reshaped as a grid of local learners."

**Why this is a real (iv) claim, stated honestly.** This is *exactly* the "refine an already-good
cheap learner" pattern that A.1 / B.1 / `gated_mix` all show **usually fails** — the routing/gating
overhead tends to exceed the savings (`gated_mix` was Pareto-hollow for precisely this reason). Three
things make it auditable rather than romantic:
1. The router adds only `O(C)` (a few·`C` FLOPs) on top of the bar's ~24k FLOPs/byte, so it
   **cannot** dominate the cost — the comparison stays at essentially matched FLOP/byte.
2. **`C = 1` is bit-identical to `delta_mix`** (degenerate identity) and **gate-off** (frozen hash
   route) is an exact static-routing baseline — two clean ablation anchors isolate the lever.
3. The kill-test decides it cheaply before any full-corpus run; the online mixer drives a useless
   delta row's weight → 0 (bounded downside). **Failures are data.** "Cortical columns → emergence"
   is not the claim; *one specific routed delta unit with a concrete per-FLOP argument* is.

## Mechanism — `column_mix`

`@register_model("column_mix") class ColumnMix(DeltaMix)`. Inherits the warm prior→eval handoff, the
hashed count store, the logistic mixing, the online mixer-SGD, the delta feature map `φ` / delta
update, and **all** count- and delta-side FLOP accounting. Adds the column array + the router.

**State** (extends `_MixerState`; warm-started then deep-copied per stream, like `W`/`tables`).
Active only when the column path is on (`n_columns > 1 and delta_orders`); otherwise these stay
`None` and every override delegates to `super()`:
- **Per-column fast weights** `Wcols ∈ ℝ^(C×V×d)`, init `0` (an unwritten column ⇒ `z = 0` ⇒ uniform
  abstention, like the bar's unwritten `W`). At `C = 1` the inherited 2-D `W` is used (delegate to
  super) — never `Wcols` — so the identity is exact.
- **Gate table** `gate ∈ ℝ^(B×C)` (`B = route_buckets`), initialised so `argmax gate[b,:] = b mod C`
  (the deterministic hash prior): `gate[b, b mod C] = +g0`, else `0`. With `gate_lr = 0` (frozen) the
  route is therefore the pure fixed-hash partition.
- **Per-bucket baseline** `gate_baseline ∈ ℝ^B` — EMA of the chosen column's per-byte bits at each
  bucket (the bandit baseline).
- Deferred-route scratch: `last_bucket` (int), `last_route` (the `m` chosen column ids), and the
  inherited `last_phi` / `last_p_delta` — for `m = 1`, `last_p_delta` is the chosen column's
  `softmax(z_c)`, reused for both its delta update and the gate advantage.

**Router (picks columns BEFORE computing them — the sparse-capacity win survives).**
- Bucket `b = route_slot(window[-route_order:]) mod B` — one Fibonacci hash (top `log2 B` bits) of
  the recent `route_order` bytes. `O(1)`; a salt-free constant so warmed runs reproduce.
- Gate row `g = gate[b, :]` (one gather of `C` values). Active set `S = top-m(g)`; `m = 1` ⇒ argmax.
- Exploration: with prob `route_epsilon` (default `0`, swept) route to a uniformly random column
  (seeded RNG in state, so reproducible). Off by default ⇒ deterministic.

**Prediction.** Build `φ` once (shared key). For each `c ∈ S`: `z_c = W_c @ φ` (touches `s` columns
of `W_c`). Combine into one row: `m = 1` ⇒ `z_delta = z_{c*}`; `m > 1` ⇒ gate-weighted blend
`z_delta = Σ_{c∈S} softmax(g_S)[c]·z_c`. `z_delta` is fed as the **single** `(K+1)`-th raw-logit row
into the existing mixer — **mixer width and the whole count-side path are identical to `delta_mix`**.
Mix → softmax → `next_logits`.

**Updates — two local rules, both deferred to the next byte (pending-prediction pattern; no leakage,
no global backward):**
1. **Column delta** (per active column, on the previous key): for each `c ∈ S_prev`,
   `W_c[:, j] −= η·φ_prev[j]·(softmax(z_c^prev) − onehot(revealed))` over `j ∈ supp(φ_prev)` — the
   bar's exact LMS rule, applied only to the chosen column(s). Each column learns its routed
   sub-stream's residual.
2. **Gate** (contextual bandit; no differentiation through the discrete route). Chosen column `c*`,
   bucket `b_prev`, chosen-column quality `q = −log2 softmax(z_{c*}^prev)[revealed]` (bits):
   ```
   A          = gate_baseline[b_prev] − q          # chosen beat its bucket's recent avg?
   gate[b_prev, c*] += gate_lr · A                 # reinforce / demote the chosen column
   gate_baseline[b_prev] = (1−ρ)·gate_baseline[b_prev] + ρ·q
   ```
   `O(m)`. For `m > 1` soft routing, the standard MoE soft-mixture gradient on the active rows
   replaces the bandit step (a sweep; `m = 1` + bandit is the headline).

**Honest limitation (stated up front).** Only chosen columns receive a signal; an un-sampled column
never updates. Without `route_epsilon > 0` the gate **refines per-bucket trust but inherits the
hash partition's column assignment** — it can demote a degrading column but cannot discover that a
*different* column would serve a bucket better. The **gate-on vs gate-off** ablation measures whether
learned refinement helps at all; `route_epsilon` is the (swept) mechanism that lets the gate
reassign. This is the credit-assignment cost of staying sparse, and it is the candidate's main risk.

### FLOP honesty (every op charged via `smolml.flops`; `d` is memory/collisions only, never FLOPs)

`V = 256`, `s = |active delta_orders|`, `C = n_columns`, `m = route_top_m`. The router and the
per-column reads/writes are charged on top of the bar's breakdown. **`d` (per-column width) costs RAM
and collisions only** — sparse access touches `s` columns regardless of `d`.

`_route_increment` per byte, on top of `DeltaMix._delta_flop_breakdown`:
- *Router (always, forward):* route hash (`3` ops) + `gather(1)` gate row of `C` + `pointwise(C)`
  top-`m`/softmax over the row ⇒ `gather(1) + pointwise(3 + C)`.
- *Extra active columns:* the bar charges **one** delta stream; `column_mix` charges the delta
  read+write+softmax for **each** of the `m` active columns (`m·(2sV read + 5V softmax)` forward,
  `m·(2sV write + V scale + 1)` backward when a prediction was pending), but the single combined
  mix-row (`2V`), the `softmax`/mixer-gradient slot (`2V`) and the weight step (`2`) are charged
  **once** (one row into the mixer). **At `m = 1` this is bit-identical to the bar's
  `_delta_increment`**, so the only net charge over `delta_mix` is the router.
- *Gate update (backward, when pending):* `pointwise(~2 + C)` for the advantage, the chosen-column
  bump, and the baseline EMA.

**Correctness anchor:** at `m = 1`, `column_mix.step` FLOPs == `delta_mix._delta_flop_breakdown`
(one active column) **+ `_route_increment`**, exactly. At `C = 1` the path delegates to `super()`
so the charge is bit-identical to `delta_mix` (zero router). Cross-vendor codex review must target
the router gather/softmax + the per-column read/write multiplicity specifically (reflex: a FLOP
undercharge has slipped through on *every* prior candidate).

### Memory

`Wcols = C·V·d·8 B`. At `C = 8`, `d = 2**18`, that is `8 × 268 MB ≈ 2.1 GB` per stream (eval
deep-copies one more) — within the bar's 3–5 GiB envelope for modest `C`; `d` can be reduced
per-column to hold memory while sweeping `C` (memory is **not** the metric — only FLOPs are). The
gate table `B·C·8 B` is negligible (`2**12 × 8 × 8 B = 256 KB`).

## Config

`ColumnMixConfig(DeltaMixConfig)` adds:
- `n_columns: int = 1` — `C ≥ 1`. `C = 1` ⇒ degenerate identity to `delta_mix`.
- `route_buckets: int = 1 << 12` — `B`, power of two ≥ 1 (Fibonacci top-bits route).
- `route_order: int = 2` — `r`, recent bytes hashed for the bucket; `1 ≤ r ≤ 8`.
- `route_top_m: int = 1` — `m`, active columns per byte; `1 ≤ m ≤ C`.
- `gate_lr: float = 0.1` — `λ ≥ 0`; `0` ⇒ frozen gate = fixed-hash route (the gate-off baseline).
- `gate_baseline_decay: float = 0.02` — `ρ ∈ (0, 1]`.
- `route_epsilon: float = 0.0` — `ε ∈ [0, 1)`, exploration prob (seeded; default off).

Reuse every `DeltaMix`/`HashedMix`/`ContextMixing` knob. Validate in `__post_init__`
(`route_buckets` a power of two; `n_columns ≥ 1`; `1 ≤ route_top_m ≤ n_columns`; `1 ≤ route_order ≤
8`; `gate_lr ≥ 0`; `0 < gate_baseline_decay ≤ 1`; `0 ≤ route_epsilon < 1`).

**Degenerate identities (assert in tests):**
- `n_columns == 1` ⇒ **bit-identical** to `delta_mix` — same predictions *and* `FlopBreakdown`
  (mirrors the `delta_orders=()` identity). The column path delegates to `super()`.
- `delta_orders == ()` ⇒ no key to route ⇒ delegates to `super()` (⇒ `hashed_mix`), for any `C`.
- `gate_lr == 0` ⇒ the route is the deterministic hash partition (`b mod C`) every byte — a
  reproducible static-routing baseline (not an identity to `delta_mix` unless `C = 1`).

## Seam (zero harness changes)

- **`ColumnMixConfig(DeltaMixConfig)`** — the seven knobs above.
- **`__init__`** — store config; `Wcols`/`gate` are per-stream online state, **not** `nn.Parameter`
  ⇒ `num_params() == 0`, no AdamW. Set `_window_cap = max(max_order, max(delta_orders, default=0),
  route_order)` so the window holds the route context too.
- **`_MixerState`** — extend with `Wcols`, `gate`, `gate_baseline`, `last_bucket`, `last_route`
  (`None` for non-column models / the `C = 1` path).
- **`_fresh_cache` / `_ensure_warm`** — allocate `Wcols` (zeroed), `gate` (hash-prior init),
  `gate_baseline` (zeroed) when the column path is on; else delegate to `super()`.
- **`init_prequential_state`** — deep-copy the warm state **including `Wcols.copy()`, `gate.copy()`,
  `gate_baseline.copy()`** (leak-free; each eval stream mutates its own copies).
- **`step`** (override; reuse `mix_logits`/`softmax`/`mixer_gradient`, inherited `_build_phi` /
  `_apply_delta_update` / count seams): delegate to `super().step` when the column path is off;
  else (1) deferred mixer update; (2) deferred per-column delta update on `last_route`/`last_phi`;
  (3) deferred gate update on `last_bucket`; (4) fold counts (parent loop); (5) route → read active
  columns → combine `z_delta` → `(K+1)`-th row → mix → softmax; (6) stash `last_*`; (7) return the
  parent breakdown **+ `_route_increment`**.
- **`_steady_step_flops` / `flops`** — parent value `+ _route_increment` at full support
  (`nd = nd_prev = s`, `m` active); **`decode_step_flops`** — `.forward` only.
- **`train_step`** — inherited **unchanged** from `WarmMix`: it folds prior windows through `step`,
  so the same routed columns + gate warm on the prior corpus, every FLOP charged. No new pretraining
  path, no backprop.
- **`from_config`** — keep only `ColumnMixConfig` fields (ignore harness-injected transformer keys);
  coerce `delta_orders` list → tuple (inherited behavior).
- **Register** in `smolml/models/__init__.py` (`ColumnMix`, `ColumnMixConfig`).

## Acceptance

- **Gates green:** `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest`.
- **`tests/test_column_mix.py`** (mirror `tests/test_delta_mix.py`):
  - registration + buildable + `num_params() == 0`; `from_config` admits fields / ignores
    transformer keys / coerces list `delta_orders`; config validation (each guard above).
  - **`n_columns=1` ⇒ bit-identical predictions AND `FlopBreakdown` to `delta_mix`** (warmed, every
    eval step) — the headline degenerate identity.
  - **`delta_orders=()` ⇒ bit-identical to `hashed_mix`** for `C > 1` (inherited path).
  - route map: `route_slot` deterministic, bounded to `[0, B)`; `gate_lr=0` ⇒ chosen column ==
    `bucket mod C` every step (fixed-hash route), reproducible.
  - prequential smoke (finite, `0 < bpb ≤ 8`); **no leakage** (perturbing a future byte leaves all
    past predictions unchanged).
  - **FLOP charge matches the code** — `step`'s breakdown == `delta_mix`'s breakdown (one active
    column) + the analytic `_route_increment`, exactly, with a hand-formula pin (`gather(1) +
    pointwise(3 + C)` forward, gate-update backward).
  - **specialization existence proof (the (iv) claim)** — on a fixture of `C` distinct
    sub-distributions selectable by the route context, the routed sheet (`C > 1`) beats the single
    delta (`C = 1`) in next-byte bpb by a clear margin. (Analog of the bar's "error-correction beats
    Hebbian" test.)
  - **learned-routing existence proof** — on a fixture where the hash partition is suboptimal but a
    better assignment exists, `gate-on` (`gate_lr>0`, `ε>0`) reaches lower bpb than `gate-off`
    (`gate_lr=0`). If this cannot be shown cheaply, document why and lean on the kill-test.
  - deep-copy isolation (an eval stream's `Wcols`/`gate` write never touches the warm state or
    another stream); reproducible under a fixed seed.
- **Kill-test experiment (CI-fast, a few-MB real-enwik8 slice — `smolml/experiments/
  column_mix_enwik8.py`, mirroring `delta_mix_enwik8.py`). MUST run and be reported BEFORE any
  full-corpus claim.** Matched **total** FLOPs:
  - (a) `delta_mix` (`C = 1`, the bar) warmed to budget `P`;
  - (b) `column_mix` (`C > 1`, learned gate) at the **same total** FLOPs;
  - (c) `column_mix` (`C > 1`, **gate-off** fixed-hash route) at matched FLOPs.
  **Kill unless (b) beats (a)** in bpb (routing earns its keep). Report (b) vs (c) for the
  learned-vs-static-routing verdict. Diagnostics, logged: per-column load (gate collapse to one
  column?), per-column conditional bpb (real specialization?), the delta-row mixer weight, and gate
  drift off the hash prior.
- **Full ADR carve** (extend `smolml/experiments/full_corpus.py` with a `column_mix` entrant on the
  full 95 MB prior / 5 MB eval; detached, multi-hour) — run **only if the kill-test passes**. Plot
  bpb-vs-total-FLOP over `C ∈ {2,4,8,16}` against the `1.8485 @ 1.322e12` bar; report peak RAM and
  the gate diagnostics. Honest either way.

## Out of scope (this spec)

- **Phase 1b — lateral predictive columns** (the brain-like depth move): columns predict neighbors'
  `z_c`; each contributes only its *residual* (lateral inhibition / sparse coding). Designed in its
  own spec **only if the flat routed sheet beats the bar** — honors "flat first, then depth" and
  B.1's Pareto-hollow warning.
- **Phase 2 — embodied snake rung** (sensorimotor / scalar-reward control): its own brainstorm after
  phase 1 lands.
- **`m > 1` soft routing** — a sweep extension, not the headline; the bandit gate at `m = 1` is the
  designed mechanism.
- **A learned dense/projected route encoder** — the cheap `(B×C)` gate table over a fixed hash bucket
  is the whole point (`O(C)` routing); a learned encoder would be a different, heavier candidate.
