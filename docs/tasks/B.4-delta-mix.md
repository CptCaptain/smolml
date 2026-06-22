# Task B.4 — online delta-rule fast-weight memory (generalizing context mixing)

- Status: SPEC — design approved (this session: **`delta_mix` first, `hopfield_mix` queued next**).
  Not yet implemented.
- The next Space-B **learning-rule** candidate under the Source-(iv) filter (ADR 0003): a
  pure-online, error-correcting fast-weight predictor mixed into the warmed count ensemble. The
  user's steer for this candidate was explicit — *a learning dynamic that beats SGD per-FLOP*, not
  another structure-capture mixer.
- **Bar to beat** (real 5 MB enwik8 ADR eval, all FLOPs counted): `hashed_mix` order-6 warmfull
  **2.0157 bpb @ 1.48e12 total FLOPs**; transformer 5.4770 @ 1.46e14. Beating the transformer is
  trivial (~100,000× the FLOPs); this candidate must beat the **cheap online mixer**.

## The bet (Source-(iv))

A count table is a *degenerate associative memory*: a **one-hot-key Hebbian** store — each exact
context owns its own cell, updated by a `+1` increment. Two structural limits follow:

1. **Zero generalization.** A novel/unseen exact k-gram makes that order *abstain*; at full corpus
   the order-6 hashed table is overloaded, so most eval contexts are unseen.
2. **Only `K=7` global mixer weights** combine orders — "order-3 is worth `w_3` *everywhere*." The
   mixer cannot learn that *some* trigrams are far more predictive than others.

`delta_mix` adds **one** online predictor whose learning rule is an **error-correcting delta (LMS)
update** on a *distributed* key, so it holds `d×V` **per-feature, per-byte** affinities: it learns
"`tio`→`n` is a strong rule" once and shares it across every context containing `tio`, weighted by
its *measured residual utility*. A never-seen order-6 context then gets a calibrated blend of its
sub-n-gram signals from a single weight matrix instead of an order-6 abstention. It also reaches
**beyond the order-6 ladder** (orders 7–8 in the key at ~free marginal cost, where adding exact
order-7/8 tables is memory/noise-prohibitive).

**Why this is a real (iv) claim, stated honestly.** It *is* online gradient descent — but the model
is **linear in a fixed feature map** and the loss is **convex in `W`**, so the exact gradient is
`(p − target) ⊗ φ`: a single rank-1 outer product with **zero backward pass, no chain rule, no 2×
backprop tax, no bad minima to escape**. That is the legitimate "more loss-reduction per FLOP than
SGD-on-a-net" edge — the same dynamic, but the architecture makes the gradient *free* and the
landscape convex. The structural win over the count-mixer is the per-feature generalization above.

## Mechanism — `delta_mix`

`@register_model("delta_mix") class DeltaMix(HashedMix)`. Inherits the warm prior→eval handoff, the
hashed count store, the logistic mixing, the online mixer-SGD, and **all** count-side FLOP
accounting; adds exactly one mixture stream.

**State** (lives in `DecodeState.cache`, per stream; warm-started then deep-copied like the tables):
- Count store `tables` + mixer weights `weights ∈ ℝ^(K+1)` — inherited (`K = max_order+1 = 7`); the
  mixer vector grows by one slot for the delta stream.
- **Fast weights** `W ∈ ℝ^(V×d)`, `V = 256`, default **`d = 2**18`** (memory/collision knob only —
  see FLOP honesty), initialised to `0` (an unwritten `W` ⇒ `z = 0` ⇒ uniform contribution ⇒ a clean
  abstention, exactly like an unseen count cell).
- Deferred-update scratch: `last_phi` (the sparse key built last step), `last_p_delta`
  (`softmax(z)` last step) — alongside the inherited `last_stretched`/`last_probs`.

**Feature map `φ(ctx)` — fixed, cheap, sparse, signed, distributed.** For the context window ending
at the current position, take the byte n-grams for `n ∈ delta_orders` (default **`(3,4,5,6,7,8)`**,
`s = |delta_orders| = 6`). Hash each n-gram with the **same** Fibonacci hash `HashedMix._slot`
already ships into a bucket `j ∈ [0,d)`, and set `φ[j] = ξ(ngram)` where `ξ ∈ {+1,−1}` is a 1-bit
sign from a second hash (signed feature-hashing — colliding features cancel in expectation instead
of piling up). `φ` is **`s`-sparse** (≤ 6 nonzeros), stored as `(indices[s], signs[s])`. Orders
0–2 are **not** in the key — they saturate instantly and are already perfectly served by the exact
count tables (nothing to generalize there).

**Prediction.** `z = W @ φ = Σ_{j∈supp(φ)} φ[j]·W[:,j] ∈ ℝ^V` — touches only `s` columns. `z` is
fed as one more **raw-logit** "stretched" row into the existing mixer (softmax is shift-invariant, so
a raw logit is exact — no log-softmax needed for the *mix*). Mixed logits `z_mix = weights @
stretched` over `K+1` rows; `next_logits = z_mix`.

**Update — error-correcting delta**, applied to the *previous* step's key when its byte is revealed
(the existing pending-prediction pattern — the update for byte `pos` uses only byte `pos`, no
leakage):
```
p   = softmax(z_prev)                 # = last_p_delta
err = p − onehot(revealed)            # the softmax-CE gradient w.r.t. z
W[:, j] −= η · φ_prev[j] · err        for j ∈ supp(φ_prev)   # rank-1, s columns only
```
This is exactly `W ← W − η (softmax(W·φ) − target) φᵀ` restricted to the sparse support. The
mixer's own `(K+1)`-dim SGD weight on the delta stream is graded online by the existing
`mixer_gradient`, so the ensemble decides *how much to trust* the delta stream byte-by-byte.

**Delta, NOT plain Hebbian (load-bearing).** Vanilla Hebbian (`W[:,j] += φ[j]·onehot(byte)`, no
error) on a *superposed* key is fatal: every feature is pulled toward the byte marginal, overlapping
n-grams double-count, and `W` collapses to "predict the global frequency" — precisely the A.1
failure mode ("memory does the same weak thing regardless of context"). The delta rule subtracts
what `W·φ` *already* predicts, so each bucket learns its **residual** contribution and correlated
features decorrelate. Error-correction is what makes a distributed key work.

### FLOP honesty (the feasibility crux — every op charged via `smolml.flops`)

`V = 256`, `s = 6`. `d` affects **memory and collisions only — never FLOPs** (sparse access touches
`s` columns regardless of `d`). A *dense* `d`-key would cost `2dV` each way — at `d = 2**18` that is
~5600× the entire bar, dead. Sparsity makes both the matvec and the rank-1 write `O(sV)`, charged at
the true `2sV`; `s` is bounded by `|delta_orders|`, so the charge is exact.

Delta add-on per byte, on top of `HashedMix`'s inherited per-byte breakdown:
- *Forward (prediction):* feature build `s` Fibonacci hashes + `s` sign hashes ≈ `pointwise(~4s)`;
  sparse matvec `z = Wφ` = `2sV`; delta row in the `(K+1)`-stream mix = `+2V`.
  **forward ≈ 2sV + 2V + 4s = 3608.**
- *Adaptation (this model's own update):* `softmax(z)` = `pointwise(5V)`; `err = p − onehot` (1) +
  scale by η `pointwise(V)`; rank-1 write+accumulate over `s` columns = `2sV` (dense outer-product
  cost, charged exactly as A.1 charged its write — no one-hot shortcut); delta's slot in the
  `2(K+1)` mixer update = `+2`. **backward ≈ 2sV + 6V + 3 = 4611.**
- **Delta total ≈ 8219 FLOPs/byte**, exposed via a new `_delta_step_flops()` and added to the
  parent breakdown in `step`/`flops`/`decode_step_flops`/`_steady_step_flops`. Combined with the
  bar's ~15.7k ⇒ **≈2.39e4 FLOPs/byte**.

The n-gram-key materialization (`int.from_bytes` per order) is the `O(k)` analogue of the dict/hash
key hashing `hashed_mix` already bundles into its single gather — charge it the **same** way the
parent does, consistently. No compute hides in the hashing. (Reflex #3: cross-vendor review has
caught an undercharge in *every* prior candidate — the review prompt for this one must target the
delta write + feature-build charge specifically.)

### Memory

`W = V·d·4 B = 256·2**18·4 ≈ 268 MB` per stream (eval deep-copies one more), well under the bar's
3–5 GiB. `delta_dim` trades RAM for collision crosstalk.

## Config

`DeltaMixConfig(HashedMixConfig)` adds: `delta_dim: int = 1 << 18`, `delta_eta: float` (LMS step,
tune; logistic-LMS is robust around `0.05–0.5`), `delta_orders: tuple[int, ...] = (3,4,5,6,7,8)`,
`delta_signed: bool = True`. Reuse `max_order`, `alpha`, `lr`, `table_bits`, `hash_min_order`, and
the `WarmMix` warmup. Validate in `__post_init__` (`delta_dim` a power of two; every entry in
`delta_orders` ≥ 1; `delta_eta > 0`). Empty `delta_orders` is **valid** — it disables the stream
(the degenerate identity below).

**Degenerate identity.** With `delta_orders = ()` (empty) the delta stream is disabled and the mixer
is `K`-wide: `delta_mix` is then **bit-identical** to `hashed_mix` — same predictions *and* same
`FlopBreakdown` (assert in tests, mirroring the `hashed_mix == warm_mix` identity).

## The honest expectation / break-even (report straight)

The delta add-on (~8.2k FLOPs/byte, ≈ +52% per-byte) is paid by warming on **fewer prior bytes** at
a fixed total budget. At ~2.39e4 FLOPs/byte:
- Full budget (1.5e12) ⇒ ~**58 MB** warm. Count-only at 58 MB ≈ 2.034 bpb (docs extrapolation,
  ~−0.083 bpb/decade). The delta stream must subtract **≥ ~0.018 bpb** of generalization to tie,
  more to win.
- Or warm **~30 MB** at ~8.4e11 total (count-only ≈ 2.057); delta must buy **≥ ~0.041 bpb** for a
  clean down-AND-left win.

This is a **real but unconfirmed** bet, and it is exactly the "refine an already-good cheap learner"
claim that A.1 / B.1 / B.2-gated all show *usually fails*. Two things make it different from A.1:
error-correction prevents the marginal-collapse, and the online mixer drives a useless stream's
weight → 0 with **no bpb blow-up** (bounded downside). The kill-test below decides it cheaply.

## Seam (zero harness changes)

- **`DeltaMixConfig(HashedMixConfig)`** — the four knobs above.
- **`__init__`** — store config; `W` is **not** an `nn.Parameter` (per-stream online fast weights,
  like the count tables) ⇒ `num_params() == 0`, no AdamW.
- **`_MixerState`** — extend with `W`, `last_phi`, `last_p_delta`; `weights` length becomes `K+1`
  **only when the delta stream is enabled** (length `K` when `delta_orders=()`, preserving the
  degenerate identity).
- **`init_prequential_state`** — deep-copy the warm `_MixerState` **including `W.copy()`**
  (mirrors `HashedMix._copy_tables`; leak-free — each eval stream mutates its own `W`).
- **`step`** (override; reuse `laplace_prob`/`mix_logits`/`softmax`/`mixer_gradient` and inherited
  `_fold_one`/`_lookup_one`/`_slot`): (1) deferred mixer-weight update over `K+1` rows; (2) deferred
  delta `W` update on `last_phi`/`last_p_delta`; (3) fold counts (parent loop); (4) build new `φ`,
  `z = Wφ`, stack as the `(K+1)`-th stretched row, mix, softmax; (5) stash `last_*`; (6) return the
  parent count breakdown **+** `_delta_step_flops()`.
- **`_steady_step_flops` / `flops`** — parent value `+ _delta_step_flops()` (full fwd+bwd);
  **`decode_step_flops`** — parent value `+ _delta_step_flops().forward` only (prediction excludes
  the update, matching the parent's forward-only convention).
- **`train_step`** — inherited **unchanged** from `WarmMix`: it folds prior windows through `step`,
  so the *same* cheap online delta rule warms `W` on the prior corpus, every FLOP charged. No new
  pretraining path, no backprop, no expensive-pretrain trap.

## Acceptance

- **Gates green:** `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest`.
- **`tests/test_delta_mix.py`:**
  - `delta_orders=()` ⇒ bit-identical predictions **and** `FlopBreakdown` to `hashed_mix`
    (degenerate identity);
  - **no leakage** — perturbing a future byte leaves all past predictions unchanged (the
    pending-prediction update for byte `pos` uses only byte `pos`);
  - **FLOP charge matches the code** — `step`'s breakdown equals the parent count breakdown plus the
    analytic `_delta_step_flops()` (`2sV` read + `2sV` write + softmax + mixer slot), exactly;
  - **error-correction beats Hebbian** — on a fixture with overlapping/superposed n-grams, the delta
    rule's next-byte bpb is below a plain-Hebbian variant's (the mechanism's load-bearing claim);
  - **generalization existence proof** — `delta`-only beats `order-6`-only next-byte bpb on contexts
    held out of warm (the (iv) claim is true iff this holds);
  - deep-copy isolation (one eval stream's `W` write never touches another's / the warm state);
    reproducible seed; registration.
- **Kill-test experiment (CI-fast, a few-MB enwik8 slice — `smolml/experiments/delta_mix_enwik8.py`,
  mirroring `warm_mix_enwik8.py`). MUST run and be reported BEFORE any full-corpus claim.**
  Matched **total** FLOPs, three configs:
  - (a) `hashed_mix` counts-only, warmed to the budget;
  - (b) `delta_mix` (counts + delta), warmed to the **same** budget (⇒ fewer warm bytes);
  - (c) `hashed_mix` counts-only with the delta's FLOPs reallocated to **more** warm bytes.
  **Kill unless (b) beats *both* (a) and (c)** in bpb. Two mechanistic diagnostics, logged: the
  mixer's learned weight on the delta stream (→ 0 ⇒ dead weight, the A.1 "gate weight identical"
  tell), and (b)'s delta-only vs order-6-only bpb on warm-unseen contexts.
- **Full ADR carve** (extend `smolml/experiments/full_corpus.py` with a `delta_mix` entrant on the
  full 95 MB prior / 5 MB eval; detached, multi-hour) — run **only if the kill-test passes**. Plot
  bpb-vs-total-FLOP against the bar; report peak RAM and the delta stream's final mixer weight.
  Honest either way.

## Out of scope

- Meta-learned / backprop-pretrained update generators (Family A `MetaFWP`) — self-conceded
  Pareto-hollow (reintroduces the A.1/B.1 expensive-pretrain trap); dropped.
- Long-range modern-Hopfield retrieval (Family C `hopfield_mix`) — **queued as the next candidate**,
  not built here (KISS: scout one clean learning-rule mechanism so the kill-test isolates the delta
  lever).
- A learned / dense / projected key — the whole feasibility crux is the *fixed sparse hashed* key
  (`O(sV)` not `O(dV)`); a learned encoder would be a different (and likely hollow) candidate.
