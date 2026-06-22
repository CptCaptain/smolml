# delta_mix Implementation Plan

> **For agentic workers:** implement task-by-task, TDD, frequent commits. Steps use `- [ ]`.
> Design of record: `docs/tasks/B.4-delta-mix.md` (read it first for mechanism, the (iv) story, and
> the kill-test). This plan adds the implementation-critical decisions the spec leaves open.

**Goal:** Add `delta_mix` — a `HashedMix` subclass with one online error-correcting delta-rule
fast-weight predictor on a fixed sparse signed hashed bag-of-n-grams key, mixed as an extra stream —
and a CI-fast matched-FLOP kill-test that decides it before any full-corpus claim.

**Architecture:** `DeltaMix(HashedMix)` inherits the warm handoff, hashed count store, logistic
mixing, online mixer-SGD, and all count-side FLOP accounting. It adds exactly one mixture stream:
fast weights `W ∈ ℝ^(V×d)` updated by the LMS/delta rule on a sparse key. No backprop, no
`nn.Parameter`, `num_params()==0`.

**Tech Stack:** Python 3.12, numpy (the count-mixing lineage is numpy, not torch — match it), torch
only for the `(V,)` logits tensor `step` returns. `uv run` for everything.

## Global Constraints (verbatim from spec / AGENTS.md)

- Lines ≤ 100 chars; 4-space indent; modern typing (`list[str]`, `X | None`); type hints everywhere.
- KISS, hard cutover, no backward-compat shims. Keep deps to torch/numpy. Reproducible seeds.
- **FLOP honesty is the product:** every op charged via `smolml.flops`; non-matmul work that isn't
  charged scores as free and invalidates the result.
- Gates before PR: `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest`.
- Bar to beat: `hashed_mix` order-6 warmfull **2.0157 bpb @ 1.48e12** on the 5 MB ADR eval.

---

## Key implementation decisions (read before any task)

**D1 — `_MixerState` gets three optional fields (shared base, harmless to others).** In
`smolml/models/context_mixing.py`, append to `_MixerState`:
```python
    W: np.ndarray | None = None              # (V, d) fast weights; None = delta stream off
    last_phi: tuple[np.ndarray, np.ndarray] | None = None   # (indices[s], signs[s]) last key
    last_p_delta: np.ndarray | None = None   # (V,) softmax(z_delta) of the pending delta pred
```
Defaults `None`, appended last → other models (warm/hashed/gated) construct positionally unchanged
and never read them. This avoids a parallel state class and its construction-site overrides.

**D2 — Degenerate delegation guarantees the identity.** Every `DeltaMix` override begins
`if not self.config.delta_orders: return super().<m>(...)`. With `delta_orders=()` the delta stream
is off and `delta_mix` runs the *exact* `HashedMix` code path → bit-identical predictions AND
`FlopBreakdown`. The identity is structural, not re-derived.

**D3 — Widen the context window.** `delta_orders` up to 8 need 8 context bytes; `ContextMixing`
caps the rolling window at `max_order`. Define
`self._window_cap = max(cfg.max_order, max(cfg.delta_orders, default=0))`; `step` caps `new_window`
at `self._window_cap` and `context_window` returns it (delta on). With delta off, both equal
`max_order` (parent behaviour). Guard `max(delta_orders) <= 8` in `__post_init__` (the Fibonacci
hash only mixes the low 8 bytes — orders > 8 would alias, same limit `HashedMix._slot` documents).

**D4 — `num_predictors` stays `K = max_order+1`; the delta stream is "extra".** The count-side
`super()._flop_breakdown` keeps charging the K count streams correctly; `DeltaMix` adds a precise
*increment* for the (K+1)-th stream. The mixer weight vector is length `n_streams = K + 1` (delta
on) / `K` (off). The stretched matrix is `(n_streams, V)`: rows `0..K-1` are the log-Laplace count
predictions (parent), row `K` is the raw delta logit `z_delta = W·φ` (no log-stretch — softmax is
shift-invariant, so a raw logit is exact in the mix and the mixer gradient, since `err` sums to 0).

**D5 — Two distinct probability vectors.** `last_probs` = `softmax(z_mix)` (the mixed dist, grades
the **mixer** weights via `mixer_gradient`, parent-owned). `last_p_delta` = `softmax(z_delta)` (the
delta stream's OWN dist, grades the **delta `W`** via the LMS rule). Don't conflate them.

---

### Task 1: config, state field, registration skeleton, degenerate identity

**Files:**
- Modify: `smolml/models/context_mixing.py` (the `_MixerState` fields, D1)
- Create: `smolml/models/delta_mix.py` (`DeltaMixConfig`, `DeltaMix` skeleton)
- Modify: `smolml/models/__init__.py` (import + `__all__`)
- Test: `tests/test_delta_mix.py`

**Interfaces produced:**
- `DeltaMixConfig(HashedMixConfig)`: `delta_dim: int = 1<<18`, `delta_eta: float = 0.1`,
  `delta_orders: tuple[int, ...] = (3,4,5,6,7,8)`, `delta_signed: bool = True`.
- `@register_model("delta_mix") class DeltaMix(HashedMix)` with `from_config` (keep only
  `DeltaMixConfig` fields, like `HashedMix.from_config`).

- [ ] **Step 1 — failing test (degenerate identity, predictions + FLOPs).** A `delta_mix` with
  `delta_orders=()` must match `hashed_mix` byte-for-byte on a fixture stream. Build both with the
  same `{max_order, table_bits, hash_min_order, seed}`, run `prequential_bpb` (or a manual `step`
  loop) on `synthetic_text8(2000, seed=0).data`, assert equal `bpb` AND equal summed `total_flops`.
  Also `__post_init__` validation: `delta_dim` power of two ≥ 2; `delta_eta>0`; entries ≥ 1 and ≤ 8;
  empty `delta_orders` is valid.
- [ ] **Step 2 — run, verify it fails** (`DeltaMix` undefined). `uv run pytest
  tests/test_delta_mix.py -x`.
- [ ] **Step 3 — implement.** Add the D1 fields. Write `DeltaMixConfig` + validation. Write
  `DeltaMix` with: `__init__` storing `_window_cap` (D3); `from_config`; and **all** overrides
  delegating to `super()` when `not self.config.delta_orders` (D2): `step`, `flops`,
  `decode_step_flops`, `_steady_step_flops`, `context_window`, `init_prequential_state`,
  `_ensure_warm`. (Delta-on bodies are filled by later tasks; for now they may also delegate so the
  module imports and the degenerate test passes.) Register import in `__init__.py`.
- [ ] **Step 4 — run, verify pass.**
- [ ] **Step 5 — commit:** `feat(B.4): delta_mix config + degenerate-identity skeleton`.

---

### Task 2: the feature map (`_build_phi` + `_delta_slot`)

**Files:** Modify `smolml/models/delta_mix.py`; Test `tests/test_delta_mix.py`.

**Interfaces produced:**
- `_delta_slot(self, ngram: bytes) -> tuple[int, float]` — `(idx, sign)`: idx = Fibonacci hash of
  `int.from_bytes(ngram, "little")` to `log2(delta_dim)` bits (reuse `_KNUTH`/`_MASK64` from
  `hashed_mix`); sign from a **second** odd constant `_KNUTH2` (top bit) → `+1.0`/`-1.0` when
  `delta_signed` else `+1.0`. Distinct constant so idx and sign decorrelate.
- `_build_phi(self, window: list[int]) -> tuple[np.ndarray, np.ndarray]` — for each `n in
  delta_orders` with `len(window) >= n`, hash `bytes(window[-n:])`; return `(indices, signs)` as
  `np.int64`/`np.float64` arrays of length `nd ≤ s` (the active delta features).

- [ ] **Step 1 — failing test.** Determinism (same ngram → same `(idx,sign)` across calls/instances
  — deterministic, NOT Python's salted `hash`); `idx ∈ [0, delta_dim)`; `sign ∈ {-1,+1}`;
  `delta_signed=False` ⇒ all `+1`; `_build_phi` returns `nd` features = count of orders with enough
  context (e.g. window of 5 bytes, `delta_orders=(3,4,5,6,7,8)` ⇒ `nd=3`).
- [ ] **Step 2 — run, verify fail.**
- [ ] **Step 3 — implement** `_KNUTH2 = 0x2545F4914F6CDD1D` (or any odd 64-bit ≠ `_KNUTH`),
  `_delta_slot`, `_build_phi`.
- [ ] **Step 4 — run, verify pass.**
- [ ] **Step 5 — commit:** `feat(B.4): sparse signed hashed bag-of-n-grams feature map`.

---

### Task 3: `step` prediction path (matvec + K+1 mix), delta-on

**Files:** Modify `smolml/models/delta_mix.py`; Test `tests/test_delta_mix.py`.

Override `step` (delta-on body). Reuse module-level `laplace_prob`/`mix_logits`/`softmax`/
`mixer_gradient` and inherited `_fold_one`/`_lookup_one`. Order of operations mirrors
`ContextMixing.step` with the delta additions (the *update* steps land in Task 4 — here, predict
only, and stash `last_*`):

```python
def step(self, state, revealed_byte, pos):
    if not self.config.delta_orders:
        return super().step(state, revealed_byte, pos)
    cfg = self.config; ms = state.cache; v = cfg.vocab_size
    window = state.tokens; K = self.num_predictors
    did_update = ms.last_probs is not None
    if did_update:                                   # (mixer SGD — Task 4 charges it)
        ms.weights -= cfg.lr * mixer_gradient(ms.last_probs, ms.last_stretched, revealed_byte)
    # (delta W update — Task 4)
    n_fold = 0                                        # fold counts (parent loop)
    for k in range(K):
        if k == 0 or len(window) >= k:
            n_fold += 1
            self._fold_one(ms.tables, k, bytes(window[-k:]) if k else b"", revealed_byte)
    cap = self._window_cap
    new_window = [*window, revealed_byte][-cap:] if cap else []
    stretched = np.empty((K + 1, v)); n_active = 0; n_laplace = 0
    for k in range(K):                                # count predictions (parent logic)
        cell = None
        if k == 0 or len(new_window) >= k:
            n_active += 1
            cell = self._lookup_one(ms.tables, k, bytes(new_window[-k:]) if k else b"")
        if cell is not None:
            n_laplace += 1; stretched[k] = np.log(laplace_prob(cell, cfg.alpha))
        else:
            stretched[k] = np.log(self._uniform)
    idxs, signs = self._build_phi(new_window)         # delta stream
    nd = idxs.shape[0]
    z_delta = (ms.W[:, idxs] * signs[None, :]).sum(axis=1) if nd else np.zeros(v)
    stretched[K] = z_delta
    z = mix_logits(stretched, ms.weights); probs = softmax(z)
    ms.last_stretched = stretched; ms.last_probs = probs
    ms.last_phi = (idxs, signs); ms.last_p_delta = softmax(z_delta)
    new_state = DecodeState(tokens=new_window, cache=ms, length=state.length + 1)
    flops = self._delta_flop_breakdown(did_update=did_update, n_fold=n_fold,
        n_active=n_active, n_laplace=n_laplace, nd=nd, nd_prev=..., delta_updated=...)  # Task 5
    return new_state, torch.from_numpy(z.astype(np.float32)), flops
```
(`W` is created in Task 6's state handoff; until then a temporary `_ensure_warm` that zeros `W`
suffices to test prediction.)

- [ ] **Step 1 — failing test.** With a freshly-zeroed `W` the delta row is all-zeros ⇒ the mixed
  prediction equals the count-only mix of a `hashed_mix` with the **same** wider window — assert a
  `delta_mix` (delta on, `W=0`, `eta=0` so no update) gives the same first-byte distribution as the
  count streams alone (delta contributes a constant 0 logit). Also assert `stretched.shape == (K+1,
  V)` and `next_logits.shape == (V,)`.
- [ ] **Step 2–4 — fail / implement / pass.**
- [ ] **Step 5 — commit:** `feat(B.4): delta_mix prediction path (sparse matvec + K+1 mix)`.

---

### Task 4: `step` update path (delta rule) + no-leakage + error-correction

**Files:** Modify `smolml/models/delta_mix.py`; Test `tests/test_delta_mix.py`.

Insert the deferred delta `W` update (before the count fold, using the PREVIOUS key):
```python
    delta_updated = ms.last_p_delta is not None
    if delta_updated:
        pidx, psign = ms.last_phi
        err = ms.last_p_delta.copy(); err[revealed_byte] -= 1.0     # p_delta - onehot
        if pidx.shape[0]:
            ms.W[:, pidx] -= cfg.delta_eta * np.outer(err, psign)   # rank-1, nd_prev cols
```
Track `nd_prev = pidx.shape[0]` and `delta_updated` for the FLOP charge (Task 5).

- [ ] **Step 1 — failing tests.**
  - **No leakage:** perturb a future byte in the eval stream; assert every past `step`'s
    `next_logits` is unchanged (the update for byte `pos` uses only byte `pos`). Mirror
    `test_fast_weight.py::test_prediction_at_t_cannot_see_byte_t`.
  - **Error-correction beats Hebbian:** on a fixture with overlapping n-grams (e.g. a stream where
    `"abcabc"` patterns recur), a `delta_mix` (delta rule) reaches lower delta-stream bpb than a
    Hebbian variant (`W[:,j] += sign·onehot`, no error) at matched steps — the load-bearing claim.
    Implement the Hebbian variant inline in the test (subclass overriding the update) — do NOT ship
    it.
- [ ] **Step 2–4 — fail / implement / pass.**
- [ ] **Step 5 — commit:** `feat(B.4): online error-correcting delta-rule W update (no leakage)`.

---

### Task 5: FLOP accounting (`_delta_flop_breakdown` / `_delta_step_flops`)

**Files:** Modify `smolml/models/delta_mix.py`; Test `tests/test_delta_mix.py`.

`_delta_flop_breakdown(...)` = `super()._flop_breakdown(did_update, n_fold, n_active, n_laplace)`
(the K count streams, unchanged) **+** the delta increment below. `V = vocab_size`.

*Forward increment (always, delta on):*
- column access of `nd` W-columns → `gather_flops(nd)`
- hash arithmetic for the key → `pointwise_flops(6 * nd)` (Fibonacci idx 3 + sign 3 per feature;
  `int.from_bytes` bundled in, as `HashedMix` bundles its key hashing into the gather)
- sparse matvec `z_delta` (`nd·V` mul by sign + `nd·V` accumulate) → `pointwise_flops(2 * nd * V)`
- the (K+1)-th row in `weights @ stretched` → `pointwise_flops(2 * V)`
- `softmax(z_delta)` for `last_p_delta` → `pointwise_flops(5 * V)`

*Backward increment (only when `delta_updated`):*
- column access of `nd_prev` columns → `gather_flops(nd_prev)`
- `err = p_delta − onehot` one-hot subtract → `pointwise_flops(1)` (match the parent's `+1`)
- `eta·err` scale → `pointwise_flops(V)`
- rank-1 outer write+accumulate over `nd_prev` cols → `pointwise_flops(2 * nd_prev * V)`
- (only when `did_update`) the delta row in the mixer gradient `stretched @ err` → `+2*V`; the delta
  weight in the step → `+2`. → `pointwise_flops(2 * V + 2)`

Expose a pure-analytic `_delta_step_flops()` (all `nd = nd_prev = s`, `did_update=delta_updated=
True`) for `flops`/`_steady_step_flops`/`decode_step_flops` (the last uses `.forward` only, D4).

- [ ] **Step 1 — failing test.** Drive one `step` with known `nd`, `nd_prev`, `did_update`; assert
  the returned `FlopBreakdown` equals `super()._flop_breakdown(...)` plus the hand-computed delta
  increment, **exactly** (mirror `test_fast_weight.py::test_step_flops_equal_core_plus_memory`).
  Also re-assert the degenerate (`delta_orders=()`) per-byte FLOP identity to `hashed_mix`.
- [ ] **Step 2–4 — fail / implement / pass.**
- [ ] **Step 5 — commit:** `feat(B.4): exact FLOP charge for the delta stream (read+write+mix)`.

---

### Task 6: warm/eval state handoff (`_fresh_cache`, `_ensure_warm`, `init_prequential_state`)

**Files:** Modify `smolml/models/delta_mix.py`; Test `tests/test_delta_mix.py`.

```python
def _fresh_cache(self) -> _MixerState:
    cfg = self.config
    n = self.num_predictors + (1 if cfg.delta_orders else 0)
    W = np.zeros((cfg.vocab_size, cfg.delta_dim)) if cfg.delta_orders else None
    return _MixerState(tables=self._new_tables(), weights=np.full(n, 1.0 / n), W=W)

def _ensure_warm(self):                       # delta on: delta-sized fresh state
    if not self.config.delta_orders: return super()._ensure_warm()
    if self._warm is None: self._warm = self._fresh_cache()
    return self._warm

def init_prequential_state(self):             # delta on: deep-copy warm incl. W
    if not self.config.delta_orders: return super().init_prequential_state()
    warm = self._ensure_warm()
    cache = _MixerState(tables=self._copy_tables(warm.tables), weights=warm.weights.copy(),
                        W=warm.W.copy())
    return DecodeState(tokens=[], cache=cache)
```
`train_step` is inherited from `WarmMix` unchanged — it folds prior rows through `DeltaMix.step`, so
the same delta rule warms `W` on the prior corpus, every FLOP charged. (Confirm `WarmMix.train_step`
resets only `last_stretched`/`last_probs` per row; also reset `last_phi`/`last_p_delta` there if a
per-row reset is needed — if so, override `train_step` minimally to reset all four, else inherit.)

- [ ] **Step 1 — failing tests.**
  - **Deep-copy isolation:** two eval streams from the same warmed model; folding bytes into one
    leaves the other's `W` and the persistent `self._warm.W` unmutated.
  - **Warm reproducibility:** same seed/prior ⇒ identical warmed `W` and identical eval bpb.
  - **Generalization existence proof (the (iv) claim):** warm a `delta_mix` on a prior, then on
    eval-stream contexts whose exact order-6 k-gram was NOT seen in warm, the **delta-only** next-byte
    bpb beats the **order-6-only** next-byte bpb. (Construct via a prior/eval carve where eval has
    novel 6-grams whose sub-n-grams appeared in warm.) The claim is true iff delta wins here.
- [ ] **Step 2–4 — fail / implement / pass.**
- [ ] **Step 5 — commit:** `feat(B.4): warmed W prior->eval handoff (leak-free deep copy)`.

---

### Task 7: kill-test runner + full-corpus entrant

**Files:**
- Create: `smolml/experiments/delta_mix_enwik8.py` (mirror `warm_mix_enwik8.py`)
- Modify: `smolml/experiments/full_corpus.py` (add a `delta_mix` entrant)

- [ ] **Step 1 — implement the CI-fast kill-test** on a few-MB enwik8 slice
  (`prepare_enwik8(n_bytes=...)`, ADR carve). At **matched total FLOPs** run three configs and print
  a small table + the two diagnostics:
  - (a) `hashed_mix` counts-only warmed to the budget;
  - (b) `delta_mix` (counts + delta) warmed to the SAME budget (⇒ fewer warm bytes);
  - (c) `hashed_mix` counts-only with delta's FLOPs reallocated to MORE warm bytes.
  Log: (b)'s final mixer weight on the delta stream (→0 ⇒ dead), and (b)'s delta-only vs
  order-6-only bpb on warm-unseen contexts. **Verdict: PASS iff (b) beats both (a) and (c).**
- [ ] **Step 2 — add the `delta_mix` entrant to `full_corpus.py`** (full 95 MB prior / 5 MB eval,
  detached, multi-hour) — to be run **only if the kill-test passes**. Report peak RAM + the delta
  stream's final mixer weight.
- [ ] **Step 3 — commit:** `feat(B.4): matched-FLOP kill-test runner + full-corpus entrant`.

---

## Self-review (run after writing the code, before cross-review)

1. **Spec coverage:** every spec acceptance bullet maps to a Task-N test (degenerate identity ×2,
   no-leakage, FLOP charge, error-correction, generalization, deep-copy isolation, kill-test).
2. **No undercharge (reflex #3):** the delta write `2·nd·V`, the matvec `2·nd·V`, the hashes, the
   extra mix/softmax rows are ALL charged; no one-hot shortcut on the write. This is the cross-vendor
   review's first target.
3. **No leakage:** the update for byte `pos` uses `last_phi`/`last_p_delta` built at `pos-1` from
   bytes `< pos`; eval folds into a deep copy only.
4. **Type/name consistency:** `_delta_slot`, `_build_phi`, `_fresh_cache`, `_delta_flop_breakdown`,
   `_delta_step_flops`, `_window_cap` used identically across tasks.

## Out of scope (do not build)

`MetaFWP` (dropped), `hopfield_mix` (queued next), learned/dense keys, GPU paths.
