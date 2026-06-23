"""Online delta-rule fast-weight memory (Task B.4) — generalizing context mixing.

``delta_mix`` is :class:`HashedMix` plus exactly one extra mixture stream: an online,
error-correcting **fast-weight** predictor ``W`` keyed on a *fixed, sparse, signed hashed
bag of byte n-grams*. The count tables are a degenerate associative memory — a one-hot
context key updated by a ``+1`` (Hebbian) increment — so they cannot generalize: a novel
k-gram makes that order abstain, and the only cross-context combination is the ``K`` global
mixer weights. ``W`` holds ``V x d`` per-feature, per-byte learned affinities updated by the
LMS / delta rule, so it learns "this trigram predicts that byte" once and shares it across
every context containing the trigram, and reaches *beyond* the order-``max_order`` ladder.

Mechanism
---------
- **Key** ``phi(ctx)``: for each ``n in delta_orders`` with enough context, hash the n-gram
  ``ctx[-n:]`` (the same Fibonacci hash :meth:`HashedMix._slot` ships, to ``delta_dim``
  buckets) and a 1-bit sign from a *second* constant (signed feature-hashing — colliding
  features cancel in expectation). ``phi`` is ``s``-sparse (``s = |active delta_orders|``),
  stored as ``(indices, signs)``.
- **Predict** ``z_delta = W @ phi`` (touches only ``s`` columns) — fed as one more **raw
  logit** row into the existing logistic mixer (softmax is shift-invariant, so a raw logit
  is exact in both the mix and the mixer gradient, since the error sums to 0).
- **Learn** (error-correcting delta, deferred to the next byte — the pending-prediction
  pattern, so the update for byte ``pos`` uses only byte ``pos``): on the previous key,
  ``W[:, j] -= eta * phi[j] * (softmax(z_delta) - onehot(byte))``. This is online multinomial
  logistic regression restricted to the sparse support — the gradient *is* the prediction
  error (no backward pass, no 2x tax, convex). The delta update is gated on the SAME pending
  flag as the mixer update, so a warm-row boundary (which clears the pending prediction)
  cleanly skips it — no cross-row leakage and no ``train_step`` override needed.

Why delta, not plain Hebbian
----------------------------
Unconditional Hebbian (``W[:, j] += phi[j] * onehot``) on a *superposed* key pulls every
feature toward the byte marginal and double-counts overlapping n-grams, collapsing ``W`` to
"predict the global frequency" — the A.1 failure mode. The delta rule subtracts what ``W phi``
*already* predicts, so each bucket learns its **residual** and correlated features decorrelate.

FLOP honesty
------------
The key is ``s``-sparse, so the matvec AND the rank-1 outer-product write are ``O(sV)`` (the
``s`` touched columns x ``V``), charged at the true ``2sV`` each — NOT ``O(dV)`` (``d`` costs
RAM and collisions only, never FLOPs; this is the feasibility crux). The added per-byte charge
on top of :class:`HashedMix`'s breakdown is the explicit delta increment in
:meth:`_delta_increment`: column gathers, the key hashes, the matvec, the extra mix row, the
``softmax(z_delta)``, and (when a prediction was pending) the delta write + the (K+1)-th
mixer-gradient/step terms. Nothing hides in the stream.

Degenerate identity
-------------------
With ``delta_orders = ()`` the stream is disabled: every override delegates to ``super()``,
so ``delta_mix`` runs the exact :class:`HashedMix` code path — **bit-identical** predictions
*and* :class:`FlopBreakdown`.

No leakage
----------
Eval folds into its own deep copy (:meth:`init_prequential_state` copies the count store, the
mixer weights, AND ``W``), never the persistent warm state.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch

from smolml.flops import FlopBreakdown, gather_flops, pointwise_flops
from smolml.models.context_mixing import (
    _MixerState,
    laplace_prob,
    mix_logits,
    mixer_gradient,
    softmax,
)
from smolml.models.hashed_mix import _KNUTH, _MASK64, HashedMix, HashedMixConfig
from smolml.models.registry import DecodeState, register_model

# A second odd 64-bit constant (distinct from _KNUTH) for the sign hash, so a feature's bucket
# index and its +/-1 sign are independent (signed feature-hashing / the hashing trick).
_KNUTH2: int = 0x2545F4914F6CDD1D


@dataclass
class DeltaMixConfig(HashedMixConfig):
    """:class:`HashedMixConfig` plus the delta fast-weight knobs.

    ``delta_orders`` selects which byte n-gram orders form the sparse key; **empty disables the
    stream** (then ``delta_mix`` is bit-identical to ``hashed_mix``). ``delta_dim`` is the
    fast-weight column count (memory / collision knob only — never FLOPs, since the key is
    sparse). ``delta_eta`` is the LMS step; ``delta_signed`` toggles signed feature-hashing.
    """

    delta_dim: int = 1 << 18
    delta_eta: float = 0.1
    delta_orders: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    delta_signed: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.delta_orders = tuple(self.delta_orders)  # JSON run logs may store a list
        if self.delta_dim < 2 or (self.delta_dim & (self.delta_dim - 1)) != 0:
            raise ValueError(f"delta_dim must be a power of two >= 2, got {self.delta_dim}")
        if self.delta_eta <= 0.0:
            raise ValueError(f"delta_eta must be > 0, got {self.delta_eta}")
        if any(n < 1 for n in self.delta_orders):
            raise ValueError(f"delta_orders entries must be >= 1, got {self.delta_orders}")
        # int.from_bytes + the Fibonacci hash mix only the low 8 bytes, so an n-gram > 8 bytes
        # would alias on the ignored bytes (same limit HashedMix._slot documents).
        if self.delta_orders and max(self.delta_orders) > 8:
            raise ValueError(
                f"delta_orders must be <= 8 bytes (got max {max(self.delta_orders)}); "
                "widen the key hash to cover all bytes first"
            )


@register_model("delta_mix")
class DeltaMix(HashedMix):
    """Warmed hashed context-mixing plus one online delta-rule fast-weight stream.

    Inherits the warm prior->eval handoff, the hashed count store, the logistic mixing, the
    online mixer-SGD, and **all** count-side FLOP accounting; adds the delta stream and its
    exact FLOP charge. ``num_predictors`` stays ``K = max_order + 1`` (the delta stream is the
    extra ``(K+1)``-th); ``W`` is per-stream online state, NOT an ``nn.Parameter`` (so
    ``num_params() == 0`` and there is no AdamW).
    """

    config: DeltaMixConfig

    def __init__(self, config: DeltaMixConfig) -> None:
        super().__init__(config)
        # log2(delta_dim) — the bucket index takes the top this-many bits of the Fibonacci hash.
        self._delta_bits: int = config.delta_dim.bit_length() - 1
        # The rolling window must hold the longest delta n-gram (orders can exceed max_order);
        # equals max_order when the delta stream is disabled, so degenerate == hashed_mix.
        self._window_cap: int = max(config.max_order, max(config.delta_orders, default=0))

    # --- delta feature map (fixed, cheap, sparse, signed) ---------------------------

    def _delta_slot(self, ngram: bytes) -> tuple[int, float]:
        """Hash an n-gram to a ``(bucket, sign)`` pair: Fibonacci hash -> top ``_delta_bits``
        for the bucket; a second constant's top bit -> ``+1.0``/``-1.0`` (``+1.0`` when
        ``delta_signed`` is off). Deterministic (salt-free) so warmed runs reproduce."""
        x = int.from_bytes(ngram, "little")
        idx = ((x * _KNUTH) & _MASK64) >> (64 - self._delta_bits)
        if not self.config.delta_signed:
            return idx, 1.0
        sign = 1.0 if (((x * _KNUTH2) & _MASK64) >> 63) else -1.0
        return idx, sign

    def _build_phi(self, window: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Sparse signed key for ``window``: one ``(bucket, sign)`` per ``delta_orders`` entry
        whose n-gram fits the window. Returns ``(indices, signs)`` of length ``nd <= s``."""
        orders = [n for n in self.config.delta_orders if len(window) >= n]
        nd = len(orders)
        idxs = np.empty(nd, dtype=np.int64)
        signs = np.empty(nd, dtype=np.float64)
        for i, n in enumerate(orders):
            idx, sign = self._delta_slot(bytes(window[-n:]))
            idxs[i] = idx
            signs[i] = sign
        return idxs, signs

    # --- prequential step (fold counts + adapt W + predict, all charged) ------------

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        """Fold the revealed byte, adapt the mixer AND the delta ``W``, predict the next byte.

        Mirrors :meth:`ContextMixing.step` with the delta additions; delegates entirely to the
        parent (== ``hashed_mix``) when the delta stream is disabled."""
        cfg = self.config
        if not cfg.delta_orders:
            return super().step(state, revealed_byte, pos)
        ms: _MixerState = state.cache
        v = cfg.vocab_size
        window = state.tokens
        k_pred = self.num_predictors

        # 1. Online mixer update on the just-revealed byte (graded prediction).
        did_update = ms.last_probs is not None
        if did_update:
            grad = mixer_gradient(ms.last_probs, ms.last_stretched, revealed_byte)
            ms.weights -= cfg.lr * grad

        # 2. Online delta-rule W update on the PREVIOUS delta key (same pending gate as the
        #    mixer, so a warm-row boundary skips it -> no cross-row leakage).
        nd_prev = 0
        if did_update:
            pidx, psign = ms.last_phi
            nd_prev = int(pidx.shape[0])
            if nd_prev:
                self._apply_delta_update(ms, pidx, psign, revealed_byte)

        # 3. Fold the revealed byte into each available order-k count table (parent stores).
        n_fold = 0
        for k in range(k_pred):
            if k == 0 or len(window) >= k:
                n_fold += 1
                key = bytes(window[-k:]) if k else b""
                self._fold_one(ms.tables, k, key, revealed_byte)

        # 4. New context window (wide enough for the delta n-grams) + the count predictions.
        cap = self._window_cap
        new_window = [*window, revealed_byte][-cap:] if cap else []
        stretched = np.empty((k_pred + 1, v))
        n_active = 0
        n_laplace = 0
        for k in range(k_pred):
            cell = None
            if k == 0 or len(new_window) >= k:
                n_active += 1
                key = bytes(new_window[-k:]) if k else b""
                cell = self._lookup_one(ms.tables, k, key)
            if cell is not None:
                n_laplace += 1
                stretched[k] = np.log(laplace_prob(cell, cfg.alpha))
            else:
                stretched[k] = np.log(self._uniform)

        # 5. Delta stream: sparse key -> z_delta = W phi -> the (K+1)-th raw-logit row.
        idxs, signs = self._build_phi(new_window)
        nd = int(idxs.shape[0])
        z_delta = (ms.W[:, idxs] * signs[None, :]).sum(axis=1) if nd else np.zeros(v)
        stretched[k_pred] = z_delta

        # 6. Mix, predict, stash the pending prediction (mixer + delta).
        z = mix_logits(stretched, ms.weights)
        probs = softmax(z)
        ms.last_stretched = stretched
        ms.last_probs = probs
        ms.last_phi = (idxs, signs)
        ms.last_p_delta = softmax(z_delta)

        next_logits = torch.from_numpy(z.astype(np.float32))
        new_state = DecodeState(tokens=new_window, cache=ms, length=state.length + 1)
        flops = self._delta_flop_breakdown(
            did_update=did_update,
            n_fold=n_fold,
            n_active=n_active,
            n_laplace=n_laplace,
            nd=nd,
            nd_prev=nd_prev,
        )
        return new_state, next_logits, flops

    def _apply_delta_update(
        self, ms: _MixerState, pidx: np.ndarray, psign: np.ndarray, revealed_byte: int
    ) -> None:
        """Error-correcting delta (LMS) write on the previous key — the plasticity seam.

        ``W[:, j] -= eta * phi[j] * (softmax(z_delta) - onehot(byte))`` over the ``nd_prev`` touched
        columns: each bucket learns its *residual* contribution (correlated features decorrelate).
        Extracted so an ablation can swap the learning rule (e.g. plain Hebbian) without
        re-deriving :meth:`step`; the charged cost (:meth:`_delta_increment`) is this rule's."""
        err = ms.last_p_delta.copy()
        err[revealed_byte] -= 1.0  # softmax-CE gradient of the delta stream w.r.t. z_delta
        scaled = self.config.delta_eta * err  # scale err (V) BEFORE the outer -> V + 2*nd_prev*V
        # Accumulate over the touched columns (np.add.at, NOT ``W[:, pidx] -=``): if two n-grams in
        # the key collide to one bucket, both signed contributions must SUM — exactly as the read
        # ``(W[:, idxs] * signs).sum`` sums them. A plain fancy-index ``-=`` is last-write-wins and
        # would silently drop a colliding feature's update (read/write asymmetry).
        np.add.at(ms.W, (slice(None), pidx), -np.outer(scaled, psign))

    # --- FLOP accounting (parent count charge + the exact delta increment) ----------
    def _delta_increment(self, *, nd: int, nd_prev: int, did_update: bool) -> FlopBreakdown:
        """The delta stream's exact per-byte FLOPs on top of the parent count breakdown.

        Forward (always): ``nd`` column gathers + key hashes (``3`` ops/feature, idx only, plus
        ``3`` more for the sign hash when ``delta_signed``) + sparse matvec ``2*nd*V`` + the
        (K+1)-th mix row ``2V`` + ``softmax(z_delta)`` ``5V``. Backward (only when a prediction was
        pending, ``did_update``): the (K+1)-th mixer-gradient row ``2V`` + weight-step ``2`` ALWAYS;
        plus, only when the prior key had support (``nd_prev > 0``), ``nd_prev`` column gathers +
        one-hot subtract ``1`` + ``eta`` scale ``V`` + rank-1 write ``2*nd_prev*V`` (matching the
        ``if nd_prev`` guard in :meth:`step`, so the charge never bills work the code skips)."""
        v = self.config.vocab_size
        hash_ops = 6 if self.config.delta_signed else 3  # sign hash is skipped when unsigned
        forward = gather_flops(nd) + pointwise_flops(hash_ops * nd + 2 * nd * v + 2 * v + 5 * v)
        backward = 0
        if did_update:
            backward = pointwise_flops(2 * v + 2)  # (K+1)-th mixer-gradient row + weight step
            if nd_prev:
                backward += gather_flops(nd_prev) + pointwise_flops(1 + v + 2 * nd_prev * v)
        return FlopBreakdown(forward=forward, backward=backward)

    def _delta_flop_breakdown(
        self, *, did_update: bool, n_fold: int, n_active: int, n_laplace: int, nd: int, nd_prev: int
    ) -> FlopBreakdown:
        """Parent (count) breakdown + the delta increment. Named distinctly from
        ``_flop_breakdown`` so the degenerate ``super().step`` path keeps charging exactly the
        :class:`HashedMix` cost."""
        base = super()._flop_breakdown(
            did_update=did_update, n_fold=n_fold, n_active=n_active, n_laplace=n_laplace
        )
        return base + self._delta_increment(nd=nd, nd_prev=nd_prev, did_update=did_update)

    def _steady_step_flops(self) -> FlopBreakdown:
        """Analytic steady-state per-byte cost: the parent estimate plus the delta increment
        at full support (``nd = nd_prev = s``). ``flops`` / ``decode_step_flops`` inherit this
        (the latter takes ``.forward`` only — prediction excludes the update)."""
        if not self.config.delta_orders:
            return super()._steady_step_flops()
        s = len(self.config.delta_orders)
        return super()._steady_step_flops() + self._delta_increment(
            nd=s, nd_prev=s, did_update=True
        )

    # --- warmed state + leak-free eval handoff (carry W alongside the count store) --

    @property
    def context_window(self) -> int:
        """Bytes conditioned on per prediction = the widest n-gram (count or delta). Equals
        ``max_order`` when the delta stream is off (degenerate parity)."""
        return self._window_cap

    def _fresh_cache(self) -> _MixerState:
        """A fresh online state sized for the delta stream: ``K+1`` mixer weights and a
        zeroed ``W`` (``W = None`` and ``K`` weights when the stream is disabled)."""
        cfg = self.config
        n = self.num_predictors + (1 if cfg.delta_orders else 0)
        w = np.zeros((cfg.vocab_size, cfg.delta_dim)) if cfg.delta_orders else None
        return _MixerState(tables=self._new_tables(), weights=np.full(n, 1.0 / n), W=w)

    def _ensure_warm(self) -> _MixerState:
        """Lazily create the persistent warm state with the delta-sized weights + zeroed ``W``
        (delegates to the parent when the stream is disabled, for the degenerate identity)."""
        if not self.config.delta_orders:
            return super()._ensure_warm()
        if self._warm is None:
            self._warm = self._fresh_cache()
        return self._warm

    def init_prequential_state(self) -> DecodeState:
        """A deep copy of the warm state for one eval stream — count store, mixer weights, AND
        ``W`` — so eval mutates only its own copy (leak-free)."""
        if not self.config.delta_orders:
            return super().init_prequential_state()
        warm = self._ensure_warm()
        cache = _MixerState(
            tables=self._copy_tables(warm.tables),
            weights=warm.weights.copy(),
            W=warm.W.copy(),
        )
        return DecodeState(tokens=[], cache=cache)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "DeltaMix":
        """Build from a config dict, keeping only :class:`DeltaMixConfig` fields (so the
        harness-injected transformer keys are ignored, like the parent)."""
        fields = {f.name for f in dataclasses.fields(DeltaMixConfig)}
        kwargs = {key: val for key, val in config.items() if key in fields}
        return cls(DeltaMixConfig(**kwargs))
