"""Routed sheet of delta columns (Task B.5) — context-conditional selection over context mixing.

``column_mix`` is :class:`DeltaMix` with exactly one structural change: its single online delta-rule
fast-weight predictor ``W ∈ ℝ^(V×d)`` is replaced by **C columns** ``W_0..W_{C-1}`` (each the *same*
delta predictor on the *same* sparse signed hashed n-gram key ``φ``), plus a cheap **router** that
activates exactly **one** column per byte. The chosen column's prediction is fed as the single
``(K+1)``-th raw-logit row into the existing mixer, so the count-side path and mixer width are
unchanged from the bar.

The lever (Source-(iv))
-----------------------
A single linear ``W`` computes one fixed linear function of ``φ`` — the same n-gram features always
map to the same logits. Raising ``delta_dim`` to ``C·d`` buys the bar the same capacity for **zero**
extra FLOPs (sparse access is ``O(sV)`` regardless of ``d``), so capacity is *not* the lever. The
only thing ``C`` routed columns add is **route-conditional selection**: ``z = W_{route(ctx)} @ φ``
switches weight matrices on a context bucket, representing a *multiplicative* context×feature
interaction a linear-in-``φ`` predictor cannot. The bet: when the next byte depends on such an
interaction, routed columns extract it at ~the bar's per-byte FLOPs (one delta read/write + an
``O(C)`` router). High Pareto-hollow risk (count redundancy, thin per-column data, gate collapse) —
the kill-test's matched-capacity control decides it. See ``docs/tasks/B.5-column-mix.md``.

Router + gate
-------------
- Bucket ``b = route_slot(window[-route_order:])`` — one Fibonacci hash to the top ``log2 B`` bits.
- ``gate[b, c]`` is a per-arm **reward** estimate (``= −bits``) for column ``c`` at bucket ``b``,
  init so ``argmax gate[b,:] = b mod C`` (the hash prior). Route = ε-greedy argmax over the row.
- Two deferred local updates per byte (both gated on a pending prediction, so warm-row boundaries
  skip them — no leakage, no ``train_step`` override): the chosen column's delta (LMS) write, and a
  per-arm contextual-bandit EMA of the chosen column toward its observed reward. No global backward.

Degenerate identities
----------------------
- ``n_columns == 1`` (or ``delta_orders == ()``) ⇒ the column path is off and **every** override
  delegates to :class:`DeltaMix` — bit-identical predictions *and* ``FlopBreakdown`` (per-step and
  analytic). With ``delta_orders == ()`` that delegation continues to :class:`HashedMix`.
- ``gate_lr == 0 and route_epsilon == 0`` ⇒ the route is the deterministic hash partition
  (``b mod C``) every byte — a reproducible static-routing baseline.

FLOP honesty
------------
``step`` FLOPs = ``super()._delta_flop_breakdown(...)`` + ``_route_increment``. The chosen column
**is** the bar's one delta stream, so the count/delta charge is the parent's, reused verbatim; the
route increment adds only the router + the ``O(1)`` bandit update. At ``C = 1`` the path delegates,
so the charge is bit-identical to the bar (zero router). ``d``/``B`` cost memory only, never FLOPs.
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
from smolml.models.delta_mix import DeltaMix, DeltaMixConfig
from smolml.models.hashed_mix import _KNUTH, _MASK64
from smolml.models.registry import DecodeState, register_model


@dataclass
class ColumnMixConfig(DeltaMixConfig):
    """:class:`DeltaMixConfig` plus the routed-column knobs.

    ``n_columns`` is ``C`` (``1`` ⇒ identity to ``delta_mix``). ``route_buckets`` is ``B``
    (power of two; the Fibonacci-hash route partition). ``route_order`` is how many recent bytes the
    route bucket hashes. ``gate_lr`` is the per-arm bandit EMA step (``0`` ⇒ frozen gate values);
    ``route_epsilon`` the ε-greedy exploration prob. ``gate_init_other`` is the init reward for
    non-prior arms (below the worst possible reward ``−8``, so an unvisited sibling never wins the
    argmax without evidence). ``seed`` seeds the per-stream route RNG. Exactly **one** column is
    active per byte (``route_top_m`` is fixed at 1; ``m > 1`` soft routing is future work).
    """

    n_columns: int = 1
    route_buckets: int = 1 << 12
    route_order: int = 2
    gate_lr: float = 0.1
    route_epsilon: float = 0.05
    gate_init_other: float = -10.0
    seed: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_columns < 1:
            raise ValueError(f"n_columns must be >= 1, got {self.n_columns}")
        if self.route_buckets < 2 or (self.route_buckets & (self.route_buckets - 1)) != 0:
            raise ValueError(f"route_buckets must be a power of two >= 2, got {self.route_buckets}")
        if not 1 <= self.route_order <= 8:
            raise ValueError(f"route_order must be in [1, 8], got {self.route_order}")
        if self.gate_lr < 0.0:
            raise ValueError(f"gate_lr must be >= 0, got {self.gate_lr}")
        if not 0.0 <= self.route_epsilon < 1.0:
            raise ValueError(f"route_epsilon must be in [0, 1), got {self.route_epsilon}")


@register_model("column_mix")
class ColumnMix(DeltaMix):
    """Warmed hashed context-mixing + one online delta stream routed across ``C`` columns.

    Inherits the warm prior→eval handoff, the hashed count store, the logistic mixing, the online
    mixer-SGD, the delta feature map / delta write, and **all** count- and delta-side FLOP
    accounting; adds the column array + the router. The column path is active only when
    ``n_columns > 1 and delta_orders``; otherwise every override delegates to :class:`DeltaMix`.
    """

    config: ColumnMixConfig

    def __init__(self, config: ColumnMixConfig) -> None:
        super().__init__(config)
        # bucket index = top this-many bits of the Fibonacci hash.
        self._route_bits: int = config.route_buckets.bit_length() - 1
        # The route grid is live only with >1 column AND a delta key to route across.
        self._columns_on: bool = config.n_columns > 1 and bool(config.delta_orders)
        # Widen the window for the route context only when the column path is on, so the inherited
        # context_window stays bit-identical to delta_mix at C == 1.
        if self._columns_on:
            self._window_cap = max(self._window_cap, config.route_order)

    # --- router (fixed hash bucket + per-arm bandit choice) -------------------------

    def _route_slot(self, window: list[int]) -> int:
        """Fibonacci hash of the last ``route_order`` bytes (the available prefix when shorter; the
        empty prefix at pos 0 → bucket 0) to the top ``_route_bits`` bits. Deterministic / salt-free
        so warmed runs reproduce."""
        ctx = bytes(window[-self.config.route_order :])
        x = int.from_bytes(ctx, "little")
        return ((x * _KNUTH) & _MASK64) >> (64 - self._route_bits)

    def _fresh_gate(self) -> np.ndarray:
        """``(B, C)`` per-arm reward table init to the hash prior: ``gate[b, b mod C] = 0`` (the
        prior arm), others ``gate_init_other`` (< the worst reward ``−8``), so cold ``argmax`` is
        the deterministic ``b mod C`` route and an unvisited sibling is reached only via ε."""
        cfg = self.config
        g = np.full((cfg.route_buckets, cfg.n_columns), cfg.gate_init_other)
        buckets = np.arange(cfg.route_buckets)
        g[buckets, buckets % cfg.n_columns] = 0.0
        return g

    def _route_choose(self, ms: _MixerState, bucket: int) -> int:
        """ε-greedy column choice: with prob ``route_epsilon`` a uniform column from the per-stream
        RNG, else ``argmax gate[bucket, :]``."""
        cfg = self.config
        if cfg.route_epsilon > 0.0 and ms.rng.random() < cfg.route_epsilon:
            return int(ms.rng.integers(cfg.n_columns))
        return int(np.argmax(ms.gate[bucket]))

    # --- deferred local updates (chosen column delta + per-arm gate bandit) ---------

    def _apply_column_update(
        self, ms: _MixerState, col: int, pidx: np.ndarray, psign: np.ndarray, revealed_byte: int
    ) -> None:
        """Error-correcting delta (LMS) write on the chosen column's previous key — the bar's exact
        rule applied to ``Wcols[col]`` (a view, so ``np.add.at`` accumulates colliding buckets in
        place exactly as the read sums them)."""
        err = ms.last_p_delta.copy()
        err[revealed_byte] -= 1.0
        scaled = self.config.delta_eta * err
        np.add.at(ms.Wcols[col], (slice(None), pidx), -np.outer(scaled, psign))

    def _apply_gate_update(self, ms: _MixerState, revealed_byte: int) -> None:
        """Per-arm contextual-bandit EMA of the chosen column toward its observed reward
        ``r = log2 p_chosen[revealed]`` (= −bits): ``gate[b,c] += gate_lr·(r − gate[b,c])``.
        Only the chosen arm updates. No-op when ``gate_lr == 0`` (frozen gate)."""
        cfg = self.config
        if cfg.gate_lr <= 0.0:
            return
        reward = float(np.log2(max(float(ms.last_p_delta[revealed_byte]), 1e-300)))
        b, c = ms.last_bucket, ms.last_route
        ms.gate[b, c] += cfg.gate_lr * (reward - ms.gate[b, c])

    # --- prequential step (fold counts + adapt columns/gate + route + predict) ------

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        """Fold the byte, adapt the mixer + the chosen column + the gate, route, predict.

        Delegates entirely to the parent (``delta_mix`` then ``hashed_mix``) when the column path
        is off, preserving the degenerate identities."""
        if not self._columns_on:
            return super().step(state, revealed_byte, pos)
        cfg = self.config
        ms: _MixerState = state.cache
        v = cfg.vocab_size
        window = state.tokens
        k_pred = self.num_predictors

        # 1. Online mixer update on the just-revealed byte (graded prediction).
        did_update = ms.last_probs is not None
        if did_update:
            grad = mixer_gradient(ms.last_probs, ms.last_stretched, revealed_byte)
            ms.weights -= cfg.lr * grad

        # 2. Chosen-column delta update on the PREVIOUS key (same pending gate as the mixer).
        nd_prev = 0
        if did_update:
            pidx, psign = ms.last_phi
            nd_prev = int(pidx.shape[0])
            if nd_prev:
                self._apply_column_update(ms, ms.last_route, pidx, psign, revealed_byte)

        # 3. Per-arm gate bandit update on the PREVIOUS bucket/route.
        if did_update:
            self._apply_gate_update(ms, revealed_byte)

        # 4. Fold the revealed byte into each available order-k count table (parent stores).
        n_fold = 0
        for k in range(k_pred):
            if k == 0 or len(window) >= k:
                n_fold += 1
                key = bytes(window[-k:]) if k else b""
                self._fold_one(ms.tables, k, key, revealed_byte)

        # 5. New context window + the count predictions.
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

        # 6. Route → read the ONE active column → the (K+1)-th raw-logit row.
        bucket = self._route_slot(new_window)
        col = self._route_choose(ms, bucket)
        idxs, signs = self._build_phi(new_window)
        nd = int(idxs.shape[0])
        z_delta = (ms.Wcols[col][:, idxs] * signs[None, :]).sum(axis=1) if nd else np.zeros(v)
        stretched[k_pred] = z_delta

        # 7. Mix, predict, stash the pending prediction (mixer + column + route).
        z = mix_logits(stretched, ms.weights)
        probs = softmax(z)
        ms.last_stretched = stretched
        ms.last_probs = probs
        ms.last_phi = (idxs, signs)
        ms.last_p_delta = softmax(z_delta)
        ms.last_bucket = bucket
        ms.last_route = col

        next_logits = torch.from_numpy(z.astype(np.float32))
        new_state = DecodeState(tokens=new_window, cache=ms, length=state.length + 1)
        flops = self._delta_flop_breakdown(
            did_update=did_update,
            n_fold=n_fold,
            n_active=n_active,
            n_laplace=n_laplace,
            nd=nd,
            nd_prev=nd_prev,
        ) + self._route_increment(did_update=did_update)
        return new_state, next_logits, flops

    # --- FLOP accounting (parent delta charge + the exact route increment) ----------

    def _route_increment(self, *, did_update: bool) -> FlopBreakdown:
        """The router's exact per-byte FLOPs on top of the parent delta breakdown.

        Forward (always): route hash (``3``) + ``gather(1)`` gate row of ``C`` + ``pointwise(C)``
        argmax. Backward (when a prediction was pending AND the gate learns): the bandit update —
        reward ``log2 p[revealed]`` (``2``) + EMA step (``3``) = ``pointwise(5)``. The ε-greedy PRNG
        draw is non-arithmetic and omitted; on an ε byte the gather+argmax above is conservatively
        charged though the random branch skips it (an overcharge, never a subsidy)."""
        c = self.config.n_columns
        forward = gather_flops(1) + pointwise_flops(3 + c)
        backward = 0
        if did_update and self.config.gate_lr > 0.0:
            backward = pointwise_flops(5)
        return FlopBreakdown(forward=forward, backward=backward)

    def _steady_step_flops(self) -> FlopBreakdown:
        """Analytic steady-state per-byte cost: the parent (delta) estimate plus the route
        increment. Guarded so ``C == 1`` / ``delta_orders == ()`` delegate (bit-identical analytic
        ``flops`` / ``decode_step_flops`` to ``delta_mix``, and pretrain-budget parity)."""
        if not self._columns_on:
            return super()._steady_step_flops()
        return super()._steady_step_flops() + self._route_increment(did_update=True)

    # --- warmed state + leak-free eval handoff (carry Wcols/gate alongside counts) --

    def _fresh_cache(self) -> _MixerState:
        """A fresh online state sized for the routed sheet: ``K+1`` mixer weights, a zeroed
        ``(C, V, d)`` column stack, the hash-prior gate, and a fresh per-stream RNG. Delegates to
        the parent when the column path is off."""
        if not self._columns_on:
            return super()._fresh_cache()
        cfg = self.config
        n = self.num_predictors + 1
        return _MixerState(
            tables=self._new_tables(),
            weights=np.full(n, 1.0 / n),
            Wcols=np.zeros((cfg.n_columns, cfg.vocab_size, cfg.delta_dim)),
            gate=self._fresh_gate(),
            rng=np.random.default_rng(cfg.seed),
        )

    def _ensure_warm(self) -> _MixerState:
        """Lazily create the persistent warm state with the routed-sheet fields (delegates to the
        parent when the column path is off)."""
        if not self._columns_on:
            return super()._ensure_warm()
        if self._warm is None:
            self._warm = self._fresh_cache()
        return self._warm

    def init_prequential_state(self) -> DecodeState:
        """A deep copy of the warm state for one eval stream — count store, mixer weights, ``Wcols``
        AND ``gate`` — plus a **fresh** per-stream RNG (never copied, so streams are deterministic
        and isolated). Leak-free: eval mutates only its own copies."""
        if not self._columns_on:
            return super().init_prequential_state()
        warm = self._ensure_warm()
        cache = _MixerState(
            tables=self._copy_tables(warm.tables),
            weights=warm.weights.copy(),
            Wcols=warm.Wcols.copy(),
            gate=warm.gate.copy(),
            rng=np.random.default_rng(self.config.seed),
        )
        return DecodeState(tokens=[], cache=cache)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ColumnMix":
        """Build from a config dict, keeping only :class:`ColumnMixConfig` fields (so the
        harness-injected transformer keys are ignored, like the parent)."""
        fields = {f.name for f in dataclasses.fields(ColumnMixConfig)}
        kwargs = {key: val for key, val in config.items() if key in fields}
        return cls(ColumnMixConfig(**kwargs))
