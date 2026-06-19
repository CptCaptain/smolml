"""Gated order escalation on the warmed backbone (Task B.2, Phase 2).

``gated_mix`` is :class:`WarmMix` with one new idea — the prediction no longer
pays for *every* order on *every* byte. It still **folds** the revealed byte into
all available order tables (so the high-order statistics stay complete) and warms
exactly like the parent, but when it **predicts** it escalates orders
cheapest-first: always the floor ``0..min_order``, then one higher order at a
time, re-mixing, and **stops** as soon as the running partial mix is confident
enough (a pre-reveal ``1 - max p`` gate below ``gate_threshold``), no higher
already-seen context exists, or ``max_order`` is reached. Only the orders actually
evaluated — plus the per-step gate arithmetic — are charged, so the per-byte cost
is dynamic and *strictly* the work done. This is the A and C fusion: the expensive
high orders fire only on high-surprise bytes.

Everything load-bearing is inherited: the fold (step 1 below), the online
mixer-SGD, the Laplace/stretch/mix/softmax primitives, the warmup budget loop and
the deep-copy eval handoff all come from :class:`WarmMix` / :class:`ContextMixing`
unchanged. The only overrides are :meth:`step` (the escalation loop + its honest
dynamic charge), :meth:`_flop_breakdown` / :meth:`_steady_step_flops` (extended to
the evaluated-order counts), and :meth:`from_config` (to admit the two new fields).

**Degenerate identity.** With ``min_order == max_order`` the floor already covers
every order, the escalation loop never runs, the gate is never computed, and one
mix + one softmax are charged — so ``gated_mix`` is then **bit-identical** to
fixed-order ``warm_mix`` (same predictions *and* same :class:`FlopBreakdown`). A
degenerate ``gate_threshold == 0`` (the ``1 - max p`` gate can never fire) forces
escalation to the deepest already-seen order every byte, but is **not** identical to
``warm_mix`` when ``min_order < max_order``: the charge is lower (it skips the unseen /
short-context "dead" orders ``warm_mix`` still evaluates as abstaining) *and* the
predictions drift, because the online mixer here grades only the *evaluated* prefix, so
its weight trajectory diverges from ``warm_mix``'s all-orders update. ``min_order ==
max_order`` is therefore the *unique* bit-identical config — the only one whose every
step evaluates all ``K`` orders and updates all ``K`` mixer weights, exactly like
``warm_mix`` (see the module tests).

**No leakage.** The gate reads only the partial-mix prediction (a function of the
already-revealed bytes), never the byte being predicted.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch

from smolml.flops import FlopBreakdown, gather_flops, pointwise_flops
from smolml.models.context_mixing import (
    ContextMixingConfig,
    _MixerState,
    laplace_prob,
    mix_logits,
    mixer_gradient,
    softmax,
)
from smolml.models.registry import DecodeState, register_model
from smolml.models.warm_mix import WarmMix


@dataclass
class GatedMixConfig(ContextMixingConfig):
    """:class:`ContextMixingConfig` plus the two gated-escalation knobs.

    ``min_order`` is the always-evaluated floor (orders ``0..min_order`` run every
    byte, like the parent); ``gate_threshold`` is the ``1 - max p`` confidence
    cutoff that stops escalation early — ``0`` never fires (full escalation),
    larger values stop sooner. ``max_order`` / ``alpha`` / ``lr`` / ``vocab_size``
    are reused unchanged.
    """

    min_order: int = 1
    gate_threshold: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.min_order <= self.max_order:
            raise ValueError(
                f"min_order must be in [0, max_order={self.max_order}], got {self.min_order}"
            )
        if self.gate_threshold < 0.0:
            raise ValueError(f"gate_threshold must be >= 0, got {self.gate_threshold}")


@register_model("gated_mix")
class GatedMix(WarmMix):
    """Warmed context-mixing with pre-reveal gated order escalation.

    Inherits the warm state, warmup loop and eval handoff from :class:`WarmMix`;
    overrides only the per-byte prediction (:meth:`step`) and its FLOP accounting.
    """

    config: GatedMixConfig

    def _stretch(
        self, ms: _MixerState, new_window: list[int], k: int
    ) -> tuple[np.ndarray, bool, bool]:
        """Stretched (log-prob) row for order ``k`` over ``new_window``.

        Returns ``(row, active, seen)``: ``active`` if the context is long enough
        (a real table lookup happened, charged as one gather), ``seen`` if that
        context already had counts (Laplace applies; otherwise the order abstains
        with the uniform row, exactly like the parent).
        """
        active = k == 0 or len(new_window) >= k
        cell = ms.tables[k].get(bytes(new_window[-k:]) if k else b"") if active else None
        prob = laplace_prob(cell, self.config.alpha) if cell is not None else self._uniform
        return np.log(prob), active, cell is not None

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        """Fold the byte into all orders, grade the previous prediction, then
        predict by gated order escalation. Every executed branch is charged."""
        cfg = self.config
        ms: _MixerState = state.cache
        v = cfg.vocab_size
        window = state.tokens

        # 1. Online mixer update over the orders that produced the previous
        #    prediction (always a contiguous prefix 0..last_depth), graded on the
        #    byte just revealed. Skipped on the first byte (no pending prediction).
        did_update = ms.last_probs is not None
        n_eval_prev = 0
        if did_update:
            n_eval_prev = ms.last_stretched.shape[0]
            grad = mixer_gradient(ms.last_probs, ms.last_stretched, revealed_byte)
            ms.weights[:n_eval_prev] -= cfg.lr * grad

        # 2. Fold the revealed byte into every *available* order table — UNCHANGED
        #    from the parent so the high-order statistics stay complete.
        n_fold = 0
        for k in range(self.num_predictors):
            if k == 0 or len(window) >= k:
                n_fold += 1
                key = bytes(window[-k:]) if k else b""
                cell = ms.tables[k].get(key)
                if cell is None:
                    cell = np.zeros(v)
                    ms.tables[k][key] = cell
                cell[revealed_byte] += 1.0

        # 3. Predict the next byte by escalating orders cheapest-first.
        cap = cfg.max_order
        new_window = [*window, revealed_byte][-cap:] if cap else []

        # --- always-evaluated floor: orders 0..min_order ---
        stretched_rows: list[np.ndarray] = []
        n_active = 0
        n_laplace = 0
        for k in range(cfg.min_order + 1):
            row, active, seen = self._stretch(ms, new_window, k)
            stretched_rows.append(row)
            n_active += active
            n_laplace += seen
        n_eval = cfg.min_order + 1
        # Mix the floor as one matmul: bit-identical to the parent when
        # min_order == max_order; escalation then extends z incrementally so the
        # total mix work stays 2*n_eval*V either way.
        z = mix_logits(np.stack(stretched_rows), ms.weights[:n_eval])

        # --- escalation: add higher orders until confident / exhausted ---
        softmax_count = 0
        gate_flops = 0
        probs: np.ndarray | None = None
        d = cfg.min_order
        while d < cap:
            if cfg.gate_threshold > 0.0:
                probs = softmax(z)
                softmax_count += 1
                gate_flops += v + 1  # max over V + the (1 - max p) subtract
                if 1.0 - float(probs.max()) < cfg.gate_threshold:
                    break  # confident enough -> stop (probs is the prediction)
            nxt = d + 1
            if len(new_window) < nxt:
                break  # no higher context can be formed
            cell = ms.tables[nxt].get(bytes(new_window[-nxt:]))
            n_active += 1  # the escalation probe is a real gather
            if cell is None:
                break  # no higher *already-seen* context -> stop
            row = np.log(laplace_prob(cell, cfg.alpha))
            stretched_rows.append(row)
            n_laplace += 1
            z = z + ms.weights[nxt] * row
            n_eval += 1
            d = nxt
            probs = None  # z changed; any cached gate softmax is stale
        if probs is None:
            probs = softmax(z)
            softmax_count += 1

        ms.last_stretched = np.stack(stretched_rows)
        ms.last_probs = probs
        next_logits = torch.from_numpy(z.astype(np.float32))
        new_state = DecodeState(tokens=new_window, cache=ms, length=state.length + 1)
        flops = self._flop_breakdown(
            did_update=did_update,
            n_eval_prev=n_eval_prev,
            n_fold=n_fold,
            n_active=n_active,
            n_laplace=n_laplace,
            n_eval=n_eval,
            softmax_count=softmax_count,
            gate_flops=gate_flops,
        )
        return new_state, next_logits, flops

    def _flop_breakdown(
        self,
        *,
        did_update: bool,
        n_eval_prev: int,
        n_fold: int,
        n_active: int,
        n_laplace: int,
        n_eval: int,
        softmax_count: int,
        gate_flops: int,
    ) -> FlopBreakdown:
        """Exact FLOPs for the branches :meth:`step` ran, extending the parent's
        per-byte charge to the *evaluated* order count plus the gate arithmetic.

        Prediction (forward): Laplace for the ``n_laplace`` seen evaluated orders
        (``3*V`` each), one stretch (``V``) and one mix increment (``2*V``) per
        evaluated order, ``5*V`` per softmax (one per gate check + the prediction),
        the ``gate_flops`` confidence arithmetic, and ``n_active`` context lookups.
        Adaptation (backward): the unchanged fold (``n_fold`` increments + lookups)
        and, when a prediction was graded, the mixer update over the
        previously-evaluated prefix (``1 + 2*n_eval_prev*V + 2*n_eval_prev``).
        """
        v = self.config.vocab_size
        forward = pointwise_flops(
            3 * v * n_laplace + n_eval * v + 2 * n_eval * v + 5 * v * softmax_count + gate_flops
        )
        forward += gather_flops(n_active)
        backward_pointwise = n_fold
        if did_update:
            backward_pointwise += 1 + 2 * n_eval_prev * v + 2 * n_eval_prev
        backward = pointwise_flops(backward_pointwise) + gather_flops(n_fold)
        return FlopBreakdown(forward=forward, backward=backward)

    def _steady_step_flops(self) -> FlopBreakdown:
        """Worst-case per-byte cost (the budget guard's upper bound): full
        escalation to ``max_order`` with the gate evaluated at every step. This
        bounds — never undercuts — any actual :meth:`step` (fewer evaluated orders,
        fewer softmaxes, no gate when ``gate_threshold == 0``), so the pretrain
        budget loop (which guards on :meth:`flops`) cannot overspend."""
        cfg = self.config
        k = self.num_predictors
        extra = cfg.max_order - cfg.min_order  # escalation steps at full depth
        return self._flop_breakdown(
            did_update=True,
            n_eval_prev=k,
            n_fold=k,
            n_active=k,
            n_laplace=k,
            n_eval=k,
            softmax_count=extra + 1,
            gate_flops=extra * (cfg.vocab_size + 1),
        )

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "GatedMix":
        """Build from a config dict, keeping only :class:`GatedMixConfig` fields
        (so the harness-injected transformer keys are ignored, like the parent)."""
        fields = {f.name for f in dataclasses.fields(GatedMixConfig)}
        kwargs = {key: val for key, val in config.items() if key in fields}
        return cls(GatedMixConfig(**kwargs))
