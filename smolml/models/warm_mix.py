"""Warmed online context-mixing (Task B.2, Phase 1).

``warm_mix`` is the context-mixing reference with exactly one new mechanism: a
**stateful prior->eval handoff**. The cold reference (``context_mixing``) is purely
transductive — it throws away everything it learns on the prior corpus because the
harness hands no prior online state to the eval stream (``docs/harness.md`` §5 defers
this "until a stateful candidate needs it"). ``warm_mix`` *is* that candidate: it folds
the prior corpus into a persistent :class:`_MixerState` during pretraining (the budget
loop's :meth:`train_step` calls), then **deep-copies** that warmed state into every eval
stream so prediction starts from warmed count tables + mixer weights instead of cold
uniform priors.

Everything else is inherited from :class:`ContextMixing` unchanged — the fold, the
logistic mixing, the online mixer-SGD, and (critically) the honest per-byte FLOP
accounting (:meth:`ContextMixing._flop_breakdown`). It is never re-derived here. The only
overrides are the three seams the handoff needs:

- :meth:`train_step` — fold prior windows into the persistent warm state (one independent
  online episode per row), charging the **exact** per-byte cost via :meth:`step`.
- :meth:`flops` — the analytic per-row warmup cost so the budget loop terminates.
- :meth:`init_prequential_state` — a deep copy of the warm state per eval stream.

**No leakage.** Warmup folds only prior bytes (the carve keeps prior and eval structurally
disjoint). Each eval stream folds its bytes into its *own* deep copy, so the persistent
warm state is never mutated by eval and concurrent eval streams never see each other. At
budget 0 the warm state is never warmed, so an un-warmed ``warm_mix`` is **bit-identical**
to the cold reference (asserted in the tests).
"""

import torch

from smolml.flops import FlopBreakdown
from smolml.models.context_mixing import ContextMixing, ContextMixingConfig, _MixerState
from smolml.models.registry import DecodeState, register_model


@register_model("warm_mix")
class WarmMix(ContextMixing):
    """Context-mixing reference with a warmed prior->eval state handoff.

    Reuses every mixing + FLOP-accounting mechanism of :class:`ContextMixing`; adds only a
    persistent warmed :class:`_MixerState` (``self._warm``) that pretraining fills and each
    eval stream deep-copies. ``step`` / ``_flop_breakdown`` / ``from_config`` /
    ``configure_optimizer`` are inherited unchanged.
    """

    def __init__(self, config: ContextMixingConfig) -> None:
        super().__init__(config)
        # Persistent warmed state (count tables + mixer weights), created lazily on first
        # use so that — until something warms it — it is shape-identical to a fresh
        # ContextMixing state (=> budget-0 is bit-identical to the cold reference).
        self._warm: _MixerState | None = None

    # --- persistent warm state -------------------------------------------------------

    def _ensure_warm(self) -> _MixerState:
        """Lazily create the persistent warm state, identical to a fresh
        :meth:`ContextMixing.init_prequential_state` cache (same empty tables, same
        ``1/K`` mixer weights), so an un-warmed ``warm_mix`` matches the cold reference."""
        if self._warm is None:
            self._warm = super().init_prequential_state().cache
        return self._warm

    # --- warmup (pretraining) seam ---------------------------------------------------

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        *,
        grad_clip: float = 1.0,
    ) -> tuple[torch.Tensor, FlopBreakdown]:
        """Warm the persistent state on a batch of prior windows.

        Each row of ``x`` is folded **in order** into ``self._warm`` as one independent
        online episode: the rolling context window and the pending-prediction cache reset
        per row, while the count tables and mixer weights persist and accumulate across
        rows and across pretrain steps (that accumulation *is* the warming). Folding reuses
        :meth:`ContextMixing.step` verbatim — same fold, same online mixer-SGD, same exact
        per-byte FLOP charge — and the returned :class:`FlopBreakdown` is the honest sum
        over every byte folded. ``optimizer`` is unused (the mixer updates itself), and the
        returned loss is a placeholder zero: the budget loop ignores it, and recomputing a real
        warmup cross-entropy here would be uncharged diagnostic compute (dropped for honesty).
        """
        warm = self._ensure_warm()
        x, _y = batch
        rows = x.detach().cpu().numpy()
        n_rows, seq_len = rows.shape
        spent = FlopBreakdown()
        for r in range(n_rows):
            row = rows[r]
            # Context reset per row: fresh window + no pending prediction, but the SAME
            # persistent tables + weights (the warm carry-over) threaded through step.
            warm.last_stretched = None
            warm.last_probs = None
            state = DecodeState(tokens=[], cache=warm)
            for pos in range(seq_len):
                state, _logits, step_flops = self.step(state, int(row[pos]), pos)
                spent += step_flops
        return torch.zeros((), dtype=torch.float32), spent

    def flops(self, seq_len: int) -> FlopBreakdown:
        """Honest analytic warmup cost of folding **one** ``seq_len`` row.

        The steady-state per-byte estimate replicated ``seq_len`` times — the same analytic
        upper bound :class:`ContextMixing` uses. It bounds (never undercuts) the exact
        per-byte sum :meth:`train_step` charges: the first ``max_order`` bytes of a row fold
        fewer orders and the very first does no mixer update, so they cost strictly less,
        and every byte's charge is componentwise <= the steady estimate. The pretrain budget
        loop guards on this estimate but accumulates the exact charge, so it always
        terminates without overspending.
        """
        return self._steady_step_flops().scale(seq_len)

    # --- eval handoff seam -----------------------------------------------------------

    def init_prequential_state(self) -> DecodeState:
        """A **deep copy** of the persistent warm state for one eval stream.

        Eval continues from the warmed count tables + mixer weights but folds its own bytes
        into the copy — never the persistent warm state — so eval is leak-free (no eval byte
        touches ``self._warm``; concurrent eval streams are isolated) and reproducible. The
        pending-prediction cache resets (a fresh stream has no pending prediction) and the
        window starts empty. At budget 0 the warm state is a fresh :class:`ContextMixing`
        cache, so this copy is bit-identical to the cold reference's initial state.
        """
        warm = self._ensure_warm()
        tables = [{ctx: counts.copy() for ctx, counts in table.items()} for table in warm.tables]
        cache = _MixerState(tables=tables, weights=warm.weights.copy())
        return DecodeState(tokens=[], cache=cache)
