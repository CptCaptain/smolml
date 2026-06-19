"""Bounded (hashed) count tables on the warmed backbone (Task B.3).

``hashed_mix`` is :class:`WarmMix` with exactly one swap: the **count store** for the
high orders. ``warm_mix``/``gated_mix`` keep one growable Python ``dict`` per order
(context ``bytes`` -> 256-wide count vector); on a full-corpus warmup the high-order
dicts grow without bound (~0.5 new contexts/byte, still climbing at 2 MB) and OOM. The
standard PAQ/cmix fix is a **fixed-size hashed table**: hash the context to a slot in a
pre-allocated array and accept that distinct contexts occasionally share a slot. Memory
is then ``2**table_bits`` slots *regardless of how many contexts are folded* — the whole
point — at the cost of collision noise in the mixed prediction.

Mechanism
---------
- Orders ``k >= hash_min_order`` use a :class:`_HashTable`: a ``(T, 256)`` ``uint16``
  count array (``T = 2**table_bits``) plus a ``(T,)`` occupancy bitmap. ``context ->
  hash(context) % T -> row``; counts halve on ``uint16`` overflow to stay fixed-width.
- Orders ``k < hash_min_order`` keep the **exact dict store** (orders 0..3 are small and
  saturate quickly — bounding them buys nothing and only adds collision noise).
- Everything else is inherited from :class:`WarmMix` / :class:`ContextMixing` **unchanged**:
  the per-byte fold, logistic mixing, the online mixer-SGD, the warm prior->eval handoff
  (deep-copied per stream), and the prequential ``step`` semantics. Only the count
  storage + its lookup/fold seams (:meth:`_new_tables`, :meth:`_copy_tables`,
  :meth:`_fold_one`, :meth:`_lookup_one`) change, plus the FLOP add-on below.

FLOP honesty
------------
A hashed resolve is charged **exactly like the dict gather it replaces**: hashed orders
are counted in the parent's ``gather(n_active)`` / ``gather(n_fold)`` just like dict
orders, and the occupancy "seen" test is O(1) like a dict's membership check (so
``n_laplace`` counts exactly the already-seen orders — Laplace/stretch/mix/softmax/update
costs are untouched). The **only** added charge is the explicit slot arithmetic the dict
does not do: a Fibonacci multiply, a 64-bit mask, and a shift = ``_HASH_FLOPS_PER_ORDER``
pointwise ops per hashed order, per resolve (one fold + one lookup per byte at steady
state). The ``int.from_bytes`` materialization is the O(k) analogue of the dict's own
O(k) key hashing and is bundled into the single gather, exactly as the dict bundles its.
The halve-on-overflow (``row >>= 1``, an O(V) op fired <= once per ``2**16`` increments per
slot) IS charged: ``pointwise_flops(V)`` per halve event, accumulated over the step's folds. The
per-byte charge therefore stays dynamic and is summed by the harness — **no compute hides
in the hashing**.

Degenerate identity
-------------------
With ``hash_min_order > max_order`` no order is hashed: every order uses the exact dict
store and the hash add-on is zero, so ``hashed_mix`` is then **bit-identical** to
fixed-order ``warm_mix`` — same predictions *and* same :class:`FlopBreakdown` (asserted in
the tests; it is also the default, since ``hash_min_order=4 > max_order=3``).

Memory & honest expectation
---------------------------
``T = 2**21`` (2 M) slots × 256 × 2 B = 1 GiB per hashed order (+2 MiB occupancy); bounding
orders 4..6 ⇒ ~3 GiB **fixed**, independent of corpus size. ``table_bits`` is the
memory<->collision knob. At full-corpus order-6 the table is overloaded (tens of millions
of contexts into 2 M slots), so the order-6 advantage may erode toward order-3 — whether
the order-6 win survives bounding is the question the full carve answers, honest either way.

**No leakage.** Eval folds into its own deep copy (:meth:`_copy_tables` ``.copy()``s the
``uint16`` count arrays *and* the occupancy bitmaps), never the persistent warm tables.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from smolml.flops import FlopBreakdown, pointwise_flops
from smolml.models.context_mixing import ContextMixingConfig
from smolml.models.registry import register_model
from smolml.models.warm_mix import WarmMix

# Fibonacci (Knuth multiplicative) hashing: a single 64-bit multiply by 2**64/phi, then
# take the top ``table_bits`` bits. Deterministic across runs/processes (unlike Python's
# salted ``hash(bytes)``), so warmed runs are reproducible.
_KNUTH: int = 0x9E3779B97F4A7C15
_MASK64: int = (1 << 64) - 1
_UINT16_MAX: int = 0xFFFF  # halve a slot's counts before any would exceed this


@dataclass
class _HashTable:
    """A fixed-size hashed count table for one hashed order.

    ``counts[slot]`` is a 256-wide ``uint16`` byte-count vector; ``occupied[slot]`` marks
    slots that have received at least one fold — the O(1) "seen" test mirroring a dict's
    membership check (so an all-unseen slot abstains -> uniform, exactly like a dict miss).
    Both arrays are ``2**table_bits`` long regardless of how many contexts are folded;
    distinct contexts sharing a slot (collisions) are accepted as bounded-memory noise.
    """

    counts: np.ndarray  # (T, 256) uint16
    occupied: np.ndarray  # (T,) bool

    def copy(self) -> "_HashTable":
        """Deep copy (independent arrays) for the leak-free warm->eval handoff."""
        return _HashTable(self.counts.copy(), self.occupied.copy())


@dataclass
class HashedMixConfig(ContextMixingConfig):
    """:class:`ContextMixingConfig` plus the two bounded-table knobs.

    ``table_bits`` sets each hashed order's slot count ``T = 2**table_bits`` (the
    memory<->collision knob); ``hash_min_order`` is the lowest order that is hashed (orders
    below it stay exact dicts). ``max_order`` / ``alpha`` / ``lr`` / ``vocab_size`` and the
    ``WarmMix`` warmup are reused unchanged.
    """

    table_bits: int = 21
    hash_min_order: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.table_bits < 1:
            raise ValueError(f"table_bits must be >= 1, got {self.table_bits}")
        if self.hash_min_order < 0:
            raise ValueError(f"hash_min_order must be >= 0, got {self.hash_min_order}")
        # _slot hashes only the low 8 context bytes (int.from_bytes & _MASK64), so a hashed
        # order > 8 bytes would alias on the ignored bytes. Guard it (the order-6 target is fine).
        if self.hash_min_order <= self.max_order and self.max_order > 8:
            raise ValueError(
                f"hashed orders must be <= 8 bytes (max_order={self.max_order}, "
                f"hash_min_order={self.hash_min_order}); widen _slot to hash all bytes first"
            )


@register_model("hashed_mix")
class HashedMix(WarmMix):
    """Warmed context-mixing with fixed-size hashed count tables for the high orders.

    Inherits the warm state, warmup loop, eval handoff, mixing, online mixer-SGD and FLOP
    accounting from :class:`WarmMix` / :class:`ContextMixing`; overrides only the four
    count-store seams, the hash add-on in :meth:`_flop_breakdown`, and :meth:`from_config`.
    """

    config: HashedMixConfig

    # Explicit slot arithmetic the dict store does not do: a Fibonacci multiply, a 64-bit
    # mask, and a shift = 3 pointwise ops per hashed order per resolve (see module docstring).
    _HASH_FLOPS_PER_ORDER: int = 3

    # --- count-store seams (swap the store for orders >= hash_min_order) ------------

    def _new_tables(self) -> list:
        """Per-order store: an exact ``dict`` for ``k < hash_min_order``, a fixed-size
        :class:`_HashTable` for ``k >= hash_min_order`` (allocated once, never grown)."""
        cfg = self.config
        size = 1 << cfg.table_bits
        tables: list = []
        for k in range(self.num_predictors):
            if k < cfg.hash_min_order:
                tables.append({})
            else:
                tables.append(
                    _HashTable(
                        counts=np.zeros((size, cfg.vocab_size), dtype=np.uint16),
                        occupied=np.zeros(size, dtype=bool),
                    )
                )
        return tables

    def _copy_tables(self, tables: list) -> list:
        """Deep-copy the mixed store: ``.copy()`` the hashed arrays, clone the dicts —
        so an eval stream never mutates the shared warm tables (leak-free handoff)."""
        out: list = []
        for table in tables:
            if isinstance(table, _HashTable):
                out.append(table.copy())
            else:
                out.append({ctx: counts.copy() for ctx, counts in table.items()})
        return out

    def _slot(self, ctx: bytes) -> int:
        """Fibonacci hash of the (fixed-length, per-order) context bytes to a slot in
        ``[0, 2**table_bits)`` — one multiply, one mask, one shift (the charged add-on)."""
        x = int.from_bytes(ctx, "little")
        return ((x * _KNUTH) & _MASK64) >> (64 - self.config.table_bits)

    def _fold_one(self, tables: list, k: int, ctx: bytes, byte: int) -> None:
        """Increment order-``k``'s count for ``byte``. Low orders use the exact dict
        store (parent); hashed orders resolve ``ctx`` to a slot, mark it occupied, and
        increment in place, halving the slot's counts first if one would overflow ``uint16``."""
        if k < self.config.hash_min_order:
            super()._fold_one(tables, k, ctx, byte)
            return
        table: _HashTable = tables[k]
        slot = self._slot(ctx)
        table.occupied[slot] = True
        row = table.counts[slot]
        if row[byte] >= _UINT16_MAX:
            row >>= 1  # halve-on-overflow keeps the table fixed-width (O(V); charged below)
            self._halves = getattr(self, "_halves", 0) + 1
        row[byte] += 1

    def _lookup_one(self, tables: list, k: int, ctx: bytes) -> np.ndarray | None:
        """Order-``k``'s count vector for ``ctx``, or ``None`` if unseen. Low orders use
        the dict store (parent); a hashed order returns its slot's row when that slot is
        occupied (the O(1) "seen" test), else ``None`` so the order abstains -> uniform."""
        if k < self.config.hash_min_order:
            return super()._lookup_one(tables, k, ctx)
        table: _HashTable = tables[k]
        slot = self._slot(ctx)
        if not table.occupied[slot]:
            return None
        return table.counts[slot]

    # --- FLOP add-on (only the explicit hash arithmetic; the rest is the parent's) --

    def _flop_breakdown(
        self, *, did_update: bool, n_fold: int, n_active: int, n_laplace: int
    ) -> FlopBreakdown:
        """The parent's exact per-byte charge plus the explicit slot arithmetic for the
        hashed orders. Folded orders are the contiguous prefix ``{0..n_fold-1}`` and active
        orders ``{0..n_active-1}``, so the hashed ones are those ``>= hash_min_order`` —
        ``max(0, n - hash_min_order)`` of them — charged ``_HASH_FLOPS_PER_ORDER`` pointwise
        ops each (lookups in forward, folds in backward). Gather/Laplace/mix/update costs are
        the parent's, unchanged."""
        base = super()._flop_breakdown(
            did_update=did_update, n_fold=n_fold, n_active=n_active, n_laplace=n_laplace
        )
        hmo = self.config.hash_min_order
        n_hash_lookup = max(0, n_active - hmo)
        n_hash_fold = max(0, n_fold - hmo)
        ops = self._HASH_FLOPS_PER_ORDER
        # Charge the O(V) halve-on-overflow folds that fired this step (counted in _fold_one);
        # reset so the next step starts clean. Amortized-tiny but not free.
        halves = getattr(self, "_halves", 0)
        self._halves = 0
        halve_flops = pointwise_flops(self.config.vocab_size * halves)
        return base + FlopBreakdown(
            forward=pointwise_flops(ops * n_hash_lookup),
            backward=pointwise_flops(ops * n_hash_fold) + halve_flops,
        )

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "HashedMix":
        """Build from a config dict, keeping only :class:`HashedMixConfig` fields (so the
        harness-injected transformer keys are ignored, like the parent)."""
        fields = {f.name for f in dataclasses.fields(HashedMixConfig)}
        kwargs = {key: val for key, val in config.items() if key in fields}
        return cls(HashedMixConfig(**kwargs))
