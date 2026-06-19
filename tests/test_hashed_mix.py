"""Bounded (hashed) count tables (Task B.3): the high-order count store swaps a growable
dict for a fixed-size hashed table, while everything else (mixing, online mixer-SGD, the
warm prior->eval handoff, prequential ``step`` semantics, FLOP accounting) is inherited
from ``warm_mix`` / ``context_mixing`` unchanged.

The contract this pins down:
- ``hash_min_order > max_order`` => no order hashed => bit-identical to ``warm_mix``
  (predictions AND ``FlopBreakdown``);
- a large table is collision-free on a tiny fixture and then reproduces the exact-dict
  predictions byte-for-byte;
- memory is BOUNDED: the hashed table's slot count is constant regardless of how many
  bytes / distinct contexts are folded (a short vs a 100x-longer, far more diverse stream
  leave the table shape unchanged), while the exact dict store grows;
- the per-byte FLOP charge equals the exact-store charge plus exactly the documented hash
  add-on (nothing hidden) in the collision-free case;
- the hash is salt-free, so warmed runs are reproducible; and the model registers.
"""

import numpy as np
import pytest
import torch

from smolml.data import synthetic_text8
from smolml.flops import pointwise_flops
from smolml.models import build_model, list_models
from smolml.models.hashed_mix import (
    _KNUTH,
    _MASK64,
    _UINT16_MAX,
    HashedMix,
    HashedMixConfig,
    _HashTable,
)
from smolml.prequential import prequential_bpb, pretrain

CPU = torch.device("cpu")


def _warm(model, prior, *, flop_budget, seq_len=64, batch_size=4, seed=7) -> int:
    """Warm ``model`` on ``prior`` through the real pretrain budget loop (the warmup)."""
    return pretrain(
        model,
        prior,
        flop_budget=flop_budget,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=0.02,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        seed=seed,
        device=CPU,
    )


def _active_count(window_len: int, num_predictors: int) -> int:
    """Orders that activate for a context window of ``window_len`` bytes — order 0 always,
    order k>=1 iff the window is at least k long (mirrors ``step``'s contiguous prefix)."""
    return sum(1 for k in range(num_predictors) if k == 0 or window_len >= k)


def _fold_stream(model, stream):
    """Fold ``stream`` byte by byte through the real ``step`` channel; return the state."""
    state = model.init_prequential_state()
    for pos in range(len(stream)):
        state, _logits, _flops = model.step(state, int(stream[pos]), pos)
    return state


# --- registration ------------------------------------------------------------
def test_registered_and_buildable():
    assert "hashed_mix" in list_models()
    model = build_model("hashed_mix", {"max_order": 5, "hash_min_order": 4, "table_bits": 12})
    assert isinstance(model, HashedMix)
    assert model.num_params() == 0  # transductive: no gradient parameters


def test_from_config_admits_fields_and_ignores_transformer_keys():
    # The CLI injects transformer keys; from_config keeps only HashedMixConfig fields.
    model = HashedMix.from_config(
        {"max_order": 5, "hash_min_order": 4, "table_bits": 18, "d_model": 128, "n_layers": 4}
    )
    assert model.config.table_bits == 18
    assert model.config.hash_min_order == 4
    assert model.num_predictors == 6


def test_config_validation():
    with pytest.raises(ValueError):
        HashedMixConfig(table_bits=0)
    with pytest.raises(ValueError):
        HashedMixConfig(hash_min_order=-1)
    # inherited ContextMixingConfig validation still runs through super().__post_init__.
    with pytest.raises(ValueError):
        HashedMixConfig(max_order=-1)
    with pytest.raises(ValueError):
        HashedMixConfig(alpha=0.0)


# --- (a) no order hashed == fixed-order warm_mix, bit for bit -----------------
def test_no_order_hashed_is_bit_identical_to_warm_mix():
    # hash_min_order (4) > max_order (3): every order uses the exact dict store and the hash
    # add-on is zero, so a warmed hashed_mix is byte-for-byte warm_mix — same warm state,
    # same predictions, same FlopBreakdown on every eval step.
    prior = synthetic_text8(4000, seed=1).data
    eval_stream = synthetic_text8(1500, seed=2).prequential_carve(eval_bytes=300)[1]
    cfg = {"max_order": 3, "alpha": 0.5, "lr": 0.02}

    warm = build_model("warm_mix", cfg)
    _warm(warm, prior, flop_budget=2e7)
    hashed = build_model("hashed_mix", {**cfg, "hash_min_order": 4, "table_bits": 16})
    _warm(hashed, prior, flop_budget=2e7)

    assert all(isinstance(t, dict) for t in hashed._warm.tables)  # nothing hashed

    ws = warm.init_prequential_state()
    hs = hashed.init_prequential_state()
    for pos in range(len(eval_stream) - 1):
        b = int(eval_stream[pos])
        ws, wl, wf = warm.step(ws, b, pos)
        hs, hl, hf = hashed.step(hs, b, pos)
        assert torch.equal(wl, hl)
        assert wf == hf


# --- (b) collision-free reproduces the exact-dict predictions -----------------
def test_collision_free_reproduces_exact_dict_predictions():
    # A repeating range gives only ``period`` distinct k-grams; a 2**16-slot table hashes
    # them without collision (asserted below), so each hashed lookup returns the SAME counts
    # as the exact dict -> bit-identical predictions byte-for-byte.
    stream = np.array(list(range(20)) * 6, dtype=np.uint8)  # 20 distinct contexts / order
    cfg = {"max_order": 3, "alpha": 0.5, "lr": 0.02}
    exact = build_model("hashed_mix", {**cfg, "hash_min_order": 4, "table_bits": 16})  # all dict
    hashed = build_model("hashed_mix", {**cfg, "hash_min_order": 2, "table_bits": 16})  # 2,3 hashed

    es = exact.init_prequential_state()
    hs = hashed.init_prequential_state()
    for pos in range(len(stream)):
        b = int(stream[pos])
        es, el, _ = exact.step(es, b, pos)
        hs, hl, _ = hashed.step(hs, b, pos)
        assert torch.equal(el, hl)  # same logits, bit for bit

    # Collision-free precondition: every distinct context occupies its own slot.
    for k in range(hashed.config.hash_min_order, hashed.num_predictors):
        n_distinct = len(es.cache.tables[k])  # exact dict: one key per distinct context
        n_occupied = int(hs.cache.tables[k].occupied.sum())
        assert n_occupied == n_distinct


# --- (c) memory is bounded, independent of #bytes and #distinct contexts ------
def test_memory_is_bounded_regardless_of_distinct_contexts():
    cfg = {"max_order": 3, "hash_min_order": 2, "table_bits": 14}
    table_t = 1 << 14
    short = build_model("hashed_mix", cfg)
    long = build_model("hashed_mix", cfg)

    short_stream = np.array(list(range(20)) * 4, dtype=np.uint8)  # ~20 distinct contexts
    long_stream = np.random.default_rng(0).integers(0, 256, size=8000, dtype=np.uint8)  # many

    ss = _fold_stream(short, short_stream)
    ls = _fold_stream(long, long_stream)

    for k in range(cfg["hash_min_order"], short.num_predictors):
        assert isinstance(ss.cache.tables[k], _HashTable)
        # Identical slot count after a short vs a 100x-longer, far more diverse stream:
        # memory does NOT grow with #bytes or #distinct contexts (the whole point).
        assert ss.cache.tables[k].counts.shape == (table_t, 256)
        assert ls.cache.tables[k].counts.shape == (table_t, 256)
        assert ss.cache.tables[k].counts.nbytes == ls.cache.tables[k].counts.nbytes
        assert ls.cache.tables[k].occupied.shape == (table_t,)
        # More distinct contexts DO fill more slots, but never add slots.
        assert ls.cache.tables[k].occupied.sum() > ss.cache.tables[k].occupied.sum()

    # Contrast: the low-order *dict* store grows with #distinct contexts (the OOM hazard).
    assert len(ls.cache.tables[1]) > len(ss.cache.tables[1])


# --- (d) per-byte FLOP charge == exact-store charge + the documented hash add-on
def test_collision_free_flop_charge_matches_exact_plus_hash():
    # In the collision-free case the hashed orders see the SAME counts and the SAME
    # seen/unseen pattern as the dict, so gather/Laplace/mix/update charges are identical;
    # the ONLY difference is the explicit slot arithmetic (_HASH_FLOPS_PER_ORDER pointwise
    # ops per hashed order, lookups in forward, folds in backward).
    stream = np.array(list(range(20)) * 6, dtype=np.uint8)
    cfg = {"max_order": 3, "alpha": 0.5, "lr": 0.02}
    exact = build_model("hashed_mix", {**cfg, "hash_min_order": 4, "table_bits": 16})  # all dict
    hashed = build_model("hashed_mix", {**cfg, "hash_min_order": 2, "table_bits": 16})  # 2,3 hashed

    k_pred = hashed.num_predictors
    hmo = hashed.config.hash_min_order
    hops = HashedMix._HASH_FLOPS_PER_ORDER
    max_order = cfg["max_order"]

    es = exact.init_prequential_state()
    hs = hashed.init_prequential_state()
    total_added = 0
    for pos in range(len(stream)):
        b = int(stream[pos])
        es, _, ef = exact.step(es, b, pos)
        hs, _, hf = hashed.step(hs, b, pos)
        # n_fold / n_active are determined by the (data-independent) window lengths.
        n_fold = _active_count(min(pos, max_order), k_pred)
        n_active = _active_count(min(pos + 1, max_order), k_pred)
        exp_fwd = pointwise_flops(hops * max(0, n_active - hmo))
        exp_bwd = pointwise_flops(hops * max(0, n_fold - hmo))
        # exact-store charge + ONLY the documented hash add-on (no hidden compute):
        assert hf.forward == ef.forward + exp_fwd
        assert hf.backward == ef.backward + exp_bwd
        total_added += exp_fwd + exp_bwd
    # The hash add-on is zero before any order >= hash_min_order is active (early bytes),
    # but the hashed orders DO fire over the stream, so real, non-zero hash work is charged.
    assert total_added > 0

    for k in range(hmo, k_pred):  # the parity argument only holds collision-free
        assert int(hs.cache.tables[k].occupied.sum()) == len(es.cache.tables[k])


# --- no leakage: eval folds into its own deep copy of the hashed arrays -------
def test_eval_does_not_mutate_warm_hashed_tables():
    # The warm->eval handoff deep-copies the _HashTable arrays (counts AND occupancy), so a
    # full eval pass leaves the persistent warm tables byte-for-byte unchanged — no eval byte
    # leaks into self._warm, and concurrent eval streams stay isolated.
    cfg = {"max_order": 5, "hash_min_order": 4, "table_bits": 12}
    warm = build_model("hashed_mix", cfg)
    _warm(warm, synthetic_text8(3000, seed=3).data, flop_budget=2e7)
    before = {
        k: (t.counts.copy(), t.occupied.copy())
        for k, t in enumerate(warm._warm.tables)
        if isinstance(t, _HashTable)
    }
    assert before and any(occ.any() for _, occ in before.values())  # warmup actually filled them
    stream = synthetic_text8(1500, seed=9).prequential_carve(eval_bytes=400)[1]
    prequential_bpb(warm, stream, device=CPU)
    for k, (counts0, occ0) in before.items():
        assert np.array_equal(warm._warm.tables[k].counts, counts0)
        assert np.array_equal(warm._warm.tables[k].occupied, occ0)


# --- (e) reproducibility + salt-free hashing ----------------------------------
def test_reproducible_under_fixed_seed():
    # Orders 4,5 are genuinely hashed here; two warmed runs must match exactly, which
    # requires a deterministic (salt-free) hash and a deterministic warmup/eval.
    cfg = {"max_order": 5, "hash_min_order": 4, "table_bits": 14}
    prior = synthetic_text8(3000, seed=4).data
    eval_stream = synthetic_text8(1500, seed=5).prequential_carve(eval_bytes=300)[1]

    def warmed_run():
        m = build_model("hashed_mix", cfg)
        _warm(m, prior, flop_budget=2e7)
        return prequential_bpb(m, eval_stream, device=CPU)

    r1 = warmed_run()
    r2 = warmed_run()
    assert r1.bpb == r2.bpb
    assert r1.eval_flops == r2.eval_flops


def test_slot_is_salt_free_multiplicative_hash():
    # The slot is the salt-free Fibonacci multiplicative hash (NOT Python's per-process
    # salted hash(bytes)) — pinned to the exact formula so a regression to a salted hash
    # (which would break cross-run reproducibility) fails here.
    model = build_model("hashed_mix", {"max_order": 4, "hash_min_order": 4, "table_bits": 20})
    ctx = b"\x01\x02\x03\x04"
    x = int.from_bytes(ctx, "little")
    expected = ((x * _KNUTH) & _MASK64) >> (64 - 20)
    assert model._slot(ctx) == expected
    assert model._slot(ctx) == model._slot(ctx)  # deterministic
    assert 0 <= expected < (1 << 20)


def test_halve_on_overflow_is_charged():
    # Force a slot to the uint16 ceiling so the next fold there halves the 256-wide row (O(V));
    # that halve must be charged — the step's backward FLOPs exceed an identical non-overflow
    # fold by exactly pointwise(V).
    m = HashedMix(HashedMixConfig(max_order=0, hash_min_order=0, table_bits=4))
    v = m.config.vocab_size
    base = m.init_prequential_state()
    _, _, f_normal = m.step(base, 7, 0)
    over = m.init_prequential_state()
    slot = m._slot(b"")
    over.cache.tables[0].occupied[slot] = True
    over.cache.tables[0].counts[slot, 7] = _UINT16_MAX
    _, _, f_over = m.step(over, 7, 0)
    assert f_over.backward == f_normal.backward + pointwise_flops(v)


def test_hashed_order_over_8_bytes_rejected():
    # _slot hashes only the low 8 context bytes, so a hashed order > 8 bytes would alias; reject.
    with pytest.raises(ValueError):
        HashedMixConfig(max_order=10, hash_min_order=4)
    # a high max_order with NO order hashed (hash_min_order > max_order) is allowed.
    HashedMixConfig(max_order=10, hash_min_order=11)
