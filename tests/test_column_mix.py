"""Routed sheet of delta columns (Task B.5): C per-column delta predictors, one active per byte,
chosen by a cheap per-arm contextual-bandit gate over a fixed hash route.

Mirrors ``tests/test_delta_mix.py``. The load-bearing checks:
- ``n_columns=1`` ⇒ bit-identical predictions + per-step/analytic FlopBreakdown to ``delta_mix``;
- ``delta_orders=()`` ⇒ bit-identical to ``hashed_mix`` for any ``C``;
- the FLOP charge == parent delta breakdown + the analytic ``_route_increment`` (charge==code);
- the **selection** lever isolated from capacity: on an interaction source a routed sheet beats a
  single column, and a single column does NOT catch up when its ``delta_dim`` is grown to ``C·d``.
"""

import numpy as np
import pytest
import torch

from smolml.data import synthetic_text8
from smolml.flops import gather_flops, pointwise_flops
from smolml.models import build_model, list_models
from smolml.models.column_mix import ColumnMix, ColumnMixConfig
from smolml.prequential import prequential_bpb, pretrain

CPU = torch.device("cpu")


def _warm(model, prior, *, flop_budget, seq_len=64, batch_size=4, seed=7) -> int:
    """Warm ``model`` on ``prior`` through the real pretrain budget loop."""
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


def _bpb(model, stream) -> float:
    return prequential_bpb(model, stream, device=CPU).bpb


def _interaction_stream(n_blocks: int, seed: int, markers=(100, 101)) -> np.ndarray:
    """Period-6 blocks ``[Rb, 0, 0, 0, 0, T]``: a regime marker ``Rb`` sits at offset −5 from the
    target ``T``, with four constant fillers at offsets −1..−4. With ``max_order<=3`` and
    ``delta_orders`` reaching ≤ offset −3, neither counts nor ``φ`` can see ``Rb`` — only the
    ``route_order=5`` router can. ``T = 1 + regime``, so the SAME context ``[0,0,0,0]`` maps to
    a DIFFERENT target by regime: a multiplicative interaction a single linear-in-``φ`` ``W`` cannot
    represent at ANY ``delta_dim``, but routed columns can (each column learns its regime's target).
    """
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for _ in range(n_blocks):
        r = int(rng.integers(0, 2))
        out += [markers[r], 0, 0, 0, 0, 1 + r]
    return np.array(out, dtype=np.uint8)


# --- registration ------------------------------------------------------------
def test_registered_and_buildable():
    assert "column_mix" in list_models()
    m = build_model("column_mix", {"n_columns": 4})
    assert isinstance(m, ColumnMix)
    assert m.num_params() == 0  # transductive: Wcols/gate are online state, not nn.Parameters


def test_from_config_admits_fields_and_ignores_transformer_keys():
    m = ColumnMix.from_config(
        {
            "n_columns": 8,
            "route_buckets": 1 << 10,
            "route_order": 4,
            "gate_lr": 0.2,
            "route_epsilon": 0.1,
            "gate_init_other": -5.0,
            "seed": 3,
            "delta_orders": [3, 4, 5],  # JSON stores a list; must coerce to a tuple
            "delta_dim": 1 << 12,
            "d_model": 256,  # harness-injected transformer key, ignored
            "n_layers": 6,
        }
    )
    assert m.config.n_columns == 8
    assert m.config.route_buckets == 1 << 10
    assert m.config.route_order == 4
    assert m.config.delta_orders == (3, 4, 5)  # coerced
    assert m.config.delta_dim == 1 << 12


def test_config_validation():
    with pytest.raises(ValueError):
        ColumnMixConfig(n_columns=0)
    with pytest.raises(ValueError):
        ColumnMixConfig(route_buckets=3)  # not a power of two
    with pytest.raises(ValueError):
        ColumnMixConfig(route_order=0)
    with pytest.raises(ValueError):
        ColumnMixConfig(route_order=9)
    with pytest.raises(ValueError):
        ColumnMixConfig(gate_lr=-0.1)
    with pytest.raises(ValueError):
        ColumnMixConfig(route_epsilon=1.0)
    with pytest.raises(ValueError):
        ColumnMixConfig(route_epsilon=-0.1)
    ColumnMixConfig(n_columns=1)  # the degenerate single-column config is valid


# --- degenerate identity: C=1 == delta_mix, bit for bit -----------------------
def test_c1_bit_identical_to_delta_mix():
    # n_columns=1 => the column path is off and every override delegates to DeltaMix: same warm
    # state, same predictions, same per-step AND analytic FlopBreakdown.
    prior = synthetic_text8(4000, seed=1).data
    eval_stream = synthetic_text8(1500, seed=2).prequential_carve(eval_bytes=300)[1]
    cfg = {
        "max_order": 4,
        "hash_min_order": 2,
        "table_bits": 14,
        "alpha": 0.5,
        "lr": 0.02,
        "delta_orders": (3, 4, 5, 6),
        "delta_dim": 1 << 12,
        "delta_eta": 0.3,
    }
    delta = build_model("delta_mix", cfg)
    _warm(delta, prior, flop_budget=2e7)
    col = build_model("column_mix", {**cfg, "n_columns": 1})
    _warm(col, prior, flop_budget=2e7)

    ds = delta.init_prequential_state()
    cs = col.init_prequential_state()
    for pos in range(len(eval_stream) - 1):
        b = int(eval_stream[pos])
        ds, dl, df = delta.step(ds, b, pos)
        cs, cl, cf = col.step(cs, b, pos)
        assert torch.equal(dl, cl)
        assert df == cf
    # analytic paths too (the _steady_step_flops guard protects these / the pretrain look-ahead).
    assert delta.flops(128) == col.flops(128)
    assert delta.decode_step_flops(64) == col.decode_step_flops(64)
    assert col.context_window == delta.context_window


def test_delta_orders_empty_is_hashed_mix():
    # delta_orders=() => no key to route => delegates through DeltaMix to hashed_mix, for any C.
    prior = synthetic_text8(4000, seed=1).data
    eval_stream = synthetic_text8(1500, seed=2).prequential_carve(eval_bytes=300)[1]
    cfg = {"max_order": 5, "hash_min_order": 4, "table_bits": 14, "alpha": 0.5, "lr": 0.02}
    hashed = build_model("hashed_mix", cfg)
    _warm(hashed, prior, flop_budget=2e7)
    col = build_model("column_mix", {**cfg, "n_columns": 8, "delta_orders": ()})
    _warm(col, prior, flop_budget=2e7)
    hs = hashed.init_prequential_state()
    cs = col.init_prequential_state()
    for pos in range(len(eval_stream) - 1):
        b = int(eval_stream[pos])
        hs, hl, hf = hashed.step(hs, b, pos)
        cs, cl, cf = col.step(cs, b, pos)
        assert torch.equal(hl, cl)
        assert hf == cf


# --- router ------------------------------------------------------------------
def test_route_slot_deterministic_bounded():
    m = build_model("column_mix", {"n_columns": 4, "route_buckets": 1 << 8, "route_order": 3})
    b1 = m._route_slot([5, 6, 7])
    assert b1 == m._route_slot([5, 6, 7])  # deterministic within an instance
    assert 0 <= b1 < (1 << 8)
    assert m._route_slot([]) == 0  # empty prefix (pos 0) -> bucket 0
    assert 0 <= m._route_slot([9]) < (1 << 8)  # window shorter than route_order is valid
    m2 = build_model("column_mix", {"n_columns": 4, "route_buckets": 1 << 8, "route_order": 3})
    assert m2._route_slot([5, 6, 7]) == b1  # salt-free across instances


def test_gate_off_is_fixed_hash_route():
    # gate_lr=0 AND route_epsilon=0 => the route is argmax(gate)=b mod C every byte (no learning,
    # no exploration); the stored last_route must equal last_bucket % C, deterministically.
    cfg = {
        "n_columns": 4,
        "route_buckets": 1 << 8,
        "route_order": 3,
        "gate_lr": 0.0,
        "route_epsilon": 0.0,
        "delta_orders": (3, 4),
        "delta_dim": 1 << 10,
    }
    m = build_model("column_mix", cfg)
    s = m.init_prequential_state()
    stream = synthetic_text8(300, seed=4).data
    for pos in range(120):
        s, _, _ = m.step(s, int(stream[pos]), pos)
        assert s.cache.last_route == s.cache.last_bucket % cfg["n_columns"]


def test_prequential_runs_and_is_finite():
    m = build_model(
        "column_mix",
        {
            "max_order": 3,
            "n_columns": 4,
            "route_buckets": 1 << 8,
            "delta_dim": 1 << 12,
            "delta_orders": (3, 4, 5),
            "delta_eta": 0.3,
            "gate_lr": 0.1,
            "route_epsilon": 0.05,
        },
    )
    r = prequential_bpb(m, synthetic_text8(900, seed=3).data, device=CPU)
    assert np.isfinite(r.bpb)
    assert 0.0 < r.bpb <= 8.0


# --- no leakage --------------------------------------------------------------
def test_prediction_at_t_cannot_see_future():
    # Prediction for byte j uses only bytes 0..j-1; the column/gate updates for the revealed byte
    # at pos use only that byte. route_epsilon=0 keeps it deterministic (no RNG draws).
    cfg = {
        "max_order": 4,
        "n_columns": 4,
        "route_buckets": 1 << 8,
        "route_order": 2,
        "delta_dim": 1 << 12,
        "delta_orders": (3, 4, 5, 6),
        "delta_eta": 0.3,
        "gate_lr": 0.3,
        "route_epsilon": 0.0,
    }
    stream = synthetic_text8(700, seed=5).data
    r1 = prequential_bpb(build_model("column_mix", cfg), stream, device=CPU, collect_logits=True)
    perturbed = stream.copy()
    perturbed[400] = (int(perturbed[400]) + 7) % 256
    r2 = prequential_bpb(build_model("column_mix", cfg), perturbed, device=CPU, collect_logits=True)
    for j in range(401):
        assert torch.equal(r1.predicted_logits[j], r2.predicted_logits[j])
    assert not torch.equal(r1.predicted_logits[401], r2.predicted_logits[401])


def test_first_step_does_not_update_columns_or_gate():
    # The deferred column/gate updates are gated on did_update (last_probs not None), so the first
    # step of a stream (or a warm-row boundary) must leave Wcols and gate untouched — the property
    # that makes the inherited WarmMix.train_step leak-free without an override.
    m = build_model(
        "column_mix",
        {"n_columns": 4, "route_buckets": 1 << 8, "delta_orders": (3, 4), "gate_lr": 0.5},
    )
    s = m.init_prequential_state()
    wcols0 = s.cache.Wcols.copy()
    gate0 = s.cache.gate.copy()
    s, _, _ = m.step(s, 65, 0)  # first step: no pending prediction yet
    assert np.array_equal(s.cache.Wcols, wcols0)
    assert np.array_equal(s.cache.gate, gate0)
    # subsequent steps DO adapt (proves the updates are merely deferred, not absent)
    for pos in range(1, 12):
        s, _, _ = m.step(s, 65 + pos, pos)
    assert not np.array_equal(s.cache.gate, gate0)


# --- FLOP honesty ------------------------------------------------------------
def test_route_increment_matches_hand_formula():
    # Pin the exact symbolic charge: router (always) + the O(1) per-arm bandit update (when pending
    # AND the gate learns). charge == code.
    c = 4
    m = build_model("column_mix", {"n_columns": c, "delta_orders": (3, 4)})
    inc = m._route_increment(did_update=True)
    assert inc.forward == gather_flops(1) + pointwise_flops(3 + c)
    assert inc.backward == pointwise_flops(5)  # reward log2 (2) + EMA step (3)
    # No pending prediction -> no gate update.
    assert m._route_increment(did_update=False).backward == 0
    # Frozen gate (gate_lr=0) -> the router still routes (forward) but charges no gate update.
    off = build_model("column_mix", {"n_columns": c, "delta_orders": (3, 4), "gate_lr": 0.0})
    assert off._route_increment(did_update=True).backward == 0
    assert off._route_increment(did_update=True).forward == gather_flops(1) + pointwise_flops(3 + c)


def test_step_flops_equal_delta_path_plus_route_increment():
    # The ONLY per-byte FLOP difference between C=1 (delegates to the delta path) and C=N (same
    # delta/count config, route_order <= max delta order so the window cap matches) is exactly the
    # analytic _route_increment. Drives both on the same stream and asserts it step by step.
    base = {
        "max_order": 4,
        "hash_min_order": 2,
        "table_bits": 16,
        "delta_orders": (3, 4, 5, 6),
        "delta_dim": 1 << 12,
        "delta_eta": 0.3,
        "route_order": 2,  # <= max(delta_orders) so C=1 and C=N share the window cap
    }
    c1 = build_model("column_mix", {**base, "n_columns": 1})
    cn = build_model("column_mix", {**base, "n_columns": 4, "gate_lr": 0.3, "route_epsilon": 0.0})
    s1 = c1.init_prequential_state()
    sn = cn.init_prequential_state()
    stream = synthetic_text8(400, seed=8).data
    saw_route_backward = False
    for pos in range(len(stream)):
        b = int(stream[pos])
        s1, _, f1 = c1.step(s1, b, pos)
        sn, _, fn = cn.step(sn, b, pos)
        inc = cn._route_increment(did_update=(pos >= 1))
        assert fn.forward == f1.forward + inc.forward
        assert fn.backward == f1.backward + inc.backward
        saw_route_backward |= inc.backward > 0
    assert saw_route_backward  # the gate update really did charge work over the stream


# --- the (iv) lever: selection, NOT capacity ---------------------------------
def test_selection_beats_single_column_and_capacity_does_not():
    # On the interaction source, a routed sheet (C=2, learned gate) beats a single column by a clear
    # margin — AND a single column does NOT catch up when its delta_dim is grown to C*d (the
    # matched-capacity control), proving the win is route-conditional SELECTION, not table size.
    ev = _interaction_stream(4000, seed=3)
    base = {
        "max_order": 3,
        "hash_min_order": 2,
        "table_bits": 14,
        "alpha": 0.5,
        "lr": 0.05,
        "delta_orders": (2, 3),
        "delta_eta": 0.5,
    }
    c1 = _bpb(build_model("column_mix", {**base, "n_columns": 1, "delta_dim": 1 << 10}), ev)
    c1_big = _bpb(  # matched-capacity control: one column at C*d width
        build_model("column_mix", {**base, "n_columns": 1, "delta_dim": 1 << 11}), ev
    )
    c2 = _bpb(
        build_model(
            "column_mix",
            {
                **base,
                "n_columns": 2,
                "delta_dim": 1 << 10,
                "route_buckets": 16,
                "route_order": 5,  # reaches the regime marker at offset -5 (phi/counts cannot)
                "gate_lr": 0.5,
                "route_epsilon": 0.1,
            },
        ),
        ev,
    )
    assert c2 < c1 - 0.02  # routing extracts the interaction
    assert c2 < c1_big - 0.02  # more capacity in one column does NOT
    assert abs(c1 - c1_big) < 1e-9  # capacity is genuinely inert here (same prediction)


def test_dead_column_abstains():
    # A column never chosen keeps its zeroed Wcols slice (=> z_delta=0 => uniform abstention). Drive
    # a constant-byte stream with the gate off: only column (bucket mod C) is ever used.
    cfg = {
        "n_columns": 4,
        "route_buckets": 1 << 6,
        "route_order": 3,
        "gate_lr": 0.0,
        "route_epsilon": 0.0,
        "delta_orders": (2, 3),
        "delta_dim": 1 << 10,
        "delta_eta": 0.5,
    }
    m = build_model("column_mix", cfg)
    s = m.init_prequential_state()
    for pos in range(80):
        s, _, _ = m.step(s, 88, pos)  # constant byte -> one bucket -> one column
    used = {c for c in range(cfg["n_columns"]) if np.any(s.cache.Wcols[c] != 0.0)}
    assert len(used) < cfg["n_columns"]  # at least one column stayed zero (abstains)
    for c in range(cfg["n_columns"]):
        if c not in used:
            assert np.array_equal(s.cache.Wcols[c], np.zeros_like(s.cache.Wcols[c]))


# --- leak-free warm handoff + reproducibility --------------------------------
def test_eval_does_not_mutate_warm_state():
    cfg = {
        "max_order": 4,
        "hash_min_order": 4,
        "table_bits": 12,
        "n_columns": 4,
        "route_buckets": 1 << 8,
        "delta_dim": 1 << 12,
        "delta_orders": (3, 4, 5),
        "delta_eta": 0.3,
        "gate_lr": 0.2,
        "route_epsilon": 0.05,
    }
    model = build_model("column_mix", cfg)
    _warm(model, synthetic_text8(3000, seed=1).data, flop_budget=3e7)
    warm_wcols = model._warm.Wcols.copy()
    warm_gate = model._warm.gate.copy()
    warm_weights = model._warm.weights.copy()
    eval_stream = synthetic_text8(1200, seed=2).prequential_carve(eval_bytes=400)[1]
    prequential_bpb(model, eval_stream, device=CPU)
    assert np.array_equal(model._warm.Wcols, warm_wcols)
    assert np.array_equal(model._warm.gate, warm_gate)
    assert np.array_equal(model._warm.weights, warm_weights)


def test_two_eval_streams_are_isolated():
    cfg = {
        "max_order": 3,
        "n_columns": 4,
        "route_buckets": 1 << 8,
        "delta_dim": 1 << 12,
        "delta_orders": (3, 4),
        "delta_eta": 0.3,
        "gate_lr": 0.2,
        "route_epsilon": 0.0,
    }
    model = build_model("column_mix", cfg)
    _warm(model, synthetic_text8(2000, seed=1).data, flop_budget=2e7)
    s1 = model.init_prequential_state()
    s2 = model.init_prequential_state()
    for pos in range(80):
        s1, _, _ = model.step(s1, pos % 7, pos)
    assert not np.array_equal(s1.cache.Wcols, s2.cache.Wcols)  # s1's writes never reach s2
    assert np.array_equal(s2.cache.Wcols, model._warm.Wcols)  # untouched copy == warm state
    assert np.array_equal(s2.cache.gate, model._warm.gate)


def test_reproducible_under_fixed_seed():
    cfg = {
        "max_order": 4,
        "hash_min_order": 4,
        "table_bits": 14,
        "n_columns": 4,
        "route_buckets": 1 << 8,
        "delta_dim": 1 << 13,
        "delta_orders": (3, 4, 5, 6),
        "delta_eta": 0.3,
        "gate_lr": 0.2,
        "route_epsilon": 0.1,  # exploration on => the per-stream RNG must make runs reproducible
        "seed": 99,
    }
    prior = synthetic_text8(3000, seed=1).data
    eval_stream = synthetic_text8(1200, seed=2).prequential_carve(eval_bytes=400)[1]
    runs = []
    for _ in range(2):
        m = build_model("column_mix", cfg)
        _warm(m, prior, flop_budget=3e7, seed=99)
        runs.append(_bpb(m, eval_stream))
    assert runs[0] == runs[1]
