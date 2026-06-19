"""Gated order escalation (Task B.2, Phase 2): pre-reveal escalation, honest
dynamic charge, and the degenerate-config identity to fixed-order ``warm_mix``.

These pin the contract that makes ``gated_mix`` trustworthy as a per-FLOP win:

- with ``min_order == max_order`` the escalation loop is structurally absent, so
  ``gated_mix`` is **bit-identical** to ``warm_mix`` — same predictions AND the
  same :class:`FlopBreakdown` — both cold (budget 0) and after an identical warmup
  (the clean baseline the per-FLOP claim is measured against);
- the per-byte charge is **exactly the orders evaluated** plus the per-step gate
  arithmetic, hand-computed on a tiny driven stream (full escalation and an
  early-firing gate);
- the gate is **pre-reveal**: an eval prediction is invariant to future bytes;
- an **aggressive** threshold evaluates fewer orders on easy bytes (mean evaluated
  depth well below ``max_order``) and the summed step FLOPs are correspondingly
  lower than full escalation;
- warming + eval is reproducible under a fixed seed; registration + config
  validation work.
"""

import numpy as np
import pytest
import torch

from smolml.data import synthetic_text8
from smolml.flops import FlopBreakdown
from smolml.models import build_model, list_models
from smolml.models.gated_mix import GatedMix, GatedMixConfig
from smolml.prequential import prequential_bpb, pretrain

CPU = torch.device("cpu")


def _warm(model, prior, *, flop_budget, seq_len=64, batch_size=8, seed=0) -> int:
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


def _drive(model, stream):
    """Step ``model`` over ``stream`` byte by byte, returning per-step FLOP
    breakdowns and the evaluated-order count (``n_eval``) realized each step."""
    state = model.init_prequential_state()
    fls: list[FlopBreakdown] = []
    n_eval: list[int] = []
    for pos in range(len(stream) - 1):
        state, _logits, fl = model.step(state, int(stream[pos]), pos)
        fls.append(fl)
        n_eval.append(state.cache.last_stretched.shape[0])
    return fls, n_eval


# --- registration + config ---------------------------------------------------
def test_registered_and_buildable():
    assert "gated_mix" in list_models()
    model = build_model("gated_mix", {"max_order": 3, "min_order": 1, "gate_threshold": 0.5})
    assert isinstance(model, GatedMix)
    assert model.num_params() == 0  # transductive: no gradient parameters


def test_config_validation():
    # min_order must be within [0, max_order]; gate_threshold must be non-negative.
    with pytest.raises(ValueError):
        GatedMixConfig(max_order=2, min_order=3)
    with pytest.raises(ValueError):
        GatedMixConfig(max_order=2, min_order=-1)
    with pytest.raises(ValueError):
        GatedMixConfig(max_order=2, min_order=1, gate_threshold=-0.1)
    # A degenerate gate_threshold of 0 (never fires) is valid.
    GatedMixConfig(max_order=2, min_order=1, gate_threshold=0.0)


def test_from_config_ignores_transformer_keys_keeps_gated_fields():
    # The CLI injects transformer keys; from_config keeps only GatedMixConfig fields.
    model = GatedMix.from_config(
        {"max_order": 4, "min_order": 2, "gate_threshold": 0.3, "d_model": 128, "n_heads": 4}
    )
    assert model.config.min_order == 2
    assert model.config.gate_threshold == 0.3
    assert model.num_predictors == 5


# --- degenerate config == fixed-order warm_mix, bit for bit -------------------
def test_min_equals_max_cold_is_bit_identical_to_warm_mix():
    # min_order == max_order: the floor covers every order, so the escalation loop
    # never runs and no gate is computed. Cold (budget 0) every prediction AND the
    # whole FLOP breakdown must match fixed-order warm_mix exactly.
    stream = synthetic_text8(2000, seed=0).prequential_carve(eval_bytes=120)[1]
    warm = prequential_bpb(
        build_model("warm_mix", {"max_order": 3}), stream, device=CPU, collect_logits=True
    )
    gated = prequential_bpb(
        build_model("gated_mix", {"max_order": 3, "min_order": 3, "gate_threshold": 0.5}),
        stream,
        device=CPU,
        collect_logits=True,
    )
    assert gated.predicted_logits is not None
    for g_logits, w_logits in zip(gated.predicted_logits, warm.predicted_logits, strict=True):
        assert torch.equal(g_logits, w_logits)
    assert gated.flops == warm.flops
    assert gated.bpb == warm.bpb


def test_min_equals_max_warmed_is_bit_identical_to_warm_mix():
    # The identity must survive the warm prior->eval handoff: warm both models on
    # the same prior/seed/budget (gated's step == warm_mix's step at min==max, so
    # the warmed states coincide), then require a bit-for-bit eval match.
    prior, eval_stream = synthetic_text8(6000, seed=1).prequential_carve(eval_bytes=400)
    warm = build_model("warm_mix", {"max_order": 3})
    gated = build_model("gated_mix", {"max_order": 3, "min_order": 3, "gate_threshold": 0.5})
    assert _warm(warm, prior, flop_budget=5e7, seed=0) == _warm(
        gated, prior, flop_budget=5e7, seed=0
    )
    rw = prequential_bpb(warm, eval_stream, device=CPU, collect_logits=True)
    rg = prequential_bpb(gated, eval_stream, device=CPU, collect_logits=True)
    for g_logits, w_logits in zip(rg.predicted_logits, rw.predicted_logits, strict=True):
        assert torch.equal(g_logits, w_logits)
    assert rg.flops == rw.flops
    assert rg.bpb == rw.bpb


# --- honest dynamic charge == the orders evaluated ---------------------------
def test_per_byte_charge_full_escalation_hand_computed():
    # Drive [97,98,97,98] cold with gate_threshold=0 (gate never fires). At pos 2
    # (revealing 97) the window is [97,98] -> new_window [98,97]:
    #   evaluated = order 0 (seen) + order 1 (context [97] seen) = n_eval 2;
    #   order-2 context [98,97] is unseen -> the escalation probe stops there.
    #   n_laplace = 2, n_active = 3 (order-0 + the two escalation probes),
    #   n_fold = 3, did_update with n_eval_prev = 1 (pos 1 evaluated order 0 only).
    #   forward  = pointwise(3V*2 + 2V + 2*2V + 5V*1) + gather(3) = 4352 + 3 = 4355
    #   backward = pointwise(3 + 1 + 2*1*V + 2*1) + gather(3)     = 518  + 3 = 521
    model = build_model("gated_mix", {"max_order": 2, "min_order": 0, "gate_threshold": 0.0})
    fls, n_eval = _drive(model, [97, 98, 97, 98])
    assert n_eval[2] == 2  # charge tracks the evaluated-order count
    assert fls[2].forward == 4355
    assert fls[2].backward == 521
    assert fls[2].total == 4876


def test_per_byte_charge_gate_fires_hand_computed():
    # Same stream, gate_threshold=2.0 (1 - max p < 2 is always true -> fires on the
    # first check). At pos 2 escalation stops at the floor (order 0): n_eval 1, but
    # the gate arithmetic (one softmax 5V + the V+1 confidence check) is still
    # charged, and the escalation probe never runs (n_active = 1):
    #   forward  = pointwise(3V*1 + 1V + 2*1V + 5V*1 + (V+1)) + gather(1)
    #            = pointwise(11V + 257) + 1 = 3073 + 1 = 3074
    #   backward = pointwise(3 + 1 + 2*1*V + 2*1) + gather(3) = 521
    model = build_model("gated_mix", {"max_order": 2, "min_order": 0, "gate_threshold": 2.0})
    fls, n_eval = _drive(model, [97, 98, 97, 98])
    assert n_eval[2] == 1  # the gate stopped escalation at the floor
    assert fls[2].forward == 3074  # includes the charged gate arithmetic (257)
    assert fls[2].backward == 521
    assert fls[2].total == 3595


# --- pre-reveal gate: no future leakage --------------------------------------
def test_gate_is_pre_reveal_invariant_to_future_bytes():
    # Two streams identical up to index k, differing at k. Predictions for bytes
    # 0..k (and the gate's escalation decision behind each) depend only on earlier
    # bytes, so they are bit-identical; the prediction for byte k+1 (first to
    # condition on the changed byte) differs. Proves the gate never peeks ahead.
    rng = np.random.default_rng(0)
    base = rng.integers(97, 123, size=40).astype(np.uint8)
    k = 20
    a = base.copy()
    b = base.copy()
    b[k] = base[k] ^ 1
    model = build_model("gated_mix", {"max_order": 3, "min_order": 1, "gate_threshold": 0.4})
    ra = prequential_bpb(model, a, device=CPU, collect_logits=True)
    rb = prequential_bpb(model, b, device=CPU, collect_logits=True)
    for i in range(k + 1):
        assert torch.equal(ra.predicted_logits[i], rb.predicted_logits[i])
    assert not torch.equal(ra.predicted_logits[k + 1], rb.predicted_logits[k + 1])


# --- the (iv) lever: fewer orders on easy bytes, at fewer FLOPs ---------------
def test_aggressive_threshold_evaluates_fewer_orders_and_costs_less():
    # On an easy periodic stream the low orders quickly predict confidently. A full
    # (threshold 0) escalation pays for every active+seen order on every byte; an
    # aggressive gate stops once confident, so it evaluates far fewer orders (mean
    # depth well below max_order) and the harness-summed step FLOPs are lower.
    stream = np.array([97, 98] * 400, dtype=np.uint8)
    max_order = 4
    full = build_model("gated_mix", {"max_order": max_order, "min_order": 0, "gate_threshold": 0.0})
    aggr = build_model("gated_mix", {"max_order": max_order, "min_order": 0, "gate_threshold": 0.5})
    fls_full, ne_full = _drive(full, stream)
    fls_aggr, ne_aggr = _drive(aggr, stream)
    depth_full = float(np.mean(ne_full)) - 1.0  # depth = n_eval - 1 (orders 0..depth)
    depth_aggr = float(np.mean(ne_aggr)) - 1.0
    total_full = sum(f.total for f in fls_full)
    total_aggr = sum(f.total for f in fls_aggr)
    assert depth_aggr < depth_full  # the gate genuinely stops earlier
    assert depth_aggr < max_order  # mean evaluated depth below the ceiling
    assert total_aggr < total_full  # and the FLOPs follow the depth down


# --- reproducibility ---------------------------------------------------------
def test_reproducible_under_fixed_seed():
    cfg = {"max_order": 3, "min_order": 1, "gate_threshold": 0.4}
    prior = synthetic_text8(3000, seed=4).data
    eval_stream = synthetic_text8(1500, seed=5).prequential_carve(eval_bytes=400)[1]

    def warmed_run():
        m = build_model("gated_mix", cfg)
        _warm(m, prior, flop_budget=2e7, batch_size=4, seed=7)
        return prequential_bpb(m, eval_stream, device=CPU)

    r1 = warmed_run()
    r2 = warmed_run()
    assert r1.bpb == r2.bpb
    assert r1.eval_flops == r2.eval_flops
