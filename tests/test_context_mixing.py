"""Context-mixing reference: hand-checkable mixing math, honest non-matmul FLOP
accounting, online learning, and a sane bpb vs. the untrained transformer.

The numbers below are derived by hand in the comments so a reviewer can confirm
the mixing and the FLOP charge without trusting the implementation.
"""

import math

import numpy as np
import torch

from smolml.data import load_sample, synthetic_text8
from smolml.flops import FlopBreakdown, gather_flops, pointwise_flops
from smolml.models import build_model, list_models
from smolml.models.context_mixing import (
    ContextMixing,
    ContextMixingConfig,
    laplace_prob,
    mix_logits,
    mixer_gradient,
    softmax,
)
from smolml.prequential import prequential_bpb

CPU = torch.device("cpu")


# --- registration ------------------------------------------------------------
def test_registered():
    assert "context_mixing" in list_models()


# --- hand-checkable mixing math ----------------------------------------------
def test_laplace_prob_hand():
    # counts [3,1,0,0], alpha=1 -> (counts+1)/(4 + 1*4) = [4,2,1,1]/8.
    p = laplace_prob(np.array([3.0, 1.0, 0.0, 0.0]), alpha=1.0)
    assert np.allclose(p, [0.5, 0.25, 0.125, 0.125])
    assert math.isclose(p.sum(), 1.0)


def test_laplace_prob_unseen_is_uniform():
    # An all-zero context smooths to the uniform distribution (the model abstains).
    p = laplace_prob(np.zeros(4), alpha=0.5)
    assert np.allclose(p, 0.25)


def test_mix_logits_hand():
    # weights [0.5,0.5], stretched rows [1,3] and [5,7] -> z = [3,5].
    stretched = np.array([[1.0, 3.0], [5.0, 7.0]])
    z = mix_logits(stretched, np.array([0.5, 0.5]))
    assert np.allclose(z, [3.0, 5.0])


def test_softmax_hand():
    assert np.allclose(softmax(np.array([0.0, 0.0])), [0.5, 0.5])
    # softmax([ln 3, 0]) = [3, 1] / 4.
    assert np.allclose(softmax(np.array([math.log(3.0), 0.0])), [0.75, 0.25])


def test_mixer_gradient_hand():
    # probs [0.5,0.5], target 0 -> err [-0.5, 0.5]; stretched [[2,4]] ->
    # grad = 2*(-0.5) + 4*(0.5) = 1.0.
    grad = mixer_gradient(np.array([0.5, 0.5]), np.array([[2.0, 4.0]]), target=0)
    assert np.allclose(grad, [1.0])


def test_one_sgd_step_lowers_loss_on_the_revealed_byte():
    # A single online mixer step must reduce −log p(target): start uniform, take
    # one step, the probability mass on the revealed byte goes up.
    stretched = np.array([[0.0, 1.0, -1.0]])  # one model, V=3
    w = np.array([0.5])
    z = mix_logits(stretched, w)
    p_before = softmax(z)
    target = 1
    w = w - 0.5 * mixer_gradient(p_before, stretched, target)
    p_after = softmax(mix_logits(stretched, w))
    assert p_after[target] > p_before[target]


# --- honest non-matmul FLOP accounting (charge == code, exactly) -------------
def test_step_charges_exactly_the_branches_executed():
    # K = max_order+1 = 2, V = 256. Trace the exact branches step() runs.
    #
    # pos=0 (fresh: no pending prediction, empty window):
    #   fold      -> only order-0 (no preceding bytes for order-1): n_fold = 1
    #   predict   -> order-0 lookup hits (just folded), order-1 lookup misses:
    #                n_active = 2, n_laplace = 1
    #   forward  = pointwise(3V*1 + KV + 2KV + 5V) + gather(2)
    #            = pointwise(768 + 512 + 1024 + 1280) + 2 = 3584 + 2 = 3586
    #   backward = pointwise(n_fold=1) + gather(1)            (no mixer update) = 2
    model = ContextMixing(ContextMixingConfig(max_order=1))
    state = model.init_prequential_state()
    state, _, f0 = model.step(state, 65, 0)  # byte 'A'
    assert f0.forward == pointwise_flops(3 * 256 + 2 * 256 + 4 * 256 + 5 * 256) + gather_flops(2)
    assert f0.forward == 3586
    assert f0.backward == pointwise_flops(1) + gather_flops(1) == 2
    assert f0.total == 3588

    # pos=1 (byte 'B' != 'A'): pending prediction -> mixer update fires; both
    #   orders fold (one preceding byte now); order-1 predict context unseen.
    #   n_fold = 2, n_active = 2, n_laplace = 1, did_update = True.
    #   forward  = 3586 (same shape as pos=0)
    #   backward = pointwise(n_fold=2 + 1 + 2KV + 2K) + gather(2)
    #            = pointwise(2 + 1 + 1024 + 4) + 2 = 1031 + 2 = 1033
    state, _, f1 = model.step(state, 66, 1)
    assert f1.forward == 3586
    assert f1.backward == pointwise_flops(2 + 1 + 2 * 2 * 256 + 2 * 2) + gather_flops(2) == 1033
    assert f1.total == 4619


def test_per_byte_cost_is_dynamic_not_constant():
    # The first byte has no pending prediction (no mixer update) and folds fewer
    # orders, so it MUST cost strictly less than a later, fully-warmed byte. This
    # is exactly the over-charge the constant per-byte charge would have hidden.
    model = ContextMixing(ContextMixingConfig(max_order=2))
    state = model.init_prequential_state()
    state, _, f0 = model.step(state, 65, 0)
    state, _, f1 = model.step(state, 66, 1)
    state, _, f2 = model.step(state, 67, 2)
    assert f0.total < f1.total < f2.total


def test_steady_step_flops_hand_computed():
    # The analytic steady-state estimate (all K active+seen, mixer updating):
    #   forward  = pointwise(6KV + 5V) + gather(K)
    #            = pointwise(6*2*256 + 5*256) + gather(2) = 4352 + 2 = 4354
    #   backward = pointwise(K_fold + 1 + 2KV + 2K) + gather(K)
    #            = pointwise(2 + 1 + 1024 + 4) + gather(2) = 1031 + 2 = 1033
    bd = ContextMixing(ContextMixingConfig(max_order=1))._steady_step_flops()
    assert bd.forward == pointwise_flops(6 * 2 * 256 + 5 * 256) + gather_flops(2) == 4354
    assert bd.backward == pointwise_flops(2 + 1 + 2 * 2 * 256 + 2 * 2) + gather_flops(2) == 1033
    assert bd.total == 5387


def test_flops_are_non_zero_and_grow_with_predictors():
    # Both prediction and adaptation cost real (pointwise/gather) FLOPs, and more
    # predictors mean more charged work.
    one = ContextMixing(ContextMixingConfig(max_order=0))._steady_step_flops()
    four = ContextMixing(ContextMixingConfig(max_order=3))._steady_step_flops()
    assert one.forward > 0 and one.backward > 0
    assert four.total > one.total


def test_flops_seq_len_is_steady_per_byte_times_length():
    model = ContextMixing(ContextMixingConfig(max_order=2))
    assert model.flops(10) == model._steady_step_flops().scale(10)


def test_prequential_total_equals_exact_sum_of_step_flops():
    # The loop must accumulate EXACTLY what each step() returns — no hidden charge,
    # no constant approximation. Replay step() independently and sum the per-byte
    # breakdowns; the prequential total must match to the FLOP.
    cfg = ContextMixingConfig(max_order=2)
    stream = synthetic_text8(2000, seed=0).prequential_carve(eval_bytes=60)[1]
    state = ContextMixing(cfg).init_prequential_state()
    replay = ContextMixing(cfg)
    total = FlopBreakdown()
    for pos in range(len(stream) - 1):
        state, _, fl = replay.step(state, int(stream[pos]), pos)
        total += fl
    result = prequential_bpb(ContextMixing(cfg), stream, device=CPU)
    assert result.n_bytes == 60
    assert result.eval_flops == total.total


# --- online learning ---------------------------------------------------------
def test_online_learning_drives_periodic_stream_to_near_zero_bpb():
    # "abababab...": an order>=1 model learns the period and predicts almost
    # perfectly, so cumulative bpb falls far below 1 bit (uniform byte = 8 bits).
    stream = np.array([65, 66] * 150, dtype=np.uint8)
    result = prequential_bpb(ContextMixing(ContextMixingConfig(max_order=2)), stream, device=CPU)
    assert result.bpb < 0.5


def test_higher_order_helps_on_real_english():
    # Real English has genuine higher-order structure, so adding orders lowers bpb.
    stream = load_sample().prequential_carve(eval_bytes=800)[1]
    bpb0 = prequential_bpb(build_model("context_mixing", {"max_order": 0}), stream, device=CPU).bpb
    bpb3 = prequential_bpb(build_model("context_mixing", {"max_order": 3}), stream, device=CPU).bpb
    assert bpb3 < bpb0


def test_beats_untrained_transformer_on_offline_clone():
    # The reference ceiling: single-pass online mixing reaches far lower bpb than a
    # zero-pretrain transformer, at a small fraction of the FLOPs.
    stream = synthetic_text8(2000, seed=0).prequential_carve(eval_bytes=120)[1]
    mix = prequential_bpb(build_model("context_mixing", {"max_order": 3}), stream, device=CPU)
    cfg = {"d_model": 16, "n_layers": 1, "n_heads": 2, "max_seq_len": 32}
    tfm = prequential_bpb(build_model("transformer", cfg), stream, device=CPU)
    assert mix.bpb < tfm.bpb - 1.0
    assert mix.eval_flops < tfm.eval_flops


# --- no future leakage -------------------------------------------------------
def test_prediction_cannot_see_the_byte_being_predicted():
    # Two streams identical up to index k, differing at k. Predictions for bytes
    # 0..k (made from bytes before them) must be bit-identical; the prediction for
    # byte k+1 (the first to condition on byte k) must change.
    rng = np.random.default_rng(0)
    base = rng.integers(97, 123, size=40).astype(np.uint8)
    k = 20
    a = base.copy()
    b = base.copy()
    b[k] = base[k] ^ 1  # flip one byte at index k
    ra = prequential_bpb(
        ContextMixing(ContextMixingConfig(max_order=3)), a, device=CPU, collect_logits=True
    )
    rb = prequential_bpb(
        ContextMixing(ContextMixingConfig(max_order=3)), b, device=CPU, collect_logits=True
    )
    for i in range(k + 1):
        assert torch.equal(ra.predicted_logits[i], rb.predicted_logits[i])
    assert not torch.equal(ra.predicted_logits[k + 1], rb.predicted_logits[k + 1])


# --- transductive: no pretrained parameters ----------------------------------
def test_transductive_has_no_parameters():
    assert ContextMixing(ContextMixingConfig()).num_params() == 0


def test_from_config_ignores_transformer_keys():
    # The CLI injects transformer keys; from_config keeps only its own fields.
    model = ContextMixing.from_config(
        {"d_model": 64, "n_layers": 3, "n_heads": 4, "max_seq_len": 256, "max_order": 2}
    )
    assert model.config.max_order == 2
    assert model.num_predictors == 3
