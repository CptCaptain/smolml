"""Surprise-gated predictive-coding refinement candidate (Task B.1).

Protects what makes this candidate trustworthy: every PC FLOP flows through ``step``'s honest
breakdown (charge == reality, hand-computed and K-dependent), the surprise gate is monotonic,
settling actually reduces loss once the readout is non-zero, the readout starts as an identity
to the core (zero correction), there is no future leakage, and a stream is reproducible.

The headline (iv) comparison — gated ≤ uniform bpb at matched total FLOPs — is the job of the
``smolml.experiments.pc_refine_sweep`` runner, not a unit test (it must be reported honestly
whichever way it lands, like A.1).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from smolml.data import synthetic_text8
from smolml.data.corpus import VOCAB_SIZE
from smolml.flops import matmul_flops, pointwise_flops
from smolml.models import build_model, list_models
from smolml.models.pc_refine import PCRefine, PCRefineConfig
from smolml.prequential import PrequentialConfig, prequential_bpb, prequential_run

CPU = torch.device("cpu")
TINY = {"d_model": 16, "n_layers": 2, "n_heads": 2, "max_seq_len": 64, "m": 8}
# The transformer core's keys only (drops the PC-module field ``m``) for like-core baselines.
CORE = {k: v for k, v in TINY.items() if k != "m"}


def _model(**overrides) -> PCRefine:
    cfg = {**TINY, **overrides}
    model = build_model("pc_refine", cfg)
    model.eval()
    return model


# --- (a) FLOP accounting equals a hand-computed K-dependent value --------------
def test_step_flops_equal_core_plus_pc_hand_computed():
    # Uniform gate => K = k_uniform is constant, so the per-step forward cost is exactly
    # hand-computable; the update gate is forced on (threshold below any -log p) so the
    # backward cost is the full applied update. Every number is derived by hand here.
    model = _model(gate="uniform", k_uniform=3, update_surprise_threshold=-1.0)
    d, m, v = model.config.d_model, model.config.m, model.config.vocab_size
    k = 3
    gate = pointwise_flops(v, 4)
    settle_iter = (
        matmul_flops(1, d, m) + matmul_flops(1, m, d) + pointwise_flops(d) + pointwise_flops(m, 5)
    )
    readout = matmul_flops(1, v, m) + pointwise_flops(v, 4)
    forward_extra = gate + k * settle_iter + readout
    update = (
        matmul_flops(1, d, m)  # residual recompute r = h - Wz at the settled latent
        + pointwise_flops(d)
        + pointwise_flops(v)  # p - e_byte
        + matmul_flops(d, m, 1)  # ΔW outer
        + matmul_flops(m, v, 1)  # ΔVmat outer
        + pointwise_flops(d * m, 3)  # W combine: keep-mult + lr-mult + add
        + pointwise_flops(m * v, 3)  # Vmat combine: keep-mult + lr-mult + sub
    )

    state = model.init_prequential_state()
    # Step 0: no pending prediction -> no update (backward 0); settling+readout in forward.
    state, _, f0 = model.step(state, 7, 0)
    assert f0.backward == 0
    assert f0.forward == model.core.decode_step_flops(1).forward + forward_extra
    # Step 1: the pending prediction's update is applied (gate forced on) -> backward > 0.
    state, _, f1 = model.step(state, 8, 1)
    assert f1.backward == pointwise_flops(1) + update
    assert f1.forward == model.core.decode_step_flops(2).forward + forward_extra


def test_update_gate_charges_only_the_decision_when_it_does_not_fire():
    # A huge threshold gates every update off: a pending step then pays only the O(1)
    # gate-decision compare in backward, never the rank-1 outer products.
    model = _model(gate="uniform", k_uniform=2, update_surprise_threshold=100.0)
    state = model.init_prequential_state()
    state, _, _ = model.step(state, 7, 0)
    _, _, f1 = model.step(state, 8, 1)
    assert f1.backward == pointwise_flops(1)


def test_disabling_decay_drops_the_decay_charge():
    model = _model(
        gate="uniform", k_uniform=1, update_surprise_threshold=-1.0, weight_decay_fast=0.0
    )
    d, m, v = model.config.d_model, model.config.m, model.config.vocab_size
    update_no_decay = (
        matmul_flops(1, d, m)
        + pointwise_flops(d)
        + pointwise_flops(v)  # p - e_byte
        + matmul_flops(d, m, 1)
        + matmul_flops(m, v, 1)
        + pointwise_flops(d * m, 2)  # W combine: lr-mult + add (no keep-mult)
        + pointwise_flops(m * v, 2)  # Vmat combine: lr-mult + sub (no keep-mult)
    )
    state = model.init_prequential_state()
    state, _, _ = model.step(state, 7, 0)
    _, _, f1 = model.step(state, 8, 1)
    assert f1.backward == pointwise_flops(1) + update_no_decay  # no d*m + m*v decay term


# --- (b) gate monotonic: higher surprise => K nondecreasing --------------------
def test_gate_is_monotonic_in_surprise():
    model = _model(gate="surprise", k_min=1, k_max=7, k_uniform=4, gate_sensitivity=1.5)
    mean, std = 0.3, 0.1  # fixed running statistics; the gate z-scores surprise against them
    surprises = [i / 10.0 for i in range(11)]  # 0.0 .. 1.0
    ks = [model._settle_depth(s, mean, std) for s in surprises]
    for lo, hi in zip(ks, ks[1:], strict=False):
        assert hi >= lo  # nondecreasing in surprise
    assert ks[0] == model.config.k_min  # low surprise floors at k_min
    assert ks[-1] == model.config.k_max  # high surprise ceils at k_max
    # Uniform mode ignores surprise entirely.
    uniform = _model(gate="uniform", k_uniform=4)
    assert all(uniform._settle_depth(s, mean, std) == 4 for s in surprises)


def test_mean_gated_depth_is_calibrated_near_k_uniform():
    # Surprises z-scored against their own mean/std average to ~k_uniform (matched settling
    # FLOPs) — the win must come from allocation, not from spending more.
    model = _model(gate="surprise", k_min=1, k_max=7, k_uniform=4, gate_sensitivity=1.5)
    rng = np.random.default_rng(0)
    mean, std = 0.5, 0.15
    samples = rng.normal(mean, std, size=4000)
    ks = [model._settle_depth(float(s), mean, std) for s in samples]
    assert abs(float(np.mean(ks)) - model.config.k_uniform) < 0.5


# --- (c) settling reduces loss on a constructed case (once Vmat is non-zero) ---
def test_more_settling_reduces_loss_once_readout_is_nonzero():
    # Scalar latent (m=1) so settling is a 1-D convex descent that monotonically approaches
    # the MAP latent without overshoot (eta*(1 + |w|^2) < 1). W aligns the latent with the
    # hidden; a non-zero readout maps the growing latent onto the target's logit, so more
    # settling iterations strictly lower the cross-entropy.
    model = _model(m=1, gate="uniform")
    d, v = model.config.d_model, model.config.vocab_size
    torch.manual_seed(0)
    w = torch.randn(d)
    w = w / w.norm() * 2.0  # |w|^2 = 4 -> eta*(1+|w|^2) = 0.5 < 1 (monotone descent)
    w_mat = w.unsqueeze(1)  # (d, 1)
    hidden = 3.0 * w  # aligned with w -> the MAP latent is positive
    v_mat = torch.zeros(1, v)
    target = 42
    v_mat[0, target] = 5.0  # the correction boosts the target as the latent grows
    core_logits = torch.zeros(v)
    losses = []
    for k in (0, 1, 2, 4, 8, 16):
        z = model._settle(torch.zeros(1), hidden, w_mat, k)
        logits = core_logits + v_mat.t() @ z
        losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target])).item())
    for lo, hi in zip(losses, losses[1:], strict=False):
        assert hi <= lo + 1e-6  # more settling never increases loss
    assert losses[-1] < losses[0] - 0.1  # and meaningfully reduces it


# --- the readout starts as identity to the core (zero correction at start) -----
def test_zero_correction_at_start_matches_the_core_exactly():
    # Vmat = 0 at reset, so the FIRST model-computed prediction must equal the bare core's
    # (same seed/weights) — predictions diverge only after the readout learns.
    stream = np.frombuffer(b"the quick brown fox. ", dtype=np.uint8)
    torch.manual_seed(0)
    core = build_model("transformer", CORE).eval()
    torch.manual_seed(0)
    pc = _model()
    r_core = prequential_bpb(core, stream, device=CPU, collect_logits=True)
    r_pc = prequential_bpb(pc, stream, device=CPU, collect_logits=True)
    # logits[1] is the first model-computed prediction (byte 1 from byte 0); Vmat is still 0.
    assert torch.allclose(r_core.predicted_logits[1], r_pc.predicted_logits[1], atol=1e-6)
    # The PC module still pays its (charged) settling/readout on top of the core.
    assert r_pc.flops.forward > r_core.flops.forward
    assert r_core.flops.backward == 0 and r_pc.flops.backward > 0


# --- (d) no future leakage ----------------------------------------------------
def test_prediction_at_t_cannot_see_byte_t():
    base = list(range(28))
    k = 12  # predictions for positions <= k use only the shared prefix bytes[0..k-1]
    stream_a = np.array(base, dtype=np.uint8)
    stream_b = np.array(base[: k + 1] + [(b + 5) % 256 for b in base[k + 1 :]], dtype=np.uint8)
    assert stream_a[k + 1] != stream_b[k + 1]
    torch.manual_seed(0)
    model = _model()
    ra = prequential_bpb(model, stream_a, device=CPU, collect_logits=True)
    rb = prequential_bpb(model, stream_b, device=CPU, collect_logits=True)
    for i in range(k + 2):  # logits[i] predicts byte i from bytes[0..i-1]
        assert torch.equal(ra.predicted_logits[i], rb.predicted_logits[i])
    assert not torch.equal(ra.predicted_logits[k + 2], rb.predicted_logits[k + 2])


# --- (e) reproducibility under a fixed seed -----------------------------------
def test_reproducible_under_a_fixed_seed():
    stream = synthetic_text8(400, seed=7).data
    torch.manual_seed(0)
    m1 = _model(gate="surprise")
    torch.manual_seed(0)
    m2 = _model(gate="surprise")
    r1 = prequential_bpb(m1, stream, device=CPU, collect_logits=True)
    r2 = prequential_bpb(m2, stream, device=CPU, collect_logits=True)
    assert r1.bpb == r2.bpb
    for a, b in zip(r1.predicted_logits, r2.predicted_logits, strict=True):
        assert torch.equal(a, b)


# --- (f) registration ---------------------------------------------------------
def test_registration_and_build():
    assert "pc_refine" in list_models()
    model = build_model("pc_refine", TINY)
    assert isinstance(model, PCRefine)


# --- amortized + prequential plumbing -----------------------------------------
def test_forward_and_flops_delegate_to_the_slow_core():
    model = _model()
    x = torch.randint(0, VOCAB_SIZE, (2, 8))
    assert torch.equal(model(x), model.core(x))
    assert model.flops(8) == model.core.flops(8)  # the PC module is eval-only, untrained


def test_prequential_run_total_is_pretrain_plus_eval(tmp_path):
    # eval_bytes (128) > max_seq_len (64) so the run crosses from the growing KV regime into
    # the bounded sliding-window recompute — exercising both decode paths.
    corpus = synthetic_text8(6000, seed=0)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=128)
    cfg = PrequentialConfig(
        model="pc_refine",
        model_config={**TINY, "max_seq_len": 64},
        pretrain_flop_budget=2e8,
        seq_len=64,
        run_name="pc-smoke",
    )
    summary = prequential_run(prior, eval_stream, cfg, runs_dir=tmp_path)
    assert summary.total_flops == summary.pretrain_flops + summary.eval_flops
    assert summary.eval_bytes == 128
    assert 0.0 < summary.bpb <= 8.5
    assert (tmp_path / "pc-smoke.jsonl").exists()


# --- config validation --------------------------------------------------------
def test_config_resolves_d_ff_and_validates():
    assert PCRefineConfig(d_model=32).d_ff == 128
    for bad in (
        {"m": 0},
        {"eta": 0.0},
        {"sigma_h": 0.0},
        {"sigma_z": -1.0},
        {"gate": "bogus"},
        {"k_min": -1},
        {"k_max": 0},
        {"k_uniform": 10, "k_min": 1, "k_max": 7},  # k_uniform > k_max
        {"gate_sensitivity": -1.0},
        {"gate_eps": 0.0},
        {"surprise_ema": 0.0},
        {"lr_readout": -0.1},
        {"weight_decay_fast": 1.0},
    ):
        try:
            PCRefineConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# --- the unigram Pareto baseline (A.1 reflex: keep the win honest) -------------
def _online_unigram_bpb(stream: np.ndarray) -> float:
    """A free predict-then-count adaptive unigram, the Pareto baseline the sweep must beat."""
    counts = np.ones(VOCAB_SIZE)
    bits = 0.0
    for byte in (int(b) for b in stream):
        bits += -math.log2(counts[byte] / counts.sum())
        counts[byte] += 1.0
    return bits / len(stream)


def test_unigram_baseline_is_well_defined():
    # Sanity anchor for the Pareto-hollow check the experiment note must report.
    bpb = _online_unigram_bpb(synthetic_text8(512, seed=1).data)
    assert 0.0 < bpb <= 8.0
