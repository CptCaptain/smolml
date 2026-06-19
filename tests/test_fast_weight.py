"""Fast-weight associative-memory candidate (Task A.1).

Protects what makes this candidate trustworthy: the memory write/read does what it
claims, every memory FLOP flows through ``step``'s honest breakdown (charge ==
reality), and the headline low-budget win over the transformer is real and stable --
while being honest that a *free* online unigram dominates both models in that regime.
"""

import math

import numpy as np
import torch

from smolml.data import synthetic_text8
from smolml.data.corpus import VOCAB_SIZE
from smolml.flops import matmul_flops, pointwise_flops
from smolml.models import build_model
from smolml.models.fast_weight import FastWeightConfig, FastWeightMemory
from smolml.prequential import PrequentialConfig, prequential_bpb, prequential_run

CPU = torch.device("cpu")
TINY = {"d_model": 16, "n_layers": 2, "n_heads": 2, "max_seq_len": 64}


def _model(**overrides) -> FastWeightMemory:
    cfg = {**TINY, **overrides}
    model = build_model("fast_weight", cfg)
    model.eval()
    return model


def _online_unigram_bpb(stream: np.ndarray) -> float:
    """A free predict-then-count adaptive unigram (~1e5 FLOPs), the Pareto baseline."""
    counts = np.ones(VOCAB_SIZE)
    bits = 0.0
    for byte in (int(b) for b in stream):
        bits += -math.log2(counts[byte] / counts.sum())
        counts[byte] += 1.0
    return bits / len(stream)


# --- memory write / read correctness -----------------------------------------
def test_write_then_read_recalls_the_written_byte():
    model = _model(memory_decay=1.0, memory_write_gain=1.0)
    d, v = model.config.d_model, model.config.vocab_size
    key = torch.randn(d)
    key = key / key.norm()
    memory = model._write(torch.zeros(d, v), key, 65)
    # With an empty store + unit gain, column 65 is exactly the key; recall of the
    # same key is |key|^2 = 1 on byte 65 and 0 elsewhere.
    assert torch.allclose(memory[:, 65], key)
    recall = key @ memory
    assert recall.argmax().item() == 65
    assert math.isclose(recall[65].item(), 1.0, rel_tol=1e-5)
    assert torch.count_nonzero(recall) == 1


def test_distinct_keys_recall_distinct_bytes_and_accumulate():
    model = _model(memory_decay=1.0)
    d, v = model.config.d_model, model.config.vocab_size
    # two (near-)orthogonal keys associated with two different bytes
    k1 = torch.zeros(d)
    k1[0] = 1.0
    k2 = torch.zeros(d)
    k2[1] = 1.0
    memory = torch.zeros(d, v)
    memory = model._write(memory, k1, 65)
    memory = model._write(memory, k2, 66)
    assert (k1 @ memory).argmax().item() == 65
    assert (k2 @ memory).argmax().item() == 66
    # writing the same association again accumulates (superposition -> stronger recall)
    before = (k1 @ memory)[65].item()
    memory = model._write(memory, k1, 65)
    assert (k1 @ memory)[65].item() > before


def test_decay_forgets_old_writes():
    model = _model(memory_decay=0.5, memory_write_gain=1.0)
    d, v = model.config.d_model, model.config.vocab_size
    k = torch.zeros(d)
    k[0] = 1.0
    memory = model._write(torch.zeros(d, v), k, 65)  # recall 65 == 1.0
    # a later unrelated write decays the earlier association by the factor 0.5.
    other = torch.zeros(d)
    other[1] = 1.0
    memory = model._write(memory, other, 66)
    assert math.isclose((k @ memory)[65].item(), 0.5, rel_tol=1e-6)


# --- memory FLOPs are counted in step's breakdown (charge == reality) ---------
def test_step_flops_equal_core_plus_memory():
    model = _model(memory_decay=0.999)
    d, v = model.config.d_model, model.config.vocab_size
    read = matmul_flops(1, v, d) + pointwise_flops(d, 6) + pointwise_flops(v, 8)
    write = matmul_flops(d, v, 1) + pointwise_flops(d * v)

    state = model.init_prequential_state()
    # Step 0: no pending key -> no write (backward 0); read still happens (forward).
    state, _, f0 = model.step(state, 7, 0)
    assert f0.backward == 0
    assert f0.forward == model.core.decode_step_flops(1).forward + read
    # Step 1: the previous key is written (decay + outer product) -> backward > 0.
    state, _, f1 = model.step(state, 8, 1)
    assert f1.backward == write
    assert f1.forward == model.core.decode_step_flops(2).forward + read


def test_disabling_decay_drops_the_decay_charge():
    model = _model(memory_decay=1.0)
    d, v = model.config.d_model, model.config.vocab_size
    state = model.init_prequential_state()
    state, _, _ = model.step(state, 7, 0)
    _, _, f1 = model.step(state, 8, 1)
    assert f1.backward == matmul_flops(d, v, 1)  # write only, no dV decay term


# --- the memory adds NOTHING but a counted read when its gate is zero ----------
def test_zero_gate_matches_the_frozen_core_exactly():
    # alpha=0 => the recall mixture weight is zero, so the candidate's prediction
    # distribution must equal the bare transformer's with the SAME pretrained core;
    # the ONLY extra FLOPs are the (still-charged) memory read/write. This pins the
    # decode path to the baseline and proves the memory is a pure, honest add-on.
    stream = np.frombuffer(b"the quick brown fox. " * 6, dtype=np.uint8)
    torch.manual_seed(0)
    transformer = build_model("transformer", TINY).eval()
    torch.manual_seed(0)
    fast = _model(memory_alpha=0.0)
    r_tr = prequential_bpb(transformer, stream, device=CPU, collect_logits=True)
    r_fw = prequential_bpb(fast, stream, device=CPU, collect_logits=True)
    for a, b in zip(r_tr.predicted_logits, r_fw.predicted_logits, strict=True):
        assert torch.allclose(torch.softmax(a, -1), torch.softmax(b, -1), atol=1e-6)
    # core forward identical; fast-weight pays exactly the memory read/write on top.
    assert r_fw.flops.forward > r_tr.flops.forward
    assert r_tr.flops.backward == 0 and r_fw.flops.backward > 0


def test_sliding_regime_zero_gate_still_matches_core():
    # A stream longer than the context window forces the bounded sliding recompute;
    # the zero-gate candidate must still track the bare transformer exactly there.
    cfg = {"d_model": 16, "n_layers": 2, "n_heads": 2, "max_seq_len": 16}
    stream = np.frombuffer(bytes((i * 7 + 3) % 256 for i in range(40)), dtype=np.uint8)
    torch.manual_seed(0)
    transformer = build_model("transformer", cfg).eval()
    torch.manual_seed(0)
    fast = build_model("fast_weight", {**cfg, "memory_alpha": 0.0}).eval()
    r_tr = prequential_bpb(transformer, stream, device=CPU, collect_logits=True)
    r_fw = prequential_bpb(fast, stream, device=CPU, collect_logits=True)
    for a, b in zip(r_tr.predicted_logits, r_fw.predicted_logits, strict=True):
        assert torch.allclose(torch.softmax(a, -1), torch.softmax(b, -1), atol=1e-5)


# --- behavioral: the memory measurably lowers bpb on repeated substrings ------
def test_memory_lowers_bpb_on_repeated_substrings():
    # A fixed phrase repeated many times: identical frozen-core context -> identical
    # key -> exact recall of the byte that followed last time. The hybrid should
    # beat its OWN frozen slow core (same seed/weights) by a wide margin.
    stream = np.frombuffer(b"the quick brown fox jumps over the lazy dog. " * 8, dtype=np.uint8)
    torch.manual_seed(0)
    core_only = build_model("transformer", TINY).eval()
    torch.manual_seed(0)
    hybrid = _model()  # same seed -> identical pretrained-equivalent core weights
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            core_only.state_dict().values(), hybrid.core.state_dict().values(), strict=True
        )
    )
    r_core = prequential_bpb(core_only, stream, device=CPU)
    r_hybrid = prequential_bpb(hybrid, stream, device=CPU)
    assert r_hybrid.bpb < r_core.bpb - 1.0
    # ...and its online adaptation compute is counted (frozen core spends none).
    assert r_core.flops.backward == 0
    assert r_hybrid.flops.backward > 0


# --- no future leakage --------------------------------------------------------
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


# --- amortized + prequential plumbing -----------------------------------------
def test_forward_and_flops_delegate_to_the_slow_core():
    model = _model()
    x = torch.randint(0, VOCAB_SIZE, (2, 8))
    assert torch.equal(model(x), model.core(x))
    assert model.flops(8) == model.core.flops(8)  # memory is eval-only, untrained


def test_prequential_run_total_is_pretrain_plus_eval(tmp_path):
    corpus = synthetic_text8(8000, seed=0)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=128)
    cfg = PrequentialConfig(
        model="fast_weight",
        model_config={"d_model": 32, "n_layers": 2, "n_heads": 2, "max_seq_len": 256},
        pretrain_flop_budget=2e8,
        seq_len=64,
        run_name="fw-smoke",
    )
    summary = prequential_run(prior, eval_stream, cfg, runs_dir=tmp_path)
    assert summary.total_flops == summary.pretrain_flops + summary.eval_flops
    assert summary.eval_bytes == 128
    assert 0.0 < summary.bpb <= 8.5
    assert (tmp_path / "fw-smoke.jsonl").exists()


def test_config_resolves_d_ff_and_validates_memory():
    assert FastWeightConfig(d_model=32).d_ff == 128
    for bad in (
        {"memory_decay": 0.0},
        {"memory_decay": 1.5},
        {"memory_alpha": 1.5},
        {"memory_beta": -1.0},
        {"memory_center_ema": 0.0},
    ):
        try:
            FastWeightConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# --- headline: the low-budget win is real, stable, and Pareto-hollow ----------
def test_low_budget_win_is_real_and_pareto_hollow():
    # At budget 0 (untrained, equal-total-FLOP) the hybrid beats its identical
    # transformer core on EVERY i.i.d. stream (a real, stable effect, not noise) --
    # but a free online unigram dominates BOTH, so the win is Pareto-hollow. Testing
    # both halves keeps the headline from ever being oversold.
    cfg = {"d_model": 16, "n_layers": 2, "n_heads": 2, "max_seq_len": 256}
    deltas = []
    for seed in range(5):
        stream = synthetic_text8(256, seed=3000 + seed).data
        torch.manual_seed(0)
        transformer = build_model("transformer", cfg).eval()
        torch.manual_seed(0)
        hybrid = build_model("fast_weight", cfg).eval()
        tr = prequential_bpb(transformer, stream, device=CPU).bpb
        fw = prequential_bpb(hybrid, stream, device=CPU).bpb
        uni = _online_unigram_bpb(stream)
        assert fw < tr  # the equal-total-FLOP win, on every stream (stability)
        assert uni < fw and uni < tr  # ...but the free baseline dominates both
        deltas.append(fw - tr)
    assert np.mean(deltas) < -0.2  # robust margin (measured ~-0.54)
