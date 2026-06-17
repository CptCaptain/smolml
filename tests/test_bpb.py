"""Hand-checkable bits-per-byte correctness."""

import math

import torch

from smolml.eval import bits_per_byte, cross_entropy_bits


def test_uniform_over_256_bytes_is_8_bpb():
    # Equal logits -> uniform softmax over 256 -> -log2(1/256) = 8 bits per byte,
    # regardless of which byte is the target.
    logits = torch.zeros(10, 256)
    targets = torch.randint(0, 256, (10,))
    assert math.isclose(bits_per_byte(logits, targets), 8.0, rel_tol=1e-6)


def test_uniform_over_k_classes_is_log2_k():
    # Equal logits over k classes -> exactly log2(k) bits.
    for k in (2, 4, 16):
        logits = torch.zeros(7, k)
        targets = torch.randint(0, k, (7,))
        assert math.isclose(bits_per_byte(logits, targets), math.log2(k), rel_tol=1e-6)


def test_p_half_is_one_bit():
    # Two classes, equal logits -> p(true) = 0.5 -> -log2(0.5) = 1 bit.
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([0])
    assert math.isclose(bits_per_byte(logits, targets), 1.0, rel_tol=1e-6)


def test_confident_correct_prediction():
    # Logits [ln 9, 0] -> softmax = [0.9, 0.1]; target 0 -> -log2(0.9) bits.
    logits = torch.tensor([[math.log(9.0), 0.0]])
    targets = torch.tensor([0])
    expected = -math.log2(0.9)  # ~= 0.152
    assert math.isclose(bits_per_byte(logits, targets), expected, rel_tol=1e-6)


def test_total_bits_aggregates_across_a_batch():
    # 3 positions, uniform over 4 classes -> 2 bits each -> 6 bits total, n=3.
    logits = torch.zeros(3, 4)
    targets = torch.tensor([0, 1, 2])
    total_bits, n = cross_entropy_bits(logits, targets)
    assert n == 3
    assert math.isclose(total_bits, 6.0, rel_tol=1e-6)
