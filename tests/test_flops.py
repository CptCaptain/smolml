"""Hand-computed checks for the FLOP counter — the critical correctness surface.

Every number below is derived by hand in the comments so a reviewer can confirm
the accounting without trusting the implementation.
"""

from smolml.flops import (
    BACKWARD_MULTIPLIER,
    MAC_FLOPS,
    FlopBreakdown,
    causal_attention_flops,
    gather_flops,
    linear_flops,
    matmul_flops,
    pointwise_flops,
)


def test_matmul_is_two_mnk():
    # (2,3) @ (3,4): 2*4 = 8 outputs, each a length-3 dot product = 3 MACs.
    # FLOPs = 2 * m * n * k = 2 * 2 * 4 * 3 = 48.
    assert matmul_flops(2, 4, 3) == 48
    assert MAC_FLOPS == 2


def test_linear_matches_matmul():
    # Linear(in=5, out=7) over 11 tokens == matmul (11,5)@(5,7) = 2*11*7*5 = 770.
    assert linear_flops(11, 5, 7) == 2 * 11 * 7 * 5
    assert linear_flops(11, 5, 7) == 770


def test_causal_attention_pairs_and_head_independence():
    # T=4 -> causal (query,key) pairs P = 4*5/2 = 10.
    # attention = 2 * MAC_FLOPS * d * P = 2 * 2 * 8 * 10 = 320.
    assert causal_attention_flops(4, 8) == 320
    # Independent of head count: only d_model enters the formula.
    assert causal_attention_flops(16, 64) == 2 * 2 * 64 * (16 * 17 // 2)


def test_pointwise_and_gather_primitives():
    # Non-matmul primitives for lookup/mixing-dominated mechanisms (Task 0.3).
    assert pointwise_flops(10) == 10  # default 1 op/element
    assert pointwise_flops(10, per_elem=3) == 30
    assert gather_flops(7) == 7  # nominal 1 op/lookup
    assert gather_flops(7, cost_per_lookup=0) == 0


def test_breakdown_total_and_backward_multiplier():
    b = FlopBreakdown(forward=100, backward=200)
    assert b.total == 300
    fwd = FlopBreakdown.from_forward(100)
    assert fwd.backward == BACKWARD_MULTIPLIER * 100 == 200
    assert fwd.total == 300


def test_breakdown_add_and_scale():
    a = FlopBreakdown(forward=10, backward=20)
    b = FlopBreakdown(forward=1, backward=2)
    assert (a + b) == FlopBreakdown(11, 22)
    assert a.scale(3) == FlopBreakdown(30, 60)


def test_transformer_layer_stack_hand_computed():
    """Full forward/backward FLOPs for a known tiny transformer, by hand.

    Config: d_model=8, n_layers=2, d_ff=16, vocab=256, seq_len T=4.
    (Head count does not affect attention FLOPs.)

    Per layer, per sequence (T tokens), forward matmul FLOPs:
      qkv  : Linear(d -> 3d) over T tokens = 2*T*(3d)*d = 6*d^2*T = 6*64*4 = 1536
      out  : Linear(d -> d)  over T tokens = 2*T*d*d     = 2*d^2*T = 2*64*4 =  512
      ffn  : Linear(d->d_ff)+Linear(d_ff->d)            = 4*d*d_ff*T          = 2048
             (= 2*T*d_ff*d + 2*T*d*d_ff = 4*8*16*4)
      attn : 4*d*P, P = T*(T+1)/2 = 10                  = 4*8*10              =  320
      ----> per layer = 1536 + 512 + 2048 + 320 = 4416
    blocks  = n_layers * per_layer = 2 * 4416 = 8832
    head    : Linear(d -> vocab) over T tokens = 2*T*d*V = 2*4*8*256          = 16384
    forward = blocks + head = 8832 + 16384 = 25216
    backward = 2 * forward = 50432
    total    = 75648
    """
    d, n_layers, d_ff, vocab, t = 8, 2, 16, 256, 4

    qkv = linear_flops(t, d, 3 * d)
    out = linear_flops(t, d, d)
    ffn = linear_flops(t, d, d_ff) + linear_flops(t, d_ff, d)
    attn = causal_attention_flops(t, d)
    per_layer = qkv + out + ffn + attn
    assert (qkv, out, ffn, attn, per_layer) == (1536, 512, 2048, 320, 4416)

    blocks = n_layers * per_layer
    head = linear_flops(t, d, vocab)
    forward = blocks + head
    assert (blocks, head, forward) == (8832, 16384, 25216)

    bd = FlopBreakdown.from_forward(forward)
    assert bd.forward == 25216
    assert bd.backward == 50432
    assert bd.total == 75648
