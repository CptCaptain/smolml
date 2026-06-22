"""Online delta-rule fast-weight memory (Task B.4): one error-correcting fast-weight stream
keyed on a fixed sparse signed hashed bag-of-n-grams, mixed into the warmed count ensemble.

The tests pin every load-bearing claim from the spec (``docs/tasks/B.4-delta-mix.md``):
- **degenerate identity** — ``delta_orders=()`` is bit-identical (predictions AND FlopBreakdown)
  to ``hashed_mix`` (the stream delegates entirely to the parent);
- **FLOP honesty** — ``step``'s breakdown equals the parent count breakdown plus the exact
  analytic delta increment, charge == code;
- **no leakage** — a prediction never depends on the byte it is scored against or any later byte;
- **error-correction is load-bearing** — the delta rule beats a plain-Hebbian ablation on a
  superposed-key source;
- **value beyond the count cap** — the delta stream captures order structure the count ladder is
  capped below, and does NOT abstain on novel high-order contexts (the generalization mechanism);
- **leak-free warm handoff + reproducibility**.

The per-FLOP *verdict* (does generalization beat warming the cheap ladder on more bytes?) is the
matched-FLOP kill-test's job, not a unit test — see ``smolml/experiments/delta_mix_enwik8.py``.
"""

import numpy as np
import pytest
import torch

from smolml.data import synthetic_text8
from smolml.flops import gather_flops, pointwise_flops
from smolml.models import build_model, list_models
from smolml.models.delta_mix import _KNUTH2, DeltaMix, DeltaMixConfig
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


def _markov_stream(n: int, *, order: int, alphabet: int, seed: int) -> np.ndarray:
    """A deterministic order-``order`` source: the next byte is a fixed (random) function of the
    last ``order`` bytes. Order-<``order`` models cannot fully predict it; the order-``order``
    context can — useful for probing what the delta stream captures beyond the count cap."""
    rng = np.random.default_rng(seed)
    data = list(int(b) for b in rng.integers(0, alphabet, size=order))
    table: dict[tuple[int, ...], int] = {}
    for _ in range(n - order):
        ctx = tuple(data[-order:])
        if ctx not in table:
            table[ctx] = int(rng.integers(0, alphabet))
        data.append(table[ctx])
    return np.array(data, dtype=np.uint8)


def _bpb(model, stream) -> float:
    return prequential_bpb(model, stream, device=CPU).bpb


# --- registration ------------------------------------------------------------
def test_registered_and_buildable():
    assert "delta_mix" in list_models()
    m = build_model("delta_mix", {"max_order": 3})
    assert isinstance(m, DeltaMix)
    assert m.num_params() == 0  # transductive: W is online state, not an nn.Parameter


def test_from_config_admits_fields_and_ignores_transformer_keys():
    # The CLI injects transformer keys; from_config keeps only DeltaMixConfig fields. A list
    # delta_orders (as JSON would store) is coerced to a tuple.
    m = DeltaMix.from_config(
        {
            "max_order": 5,
            "table_bits": 16,
            "hash_min_order": 4,
            "delta_dim": 1 << 12,
            "delta_orders": [3, 4, 5],
            "delta_eta": 0.2,
            "d_model": 64,
            "n_layers": 3,
            "n_heads": 4,
            "max_seq_len": 128,
        }
    )
    assert m.num_predictors == 6
    assert m.config.delta_orders == (3, 4, 5)
    assert m.config.delta_dim == 1 << 12


def test_config_validation():
    with pytest.raises(ValueError):
        DeltaMixConfig(delta_dim=3)  # not a power of two
    with pytest.raises(ValueError):
        DeltaMixConfig(delta_eta=0.0)
    with pytest.raises(ValueError):
        DeltaMixConfig(delta_orders=(0, 3))  # entry < 1
    with pytest.raises(ValueError):
        DeltaMixConfig(delta_orders=(3, 9))  # n-gram > 8 bytes would alias the hash
    DeltaMixConfig(delta_orders=())  # empty disables the stream — valid


# --- degenerate identity: delta off == hashed_mix, bit for bit ----------------
def test_delta_disabled_is_bit_identical_to_hashed_mix():
    # delta_orders=() => every override delegates to super(); warmed delta_mix is byte-for-byte
    # hashed_mix — same warm state, same predictions, same FlopBreakdown on every eval step.
    prior = synthetic_text8(4000, seed=1).data
    eval_stream = synthetic_text8(1500, seed=2).prequential_carve(eval_bytes=300)[1]
    cfg = {"max_order": 5, "hash_min_order": 4, "table_bits": 14, "alpha": 0.5, "lr": 0.02}

    hashed = build_model("hashed_mix", cfg)
    _warm(hashed, prior, flop_budget=2e7)
    delta = build_model("delta_mix", {**cfg, "delta_orders": ()})
    _warm(delta, prior, flop_budget=2e7)

    hs = hashed.init_prequential_state()
    ds = delta.init_prequential_state()
    for pos in range(len(eval_stream) - 1):
        b = int(eval_stream[pos])
        hs, hl, hf = hashed.step(hs, b, pos)
        ds, dl, df = delta.step(ds, b, pos)
        assert torch.equal(hl, dl)
        assert hf == df


# --- feature map --------------------------------------------------------------
def test_delta_slot_deterministic_bounded_signed():
    m = build_model("delta_mix", {"delta_dim": 1 << 10})
    i1, s1 = m._delta_slot(b"abc")
    assert (i1, s1) == m._delta_slot(b"abc")  # deterministic within an instance
    assert 0 <= i1 < (1 << 10)
    assert s1 in (-1.0, 1.0)
    # Salt-free: a fresh instance/process hashes identically (warmed runs reproduce).
    m2 = build_model("delta_mix", {"delta_dim": 1 << 10})
    assert m2._delta_slot(b"abc") == (i1, s1)
    # Distinct n-grams generally land in distinct buckets and need not share a sign.
    assert _KNUTH2 != 0


def test_delta_slot_unsigned_is_all_plus():
    m = build_model("delta_mix", {"delta_dim": 1 << 10, "delta_signed": False})
    assert all(m._delta_slot(bytes([x, x + 1]))[1] == 1.0 for x in range(0, 200, 7))


def test_build_phi_sparsity_tracks_active_orders():
    m = build_model("delta_mix", {"delta_orders": (3, 4, 5, 6, 7, 8)})
    idxs, signs = m._build_phi([1, 2, 3, 4, 5])  # 5 bytes -> orders 3,4,5 fit
    assert idxs.shape == (3,)
    assert signs.shape == (3,)
    assert set(signs.tolist()) <= {-1.0, 1.0}
    assert m._build_phi([1, 2])[0].shape == (0,)  # < 3 bytes -> no delta feature


# --- prequential smoke --------------------------------------------------------
def test_prequential_runs_and_is_finite():
    m = build_model(
        "delta_mix",
        {"max_order": 3, "delta_dim": 1 << 12, "delta_orders": (3, 4, 5), "delta_eta": 0.3},
    )
    r = prequential_bpb(m, synthetic_text8(800, seed=3).data, device=CPU)
    assert np.isfinite(r.bpb)
    assert 0.0 < r.bpb <= 8.0  # never worse than the uniform prior over a stream


# --- no leakage ---------------------------------------------------------------
def test_prediction_at_t_cannot_see_future():
    # The prediction for byte j uses only bytes 0..j-1; the delta W update for the byte revealed
    # at pos uses only that byte. So perturbing byte 400 leaves predictions 0..400 untouched and
    # changes prediction 401 (the first that folds byte 400).
    cfg = {"max_order": 4, "delta_dim": 1 << 12, "delta_orders": (3, 4, 5, 6), "delta_eta": 0.3}
    stream = synthetic_text8(700, seed=5).data
    r1 = prequential_bpb(build_model("delta_mix", cfg), stream, device=CPU, collect_logits=True)
    perturbed = stream.copy()
    perturbed[400] = (int(perturbed[400]) + 7) % 256
    r2 = prequential_bpb(build_model("delta_mix", cfg), perturbed, device=CPU, collect_logits=True)
    for j in range(401):  # predictions for bytes 0..400 cannot see byte 400
        assert torch.equal(r1.predicted_logits[j], r2.predicted_logits[j])
    assert not torch.equal(r1.predicted_logits[401], r2.predicted_logits[401])


# --- FLOP honesty: step charge == parent count charge + the analytic delta increment ---
def test_step_flops_equal_hashed_plus_delta_increment():
    cfg = {"max_order": 4, "hash_min_order": 2, "table_bits": 16, "alpha": 0.5, "lr": 0.02}
    dcfg = {**cfg, "delta_orders": (3, 4, 5, 6), "delta_dim": 1 << 12, "delta_eta": 0.3}
    hashed = build_model("hashed_mix", cfg)
    delta = build_model("delta_mix", dcfg)
    hs = hashed.init_prequential_state()
    ds = delta.init_prequential_state()
    stream = synthetic_text8(400, seed=8).data
    cap = delta._window_cap
    orders = delta.config.delta_orders
    prev_nd = 0
    total_inc = 0
    for pos in range(len(stream)):
        b = int(stream[pos])
        hs, _, hf = hashed.step(hs, b, pos)  # count side evolves identically (same keys/counts)
        ds, _, df = delta.step(ds, b, pos)
        nd = sum(1 for n in orders if min(pos + 1, cap) >= n)
        did_update = pos >= 1  # last_probs is set after the first step
        inc = delta._delta_increment(nd=nd, nd_prev=prev_nd, did_update=did_update)
        assert df.forward == hf.forward + inc.forward
        assert df.backward == hf.backward + inc.backward
        total_inc += inc.forward + inc.backward
        prev_nd = nd
    assert total_inc > 0  # the delta stream really did charge work over the stream


def test_delta_increment_matches_hand_formula():
    # Pin the exact symbolic charge so a regression in the accounting fails here.
    m = build_model("delta_mix", {"delta_dim": 1 << 12, "delta_orders": (3, 4, 5)})
    v = m.config.vocab_size
    nd, nd_prev = 3, 2
    inc = m._delta_increment(nd=nd, nd_prev=nd_prev, did_update=True)
    exp_fwd = gather_flops(nd) + pointwise_flops(6 * nd + 2 * nd * v + 2 * v + 5 * v)
    exp_bwd = (
        pointwise_flops(2 * v + 2)
        + gather_flops(nd_prev)
        + pointwise_flops(1 + v + 2 * nd_prev * v)
    )
    assert inc.forward == exp_fwd
    assert inc.backward == exp_bwd
    # No pending prediction -> no backward delta cost at all.
    assert m._delta_increment(nd=nd, nd_prev=0, did_update=False).backward == 0
    # Pending but the prior key had no support -> only the (K+1)-th mixer terms, no W write.
    assert m._delta_increment(nd=nd, nd_prev=0, did_update=True).backward == pointwise_flops(
        2 * v + 2
    )


# --- error-correction is load-bearing (delta rule beats plain Hebbian) --------
class _HebbianMix(DeltaMix):
    """Ablation: replace the error-correcting delta write with a plain Hebbian one
    (``W[byte, j] += eta * sign``, no error term). On a superposed key this collapses toward
    the byte marginal — the failure mode the delta rule's residual update avoids."""

    def _apply_delta_update(self, ms, pidx, psign, revealed_byte):
        ms.W[revealed_byte, pidx] += self.config.delta_eta * psign


def test_error_correction_beats_hebbian_on_superposed_key():
    # An order-3 deterministic source carried ONLY by the delta stream (max_order=0 => counts give
    # just the global marginal). The key superposes orders 1..4, so low orders are shared across
    # many contexts: the error-correcting rule learns each bucket's residual and decorrelates;
    # plain Hebbian double-counts the shared features and stays blurry.
    stream = _markov_stream(6000, order=3, alphabet=6, seed=11)
    cfg = {"max_order": 0, "delta_dim": 1 << 14, "delta_orders": (1, 2, 3, 4), "delta_eta": 0.3}
    delta_bpb = _bpb(build_model("delta_mix", cfg), stream)
    hebb_bpb = _bpb(_HebbianMix(DeltaMixConfig(**cfg)), stream)
    assert delta_bpb < hebb_bpb - 0.05  # clear, not within noise


# --- value beyond the count cap + no abstention on novel high-order contexts --
def test_delta_captures_structure_beyond_count_cap():
    # Order-3 deterministic source; cap the counts at order-1 (cannot represent the rule) and let
    # the delta stream carry orders 2,3. The delta stream learns the order-3 structure, so the
    # full model beats a count-only hashed_mix with the SAME order-1 cap by a wide margin.
    stream = _markov_stream(8000, order=3, alphabet=6, seed=13)
    base = {"max_order": 1, "alpha": 0.5, "lr": 0.02}
    capped = _bpb(
        build_model("hashed_mix", {**base, "hash_min_order": 2, "table_bits": 16}), stream
    )
    delta = _bpb(
        build_model(
            "delta_mix",
            {
                **base,
                "hash_min_order": 2,
                "table_bits": 16,
                "delta_dim": 1 << 14,
                "delta_orders": (2, 3, 4),
                "delta_eta": 0.3,
            },
        ),
        stream,
    )
    assert delta < capped - 0.3  # the delta stream captures structure the order-1 ladder cannot


def test_delta_stream_does_not_abstain_on_novel_high_order_context():
    # The generalization MECHANISM: an exact order-k count abstains (uniform) on a never-seen
    # k-gram, but the distributed delta key carries signal from the lower-order suffixes it HAS
    # seen. After warming, the delta stream's standalone prediction beats the 8-bit uniform on a
    # context whose full n-gram is novel but whose order-2 suffix recurs in the source.
    stream = _markov_stream(6000, order=2, alphabet=5, seed=17)  # order-2 rule, long contexts
    m = build_model(
        "delta_mix",
        {"max_order": 0, "delta_dim": 1 << 16, "delta_orders": (2, 3, 4, 5), "delta_eta": 0.3},
    )
    # Warm the live online state by folding the stream (prequential_bpb does not return state).
    state = m.init_prequential_state()
    for pos in range(len(stream) - 200):
        state, _, _ = m.step(state, int(stream[pos]), pos)
    ms = state.cache
    # Build a NOVEL long context: a fresh 5-gram whose order-2 suffix obeys the source's rule.
    suffix = [int(stream[10]), int(stream[11])]  # a 2-gram that occurs in the source
    novel = [200, 201, 202, *suffix]  # leading bytes never seen -> the full 5-gram is novel
    idxs, signs = m._build_phi(novel)
    z_delta = (ms.W[:, idxs] * signs[None, :]).sum(axis=1)
    # The order-2 rule's true continuation for `suffix`:
    from collections import Counter

    nxt = Counter()
    for pos in range(1, len(stream) - 1):
        if int(stream[pos - 1]) == suffix[0] and int(stream[pos]) == suffix[1]:
            nxt[int(stream[pos + 1])] += 1
    true_byte = nxt.most_common(1)[0][0]
    log_probs = torch.log_softmax(torch.from_numpy(z_delta.astype(np.float32)), dim=-1)
    bits_true = float(-log_probs[true_byte].item() / np.log(2))
    assert bits_true < 8.0  # carries real signal — does NOT abstain to uniform like an exact count


# --- leak-free warm handoff + reproducibility ---------------------------------
def test_eval_does_not_mutate_warm_state():
    cfg = {
        "max_order": 4,
        "hash_min_order": 4,
        "table_bits": 12,
        "delta_dim": 1 << 12,
        "delta_orders": (3, 4, 5),
        "delta_eta": 0.3,
    }
    model = build_model("delta_mix", cfg)
    _warm(model, synthetic_text8(3000, seed=1).data, flop_budget=3e7)
    warm_w = model._warm.W.copy()
    warm_weights = model._warm.weights.copy()
    eval_stream = synthetic_text8(1200, seed=2).prequential_carve(eval_bytes=400)[1]
    prequential_bpb(model, eval_stream, device=CPU)  # a full eval pass mutates only its own copy
    assert np.array_equal(model._warm.W, warm_w)
    assert np.array_equal(model._warm.weights, warm_weights)


def test_two_eval_streams_are_isolated():
    cfg = {"max_order": 3, "delta_dim": 1 << 12, "delta_orders": (3, 4), "delta_eta": 0.3}
    model = build_model("delta_mix", cfg)
    _warm(model, synthetic_text8(2000, seed=1).data, flop_budget=2e7)
    s1 = model.init_prequential_state()
    s2 = model.init_prequential_state()
    for pos in range(60):
        s1, _, _ = model.step(s1, pos % 7, pos)
    assert not np.array_equal(s1.cache.W, s2.cache.W)  # s1's writes never reach s2's copy
    assert np.array_equal(s2.cache.W, model._warm.W)  # untouched copy still equals the warm state


def test_reproducible_under_fixed_seed():
    cfg = {
        "max_order": 4,
        "hash_min_order": 4,
        "table_bits": 14,
        "delta_dim": 1 << 13,
        "delta_orders": (3, 4, 5, 6),
        "delta_eta": 0.3,
    }
    prior = synthetic_text8(3000, seed=1).data
    eval_stream = synthetic_text8(1200, seed=2).prequential_carve(eval_bytes=400)[1]
    b = []
    for _ in range(2):
        m = build_model("delta_mix", cfg)
        _warm(m, prior, flop_budget=3e7, seed=99)
        b.append(_bpb(m, eval_stream))
    assert b[0] == b[1]
