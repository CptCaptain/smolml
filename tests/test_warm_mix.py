"""Warmed context-mixing (Task B.2, Phase 1): the stateful prior->eval handoff.

These mirror ``tests/test_context_mixing.py`` and pin the contract that makes ``warm_mix``
trustworthy as a candidate AND a Tier-0 backbone:

- budget-0 ``warm_mix`` is **bit-identical** to the cold ``context_mixing`` reference
  (predictions and the exact FLOP breakdown), so the warmed result is measured against a
  clean, un-cheated baseline;
- warming on a disjoint prior **strictly lowers** held-out bpb;
- the warmup FLOP charge is the parent's exact per-byte cost summed over every folded byte
  (nothing hidden, nothing double-folded), and is bounded by ``flops(seq_len)``;
- **no leakage** — an eval prediction is invariant to future eval bytes, and an eval pass
  never mutates the persistent warm state (deep-copy isolation);
- warming + eval is reproducible under a fixed seed; registration works.
"""

import numpy as np
import torch

from smolml.data import synthetic_text8
from smolml.flops import FlopBreakdown
from smolml.models import build_model, list_models
from smolml.models.context_mixing import ContextMixing, ContextMixingConfig
from smolml.models.registry import DecodeState
from smolml.models.warm_mix import WarmMix
from smolml.prequential import prequential_bpb, pretrain

CPU = torch.device("cpu")


def _warm(model, prior, *, flop_budget, seq_len=64, batch_size=8, seed=0) -> int:
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


# --- registration ------------------------------------------------------------
def test_registered_and_buildable():
    assert "warm_mix" in list_models()
    model = build_model("warm_mix", {"max_order": 2})
    assert isinstance(model, WarmMix)
    assert model.num_params() == 0  # transductive: no gradient parameters


def test_from_config_ignores_transformer_keys():
    # The CLI injects transformer keys; the inherited from_config keeps only its own fields.
    model = WarmMix.from_config(
        {"max_order": 3, "d_model": 64, "n_layers": 4, "n_heads": 8, "max_seq_len": 128}
    )
    assert isinstance(model, WarmMix)
    assert model.num_predictors == 4


# --- budget 0 == the cold reference, bit for bit ----------------------------
def test_budget_zero_is_bit_identical_to_cold_reference():
    # Never warmed => the deep-copied warm state is a fresh ContextMixing cache, so every
    # prediction AND the whole FLOP breakdown match the cold reference exactly.
    cfg = {"max_order": 3}
    stream = synthetic_text8(2000, seed=0).prequential_carve(eval_bytes=120)[1]
    cold = prequential_bpb(
        build_model("context_mixing", cfg), stream, device=CPU, collect_logits=True
    )
    warm = prequential_bpb(build_model("warm_mix", cfg), stream, device=CPU, collect_logits=True)
    assert warm.predicted_logits is not None
    assert len(warm.predicted_logits) == len(cold.predicted_logits)
    for w_logits, c_logits in zip(warm.predicted_logits, cold.predicted_logits, strict=True):
        assert torch.equal(w_logits, c_logits)
    assert warm.flops == cold.flops
    assert warm.bpb == cold.bpb


# --- warming strictly lowers held-out bpb -----------------------------------
def test_warmup_strictly_lowers_eval_bpb():
    # Prior and eval are structurally disjoint (tail carve), so this is a genuine
    # generalization win, not memorization: warmed count tables + mixer weights start the
    # eval stream below the cold-start curve.
    corpus = synthetic_text8(6000, seed=1)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=1000)
    cfg = {"max_order": 2}
    cold_bpb = prequential_bpb(build_model("context_mixing", cfg), eval_stream, device=CPU).bpb
    warm = build_model("warm_mix", cfg)
    spent = _warm(warm, prior, flop_budget=5e7, seed=0)
    assert spent > 0  # the budget actually warmed something
    warm_bpb = prequential_bpb(warm, eval_stream, device=CPU).bpb
    assert warm_bpb < cold_bpb


# --- honest warmup FLOP accounting ------------------------------------------
def test_warmup_flops_match_parent_per_byte_replay():
    # The warmup charge MUST equal the parent's EXACT per-byte cost summed over every folded
    # byte — no hidden compute, nothing double-folded. Replay the same rows through
    # ContextMixing.step into one shared cache (window + pending-prediction reset per row,
    # exactly as warm_mix folds) and require a FLOP-for-FLOP match.
    cfg = ContextMixingConfig(max_order=2)
    rows = np.array(
        [[72, 73, 74, 75, 76, 77, 32, 72], [72, 73, 80, 81, 72, 73, 74, 75]], dtype=np.int64
    )
    x = torch.from_numpy(rows)
    warm = build_model("warm_mix", {"max_order": 2})
    _, spent = warm.train_step((x, x), optimizer=None)

    ref = ContextMixing(cfg)
    cache = ref.init_prequential_state().cache
    replay = FlopBreakdown()
    for r in range(rows.shape[0]):
        cache.last_stretched = None
        cache.last_probs = None
        state = DecodeState(tokens=[], cache=cache)
        for pos in range(rows.shape[1]):
            state, _, fl = ref.step(state, int(rows[r, pos]), pos)
            replay += fl
    assert spent.forward == replay.forward
    assert spent.backward == replay.backward
    assert spent.total == replay.total


def test_warmup_flops_hand_number_and_within_upper_bound():
    # Single row [65, 66], max_order=1, hand-derived from the parent's per-byte charge:
    #   byte 65: no pending prediction (no mixer update), only order-0 folds ->
    #            forward 3586 + backward 2    = 3588
    #   byte 66: prediction graded (mixer update), both orders fold ->
    #            forward 3586 + backward 1033 = 4619
    #   bytes folded total = 3588 + 4619 = 8207
    x = torch.tensor([[65, 66]])
    warm = build_model("warm_mix", {"max_order": 1})
    _, spent = warm.train_step((x, x), optimizer=None)
    assert spent.total == 8207
    # flops(seq_len) is the steady-state analytic upper bound; the exact charge is strictly
    # less here (byte 0 ramps up), confirming flops() never undercounts the budget guard.
    assert spent.total < warm.flops(2).total
    assert warm.flops(2) == warm._steady_step_flops().scale(2)


# --- no future leakage -------------------------------------------------------
def test_eval_prediction_invariant_to_future_bytes():
    # On a WARMED model: two eval streams identical up to index k, differing at k. Each gets
    # its own deep copy of the same warm state, so predictions for bytes 0..k (made only from
    # earlier bytes) are bit-identical; the prediction for byte k+1 (first to condition on the
    # changed byte) differs. Proves eval never peeks ahead and the warm prior is shared.
    rng = np.random.default_rng(0)
    base = rng.integers(97, 123, size=40).astype(np.uint8)
    k = 20
    a = base.copy()
    b = base.copy()
    b[k] = base[k] ^ 1  # flip one byte at index k
    warm = build_model("warm_mix", {"max_order": 3})
    _warm(warm, synthetic_text8(3000, seed=2).data, flop_budget=2e7, batch_size=4, seed=1)
    ra = prequential_bpb(warm, a, device=CPU, collect_logits=True)
    rb = prequential_bpb(warm, b, device=CPU, collect_logits=True)
    for i in range(k + 1):
        assert torch.equal(ra.predicted_logits[i], rb.predicted_logits[i])
    assert not torch.equal(ra.predicted_logits[k + 1], rb.predicted_logits[k + 1])


def test_eval_does_not_mutate_warm_state():
    # Deep-copy isolation: a full eval pass folds into its OWN copy, leaving the persistent
    # warm tables + mixer weights byte-for-byte unchanged.
    warm = build_model("warm_mix", {"max_order": 2})
    _warm(warm, synthetic_text8(3000, seed=3).data, flop_budget=2e7, batch_size=4, seed=2)
    weights_before = warm._warm.weights.copy()
    tables_before = [{ctx: c.copy() for ctx, c in t.items()} for t in warm._warm.tables]
    stream = synthetic_text8(1500, seed=9).prequential_carve(eval_bytes=400)[1]
    prequential_bpb(warm, stream, device=CPU)
    assert np.array_equal(warm._warm.weights, weights_before)
    assert len(warm._warm.tables) == len(tables_before)
    for after, before in zip(warm._warm.tables, tables_before, strict=True):
        assert after.keys() == before.keys()
        for ctx in before:
            assert np.array_equal(after[ctx], before[ctx])


# --- reproducibility ---------------------------------------------------------
def test_reproducible_under_fixed_seed():
    cfg = {"max_order": 2}
    prior = synthetic_text8(3000, seed=4).data
    eval_stream = synthetic_text8(1500, seed=5).prequential_carve(eval_bytes=400)[1]

    def warmed_run():
        m = build_model("warm_mix", cfg)
        _warm(m, prior, flop_budget=2e7, batch_size=4, seed=7)
        return prequential_bpb(m, eval_stream, device=CPU)

    r1 = warmed_run()
    r2 = warmed_run()
    assert r1.bpb == r2.bpb
    assert r1.eval_flops == r2.eval_flops
