"""Prequential-mode guarantees: no leakage, honest FLOP accounting, hand-checked bpb.

These protect the *online* metric the same way ``test_metric_guards.py`` protects
the amortized one.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from smolml.data import synthetic_text8
from smolml.data.corpus import VOCAB_SIZE, ByteCorpus
from smolml.flops import FlopBreakdown
from smolml.models import Transformer, TransformerConfig
from smolml.models.registry import DecodeState, LanguageModel
from smolml.prequential import (
    PrequentialConfig,
    prequential_bpb,
    prequential_run,
    score_bits,
)

CPU = torch.device("cpu")


@dataclass
class _StubConfig:
    max_seq_len: int = 100_000
    vocab_size: int = VOCAB_SIZE


class _ConstantModel(LanguageModel):
    """A model that always predicts the same fixed logits (ignores context).

    Makes prequential bpb hand-computable: every byte costs −log2 p of the same
    distribution, so the cumulative bits are exactly predictable.
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.config = _StubConfig()
        self._logits = logits

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        return self._logits.view(1, 1, -1).expand(b, t, -1)

    def flops(self, seq_len: int) -> FlopBreakdown:
        return FlopBreakdown(forward=seq_len, backward=0)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "_ConstantModel":
        return cls(torch.zeros(VOCAB_SIZE))

    # Prequential: constant prediction, O(1) observe charging 1 decode FLOP.
    def init_prequential_state(self) -> DecodeState:
        return DecodeState(next_logits=self._logits)

    def observe(
        self, state: DecodeState, token: int, pos: int
    ) -> tuple[DecodeState, FlopBreakdown]:
        return DecodeState(next_logits=self._logits), self.decode_step_flops(pos + 1)

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        return FlopBreakdown(forward=1, backward=0)


class _AdaptingModel(_ConstantModel):
    """A constant model that ALSO pays a fixed cost every adaptation step."""

    ADAPT = FlopBreakdown(forward=10, backward=20)

    def adapt(self, state, optimizer, *, grad_clip=1.0):
        return state, self.ADAPT

    def adapt_step_flops(self, context_len: int) -> FlopBreakdown:
        return self.ADAPT


# --- (c) hand-computed prequential bpb ---------------------------------------
def test_uniform_constant_model_is_8_bpb():
    model = _ConstantModel(torch.zeros(VOCAB_SIZE))  # uniform over 256
    stream = np.frombuffer(bytes([1, 2, 3, 4, 5, 6, 7, 8]), dtype=np.uint8)
    result = prequential_bpb(model, stream, device=CPU)
    assert math.isclose(result.bpb, 8.0, rel_tol=1e-6)
    assert result.n_bytes == 8


def test_constant_model_with_known_probability():
    # logits: byte 65 gets logit ln(255), all others 0 -> p(65) = 255/510 = 0.5.
    logits = torch.zeros(VOCAB_SIZE)
    logits[65] = math.log(255.0)
    model = _ConstantModel(logits)
    stream = np.full(6, 65, dtype=np.uint8)  # all 'A' -> each byte costs -log2(0.5) = 1 bit
    result = prequential_bpb(model, stream, device=CPU)
    assert math.isclose(result.bpb, 1.0, rel_tol=1e-4)
    assert math.isclose(result.total_bits, 6.0, rel_tol=1e-4)


def test_score_bits_hand_cases():
    assert math.isclose(score_bits(torch.zeros(256), 0), 8.0, rel_tol=1e-6)
    assert math.isclose(score_bits(torch.zeros(2), 0), 1.0, rel_tol=1e-6)


# --- (a) no future leakage ---------------------------------------------------
def test_prediction_at_t_cannot_see_byte_t():
    torch.manual_seed(0)
    model = Transformer(
        TransformerConfig(d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=64)
    )
    model.eval()
    base = list(range(24))
    k = 10  # predictions for positions 0..k depend only on bytes 0..k-1 (shared)
    stream_a = np.array(base, dtype=np.uint8)
    stream_b = np.array(base[: k + 1] + [(b + 1) % 256 for b in base[k + 1 :]], dtype=np.uint8)
    assert stream_a[k + 1] != stream_b[k + 1]  # the futures genuinely differ

    ra = prequential_bpb(model, stream_a, device=CPU, collect_logits=True)
    rb = prequential_bpb(model, stream_b, device=CPU, collect_logits=True)
    # The distribution for byte k+1 is computed BEFORE it is revealed, from the
    # shared prefix bytes[0..k] -> it must be byte-for-byte identical.
    for i in range(k + 2):
        assert torch.equal(ra.predicted_logits[i], rb.predicted_logits[i])
    # And once the streams diverge (byte k+1 observed), later predictions differ.
    assert not torch.equal(ra.predicted_logits[k + 2], rb.predicted_logits[k + 2])


def test_carve_eval_stream_disjoint_from_prior():
    # prior = all 0x00, eval = all 0xFF: a 0xFF in the prior would prove leakage.
    data = np.concatenate([np.zeros(800, np.uint8), np.full(200, 255, np.uint8)])
    prior, eval_stream = ByteCorpus(data).prequential_carve(eval_bytes=200)
    assert (len(prior), len(eval_stream)) == (800, 200)
    assert np.unique(prior).tolist() == [0]
    assert np.unique(eval_stream).tolist() == [255]


# --- (b) adaptation FLOPs are counted ----------------------------------------
def test_adaptation_flops_strictly_increase_total():
    stream = np.arange(20, dtype=np.uint8)
    frozen = prequential_bpb(_ConstantModel(torch.zeros(VOCAB_SIZE)), stream, device=CPU)
    adapting = prequential_bpb(
        _AdaptingModel(torch.zeros(VOCAB_SIZE)), stream, device=CPU, adapt_interval=1
    )
    assert frozen.adapt_flops.total == 0
    # adapt called once per byte after the first reveal -> (n-1) times.
    n_adapt = len(stream) - 1
    assert adapting.adapt_flops.total == n_adapt * _AdaptingModel.ADAPT.total
    assert adapting.eval_flops > frozen.eval_flops  # strictly more total FLOPs


# --- transformer decode honesty ----------------------------------------------
def test_kv_cache_decode_matches_full_forward():
    torch.manual_seed(0)
    cfg = TransformerConfig(d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=32)
    model = Transformer(cfg)
    model.eval()
    stream = list(range(12))
    with torch.no_grad():
        full = model(torch.tensor([stream]))[0]  # full[t] predicts byte t+1
    state = model.init_prequential_state()
    for pos in range(len(stream) - 1):
        state, _ = model.observe(state, stream[pos], pos)
        assert torch.allclose(state.next_logits, full[pos], atol=1e-5)


def test_decode_step_flops_forward_only_and_grows_with_context():
    model = Transformer(TransformerConfig(d_model=16, n_layers=2, n_heads=2, max_seq_len=64))
    assert model.decode_step_flops(5).backward == 0
    # Only the attention term depends on context, so cost is strictly increasing.
    assert model.decode_step_flops(20).forward > model.decode_step_flops(5).forward


# --- integration: prequential run accounting ---------------------------------
def test_prequential_run_total_is_pretrain_plus_eval(tmp_path):
    corpus = synthetic_text8(8000, seed=0)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=120)
    cfg = PrequentialConfig(
        model="transformer",
        model_config={"d_model": 16, "n_layers": 2, "n_heads": 2, "max_seq_len": 256},
        pretrain_flop_budget=2e8,
        batch_size=8,
        seq_len=32,
        checkpoint_interval=40,
        seed=0,
        device="cpu",
        run_name="preq-smoke",
    )
    summary = prequential_run(prior, eval_stream, cfg, runs_dir=tmp_path)
    assert summary.total_flops == summary.pretrain_flops + summary.eval_flops
    assert summary.eval_flops > 0  # inference was counted
    assert summary.eval_bytes == 120
    assert 0.0 < summary.bpb <= 8.5
    assert (tmp_path / "preq-smoke.jsonl").exists()
