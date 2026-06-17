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

    # Unified seam: fold the byte (no-op), predict the constant, charge 1 FLOP.
    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        return (
            DecodeState(length=state.length + 1),
            self._logits,
            FlopBreakdown(forward=1, backward=0),
        )


class _OnlineFrequencyModel(LanguageModel):
    """An online Laplace frequency model: each step counts the revealed byte and
    predicts proportional to counts. It genuinely learns at test time, so on a
    repetitive stream its cumulative bpb drops far below a frozen model's. Charges
    a `backward=1` per step to mark the adaptation as counted compute."""

    def __init__(self):
        super().__init__()
        self.config = _StubConfig()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        return torch.zeros(b, t, VOCAB_SIZE)

    def flops(self, seq_len: int) -> FlopBreakdown:
        return FlopBreakdown(forward=seq_len, backward=0)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "_OnlineFrequencyModel":
        return cls()

    def init_prequential_state(self) -> DecodeState:
        return DecodeState(cache=torch.ones(VOCAB_SIZE))  # Laplace prior

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        counts = state.cache.clone()
        counts[revealed_byte] += 1.0  # online update from the revealed byte
        next_logits = torch.log(counts)  # predict ∝ counts
        # forward (predict) + backward (the online update) — both counted.
        return (
            DecodeState(cache=counts, length=state.length + 1),
            next_logits,
            FlopBreakdown(forward=1, backward=1),
        )


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
    stream = np.full(6, 65, dtype=np.uint8)  # all 'A'
    result = prequential_bpb(model, stream, device=CPU)
    # byte 0 = uniform prior (8 bits, no context); bytes 1..5 = -log2(0.5) = 1 bit
    # each -> total = 8 + 5 = 13 bits over 6 bytes.
    assert math.isclose(result.total_bits, 13.0, rel_tol=1e-4)
    assert math.isclose(result.bpb, 13.0 / 6.0, rel_tol=1e-4)


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


# --- (b) adaptation actually moves the scored prediction AND is counted -------
def test_adaptation_lowers_bpb_and_is_counted():
    stream = np.full(40, 65, dtype=np.uint8)  # learnable: one repeated byte
    frozen = prequential_bpb(_ConstantModel(torch.zeros(VOCAB_SIZE)), stream, device=CPU)
    adapting = prequential_bpb(_OnlineFrequencyModel(), stream, device=CPU, collect_logits=True)
    # Frozen never learns -> stays at the 8.0 uniform baseline.
    assert math.isclose(frozen.bpb, 8.0, rel_tol=1e-6)
    # Online adaptation MEASURABLY lowers cumulative bpb (the whole point).
    assert adapting.bpb < frozen.bpb - 1.0
    # Adaptation compute is counted through the single step channel (backward > 0).
    assert adapting.flops.backward > 0
    assert frozen.flops.backward == 0
    # The scored prediction genuinely sharpens toward the seen byte over time.
    early = torch.softmax(adapting.predicted_logits[2], dim=-1)[65].item()
    late = torch.softmax(adapting.predicted_logits[30], dim=-1)[65].item()
    assert late > early


# --- transformer decode honesty ----------------------------------------------
def test_kv_cache_decode_matches_full_forward():
    # Growing regime (stream fits the context): incremental decode == full forward.
    torch.manual_seed(0)
    cfg = TransformerConfig(d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=32)
    model = Transformer(cfg)
    model.eval()
    stream = list(range(12))
    with torch.no_grad():
        full = model(torch.tensor([stream]))[0]  # full[t] predicts byte t+1
    state = model.init_prequential_state()
    for pos in range(len(stream) - 1):
        state, next_logits, _ = model.step(state, stream[pos], pos)
        assert torch.allclose(next_logits, full[pos], atol=1e-5)


def test_sliding_window_decode_matches_windowed_forward():
    # Stream longer than the context window forces the bounded sliding regime.
    torch.manual_seed(0)
    w = 8
    cfg = TransformerConfig(d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=w)
    model = Transformer(cfg)
    model.eval()
    stream = list(range(20))
    state = model.init_prequential_state()
    checked_sliding = False
    for pos in range(len(stream) - 1):
        state, next_logits, flops = model.step(state, stream[pos], pos)
        if pos >= w:  # new_len = pos+1 > w -> sliding (bounded recompute)
            window = stream[pos - w + 1 : pos + 1]  # last w revealed bytes
            with torch.no_grad():
                ref = model(torch.tensor([window]))[0, -1]
            assert torch.allclose(next_logits, ref, atol=1e-5)
            assert len(state.tokens) == w  # bounded memory: window capped at w
            assert flops.forward == model.flops(w).forward  # recompute cost charged
            checked_sliding = True
    assert checked_sliding


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


def test_prequential_run_is_deterministic(tmp_path):
    corpus = synthetic_text8(8000, seed=0)
    prior, eval_stream = corpus.prequential_carve(eval_bytes=96)
    model_config = {"d_model": 16, "n_layers": 2, "n_heads": 2, "max_seq_len": 256}

    def run(name: str):
        cfg = PrequentialConfig(
            model="transformer",
            model_config=model_config,
            pretrain_flop_budget=2e8,
            batch_size=8,
            seq_len=32,
            seed=0,
            device="cpu",
            run_name=name,
        )
        return prequential_run(prior, eval_stream, cfg, runs_dir=tmp_path)

    a, b = run("det-a"), run("det-b")
    assert a.bpb == b.bpb
    assert a.total_flops == b.total_flops
    assert a.pretrain_flops == b.pretrain_flops
