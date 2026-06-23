"""Task 4 acceptance tests: the ``evaluate_control`` interactive rollout scorer."""

import math
from types import SimpleNamespace

import torch

from smolml.control_eval import evaluate_control
from smolml.envs.chemotaxis import ChemoConfig, action_token, vocab_size
from smolml.flops import FlopBreakdown
from smolml.models.registry import LanguageModel


class _FixedActionModel(LanguageModel):
    """Stub: always favors one absolute action token; uniform over concentrations."""

    def __init__(self, vocab: int, max_seq_len: int, fav_token: int):
        super().__init__()
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self._vocab, self._fav = vocab, fav_token
        # The base-class recompute ``step`` reads the device from a parameter; a
        # real model always has one, so anchor a trivial param for the stub.
        self._anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        logits = torch.zeros(b, t, self._vocab)
        logits[..., self._fav] = 10.0
        return logits

    def flops(self, seq_len: int) -> FlopBreakdown:
        return FlopBreakdown.from_forward(seq_len)

    @classmethod
    def from_config(cls, config: dict) -> "_FixedActionModel":
        return cls(**config)


def test_uniform_world_model_bits_near_log2_levels():
    cfg = ChemoConfig(width=16, levels=8, horizon=8)
    model = _FixedActionModel(vocab_size(cfg), 2 * cfg.horizon + 1, action_token(cfg, 1))
    res = evaluate_control(
        model, cfg, split="eval", n_episodes=4, seed=0, device=torch.device("cpu")
    )
    assert math.isfinite(res.world_model_bits)
    assert abs(res.world_model_bits - math.log2(cfg.levels)) < 0.2


def test_env_responds_to_actions_no_predetermined_feedback():
    cfg = ChemoConfig(width=16, levels=8, horizon=12)
    left = _FixedActionModel(vocab_size(cfg), 2 * cfg.horizon + 1, action_token(cfg, 0))
    right = _FixedActionModel(vocab_size(cfg), 2 * cfg.horizon + 1, action_token(cfg, 2))
    rl = evaluate_control(
        left,
        cfg,
        split="eval",
        n_episodes=1,
        seed=3,
        device=torch.device("cpu"),
        greedy=True,
        record=True,
    )
    rr = evaluate_control(
        right,
        cfg,
        split="eval",
        n_episodes=1,
        seed=3,
        device=torch.device("cpu"),
        greedy=True,
        record=True,
    )
    assert rl.trajectory.pos != rr.trajectory.pos  # opposite moves -> different trajectories


def test_rollout_flop_accounting_matches_analytic():
    from smolml.models.transformer import Transformer, TransformerConfig

    cfg = ChemoConfig(width=16, levels=8, horizon=8)
    tcfg = TransformerConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        vocab_size=vocab_size(cfg),
        max_seq_len=2 * cfg.horizon + 1,
    )
    model = Transformer(tcfg)
    res = evaluate_control(
        model, cfg, split="eval", n_episodes=1, seed=1, device=torch.device("cpu")
    )
    expected = sum(model.decode_step_flops(k).forward for k in range(1, 2 * cfg.horizon + 1))
    assert res.flops.forward == expected
