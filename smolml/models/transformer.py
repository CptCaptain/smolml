"""Transformer baseline: pre-norm, RMSNorm, RoPE, causal self-attention.

Plain and honestly counted — the bar every candidate must clear. Sizes are
configurable so the same architecture spans tiny smoke runs to larger budgets.
The FFN is a 2-layer GELU MLP (``d -> d_ff -> d``), which is exactly what
:meth:`Transformer.flops` charges.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from smolml.data.corpus import VOCAB_SIZE
from smolml.flops import MAC_FLOPS, FlopBreakdown, causal_attention_flops, linear_flops
from smolml.models.registry import DecodeState, LanguageModel, register_model


@dataclass
class TransformerConfig:
    """Hyperparameters for the transformer baseline.

    ``d_ff`` defaults to ``4 * d_model``; ``d_model`` must be divisible by
    ``n_heads`` and the resulting head dim must be even (RoPE rotates pairs).
    """

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int | None = None
    max_seq_len: int = 256
    vocab_size: int = VOCAB_SIZE
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head dim (d_model / n_heads) must be even for RoPE")


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) of shape ``(seq_len, head_dim)`` for rotary embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to ``x`` of shape ``(B, n_heads, T, head_dim)``."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE on queries and keys."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.proj(out)

    def decode_step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-token incremental step with a KV cache.

        ``x`` is ``(B, 1, C)`` (the one new token); ``cos``/``sin`` are the rotary
        factors at its absolute position; ``kv`` is the cached (keys, values) from
        earlier positions, or ``None``. The new query attends to all cached keys
        plus itself (no mask needed — every cached key is in the past), exactly
        reproducing the last row of the full causal attention. Returns the
        projected output and the grown cache.
        """
        b, _, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv is not None:
            k = torch.cat([kv[0], k], dim=2)
            v = torch.cat([kv[1], v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(b, 1, c)
        return self.proj(out), (k, v)


class MLP(nn.Module):
    """Position-wise feed-forward: Linear -> GELU -> Linear."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)); x + mlp(norm(x))."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

    def decode_step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        a, new_kv = self.attn.decode_step(self.norm1(x), cos, sin, kv)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


@register_model("transformer")
class Transformer(LanguageModel):
    """The byte-level transformer baseline."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        head_dim = config.d_model // config.n_heads
        cos, sin = build_rope_cache(config.max_seq_len, head_dim, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        if t > self.config.max_seq_len:
            raise ValueError(f"seq_len {t} exceeds max_seq_len {self.config.max_seq_len}")
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        return self.head(x)

    def flops(self, seq_len: int) -> FlopBreakdown:
        cfg = self.config
        t, d = seq_len, cfg.d_model
        per_layer = (
            linear_flops(t, d, 3 * d)  # qkv projection
            + linear_flops(t, d, d)  # output projection
            + linear_flops(t, d, cfg.d_ff)  # ffn up
            + linear_flops(t, cfg.d_ff, d)  # ffn down
            + causal_attention_flops(t, d)  # scores + value mixing
        )
        forward = cfg.n_layers * per_layer + linear_flops(t, d, cfg.vocab_size)
        return FlopBreakdown.from_forward(forward)

    # --- Prequential decode (KV-cache override; O(context_len) per byte) -------

    def init_prequential_state(self) -> DecodeState:
        # First byte predicted from a uniform prior; one empty KV slot per layer.
        return DecodeState(
            next_logits=torch.zeros(self.config.vocab_size),
            cache=[None] * self.config.n_layers,
        )

    def observe(
        self, state: DecodeState, token: int, pos: int
    ) -> tuple[DecodeState, FlopBreakdown]:
        cfg = self.config
        if pos >= cfg.max_seq_len:
            raise ValueError(
                f"prequential position {pos} reaches max_seq_len {cfg.max_seq_len}; "
                "size max_seq_len >= eval-stream length (sliding-window decode is future work)"
            )
        device = self.rope_cos.device
        x = self.tok_emb(torch.tensor([[token]], dtype=torch.long, device=device))
        cos, sin = self.rope_cos[pos : pos + 1], self.rope_sin[pos : pos + 1]
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for block, kv in zip(self.blocks, state.cache, strict=True):
                x, nkv = block.decode_step(x, cos, sin, kv)
                new_cache.append(nkv)
            logits = self.head(self.norm_f(x))[0, -1].detach()
        # This step attended pos+1 keys (the cache of pos plus the new token).
        return DecodeState(next_logits=logits, cache=new_cache), self.decode_step_flops(pos + 1)

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        """Incremental per-byte prediction cost with a KV cache: O(d^2) projections
        for the one new token + O(context_len * d) attention (1 query vs
        ``context_len`` cached keys). Forward-only (inference)."""
        cfg = self.config
        d = cfg.d_model
        per_layer = (
            linear_flops(1, d, 3 * d)  # qkv for the single new token
            + linear_flops(1, d, d)  # output projection
            + linear_flops(1, d, cfg.d_ff)  # ffn up
            + linear_flops(1, cfg.d_ff, d)  # ffn down
            + 2 * MAC_FLOPS * d * context_len  # Q·Kᵀ + softmax·V: 1 query × context_len keys
        )
        forward = cfg.n_layers * per_layer + linear_flops(1, d, cfg.vocab_size)
        return FlopBreakdown(forward=forward, backward=0)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "Transformer":
        return cls(TransformerConfig(**config))
