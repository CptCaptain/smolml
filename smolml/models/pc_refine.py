"""Surprise-gated predictive-coding refinement over a frozen transformer core (Task B.1).

The second Source-(iv) candidate (config **c**, variant **α**). **The bet:** a frozen,
backprop-pretrained transformer core proposes a next-byte distribution, and a small
gradient-free **predictive-coding (PC) module refines it by iterative error-minimization
("settling")** while **settling depth + the online weight update are gated by per-byte
surprise**, so loss-reducing compute concentrates on hard bytes. The falsifiable claim:
*at equal total FLOPs, surprise-gated settling reaches lower bpb than uniform settling,
because we don't waste settling/learning on easy bytes.*

Mechanism (variant α — logit-correction PC)
-------------------------------------------
Slow core = the existing :class:`~smolml.models.transformer.Transformer`, pretrained by
the default backprop ``train_step`` and **frozen** at eval (``forward``/``flops`` delegate
to it, so amortized pretraining and its accounting are exactly the baseline's). Per
position the core yields hidden ``h ∈ ℝ^d`` and base logits ``ℓ_core = head(h) ∈ ℝ^V``.

The PC module is gradient-free runtime state in :class:`DecodeState.cache` (plain tensors,
never ``nn.Parameter`` — exactly like the fast-weight memory). Latent ``z ∈ ℝ^m`` and two
runtime matrices: generative ``W ∈ ℝ^{d×m}`` (predicts the hidden ``ĥ = W z``) and readout
``Vmat ∈ ℝ^{m×V}`` (emits the logit correction ``c = Vmatᵀ z``), **initialized to 0** so
the correction is zero at start (identity to the core — bounds the worst case and answers
the A.1 lesson of a confident-wrong module starving the truth).

*Inference = settling* (free-energy descent on the latent). Minimize
``F(z) = ½‖h − Wz‖²/σ_h² + ½‖z‖²/σ_z²`` by ``K`` gradient steps
``z ← z − η[ z/σ_z² − Wᵀ(h − Wz)/σ_h² ]``, ``z`` warm-started from the previous step.
After settling, refined logits ``ℓ = ℓ_core + Vmatᵀ z``; predict ``softmax(ℓ)``.

*Online learning* (after the target byte is revealed) — gradient-free, local, charged
honestly: readout ``ΔVmat = −lr_readout · z (p − e_byte)ᵀ`` (the exact CE gradient w.r.t. a
linear readout, computed locally — no autograd through the core) and generative
``ΔW = +lr_gen · (h − Wz) zᵀ`` (the PC prediction-error rule), with optional decay toward 0.
Uses the **pending-prediction pattern** from ``fast_weight.step``: stash ``(z, p, h)`` of the
prediction just made; on the next step, when its target byte is revealed, apply the update.
**No leakage** — the update for the prediction at ``pos`` uses only the byte revealed at
``pos``, and the prediction for ``pos+1`` never reads byte ``pos+1``.

Surprise gate
-------------
*Settling depth* ``K`` is picked from a **pre-reveal** proxy (``1 − max softmax(ℓ_core)`` —
never peeks at the future byte): ``uniform`` mode → ``K = k_uniform`` constant; ``surprise``
mode → ``K ∈ [k_min, k_max]`` increasing in surprise, centered on a running surprise mean so
the realized mean ``K ≈ k_uniform`` (⇒ matched total settling FLOPs; the win must come from
*allocation*, not from spending more). *Update gating* applies the weight update only when
**post-reveal** surprise ``−log p(byte) > θ``; update FLOPs are charged only when applied.

FLOP honesty (the critical surface — see :mod:`smolml.flops`, the 0.3 finding)
-----------------------------------------------------------------------------
PC's dominant compute is small matvecs and elementwise mixing, *not* the big projection
matmuls — exactly the regime the instrument would otherwise score as nearly free. So every
op is charged via the shared primitives and returned by ``step`` (the harness sums it):
``forward = core decode + gate-arith + K·settling + readout`` and ``backward = applied
update``. ``K`` is data-dependent, so total eval FLOPs differ per policy — honest by
construction. No compute may be free by omission.
"""

from dataclasses import dataclass

import torch

from smolml.data.corpus import VOCAB_SIZE
from smolml.flops import FlopBreakdown, matmul_flops, pointwise_flops
from smolml.models.registry import DecodeState, LanguageModel, register_model
from smolml.models.transformer import Transformer, TransformerConfig


@dataclass
class PCRefineConfig:
    """Slow-core (transformer) hyperparameters plus predictive-coding/gate scalars.

    The core fields mirror :class:`~smolml.models.transformer.TransformerConfig` (so the
    slow core is the baseline architecture and the comparison isolates the PC module). The
    PC fields are gradient-free runtime scalars, not trained parameters.
    """

    # slow-core (transformer) hyperparameters
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int | None = None
    max_seq_len: int = 256
    vocab_size: int = VOCAB_SIZE
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    # predictive-coding module (gradient-free runtime state)
    m: int = 64  # latent dimension
    eta: float = 0.1  # settling (free-energy descent) step size
    sigma_h: float = 1.0  # hidden-prediction noise scale (data term weight 1/sigma_h^2)
    sigma_z: float = 1.0  # latent prior scale (prior term weight 1/sigma_z^2)
    w_init_scale: float = 0.1  # std of the gradient-free generative weights W at reset
    init_seed: int = 0  # seed for the per-stream W init (Vmat starts at 0)
    # surprise gate (settling depth)
    gate: str = "surprise"  # "uniform" (K=k_uniform) | "surprise" (K from the proxy)
    k_min: int = 1  # min settling iterations (surprise mode)
    k_max: int = 7  # max settling iterations (surprise mode)
    k_uniform: int = 4  # constant K (uniform mode) and the calibration center for the gate
    gate_sensitivity: float = 1.5  # K change per surprise standard-deviation (z-score gate)
    gate_eps: float = 1e-3  # noise floor on the surprise std (suppresses sub-signal gating)
    surprise_ema: float = 0.05  # running mean/variance rate for the surprise reference
    # online learning (gradient-free, local) + update gate
    update_surprise_threshold: float = 0.5  # apply update iff -log p(byte) > θ (nats)
    lr_readout: float = 0.2  # ΔVmat step (exact CE gradient of the linear readout)
    lr_gen: float = 0.05  # ΔW step (PC prediction-error rule)
    weight_decay_fast: float = 0.01  # decay of W/Vmat toward 0 each applied update (in [0,1))

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        if self.eta <= 0.0:
            raise ValueError(f"eta must be positive, got {self.eta}")
        if self.sigma_h <= 0.0 or self.sigma_z <= 0.0:
            raise ValueError(
                f"sigma_h/sigma_z must be positive, got {self.sigma_h}, {self.sigma_z}"
            )
        if self.gate not in ("uniform", "surprise"):
            raise ValueError(f"gate must be 'uniform' or 'surprise', got {self.gate!r}")
        if not 0 <= self.k_min <= self.k_uniform <= self.k_max:
            raise ValueError(
                "settling depths must satisfy 0 <= k_min <= k_uniform <= k_max, got "
                f"k_min={self.k_min}, k_uniform={self.k_uniform}, k_max={self.k_max}"
            )
        if self.k_max < 1:
            raise ValueError(f"k_max must be >= 1, got {self.k_max}")
        if self.gate_sensitivity < 0.0:
            raise ValueError(f"gate_sensitivity must be >= 0, got {self.gate_sensitivity}")
        if self.gate_eps <= 0.0:
            raise ValueError(f"gate_eps must be positive, got {self.gate_eps}")
        if not 0.0 < self.surprise_ema <= 1.0:
            raise ValueError(f"surprise_ema must be in (0, 1], got {self.surprise_ema}")
        if self.lr_readout < 0.0 or self.lr_gen < 0.0:
            raise ValueError(f"learning rates must be >= 0, got {self.lr_readout}, {self.lr_gen}")
        if not 0.0 <= self.weight_decay_fast < 1.0:
            raise ValueError(f"weight_decay_fast must be in [0, 1), got {self.weight_decay_fast}")

    def core_config(self) -> TransformerConfig:
        return TransformerConfig(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            vocab_size=self.vocab_size,
            rope_base=self.rope_base,
            dropout=self.dropout,
            tie_embeddings=self.tie_embeddings,
        )


@dataclass
class _PCRefineCache:
    """Per-stream PC state threaded through :meth:`PCRefine.step`.

    ``W``/``Vmat`` are the gradient-free generative/readout matrices; ``z`` is the settled
    latent of the *previous* prediction (the warm start for the next settling and the latent
    used to apply that prediction's update); ``pending_p``/``pending_h`` are the predicted
    distribution and hidden of the previous prediction (its update is applied when its target
    byte is revealed); ``surprise_mean``/``surprise_var`` are the running surprise statistics
    that calibrate the z-score gate (so realized mean ``K ≈ k_uniform`` at any surprise scale);
    ``kv`` is the slow core's per-layer KV cache in the growing regime (``None`` once decode
    switches to the bounded windowed recompute).
    """

    W: torch.Tensor
    Vmat: torch.Tensor
    z: torch.Tensor
    pending_p: torch.Tensor | None
    pending_h: torch.Tensor | None
    surprise_mean: float | None
    surprise_var: float | None
    kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None


@register_model("pc_refine")
class PCRefine(LanguageModel):
    """Frozen slow transformer core + a surprise-gated predictive-coding refinement."""

    def __init__(self, config: PCRefineConfig):
        super().__init__()
        self.config = config
        # The only trained parameters are the slow core's; the PC module is gradient-free
        # runtime state (lives in DecodeState, never an nn.Parameter).
        self.core = Transformer(config.core_config())

    # --- Amortized path: the slow core only (the PC module is eval-only) ---------

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.core(idx)

    def flops(self, seq_len: int) -> FlopBreakdown:
        return self.core.flops(seq_len)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "PCRefine":
        return cls(PCRefineConfig(**config))

    # --- PC FLOP accounting (charge == reality; see module docstring) ------------

    def _gate_flops(self) -> int:
        """Forward FLOPs of the settling-depth gate: ``softmax(ℓ_core)`` plus the ``1 − max``
        surprise proxy over the ``V`` logits (charged in both modes so the forward path is
        identical and the comparison isolates only the K allocation). The O(1) scalar gate
        bookkeeping (z-score, EMA mean/variance) is dominated by this O(V) softmax and folded
        per the :mod:`smolml.flops` elementwise-omission convention."""
        return pointwise_flops(self.config.vocab_size, per_elem=4)  # softmax(3) + (1-max)(1)

    def _settle_iter_flops(self) -> int:
        """Forward FLOPs of one settling iteration: the ``Wz`` and ``Wᵀr`` matvecs, the
        residual ``r = h − Wz`` (``d`` subtracts), and the latent update ``z ← z − η·grad``
        (``5m`` elementwise: ``z/σ_z²``, ``Wᵀr/σ_h²``, their difference, ``η·grad``, subtract)."""
        d, m = self.config.d_model, self.config.m
        return (
            matmul_flops(1, d, m)  # W z : (d,m) @ (m,) -> (d,)
            + matmul_flops(1, m, d)  # Wᵀ r : (m,d) @ (d,) -> (m,)
            + pointwise_flops(d)  # residual r = h - Wz
            + pointwise_flops(m, per_elem=5)  # z <- z - eta*(z/σ_z² - Wᵀr/σ_h²)
        )

    def _readout_flops(self) -> int:
        """Forward FLOPs of the readout: ``Vmatᵀz`` plus the add to the core logits and
        the ``softmax`` that forms ``p`` (stashed for the next step's update)."""
        m, v = self.config.m, self.config.vocab_size
        return matmul_flops(1, v, m) + pointwise_flops(v, per_elem=4)  # add(1) + softmax(3)

    def _update_gate_flops(self) -> int:
        """Backward FLOPs of the update-gate decision: index ``p[byte]``, the ``−log p(byte)``,
        and the ``> θ`` compare — charged whenever a pending prediction exists (even when the
        update is gated off) so no online compute is free by omission. This O(1) work is
        charged nominally as ``pointwise_flops(1)`` (dominated by the step's O(V)/O(dm) terms)."""
        return pointwise_flops(1)

    def _update_flops(self) -> int:
        """Backward FLOPs of one applied gradient-free update, charged in full because the
        PC update's elementwise work is the *same order* as its rank-1 matmuls (so the
        :mod:`smolml.flops` dominance exception does NOT apply): residual recompute at the
        settled latent, the error ``p − e_byte`` (``V``), the two rank-1 outer products
        (``ΔW``, ``ΔVmat``), and each matrix's combine — ``lr``-scale + add/sub, plus the
        decay multiply when ``weight_decay_fast > 0`` (3 elementwise passes vs 2)."""
        d, m, v = self.config.d_model, self.config.m, self.config.vocab_size
        residual = matmul_flops(1, d, m) + pointwise_flops(d)  # r = h - Wz at the settled z
        err = pointwise_flops(v)  # p - e_byte
        grad_gen = matmul_flops(d, m, 1)  # ΔW = (h - Wz) zᵀ outer
        grad_readout = matmul_flops(m, v, 1)  # ΔVmat = z (p - e_byte)ᵀ outer
        passes = 3 if self.config.weight_decay_fast > 0.0 else 2  # keep-mult? + lr-mult + add/sub
        combine = pointwise_flops(d * m, passes) + pointwise_flops(m * v, passes)
        return residual + err + grad_gen + grad_readout + combine

    # --- Prequential / online decode seam ---------------------------------------

    def init_prequential_state(self) -> DecodeState:
        cfg = self.config
        device = self.core.rope_cos.device
        # W is gradient-free; seed it deterministically (off the global RNG) so a stream is
        # reproducible regardless of surrounding RNG state. Vmat starts at 0 (zero correction).
        gen = torch.Generator().manual_seed(cfg.init_seed)
        w = torch.randn(cfg.d_model, cfg.m, generator=gen) * cfg.w_init_scale
        cache = _PCRefineCache(
            W=w.to(device),
            Vmat=torch.zeros(cfg.m, cfg.vocab_size, device=device),
            z=torch.zeros(cfg.m, device=device),
            pending_p=None,
            pending_h=None,
            surprise_mean=None,
            surprise_var=None,
            kv=[None] * cfg.n_layers,
        )
        return DecodeState(cache=cache)

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        cfg = self.config
        cache: _PCRefineCache = state.cache
        new_len = state.length + 1
        window_cap = cfg.max_seq_len
        window = [*state.tokens, revealed_byte][-window_cap:]

        w_mat, v_mat, z = cache.W, cache.Vmat, cache.z
        backward = 0
        with torch.no_grad():
            # (1) Adapt: apply the PREVIOUS prediction's gated local update against the byte
            # it was predicting, now revealed. Uses only past/present bytes (no leakage).
            if cache.pending_p is not None:
                w_mat, v_mat, backward = self._adapt(
                    w_mat, v_mat, z, cache.pending_h, cache.pending_p, revealed_byte
                )

            # (2) Frozen slow core: decode the revealed byte to a hidden state + logits.
            if cache.kv is not None and new_len <= window_cap:
                if pos != new_len - 1:
                    raise ValueError(
                        f"step expects consecutive positions: pos={pos}, length={new_len - 1}"
                    )
                hidden, core_logits, new_kv = self._decode_incremental(revealed_byte, pos, cache.kv)
                core_fwd = self.core.decode_step_flops(new_len).forward
            else:
                hidden, core_logits = self._decode_window(window)
                new_kv = None
                core_fwd = self.core.flops(len(window)).forward

            # (3) Gate: pick the settling depth K from the pre-reveal surprise proxy, z-scored
            # against the running surprise mean/variance (calibrated, scale-adaptive). The
            # current surprise is pre-reveal, the statistics are from past bytes -> no leakage.
            surprise = 1.0 - float(torch.softmax(core_logits, dim=-1).max())
            mean = surprise if cache.surprise_mean is None else cache.surprise_mean
            var = 0.0 if cache.surprise_var is None else cache.surprise_var
            k = self._settle_depth(surprise, mean, var**0.5)
            delta = surprise - mean
            new_mean = mean + cfg.surprise_ema * delta
            new_var = (1.0 - cfg.surprise_ema) * (var + cfg.surprise_ema * delta * delta)

            # (4) Settling: free-energy descent on the latent (warm-started from z).
            z_new = self._settle(z, hidden, w_mat, k)

            # (5) Readout: refined logits = core + Vmatᵀz; stash p/h for the next adapt.
            next_logits = core_logits + v_mat.t() @ z_new
            p = torch.softmax(next_logits, dim=-1)

        forward = (
            core_fwd + self._gate_flops() + k * self._settle_iter_flops() + self._readout_flops()
        )
        flops = FlopBreakdown(forward=forward, backward=backward)
        new_cache = _PCRefineCache(
            W=w_mat,
            Vmat=v_mat,
            z=z_new,
            pending_p=p,
            pending_h=hidden,
            surprise_mean=new_mean,
            surprise_var=new_var,
            kv=new_kv,
        )
        new_state = DecodeState(tokens=window, cache=new_cache, length=new_len)
        return new_state, next_logits.detach(), flops

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        """Forward-only per-byte cost at the representative (uniform) settling depth: the
        core's incremental decode + gate + ``k_uniform`` settling iters + readout. The true
        per-byte cost (with data-dependent ``K``) is what :meth:`step` returns and the
        harness sums; this analytic estimate uses the constant ``k_uniform``."""
        core = self.core.decode_step_flops(context_len).forward
        forward = (
            core
            + self._gate_flops()
            + self.config.k_uniform * self._settle_iter_flops()
            + self._readout_flops()
        )
        return FlopBreakdown(forward=forward, backward=0)

    # --- internals ---------------------------------------------------------------

    def _settle_depth(self, surprise: float, mean: float, std: float) -> int:
        """Map the pre-reveal surprise proxy to a settling depth ``K``.

        ``uniform`` mode → constant ``k_uniform``. ``surprise`` mode → ``k_uniform`` plus
        ``gate_sensitivity`` per standard-deviation of surprise above the running mean
        (z-scored, so the gate adapts to any surprise scale and the realized mean ``K ≈
        k_uniform`` ⇒ matched total settling FLOPs); ``gate_eps`` floors the denominator so a
        sub-signal (noise-level) surprise spread does not drive spurious allocation. Clamped
        to ``[k_min, k_max]`` and monotonic nondecreasing in surprise."""
        cfg = self.config
        if cfg.gate == "uniform":
            return cfg.k_uniform
        z = (surprise - mean) / (std + cfg.gate_eps)
        raw = cfg.k_uniform + cfg.gate_sensitivity * z
        return max(cfg.k_min, min(cfg.k_max, int(round(raw))))

    def _settle(
        self, z: torch.Tensor, hidden: torch.Tensor, w_mat: torch.Tensor, k: int
    ) -> torch.Tensor:
        """``k`` free-energy descent steps ``z ← z − η[ z/σ_z² − Wᵀ(h − Wz)/σ_h² ]`` from the
        warm start ``z`` (returns a new tensor; ``k=0`` returns the warm start unchanged)."""
        cfg = self.config
        inv_h = 1.0 / (cfg.sigma_h * cfg.sigma_h)
        inv_z = 1.0 / (cfg.sigma_z * cfg.sigma_z)
        eta = cfg.eta
        for _ in range(k):
            residual = hidden - w_mat @ z  # (d,)
            grad = z * inv_z - (w_mat.t() @ residual) * inv_h  # (m,)
            z = z - eta * grad
        return z

    def _adapt(
        self,
        w_mat: torch.Tensor,
        v_mat: torch.Tensor,
        z: torch.Tensor,
        hidden: torch.Tensor,
        p: torch.Tensor,
        byte: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply the gated gradient-free update for the pending prediction.

        Update gate: only when the post-reveal surprise ``−log p(byte)`` exceeds the
        threshold. Readout follows the exact CE gradient of the linear readout
        ``ΔVmat = −lr_readout·z (p − e_byte)ᵀ``; the generative rule reduces the PC
        prediction error ``ΔW = +lr_gen·(h − Wz) zᵀ``. Returns ``(W, Vmat, backward_flops)``.
        """
        cfg = self.config
        backward = self._update_gate_flops()
        neg_log_p = -float(torch.log(p[byte] + 1e-12))
        if neg_log_p <= cfg.update_surprise_threshold:
            return w_mat, v_mat, backward
        residual = hidden - w_mat @ z  # prediction error at the settled latent
        target = torch.zeros_like(p)
        target[byte] = 1.0
        d_v = torch.outer(z, p - target)  # (m, V)
        d_w = torch.outer(residual, z)  # (d, m)
        keep = 1.0 - cfg.weight_decay_fast
        if cfg.weight_decay_fast > 0.0:
            v_mat = v_mat * keep - cfg.lr_readout * d_v
            w_mat = w_mat * keep + cfg.lr_gen * d_w
        else:
            v_mat = v_mat - cfg.lr_readout * d_v
            w_mat = w_mat + cfg.lr_gen * d_w
        backward += self._update_flops()
        return w_mat, v_mat, backward

    def _decode_incremental(
        self,
        revealed_byte: int,
        pos: int,
        kv: list[tuple[torch.Tensor, torch.Tensor] | None],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Growing-regime KV-cache decode of one byte; returns (hidden, logits, kv).

        Replays the core's incremental decode (reusing its blocks/norm/head/rope) so the cost
        equals ``core.decode_step_flops`` exactly, while exposing the final hidden state."""
        core = self.core
        device = core.rope_cos.device
        x = core.tok_emb(torch.tensor([[revealed_byte]], dtype=torch.long, device=device))
        cos, sin = core.rope_cos[pos : pos + 1], core.rope_sin[pos : pos + 1]
        new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block, layer_kv in zip(core.blocks, kv, strict=True):
            x, nkv = block.decode_step(x, cos, sin, layer_kv)
            new_kv.append(nkv)
        hidden = core.norm_f(x)[0, -1]
        return hidden, core.head(hidden), new_kv

    def _decode_window(self, window: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Sliding-regime full recompute over the last ``window`` bytes; returns the
        final-position (hidden, logits). Bounded memory, length-matched, exact."""
        core = self.core
        device = core.rope_cos.device
        idx = torch.tensor([window], dtype=torch.long, device=device)
        t = idx.shape[1]
        x = core.tok_emb(idx)
        cos, sin = core.rope_cos[:t], core.rope_sin[:t]
        for block in core.blocks:
            x = block(x, cos, sin)
        hidden = core.norm_f(x)[0, -1]
        return hidden, core.head(hidden)
