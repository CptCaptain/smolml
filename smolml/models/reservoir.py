"""Echo-state reservoir + trained linear readout (Task C.A.1, minimal-organism control).

The bio thesis (ADR 0007 + ADR 0003): a minimal organism with a **fixed** recurrent
substrate and a cheap, locally-trained readout can do in-context control. A transformer
recomputes ``O(n_layers·d² + context·d)`` per decode step; a reservoir (echo-state
network) rolls its state in ``O(d_res²)`` per step — **independent of context length** —
and only a **linear readout** is trained. The loss-per-FLOP edge is structural: the
expensive recurrent dynamics are **never trained** (0 backward), so the learning cost is
just the readout's ``dW_out`` outer product.

Per token (the input is the token id):

    x_t    = W_in[:, token]                          # fixed-random column = a gather
    pre    = x_t + W_res @ h_{t-1}                    # the dominant matvec, O(d_res²)
    h_t    = (1 - leak) * h_{t-1} + leak * tanh(pre)  # leaky-integrator state
    logits = W_out @ h_t + b_out                      # the TRAINED readout, full vocab

``W_in`` and ``W_res`` are ``nn.Parameter(requires_grad=False)`` (seeded random, ``W_res``
rescaled to spectral radius ``rho`` for the echo-state property). They are **counted in
``num_params``** (memory parity with the transformer bar) but **excluded from the
optimizer** and **charged 0 backward**. ``W_out``/``b_out`` are the only trainable params.

FLOP accounting (honest, via :mod:`smolml.flops` — the product)
---------------------------------------------------------------
- Per-token forward = ``matmul_flops(1, d_res, d_res)`` (``W_res·h``) +
  ``matmul_flops(1, vocab, d_res)`` (readout) + the gather/add/leak+tanh/bias pointwise
  work (``O(d_res)``). ``flops(T).forward = T ×`` that.
- ``flops(T).backward = T × matmul_flops(1, vocab, d_res)`` — the readout ``dW_out`` outer
  product **only**. The recurrence builds **no autograd graph** (frozen params + a constant
  ``h_0`` ⇒ ``h`` does not require grad), so backprop reaches ``W_out``/``b_out`` and stops;
  there is no ``dh`` matmul. This is why :meth:`flops` must NOT use
  ``FlopBreakdown.from_forward`` (that would charge a full 2× backward, which is wrong here).
- ``step``/``decode_step_flops`` = per-token forward, backward 0 (the readout is frozen
  after distillation), ``O(d_res²)`` and context-independent (``h`` is the bounded memory).
"""

from dataclasses import dataclass

import torch
from torch import nn

from smolml.data.corpus import VOCAB_SIZE
from smolml.envs.chemotaxis import N_ACTIONS
from smolml.flops import FlopBreakdown, gather_flops, matmul_flops, pointwise_flops
from smolml.models.registry import DecodeState, LanguageModel, register_model


@dataclass
class ReservoirConfig:
    """Reservoir hyperparameters. ``vocab_size``/``max_seq_len`` are injected by
    ``distill_train_run``; the rest size and seed the frozen echo-state core.

    ``d_res=374`` lands ``num_params`` just under the transformer bar's 148,608 at the
    control vocab of 11 (8 concentration levels + 3 actions): 374² + 2·11·374 + 11 =
    148,115 params (so the regret-per-FLOP comparison is fair on the param axis).
    """

    d_res: int = 374
    leak: float = 0.6
    spectral_radius: float = 0.9
    seed: int = 0
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 256

    def __post_init__(self):
        if not 0.0 < self.leak <= 1.0:
            raise ValueError(f"leak must be in (0, 1], got {self.leak}")
        if self.spectral_radius <= 0.0:
            raise ValueError(f"spectral_radius must be positive, got {self.spectral_radius}")


class _ReservoirCore(nn.Module):
    """The shared **frozen** echo-state substrate: a fixed input embedding ``W_in`` and a
    recurrent matrix ``W_res`` rescaled to spectral radius ``rho`` (the echo-state
    property — past inputs fade, so ``h`` is a bounded, fading summary of the trajectory).

    Both matrices are ``nn.Parameter(requires_grad=False)``: counted in ``num_params`` for
    memory parity, but never trained. The recurrence runs under ``no_grad`` so it builds no
    autograd graph — a downstream readout's backward reaches only the readout's weights.
    Reused unchanged by the (C.A.1b) online-readout sibling.
    """

    def __init__(self, d_res: int, vocab_size: int, spectral_radius: float, seed: int):
        super().__init__()
        self.d_res = d_res
        gen = torch.Generator().manual_seed(seed)
        w_in = torch.randn(d_res, vocab_size, generator=gen)
        w_res = torch.randn(d_res, d_res, generator=gen)
        # Rescale to spectral radius rho: divide by the largest |eigenvalue|, times rho.
        radius = torch.linalg.eigvals(w_res).abs().max()
        w_res = w_res * (spectral_radius / radius)
        self.W_in = nn.Parameter(w_in, requires_grad=False)
        self.W_res = nn.Parameter(w_res, requires_grad=False)

    def initial_state(self) -> torch.Tensor:
        """The constant ``h_0 = 0`` (a ``d_res`` vector) that seeds every rollout."""
        return torch.zeros(self.d_res, device=self.W_in.device, dtype=self.W_in.dtype)

    @torch.no_grad()
    def run(self, idx: torch.Tensor, leak: float) -> torch.Tensor:
        """Sequential recurrence over ``idx`` ``(B, T)`` from ``h_0 = 0`` → states
        ``(B, T, d_res)``. Runs under ``no_grad`` so the returned states carry no graph."""
        b, t = idx.shape
        h = torch.zeros(b, self.d_res, device=self.W_in.device, dtype=self.W_in.dtype)
        states = []
        for pos in range(t):
            x = self.W_in.index_select(1, idx[:, pos]).t()  # (B, d_res) gather of columns
            h = (1.0 - leak) * h + leak * torch.tanh(x + h @ self.W_res.t())
            states.append(h)
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def fold(self, h: torch.Tensor, token: int, leak: float) -> torch.Tensor:
        """Advance one token: fold ``token`` into state ``h`` ``(d_res,)`` → ``h_new``."""
        x = self.W_in[:, token]  # (d_res,) fixed-embedding column
        return (1.0 - leak) * h + leak * torch.tanh(x + self.W_res @ h)

    def recurrence_flops(self) -> int:
        """Forward FLOPs of one recurrence step: the ``W_res·h`` matvec plus the column
        gather, the residual add, and the leaky-integrator ``tanh`` update."""
        d = self.d_res
        return (
            matmul_flops(1, d, d)  # W_res @ h : the dominant O(d_res²) matvec
            + gather_flops(d)  # x = W_in[:, token] : a fixed-embedding column gather
            + pointwise_flops(d)  # pre = x + W_res@h : the residual add
            + pointwise_flops(d, per_elem=4)  # (1-leak)*h + leak*tanh(pre): tanh + 2 scales + add
        )


@register_model("reservoir")
class Reservoir(LanguageModel):
    """Frozen echo-state core + a distilled-frozen linear readout (C.A.1).

    Trained by ``distill_train_run(model="reservoir", …)`` exactly like the transformer
    bar; the only trained tensors are ``readout.weight``/``bias``. In-context adaptation at
    eval is the reservoir **state** echoing the recent trajectory — the readout is frozen
    after distillation (no weight change in :meth:`step`).
    """

    def __init__(self, config: ReservoirConfig):
        super().__init__()
        self.config = config
        self.core = _ReservoirCore(
            config.d_res, config.vocab_size, config.spectral_radius, config.seed
        )
        self.readout = nn.Linear(config.d_res, config.vocab_size)
        # Deterministic readout init from a stream distinct from the core's (so a fixed
        # seed pins the whole model's forward output for the determinism test).
        gen = torch.Generator().manual_seed(config.seed + 1)
        with torch.no_grad():
            self.readout.weight.normal_(0.0, 0.02, generator=gen)
            self.readout.bias.zero_()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        states = self.core.run(idx, self.config.leak)  # (B,T,d_res), no grad through core
        return self.readout(states)  # (B,T,vocab); backward flows only into the readout

    def _per_token_forward(self) -> int:
        """Per-token forward FLOPs: the recurrence + the readout matvec + the bias add."""
        d, v = self.config.d_res, self.config.vocab_size
        return self.core.recurrence_flops() + matmul_flops(1, v, d) + pointwise_flops(v)

    def _readout_backward(self) -> int:
        """Per-token backward FLOPs: the ``dW_out = dlogits ⊗ h`` outer product plus the
        elementwise ``db_out`` bias gradient.

        ``h`` does not require grad (frozen core + constant ``h_0``), so backprop computes
        ``dW_out``/``db_out`` and stops — no ``dh`` matmul, no gradient into the recurrence.
        Hence backward ≠ 2× forward."""
        v = self.config.vocab_size
        return matmul_flops(1, v, self.config.d_res) + pointwise_flops(v)

    def flops(self, seq_len: int) -> FlopBreakdown:
        return FlopBreakdown(
            forward=seq_len * self._per_token_forward(),
            backward=seq_len * self._readout_backward(),
        )

    def configure_optimizer(
        self, *, lr: float, weight_decay: float, betas: tuple[float, float]
    ) -> torch.optim.Optimizer:
        """AdamW over the trainable readout only; the frozen core is excluded."""
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=lr, betas=betas, weight_decay=weight_decay)

    # --- Prequential / online decode seam: h IS the bounded memory --------------

    def init_prequential_state(self) -> DecodeState:
        return DecodeState(cache=self.core.initial_state())

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        """Fold one token into the reservoir state and read out the next distribution.

        ``state.cache`` is the ``d_res`` reservoir state (the entire bounded memory);
        ``state.tokens`` is ignored (no token window is needed). Forward-only: the readout
        is frozen, so backward is 0 and the cost is ``O(d_res²)``, context-independent."""
        h_new = self.core.fold(state.cache, revealed_byte, self.config.leak)
        with torch.no_grad():
            logits = self.readout(h_new).detach()
        flops = FlopBreakdown(forward=self._per_token_forward(), backward=0)
        return DecodeState(cache=h_new, length=state.length + 1), logits, flops

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        """Forward-only per-step cost: ``O(d_res²)``, independent of ``context_len``."""
        return FlopBreakdown(forward=self._per_token_forward(), backward=0)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "Reservoir":
        return cls(ReservoirConfig(**config))


@dataclass
class ReservoirPlasticConfig:
    """Frozen echo-state core + an ONLINE reward-modulated PLASTIC readout (C.A.1b).

    Mirrors :class:`ReservoirConfig` (same frozen core, same default sizing) and adds the
    three local-rule rates. The seed ``nn.Linear`` readout is held as an ``nn.Parameter``
    (counted in ``num_params`` for memory parity, like ``reservoir``) and cloned into the
    decode cache as the PLASTIC ``(W, b)`` that adapt online in :meth:`ReservoirPlastic.step`.
    The headline run does ~0 distillation: all learning happens online in ``step``.

    ``d_res=374`` lands ``num_params`` at 148,115 (same tensors as ``reservoir``), under the
    transformer bar's 148,608, so the regret-per-FLOP comparison is fair on the param axis.
    """

    d_res: int = 374
    leak: float = 0.6
    spectral_radius: float = 0.9
    seed: int = 0
    lr_wm: float = 0.5  # world-model delta-rule rate (conc_slice columns)
    lr_pol: float = 0.03  # reward-modulated Hebbian policy rate (action_slice columns)
    reward_decay: float = 0.7  # leaky reward-baseline rate (fast ⇒ adv ≈ Δconcentration)
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 256

    def __post_init__(self):
        if not 0.0 < self.leak <= 1.0:
            raise ValueError(f"leak must be in (0, 1], got {self.leak}")
        if self.spectral_radius <= 0.0:
            raise ValueError(f"spectral_radius must be positive, got {self.spectral_radius}")
        if self.vocab_size - N_ACTIONS < 2:
            raise ValueError(
                f"vocab_size must leave >=2 concentration levels, got {self.vocab_size}"
            )


@dataclass
class _PlasticCache:
    """Per-stream online-readout state threaded through :meth:`ReservoirPlastic.step`.

    ``h`` is the frozen reservoir state (the bounded memory); ``W``/``b`` are the PLASTIC
    readout — clones of the seed ``nn.Linear``, never ``nn.Parameter`` (so eval changes no
    model weights); ``baseline`` is the leaky reward baseline. The remaining fields are the
    one-step-late bookkeeping the next concentration-fold update consumes (the reward and the
    world-model target are only observable one step after the action / prediction):

    - ``h_conc``: ``h`` at the most recent CONC fold — the state the action was sampled from
      (presynaptic activity for the policy Hebbian term).
    - ``action_token``: the action token folded at the most recent ACTION fold.
    - ``conc_pred_logits``: full-vocab logits at that ACTION fold — its ``conc_slice``
      predicted the concentration now being revealed (the delta-rule target).
    - ``h_action``: ``h`` at that ACTION fold — presynaptic activity for the delta-rule term.
    """

    h: torch.Tensor
    W: torch.Tensor
    b: torch.Tensor
    baseline: float
    h_conc: torch.Tensor | None
    action_token: int | None
    conc_pred_logits: torch.Tensor | None
    h_action: torch.Tensor | None


@register_model("reservoir_plastic")
class ReservoirPlastic(LanguageModel):
    """Frozen echo-state core + an ONLINE reward-modulated plastic readout (C.A.1b).

    Reuses :class:`_ReservoirCore` unchanged. The readout is no longer distilled-and-frozen:
    a working copy of ``(W, b)`` lives in :attr:`DecodeState.cache` and is adapted by a
    gradient-free LOCAL rule inside :meth:`step`. ``evaluate_control`` is ``@torch.no_grad()``
    so the update is plain tensor ops (no autograd). ~0 distillation — all learning is online.

    Two local rules, both charged to ``step``'s ``backward`` (ADR 0004 — eval compute is the
    product), fire only on a CONCENTRATION fold at ``pos>=2`` (when the reward AND the
    world-model target are first observable):

    - **world model** (``conc_slice`` columns): an online softmax **delta rule**, supervised
      by the just-revealed concentration against the prediction made at the preceding
      action-fold — ``W[conc] += lr_wm·(onehot(c) − softmax(pred)) ⊗ h_action``.
    - **policy** (``action_slice`` columns): **reward-modulated Hebbian** with a leaky
      baseline, reward proxy ``r = c/(levels−1)`` —
      ``W[action] += lr_pol·(r − baseline)·onehot(a_taken) ⊗ h_conc``.

    Gradient-free, local, context-independent (``O(d_res²)``/step). The seed ``nn.Linear``
    readout (an ``nn.Parameter``, counted for memory parity) seeds the cache copy and also
    drives a backprop distill path (``forward``/``train_step``) so the model still fits the
    distillation harness — but the headline run trains 0 steps.
    """

    def __init__(self, config: ReservoirPlasticConfig):
        super().__init__()
        self.config = config
        self.core = _ReservoirCore(
            config.d_res, config.vocab_size, config.spectral_radius, config.seed
        )
        self.readout = nn.Linear(config.d_res, config.vocab_size)
        # Deterministic seed readout from a stream distinct from the core's (so a fixed seed
        # pins forward's output for the determinism test), mirroring ``Reservoir``.
        gen = torch.Generator().manual_seed(config.seed + 1)
        with torch.no_grad():
            self.readout.weight.normal_(0.0, 0.02, generator=gen)
            self.readout.bias.zero_()
        self._levels = config.vocab_size - N_ACTIONS
        self._conc = slice(0, self._levels)
        self._action = slice(self._levels, config.vocab_size)

    # --- distill path: forward + the default backprop train_step ----------------

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        states = self.core.run(idx, self.config.leak)  # (B,T,d_res), no grad through core
        return self.readout(states)  # (B,T,vocab); backward flows only into the seed readout

    def _per_token_forward(self) -> int:
        """Per-step forward (predict): recurrence + plastic-readout matvec + the bias add —
        identical to ``Reservoir``'s per-token forward."""
        d, v = self.config.d_res, self.config.vocab_size
        return self.core.recurrence_flops() + matmul_flops(1, v, d) + pointwise_flops(v)

    def _readout_backward(self) -> int:
        """Distill-path backward: the readout ``dW_out`` outer product + ``db_out`` (the
        frozen core gets 0). Drives only the rarely-exercised distillation harness."""
        v = self.config.vocab_size
        return matmul_flops(1, v, self.config.d_res) + pointwise_flops(v)

    def flops(self, seq_len: int) -> FlopBreakdown:
        """The (mostly unused) distill path: forward = ``seq_len ×`` per-token forward;
        backward = ``seq_len ×`` readout-only (NOT 2× forward — the frozen core gets 0)."""
        return FlopBreakdown(
            forward=seq_len * self._per_token_forward(),
            backward=seq_len * self._readout_backward(),
        )

    def configure_optimizer(
        self, *, lr: float, weight_decay: float, betas: tuple[float, float]
    ) -> torch.optim.Optimizer:
        """AdamW over the trainable seed readout only; the frozen core is excluded. (The
        headline run does ~0 distillation, so this optimizer is rarely stepped.)"""
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=lr, betas=betas, weight_decay=weight_decay)

    # --- online-update FLOP accounting (charged to step.backward) ---------------

    def _wm_update_flops(self) -> int:
        """World-model delta-rule cost on the ``conc_slice`` columns (levels × d_res)."""
        d, lv = self.config.d_res, self._levels
        return (
            pointwise_flops(lv, per_elem=3)  # softmax(prev conc-pred logits) over levels
            + pointwise_flops(lv, per_elem=2)  # onehot(c) target + (target − pred)
            + matmul_flops(d, lv, 1)  # outer(err, h_action) : the delta-rule ΔW
            + pointwise_flops(d * lv, per_elem=2)  # W[conc] += lr_wm·ΔW : scale + add
            + pointwise_flops(lv, per_elem=2)  # b[conc] += lr_wm·err : scale + add
        )

    def _policy_update_flops(self) -> int:
        """Reward-modulated Hebbian cost on the ``action_slice`` columns (N_ACTIONS × d_res)."""
        d, na = self.config.d_res, N_ACTIONS
        return (
            pointwise_flops(5)  # r (1), adv = r-baseline (1), leaky baseline b+decay·(r-b) (3)
            + pointwise_flops(na)  # onehot(a_taken)
            + matmul_flops(d, na, 1)  # outer(onehot_a, h_conc) : the Hebbian ΔW
            + pointwise_flops(d * na, per_elem=2)  # W[action] += lr_pol·adv·ΔW : scale + add
            + pointwise_flops(na, per_elem=2)  # b[action] += lr_pol·adv·onehot_a : scale + add
        )

    def _online_update_flops(self) -> int:
        """Total FLOPs of one online update (fires on a conc fold at ``pos>=2``)."""
        return self._wm_update_flops() + self._policy_update_flops()

    # --- prequential / online decode seam: h + the plastic (W,b) ARE the memory -

    def init_prequential_state(self) -> DecodeState:
        """Seed the cache: ``h_0 = 0`` and a fresh CLONE of the seed readout as the plastic
        ``(W, b)`` (never an ``nn.Parameter``, so eval never mutates a model weight)."""
        with torch.no_grad():
            cache = _PlasticCache(
                h=self.core.initial_state(),
                W=self.readout.weight.detach().clone(),
                b=self.readout.bias.detach().clone(),
                baseline=0.0,
                h_conc=None,
                action_token=None,
                conc_pred_logits=None,
                h_action=None,
            )
        return DecodeState(cache=cache)

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        """Fold one token, run the online update (on a conc fold at ``pos>=2``), read out.

        Tape parity (from ``evaluate_control``): EVEN ``pos`` = concentration, ODD = action,
        in order ``c0, a0, c1, a1, …``. The update is one step late by construction: the
        reward (the concentration after the action) and the world-model target (the same
        concentration) are only revealed at the NEXT conc fold. Forward is the recurrence +
        readout; backward is the (nonzero) update cost on conc folds, 0 otherwise."""
        cfg = self.config
        c_in: _PlasticCache = state.cache
        with torch.no_grad():
            h_new = self.core.fold(c_in.h, revealed_byte, cfg.leak)
            W, b, baseline = c_in.W, c_in.b, c_in.baseline
            backward = 0
            is_conc = pos % 2 == 0
            ready = (
                c_in.h_conc is not None
                and c_in.action_token is not None
                and c_in.conc_pred_logits is not None
                and c_in.h_action is not None
            )
            if is_conc and pos >= 2 and ready:
                W, b = W.clone(), b.clone()
                lv = self._levels
                # World-model delta rule on the conc columns: push the prediction made at the
                # preceding action-fold toward the just-revealed concentration.
                target = torch.zeros(lv, device=W.device, dtype=W.dtype)
                target[revealed_byte] = 1.0
                pred = torch.softmax(c_in.conc_pred_logits[self._conc], dim=-1)
                err = target - pred
                W[self._conc] += cfg.lr_wm * torch.outer(err, c_in.h_action)
                b[self._conc] += cfg.lr_wm * err
                # Reward-modulated Hebbian policy update on the action columns, with a leaky
                # baseline; reward proxy = the observed concentration level.
                r = revealed_byte / (lv - 1)
                adv = r - baseline
                baseline = baseline + cfg.reward_decay * (r - baseline)
                oh_a = torch.zeros(N_ACTIONS, device=W.device, dtype=W.dtype)
                oh_a[c_in.action_token - lv] = 1.0
                W[self._action] += cfg.lr_pol * adv * torch.outer(oh_a, c_in.h_conc)
                b[self._action] += cfg.lr_pol * adv * oh_a
                backward = self._online_update_flops()

            logits = (W @ h_new + b).detach()  # the plastic readout (uses the updated W,b)

            if is_conc:
                # This conc-fold's state is what the upcoming action will be sampled from.
                new_cache = _PlasticCache(
                    h=h_new,
                    W=W,
                    b=b,
                    baseline=baseline,
                    h_conc=h_new,
                    action_token=c_in.action_token,
                    conc_pred_logits=c_in.conc_pred_logits,
                    h_action=c_in.h_action,
                )
            else:
                # Action fold: stash the action, its conc prediction, and the action-fold state.
                new_cache = _PlasticCache(
                    h=h_new,
                    W=W,
                    b=b,
                    baseline=baseline,
                    h_conc=c_in.h_conc,
                    action_token=revealed_byte,
                    conc_pred_logits=logits,
                    h_action=h_new,
                )
        flops = FlopBreakdown(forward=self._per_token_forward(), backward=backward)
        return DecodeState(cache=new_cache, length=state.length + 1), logits, flops

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        """Forward-only per-step predict cost: ``O(d_res²)``, independent of ``context_len``
        (the online update is adaptation, reported in :meth:`step`'s ``backward``)."""
        return FlopBreakdown(forward=self._per_token_forward(), backward=0)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ReservoirPlastic":
        return cls(ReservoirPlasticConfig(**config))
