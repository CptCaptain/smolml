"""Minimal-organism control candidate (Task C.A.2) — `chemotaxis_min`.

The purest Source-(iv) bet on the control rung: a bacterium does chemotaxis with
almost no machinery — integrate the sensed concentration, ask "am I improving?",
and bias the next move (run-and-tumble: keep heading while it rises, reverse on a
drop). The **in-context adaptation is the integrator state** — a leaky baseline
`b` tracking the recent concentration — not weight learning. So the mechanism is a
*handful of learnable scalars* plus a few pointwise ops per `step`, and its total
FLOPs is dominated by the (cheap) eval rollout, counted honestly through `step`
(ADR 0004). If it competes with a distilled transformer on **regret per total
FLOP**, that is the cleanest possible source-(iv) result — and the FLOP-floor
reference for the control rung.

The learnable scalars (five `nn.Parameter`s, so `num_params == 5`):

- ``leak_logit`` — leak ``λ = sigmoid(leak_logit)`` of the baseline EMA;
- ``g`` — policy gain on the surprise;
- ``stay_bias`` — standing logit for the STAY action;
- ``climb`` / ``sharpness`` — the small world-model predictor (peak location shift
  and peakedness).

They are differentiable through the whole recurrence, so a short distillation can
tune them (the default backprop `train_step` applies) — but the dominant adaptation
is ``b`` updating each step, which costs **no weight change**.

Tape parity (see ``smolml/envs/chemotaxis.py``): EVEN positions are concentration
levels, ODD positions are action tokens. Every `step`/`forward` position emits
**full-vocab** logits ``[conc-head over levels | action-head over N_ACTIONS]``; the
scorer reads the slice for the *next* position's parity (it always populates both).

FLOP accounting (the product — honest, via `smolml.flops`)
----------------------------------------------------------
All compute is **pointwise** (no matmul dominates), so it MUST be charged via
``pointwise_flops`` — the flops module's conditional-omission rule (``flops.py``
docstring) says a non-matmul mechanism charges its real work or the instrument
scores it as free. The per-step forward op count is fixed and context-independent
(:meth:`ChemotaxisMin._per_step_ops`); :meth:`flops` charges ``seq_len`` of them
(backward = the standard ``2x`` for the distilled scalars, which participate
throughout — :meth:`~smolml.flops.FlopBreakdown.from_forward`), :meth:`step` and
:meth:`decode_step_flops` charge exactly one (``backward = 0`` — the integrator
update is forward compute, no weight change).
"""

import dataclasses
import math
from dataclasses import dataclass

import torch
from torch import nn

from smolml.envs.chemotaxis import N_ACTIONS
from smolml.flops import FlopBreakdown, pointwise_flops
from smolml.models.registry import DecodeState, LanguageModel, register_model

# Action indices (mirror ``ACTION_DELTAS = (-1, 0, 1)`` in smolml/envs/chemotaxis.py).
LEFT, STAY, RIGHT = 0, 1, 2

# --- Per-step pointwise op count (charged honestly; hand-checkable) ------------
# Every elementwise op the mechanism performs for ONE token, counted as one pointwise
# FLOP (conservative: the heavier EVEN/sense branch is charged on every step). Includes
# the action-head compare + where-selects + stack and the full-vocab cat construction,
# so the mechanism is never scored cheaper than the elementwise work it actually does.
_LEAK_OPS = 5  # leak = sigmoid(leak_logit): neg, exp, +1, recip (4) + (1-leak) (1)
_SENSE_OPS = 4  # surprise s = c - b (1); EMA b' = (1-λ)·b + λ·c = 2 mul + 1 add (3)
_ACTION_OPS = 8  # keep=g·s (1), reverse=-keep (1), is_left compare (1), 2 where (2), stack-3 (3)
_CENTER_OPS = 6  # sign(s)(1)+climb·sign(1)+c+·(1)+clamp lo(1)+clamp hi(1)+unsqueeze(1)
_CAT_ACTION_OPS = 3  # action-slice (N_ACTIONS) copy in the final full-vocab cat
_PER_LEVEL_OPS = 5  # per level: arange(1)+diff(1)+square(1)+·(-sharpness)(1)+conc cat copy(1)


@dataclass
class ChemoMinState:
    """Per-stream integrator state threaded through :meth:`ChemotaxisMin.step`.

    This *is* the in-context memory: ``b`` (the leaky concentration baseline)
    accumulates history with no weight change. ``s`` is the last surprise,
    ``c`` the last sensed level, ``last_action`` the last folded action index.
    """

    b: float
    s: float
    c: float
    last_action: int


@dataclass
class ChemoMinConfig:
    """Hyperparameters: the vocab/window the rung injects, plus scalar inits.

    ``levels`` is derived as ``vocab_size - N_ACTIONS`` (the env packs the two heads
    into one disjoint id space). The ``*_init`` values seed the learnable scalars;
    they are tuned to clear the random-policy floor *untrained* (≈0 distillation).
    """

    vocab_size: int = 8 + N_ACTIONS
    max_seq_len: int = 2 * 64 + 1
    leak_init: float = 0.85  # baseline tracks the recent concentration closely
    gain_init: float = 2.5  # confident run-and-tumble under sampled actions
    stay_bias_init: float = -1.5  # standing is mildly disfavored
    climb_init: float = 1.0  # world model expects the next level one step up-gradient
    sharpness_init: float = 1.0  # world-model peakedness
    baseline_init: float = 0.0  # initial leaky baseline b0

    def __post_init__(self) -> None:
        if self.vocab_size <= N_ACTIONS:
            raise ValueError(f"vocab_size must exceed N_ACTIONS={N_ACTIONS}, got {self.vocab_size}")
        if not 0.0 < self.leak_init < 1.0:
            raise ValueError(f"leak_init must be in (0, 1), got {self.leak_init}")


@register_model("chemotaxis_min")
class ChemotaxisMin(LanguageModel):
    """Hand-structured run-and-tumble controller; five learnable scalars; the
    in-context adaptation is a leaky integrator (no weight change at eval)."""

    def __init__(self, config: ChemoMinConfig):
        super().__init__()
        self.config = config
        self.levels = config.vocab_size - N_ACTIONS
        leak_logit = math.log(config.leak_init / (1.0 - config.leak_init))
        self.leak_logit = nn.Parameter(torch.tensor(leak_logit))
        self.g = nn.Parameter(torch.tensor(float(config.gain_init)))
        self.stay_bias = nn.Parameter(torch.tensor(float(config.stay_bias_init)))
        self.climb = nn.Parameter(torch.tensor(float(config.climb_init)))
        self.sharpness = nn.Parameter(torch.tensor(float(config.sharpness_init)))

    # --- the shared head: full-vocab logits from the current integrator state ---

    def _emit_logits(
        self, s: torch.Tensor, c_last: torch.Tensor, last_action: torch.Tensor
    ) -> torch.Tensor:
        """Full-vocab logits ``[conc(levels) | action(N_ACTIONS)]`` from state.

        Works for any leading shape (a 0-dim scalar in :meth:`step`, a ``(B,)``
        batch per position in :meth:`forward`), so both channels run the *identical*
        arithmetic and ``step`` matches ``forward`` byte-for-byte.
        """
        levels = self.levels
        # Policy (action) head — run-and-tumble around the current heading.
        keep = self.g * s  # improving (s>0) reinforces the kept direction
        reverse = -keep
        is_left = last_action == LEFT  # heading LEFT keeps LEFT; else keep RIGHT
        left_logit = torch.where(is_left, keep, reverse)
        right_logit = torch.where(is_left, reverse, keep)
        stay_logit = self.stay_bias.expand_as(s)
        action = torch.stack([left_logit, stay_logit, right_logit], dim=-1)
        # World-model (concentration) head — a peak at the level one step up-gradient.
        center = torch.clamp(c_last + self.climb * torch.sign(s), 0.0, float(levels - 1))
        grid = torch.arange(levels, device=s.device, dtype=s.dtype)
        diff = grid - center.unsqueeze(-1)
        conc = (-self.sharpness) * diff * diff
        return torch.cat([conc, action], dim=-1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Teacher-forced recurrence over ``(B, T)`` -> ``(B, T, vocab)`` logits.

        Sequential in ``b``/``last_action`` (a Python scan over ``T``), but
        differentiable in the five scalars so ``distill_train_run`` can next-token
        train them. Position parity is the within-row index (tapes start at ``c0``,
        so it equals the absolute tape parity ``step`` sees)."""
        if idx.dim() != 2:
            raise ValueError(f"expected (batch, seq_len) int ids, got shape {tuple(idx.shape)}")
        cfg = self.config
        batch, length = idx.shape
        device = self.g.device
        dtype = self.g.dtype
        levels = self.levels
        leak = torch.sigmoid(self.leak_logit)
        one_minus = 1.0 - leak

        b = torch.full((batch,), cfg.baseline_init, device=device, dtype=dtype)
        s = torch.zeros(batch, device=device, dtype=dtype)
        c_last = torch.zeros(batch, device=device, dtype=dtype)
        last_action = torch.full((batch,), RIGHT, device=device, dtype=torch.long)
        out = torch.empty((batch, length, cfg.vocab_size), device=device, dtype=dtype)

        for t in range(length):
            tok = idx[:, t]
            if t % 2 == 0:  # EVEN — a sensed concentration level
                c = tok.to(dtype)
                s = c - b  # surprise BEFORE the baseline update
                b = one_minus * b + leak * c
                c_last = c
            else:  # ODD — an action token
                last_action = tok - levels
            out[:, t] = self._emit_logits(s, c_last, last_action)
        return out

    def flops(self, seq_len: int) -> FlopBreakdown:
        """Analytic per-sequence cost: ``seq_len`` pointwise per-step ops, with the
        standard ``2x`` backward for the distilled scalars (they participate at every
        position, so the distill path is an honest ``3x`` of forward)."""
        forward = pointwise_flops(seq_len * self._per_step_ops())
        return FlopBreakdown.from_forward(forward)

    def _per_step_ops(self) -> int:
        """Pointwise op count for ONE token (fold + full-vocab logit assembly).
        Context-independent (the integrator is O(1)); linear in ``levels``."""
        return (
            _LEAK_OPS
            + _SENSE_OPS
            + _ACTION_OPS
            + _CENTER_OPS
            + _CAT_ACTION_OPS
            + _PER_LEVEL_OPS * self.levels
        )

    # --- prequential / online control seam --------------------------------------

    def init_prequential_state(self) -> DecodeState:
        """Fresh per-rollout integrator: baseline at ``baseline_init``, heading
        RIGHT, no surprise yet. O(1) memory — no token window is retained."""
        cache = ChemoMinState(b=self.config.baseline_init, s=0.0, c=0.0, last_action=RIGHT)
        return DecodeState(cache=cache)

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        """Fold ``revealed_byte`` (branch on parity), update the integrator, emit the
        NEXT full-vocab logits. The leaky-baseline update IS the in-context
        adaptation — no weight change — so ``backward = 0`` and the forward charge is
        the same per-step pointwise count as :meth:`decode_step_flops`."""
        st: ChemoMinState = state.cache
        levels = self.levels
        dtype = self.g.dtype
        leak = torch.sigmoid(self.leak_logit)
        one_minus = 1.0 - leak

        if pos % 2 == 0:  # EVEN — a sensed concentration level
            c = torch.tensor(float(revealed_byte), dtype=dtype, device=self.g.device)
            b_prev = torch.tensor(st.b, dtype=dtype, device=self.g.device)
            s = c - b_prev
            b_new = one_minus * b_prev + leak * c
            new_st = ChemoMinState(
                b=float(b_new.item()),
                s=float(s.item()),
                c=float(c.item()),
                last_action=st.last_action,
            )
        else:  # ODD — an action token
            new_st = ChemoMinState(b=st.b, s=st.s, c=st.c, last_action=int(revealed_byte) - levels)

        s_t = torch.tensor(new_st.s, dtype=dtype, device=self.g.device)
        c_t = torch.tensor(new_st.c, dtype=dtype, device=self.g.device)
        a_t = torch.tensor(new_st.last_action, dtype=torch.long, device=self.g.device)
        logits = self._emit_logits(s_t, c_t, a_t)
        flops = self.decode_step_flops(state.length)
        return DecodeState(cache=new_st, length=state.length + 1), logits, flops

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        """Forward-only per-step cost (context-independent); ``backward = 0`` — the
        integrator update is forward compute, not a weight update."""
        return FlopBreakdown(forward=pointwise_flops(self._per_step_ops()), backward=0)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ChemotaxisMin":
        """Build from a config dict, ignoring harness-injected transformer keys
        (e.g. ``d_model``) this model does not use, so it runs the CLI/driver
        unchanged."""
        fields = {f.name for f in dataclasses.fields(ChemoMinConfig)}
        kwargs = {k: v for k, v in config.items() if k in fields}
        return cls(ChemoMinConfig(**kwargs))
