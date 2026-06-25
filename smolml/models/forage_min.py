"""Per-type contingency-tracker control candidate (Task C.A.4) — `forage_min`.

The bandit analogue of `chemotaxis_min` on the reflex-proof forage rung. Forage is a
contextual bandit over `K` cue types: exactly one latent type `g` pays `+1` on EAT
(others poison `-1`), fresh per episode, and the agent senses only its current cell's
combined `(type, last_reward)` obs. The optimal play is to infer `g` from its own
eat-outcomes, then camp a `g` cell — pure in-context contingency identification.

The mechanism swaps `chemotaxis_min`'s scalar leaky-integrator baseline for a **per-type
value vector** ``v[K]`` — the in-context memory (reset every episode, NOT a parameter).
A local delta rule credits the eaten type from the revealed reward; a distilled-scalar
softmax policy eats high-value types and searches past poison. This is win-stay-lose-shift
expressed as a learned, differentiable, FLOP-counted ``LanguageModel`` — perfect O(1) local
credit assignment (the agent knows exactly which type it ate, since EAT does not move).

The learnable scalars (eight `nn.Parameter`s, so `num_params == 8`):

- ``lr_logit`` — delta-rule rate ``lr = sigmoid(lr_logit)`` of the per-type value EMA;
- ``v_init`` — optimistic value prior (drives explore-by-eating of unknown types);
- ``g`` — policy gain on the current type's value;
- ``b_eat`` / ``b_left`` / ``b_right`` — standing action logits (search bias);
- ``g_wm`` — world-model gain mapping the value belief to the predicted reward level;
- ``stick`` — world-model type stickiness (EAT stays on the same cell type).

All are differentiable through the recurrence, so a short distillation can tune them, but
the dominant adaptation is ``v`` updating each step — no weight change at eval.

Tape parity (see ``smolml/envs/forage.py``): EVEN positions are combined obs
``type*3 + (reward+1)``, ODD positions are action tokens ``3K + action``. Every
`step`/`forward` position emits **full-vocab** logits ``[obs(3K) | action(N_ACTIONS)]``;
the scorer reads the slice for the *next* position's parity (it always populates both).

FLOP accounting (the product — honest, via `smolml.flops`)
----------------------------------------------------------
All compute is **pointwise** (no matmul dominates), so it MUST be charged via
``pointwise_flops`` — the flops module's conditional-omission rule says a non-matmul
mechanism charges its real work or the instrument scores it as free. The per-step op count
is fixed and context-independent (:meth:`ForageMin._per_step_ops`, hand-checkable named
constants); :meth:`flops` charges ``seq_len`` of them (backward = the standard ``2x`` for
the distilled scalars, which participate throughout — :meth:`~smolml.flops.FlopBreakdown.
from_forward`), :meth:`step` and :meth:`decode_step_flops` charge exactly one (``backward =
0`` — the ``v`` delta-rule update is forward compute, no weight change).
"""

import dataclasses
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from smolml.envs.forage import EAT, N_ACTIONS, REWARD_LEVELS, RIGHT
from smolml.flops import FlopBreakdown, pointwise_flops
from smolml.models.registry import DecodeState, LanguageModel, register_model

# --- Per-step pointwise op count (charged honestly; hand-checkable) ------------
# Every elementwise op the mechanism performs for ONE token, counted as one pointwise FLOP
# (conservative: the heavier EVEN/obs branch — decode + delta update — is charged on every
# step). The world-model head's per-type and per-combined-symbol work scales with K.
_LR_OPS = 4  # lr = sigmoid(lr_logit): neg, exp, +1, recip
_DECODE_OPS = 3  # t = byte // 3 (1); rem = byte % 3 (1); r = rem - 1 (1)
_UPDATE_OPS = 6  # gather v[t] (1); r - v[t] (1); lr * (.) (1); v[t] + (.) (1); EAT-gate (1); scatter (1)
_VCUR_OPS = 1  # v_cur = v[current_type] : a gather for the heads
_POLICY_OPS = 7  # eat = g*v_cur (1) + b_eat (1); left/right expand (2); stack-3 (3)
_REW_OPS = 11  # gv = g_wm*v_cur (1), -gv (1); stack eat-row (3); stack move-row (3); where over 3 (3)
_PER_TYPE_OPS = 3  # per type: one-hot build (1); * stick (1); where vs zeros (1)
_PER_SYMBOL_OPS = 2  # per combined obs symbol: outer-sum add (1); final cat copy (1)


@dataclass
class ForageMinState:
    """Per-stream tracker state threaded through :meth:`ForageMin.step`.

    ``v`` is the per-type value vector (the in-context memory; reset each episode, NOT a
    parameter). ``current_type`` is the most recently sensed cue type (the cell the agent is
    on), ``last_action`` is the most recent action index (``RIGHT`` at reset, before any act).
    """

    v: list[float]
    current_type: int
    last_action: int


@dataclass
class ForageMinConfig:
    """Hyperparameters: the vocab/window the rung injects, plus scalar inits.

    ``K`` (number of cue types) is derived from the injected ``vocab_size``:
    ``K = (vocab_size - N_ACTIONS) // REWARD_LEVELS``.
    """

    vocab_size: int = REWARD_LEVELS * 3 + N_ACTIONS
    max_seq_len: int = 256
    lr_init: float = 0.8  # delta-rule rate (fast: one observation flips a type's value)
    v_init: float = 0.3  # optimistic prior -> explore unknown types by eating
    gain_init: float = 8.0  # policy gain: large => firm (near-greedy) camping under sampling
    eat_bias_init: float = 0.0
    left_bias_init: float = -2.0  # discourage LEFT; search goes RIGHT
    right_bias_init: float = 0.5  # default search direction
    wm_gain_init: float = 2.0  # world-model: value -> predicted reward level
    stick_init: float = 2.0  # world-model: EAT stays on the same cue type

    def __post_init__(self):
        levels = self.vocab_size - N_ACTIONS
        if levels < REWARD_LEVELS or levels % REWARD_LEVELS != 0:
            raise ValueError(
                f"vocab_size must leave a multiple of {REWARD_LEVELS} obs symbols, got {levels}"
            )
        if not 0.0 < self.lr_init < 1.0:
            raise ValueError(f"lr_init must be in (0, 1), got {self.lr_init}")


@register_model("forage_min")
class ForageMin(LanguageModel):
    """Per-type contingency tracker; eight learnable scalars; the in-context adaptation is a
    per-type value EMA (no weight change at eval)."""

    def __init__(self, config: ForageMinConfig):
        super().__init__()
        self.config = config
        self.obs_len = config.vocab_size - N_ACTIONS
        self.K = self.obs_len // REWARD_LEVELS
        lr_logit = math.log(config.lr_init / (1.0 - config.lr_init))
        self.lr_logit = nn.Parameter(torch.tensor(lr_logit))
        self.v_init = nn.Parameter(torch.tensor(float(config.v_init)))
        self.g = nn.Parameter(torch.tensor(float(config.gain_init)))
        self.b_eat = nn.Parameter(torch.tensor(float(config.eat_bias_init)))
        self.b_left = nn.Parameter(torch.tensor(float(config.left_bias_init)))
        self.b_right = nn.Parameter(torch.tensor(float(config.right_bias_init)))
        self.g_wm = nn.Parameter(torch.tensor(float(config.wm_gain_init)))
        self.stick = nn.Parameter(torch.tensor(float(config.stick_init)))

    # --- the shared head: full-vocab logits from the current value belief -------

    def _emit_logits(
        self, v_cur: torch.Tensor, type_onehot: torch.Tensor, last_action: torch.Tensor
    ) -> torch.Tensor:
        """Full-vocab logits ``[obs(3K) | action(N_ACTIONS)]`` from the current type's value.

        Shape-polymorphic in the leading dims: a 0-dim scalar path in :meth:`step`, a ``(B,)``
        batch per position in :meth:`forward` (with ``v_cur`` ``(...)``, ``type_onehot``
        ``(..., K)``, ``last_action`` ``(...)``), so both channels run the *identical*
        arithmetic and ``step`` matches ``forward``.
        """
        zeros = torch.zeros_like(v_cur)
        # Policy (action) head: eat high-value types, search past poison.
        eat = self.g * v_cur + self.b_eat
        left = self.b_left.expand_as(v_cur)
        right = self.b_right.expand_as(v_cur)
        action = torch.stack([left, eat, right], dim=-1)  # order LEFT, EAT, RIGHT
        # World-model (obs) head: predict the next combined (type, reward) symbol.
        is_eat = (last_action == EAT).unsqueeze(-1)
        gv = self.g_wm * v_cur
        rew_eat = torch.stack([-gv, zeros, gv], dim=-1)  # reward {-1,0,+1}: follow the belief
        stick_b = self.stick.expand_as(v_cur)
        rew_move = torch.stack([zeros, stick_b, zeros], dim=-1)  # a move -> reward 0 is certain
        rew = torch.where(is_eat, rew_eat, rew_move)  # (..., REWARD_LEVELS)
        typ = torch.where(is_eat, self.stick * type_onehot, torch.zeros_like(type_onehot))
        # Combined obs logits: obs[t*REWARD_LEVELS + rho] = typ[t] + rew[rho].
        grid = typ.unsqueeze(-1) + rew.unsqueeze(-2)  # (..., K, REWARD_LEVELS)
        obs = grid.reshape(*grid.shape[:-2], self.obs_len)
        return torch.cat([obs, action], dim=-1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Teacher-forced recurrence over ``(B, T)`` -> ``(B, T, vocab)`` logits.

        Sequential in the per-type value ``v`` (a Python scan over ``T``), but differentiable
        in the eight scalars so ``distill_train_run`` can next-token train them. Position
        parity is the within-row index (tapes start at obs ``c0``, so it equals the absolute
        tape parity ``step`` sees)."""
        if idx.dim() != 2:
            raise ValueError(f"expected (batch, seq_len) int ids, got shape {tuple(idx.shape)}")
        batch, length = idx.shape
        device, dtype = self.g.device, self.g.dtype
        lr = torch.sigmoid(self.lr_logit)

        v = torch.full((batch, self.K), float(self.v_init.item()), device=device, dtype=dtype)
        # Keep v differentiable in v_init (the optimistic prior is a learnable scalar).
        v = v - float(self.v_init.item()) + self.v_init
        current_type = torch.zeros(batch, device=device, dtype=torch.long)
        last_action = torch.full((batch,), RIGHT, device=device, dtype=torch.long)
        out = torch.empty((batch, length, self.config.vocab_size), device=device, dtype=dtype)

        for t in range(length):
            tok = idx[:, t]
            if t % 2 == 0:  # EVEN — a combined obs symbol
                cur = tok // REWARD_LEVELS
                r = (tok % REWARD_LEVELS - 1).to(dtype)
                v_cur_old = v.gather(1, cur.unsqueeze(1)).squeeze(1)
                target = v_cur_old + lr * (r - v_cur_old)
                is_eat = last_action == EAT
                v_cur_new = torch.where(is_eat, target, v_cur_old)
                v = v.scatter(1, cur.unsqueeze(1), v_cur_new.unsqueeze(1))
                current_type = cur
            else:  # ODD — an action token
                last_action = tok - self.obs_len
            v_cur = v.gather(1, current_type.unsqueeze(1)).squeeze(1)
            type_onehot = F.one_hot(current_type, self.K).to(dtype)
            out[:, t] = self._emit_logits(v_cur, type_onehot, last_action)
        return out

    def _per_step_ops(self) -> int:
        """Pointwise op count for ONE token (fold + delta update + full-vocab logit assembly).
        Context-independent (the tracker is O(K)); linear in ``K``."""
        return (
            _LR_OPS
            + _DECODE_OPS
            + _UPDATE_OPS
            + _VCUR_OPS
            + _POLICY_OPS
            + _REW_OPS
            + _PER_TYPE_OPS * self.K
            + _PER_SYMBOL_OPS * self.obs_len
        )

    def flops(self, seq_len: int) -> FlopBreakdown:
        """Analytic per-sequence cost: ``seq_len`` pointwise per-step ops, with the standard
        ``2x`` backward for the distilled scalars (they participate at every position, so the
        distill path is an honest ``3x`` of forward)."""
        return FlopBreakdown.from_forward(pointwise_flops(seq_len * self._per_step_ops()))

    # --- prequential / online control seam --------------------------------------

    def init_prequential_state(self) -> DecodeState:
        """Fresh per-rollout tracker: every type at the optimistic ``v_init`` prior, heading
        RIGHT, no type sensed yet. O(K) memory — no token window is retained."""
        cache = ForageMinState(
            v=[float(self.v_init.item())] * self.K, current_type=0, last_action=RIGHT
        )
        return DecodeState(cache=cache)

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        """Fold ``revealed_byte`` (branch on parity), run the local delta update on a post-EAT
        obs fold, emit the NEXT full-vocab logits. The per-type value update IS the in-context
        adaptation — no weight change — so ``backward = 0`` and the forward charge equals
        :meth:`decode_step_flops`."""
        st: ForageMinState = state.cache
        device, dtype = self.g.device, self.g.dtype
        v = torch.tensor(st.v, dtype=dtype, device=device)

        if pos % 2 == 0:  # EVEN — a combined obs symbol
            cur = int(revealed_byte) // REWARD_LEVELS
            r = float(int(revealed_byte) % REWARD_LEVELS - 1)
            if st.last_action == EAT:  # credit the eaten type (EAT does not move)
                lr = torch.sigmoid(self.lr_logit)
                v[cur] = v[cur] + lr * (r - v[cur])
            new_st = ForageMinState(v=v.tolist(), current_type=cur, last_action=st.last_action)
        else:  # ODD — an action token
            new_st = ForageMinState(
                v=st.v, current_type=st.current_type, last_action=int(revealed_byte) - self.obs_len
            )

        v_now = torch.tensor(new_st.v, dtype=dtype, device=device)
        v_cur = v_now[new_st.current_type]
        type_onehot = F.one_hot(
            torch.tensor(new_st.current_type, device=device), self.K
        ).to(dtype)
        last_action = torch.tensor(new_st.last_action, device=device)
        logits = self._emit_logits(v_cur, type_onehot, last_action)
        flops = self.decode_step_flops(state.length)
        return DecodeState(cache=new_st, length=state.length + 1), logits, flops

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        """Forward-only per-step cost (context-independent); ``backward = 0`` — the per-type
        value update is forward compute, not a weight update."""
        return FlopBreakdown(forward=pointwise_flops(self._per_step_ops()), backward=0)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ForageMin":
        """Build from a config dict, ignoring harness-injected transformer keys (e.g.
        ``d_model``) this model does not use, so it runs the CLI/driver unchanged."""
        fields = {f.name for f in dataclasses.fields(ForageMinConfig)}
        kwargs = {k: v for k, v in config.items() if k in fields}
        return cls(ForageMinConfig(**kwargs))
