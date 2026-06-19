"""Online context-mixing reference (PAQ / cmix / nncp lineage).

A **non-candidate yardstick**, not an entrant. It measures the bits-per-byte that
*single-pass online learning* can reach per FLOP — the target a real candidate
should try to approach. The lineage is the Hutter-Prize winners: the best enwik8
compressors are not trained transformers but online **context-mixing** predictors
that combine many cheap sub-models with weights learned in a single streaming
pass (see ``docs/candidates.md`` -> "Reality check: our metric *is* the Hutter
Prize").

Mechanism (deliberately simple)
-------------------------------
- A handful of **order-k byte models** (k = 0..``max_order``). Order-k predicts the
  next byte from a conditional frequency table keyed by the last ``k`` bytes.
- Their predictions are combined by **logistic mixing**: each model's probability
  vector ``p_k`` is *stretched* to ``s_k = log p_k``, the stretched vectors are
  combined as ``z = Σ_k w_k · s_k``, and ``softmax(z)`` is the mixed distribution
  (multiclass generalisation of PAQ's ``squash(Σ w_k · stretch(p_k))``).
- The mixing weights ``w`` are learned **online** by one SGD step of multinomial
  logistic regression on the revealed byte. There is **no gradient pretraining**:
  ``train_step`` is trivial; *all* learning happens inside :meth:`step`.

FLOP honesty (the whole point — ADR 0003/0004, ``smolml.flops``)
----------------------------------------------------------------
This mechanism's dominant compute is **not** matmuls — it is table lookups, count
updates, and logistic mixing. Charging only matmuls would score it as nearly free
(a silent cheat). So every per-byte op is charged through the non-matmul
primitives :func:`~smolml.flops.pointwise_flops` and
:func:`~smolml.flops.gather_flops` — and charged **exactly for the branches
executed** that byte (no constant over-estimate). The derivation lives in
:meth:`ContextMixing._flop_breakdown` and the concept page
``docs/learning/concepts/context-mixing.md``.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from smolml.flops import FlopBreakdown, gather_flops, pointwise_flops
from smolml.models.registry import DecodeState, LanguageModel, register_model


@dataclass
class ContextMixingConfig:
    """Hyperparameters for the online context-mixing reference.

    ``max_order`` selects which order-k models run (k = 0..max_order), so the
    number of sub-predictors is ``max_order + 1``. Sweeping ``max_order`` traces
    the reference's bpb-vs-FLOP curve (more orders = more compute per byte).
    """

    max_order: int = 3
    alpha: float = 0.5  # Laplace smoothing added to every byte count
    lr: float = 0.02  # online learning rate for the logistic mixer
    vocab_size: int = 256

    def __post_init__(self) -> None:
        if self.max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {self.max_order}")
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")


@dataclass
class _MixerState:
    """Per-stream online state threaded through :meth:`ContextMixing.step`.

    ``tables[k]`` maps a length-k context (as ``bytes``) to its byte-count vector;
    ``weights`` are the mixer weights; ``last_*`` cache the prediction made for the
    byte that is about to be revealed, so the mixer can be graded on it next step.
    """

    tables: list[dict[bytes, np.ndarray]]
    weights: np.ndarray
    last_stretched: np.ndarray | None = None  # (K, V) stretched inputs of the pending prediction
    last_probs: np.ndarray | None = None  # (V,) the pending predicted distribution


def laplace_prob(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Add-``alpha`` (Laplace) smoothed distribution from a byte-count vector.

    An all-zero (unseen) context yields the uniform distribution, so an order-k
    model that has no statistics yet abstains (its stretched vector is constant
    and cancels under the mixer's softmax).
    """
    return (counts + alpha) / (counts.sum() + alpha * counts.shape[0])


def mix_logits(stretched: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Logistic-mixing combination ``z[b] = Σ_k weights[k] · stretched[k, b]``."""
    return weights @ stretched


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax of a 1-D logit vector."""
    shifted = z - z.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def mixer_gradient(probs: np.ndarray, stretched: np.ndarray, target: int) -> np.ndarray:
    """Gradient of the cross-entropy loss w.r.t. the mixer weights.

    For ``L = −log softmax(z)[target]`` with ``z = Σ_k w_k · s_k``, the softmax-CE
    gradient is ``∂L/∂z[b] = probs[b] − 1{b == target}`` and ``∂z[b]/∂w_k =
    s_k[b]``, so ``∂L/∂w_k = Σ_b (probs[b] − 1{b == target}) · s_k[b]``.
    """
    err = probs.copy()
    err[target] -= 1.0
    return stretched @ err


@register_model("context_mixing")
class ContextMixing(LanguageModel):
    """Online context-mixing reference: order-k byte models + logistic mixing.

    Purely transductive: it carries no pretrained parameters (``num_params == 0``)
    and learns entirely inside :meth:`step`. Labelled a **reference (not a
    candidate)** on the leaderboard.
    """

    def __init__(self, config: ContextMixingConfig):
        super().__init__()
        self.config = config
        self.num_predictors = config.max_order + 1
        self._uniform = np.full(config.vocab_size, 1.0 / config.vocab_size)

    # --- amortized interface (trivial: this reference does not pretrain) --------

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward: replay each row as an independent online pass.

        Position 0 has no context (uniform logits); position ``t`` is predicted
        from bytes ``0..t-1`` with within-row counts and online-mixed weights.
        Used only by the amortized path; the prequential metric uses :meth:`step`.
        """
        arr = idx.detach().cpu().numpy()
        batch, length = arr.shape
        out = np.zeros((batch, length, self.config.vocab_size), dtype=np.float32)
        for i in range(batch):
            state = self.init_prequential_state()
            for pos in range(length - 1):
                state, logits, _ = self.step(state, int(arr[i, pos]), pos)
                out[i, pos + 1] = logits.numpy()
        return torch.from_numpy(out).to(idx.device)

    def flops(self, seq_len: int) -> FlopBreakdown:
        """Analytic per-sequence cost = the *steady-state* per-byte estimate
        (:meth:`_steady_step_flops`) replicated ``seq_len`` times. Config-derived
        (not data-dependent), per the harness contract; the measured prequential
        curve uses :meth:`step`'s exact per-byte charge instead."""
        return self._steady_step_flops().scale(seq_len)

    def configure_optimizer(self, *, lr: float, weight_decay: float, betas: tuple[float, float]):
        """No gradient parameters to optimize; return a trivial optimizer that the
        overridden :meth:`train_step` never steps (kept so the harness can call
        it)."""
        return torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)

    def train_step(self, batch, optimizer, *, grad_clip: float = 1.0):
        """No-op pretraining: this reference is transductive, so pretraining cannot
        help (the harness does not hand prior-corpus online state to the eval
        stream — ``docs/harness.md`` §5). Returns the honest forward cost so the
        pretrain budget loop still terminates; runs use ``pretrain_budget = 0``."""
        x, y = batch
        b, t = x.shape
        logits = self(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return loss, self.flops(t).scale(b)

    # --- prequential / online seam (the only honest per-byte channel) ----------

    @property
    def context_window(self) -> int:
        """Bytes conditioned on per prediction = the highest order used."""
        return self.config.max_order

    def init_prequential_state(self) -> DecodeState:
        k = self.num_predictors
        tables: list[dict[bytes, np.ndarray]] = [{} for _ in range(k)]
        weights = np.full(k, 1.0 / k)  # geometric-mean init; abstaining models cancel
        return DecodeState(tokens=[], cache=_MixerState(tables=tables, weights=weights))

    def step(
        self, state: DecodeState, revealed_byte: int, pos: int
    ) -> tuple[DecodeState, torch.Tensor, FlopBreakdown]:
        """Fold ``revealed_byte``, adapt the mixer, predict the next byte.

        Order of operations (every FLOP charged exactly for the branches actually
        executed this byte — see :meth:`_flop_breakdown`):
        1. grade the pending prediction on ``revealed_byte`` -> one SGD step on the
           mixer weights (skipped on the first byte, which had no prediction);
        2. fold ``revealed_byte`` into each *available* order-k count table
           (context = the bytes preceding it);
        3. predict the next byte: Laplace prob (only for active, already-seen
           contexts) -> stretch (log) -> mix -> softmax; return ``z`` as logits.
        """
        cfg = self.config
        ms: _MixerState = state.cache
        v = cfg.vocab_size
        window = state.tokens  # up to max_order bytes preceding `pos`

        # 1. Online mixer update on the just-revealed byte (graded prediction).
        did_update = ms.last_probs is not None
        if did_update:
            grad = mixer_gradient(ms.last_probs, ms.last_stretched, revealed_byte)
            ms.weights -= cfg.lr * grad

        # 2. Fold the revealed byte into each *available* order-k table.
        n_fold = 0
        for k in range(self.num_predictors):
            if k == 0 or len(window) >= k:
                n_fold += 1
                key = bytes(window[-k:]) if k else b""
                cell = ms.tables[k].get(key)
                if cell is None:
                    cell = np.zeros(v)
                    ms.tables[k][key] = cell
                cell[revealed_byte] += 1.0

        # 3. New context window ending at `pos`, then predict the next byte.
        cap = cfg.max_order
        new_window = [*window, revealed_byte][-cap:] if cap else []
        stretched = np.empty((self.num_predictors, v))
        n_active = 0
        n_laplace = 0
        for k in range(self.num_predictors):
            cell = None
            if k == 0 or len(new_window) >= k:
                n_active += 1
                key = bytes(new_window[-k:]) if k else b""
                cell = ms.tables[k].get(key)
            if cell is not None:
                n_laplace += 1
                prob = laplace_prob(cell, cfg.alpha)
            else:
                prob = self._uniform
            stretched[k] = np.log(prob)
        z = mix_logits(stretched, ms.weights)
        probs = softmax(z)

        ms.last_stretched = stretched
        ms.last_probs = probs
        next_logits = torch.from_numpy(z.astype(np.float32))
        new_state = DecodeState(tokens=new_window, cache=ms, length=state.length + 1)
        flops = self._flop_breakdown(
            did_update=did_update, n_fold=n_fold, n_active=n_active, n_laplace=n_laplace
        )
        return new_state, next_logits, flops

    def _flop_breakdown(
        self, *, did_update: bool, n_fold: int, n_active: int, n_laplace: int
    ) -> FlopBreakdown:
        """Exact FLOPs for the branches a :meth:`step` actually executed — every
        non-matmul op charged through ``pointwise``/``gather`` (charge == code, not
        a conservative over-estimate).

        Let ``K = num_predictors`` and ``V = vocab_size``.

        Prediction (``forward``):
          - ``n_active`` context lookups                       -> ``gather(n_active)``
          - Laplace prob (sum+add+divide ≈ 3V), only for the ``n_laplace`` active,
            already-seen contexts                              -> ``3·V·n_laplace``
          - stretch ``log p_k`` for every predictor           -> ``K·V``
          - mix ``z = Σ_k w_k·s_k`` (K·V mul + K·V add)        -> ``2·K·V``
          - softmax (max+sub+exp+sum+div ≈ 5V)                 -> ``5·V``

        Adaptation (``backward`` = the model's own online update):
          - fold: ``n_fold`` count increments + ``n_fold`` lookups
                                                               -> ``n_fold`` + ``gather(n_fold)``
          - mixer update, only when a pending prediction exists (``did_update``):
            one-hot subtract (1) + gradient matvec (2·K·V) + weight step (2·K).
        """
        k = self.num_predictors
        v = self.config.vocab_size
        forward = pointwise_flops(3 * v * n_laplace + k * v + 2 * k * v + 5 * v)
        forward += gather_flops(n_active)
        backward_pointwise = n_fold
        if did_update:
            backward_pointwise += 1 + 2 * k * v + 2 * k
        backward = pointwise_flops(backward_pointwise) + gather_flops(n_fold)
        return FlopBreakdown(forward=forward, backward=backward)

    def _steady_step_flops(self) -> FlopBreakdown:
        """Analytic *steady-state* per-byte cost: all K predictors active and
        already-seen, mixer updating. The config-derived upper bound used by
        :meth:`flops`/:meth:`decode_step_flops`; the measured prequential curve
        uses :meth:`step`'s exact per-byte charge, not this estimate."""
        k = self.num_predictors
        return self._flop_breakdown(did_update=True, n_fold=k, n_active=k, n_laplace=k)

    def decode_step_flops(self, context_len: int) -> FlopBreakdown:
        """Forward-only (prediction) per-byte cost; independent of context length
        (mixing is O(K·V) regardless of how many bytes have been seen). Analytic
        steady-state — see :meth:`_steady_step_flops`."""
        return FlopBreakdown(forward=self._steady_step_flops().forward, backward=0)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ContextMixing":
        """Build from a config dict, ignoring harness-injected transformer keys
        (``d_model``/``n_layers``/``n_heads``/``max_seq_len``) this model does not
        use, so it runs through the existing CLI unchanged."""
        fields = {f.name for f in dataclasses.fields(ContextMixingConfig)}
        kwargs = {k: v for k, v in config.items() if k in fields}
        return cls(ContextMixingConfig(**kwargs))
