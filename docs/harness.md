# The baseline harness (Task 0.1)

The foundation every candidate plugs into. This document is the **contract**: a
new mechanism is added by reading *this*, not the harness source. It is amortized
(train/val) for now; the prequential/online mode and inference-FLOP accounting are
Task 0.2, built on top of these same interfaces without redesign.

The one metric is **validation bits-per-byte at a fixed training-FLOP budget** on
a tiny byte-level corpus. Lower bpb at equal FLOPs wins; nothing else counts.

```mermaid
graph LR
    DATA[byte corpus] --> TRAIN[train loop<br/>fixed FLOP budget]
    MODEL[registered model] --> TRAIN
    FLOP[FLOP counter] --> TRAIN
    TRAIN --> JSONL[runs/*.jsonl]
    JSONL --> LB[leaderboard<br/>table + plot]
```

## 1. Model interface & registry

`smolml/models/registry.py`. The harness only ever speaks to a model through
`LanguageModel`, so it never needs to know the mechanism behind a name.

```python
class LanguageModel(nn.Module, abc.ABC):
    config: object                                   # the model's own config dataclass

    def forward(self, idx: Tensor) -> Tensor: ...    # (B, T) int64 -> (B, T, 256) float logits
    def flops(self, seq_len: int) -> FlopBreakdown:  # analytic matmul FLOPs for ONE sequence
    @classmethod
    def from_config(cls, config: dict) -> "LanguageModel": ...
    def num_params(self) -> int                      # provided; tied tensors counted once
```

- **Vocabulary is fixed at 256** (raw byte values) — no tokenizer choices. `forward`
  returns **logits** (not probabilities); the loss applies softmax.
- `flops(seq_len)` returns a `FlopBreakdown` (see §2). It is **analytic** — derived
  from the config, not profiled — so it is deterministic and hand-checkable.
- `from_config(dict)` rebuilds the model from the plain config dict that is stored
  verbatim in the run log, so runs are reproducible.

### Registry API

```python
register_model(name)        # class decorator: register a LanguageModel subclass
get_model(name)  -> type    # look up a class
list_models()    -> list[str]
build_model(name, config: dict) -> LanguageModel   # what the harness calls
```

### Adding a candidate (zero harness changes)

```python
from smolml.flops import FlopBreakdown, linear_flops
from smolml.models.registry import LanguageModel, register_model

@register_model("my_mechanism")
class MyMechanism(LanguageModel):
    def __init__(self, config: MyConfig):
        super().__init__()
        self.config = config
        ...  # nn.Module layers

    def forward(self, idx):           # (B,T) -> (B,T,256) logits
        ...

    def flops(self, seq_len):         # compose smolml.flops primitives — see §2
        fwd = ...                     # forward matmul FLOPs for one sequence
        return FlopBreakdown.from_forward(fwd)

    @classmethod
    def from_config(cls, config):
        return cls(MyConfig(**config))
```

Then it runs under the existing train loop, eval, and leaderboard. The **only**
rule is: account for compute through the shared `smolml.flops` primitives so the
referee is identical for every entrant.

## 2. The FLOP counter (the critical correctness surface)

`smolml/flops.py`. A bug here silently invalidates every comparison, so the
accounting is explicit and unit-tested against hand-computed values
(`tests/test_flops.py`).

### Conventions (assumptions, made explicit)

- **MAC = 2 FLOPs** — a multiply-accumulate is 1 multiply + 1 add.
- **Matmul `(m,k)·(k,n) -> (m,n)` costs `2·m·n·k`** — `m·n` outputs, each a length-`k`
  dot product (`k` MACs).
- **Counted:** the matmuls that dominate compute — linear/projection layers
  (`O(tokens·d²)`) and the attention score/value matmuls (`O(tokens²·d)`).
- **Ignored, and why:** elementwise ops (activations, RMSNorm, residual adds,
  softmax normalization, RoPE rotations, dropout) and embedding lookups (gathers,
  no multiply-add). These are `O(tokens·d)` vs. `O(tokens·d²)` for the matmuls —
  asymptotically negligible, and counting them exactly is framework-dependent
  without moving the metric.
- **Backward = 2× forward (matmul FLOPs).** For `Y = A·B` with both operands
  feeding gradients, backprop computes `dA = dY·Bᵀ` and `dB = Aᵀ·dY` — two matmuls
  of the same magnitude as the forward one. So a training step costs
  `forward + 2·forward = 3·forward`. This is exactly the textbook **`C ≈ 6·N·D`**
  rule (2 FLOPs/param/token forward, 4 backward), generalized to also charge the
  attention activation matmuls.

### API

```python
MAC_FLOPS = 2
BACKWARD_MULTIPLIER = 2

matmul_flops(m, n, k)                        -> int   # 2*m*n*k
linear_flops(tokens, in_features, out_features) -> int  # Linear over `tokens` rows
causal_attention_flops(seq_len, d_model)     -> int   # scores + value mixing, one layer

@dataclass(frozen=True)
class FlopBreakdown:
    forward: int
    backward: int
    total            # forward + backward (one training step)
    __add__, scale(factor)
    from_forward(forward)   # charges BACKWARD_MULTIPLIER * forward
```

`forward` and `backward` are kept separate on purpose so Task 0.2 can reuse the
same primitives for **inference** (`.forward` only) and **online adaptation**
(`.total` of the adapted submodule) with no redesign.

### Derivation — transformer baseline `flops(T)`

Let `d = d_model`, `L = n_layers`, `d_ff` = FFN hidden, `V = 256`, sequence length
`T`. (Head count does **not** affect attention FLOPs — see below.) Per layer, per
sequence, forward matmul FLOPs:

| term | shape | FLOPs |
| --- | --- | --- |
| qkv projection | Linear(d → 3d) over T | `2·T·(3d)·d = 6·d²·T` |
| output projection | Linear(d → d) over T | `2·d²·T` |
| FFN (up + down) | Linear(d→d_ff)+Linear(d_ff→d) | `4·d·d_ff·T` |
| attention (scores + value) | causal | `4·d·P`, `P = T(T+1)/2` |

**Attention is head-count independent and causal-aware.** With `h` heads of width
`d_head = d/h`, query `i` attends keys `0..i`, so the number of (query,key) pairs
over a sequence is `P = T(T+1)/2`. Each pair is one length-`d_head` dot product per
head for `Q·Kᵀ` and one for value mixing; summed over `h` heads the `d_head`
factors recombine to `d` — so the count depends only on `d_model`:
`Q·Kᵀ = 2·d·P`, `softmax·V = 2·d·P`, total `4·d·P`.

Whole model, per sequence:

```
forward  = L·(8·d²·T + 4·d·d_ff·T + 4·d·P) + 2·d·V·T   # blocks + LM head
backward = 2 · forward
total    = 3 · forward
```

**Worked tiny example** (`d=8, L=2, d_ff=16, V=256, T=4`; `P = 10`):
per-layer `1536 + 512 + 2048 + 320 = 4416`; blocks `8832`; head `16384`;
`forward = 25216`, `backward = 50432`, `total = 75648`. This exact triple is
asserted in `tests/test_flops.py` and against `Transformer.flops(4)`.

The train loop charges **`flops(seq_len).total · batch_size`** per optimizer step.

## 3. Run logging — the JSONL schema

`smolml/train.py` writes one run to `runs/<run>.jsonl`, one JSON object per line.

- **Line 1 — meta:**
  ```json
  {"type":"meta","run":"...","model":"transformer","config":{...},"params":164288,
   "device":"cpu","seed":0,"flop_budget":5e10,"batch_size":16,"seq_len":64,
   "lr":0.003,"started_at":1750000000.0}
  ```
- **Each later line — step:**
  ```json
  {"type":"step","wallclock":3.2,"step":47,"cumulative_flops":50924523520,
   "train_loss":3.91,"val_bpb":3.80}
  ```
  - `wallclock` — seconds since training started,
  - `step` — optimizer steps taken,
  - `cumulative_flops` — **training** FLOPs (forward+backward) spent so far,
  - `train_loss` — mini-batch loss in **bits/byte** (same unit as `val_bpb`, so the
    two curves are directly comparable),
  - `val_bpb` — validation bits-per-byte at this step.

A step line is written every `eval_interval` steps and always once more at the end.
**The budget is on training FLOPs only**; the validation forward pass is a
measurement and is not charged (amortized protocol — Task 0.2 will count
inference/adaptation FLOPs into a *total*-FLOP budget).

## 4. The leaderboard

`smolml/leaderboard.py` reads every `runs/*.jsonl`, sorts by final `val_bpb`
(lowest first), renders a markdown table, and plots each run's
bpb-vs-training-FLOPs trajectory on a log-x axis, saved as a PNG.

```python
collect_runs(runs_dir) -> list[RunRecord]
build_table(records)   -> str           # markdown
plot_bpb_vs_flops(records, out_png) -> Path
regenerate(runs_dir, table_path=None, plot_path="runs/leaderboard.png") -> (table, png)
```

Regenerate after new runs land — it is reproducible and never hand-edited.

## How to run

Everything via `uv run` (never bare `python`). Device auto-detects **cuda > mps >
cpu**; the metric is FLOP-based so the device only changes wall-clock.

```bash
uv sync                                   # create the env (CPU torch wheel; see pyproject)

# train a run to a fixed FLOP budget (defaults: bundled sample, transformer)
uv run smolml train --data sample --d-model 64 --layers 3 --budget 5e10 \
    --run-name baseline-sample-d64

# the real corpus is opt-in (network); tests never need it
uv run smolml train --data enwik8 --enwik8-bytes 5000000 --budget 1e13

# CI-scale synthetic text8 clone (no network)
uv run smolml train --data synthetic --synthetic-bytes 1000000 --budget 1e11

# regenerate the leaderboard table + plot from all run logs
uv run smolml leaderboard --runs-dir runs --table runs/leaderboard.md --plot runs/leaderboard.png
```

### Data sources (`smolml/data/`)

- `load_sample()` — tiny bundled English sample committed under the package; used
  by tests and the offline smoke run. **No network.**
- `synthetic_text8(n_bytes, seed)` — deterministic, scaled `text8`-style clone
  (lowercase + space) for CI-scale runs. **No network.**
- `prepare_enwik8(cache_dir, n_bytes=None)` — the real corpus; **opt-in** network
  download. Tests never call it.

All sources yield a `ByteCorpus`; `ByteCorpus.split(val_fraction)` is a
deterministic tail split (val is the final fraction; no shuffling, no leakage).

## Gates

```bash
uvx ruff format --check
uvx ruff check
uv run pytest
```
