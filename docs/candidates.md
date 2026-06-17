# Candidate mechanisms

Running list of things to scout, ranked later by bpb-at-fixed-FLOPs vs. the transformer
baseline. Status: `idea` → `queued` → `running` → `beat-baseline` / `lost` / `parked`.

> Open question being grilled: **which space do we scout first?**
> - **Space A — alternative sequence mixers** (keep backprop + SGD + GPU; swap *attention*).
>   Lower risk, crowded, incremental loss-per-FLOP gains, trivial to test in PyTorch.
> - **Space B — alternative learning rules** (replace *backprop itself*).
>   The post's "beautiful algorithm" romance lives here. High risk, currently *worse*
>   loss-per-FLOP, but the only place a true paradigm shift could come from.

## Space A — sequence-mixing architectures (swap attention, keep backprop)

| Mechanism | One-line idea | Why it might win per-FLOP | Status |
|---|---|---|---|
| Transformer (baseline) | Self-attention + MLP, RoPE, RMSNorm | The bar to beat | idea |
| Selective SSM (Mamba-style) | Input-dependent linear state recurrence | Linear in sequence length; strong long-context per-FLOP | idea |
| Gated Linear Attention (GLA/RetNet) | Linear attention with gating | O(n) attention, parallelizable | idea |
| minRNN (minGRU/minLSTM) | Stripped-down parallelizable RNN | "Were RNNs all we needed?" — cheap, surprisingly strong small-scale | idea |
| Hyena / long convolutions | Implicit long conv via FFT | Subquadratic mixing, no attention | idea |
| gMLP / MLP-mixer token mixing | Static token-mixing MLP + gating | No attention at all; very cheap per step | idea |
| Mixture-of-Experts | Sparse conditional compute | More params per FLOP via routing | idea |

## Space B — learning rules (replace backprop)

| Mechanism | One-line idea | Why it might win per-FLOP | Status |
|---|---|---|---|
| Forward-Forward (Hinton 2022) | Two forward passes, local goodness objective | No backward pass; local, parallel | idea |
| Predictive coding | Local error-driven updates toward equilibrium | Biologically-plausible, local | idea |
| Equilibrium propagation | Energy-based local learning | Single mechanism for inference + learning | idea |
| Feedback alignment / target prop | Replace exact gradients with cheaper signals | Avoids weight transport; cheaper backward | idea |
| Fast-weight programmers | Network writes its own fast weights | Schmidhuber's "learning to learn"; meta-efficiency | idea |
| Evolution strategies / zeroth-order | Gradient-free parameter search | No backward pass; embarrassingly parallel | idea |

## Notes / honest priors

- At *tiny* scale, plain transformers are brutally strong per-FLOP. Beating them is real work.
- Space A wins tend to be *incremental* (and someone has often tried them) — but cheap and reproducible.
- Space B is where the post's romance points; it is also where most ideas currently *lose*
  on loss-per-FLOP. That doesn't make it wrong — it makes it the actual frontier, with
  frontier-level risk.

## Space B under the Source-(iv) filter (LOCKED — see ADR 0003)

Order locked: build & calibrate the harness on Space A, then hunt Space B. A B candidate
qualifies **only** via Source (iv) (more loss-reduction per FLOP). Parallelism/locality is
out of scope.

Parked under (iv) (advantage is mainly (i)/(ii), no clear per-FLOP loss-reduction story):
forward-forward, equilibrium propagation, feedback alignment. Revisit only with a real (iv)
argument.

### Reality check: our metric *is* the Hutter Prize
bpb on `enwik8` is literally the Hutter Prize benchmark. The best results there are **not**
conventionally-trained transformers — they are **online context-mixing** predictors
(PAQ / cmix / nncp lineage): single-pass, online-learned mixtures of many cheap predictors,
reaching ~0.9–1.0 bpb. So "weird ideas that beat transformers on enwik8 bpb" *partly already
exist*, and they're Source-(iv) by nature (single-pass online learning = lots of
loss-reduction per FLOP). This both grounds the hunt and further falsifies the post's
"nobody tried" framing.

### Weird (iv)-qualifying candidates to scout (shortlist)

| Mechanism | Out-of-the-box idea | Source-(iv) story | Amortized? | Status |
|---|---|---|---|---|
| Online context-mixing | Mixture of many cheap predictors, weights learned online in one pass | Single-pass online learning; proven low bpb on enwik8 | transductive | idea |
| Fast-weight associative memory + slow core | O(1) Hebbian/Hopfield store for rote memorization; gradient core for generalization | Memorization made FLOP-cheap frees compute for generalization | amortized | idea |
| Active data / loss selection | Spend FLOPs only on high-information bytes (RHO-loss / online hard-example mining) | Same FLOPs, more loss reduced — pure (iv) | amortized | idea |
| Growing / morphing networks | Start tiny, add capacity only at loss plateaus | No FLOPs wasted on capacity that can't yet be used | amortized | idea |
| Cheap preconditioned updates | Lightweight 2nd-order-ish steps (Sophia/Shampoo-lite) | Fewer steps to a given loss (less "weird", solid (iv)) | amortized | idea |

Open boundary question (being grilled): **amortized vs. transductive** — must a candidate
yield a reusable, promptable model, or is single-pass online prediction allowed? This decides
whether the whole context-mixing literature is in-scope or merely an inspiration.
