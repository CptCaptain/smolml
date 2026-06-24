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
| Predictive coding | Local error-driven updates; **surprise-gated settling** (B.1) | Spend settling only on hard bytes — more loss-reduction per FLOP | **tested (B.1): lever real, mechanism Pareto-hollow on synthetic; enwik8 control pending** |
| Equilibrium propagation | Energy-based local learning | Single mechanism for inference + learning | idea |
| Feedback alignment / target prop | Replace exact gradients with cheaper signals | Avoids weight transport; cheaper backward | idea |
| Fast-weight programmers / delta-rule memory | Online error-correcting (delta/LMS) write on a distributed key — `delta_mix` (B.4) | Generalizes across contexts exact count tables abstain on; exact gradient = error, no backward pass | **WON (B.4): beats the bar — 1.8485 @ 1.322e12 strictly dominates 2.0157 @ 1.478e12; first Space-B win** |
| Routed sheet of delta columns | C local delta predictors, one routed per byte by a learned gate — `column_mix` (B.5) | Route-conditional *selection* (context×feature interaction) at ~constant FLOPs | **LOST (B.5): Pareto-hollow on enwik8 — best routed 2.4427 loses to the bar 2.4376 AND a matched-capacity single `W` 2.4181; capacity > selection** |
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
| Fast-weight associative memory + slow core | O(1) Hebbian/Hopfield store for rote memorization; gradient core for generalization | Memorization made FLOP-cheap frees compute for generalization | hybrid | **PHASE A — queued** |
| Active data / loss selection | Spend FLOPs only on high-information bytes (RHO-loss / online hard-example mining) | Same FLOPs, more loss reduced — pure (iv) | amortized | idea |
| Growing / morphing networks | Start tiny, add capacity only at loss plateaus | No FLOPs wasted on capacity that can't yet be used | amortized | idea |
| Cheap preconditioned updates | Lightweight 2nd-order-ish steps (Sophia/Shampoo-lite) | Fewer steps to a given loss (less "weird", solid (iv)) | amortized | idea |

Boundary RESOLVED: **hybrid / continual-learning models are the chosen direction.** Amortized
pretraining is allowed; test-time adaptation is allowed *iff its FLOPs are counted*. Pure
transductive compression is inspiration + a per-FLOP ceiling, not the target.

### B.1 result — surprise-gated predictive-coding refinement (`pc_refine`)
First Space-B fusion scouted (config c / variant α): a frozen core + a gradient-free PC module
whose settling depth + online update are **surprise-gated**. Verdict: the (iv) *gating lever* is
real and directional (gated −0.0045 bpb vs uniform at *matched* total FLOPs, allocation-only), but
the mechanism is **Pareto-hollow** on the synthetic clone — it loses to the bare core (+0.034 bpb,
more FLOPs) and is dominated per-FLOP by the context-mixing reference. Next gate before config
(b)/variant β: a **real-enwik8** control where per-byte difficulty actually varies. See
`docs/learning/experiments/B.1-surprise-gated-pc-refinement.md`.

### B.2 result — warmed online mixing (`warm_mix`) + gated escalation (`gated_mix`)
First **genuine per-FLOP win** in the project, on the **real enwik8** corpus. `warm_mix` =
context-mixing + a FLOP-counted prior→eval warm-start: 2.7700 bpb @ 1.03e10 vs the transformer's
5.5453 @ 9.71e11 — strictly dominates per FLOP (~94× cheaper); cold-start == the reference. The bold
A∩C fusion `gated_mix` (gated order escalation) is **Pareto-hollow** against it — the gate's
per-escalation overhead exceeds the savings on already-cheap mixing. Lesson: the reference's loss was
its *transductive handicap*, not its structure; the bar is now `warm_mix`. See
`docs/learning/experiments/B.2-warmed-mixing.md`.

### B.3 result — bounded (hashed) tables → full-corpus mixing (`hashed_mix`)
The engineering unlock that ran the **full enwik8 ADR carve** (5 MB eval / 95 MB prior): fixed-size
hashed count tables let the order-6 win run without the ~58 GB OOM unbounded dicts hit. On the full
5 MB ADR eval, bounded order-6 **survives** — cold 2.2570, ~7 MB-warmed **2.1111** bpb (vs order-3's
2.6224), in fixed ≤5.0 GiB, far below the transformer. Pre-warming pays all the way to the full
95 MB prior — **2.0157 bpb @ 1.48e12** (the table did NOT saturate: 2.11→2.02); the transformer
landed at 5.4770 @ 1.46e14 (~100,000× the FLOPs). See `docs/learning/experiments/B.3-hashed-mix-full-corpus.md`.

### B.4 result — online delta-rule fast-weight memory (`delta_mix`)
The **first non-Pareto-hollow Space-B (learning-rule) candidate.** `delta_mix` = the warmed hashed
ladder + ONE online error-correcting **delta-rule** fast-weight predictor `W∈ℝ^(256×d)` keyed on a
*fixed sparse signed hashed bag of byte n-grams* (orders 3–8), mixed as one extra raw-logit stream.
Sparsity makes the read AND the rank-1 write `O(sV)` (not `O(dV)`) — the feasibility crux; `d` is
RAM/collisions only, never FLOPs. The (iv) edge over the count mixer: a distributed key
**generalizes** across near-miss/unseen contexts the exact order-k tables must abstain on, and the
convex linear-in-fixed-features loss makes the gradient *be* the error (no backward pass). On the
CI matched-FLOP kill-test (real enwik8 4 MB slice, total ≈1.07e10): **delta 2.4181 bpb beats BOTH
`counts_only` 2.4353 AND `counts_more_warm` 2.4327** — i.e. spending the FLOPs on the learning rule
beats spending them on more warm counts (−0.0146 bpb at matched total). Diagnostics: the mixer
learns weight **+0.86** on the delta stream (load-bearing, NOT A.1's dead gate); on 20,051
top-order-**unseen** contexts delta-only scores **3.73 bpb vs the abstaining count's 8.0** (the
generalization mechanism, confirmed at scale). **Full 5 MB-ADR-carve: `delta_o6_warmfull` =
1.8485 bpb @ 1.322e12 FLOPs — strictly dominates the 2.0157 @ 1.478e12 bar (−0.167 bpb AND fewer
FLOPs).** The first candidate to beat the warmed-mixing bar; `delta_mix` is the new bar. See
`docs/learning/experiments/B.4-delta-mix.md`.

### B.5 result — routed sheet of delta columns (`column_mix`)
**A clean Source-(iv) NEGATIVE: routing the delta stream is Pareto-hollow on byte prediction.**
`column_mix` = `delta_mix` with its single delta `W` replaced by **C columns**, one routed per byte
by a cheap per-arm contextual-bandit gate over a fixed hash bucket (`C=1` ≡ `delta_mix`
bit-identically; gate-off ≡ a static hash route). The bet was **route-conditional selection** — a
multiplicative context×feature interaction one linear-in-φ `W` cannot represent — NOT capacity, since
the bar can raise `delta_dim` to `C·d` for free (sparse read is `O(sV)` regardless of `d`). A unit
fixture confirms the lever **exists**: on a synthetic interaction source a routed sheet beats one
column AND one column does **not** catch up when its `delta_dim` is grown to `C·d`. But on the **real
enwik8** CI matched-FLOP kill-test (4 MB slice, total ≈1.07e10, C=4) it does **not pay**: (a) bar
`delta_mix` **2.4376** / (b) learned gate 2.4577 / (c) gate-off 2.4427 / (d) `delta_mix` at
`delta_dim=C·d` (matched capacity, no router) **2.4181**. **KILL:** best routed (2.4427) loses to BOTH
the bar and the matched-capacity control — the same FLOPs on one *bigger shared* `W` beat
*partitioning* the table across routes. Diagnostics: ~25% column load (no collapse) but each column
starved to ~1/C the data, the learned gate **worse** than gate-off (bandit adds noise; 16/4096 buckets
reassigned), no per-column specialization. Pre-registered failure modes (count-ladder redundancy, data
starvation, capacity > selection) all materialized. Cross-vendor (gpt-5.5) FLOP audit: **no
undercharge** — the kill rests on a fair matched-FLOP comparison. `delta_mix` remains the bar; phase-1b
(lateral predictive columns) and the snake rung stay locked (gated on a flat-sheet win). See
`docs/tasks/B.5-column-mix.md`.

LOCKED (ADR 0004) — evaluation protocol: **prequential (one-step-ahead online) bpb vs.
total FLOPs.** The model predicts each byte *before* seeing it (so memorizing the past cannot
leak the future — honest generalization with no held-out split needed), then may adapt on the
revealed byte. Score = cumulative bpb over the evaluation stream at a fixed *total*-FLOP budget
(pretraining + online + prediction). This protocol subsumes amortized (all FLOPs up front,
then frozen), transductive (zero pretraining), and hybrid as points on one spectrum, compared
fairly. Amends ADR 0001's eval protocol (see ADR 0004); requires a follow-up harness task.

Data carve (LOCKED, ADR 0004): enwik8 byte-level; **final 5 MB = fixed prequential eval
stream** (no leakage); first ~95 MB = freely-usable prior corpus capped by the total-FLOP
budget; adaptation during eval allowed (FLOPs counted); CI uses a scaled `text8` clone.
