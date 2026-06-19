# Predictive coding & surprise-gated settling

## Intuition

Backprop learns by pushing one global error signal backward through every layer. **Predictive
coding (PC)** does something more local and more dynamic: each layer holds a *guess* about the
layer below, the mismatch between guess and reality is a **prediction error**, and the network
*settles* — iteratively nudging its internal latents to shrink those errors — before it commits
to an answer. "Inference" is not one forward pass; it is a short optimization (a few descent
steps) that the model runs at use-time.

That dynamism is also PC's per-FLOP weakness: it usually settles **uniformly** — every input gets
the same number of iterations — so it pays full settling cost on trivial inputs and tends to lose
on loss-per-FLOP. The idea we test in [B.1](../experiments/B.1-surprise-gated-pc-refinement.md) is
to make settling **surprise-gated**: spend many iterations on the bytes the model finds hard, one
or zero on the easy ones. Same total settling budget, *concentrated where it reduces loss* — a
[Source-(iv)](source-iv-advantage.md) bet about *dynamics*, not architecture.

## The math (logit-correction variant)

A frozen core gives a hidden state `h` and base logits `ℓ_core`. A small PC module adds a latent
`z`, a generative map `W` (it predicts the hidden, `ĥ = Wz`), and a readout `Vmat` (it emits a
*correction* to the logits). Inference minimizes a free energy that trades fitting the hidden
against a latent prior,

$$F(z) = \tfrac{1}{2\sigma_h^2}\lVert h - Wz\rVert^2 + \tfrac{1}{2\sigma_z^2}\lVert z\rVert^2,$$

by `K` gradient (settling) steps

$$z \leftarrow z - \eta\Big[\tfrac{z}{\sigma_z^2} - \tfrac{W^\top(h - Wz)}{\sigma_h^2}\Big].$$

The refined prediction is `ℓ = ℓ_core + Vmatᵀz`. Learning is **local and gradient-free**: after the
true byte is revealed, the readout follows the exact cross-entropy gradient `ΔVmat ∝ −z(p−e)ᵀ`
and the generative weights follow the PC error rule `ΔW ∝ (h−Wz)zᵀ`. No backward pass through the
core.

The **gate** reads a *pre-reveal* surprise proxy (how flat the current prediction is, `1 − max p`)
and maps it to the settling depth `K` — calibrated so the *average* `K` equals the uniform
baseline's, so the win can only come from **allocation**, never from spending more.

## Worked example (why it can — and here doesn't — pay off)

Start `Vmat = 0`, so the correction is zero and the model is exactly the core. As bytes stream
by, the readout learns which latent directions predict which bytes; on a *hard* byte the gate
grants more settling steps, `z` moves further toward explaining `h`, and the learned readout turns
that into a sharper logit — *if* the core's confidence actually varies byte-to-byte.

The catch B.1 measured: that "if" is load-bearing. On order-0 synthetic data a lightly-trained
core is uniformly unsure, the surprise proxy barely moves, and the gate degenerates to the uniform
schedule. The lever only bites once the core is trained enough to be *differentially* confident —
and even then, on this corpus, the PC layer costs more than it returns. The mechanism is sound and
honestly metered; the data has to *have* hard bytes for "spend more on hard bytes" to mean
anything.

## See also

- [Source-(iv) advantage](source-iv-advantage.md) — the only qualifying kind of win.
- [Prequential evaluation](prequential-evaluation.md) — why the gate must use a *pre-reveal* proxy.
- [Loss per FLOP](loss-per-flop-and-scaling-laws.md) — the axis the gate tries to move.
- [Experiment B.1](../experiments/B.1-surprise-gated-pc-refinement.md) — the controlled run.
