# Researcher note — C.A.4 forage local-learning candidates (for the docs-builder)

> ADR 0006: researchers do NOT hand-write `docs/learning/`. This is the written explanation +
> intuition + math + worked example + what-to-visualize for the docs-builder to turn into an MDX
> experiment page (likely `docs/learning/src/pages/experiments/C.A.4-forage-local-learners.mdx`) and a
> refresh of the `in-context-control` concept page. Confirm the page back with a researcher when drafted.

## One-paragraph intuition

The forage rung is a **contextual bandit over cue types**: each episode picks a hidden "good" type `g`
that pays `+1` on EAT while every other type poisons (`−1`), with a fresh `g` and layout each episode.
The whole game is: *figure out which cue type pays, from your own eat-outcomes, then camp a cell of that
type.* A transformer learns this by Algorithm Distillation — copying a within-episode learner into
attention over the tape, at `O(window·d)` per step plus a distillation bill. The C.A.4 question: can a
**tiny local rule** learn the same contingency online, far cheaper per FLOP? Two candidates answer it.

## `forage_min` — the per-type contingency tracker (the headline)

The bandit analogue of `chemotaxis_min`. The in-context memory is a **per-type value vector** `v[K]`
(K=3), reset each episode — NOT a learned weight. Two moving parts:

- **Local credit assignment (a delta rule).** When the agent EATs a cell of type `t` and the next
  observation reveals reward `r`, it nudges that type's value toward the outcome:
  `v[t] ← v[t] + lr·(r − v[t])`. Because EAT does not move, the eaten type equals the currently-sensed
  type — so credit is assigned **exactly and locally in O(1)**, no backprop, no attention over history.
- **A distilled-scalar softmax policy.** Action logits are read off the current type's value:
  `logit(EAT) = g·v[t] + b_eat`, with standing logits for LEFT/RIGHT. Optimistic initial value
  (`v_init>0`) makes unknown types worth eating → it **explores by eating**; one poison flips a type's
  value negative → it then **moves past** that type to search; a positive type → it **camps**. This is
  win-stay-lose-shift, but expressed as a *learned, differentiable, FLOP-counted* policy (≈8 scalar
  params), not a hand-coded if/else.

It emits full-vocab logits every step (a small world-model head over the obs slice predicts the next
`(type, reward)` symbol — its reward component *is* the contingency belief). All compute is pointwise;
the value update is forward compute, so `step` charges `backward = 0`. ~0 distillation: the headline run
trains 0 steps and all learning happens online.

## `forage_reservoir` — the generic-capacity control

The `reservoir_plastic` mechanism (a frozen random echo-state core + an online reward-modulated plastic
readout, ~148k params for memory-parity with the transformer) ported to forage by decoding the eat-reward
from the combined obs (`obs % 3 − 1`). Same online local rule, **no per-type structure**: the reservoir
state is a fading summary of the *whole* trajectory, so per-type credit through one plastic readout is
muddy. It is the contrast that isolates whether *structure* or *capacity* is the lever — this exact shape
already lost on chemotaxis.

## Worked example (one episode, K=3, optimistic `v=[+0.3,+0.3,+0.3]`, lr=0.8, g=8)

1. On a type-1 cell, `v[1]=+0.3` → `logit(EAT)=8·0.3=+2.4` ≫ move logits → **EAT**. Reward `−1` (type 1
   is poison). Next obs reveals it: `v[1] ← 0.3 + 0.8·(−1−0.3) = −0.74`.
2. Back on a type-1 cell, `logit(EAT)=8·(−0.74)=−5.9` → **move RIGHT** to search.
3. On a type-0 cell (still optimistic `+0.3`) → **EAT**. Reward `+1` (type 0 is `g`):
   `v[0] ← 0.3 + 0.8·(1−0.3) = +0.86` → `logit(EAT)=+6.9` → **camps** type-0 cells for `+1`/step.

The agent identified `g` in ~2 eat-outcomes and camped — the within-episode learning curve the metric
rewards (2nd-half reward ≫ 1st-half).

## Results (regret-per-FLOP at fixed memory, held-out eval, H=64, 32 episodes)

| candidate | params | distillation | total FLOPs | regret | within-episode (2nd−1st) |
|---|---|---|---|---|---|
| `forage_min` (headline) | 8 | ~0 (0 steps) | **2.66e5** | **0.047** | +0.16 |
| `forage_min` @ 100 distill steps | 8 | 100 | 8.0e7 | 0.056 | +0.16 |
| `forage_min` @ 400 distill steps | 8 | 400 | 3.2e8 | 0.161 | +0.17 |
| transformer bar (swept winner) | 148,672 | 150 steps | 6.39e11 | 0.113 | — |
| transformer bar | 148,672 | 400 steps | 1.70e12 | 0.132 | — |
| transformer bar | 148,672 | 900 steps | 3.83e12 | 0.135 | — |
| `forage_reservoir` (control) | 148,093 | ~0 (0 steps) | 1.22e9 | 0.456 | +0.31 |
| `forage_reservoir` @ 200 steps | 148,093 | 200 | 2.46e11 | 0.397 | +0.18 |

Reference policies (MC-pinned): oracle ≈ +0.96 (regret 0); `win_stay_lose_shift` ≈ +0.85 (regret ≈ 0.11);
random ≈ −0.11; `always_eat` ≈ −0.33.

**Robustness (the eval-overfit guard).** The inits (`gain=8, lr=0.8`) were chosen on held-out eval at
seed 0 — symmetric with how the transformer bar is the best of a hyperparameter sweep on held-out eval.
Re-evaluating the *fixed* inits on seeds 1–7 (disjoint episodes) gives regret **0.0435 ± 0.009 (max
0.058)** — every seed well below the bar's 0.113, so the headline is not seed-overfit.

**Headline:** `forage_min` reaches regret ≈ 0.047 with 8 params and ~2.7e5 total FLOPs — ~6 orders of
magnitude below the transformer bar's 6.4e11 — and *beats* both `win_stay_lose_shift` and the swept
distilled transformer (0.113) on regret, because it exploits firmly instead of forced ε-exploration.
**Distillation does not help it** (regret rises to 0.16 by 400 steps): the principled inits are already
near-optimal, so spending FLOPs on backprop is wasted — the cleanest possible Source-(iv) result. The
bar's regret is flat across its FLOP curve (0.113→0.135), confirming tuning-not-compute is its lever (an
honest bar). `forage_reservoir` (generic capacity, no per-type structure) lands at regret ≈ 0.40–0.46 at
~1e9–2.5e11 FLOPs: **structure, not capacity, is the per-FLOP lever.**

## Honest framing (state it on the page)

Forage is **reflex-proof** — no *fixed* policy is near-optimal — and `forage_min` is a genuine *online
learner* (it adapts within-episode), so a win is the (iv) thesis, not a strawman. But a tracker so
well-matched to the bandit winning also re-shows (like `chemotaxis_min` on chemotaxis) that this rung's
optimum is cheaply learnable by a local rule. The transformer bar is honest (best of a swept baseline),
so the comparison is fair; the result says local credit assignment extracts the contingency at a tiny
fraction of the distilled transformer's FLOPs.

## What to visualize

1. **The regret-vs-total-FLOP scatter** (log-x): `forage_min` (bottom-left, ~5e5 FLOPs / 0.045 regret),
   `forage_reservoir` (~2.4e9 / 0.49), and the transformer bar curve (~3e11 / ~0.16) — the ~6-OOM gap is
   the whole story. (`runs/forage/leaderboard.png` is the static version.)
2. **A within-episode `v[t]` trace** for one episode: three lines (one per cue type) starting at `v_init`,
   the poison types diving negative on their first eat, `g` climbing toward +1 — overlaid with the
   per-step reward, showing identify-then-camp. (Interactive scrubber would be ideal.)
3. **The forage spacetime raster** (`render_rollout`): ring cells colored by cue type, the agent's path,
   eaten cells, cumulative reward vs oracle — `forage_min` vs `forage_reservoir` side by side makes the
   "clean camp vs muddy wander" contrast visible.
