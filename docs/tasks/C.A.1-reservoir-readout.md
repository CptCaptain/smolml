# Task C.A.1 — reservoir + plastic readout (minimal-organism control candidate)

- Status: SPEC — design approved in brainstorm (forks: ship distilled-frozen `reservoir` first,
  then online `reservoir_plastic` as C.A.1b reusing the same core; reservoir matrices counted in
  `num_params` for memory parity).
- Depends on: the **C.A.0 control rung** (merged to `main`): `smolml/envs/chemotaxis.py`,
  `smolml/control_eval.py::evaluate_control`, `smolml/control_train.py::distill_train_run`,
  `smolml/leaderboard.py::regenerate_control`. **Zero rung changes** — a candidate is just a
  registered `LanguageModel` (`smolml/models/registry.py`) that also implements the per-step seam.
- Branch: `task/C.A.1-reservoir` off `main` (worktree `../smolml.worktrees/C.A.1-reservoir`). Own PR;
  do not merge (human merges). C.A.1b branches off this branch (shares the core).

## Why (ADR 0007 + ADR 0003, the source-(iv) filter)

The bio thesis: a minimal organism with a **fixed** recurrent substrate and a cheap, locally-trained
readout can do in-context control. A transformer recomputes `O(n_layers·d² + context·d)` per decode
step; a reservoir (echo-state network) rolls its state in `O(d_res²)` per step, **independent of
context length**, and only a **linear readout** is trained. The loss-per-FLOP edge is structural:
the expensive recurrent dynamics are **never trained** (0 backward), so the learning cost is just the
readout. A candidate qualifies under ADR 0003 only if its *learning dynamics* extract more
regret-reduction per total FLOP than the transformer bar — that is exactly what this measures.

## The bar to beat (transformer baseline, held-out drifting eval)

regret **0.229 → 0.171 → 0.141** (reward 0.704 → 0.762 → 0.792) at total FLOPs **2.97e11 → 1.19e12 →
2.96e12** (150/600/1500 distillation steps), at **148,608 params**. Headline = **regret vs oracle per
total FLOP at fixed params** (lower-left wins). Beat it = lower regret at ≤ FLOPs and ≤ params.

## The mechanism — `_ReservoirCore` + two registered readouts (`smolml/models/reservoir.py`)

A shared **frozen echo-state core** plus a readout. Per token (input = the token id):

```
x_t   = W_in[:, token]                          # fixed random column = a fixed embedding (a gather)
pre   = x_t + W_res @ h_{t-1}                    # W_res @ h : the dominant matvec, O(d_res²)
h_t   = (1 - leak) * h_{t-1} + leak * tanh(pre)  # leaky-integrator reservoir state
logits = W_out @ h_t + b_out                     # the TRAINED readout, full vocab (levels + 3)
```

- `W_in` (`d_res × vocab`), `W_res` (`d_res × d_res`) are `nn.Parameter(requires_grad=False)`,
  initialized random with `W_res` **spectral-radius-scaled** (e.g. rescale to `rho≈0.9`) for the
  echo-state property; reproducible from `seed`. They are **counted in `num_params`** (memory parity)
  but **excluded from the optimizer** and **charged 0 backward** in `flops`.
- `W_out` (`vocab × d_res`) + `b_out` are the **only trainable params** (~`vocab·d_res`).
- **Size `d_res` so total `num_params` ≤ the bar's 148,608** (so the comparison is fair on the param
  axis); `d_res ≈ 374` lands just under — implementer computes the exact value and reports it.
- The reservoir state `h` (a `d_res` vector) **is** the bounded in-context memory; no token window is
  needed at decode (the candidate overrides `step` and keys off `h`, not `state.tokens`).

### `reservoir` (C.A.1, distilled-frozen readout)

- `forward(idx)`: run the recurrence over `T` (sequential; `h_0 = 0`) to produce per-position states,
  then apply the readout → `(B, T, vocab)`. Compute the recurrence so it builds **no autograd graph
  through the frozen core** (frozen params + constant `h_0` ⇒ `h` does not require grad ⇒ `backward`
  flows only into `W_out`). In-context adaptation at eval = the reservoir **state** echoing the recent
  trajectory; the readout is **frozen** after distillation (no weight change in `step`).
- Trained by `distill_train_run(model="reservoir", …)` exactly like the transformer bar.

### `reservoir_plastic` (C.A.1b, online reward-modulated readout) — follow-on PR off this branch

- Same frozen core; the readout **adapts online in `step`** (≈ zero distillation), all update FLOPs
  charged in the returned breakdown's `backward`. Explore: world-model columns (`conc_slice`) updated
  by an online **delta rule** (supervised by the next observed concentration); action columns
  (`action_slice`) updated by a **reward-modulated Hebbian** rule with a leaky reward baseline (reward
  proxy = the observed concentration level). Report honestly what works; if it cannot beat the random
  floor, that is a documented source-(iv) finding (do NOT stub it — implement a real rule).

## FLOP accounting (honest, via `smolml.flops` — the product)

Per token / per `step`:
- forward = `matmul_flops(1, d_res, d_res)` (W_res·h) + `matmul_flops(1, vocab, d_res)` (readout) +
  `pointwise_flops` for the gather/leak/tanh (~`d_res` elems, a few per elem). Input gather is O(d_res).
- `flops(seq_len)`: forward = `seq_len ×` the above; **backward = `seq_len × matmul_flops(1, vocab,
  d_res)`** (the readout `dW_out` outer product only — the frozen recurrence gets **0** backward; do
  NOT use `FlopBreakdown.from_forward`, which would charge a full 2× backward).
- `reservoir.step`: forward as above, **backward = 0** (frozen readout). O(d_res²)/step,
  context-independent. `reservoir_plastic.step`: add the online-update FLOPs to `backward`.
- `decode_step_flops(context_len)`: forward-only per-step cost (context-independent here).
- `configure_optimizer`: AdamW over **trainable params only** (`p for p in self.parameters() if
  p.requires_grad`). `train_step`: the default backprop step is correct (frozen core ⇒ autograd only
  touches `W_out`); the overridden `flops` makes the *charge* match.

## Files

- `smolml/models/reservoir.py` — `ReservoirConfig`, `_ReservoirCore`, `@register_model("reservoir")`
  `Reservoir`, and (C.A.1b) `@register_model("reservoir_plastic")` `ReservoirPlastic`.
- `smolml/models/__init__.py` — import + register the new model(s).
- `smolml/experiments/reservoir_control.py` — driver mirroring `control_baseline.py`: distill-train
  `reservoir` across a small step sweep, eval held-out, write a `runs/control/*.jsonl` row, print the
  comparison to the bar. (C.A.1b: a sibling driver / sweep with tiny budgets.)
- `tests/test_reservoir.py` — acceptance tests below.
- `docs/harness.md` — a short note under §6 that `reservoir`/`reservoir_plastic` are registered
  control candidates (the seam is unchanged).

## Acceptance criteria

- Gates green: `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest` (paste outputs).
- `tests/test_reservoir.py`:
  - **forward shape & determinism**: `forward((B,T))` → `(B,T,vocab)`; fixed seed ⇒ identical output.
  - **frozen core**: `W_in`/`W_res` have `requires_grad=False`, are in `num_params`, and are NOT in
    `configure_optimizer`'s param groups; after a `train_step` only `W_out`/`b_out` changed.
  - **FLOP analytic match**: `flops(T).forward` equals the hand-derived recurrence+readout sum;
    `flops(T).backward` equals `T · matmul_flops(1, vocab, d_res)` (readout-only) — **not** 2× forward.
  - **step ≡ forward**: summed `step` logits/states match `forward` on the same token sequence
    (last-position equality) to tolerance; summed `step` FLOPs match `T · decode_step_flops`.
  - **param budget**: `num_params ≤ 148_608`.
  - **end-to-end smoke**: `distill_train_run` a few FLOP-budgeted steps → `evaluate_control` on
    held-out envs returns finite **mean reward strictly above the random-policy floor** and
    `second_half_reward > first_half_reward`; a `runs/control` row is written and `regenerate_control`
    ranks it next to the bar.
- (C.A.1b) `reservoir_plastic`: online update is FLOP-counted in `step.backward`; an end-to-end smoke
  beats the random floor (or the inability to is documented as a finding).

## Out of scope

- The chemotaxis-minimal candidate (C.A.2 — its own task/PR).
- Any change to `ChemoEnv`, `evaluate_control`, `distill_train_run`, the leaderboard, or the renderer.
- A within-episode-learner distillation source (C.A.0's documented follow-up) — orthogonal.
