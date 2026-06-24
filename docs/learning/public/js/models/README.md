# Interactive-demo model layer — engineer → docs-builder handoff

Validated, parity-gated JS inference ports of the smolml compendium models, for
the two interactive demos (autocomplete / live-bpb race; chemotaxis cursor-follow).
**This layer is the math/weights only — no UI.** The docs-builder consumes these
modules (ADR 0006). Everything here is self-contained: the browser never imports
Python; the JS ports read weights/config from the JSON files below.

## Files

- Ports (classic `file://`-safe scripts, no bundler) — `public/js/models/`:
  `_demo_common.js` (shared helpers; **load first**), `context_mixing.js`,
  `delta_mix.js`, `transformer.js`, `chemotaxis_min.js`, `reservoir.js`,
  `reservoir_plastic.js`.
- Weights + fixtures — `public/data/demos/`: `<name>.weights.json` (transformer,
  reservoir, reservoir_plastic) and `<name>.fixture.json` (all six). Read these at
  build time and inline them into the widget's `<script type="application/json">`
  exactly like `ControlRollout.astro` does with `icl_control_rollout.json`.
- Parity gate: `docs/learning/scripts/parity.mjs` — `node docs/learning/scripts/parity.mjs`
  (exits non-zero on any mismatch; ~30 s, dominated by the transformer).
- Regeneration: `uv run python -m smolml.experiments.export_demo_fixtures`.

## Mounting

Load `_demo_common.js`, then any model scripts. They attach to two globals:
`SmolModels.<name>` (the model) and `SmolDemos` (shared helpers: `softmax`,
`argmax(v, lo?, hi?)`, `concentration`, `ChemoEnv`, `N_ACTIONS`, `ACTION_DELTAS`).

```js
var m = SmolModels.context_mixing;
var state = m.create({ config: fixture.config });          // byte: config only
var out = m.step(state, revealedByte, pos);                // { probs:Float64Array(256), state }
// cumulative bpb = sum over revealed bytes of -log2(out.probs[nextByte]); byte 0 = 8 bits.
```

### API per model

All `create(opts) -> state` and `step(state, tokenOrByte, pos) -> { …, state }`.
State is plain typed-array data; mutate-and-return (pass the returned `state` back in).

| model | kind | `create(opts)` needs | `step` returns | weights |
|---|---|---|---|---|
| `context_mixing` | byte | `opts.config` (max_order, alpha, lr, vocab_size) | `{ probs }` | none (warms live) |
| `delta_mix` | byte | `opts.config` (+ delta_dim, delta_eta, delta_orders[], delta_signed) | `{ probs }` | none (warms live) |
| `transformer` | byte | `opts.weights` = `transformer.weights.json` | `{ probs }` | exported |
| `chemotaxis_min` | control | `opts.config` (vocab_size, *_init scalars) | `{ logits }` | none (config defaults) |
| `reservoir` | control | `opts.weights` = `reservoir.weights.json` | `{ logits }` | exported (core + distilled readout) |
| `reservoir_plastic` | control | `opts.weights` = `reservoir_plastic.weights.json` | `{ logits }` | exported (core + seed readout; adapts online) |

- **Byte models** are predict-then-learn: `step` folds the revealed byte, adapts
  online, returns the next-byte **probabilities** (post-softmax, length 256).
- **Control models** return full-vocab **logits** `[conc(levels) | action(N_ACTIONS)]`.
  Tape parity: EVEN `pos` = concentration token, ODD = action token. Sample the
  action greedily as `argmax(logits[levels … levels+3))` and read world-model
  predictions from `logits[0 … levels)`. Each control module also exposes
  `concentration(x, peakX, cfg)` (cfg = `{ width, sigma }`) — the field value at
  cell `x` for a peak at `peakX` (fractional `peakX` ok → cursor between cells).
  Use `new SmolDemos.ChemoEnv({width,levels,sigma,horizon}, { drift_rate, drift_dir, mu, p, phase })`
  for a deterministic rollout (`.reset()`, `.step(actionIdx)->{level,raw}`, `.field()`).

The fixture `config` block is the single source of truth for `create` opts; the
weights JSON carries its own `config` (the reservoir/transformer ports read that).

## HUD numbers (params + steady-state FLOPs at the demo configs)

| model | params | per-step FLOPs (HUD) |
|---|---|---|
| `context_mixing` | 0 | 7 428 fwd / **9 493** total (with online adapt) per byte |
| `delta_mix` | 0 | 12 334 fwd / **18 248** total per byte |
| `transformer` | 82 240 | **11 550 720** per byte steady-state (sliding recompute, ctx 64); 196 608 while the window is still filling (KV decode) |
| `chemotaxis_min` | 5 | **66** per token |
| `reservoir` | 5 515 | **9 995** per token (frozen readout, backward 0) |
| `reservoir_plastic` | 5 515 | 9 995 fwd + **2 886** online update per token |

The transformer's per-byte cost is ~1000× the minimal-organism controllers' — the
loss-per-FLOP contrast the demos exist to make visible.

## Seed stream (autocomplete race)

A fixed 2 048-byte enwik8 snippet at offset 2 000 000 (Wikipedia markup, e.g.
`''[[E=mc²|E&nbsp;=&nbsp;mc&sup2;]]''. <!-- … -->`), shared by all three byte
models. It is in every byte fixture as `stream` (int[]) and `seed_text` (latin-1
string) — seed the editable text field with `seed_text`.

## Parity (the gate)

All six pass `bpb/reward within 1e-3, argmax identical`:
byte dBpb ≤ 1.3e-8 (0 argmax mismatches); control dReward = 0 (0 argmax/tape
mismatches). Weights are stored float32 (compact) but the Python fixtures are
generated in float64 from those exact values, so the float64 JS ports match to
~1e-12. A port that fails parity does not ship — fix the port, never the gate.

## Regenerating the control fixtures

`export_demo_fixtures.py` imports the three control models, which live on their
candidate branches: `reservoir`/`reservoir_plastic` on
`task/C.A.1b-reservoir-plastic`, `chemotaxis_min` on `task/C.A.2-chemotaxis-minimal`.
The script runs as-is on `main` once those PRs land. The three byte models are on
`main` already (no extra dependency).
