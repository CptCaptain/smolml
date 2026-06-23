# Task C.A — interactive model demos for the compendium

- Status: SPEC — design approved (phased: vertical slice first; transformer competes via a small
  trained-weight export). Branch `task/C.A-interactive-demos` off `main`
  (worktree `../smolml.worktrees/C.A-interactive-demos`). Own PR; do not merge.
- Goal: make the compendium's models **runnable in the browser** so a reader can *feel* the
  loss-per-FLOP differences. Two demo families; **faithful** to the measured models (parity-gated).

## Role split (ADR 0006 — read before touching `docs/learning/`)
- **Model-port engineer** (this spec's first deliverable): the **validated JS inference modules** +
  the **parity harness** + the **weight exports**. The "math/weights" deliverable. NO UI work.
- **docs-builder** (follows): the interactive MDX components/UI consuming those modules. Researchers
  do NOT hand-write the components; the engineer does NOT build UI.

## The two demos
1. **Autocomplete / live-bpb race (byte models).** A text field (seeded with a fixed `enwik8`
   snippet, user-editable). Per byte: each model predicts the next-byte distribution, the true byte is
   revealed, the model **adapts online** (the real prequential `step`), accumulating **live cumulative
   bpb** (the success metric) + next-char top-1 hit-rate. 2–3 models race on the SAME stream.
   - The online byte models are **predict-then-learn** (`smolml/prequential.py::prequential_bpb`:
     `step(state, revealed_byte, pos)` folds the byte, adapts, returns the next-byte dist). So they run
     their REAL algorithm live with **NO weight export** — they warm up on the text.
2. **Chemotaxis cursor-follow (control models).** Concentration peak = the cursor's x; the organism
   senses local concentration and chemotaxes to chase it. Toggle models; live FLOP/param HUD +
   cumulative reward. The control `step` seam is `smolml/control_eval.py` /
   `smolml/envs/chemotaxis.py` (tape `c0 a0 c1 a1 …`, EVEN pos = concentration, ODD = action).

## Vertical slice (build + ship FIRST, then extend)
- **Control trio:** `chemotaxis_min` (5 scalars — JS uses config defaults, no export),
  `reservoir_plastic` (frozen core export + readout learns online), `reservoir` (frozen core + a
  short-distilled readout export).
- **Autocomplete race:** `context_mixing` (online count ladder, no export), `delta_mix` (online
  delta-rule, no export), `transformer` (**export a tiny trained model's weights**).
- **Shared infra:** the parity harness + the export tooling + the JS module contract below.
- **Extend phase (after the slice ships):** the remaining byte models (`fast_weight`, `warm_mix`,
  `gated_mix`, `hashed_mix` [demo-sized table, fills live], `pc_refine`) reuse the same component.

## JS inference module contract (engineer)
One module per model under `docs/learning/public/js/models/<name>.js` (classic scripts, file://-safe,
matching `public/js/compendium.js` conventions — no bundler). Each exposes:
- **byte models:** `create(opts) -> state`; `step(state, revealedByte, pos) -> { probs: Float32Array(256), state }`
  — faithful port of the Python `step`: fold the byte, run online adaptation, return the NEXT-byte
  distribution (probabilities, post-softmax). Cumulative bpb = `Σ −log2(probs[nextByte])`.
- **control models:** `create(opts) -> state`; `step(state, token, pos) -> { logits: Float32Array(vocab), state }`
  mirroring the control seam; plus a tiny `concentration(x, peakX, cfg)` field port from `ChemoConfig`.
- `opts` carries config + (where needed) loaded weights. Keep state plain (typed arrays) for speed.

## Parity harness (engineer — the fidelity gate, non-negotiable)
- `smolml/experiments/export_demo_fixtures.py` (or similar): for each slice model, run the REAL Python
  `step` on a FIXED stream/rollout (a ~2–4 KB `enwik8` snippet for byte models; a fixed seeded rollout
  for control) and dump to `docs/learning/public/data/demos/<name>.fixture.json`: the stream, the
  per-position predicted distribution (or at least top-8 + the scored bit value), and the cumulative
  bpb / reward. Also export weights to `docs/learning/public/data/demos/<name>.weights.json`
  (transformer tiny trained; reservoir core+readout; reservoir_plastic seed core+readout).
- A JS parity check (node, runnable; e.g. `docs/learning/scripts/parity.mjs`) loads each fixture, runs
  the JS port on the same stream, and asserts: cumulative **bpb within 1e-3** (byte) / reward within
  1e-3 (control) AND per-step argmax identical. **A port that fails parity does not ship.**
- The tiny transformer for export: train a SMALL config (e.g. d_model 64, 2–3 layers) briefly on a
  seed `enwik8` slice via the existing harness, serialize weights; the JS port mirrors its KV-cache
  decode `step`. Keep the export < ~1 MB.

## Acceptance (engineer deliverable)
- 6 JS inference modules (3 control + 3 byte) under `public/js/models/`, each **parity-validated**
  against Python (`scripts/parity.mjs` green: bpb/reward within 1e-3, argmax identical).
- Weight + fixture JSON under `public/data/demos/` (< ~1 MB each; transformer export trained).
- A short README/handoff note (module API + how to run parity + what each `opts` needs) for the
  docs-builder.
- Do NOT build UI/components. Do NOT run project-wide gates or commit (orchestrator does). You MAY run
  `uv run python …` (fixtures/export) and `node scripts/parity.mjs` to validate.

## Acceptance (docs-builder deliverable, after the engineer)
- Interactive MDX components: the autocomplete-race UI (editable text, per-model top-k bars, live bpb +
  top-1 HUD, same-stream comparison) and the cursor-follow canvas (concentration field, organism,
  FLOP/param/reward HUD), consuming the engineer's modules. Reuse `compendium.js` mount conventions;
  factor shared pieces (rule-of-two); update `PROCESS.md`.
- Cross-link into the relevant concept/experiment pages (in-context-control, context-mixing,
  fast-weight-memory, the C.A pages). `npm run build` green; cross-vendor frontend review.

## Out of scope
- WebGPU (the models are tiny — plain JS/typed-arrays runs at 60fps).
- The extend-phase byte models (separate follow-up once the slice ships).
- Any change to harness/research Python beyond the export/fixture/parity tooling.
