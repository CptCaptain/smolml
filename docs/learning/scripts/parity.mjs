/* parity.mjs — the fidelity gate for the interactive-demo model layer.
 *
 * Loads each JS port (classic scripts, run inside a node vm context that stands
 * in for `window`/`globalThis`), runs it on the SAME stream/rollout the Python
 * exporter scored, and asserts:
 *   - byte models:  cumulative bpb within 1e-3 of Python AND per-step argmax identical;
 *   - control models: cumulative reward within 1e-3 AND per-step argmax identical
 *                     (a deterministic greedy rollout in a ported ChemoEnv seeded
 *                      from the baked initial conditions in the fixture).
 * Exits non-zero on ANY mismatch. Run: `node docs/learning/scripts/parity.mjs`.
 *
 * A port that fails parity does not ship — fix the port, never loosen the gate. */
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, ".."); // docs/learning
const MODELS_DIR = path.join(ROOT, "public/js/models");
const DATA_DIR = path.join(ROOT, "public/data/demos");

const BPB_TOL = 1e-3;      // the gate: cumulative bpb / reward within 1e-3
const REWARD_TOL = 1e-3;
const PERSTEP_BITS_TOL = 1e-2; // diagnostic: localize gross per-position drift

// ── load the classic-script ports into one shared vm context ────────────────
const sandbox = { Buffer, console };
vm.createContext(sandbox);
const SCRIPTS = [
  "_demo_common.js", "context_mixing.js", "delta_mix.js", "transformer.js",
  "chemotaxis_min.js", "reservoir.js", "reservoir_plastic.js",
];
for (const f of SCRIPTS) {
  vm.runInContext(fs.readFileSync(path.join(MODELS_DIR, f), "utf8"), sandbox, { filename: f });
}
const SmolModels = sandbox.SmolModels;
const SmolDemos = sandbox.SmolDemos;

const readJSON = (name) => JSON.parse(fs.readFileSync(path.join(DATA_DIR, name), "utf8"));

const results = [];
function record(name, ok, detail) { results.push({ name, ok, detail }); }

// ── byte parity ──────────────────────────────────────────────────────────────
function parityByte(name) {
  const fx = readJSON(`${name}.fixture.json`);
  const opts = { config: fx.config };
  if (name === "transformer") opts.weights = readJSON("transformer.weights.json");
  const M = SmolModels[name];
  if (!M) return record(name, false, "no JS module");
  let state = M.create(opts);
  const stream = fx.stream, n = stream.length;
  let total = 8.0; // byte 0: uniform prior = -log2(1/256)
  let maxBits = 0, argMis = 0, firstMis = null;
  for (let pos = 0; pos < n - 1; pos++) {
    const r = M.step(state, stream[pos], pos);
    state = r.state;
    const next = stream[pos + 1];
    const p = r.probs[next];
    const bits = -Math.log2(p);
    total += bits;
    const am = SmolDemos.argmax(r.probs);
    if (am !== fx.argmax[pos + 1]) {
      argMis++;
      if (!firstMis) firstMis = { pos: pos + 1, js: am, py: fx.argmax[pos + 1] };
    }
    const db = Math.abs(bits - fx.scored_bits[pos + 1]);
    if (db > maxBits) maxBits = db;
  }
  const bpb = total / n;
  const dbpb = Math.abs(bpb - fx.cumulative_bpb);
  const ok = dbpb < BPB_TOL && argMis === 0 && maxBits < PERSTEP_BITS_TOL;
  record(name, ok, {
    pyBpb: fx.cumulative_bpb, jsBpb: bpb, dBpb: dbpb,
    argMismatches: argMis, firstMis, maxStepBitsDev: maxBits, n,
  });
}

// ── control parity ──────────────────────────────────────────────────────────
function parityControl(name) {
  const fx = readJSON(`${name}.fixture.json`);
  const opts = { config: fx.config };
  if (name === "reservoir" || name === "reservoir_plastic") {
    opts.weights = readJSON(`${name}.weights.json`);
  }
  const M = SmolModels[name];
  if (!M) return record(name, false, "no JS module");
  const env = fx.env, levels = env.levels, H = env.horizon;
  const cfg = { width: env.width, levels: levels, sigma: env.sigma, horizon: H };
  let totalReward = 0, argMis = 0, tapeMis = 0, firstMis = null;
  for (let e = 0; e < fx.episodes.length; e++) {
    const ep = fx.episodes[e];
    const chemo = new SmolDemos.ChemoEnv(cfg, ep.env_init);
    let state = M.create(opts);
    let c = chemo.reset();
    const tape = [c];
    let pos = 0;
    for (let t = 0; t < H; t++) {
      let r = M.step(state, tape[pos], pos); state = r.state;
      const a = SmolDemos.argmax(r.logits, levels, levels + SmolDemos.N_ACTIONS) - levels;
      if (a !== ep.actions[t] && !firstMis) firstMis = { ep: e, t, kind: "action", js: a, py: ep.actions[t] };
      if (a !== ep.actions[t]) argMis++;
      pos++;
      tape.push(levels + a);
      r = M.step(state, tape[pos], pos); state = r.state;
      const concA = SmolDemos.argmax(r.logits, 0, levels);
      if (concA !== ep.conc_argmax[t] && !firstMis) firstMis = { ep: e, t, kind: "conc", js: concA, py: ep.conc_argmax[t] };
      if (concA !== ep.conc_argmax[t]) argMis++;
      pos++;
      const out = chemo.step(a);
      totalReward += out.raw;
      tape.push(out.level);
    }
    for (let i = 0; i < ep.tape.length; i++) if (tape[i] !== ep.tape[i]) tapeMis++;
  }
  const dRew = Math.abs(totalReward - fx.cumulative_reward);
  const ok = dRew < REWARD_TOL && argMis === 0 && tapeMis === 0;
  record(name, ok, {
    pyReward: fx.cumulative_reward, jsReward: totalReward, dReward: dRew,
    argMismatches: argMis, tapeMismatches: tapeMis, firstMis,
  });
}

// ── run ──────────────────────────────────────────────────────────────────────
const BYTE = ["context_mixing", "delta_mix", "transformer"];
const CONTROL = ["chemotaxis_min", "reservoir", "reservoir_plastic"];
for (const m of BYTE) parityByte(m);
for (const m of CONTROL) parityControl(m);

let allOk = true;
console.log("\n  smolml interactive-demo parity gate (bpb/reward within 1e-3, argmax identical)\n");
for (const r of results) {
  allOk = allOk && r.ok;
  const tag = r.ok ? "PASS" : "FAIL";
  console.log(`  [${tag}] ${r.name}`);
  console.log("         " + JSON.stringify(r.detail));
}
console.log(allOk ? "\n  ALL 6 PORTS PARITY-GREEN\n" : "\n  PARITY FAILED\n");
process.exit(allOk ? 0 : 1);
