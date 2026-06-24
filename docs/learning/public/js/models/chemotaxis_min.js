/* chemotaxis_min — minimal-organism run-and-tumble controller, faithful port of
 * smolml/models/chemotaxis_min.py::ChemotaxisMin.step. Five learnable scalars,
 * here taken at their config defaults (no weights exported — the headline run is
 * untrained). The in-context memory is the leaky baseline `b`; no weight change.
 *
 * Tape parity: EVEN pos = sensed concentration level, ODD = action token. Every
 * step emits full-vocab logits [conc(levels) | action(N_ACTIONS)].
 *
 * API: create({ config }); step(state, token, pos) -> { logits: Float64Array(vocab), state };
 *      concentration(x, peakX, cfg) — the env field (cfg = { width, sigma }). */
(function (global) {
  "use strict";
  var H = global.SmolDemos;
  var MODELS = (global.SmolModels = global.SmolModels || {});
  var N_ACTIONS = H.N_ACTIONS, LEFT = 0;

  function create(opts) {
    opts = opts || {};
    var c = opts.config || {};
    var vocab = c.vocab_size != null ? c.vocab_size : 8 + N_ACTIONS;
    var leakInit = c.leak_init != null ? c.leak_init : 0.85;
    var leakLogit = Math.log(leakInit / (1.0 - leakInit));
    var leak = 1.0 / (1.0 + Math.exp(-leakLogit)); // sigmoid(logit) — matches torch
    var baseline = c.baseline_init != null ? c.baseline_init : 0.0;
    return {
      cfg: {
        vocab_size: vocab,
        levels: vocab - N_ACTIONS,
        leak: leak,
        oneMinus: 1.0 - leak,
        g: c.gain_init != null ? c.gain_init : 2.5,
        stay_bias: c.stay_bias_init != null ? c.stay_bias_init : -1.5,
        climb: c.climb_init != null ? c.climb_init : 1.0,
        sharpness: c.sharpness_init != null ? c.sharpness_init : 1.0,
        baseline_init: baseline,
      },
      b: baseline, s: 0.0, c: 0.0, lastAction: 2, // RIGHT
    };
  }

  function emit(cfg, s, cLast, lastAction) {
    var levels = cfg.levels, V = cfg.vocab_size;
    var out = new Float64Array(V);
    // World-model (concentration) head — peak one step up-gradient.
    var sign = s > 0 ? 1 : (s < 0 ? -1 : 0);
    var center = cLast + cfg.climb * sign;
    if (center < 0) center = 0; else if (center > levels - 1) center = levels - 1;
    for (var i = 0; i < levels; i++) { var diff = i - center; out[i] = -cfg.sharpness * diff * diff; }
    // Policy (action) head — run-and-tumble around the current heading.
    var keep = cfg.g * s, reverse = -keep, isLeft = lastAction === LEFT;
    out[levels + 0] = isLeft ? keep : reverse;     // LEFT
    out[levels + 1] = cfg.stay_bias;               // STAY
    out[levels + 2] = isLeft ? reverse : keep;     // RIGHT
    return out;
  }

  function step(state, token, pos) {
    var cfg = state.cfg, levels = cfg.levels;
    if (pos % 2 === 0) { // EVEN — sensed concentration
      var c = token;
      var s = c - state.b;            // surprise BEFORE the baseline update
      state.b = cfg.oneMinus * state.b + cfg.leak * c;
      state.s = s; state.c = c;
    } else {             // ODD — action token
      state.lastAction = token - levels;
    }
    var logits = emit(cfg, state.s, state.c, state.lastAction);
    return { logits: logits, state: state };
  }

  MODELS.chemotaxis_min = {
    create: create,
    step: step,
    concentration: H.concentration,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
