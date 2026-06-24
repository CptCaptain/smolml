/* reservoir — frozen echo-state core + distilled-frozen linear readout, faithful
 * port of smolml/models/reservoir.py::Reservoir.step.
 *
 * Per token: x = W_in[:,token]; h = (1-leak)·h + leak·tanh(x + W_res·h);
 *            logits = W_out·h + b_out.  The core (W_in, W_res) is frozen random
 * (seeded in Python, so it MUST be exported numerically — it cannot be
 * regenerated in JS); the readout is distilled then frozen. h is the entire
 * bounded in-context memory; no weight change at decode (frozen readout).
 *
 * API: create({ config, weights }); step(state, token, pos) -> { logits, state };
 *      concentration(x, peakX, cfg). weights = reservoir.weights.json. */
(function (global) {
  "use strict";
  var H = global.SmolDemos;
  var MODELS = (global.SmolModels = global.SmolModels || {});

  // h_new = (1-leak)·h + leak·tanh(W_in[:,token] + W_res·h).
  function fold(core, h, token, leak) {
    var d = core.dRes, V = core.vocab, Win = core.Win, Wres = core.Wres;
    var out = new Float64Array(d), oneMinus = 1.0 - leak;
    for (var r = 0; r < d; r++) {
      var acc = Win[r * V + token]; // fixed-embedding column
      var rowb = r * d;
      for (var cc = 0; cc < d; cc++) acc += Wres[rowb + cc] * h[cc];
      out[r] = oneMinus * h[r] + leak * Math.tanh(acc);
    }
    return out;
  }
  // logits[v] = Σ_r Wout[v,r]·h[r] + bout[v].
  function readout(Wout, bout, h, d, V) {
    var out = new Float64Array(V);
    for (var v = 0; v < V; v++) {
      var rb = v * d, acc = bout[v];
      for (var r = 0; r < d; r++) acc += Wout[rb + r] * h[r];
      out[v] = acc;
    }
    return out;
  }

  function create(opts) {
    opts = opts || {};
    var w = opts.weights;
    if (!w) throw new Error("reservoir.create needs opts.weights (reservoir.weights.json)");
    var cfg = w.config;
    var core = {
      dRes: cfg.d_res, vocab: cfg.vocab_size,
      Win: H.decodeF32(w.W_in), Wres: H.decodeF32(w.W_res),
    };
    return {
      cfg: cfg, core: core,
      Wout: H.decodeF32(w.W_out), bout: H.decodeF32(w.b_out),
      h: new Float64Array(cfg.d_res),
    };
  }

  function step(state, token, pos) {
    var cfg = state.cfg;
    state.h = fold(state.core, state.h, token, cfg.leak);
    var logits = readout(state.Wout, state.bout, state.h, cfg.d_res, cfg.vocab_size);
    return { logits: logits, state: state };
  }

  MODELS.reservoir = { create: create, step: step, fold: fold, concentration: H.concentration };
})(typeof globalThis !== "undefined" ? globalThis : this);
