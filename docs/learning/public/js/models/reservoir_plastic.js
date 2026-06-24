/* reservoir_plastic — frozen echo-state core + an ONLINE reward-modulated
 * plastic readout, faithful port of
 * smolml/models/reservoir.py::ReservoirPlastic.step.
 *
 * Same frozen core as `reservoir` (W_in, W_res exported numerically). The readout
 * is a working copy (W, b) cloned from the exported seed readout and adapted by
 * two gradient-free LOCAL rules that fire on a CONCENTRATION fold at pos>=2:
 *   - world model (conc columns): softmax delta rule toward the just-revealed
 *     concentration vs the prediction made at the preceding action-fold;
 *   - policy (action columns): reward-modulated Hebbian with a leaky baseline.
 * The update is one step late by construction (reward + world-model target are
 * only observable at the next conc fold). h + (W,b) ARE the in-context memory.
 *
 * Tape parity: EVEN pos = concentration, ODD = action. API:
 *   create({ config, weights }); step(state, token, pos) -> { logits, state };
 *   concentration(x, peakX, cfg). weights = reservoir_plastic.weights.json. */
(function (global) {
  "use strict";
  var H = global.SmolDemos;
  var MODELS = (global.SmolModels = global.SmolModels || {});
  var N_ACTIONS = H.N_ACTIONS;

  function fold(core, h, token, leak) {
    var d = core.dRes, V = core.vocab, Win = core.Win, Wres = core.Wres;
    var out = new Float64Array(d), oneMinus = 1.0 - leak;
    for (var r = 0; r < d; r++) {
      var acc = Win[r * V + token], rowb = r * d;
      for (var cc = 0; cc < d; cc++) acc += Wres[rowb + cc] * h[cc];
      out[r] = oneMinus * h[r] + leak * Math.tanh(acc);
    }
    return out;
  }

  function create(opts) {
    opts = opts || {};
    var w = opts.weights;
    if (!w) throw new Error("reservoir_plastic.create needs opts.weights");
    var cfg = w.config;
    var V = cfg.vocab_size, d = cfg.d_res;
    var core = { dRes: d, vocab: V, Win: H.decodeF32(w.W_in), Wres: H.decodeF32(w.W_res) };
    // Plastic working copies of the seed readout (never mutate the export arrays).
    var W = H.decodeF32(w.W_out).slice(); // (V*d) flat, row v = cols of byte v
    var b = H.decodeF32(w.b_out).slice(); // (V)
    return {
      cfg: cfg, core: core, V: V, d: d, levels: V - N_ACTIONS,
      lrWm: cfg.lr_wm, lrPol: cfg.lr_pol, rewardDecay: cfg.reward_decay,
      W: W, b: b, baseline: 0.0,
      h: new Float64Array(d),
      hConc: null, actionToken: null, concPredLogits: null, hAction: null,
    };
  }

  function readout(W, b, h, d, V) {
    var out = new Float64Array(V);
    for (var v = 0; v < V; v++) {
      var rb = v * d, acc = b[v];
      for (var r = 0; r < d; r++) acc += W[rb + r] * h[r];
      out[v] = acc;
    }
    return out;
  }

  function step(state, token, pos) {
    var d = state.d, V = state.V, lv = state.levels;
    var hNew = fold(state.core, state.h, token, state.cfg.leak);
    var isConc = pos % 2 === 0;
    var ready = state.hConc != null && state.actionToken != null &&
      state.concPredLogits != null && state.hAction != null;

    if (isConc && pos >= 2 && ready) {
      var W = state.W, b = state.b, hAction = state.hAction, hConc = state.hConc;
      // World-model delta rule on the conc columns (rows 0..levels-1).
      var pred = new Float64Array(lv), mx = -Infinity, i;
      for (i = 0; i < lv; i++) { var z = state.concPredLogits[i]; pred[i] = z; if (z > mx) mx = z; }
      var ssum = 0.0;
      for (i = 0; i < lv; i++) { var e = Math.exp(pred[i] - mx); pred[i] = e; ssum += e; }
      for (i = 0; i < lv; i++) pred[i] /= ssum;
      for (i = 0; i < lv; i++) {
        var err = (i === token ? 1.0 : 0.0) - pred[i];
        var scaled = state.lrWm * err, rb = i * d;
        for (var r = 0; r < d; r++) W[rb + r] += scaled * hAction[r];
        b[i] += scaled;
      }
      // Reward-modulated Hebbian policy update on the action columns.
      var rprox = token / (lv - 1), adv = rprox - state.baseline;
      state.baseline = state.baseline + state.rewardDecay * (rprox - state.baseline);
      var a = state.actionToken - lv, rowb = (lv + a) * d, coef = state.lrPol * adv;
      for (var r2 = 0; r2 < d; r2++) W[rowb + r2] += coef * hConc[r2];
      b[lv + a] += coef;
    }

    var logits = readout(state.W, state.b, hNew, d, V);

    if (isConc) {
      state.hConc = hNew; // this conc-fold state is what the next action samples from
    } else {
      state.actionToken = token;
      state.concPredLogits = logits; // its conc slice predicts the next concentration
      state.hAction = hNew;
    }
    state.h = hNew;
    return { logits: logits, state: state };
  }

  MODELS.reservoir_plastic = { create: create, step: step, fold: fold, concentration: H.concentration };
})(typeof globalThis !== "undefined" ? globalThis : this);
