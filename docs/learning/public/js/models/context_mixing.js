/* context_mixing — online context-mixing reference (PAQ/cmix lineage), faithful
 * port of smolml/models/context_mixing.py::ContextMixing.step.
 *
 * Purely transductive: no exported weights. Per byte it grades its pending
 * prediction (one SGD step on the logistic-mixer weights), folds the byte into
 * each order-k count table, then predicts the next byte (Laplace -> log-stretch
 * -> logistic mix -> softmax). State is plain JS (Maps of typed-array counts).
 *
 * API:
 *   create(opts) -> state    opts.config = { max_order, alpha, lr, vocab_size }
 *   step(state, byte, pos) -> { probs: Float64Array(vocab), state }
 *   (byte models are predict-then-learn; cumulative bpb = sum -log2 probs[next]). */
(function (global) {
  "use strict";
  var H = global.SmolDemos;
  var MODELS = (global.SmolModels = global.SmolModels || {});

  function keyOf(window, k) {
    if (k === 0) return "";
    var s = "", start = window.length - k;
    for (var i = start; i < window.length; i++) s += String.fromCharCode(window[i]);
    return s;
  }

  function create(opts) {
    opts = opts || {};
    var c = opts.config || {};
    var cfg = {
      max_order: c.max_order != null ? c.max_order : 3,
      alpha: c.alpha != null ? c.alpha : 0.5,
      lr: c.lr != null ? c.lr : 0.02,
      vocab_size: c.vocab_size != null ? c.vocab_size : 256,
    };
    var k = cfg.max_order + 1;
    var weights = new Float64Array(k);
    for (var i = 0; i < k; i++) weights[i] = 1.0 / k;
    var tables = [];
    for (var j = 0; j < k; j++) tables.push(new Map());
    return {
      cfg: cfg,
      kPred: k,
      logUniform: -Math.log(cfg.vocab_size),
      tables: tables,
      weights: weights,
      window: [],
      lastStretched: null, // Array(k) of Float64Array(V)
      lastProbs: null,     // Float64Array(V)
    };
  }

  function step(state, revealedByte, pos) {
    var cfg = state.cfg, V = cfg.vocab_size, K = state.kPred;
    var window = state.window, tables = state.tables, weights = state.weights;

    // 1. Online mixer SGD on the just-revealed byte (graded pending prediction).
    if (state.lastProbs) {
      var err = state.lastProbs; // reuse: err = probs; err[byte] -= 1
      var lp = state.lastStretched;
      err[revealedByte] -= 1.0;
      for (var kk = 0; kk < K; kk++) {
        var sk = lp[kk], g = 0.0;
        for (var b = 0; b < V; b++) g += sk[b] * err[b];
        weights[kk] -= cfg.lr * g;
      }
    }

    // 2. Fold the revealed byte into each available order-k count table.
    for (var k1 = 0; k1 < K; k1++) {
      if (k1 === 0 || window.length >= k1) {
        var key = keyOf(window, k1);
        var cell = tables[k1].get(key);
        if (!cell) { cell = new Float64Array(V); tables[k1].set(key, cell); }
        cell[revealedByte] += 1.0;
      }
    }

    // 3. New context window ending at pos, then predict the next byte.
    var cap = cfg.max_order;
    var newWindow = window.concat([revealedByte]);
    if (cap && newWindow.length > cap) newWindow = newWindow.slice(newWindow.length - cap);
    else if (!cap) newWindow = [];

    var stretched = new Array(K);
    for (var k2 = 0; k2 < K; k2++) {
      var cell2 = null;
      if (k2 === 0 || newWindow.length >= k2) {
        cell2 = tables[k2].get(keyOf(newWindow, k2));
      }
      var s = new Float64Array(V);
      if (cell2) {
        var tot = 0.0;
        for (var b2 = 0; b2 < V; b2++) tot += cell2[b2];
        var denom = tot + cfg.alpha * V;
        for (var b3 = 0; b3 < V; b3++) s[b3] = Math.log((cell2[b3] + cfg.alpha) / denom);
      } else {
        for (var b4 = 0; b4 < V; b4++) s[b4] = state.logUniform;
      }
      stretched[k2] = s;
    }

    // z[b] = Σ_k weights[k]·stretched[k][b]; probs = softmax(z).
    var z = new Float64Array(V);
    for (var k3 = 0; k3 < K; k3++) {
      var w = weights[k3], sv = stretched[k3];
      for (var bb = 0; bb < V; bb++) z[bb] += w * sv[bb];
    }
    var probs = H.softmax(z);

    state.window = newWindow;
    state.lastStretched = stretched;
    state.lastProbs = probs.slice(); // keep a clean copy (mixer mutates its err copy)
    return { probs: probs, state: state };
  }

  MODELS.context_mixing = { create: create, step: step };
})(typeof globalThis !== "undefined" ? globalThis : this);
