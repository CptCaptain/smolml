/* delta_mix — online delta-rule fast-weight memory on the count-mixing backbone,
 * faithful port of smolml/models/delta_mix.py::DeltaMix.step (with the count store
 * inherited from HashedMix/ContextMixing — at the demo config no order is hashed,
 * so the count tables are plain dicts, exactly like context_mixing).
 *
 * The extra stream over context_mixing: a sparse, signed, hashed bag of byte
 * n-grams keys a fast-weight matrix W (V x delta_dim) read as one more raw-logit
 * row in the logistic mix, learned by an error-correcting delta (LMS) rule. W is
 * sparse here (a Map bucket -> Float64Array(V), zero default) — numerically
 * identical to numpy's dense zeros, but light enough for the browser at any
 * delta_dim. No exported weights: it warms up live.
 *
 * The Fibonacci bucket hash and sign hash are 64-bit, so they run in BigInt to
 * match Python's `int.from_bytes(...,"little") * KNUTH & MASK64`.
 *
 * API: create(opts){opts.config: ContextMixing fields + delta_dim, delta_eta,
 *      delta_orders[], delta_signed}; step(state, byte, pos) -> { probs, state }. */
(function (global) {
  "use strict";
  var H = global.SmolDemos;
  var MODELS = (global.SmolModels = global.SmolModels || {});

  var KNUTH = 0x9e3779b97f4a7c15n;
  var KNUTH2 = 0x2545f4914f6cdd1dn;
  var MASK64 = (1n << 64n) - 1n;

  function keyOf(window, k) {
    if (k === 0) return "";
    var s = "", start = window.length - k;
    for (var i = start; i < window.length; i++) s += String.fromCharCode(window[i]);
    return s;
  }

  function ilog2(n) { var b = 0; while ((1 << (b + 1)) <= n) b++; return b; }

  function create(opts) {
    opts = opts || {};
    var c = opts.config || {};
    var cfg = {
      max_order: c.max_order != null ? c.max_order : 3,
      alpha: c.alpha != null ? c.alpha : 0.5,
      lr: c.lr != null ? c.lr : 0.02,
      vocab_size: c.vocab_size != null ? c.vocab_size : 256,
      delta_dim: c.delta_dim != null ? c.delta_dim : (1 << 14),
      delta_eta: c.delta_eta != null ? c.delta_eta : 0.1,
      delta_orders: (c.delta_orders || [3, 4, 5, 6, 7, 8]).slice(),
      delta_signed: c.delta_signed != null ? c.delta_signed : true,
    };
    var K = cfg.max_order + 1;
    var nMix = K + 1; // the delta stream is the (K+1)-th mixer input
    var weights = new Float64Array(nMix);
    for (var i = 0; i < nMix; i++) weights[i] = 1.0 / nMix;
    var tables = [];
    for (var j = 0; j < K; j++) tables.push(new Map());
    var maxOrder = cfg.delta_orders.length ? Math.max.apply(null, cfg.delta_orders) : 0;
    return {
      cfg: cfg,
      kPred: K,
      nMix: nMix,
      deltaBits: ilog2(cfg.delta_dim),
      windowCap: Math.max(cfg.max_order, maxOrder),
      logUniform: -Math.log(cfg.vocab_size),
      tables: tables,
      weights: weights,
      W: new Map(),       // bucket -> Float64Array(V)
      window: [],
      lastStretched: null, // Array(nMix) of Float64Array(V)
      lastProbs: null,     // Float64Array(V)
      lastPhi: null,       // { idxs: [], signs: [] }
      lastPDelta: null,    // Float64Array(V)
    };
  }

  function deltaSlot(state, window, n) {
    var x = 0n, start = window.length - n;
    for (var i = 0; i < n; i++) x += BigInt(window[start + i]) << BigInt(8 * i);
    var idx = Number(((x * KNUTH) & MASK64) >> BigInt(64 - state.deltaBits));
    if (!state.cfg.delta_signed) return { idx: idx, sign: 1.0 };
    var sign = (((x * KNUTH2) & MASK64) >> 63n) ? 1.0 : -1.0;
    return { idx: idx, sign: sign };
  }

  function buildPhi(state, window) {
    var orders = state.cfg.delta_orders, idxs = [], signs = [];
    for (var i = 0; i < orders.length; i++) {
      var n = orders[i];
      if (window.length >= n) {
        var s = deltaSlot(state, window, n);
        idxs.push(s.idx); signs.push(s.sign);
      }
    }
    return { idxs: idxs, signs: signs };
  }

  function step(state, revealedByte, pos) {
    var cfg = state.cfg, V = cfg.vocab_size, K = state.kPred, nMix = state.nMix;
    var window = state.window, tables = state.tables, weights = state.weights, W = state.W;
    var didUpdate = state.lastProbs != null;

    // 1. Online mixer SGD on the pending prediction (over K+1 inputs).
    if (didUpdate) {
      var err = state.lastProbs, lp = state.lastStretched;
      err[revealedByte] -= 1.0;
      for (var kk = 0; kk < nMix; kk++) {
        var sk = lp[kk], g = 0.0;
        for (var b = 0; b < V; b++) g += sk[b] * err[b];
        weights[kk] -= cfg.lr * g;
      }
    }

    // 2. Delta-rule W update on the PREVIOUS key (same pending gate as the mixer).
    if (didUpdate && state.lastPhi && state.lastPhi.idxs.length) {
      var pidx = state.lastPhi.idxs, psign = state.lastPhi.signs;
      var ed = state.lastPDelta; // err = softmax(prev z_delta); err[byte] -= 1
      ed[revealedByte] -= 1.0;
      var scaled = new Float64Array(V);
      for (var b2 = 0; b2 < V; b2++) scaled[b2] = cfg.delta_eta * ed[b2];
      for (var j = 0; j < pidx.length; j++) {
        var col = W.get(pidx[j]);
        if (!col) { col = new Float64Array(V); W.set(pidx[j], col); }
        var sg = psign[j];
        for (var b3 = 0; b3 < V; b3++) col[b3] -= scaled[b3] * sg; // colliding buckets accumulate
      }
    }

    // 3. Fold the revealed byte into each available order-k count table.
    for (var k1 = 0; k1 < K; k1++) {
      if (k1 === 0 || window.length >= k1) {
        var key = keyOf(window, k1);
        var cell = tables[k1].get(key);
        if (!cell) { cell = new Float64Array(V); tables[k1].set(key, cell); }
        cell[revealedByte] += 1.0;
      }
    }

    // 4. New (wide) context window + the count-prediction rows.
    var cap = state.windowCap;
    var newWindow = window.concat([revealedByte]);
    if (cap && newWindow.length > cap) newWindow = newWindow.slice(newWindow.length - cap);
    else if (!cap) newWindow = [];

    var stretched = new Array(nMix);
    for (var k2 = 0; k2 < K; k2++) {
      var cell2 = null;
      if (k2 === 0 || newWindow.length >= k2) cell2 = tables[k2].get(keyOf(newWindow, k2));
      var s = new Float64Array(V);
      if (cell2) {
        var tot = 0.0;
        for (var bt = 0; bt < V; bt++) tot += cell2[bt];
        var denom = tot + cfg.alpha * V;
        for (var bb = 0; bb < V; bb++) s[bb] = Math.log((cell2[bb] + cfg.alpha) / denom);
      } else {
        for (var bu = 0; bu < V; bu++) s[bu] = state.logUniform;
      }
      stretched[k2] = s;
    }

    // 5. Delta stream: sparse key -> z_delta = W·phi (uses the just-updated W).
    var phi = buildPhi(state, newWindow);
    var zDelta = new Float64Array(V);
    for (var j2 = 0; j2 < phi.idxs.length; j2++) {
      var c2 = W.get(phi.idxs[j2]);
      if (c2) { var sg2 = phi.signs[j2]; for (var bd = 0; bd < V; bd++) zDelta[bd] += c2[bd] * sg2; }
    }
    stretched[K] = zDelta;

    // 6. Mix, softmax, stash the pending prediction (mixer + delta).
    var z = new Float64Array(V);
    for (var k3 = 0; k3 < nMix; k3++) {
      var w = weights[k3], sv = stretched[k3];
      for (var bz = 0; bz < V; bz++) z[bz] += w * sv[bz];
    }
    var probs = H.softmax(z);

    state.window = newWindow;
    state.lastStretched = stretched;
    state.lastProbs = probs.slice();
    state.lastPhi = phi;
    state.lastPDelta = H.softmax(zDelta);
    return { probs: probs, state: state };
  }

  MODELS.delta_mix = { create: create, step: step };
})(typeof globalThis !== "undefined" ? globalThis : this);
