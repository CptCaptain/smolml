/* transformer — byte-level pre-norm RMSNorm/RoPE causal transformer, faithful
 * port of smolml/models/transformer.py decode for the prequential step.
 *
 * The Python step runs an incremental KV-cache decode while the context is below
 * max_seq_len (the GROWING regime) and a full windowed recompute once it is full
 * (the SLIDING regime, which drops the cache and resets RoPE positions to
 * 0..win-1). Both regimes equal a full causal forward over the last
 * min(len, max_seq_len) bytes read at the final position — so this port simply
 * does that forward every step and matches Python in BOTH regimes. Weights are
 * loaded from the tiny trained export (float32 base64); compute is float64.
 *
 * API: create({ config, weights }); step(state, byte, pos) -> { probs, state }.
 *   weights = transformer.weights.json (see export_demo_fixtures.py). */
(function (global) {
  "use strict";
  var H = global.SmolDemos;
  var MODELS = (global.SmolModels = global.SmolModels || {});
  var EPS = 1e-6;

  // y[t,o] = Σ_i x[t,i]·W[o,i]   (torch nn.Linear: y = x · Wᵀ, W is (out,in)).
  function linear(x, T, din, W, dout, out) {
    for (var t = 0; t < T; t++) {
      var xb = t * din, ob = t * dout;
      for (var o = 0; o < dout; o++) {
        var wb = o * din, acc = 0.0;
        for (var i = 0; i < din; i++) acc += x[xb + i] * W[wb + i];
        out[ob + o] = acc;
      }
    }
    return out;
  }

  function rmsnorm(x, T, d, weight, out) {
    for (var t = 0; t < T; t++) {
      var b = t * d, ss = 0.0, j;
      for (j = 0; j < d; j++) { var v = x[b + j]; ss += v * v; }
      var nrm = 1.0 / Math.sqrt(ss / d + EPS);
      for (j = 0; j < d; j++) out[b + j] = x[b + j] * nrm * weight[j];
    }
    return out;
  }

  function create(opts) {
    opts = opts || {};
    var w = opts.weights;
    if (!w) throw new Error("transformer.create needs opts.weights (transformer.weights.json)");
    var cfg = w.config;
    var d = cfg.d_model, nh = cfg.n_heads, hd = d / nh, dff = cfg.d_ff;
    // RoPE (cos,sin) for positions 0..max_seq_len-1 — recomputed (not exported).
    var P = cfg.max_seq_len, half = hd / 2;
    var invFreq = new Float64Array(half);
    for (var i = 0; i < half; i++) invFreq[i] = 1.0 / Math.pow(cfg.rope_base, (2 * i) / hd);
    var cos = new Float64Array(P * hd), sin = new Float64Array(P * hd);
    for (var t = 0; t < P; t++) {
      for (var k = 0; k < half; k++) {
        var ang = t * invFreq[k];
        var cv = Math.cos(ang), sv = Math.sin(ang);
        cos[t * hd + k] = cv; cos[t * hd + half + k] = cv;
        sin[t * hd + k] = sv; sin[t * hd + half + k] = sv;
      }
    }
    var blocks = w.blocks.map(function (bl) {
      return {
        norm1: H.decodeF32(bl.norm1), qkv: H.decodeF32(bl.qkv), proj: H.decodeF32(bl.proj),
        norm2: H.decodeF32(bl.norm2), fc1: H.decodeF32(bl.fc1), fc2: H.decodeF32(bl.fc2),
      };
    });
    return {
      cfg: cfg, d: d, nh: nh, hd: hd, half: half, dff: dff, V: cfg.vocab_size,
      emb: H.decodeF32(w.tok_emb), normF: H.decodeF32(w.norm_f), blocks: blocks,
      cos: cos, sin: sin,
      window: [], // rolling byte window, capped at max_seq_len
    };
  }

  function applyRope(vec, base, cos, sin, cb, half, hd) {
    // in place: result[m<half] = v[m]·cos[m] - v[m+half]·sin[m];
    //           result[m>=half] = v[m]·cos[m] + v[m-half]·sin[m].
    for (var m = 0; m < half; m++) {
      var a = vec[base + m], b = vec[base + half + m];
      var cl = cos[cb + m], sl = sin[cb + m];
      vec[base + m] = a * cl - b * sl;
      vec[base + half + m] = b * cos[cb + half + m] + a * sin[cb + half + m];
    }
  }

  function forwardLast(state, tokens) {
    var T = tokens.length, d = state.d, nh = state.nh, hd = state.hd, half = state.half;
    var dff = state.dff, V = state.V, cos = state.cos, sin = state.sin;
    var x = new Float64Array(T * d), t, j;
    for (t = 0; t < T; t++) {
      var er = tokens[t] * d, xb = t * d;
      for (j = 0; j < d; j++) x[xb + j] = state.emb[er + j];
    }
    var nrm = new Float64Array(T * d), qkv = new Float64Array(T * 3 * d);
    var attn = new Float64Array(T * d), proj = new Float64Array(T * d);
    var h2 = new Float64Array(T * d), ff = new Float64Array(T * dff), mlp = new Float64Array(T * d);
    var scale = 1.0 / Math.sqrt(hd), scores = new Float64Array(T);

    for (var L = 0; L < state.blocks.length; L++) {
      var bl = state.blocks[L];
      rmsnorm(x, T, d, bl.norm1, nrm);
      linear(nrm, T, d, bl.qkv, 3 * d, qkv);
      // RoPE on q (cols 0..d) and k (cols d..2d) per head.
      for (t = 0; t < T; t++) {
        var cb = t * hd; // window positions are 0..T-1
        for (var hh = 0; hh < nh; hh++) {
          applyRope(qkv, t * 3 * d + hh * hd, cos, sin, cb, half, hd);
          applyRope(qkv, t * 3 * d + d + hh * hd, cos, sin, cb, half, hd);
        }
      }
      // Causal attention per head -> attn (T,d).
      for (var head = 0; head < nh; head++) {
        var off = head * hd;
        for (var qi = 0; qi < T; qi++) {
          var qbase = qi * 3 * d + off, mx = -Infinity, jj, mm;
          for (jj = 0; jj <= qi; jj++) {
            var kbase = jj * 3 * d + d + off, dot = 0.0;
            for (mm = 0; mm < hd; mm++) dot += qkv[qbase + mm] * qkv[kbase + mm];
            dot *= scale; scores[jj] = dot; if (dot > mx) mx = dot;
          }
          var ssum = 0.0;
          for (jj = 0; jj <= qi; jj++) { var e = Math.exp(scores[jj] - mx); scores[jj] = e; ssum += e; }
          var obase = qi * d + off;
          for (mm = 0; mm < hd; mm++) {
            var acc = 0.0;
            for (jj = 0; jj <= qi; jj++) acc += scores[jj] * qkv[jj * 3 * d + 2 * d + off + mm];
            attn[obase + mm] = acc / ssum;
          }
        }
      }
      linear(attn, T, d, bl.proj, d, proj);
      for (j = 0; j < T * d; j++) x[j] += proj[j];
      rmsnorm(x, T, d, bl.norm2, h2);
      linear(h2, T, d, bl.fc1, dff, ff);
      for (j = 0; j < T * dff; j++) ff[j] = H.gelu(ff[j]);
      linear(ff, T, dff, bl.fc2, d, mlp);
      for (j = 0; j < T * d; j++) x[j] += mlp[j];
    }
    // Final norm + tied head on the LAST position only.
    var last = new Float64Array(d), lb = (T - 1) * d, ss = 0.0;
    for (j = 0; j < d; j++) { var vv = x[lb + j]; ss += vv * vv; }
    var fn = 1.0 / Math.sqrt(ss / d + EPS);
    for (j = 0; j < d; j++) last[j] = x[lb + j] * fn * state.normF[j];
    var logits = new Float64Array(V);
    for (var b = 0; b < V; b++) {
      var eb = b * d, acc2 = 0.0;
      for (j = 0; j < d; j++) acc2 += last[j] * state.emb[eb + j];
      logits[b] = acc2;
    }
    return logits;
  }

  function step(state, revealedByte, pos) {
    var cap = state.cfg.max_seq_len;
    var win = state.window.concat([revealedByte]);
    if (win.length > cap) win = win.slice(win.length - cap);
    state.window = win;
    var logits = forwardLast(state, win);
    return { probs: H.softmax(logits), state: state };
  }

  MODELS.transformer = { create: create, step: step };
})(typeof globalThis !== "undefined" ? globalThis : this);
