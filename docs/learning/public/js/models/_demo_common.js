/* smolml interactive-demo model layer — shared helpers.
 *
 * HARD CONSTRAINT (mirrors compendium.js): the site runs from file:// with no
 * server, so this is a plain classic script — no ES modules, no imports, no
 * fetch. It attaches helpers to a global `SmolDemos` namespace; the per-model
 * modules (also classic scripts) read them and attach to `SmolModels`. The same
 * files are loaded by docs/learning/scripts/parity.mjs inside a node `vm`
 * context (where `globalThis` stands in for `window`), so nothing here may touch
 * the DOM or any browser-only global.
 *
 * Everything computes in float64 (JS native). Exported weights are stored as
 * float32 (compact) but the Python fixtures are generated in float64 from those
 * exact float32 values, so JS float64 == Python float64 to ~1e-12 and the parity
 * gate (bpb/reward within 1e-3, argmax identical) holds with wide margin. */
(function (global) {
  "use strict";
  var NS = (global.SmolDemos = global.SmolDemos || {});

  // ── numerics ──────────────────────────────────────────────────────────────
  var LN2 = Math.log(2.0);
  NS.LN2 = LN2;
  NS.log2 = function (x) { return Math.log(x) / LN2; };

  // erf / erfc — Cephes double-precision rational approximation (~1e-15), so JS
  // gelu matches torch's libm gelu far inside the 1e-3 gate (no argmax flips).
  var ERF_T = [9.60497373987051638749e0, 9.00260197203842689217e1,
    2.23200534594684319226e3, 7.00332514112805075473e3, 5.55923013010394962768e4];
  var ERF_U = [3.35617141647503099647e1, 5.21357949780152679795e2,
    4.59432382970980127987e3, 2.26290000613890934246e4, 4.92673942608635921086e4];
  var ERFC_P = [2.46196981473530512524e-10, 5.64189564831068821977e-1,
    7.46321056442269912687e0, 4.86371970985681366614e1, 1.96520832956077098242e2,
    5.26445194995477358631e2, 9.34528527171957607540e2, 1.02755188689515710272e3,
    5.57535335369399327526e2];
  var ERFC_Q = [1.32281951154744992508e1, 8.67072140885989742329e1,
    3.54937778887819891062e2, 9.75708501743205489753e2, 1.82390916687909736289e3,
    2.24633760818710981792e3, 1.65666309194161350182e3, 5.57535340817727675546e2];
  var ERFC_R = [5.64189583547755073984e-1, 1.27536670759978104416e0,
    5.01905042251180477414e0, 6.16021097993053585195e0, 7.40974269950448939160e0,
    2.97886665372100240670e0];
  var ERFC_S = [2.26052863220117276590e0, 9.39603524938001434673e0,
    1.20489539808096656605e1, 1.70814450747565897222e1, 9.60896809063285878198e0,
    3.36907645100081516050e0];

  function polevl(x, c) {
    var r = c[0];
    for (var i = 1; i < c.length; i++) r = r * x + c[i];
    return r;
  }
  function p1evl(x, c) { // polevl with an implied leading coefficient of 1
    var r = x + c[0];
    for (var i = 1; i < c.length; i++) r = r * x + c[i];
    return r;
  }
  function erf(x) {
    if (Math.abs(x) > 1.0) return 1.0 - erfc(x);
    var z = x * x;
    return x * polevl(z, ERF_T) / p1evl(z, ERF_U);
  }
  function erfc(a) {
    var x = a < 0 ? -a : a;
    if (x < 1.0) return 1.0 - erf(a);
    var z = -a * a;
    z = Math.exp(z);
    var p, q;
    if (x < 8.0) { p = polevl(x, ERFC_P); q = p1evl(x, ERFC_Q); }
    else { p = polevl(x, ERFC_R); q = p1evl(x, ERFC_S); }
    var y = (z * p) / q;
    if (a < 0) y = 2.0 - y;
    if (y === 0.0) return a < 0 ? 2.0 : 0.0;
    return y;
  }
  NS.erf = erf;
  NS.erfc = erfc;
  var SQRT1_2 = 0.7071067811865476;
  NS.gelu = function (x) { return 0.5 * x * (1.0 + erf(x * SQRT1_2)); };

  // In-place stable softmax over a Float64Array (or plain array). Returns probs
  // in a fresh Float64Array.
  NS.softmax = function (z) {
    var n = z.length, i, m = -Infinity;
    for (i = 0; i < n; i++) if (z[i] > m) m = z[i];
    var out = new Float64Array(n), s = 0;
    for (i = 0; i < n; i++) { var e = Math.exp(z[i] - m); out[i] = e; s += e; }
    for (i = 0; i < n; i++) out[i] /= s;
    return out;
  };

  // argmax over the whole vector, or over [lo, hi) when bounds are given.
  // Ties resolve to the lowest index (matches numpy/torch argmax).
  NS.argmax = function (v, lo, hi) {
    lo = lo == null ? 0 : lo;
    hi = hi == null ? v.length : hi;
    var best = lo, bv = v[lo];
    for (var i = lo + 1; i < hi; i++) if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
  };

  // ── weight loading ──────────────────────────────────────────────────────────
  // Decode a base64 string of little-endian float32 bytes -> Float64Array. The
  // exporter writes '<f4' little-endian; all deployment targets (x86 node +
  // browsers) are little-endian, so the typed-array view is correct and fast.
  function b64ToBytes(b64) {
    if (typeof Buffer !== "undefined") {
      var buf = Buffer.from(b64, "base64");
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    }
    var bin = global.atob(b64), n = bin.length, out = new Uint8Array(n);
    for (var i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  NS.decodeF32 = function (b64) {
    var bytes = b64ToBytes(b64);
    // Copy into an aligned buffer (Buffer's offset may be non-multiple-of-4).
    var aligned = new Uint8Array(bytes.length);
    aligned.set(bytes);
    var f32 = new Float32Array(aligned.buffer, 0, aligned.length >> 2);
    var out = new Float64Array(f32.length);
    for (var i = 0; i < f32.length; i++) out[i] = f32[i];
    return out;
  };

  // ── chemotaxis environment (faithful port of smolml/envs/chemotaxis.py) ─────
  NS.N_ACTIONS = 3;
  NS.ACTION_DELTAS = [-1, 0, 1]; // LEFT, STAY, RIGHT

  // Ring distance; integer args in the env, fractional `b` allowed for the
  // interactive concentration field (cursor peak between cells).
  function ringdist(a, b, width) {
    var d = Math.abs(a - b) % width;
    return Math.min(d, width - d);
  }
  NS.ringdist = ringdist;

  // Concentration field value at cell/position `x` for a peak at `peakX`
  // (== ChemoEnv._raw with mu=peakX). cfg = { width, sigma }.
  NS.concentration = function (x, peakX, cfg) {
    var d = ringdist(x, peakX, cfg.width);
    return Math.exp(-(d * d) / (2.0 * cfg.sigma * cfg.sigma));
  };

  // Python's round() is round-half-to-even; mirror it so _level is bit-faithful.
  function roundHalfEven(v) {
    var f = Math.floor(v), diff = v - f;
    if (diff < 0.5) return f;
    if (diff > 0.5) return f + 1;
    return f % 2 === 0 ? f : f + 1;
  }
  function quantizeLevel(raw, levels) {
    var lv = roundHalfEven(raw * (levels - 1));
    if (lv < 0) lv = 0;
    if (lv > levels - 1) lv = levels - 1;
    return lv;
  }
  NS.quantizeLevel = quantizeLevel;

  // ChemoEnv with BAKED initial conditions (drift_rate/drift_dir/mu/p captured
  // from the Python rng at fixture time) — fully deterministic, no RNG port.
  function ChemoEnv(cfg, init) {
    this.cfg = cfg;
    this.drift_rate = init.drift_rate;
    this.drift_dir = init.drift_dir;
    this.mu = init.mu;
    this.p = init.p;
    this._phase = init.phase != null ? init.phase : 0.0;
  }
  ChemoEnv.prototype._raw = function (x) {
    var d = ringdist(x, this.mu, this.cfg.width);
    return Math.exp(-(d * d) / (2.0 * this.cfg.sigma * this.cfg.sigma));
  };
  ChemoEnv.prototype.reset = function () { return quantizeLevel(this._raw(this.p), this.cfg.levels); };
  ChemoEnv.prototype.step = function (actionIdx) {
    this.p = ((this.p + NS.ACTION_DELTAS[actionIdx]) % this.cfg.width + this.cfg.width) % this.cfg.width;
    this._phase += this.drift_rate;
    if (this._phase >= 1.0) {
      this.mu = ((this.mu + this.drift_dir) % this.cfg.width + this.cfg.width) % this.cfg.width;
      this._phase -= 1.0;
    }
    var raw = this._raw(this.p);
    return { level: quantizeLevel(raw, this.cfg.levels), raw: raw };
  };
  ChemoEnv.prototype.field = function () {
    var out = [];
    for (var x = 0; x < this.cfg.width; x++) out.push(this._raw(x));
    return out;
  };
  NS.ChemoEnv = ChemoEnv;
})(typeof globalThis !== "undefined" ? globalThis : this);
