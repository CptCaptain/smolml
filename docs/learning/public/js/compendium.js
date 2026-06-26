/* smolml compendium — classic interactive widget runtime.
 *
 * HARD CONSTRAINT: the site must work from file:// with no server, so this is a
 * plain classic script (no ES modules, no imports, no fetch). It auto-mounts
 * every interactive widget marked with `data-widget`, reading any build-time
 * data from an inline `<script type="application/json">`. State lives in a
 * closure; interaction re-renders via innerHTML with listeners delegated on the
 * widget root (so they survive re-render). The bpb-vs-FLOP chart keeps keyboard
 * focusable marks and updates its tooltip in place (no re-render) so focus is
 * never lost while tabbing. Mirrors the prior Preact components 1:1. */
(function () {
  "use strict";

  // Page-relative prefix to the site root (set on <html data-root> at build
  // time), so links built here resolve under file:// at any page depth.
  var ROOT = document.documentElement.getAttribute("data-root") || "./";
  function LINK(route) { return ROOT + route + "/index.html"; }

  // ── shared helpers ──────────────────────────────────────────────────────
  function readData(root) {
    var s = root.querySelector('script[type="application/json"]');
    try {
      return s ? JSON.parse(s.textContent) : {};
    } catch (e) {
      console.error("[compendium] bad widget data", e);
      return {};
    }
  }
  var SUP = { "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079", "-": "\u207b" };
  function sup(n) {
    return String(n).split("").map(function (c) { return SUP[c] || c; }).join("");
  }
  function sci(x) {
    if (x === 0) return "0";
    var e = Math.floor(Math.log10(x));
    var m = x / Math.pow(10, e);
    return m.toFixed(2) + "\u00d710" + sup(e);
  }
  function showCh(c) { return c === " " ? "\u2423" : c; }

  // ══ BpbFlopChart ═════════════════════════════════════════════════════════
  var ROLE_COLOR = { reference: "#5ea8e6", fast_weight: "#f0913e", transformer: "#5cc46a", free: "#cf8be0", pc_refine: "#e0738a", warm: "#e0654d", reservoir: "#8f86d6", reservoir_plastic: "#b3a4e8", chemotaxis: "#3aa890", forage_min: "#cf72b3", neutral: "#9a8e76" };
  var VB_W = 780, VB_H = 480, M = { top: 30, right: 26, bottom: 58, left: 66 };
  var PX0 = M.left, PX1 = VB_W - M.right, PY0 = M.top, PY1 = VB_H - M.bottom;

  function mountChart(root) {
    var data = readData(root);
    var series = data.series || [];
    var budgetLine = data.budgetLine, budgetLabel = data.budgetLabel || "equal-FLOP budget";
    var noModelLine = !!data.noModelLine, annotations = data.annotations || [];
    var metric = data.metric || {}, yKey = metric.yKey || "bpb", tipLabel = metric.tipLabel || "bpb", valDec = metric.valDecimals != null ? metric.valDecimals : 4;
    var hidden = {};
    var marks = {}; // "sid|i" -> info
    var sc = null; // current scales

    function visible() { return series.filter(function (s) { return !hidden[s.id]; }); }

    function layout() {
      var vis = visible(), xs = [], ys = [];
      vis.forEach(function (s) { s.points.forEach(function (p) { xs.push(p.flops); ys.push(p[yKey]); }); });
      if (budgetLine) xs.push(budgetLine);
      if (noModelLine) ys.push(8);
      if (!xs.length) { xs = [1e6, 1e10]; ys = [4, 8]; }
      var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
      var lx0 = Math.floor(Math.log10(minX) * 2) / 2 - 0.15;
      var lx1 = Math.ceil(Math.log10(maxX) * 2) / 2 + 0.15;
      var ylo = data.yMin != null ? data.yMin : Math.min.apply(null, ys) - 0.3;
      var yhi = data.yMax != null ? data.yMax : Math.max.apply(null, ys) + 0.3;
      var xScale = function (f) { return PX0 + ((Math.log10(f) - lx0) / (lx1 - lx0)) * (PX1 - PX0); };
      var yScale = function (b) { return PY1 - ((b - ylo) / (yhi - ylo)) * (PY1 - PY0); };
      var xTicks = []; for (var k = Math.ceil(lx0); k <= Math.floor(lx1); k++) xTicks.push(k);
      var step, yDecimals;
      if (metric.yTickStep) { step = metric.yTickStep; yDecimals = metric.yDecimals != null ? metric.yDecimals : 1; }
      else { step = (yhi - ylo) / 6 > 0.75 ? 1 : 0.5; yDecimals = step < 1 ? 1 : 0; }
      var yTicks = []; for (var v = Math.ceil(ylo / step) * step; v <= yhi + 1e-9; v += step) yTicks.push(Math.round(v / step) * step);
      if (yTicks.length === 0) { // tight range: the 0.5/1.0 grid is too coarse — pick a 1-2-5 nice step
        var raw = (yhi - ylo) / 4, p10 = Math.pow(10, Math.floor(Math.log10(raw))), cands = [1, 2, 2.5, 5, 10];
        step = 10 * p10;
        for (var ci = 0; ci < cands.length; ci++) { if (cands[ci] * p10 >= raw) { step = cands[ci] * p10; break; } }
        yDecimals = 0; for (var ss = step; yDecimals < 6 && Math.abs(Math.round(ss) - ss) > 1e-9; ss *= 10) yDecimals++;
        yTicks = []; for (var w = Math.ceil(ylo / step) * step; w <= yhi + 1e-9; w += step) yTicks.push(Math.round(w / step) * step);
      }
      return { lx0: lx0, lx1: lx1, ylo: ylo, yhi: yhi, xScale: xScale, yScale: yScale, xTicks: xTicks, yTicks: yTicks, step: step, yDecimals: yDecimals };
    }

    function svgString() {
      sc = layout(); marks = {};
      var L = sc, s = "";
      s += '<svg viewBox="0 0 ' + VB_W + ' ' + VB_H + '" role="img" aria-label="' + (metric.aria || "Bits-per-byte versus total FLOPs; lower-left is better.") + '" class="chart-svg">';
      L.xTicks.forEach(function (k) { var x = L.xScale(Math.pow(10, k)); s += '<line x1="' + x + '" x2="' + x + '" y1="' + PY0 + '" y2="' + PY1 + '" class="grid"/>'; });
      L.yTicks.forEach(function (v) { var y = L.yScale(v); s += '<line x1="' + PX0 + '" x2="' + PX1 + '" y1="' + y + '" y2="' + y + '" class="grid"/>'; });
      s += '<line x1="' + PX0 + '" x2="' + PX1 + '" y1="' + PY1 + '" y2="' + PY1 + '" class="axis"/>';
      s += '<line x1="' + PX0 + '" x2="' + PX0 + '" y1="' + PY0 + '" y2="' + PY1 + '" class="axis"/>';
      L.xTicks.forEach(function (k) { s += '<text x="' + L.xScale(Math.pow(10, k)) + '" y="' + (PY1 + 20) + '" class="tick" text-anchor="middle">10' + sup(k) + "</text>"; });
      s += '<text x="' + ((PX0 + PX1) / 2) + '" y="' + (VB_H - 10) + '" class="axis-label" text-anchor="middle">total FLOPs (log scale) \u2192</text>';
      L.yTicks.forEach(function (v) { s += '<text x="' + (PX0 - 10) + '" y="' + (L.yScale(v) + 4) + '" class="tick" text-anchor="end">' + v.toFixed(L.yDecimals) + "</text>"; });
      s += '<text class="axis-label" text-anchor="middle" transform="translate(16 ' + ((PY0 + PY1) / 2) + ') rotate(-90)">' + (metric.yLabel || "\u2190 validation bits-per-byte") + '</text>';
      if (noModelLine && L.yhi >= 7.9) {
        s += '<line x1="' + PX0 + '" x2="' + PX1 + '" y1="' + L.yScale(8) + '" y2="' + L.yScale(8) + '" class="ref-line"/>';
        s += '<text x="' + (PX1 - 6) + '" y="' + (L.yScale(8) - 6) + '" class="ref-text" text-anchor="end">8.0 bpb \u2014 uniform \u201cno model\u201d</text>';
      }
      if (budgetLine) {
        var bx = L.xScale(budgetLine);
        s += '<line x1="' + bx + '" x2="' + bx + '" y1="' + PY0 + '" y2="' + PY1 + '" class="budget-line"/>';
        s += '<text x="' + (bx - 7) + '" y="' + (PY0 + 14) + '" class="budget-text" text-anchor="end">' + budgetLabel + "</text>";
      }
      visible().forEach(function (sr) {
        var color = ROLE_COLOR[sr.role] || ROLE_COLOR.neutral;
        var pts = sr.points.slice().sort(function (a, b) { return a.flops - b.flops; });
        if (sr.kind === "curve" && pts.length > 1) {
          var path = pts.map(function (p) { return L.xScale(p.flops) + "," + L.yScale(p[yKey]); }).join(" ");
          s += '<polyline points="' + path + '" fill="none" stroke="' + color + '" stroke-width="2"' + (sr.dashed ? ' stroke-dasharray="7 5"' : "") + ' stroke-linejoin="round"/>';
        }
        pts.forEach(function (p, i) {
          var px = L.xScale(p.flops), py = L.yScale(p[yKey]), r = sr.kind === "point" ? 8 : 5;
          var key = sr.id + "|" + i;
          marks[key] = { px: px, py: py, role: sr.role, label: sr.label, val: p[yKey], flops: p.flops, tag: p.tag };
          var aria = sr.label + ": " + p[yKey].toFixed(valDec) + " " + tipLabel + " at " + p.flops.toExponential(2) + " FLOPs" + (p.tag ? " \u2014 " + p.tag : "");
          s += '<g class="chart-mark" tabindex="0" role="button" data-key="' + key + '" aria-label="' + aria.replace(/"/g, "&quot;") + '">';
          s += '<circle cx="' + px + '" cy="' + py + '" r="16" fill="transparent"/>';
          if (sr.kind === "point") {
            s += '<rect x="' + (px - r) + '" y="' + (py - r) + '" width="' + (r * 2) + '" height="' + (r * 2) + '" transform="rotate(45 ' + px + " " + py + ')" fill="' + color + '" class="mk"/>';
          } else {
            s += '<circle cx="' + px + '" cy="' + py + '" r="' + r + '" fill="' + color + '" stroke="#16130e" stroke-width="1.5" class="mk"/>';
          }
          s += "</g>";
        });
      });
      annotations.forEach(function (a) {
        var ax = sc.xScale(a.flops), ay = sc.yScale(a[yKey] != null ? a[yKey] : a.bpb), dx = a.dx == null ? 14 : a.dx, dy = a.dy == null ? -16 : a.dy;
        var tx = ax + dx, ty = ay + dy;
        s += '<line x1="' + ax + '" y1="' + ay + '" x2="' + tx + '" y2="' + ty + '" class="anno-leader"/>';
        s += '<text x="' + tx + '" y="' + ty + '" class="anno-text" text-anchor="' + (dx < 0 ? "end" : "start") + '">' + a.text + "</text>";
      });
      s += '<g class="chart-tip"></g>';
      s += "</svg>";
      return s;
    }

    function legendString() {
      var s = '<div class="legend">';
      series.forEach(function (sr) {
        var off = !!hidden[sr.id];
        s += '<button type="button" class="legend-item' + (off ? " off" : "") + '" data-toggle="' + sr.id + '" aria-pressed="' + (!off) + '">';
        s += '<span class="legend-swatch" style="background:' + (ROLE_COLOR[sr.role] || ROLE_COLOR.neutral) + '"></span>' + sr.label;
        if (sr.reconstructed) s += '<span class="legend-recon"> (reconstructed x)</span>';
        s += "</button>";
      });
      return s + "</div>";
    }

    function render() { root.innerHTML = svgString() + legendString(); }

    function showTip(key) {
      var info = marks[key]; if (!info) return;
      var tip = root.querySelector(".chart-tip"); if (!tip) return;
      var lines = [info.label, tipLabel + " " + info.val.toFixed(valDec), info.flops.toExponential(2) + " FLOPs"];
      if (info.tag) lines.push(info.tag);
      var w = 168, h = 16 + lines.length * 15, px = info.px, py = info.py;
      var bx = px + 14; if (bx + w > VB_W) bx = px - w - 14;
      var by = py - h - 10; if (by < PY0) by = py + 12;
      var t = '<rect x="' + bx + '" y="' + by + '" width="' + w + '" height="' + h + '" rx="6" class="tip-box"/>';
      t += '<rect x="' + bx + '" y="' + by + '" width="4" height="' + h + '" rx="2" fill="' + (ROLE_COLOR[info.role] || ROLE_COLOR.neutral) + '"/>';
      lines.forEach(function (ln, i) { t += '<text x="' + (bx + 12) + '" y="' + (by + 17 + i * 15) + '" class="' + (i === 0 ? "tip-title" : "tip-line") + '">' + ln + "</text>"; });
      tip.innerHTML = t;
      var g = root.querySelector('.chart-mark[data-key="' + key + '"]');
      if (g) g.classList.add("hot");
    }
    function clearTip() {
      var tip = root.querySelector(".chart-tip"); if (tip) tip.innerHTML = "";
      var hot = root.querySelector(".chart-mark.hot"); if (hot) hot.classList.remove("hot");
    }

    root.addEventListener("click", function (e) {
      var b = e.target.closest("[data-toggle]"); if (!b) return;
      var id = b.getAttribute("data-toggle");
      if (hidden[id]) delete hidden[id]; else hidden[id] = 1;
      render();
    });
    function onEnter(e) { var g = e.target.closest(".chart-mark"); if (g) showTip(g.getAttribute("data-key")); }
    function onLeave(e) { if (e.target.closest(".chart-mark")) clearTip(); }
    root.addEventListener("mouseover", onEnter);
    root.addEventListener("mouseout", onLeave);
    root.addEventListener("focusin", onEnter);
    root.addEventListener("focusout", onLeave);

    render();
  }

  // ══ Stream demos (prequential + context mixing) on a shared scaffold ══════
  function softmaxBits(p) { return -Math.log2(p); }

  // -- prequential order-0/1 model --
  var PQ_STREAM = "the cat sat on the mat. the cat ate the rat. ";
  var PQ_ALPHA = uniqSorted(PQ_STREAM);
  function uniqSorted(s) { var set = {}; s.split("").forEach(function (c) { set[c] = 1; }); return Object.keys(set).sort(); }
  function pqFresh() { var c0 = {}; PQ_ALPHA.forEach(function (c) { c0[c] = 0; }); return { pos: 0, c0: c0, c1: {}, cumBits: 0, history: [] }; }
  function pqPredict(st) {
    var A = PQ_ALPHA.length, prev = st.pos > 0 ? PQ_STREAM[st.pos - 1] : null;
    var tot0 = 0; PQ_ALPHA.forEach(function (c) { tot0 += st.c0[c]; });
    var row = prev != null ? st.c1[prev] : null, tot1 = 0;
    if (row) Object.keys(row).forEach(function (k) { tot1 += row[k]; });
    return PQ_ALPHA.map(function (ch) {
      var p0 = (st.c0[ch] + 1) / (tot0 + A);
      var p1 = row ? (((row[ch] || 0) + 1) / (tot1 + A)) : 1 / A;
      return { ch: ch, p: 0.5 * p0 + 0.5 * p1 };
    }).sort(function (a, b) { return b.p - a.p; });
  }
  function pqAdvance(st) {
    var truth = PQ_STREAM[st.pos], dist = pqPredict(st);
    var pTrue = dist.filter(function (d) { return d.ch === truth; })[0].p;
    var bits = softmaxBits(pTrue), prev = st.pos > 0 ? PQ_STREAM[st.pos - 1] : null;
    var c0 = Object.assign({}, st.c0); c0[truth] += 1;
    var c1 = Object.assign({}, st.c1);
    if (prev != null) { var r = Object.assign({}, c1[prev] || {}); r[truth] = (r[truth] || 0) + 1; c1[prev] = r; }
    return { pos: st.pos + 1, c0: c0, c1: c1, cumBits: st.cumBits + bits, history: st.history.concat([bits]) };
  }
  function pqPanel(st) {
    var done = st.pos >= PQ_STREAM.length;
    var dist = done ? [] : pqPredict(st), truth = done ? null : PQ_STREAM[st.pos];
    var top = dist.slice(0, 6), maxP = top.length ? top[0].p : 1;
    var s = '<p class="ps-h">' + (done ? "stream complete" : "model\u2019s p(next byte) \u2014 before reveal") + "</p>";
    top.forEach(function (d) {
      var isT = d.ch === truth;
      s += '<div class="ps-bar' + (isT ? " istrue" : "") + '"><span class="ps-bar-ch">' + showCh(d.ch) + '</span><span class="ps-bar-track"><span class="ps-bar-fill" style="width:' + (d.p / maxP * 100) + '%"></span></span><span class="ps-bar-p">' + (d.p * 100).toFixed(1) + "%</span></div>";
    });
    if (!done) s += '<p class="ps-pending">truth is <code>' + showCh(truth) + "</code> \u2192 pays <strong>" + softmaxBits(dist.filter(function (d) { return d.ch === truth; })[0].p).toFixed(2) + "</strong> bits</p>";
    return s;
  }

  // -- context mixing order-0/1/2 with online logistic weights --
  var CM_STREAM = "mississippi river, mississippi river. ";
  var CM_ALPHA = uniqSorted(CM_STREAM), CM_V = CM_ALPHA.length, CM_ORDERS = [0, 1, 2], CM_K = 3, CM_LR = 0.03, LN2 = Math.log(2);
  function cmFresh() { return { pos: 0, counts: CM_ORDERS.map(function () { return {}; }), w: CM_ORDERS.map(function () { return 1 / CM_K; }), cumBits: 0, history: [] }; }
  function cmSpecialists(st) {
    return CM_ORDERS.map(function (k, ki) {
      var ctx = st.pos < k ? null : CM_STREAM.slice(st.pos - k, st.pos);
      var row = ctx != null ? st.counts[ki][ctx] : null, tot = 0;
      if (row) Object.keys(row).forEach(function (key) { tot += row[key]; });
      var seen = tot > 0;
      var p = CM_ALPHA.map(function (ch) { return seen ? (((row[ch] || 0) + 1) / (tot + CM_V)) : 1 / CM_V; });
      var ss = p.map(function (x) { return Math.log(x); });
      var bi = 0; for (var j = 1; j < CM_V; j++) if (p[j] > p[bi]) bi = j;
      return { order: k, seen: seen, s: ss, top: { ch: CM_ALPHA[bi], p: p[bi] } };
    });
  }
  function cmMixed(specs, w) {
    var z = CM_ALPHA.map(function (_, b) { var a = 0; specs.forEach(function (sp, ki) { a += w[ki] * sp.s[b]; }); return a; });
    var mx = Math.max.apply(null, z), ex = z.map(function (v) { return Math.exp(v - mx); });
    var sum = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (e) { return e / sum; });
  }
  function cmAdvance(st) {
    var truth = CM_STREAM[st.pos], yi = CM_ALPHA.indexOf(truth), specs = cmSpecialists(st), P = cmMixed(specs, st.w);
    var bits = -Math.log(P[yi]) / LN2;
    var w = st.w.map(function (wk, ki) { var g = 0; for (var b = 0; b < CM_V; b++) g += (P[b] - (b === yi ? 1 : 0)) * specs[ki].s[b]; return Math.max(0, Math.min(3, wk - CM_LR * g)); });
    var counts = st.counts.map(function (tbl, ki) {
      var ord = CM_ORDERS[ki], ctx = st.pos < ord ? null : CM_STREAM.slice(st.pos - ord, st.pos);
      if (ctx == null) return tbl;
      var next = Object.assign({}, tbl), row = Object.assign({}, next[ctx] || {}); row[truth] = (row[truth] || 0) + 1; next[ctx] = row; return next;
    });
    return { pos: st.pos + 1, counts: counts, w: w, cumBits: st.cumBits + bits, history: st.history.concat([bits]) };
  }
  function cmPanel(st) {
    var done = st.pos >= CM_STREAM.length;
    var specs = done ? [] : cmSpecialists(st), truth = done ? null : CM_STREAM[st.pos];
    var P = done ? [] : cmMixed(specs, st.w), yi = truth ? CM_ALPHA.indexOf(truth) : -1;
    var maxW = Math.max.apply(null, [0.01].concat(st.w));
    var s = '<p class="cm-h">' + (done ? "stream complete" : CM_K + " specialists \u2192 online logistic mix") + "</p>";
    if (!done) {
      s += '<div class="cm-specs">';
      specs.forEach(function (sp, ki) {
        s += '<div class="cm-spec' + (sp.seen ? "" : " abstain") + '"><div class="cm-spec-top"><span class="cm-order">order-' + sp.order + '</span><span class="cm-pred">' + (sp.seen ? "\u2192 \u2018" + showCh(sp.top.ch) + "\u2019 " + (sp.top.p * 100).toFixed(0) + "%" : "abstains") + "</span></div>";
        s += '<div class="cm-wrow"><span class="cm-wlabel">w</span><span class="cm-wtrack"><span class="cm-wfill" style="width:' + (st.w[ki] / maxW * 100) + '%"></span></span><span class="cm-wval">' + st.w[ki].toFixed(2) + "</span></div></div>";
      });
      s += "</div>";
    }
    if (!done) s += '<p class="cm-h">mixed P(next byte)</p>';
    var ranked = P.map(function (p, i) { return { ch: CM_ALPHA[i], p: p, i: i }; }).sort(function (a, b) { return b.p - a.p; }).slice(0, 5);
    var maxP = ranked.length ? ranked[0].p : 1;
    ranked.forEach(function (r) {
      s += '<div class="cm-bar' + (r.i === yi ? " istrue" : "") + '"><span class="cm-bar-ch">' + showCh(r.ch) + '</span><span class="cm-bar-track"><span class="cm-bar-fill" style="width:' + (r.p / maxP * 100) + '%"></span></span><span class="cm-bar-p">' + (r.p * 100).toFixed(1) + "%</span></div>";
    });
    if (!done) s += '<p class="cm-pending">truth is <code>' + showCh(truth) + "</code> \u2192 pays <strong>" + (-Math.log(P[yi]) / LN2).toFixed(2) + "</strong> bits</p>";
    return s;
  }

  var STREAM_CFG = {
    prequential: { STREAM: PQ_STREAM, fresh: pqFresh, advance: pqAdvance, panel: pqPanel, bar: "#5ea8e6", caption: 'Every prediction is made <em>before</em> the byte is revealed, so a model that memorizes the past cannot cheat the future \u2014 an honest generalization measure with no held-out split. The cumulative bits equal the compressed length, tying straight back to <a href="' + LINK("concepts/compression-equals-prediction") + '">compression = prediction</a>.' },
    contextmixing: { STREAM: CM_STREAM, fresh: cmFresh, advance: cmAdvance, panel: cmPanel, bar: "#5ea8e6", caption: 'Each specialist is a smoothed order-k frequency table; the mixer is one-layer logistic regression on their stretched (log-prob) outputs, learned online by SGD. Watch the weights shift toward the higher orders once the repetition appears \u2014 that is the entire learning algorithm of the <a href="' + LINK("concepts/context-mixing") + '">context-mixing reference</a>.' },
  };

  function streamShell(cfg, st, playing) {
    var STREAM = cfg.STREAM, done = st.pos >= STREAM.length, bytes = st.pos, bpb = bytes > 0 ? st.cumBits / bytes : 0;
    var tape = "";
    for (var i = 0; i < STREAM.length; i++) {
      var cls = i < st.pos ? "strm-ch seen" : i === st.pos ? "strm-ch cursor" : "strm-ch future";
      tape += '<span class="' + cls + '">' + (i === st.pos && !done ? "\u25af" : showCh(STREAM[i])) + "</span>";
    }
    var spark = '<svg viewBox="0 0 220 56" class="strm-spark" role="img" aria-label="bits paid per byte"><line x1="0" y1="55" x2="220" y2="55" stroke="#3a3326"/>';
    st.history.forEach(function (b, i) { var w = 220 / STREAM.length, h = Math.min(b, 9) / 9 * 50; spark += '<rect x="' + (i * w) + '" y="' + (55 - h) + '" width="' + Math.max(1, w - 0.6) + '" height="' + h + '" fill="' + cfg.bar + '"/>'; });
    spark += "</svg>";
    var s = '<figure class="strm">';
    s += '<div class="strm-tape" aria-label="prediction stream">' + tape + "</div>";
    s += '<div class="strm-main"><div class="strm-panel">' + cfg.panel(st) + "</div>";
    s += '<div class="strm-readout"><div class="strm-metric"><span class="strm-metric-v">' + bpb.toFixed(3) + '</span><span class="strm-metric-l">running bpb</span></div>';
    s += '<div class="strm-sub"><span>' + st.cumBits.toFixed(1) + " bits</span><span>\u00f7 " + bytes + " bytes</span></div>";
    s += spark + '<p class="strm-spark-cap">bits paid per byte (it drops as the model learns)</p></div></div>';
    s += '<div class="strm-controls"><button type="button" data-act="step"' + (done ? " disabled" : "") + '>Step</button>';
    s += '<button type="button" class="primary" data-act="play"' + (done ? " disabled" : "") + ">" + (playing ? "Pause" : "Play") + "</button>";
    s += '<button type="button" data-act="reset">Reset</button></div>';
    s += '<figcaption class="figcaption">' + cfg.caption + "</figcaption></figure>";
    return s;
  }

  function mountStream(root, kind) {
    var cfg = STREAM_CFG[kind], st = cfg.fresh(), playing = false, timer = null;
    function render() { root.innerHTML = streamShell(cfg, st, playing); }
    function done() { return st.pos >= cfg.STREAM.length; }
    function step() { if (!done()) { st = cfg.advance(st); render(); } }
    function loop() {
      clearTimeout(timer);
      if (playing && !done()) timer = setTimeout(function () { if (playing) { step(); loop(); } }, 520);
      else if (done()) { playing = false; render(); }
    }
    root.addEventListener("click", function (e) {
      var b = e.target.closest("[data-act]"); if (!b) return;
      var act = b.getAttribute("data-act");
      if (act === "step") { if (done()) { playing = false; render(); } else step(); }
      else if (act === "play") { playing = !playing; render(); loop(); }
      else if (act === "reset") { playing = false; clearTimeout(timer); st = cfg.fresh(); render(); }
    });
    render();
  }

  // ══ FastWeight associative memory ════════════════════════════════════════
  var FW_D = 6, FW_BYTES = ["e", "a", "t", "o", "n"], FW_V = 5;
  function fwNorm(v) { var n = Math.sqrt(v.reduce(function (a, x) { return a + x * x; }, 0)) || 1; return v.map(function (x) { return x / n; }); }
  var FW_KEYS = [
    { label: "ctx \u03b1", hint: "distinct", vec: fwNorm([1, 0.15, 0, 0, 0.1, 0]) },
    { label: "ctx \u03b2", hint: "distinct", vec: fwNorm([0.1, 1, 0.1, 0, 0, 0]) },
    { label: "ctx \u03b3", hint: "distinct", vec: fwNorm([0, 0, 1, 0.15, 0, 0.12]) },
    { label: "ctx \u03b2\u2032", hint: "looks like \u03b2 \u2192 crosstalk", vec: fwNorm([0.18, 0.92, 0, 0.32, 0, 0]) }
  ];
  function fwZeros() { var m = []; for (var i = 0; i < FW_D; i++) { m.push([]); for (var j = 0; j < FW_V; j++) m[i].push(0); } return m; }
  function fwRead(M, q) {
    var logits = []; for (var j = 0; j < FW_V; j++) { var sm = 0; for (var i = 0; i < FW_D; i++) sm += q[i] * M[i][j]; logits.push(sm); }
    var beta = 3.0, mx = Math.max.apply(null, logits), ex = logits.map(function (l) { return Math.exp(beta * (l - mx)); });
    var z = ex.reduce(function (a, b) { return a + b; }, 0); return ex.map(function (e) { return e / z; });
  }
  function mountFastWeight(root) {
    var M = fwZeros(), decay = 0.92, selKey = 0, selVal = 2, query = 0, log = [];
    function write(ki, vi) {
      var k = FW_KEYS[ki].vec;
      for (var i = 0; i < FW_D; i++) { for (var j = 0; j < FW_V; j++) M[i][j] *= decay; M[i][vi] += k[i]; }
      log = ["write  " + FW_KEYS[ki].label + " \u2192 '" + FW_BYTES[vi] + "'"].concat(log).slice(0, 5);
    }
    function seed() {
      M = fwZeros(); [[0, 2], [1, 0], [2, 4]].forEach(function (pair) { var k = FW_KEYS[pair[0]].vec; for (var i = 0; i < FW_D; i++) { for (var j = 0; j < FW_V; j++) M[i][j] *= decay; M[i][pair[1]] += k[i]; } });
      log = ["seed   \u03b1\u2192't', \u03b2\u2192'e', \u03b3\u2192'n'"];
    }
    function fade() { for (var i = 0; i < FW_D; i++) for (var j = 0; j < FW_V; j++) M[i][j] *= Math.pow(decay, 6); log = ["fade   \u00d7 decay\u2076 (forgetting)"].concat(log).slice(0, 5); }
    function render() {
      var dist = fwRead(M, FW_KEYS[query].vec), argmax = 0; for (var j = 1; j < FW_V; j++) if (dist[j] > dist[argmax]) argmax = j;
      var flat = []; M.forEach(function (r) { r.forEach(function (x) { flat.push(Math.abs(x)); }); });
      var maxAbs = Math.max.apply(null, [0.4].concat(flat));
      function cell(x) { var a = Math.min(1, Math.abs(x) / maxAbs).toFixed(3); return x >= 0 ? "rgba(232,181,77," + a + ")" : "rgba(94,168,230," + a + ")"; }
      var s = '<figure class="fw"><div class="fw-top"><div class="fw-controls"><p class="fw-h">write an association</p><div class="fw-keys">';
      FW_KEYS.forEach(function (k, i) { s += '<button type="button" class="fw-key' + (selKey === i ? " on" : "") + '" data-key="' + i + '">' + k.label + '<span class="fw-key-hint">' + k.hint + "</span></button>"; });
      s += '</div><p class="fw-sub">value (next byte)</p><div class="fw-vals">';
      FW_BYTES.forEach(function (b, i) { s += '<button type="button" class="fw-val' + (selVal === i ? " on" : "") + '" data-val="' + i + '">' + b + "</button>"; });
      s += '</div><div class="fw-actions"><button type="button" class="primary" data-act="write">Write \u00a0' + FW_KEYS[selKey].label + " \u2192 \u2018" + FW_BYTES[selVal] + "\u2019</button></div>";
      s += '<label class="fw-decay">memory decay\u00a0<output>' + decay.toFixed(2) + '</output><input type="range" min="0.80" max="1.00" step="0.01" value="' + decay + '" data-decay></label>';
      s += '<div class="fw-actions"><button type="button" data-act="seed">Seed 3</button><button type="button" data-act="fade">Let it fade</button><button type="button" data-act="reset">Reset</button></div>';
      if (log.length) { s += '<ul class="fw-log">'; log.forEach(function (l) { s += "<li>" + l + "</li>"; }); s += "</ul>"; }
      s += '</div><div class="fw-matrix"><p class="fw-h">memory M \u00a0<span class="fw-dim">(d\u00d7V = ' + FW_D + "\u00d7" + FW_V + ')</span></p><div class="fw-grid" style="grid-template-columns:1.4rem repeat(' + FW_V + ',1fr)"><span class="fw-corner"></span>';
      FW_BYTES.forEach(function (b) { s += '<span class="fw-col-label">' + b + "</span>"; });
      M.forEach(function (row, i) { s += '<span class="fw-row-label">d' + i + "</span>"; row.forEach(function (x) { s += '<span class="fw-cell" style="background:' + cell(x) + '" title="' + x.toFixed(2) + '"></span>'; }); });
      s += '</div><p class="fw-mathnote">write: <code>M \u2190 decay\u00b7M + key \u2297 e_byte</code> \u00b7 read: <code>q @ M \u2192 softmax</code></p></div></div>';
      s += '<div class="fw-read"><div class="fw-read-head"><span class="fw-h">read: query with</span><div class="fw-keys inline">';
      FW_KEYS.forEach(function (k, i) { s += '<button type="button" class="fw-key sm' + (query === i ? " on" : "") + '" data-query="' + i + '">' + k.label + "</button>"; });
      s += '</div></div><div class="fw-bars">';
      FW_BYTES.forEach(function (b, i) { s += '<div class="fw-bar' + (i === argmax ? " best" : "") + '"><span class="fw-bar-ch">' + b + '</span><span class="fw-bar-track"><span class="fw-bar-fill" style="width:' + (dist[i] * 100) + '%"></span></span><span class="fw-bar-p">' + (dist[i] * 100).toFixed(0) + "%</span></div>"; });
      s += '</div><p class="fw-recall">recalled \u2192 <strong>\u2018' + FW_BYTES[argmax] + "\u2019</strong> at " + (dist[argmax] * 100).toFixed(0) + "% confidence</p></div>";
      s += '<figcaption class="figcaption">The write is one outer product \u2014 instant, <strong>O(1)</strong>, no gradient. Querying \u03b1/\u03b2/\u03b3 recalls their stored byte; querying <strong>\u03b2\u2032</strong> (which points almost the same way as \u03b2) recalls a blurred mix \u2014 that is the <em>crosstalk</em> of superposing associations in one matrix. Drop the decay and write more to watch old associations <em>forget</em>.</figcaption></figure>';
      root.innerHTML = s;
    }
    root.addEventListener("click", function (e) {
      var el = e.target.closest("[data-key],[data-val],[data-query],[data-act]"); if (!el) return;
      if (el.hasAttribute("data-key")) selKey = +el.getAttribute("data-key");
      else if (el.hasAttribute("data-val")) selVal = +el.getAttribute("data-val");
      else if (el.hasAttribute("data-query")) query = +el.getAttribute("data-query");
      else { var act = el.getAttribute("data-act"); if (act === "write") write(selKey, selVal); else if (act === "seed") seed(); else if (act === "fade") fade(); else if (act === "reset") { M = fwZeros(); log = []; } }
      render();
    });
    root.addEventListener("input", function (e) { if (e.target.hasAttribute("data-decay")) { decay = parseFloat(e.target.value); var o = root.querySelector(".fw-decay output"); if (o) o.textContent = decay.toFixed(2); } });
    render();
  }

  // ══ CodeLength (compression = prediction) ════════════════════════════════
  function mountCodeLength(root) {
    var W = 560, Hh = 240, PAD = { l: 46, r: 16, t: 16, b: 34 }, p = 0.5;
    var x0 = PAD.l, x1 = W - PAD.r, y0 = PAD.t, y1 = Hh - PAD.b, CAP = 12;
    function px(pp) { return x0 + pp * (x1 - x0); }
    function py(b) { return y1 - Math.min(b, CAP) / CAP * (y1 - y0); }
    var path = ""; for (var i = 1; i <= 200; i++) { var pp = i / 200; path += (i > 1 ? " " : "") + px(pp).toFixed(1) + "," + py(-Math.log2(pp)).toFixed(1); }
    var staticSvg = "";
    staticSvg += '<line x1="' + x0 + '" x2="' + x1 + '" y1="' + py(8) + '" y2="' + py(8) + '" class="cld-ref"/>';
    staticSvg += '<text x="' + (x1 - 4) + '" y="' + (py(8) - 5) + '" class="cld-reftext" text-anchor="end">8 bits \u2014 no model</text>';
    staticSvg += '<line x1="' + x0 + '" x2="' + x1 + '" y1="' + y1 + '" y2="' + y1 + '" class="cld-axis"/><line x1="' + x0 + '" x2="' + x0 + '" y1="' + y0 + '" y2="' + y1 + '" class="cld-axis"/>';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) { staticSvg += '<text x="' + px(t) + '" y="' + (y1 + 18) + '" class="cld-tick" text-anchor="middle">' + t + "</text>"; });
    [0, 4, 8, 12].forEach(function (b) { staticSvg += '<text x="' + (x0 - 6) + '" y="' + (py(b) + 4) + '" class="cld-tick" text-anchor="end">' + b + "</text>"; });
    staticSvg += '<text x="' + ((x0 + x1) / 2) + '" y="' + (Hh - 4) + '" class="cld-axislabel" text-anchor="middle">p(true next byte)</text>';
    staticSvg += '<polyline points="' + path + '" fill="none" stroke="#5ea8e6" stroke-width="2.2"/>';
    root.innerHTML =
      '<figure class="cld"><div class="cld-grid"><div class="cld-readout">' +
      '<div class="cld-bignum"><span class="cld-val">1.000</span><span class="cld-unit">bits / byte</span></div>' +
      '<p class="cld-eq"></p>' +
      '<label class="cld-label">model\u2019s probability on the <em>true</em> next byte<input type="range" min="0.01" max="0.99" step="0.01" value="0.5" data-p><output>p = 0.50</output></label>' +
      '<p class="cld-note"></p></div>' +
      '<svg viewBox="0 0 ' + W + " " + Hh + '" class="cld-svg" role="img" aria-label="Code length minus log2 p versus probability">' + staticSvg +
      '<line class="cld-drop"/><circle r="6" fill="#e8b54d" stroke="#16130e" stroke-width="1.5" class="cld-pt"/></svg></div>' +
      '<figcaption class="figcaption">Arithmetic coding stores the true next byte in <strong>\u2212log\u2082\u00a0p</strong> bits. Confident and right \u2192 nearly free; surprised \u2192 expensive; confidently wrong (small p) \u2192 worse than the 8-bit uniform prior. Sum this over a stream \u00f7 bytes = bpb.</figcaption></figure>';
    var elVal = root.querySelector(".cld-val"), elEq = root.querySelector(".cld-eq"), elNote = root.querySelector(".cld-note"), elOut = root.querySelector(".cld-label output");
    var pt = root.querySelector(".cld-pt"), drop = root.querySelector(".cld-drop");
    function update() {
      var cost = -Math.log2(p), saved = 8 - cost;
      elVal.textContent = cost.toFixed(3);
      elEq.innerHTML = "\u2212log\u2082(" + p.toFixed(2) + ") = " + cost.toFixed(3);
      elOut.textContent = "p = " + p.toFixed(2);
      elNote.className = "cld-note " + (saved >= 0 ? "good" : "bad");
      elNote.textContent = saved >= 0 ? saved.toFixed(2) + " bits cheaper than the 8-bit \u201cno model\u201d." : (-saved).toFixed(2) + " bits worse than no model \u2014 being confidently wrong costs.";
      var cx = px(p), cy = py(cost);
      pt.setAttribute("cx", cx); pt.setAttribute("cy", cy);
      drop.setAttribute("x1", cx); drop.setAttribute("x2", cx); drop.setAttribute("y1", cy); drop.setAttribute("y2", y1);
    }
    root.addEventListener("input", function (e) { if (e.target.hasAttribute("data-p")) { p = parseFloat(e.target.value); update(); } });
    update();
  }

  // ══ ScalingCalculator ════════════════════════════════════════════════════
  var HW = [{ label: "laptop GPU (~2\u00d710\u00b9\u00b3)", flops: 2e13 }, { label: "A100 (~3\u00d710\u00b9\u2074)", flops: 3.1e14 }, { label: "H100 (~1\u00d710\u00b9\u2075)", flops: 1e15 }];
  var SC_PRESETS = [{ label: "GPT-3", n: 175e9, d: 300e9 }, { label: "smolml d32 baseline", n: 32928, d: 1.5e6 }];
  function humanTime(seconds) {
    var units = [["year", 365.25 * 86400], ["day", 86400], ["hour", 3600], ["minute", 60], ["second", 1]];
    for (var i = 0; i < units.length; i++) { var s = units[i][1]; if (seconds >= s) { var v = seconds / s; return (v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(1)) + " " + units[i][0] + (v >= 1.5 ? "s" : ""); } }
    return seconds.toExponential(1) + " s";
  }
  function mountScaling(root) {
    var logN = Math.log10(175e9), logD = Math.log10(300e9), hw = 0;
    var gpt3C = 6 * 175e9 * 300e9;
    root.innerHTML =
      '<figure class="sc"><div class="sc-head"><div class="sc-eq"></div><div class="sc-time"></div></div>' +
      '<label class="sc-row"><span class="sc-name">N \u2014 parameters</span><input type="range" min="3" max="12" step="0.05" value="' + logN + '" data-n><span class="sc-val sc-vn"></span></label>' +
      '<label class="sc-row"><span class="sc-name">D \u2014 tokens seen</span><input type="range" min="4" max="12" step="0.05" value="' + logD + '" data-d><span class="sc-val sc-vd"></span></label>' +
      '<label class="sc-row"><span class="sc-name">hardware</span><select data-hw>' + HW.map(function (h, i) { return '<option value="' + i + '">' + h.label + "</option>"; }).join("") + '</select><span class="sc-val sc-vh"></span></label>' +
      '<div class="sc-bar" aria-hidden="true"><div class="sc-bar-fill"></div><span class="sc-bar-label"></span></div>' +
      '<div class="sc-presets"><span>presets:</span>' + SC_PRESETS.map(function (p, i) { return '<button type="button" data-preset="' + i + '">' + p.label + "</button>"; }).join("") + "</div>" +
      '<figcaption class="figcaption">The transformer \u201cfits on a napkin\u201d, but the <em>training</em> cost is set by C = 6\u00b7N\u00b7D. GPT-3 (N\u22481.75\u00d710\u00b9\u00b9, D\u22483\u00d710\u00b9\u00b9) lands at \u22483\u00d710\u00b2\u00b3 FLOPs \u2014 centuries on a laptop. Simplicity of the kernel \u2260 cheapness of training; that category error is the whole trap.</figcaption></figure>';
    var elEq = root.querySelector(".sc-eq"), elTime = root.querySelector(".sc-time"), vn = root.querySelector(".sc-vn"), vd = root.querySelector(".sc-vd"), vh = root.querySelector(".sc-vh");
    var fill = root.querySelector(".sc-bar-fill"), barLabel = root.querySelector(".sc-bar-label"), sel = root.querySelector("[data-hw]");
    function update() {
      var N = Math.pow(10, logN), D = Math.pow(10, logD), C = 6 * N * D, seconds = C / HW[hw].flops, frac = C / gpt3C;
      elEq.innerHTML = "C = 6 \u00b7 N \u00b7 D = <strong>" + sci(C) + "</strong> FLOPs";
      elTime.innerHTML = "\u2248 <strong>" + humanTime(seconds) + "</strong> at 100% util on " + HW[hw].label;
      vn.textContent = sci(N); vd.textContent = sci(D); vh.textContent = sci(HW[hw].flops) + " FLOP/s";
      fill.style.width = Math.max(1, Math.min(100, Math.log10(Math.max(C, 1)) / Math.log10(gpt3C) * 100)) + "%";
      barLabel.textContent = frac >= 1 ? frac.toFixed(1) + "\u00d7 GPT-3" : (frac * 100).toExponential(1) + "% of GPT-3";
    }
    root.addEventListener("input", function (e) {
      if (e.target.hasAttribute("data-n")) logN = parseFloat(e.target.value);
      else if (e.target.hasAttribute("data-d")) logD = parseFloat(e.target.value);
      else return; update();
    });
    root.addEventListener("change", function (e) { if (e.target.hasAttribute("data-hw")) { hw = parseInt(e.target.value, 10); update(); } });
    root.addEventListener("click", function (e) {
      var b = e.target.closest("[data-preset]"); if (!b) return;
      var pr = SC_PRESETS[+b.getAttribute("data-preset")]; logN = Math.log10(pr.n); logD = Math.log10(pr.d);
      root.querySelector("[data-n]").value = logN; root.querySelector("[data-d]").value = logD; update();
    });
    update();
  }

  // ══ SourceIvScreen ═══════════════════════════════════════════════════════
  var SIV = [
    { id: "i", name: "cheaper credit assignment", claim: "\u201cskip the backward pass\u201d", impact: "low", note: "Backward is only ~2\u00d7 forward, so the ceiling is ~3\u00d7 \u2014 and it is usually spent back. Modest." },
    { id: "ii", name: "locality \u2192 parallelism / async", claim: "\u201cno global sync\u201d", impact: "zero", note: "Wall-clock & scaling only. Scores ZERO on a fixed FLOP budget \u2014 out of scope." },
    { id: "iii", name: "no activation storage", claim: "\u201cfits in memory\u201d", impact: "zero", note: "Memory only; barely touches FLOPs." },
    { id: "iv", name: "better learning dynamics", claim: "\u201creduces loss faster per FLOP\u201d", impact: "high", note: "The update rule itself extracts more loss-reduction per FLOP. The only thing that moves our scoreboard." }
  ];
  var SIV_PCT = { high: 100, low: 28, zero: 4 };
  function mountSourceIv(root) {
    var on = {};
    function render() {
      var keys = Object.keys(on).filter(function (k) { return on[k]; });
      var hasIv = !!on.iv, hasAny = keys.length > 0, verdict = !hasAny ? "empty" : hasIv ? "scout" : "park";
      var s = '<figure class="siv"><p class="siv-q">Before any candidate earns a GPU-hour: <em>is there a plausible reason this reduces loss faster <strong>per FLOP</strong> \u2014 or is it just avoiding a cheap backward pass / buying parallelism we don\u2019t reward?</em></p><div class="siv-chips">';
      SIV.forEach(function (sr) { s += '<button type="button" class="siv-chip ' + (on[sr.id] ? "on " : "") + "imp-" + sr.impact + '" data-src="' + sr.id + '" aria-pressed="' + (!!on[sr.id]) + '"><span class="siv-chip-id">(' + sr.id + ")</span> " + sr.name + "</button>"; });
      s += '</div><div class="siv-rows">';
      SIV.filter(function (sr) { return on[sr.id]; }).forEach(function (sr) {
        s += '<div class="siv-row"><div class="siv-row-head"><span class="siv-row-name">(' + sr.id + ") " + sr.claim + '</span><span class="siv-tag imp-' + sr.impact + '">' + (sr.impact === "high" ? "moves the metric" : sr.impact === "low" ? "barely" : "scores zero") + "</span></div>";
        s += '<div class="siv-meter"><div class="siv-meter-fill imp-' + sr.impact + '" style="width:' + SIV_PCT[sr.impact] + '%"></div></div><p class="siv-note">' + sr.note + "</p></div>";
      });
      if (!hasAny) s += '<p class="siv-empty">Toggle the reasons the candidate claims \u2191</p>';
      s += "</div>";
      s += '<div class="siv-verdict v-' + verdict + '">';
      if (verdict === "empty") s += "<span>awaiting a claim\u2026</span>";
      else if (verdict === "scout") s += "<span><strong>Scout it.</strong> There is a real per-FLOP story (iv) \u2014 the only admissible reason to spend the compute.</span>";
      else s += "<span><strong>Parked.</strong> Only (i)/(ii)/(iii) \u2014 these buy speed, parallelism, or memory, none of which the fixed-FLOP metric rewards. Not forbidden as inspiration; just not a win here.</span>";
      s += "</div>";
      s += '<div class="siv-presets"><span>try:</span><button type="button" data-set="i,ii">Forward-Forward</button><button type="button" data-set="iv">a real (iv) candidate</button><button type="button" data-set="">clear</button></div>';
      s += '<figcaption class="figcaption"><em>Forward-Forward</em> replaces the backward pass with a second forward pass \u2014 but two forwards \u2248 one forward+backward, so its (i) saving is \u22480 and its real selling point is (ii) locality. No distinct (iv) story \u2192 parked here.</figcaption></figure>';
      root.innerHTML = s;
    }
    root.addEventListener("click", function (e) {
      var chip = e.target.closest("[data-src]"), preset = e.target.closest("[data-set]");
      if (chip) { var id = chip.getAttribute("data-src"); on[id] = !on[id]; render(); }
      else if (preset) { on = {}; preset.getAttribute("data-set").split(",").forEach(function (k) { if (k) on[k] = true; }); render(); }
    });
    render();
  }

  // ══ ControlRollout (in-context control / chemotaxis rung) ════════════════
  // Scrubbable instrument over one trained, held-out ChemoEnv rollout: the
  // unrolled W-cell ring (current concentration field with the hidden peak ▼
  // and the agent ▲), a spacetime raster (time ↓) tracing both paths, and a
  // cumulative-reward spark whose slope steepens as the agent stops climbing
  // and starts tracking. Built once, then mutated in place per step (the
  // 33×16 raster never re-renders, so scrubbing stays smooth).
  function mountControlRollout(root) {
    var d = readData(root);
    var field = d.field || [], T = field.length;
    var W = d.width || (field[0] ? field[0].length : 16), L = d.levels || 8, H = d.horizon || (T - 1);
    var mu = d.mu || [], pos = d.pos || [], tok = d.conc_token || [], reward = d.reward || [], action = d.action || [];
    var cum = [0]; for (var ci = 0; ci < reward.length; ci++) cum.push(cum[ci] + reward[ci]);
    var ACT = ["LEFT", "STAY", "RIGHT"], ACT_ARROW = ["\u2190", "\u00b7", "\u2192"];
    var AMBER = "232,181,77", GREEN = "#5cc46a", BLUE = "#5ea8e6";

    var VBW = 472, PL = 34, PR = 14, LBL = 16, MK = 14, FIELDH = 48, GAP = 22, AXIS = 30;
    var IW = VBW - PL - PR, cw = IW / W;
    var fTop = LBL + MK, fBot = fTop + FIELDH;
    var rTop = fBot + MK + GAP, rh = 8.2, RH = rh * T, rBot = rTop + RH, VBH = rBot + AXIS;
    function colX(c) { return PL + c * cw; }
    function colCX(v) { return PL + (v + 0.5) * cw; }
    function rowCY(iy) { return rTop + (iy + 0.5) * rh; }
    function heat(c) { var a = Math.pow(Math.max(0, Math.min(1, c)), 0.6); return "rgba(" + AMBER + "," + a.toFixed(3) + ")"; }

    function rasterCells() {
      var s = "";
      for (var iy = 0; iy < T; iy++) {
        var row = field[iy] || [];
        for (var c = 0; c < W; c++) s += '<rect x="' + colX(c).toFixed(1) + '" y="' + (rTop + iy * rh).toFixed(1) + '" width="' + (cw + 0.6).toFixed(1) + '" height="' + (rh + 0.6).toFixed(1) + '" fill="' + heat(row[c]) + '"/>';
      }
      return s;
    }
    // Break the path at ring-seam wraps so we never draw a misleading streak
    // straight across the raster when the agent crosses cell 15 ↔ 0.
    function pathSegs(vals, color, dashed) {
      var segs = [], cur = [];
      for (var iy = 0; iy < vals.length; iy++) {
        if (iy > 0 && Math.abs(vals[iy] - vals[iy - 1]) > W / 2) { segs.push(cur); cur = []; }
        cur.push(colCX(vals[iy]).toFixed(1) + "," + rowCY(iy).toFixed(1));
      }
      if (cur.length) segs.push(cur);
      return segs.map(function (sg) { return '<polyline points="' + sg.join(" ") + '" fill="none" stroke="' + color + '" stroke-width="1.7"' + (dashed ? ' stroke-dasharray="3 3"' : "") + ' stroke-linejoin="round" stroke-linecap="round"/>'; }).join("");
    }
    function fieldGroup(t) {
      var row = field[t] || [], s = "";
      for (var c = 0; c < W; c++) {
        var v = Math.max(0, Math.min(1, row[c] || 0)), hh = v * FIELDH;
        var isAgent = pos[t] === c;
        s += '<rect x="' + (colX(c) + 1).toFixed(1) + '" y="' + (fBot - hh).toFixed(1) + '" width="' + (cw - 2).toFixed(1) + '" height="' + Math.max(0.6, hh).toFixed(1) + '" rx="1.5" fill="rgba(' + AMBER + "," + Math.pow(v, 0.55).toFixed(3) + ')"' + (isAgent ? ' stroke="' + GREEN + '" stroke-width="1.6"' : "") + "/>";
      }
      var pk = colCX(mu[t]);
      s += '<path d="M' + (pk - 5).toFixed(1) + " " + (LBL + 2) + " L" + (pk + 5).toFixed(1) + " " + (LBL + 2) + " L" + pk.toFixed(1) + " " + (fTop - 1) + ' Z" fill="' + BLUE + '"/>';
      var ag = colCX(pos[t]);
      s += '<path d="M' + (ag - 5).toFixed(1) + " " + (fBot + MK - 1) + " L" + (ag + 5).toFixed(1) + " " + (fBot + MK - 1) + " L" + ag.toFixed(1) + " " + (fBot + 1) + ' Z" fill="' + GREEN + '"/>';
      return s;
    }

    var SVW = 240, SVH = 96, SPL = 4, SPR = 4, SPT = 8, SPB = 16;
    var sIW = SVW - SPL - SPR, sIH = SVH - SPT - SPB, cumMax = (T - 1) || 1;
    function sx(i) { return SPL + (T <= 1 ? 0 : i / (T - 1) * sIW); }
    function sy(v) { return SPT + sIH - (v / cumMax) * sIH; }
    function refLine(slope) { return sx(0).toFixed(1) + "," + sy(0).toFixed(1) + " " + sx(T - 1).toFixed(1) + "," + sy(slope * (T - 1)).toFixed(1); }
    function cumPts(upto) { var p = []; for (var iy = 0; iy <= upto; iy++) p.push(sx(iy).toFixed(1) + "," + sy(cum[iy]).toFixed(1)); return p.join(" "); }

    function stat(v, l, hi) { return '<div class="ctrl-stat' + (hi ? " hi" : "") + '"><span class="ctrl-stat-v">' + v + '</span><span class="ctrl-stat-l">' + l + "</span></div>"; }
    function scoreboard() {
      return '<div class="ctrl-score">' +
        stat(d.regret != null ? d.regret.toFixed(3) : "\u2014", "regret \u2193", true) +
        stat(d.mean_reward != null ? d.mean_reward.toFixed(3) : "\u2014", "mean reward \u2191") +
        stat("0.37", "random floor") +
        stat("0", "oracle regret") + "</div>";
    }
    function staticSvg() {
      var s = '<svg viewBox="0 0 ' + VBW + " " + VBH + '" class="ctrl-svg" role="img" aria-label="Chemotaxis rollout: the agent climbs onto the drifting concentration peak and tracks it over ' + H + ' steps.">';
      s += '<rect x="' + PL + '" y="' + rTop.toFixed(1) + '" width="' + IW + '" height="' + RH.toFixed(1) + '" fill="#16130e"/>';
      s += rasterCells();
      s += pathSegs(mu, BLUE, true);
      s += pathSegs(pos, GREEN, false);
      s += '<rect class="ctrl-hl" x="' + PL + '" width="' + IW + '" height="' + rh.toFixed(1) + '" y="' + rTop.toFixed(1) + '"/>';
      s += '<text x="' + PL + '" y="' + (LBL - 5) + '" class="ctrl-cap">the world now \u2014 the agent senses only its own cell</text>';
      s += '<g class="ctrl-field"></g>';
      [0, 4, 8, 12, 15].forEach(function (c) { s += '<text x="' + colCX(c).toFixed(1) + '" y="' + (rBot + 13) + '" class="ctrl-tick" text-anchor="middle">' + c + "</text>"; });
      s += '<text x="' + (PL + IW / 2).toFixed(1) + '" y="' + (rBot + AXIS - 2) + '" class="ctrl-tick" text-anchor="middle">ring cell 0\u201315 (wraps) \u2014 spacetime raster, time \u2193</text>';
      s += "</svg>";
      return s;
    }
    function readout() {
      return '<div class="ctrl-read"><div class="ctrl-step"><span class="ctrl-step-v">0</span><span class="ctrl-step-l">/ ' + H + ' steps</span></div>' +
        '<dl class="ctrl-kv"><dt>sensed</dt><dd class="ctrl-sense">\u2014</dd><dt>next move</dt><dd class="ctrl-action">\u2014</dd><dt>reward so far</dt><dd class="ctrl-cum">0.00</dd></dl>' +
        '<p class="ctrl-cum-h">cumulative reward</p>' +
        '<svg viewBox="0 0 ' + SVW + " " + SVH + '" class="ctrl-spark" role="img" aria-label="cumulative reward versus step">' +
        '<polyline points="' + refLine(1) + '" class="ctrl-ceil"/>' +
        '<polyline points="' + refLine(0.37) + '" class="ctrl-floor"/>' +
        '<polyline points="' + cumPts(T - 1) + '" class="ctrl-cum-full"/>' +
        '<polyline class="ctrl-cum-live" points=""/>' +
        '<circle class="ctrl-cum-dot" r="3" cx="' + sx(0).toFixed(1) + '" cy="' + sy(0).toFixed(1) + '"/></svg>' +
        '<p class="ctrl-spark-cap">faint lines: perfect 1.0/step (top) and random 0.37/step. The agent line steepens as it stops climbing and starts tracking.</p></div>';
    }
    function controls() {
      return '<div class="ctrl-controls"><button type="button" data-act="step">Step</button>' +
        '<button type="button" class="primary" data-act="play">Play</button>' +
        '<button type="button" data-act="reset">Reset</button>' +
        '<input type="range" class="ctrl-scrub" min="0" max="' + (T - 1) + '" value="0" step="1" aria-label="scrub to step"/></div>';
    }
    function caption() {
      return '<figcaption class="figcaption">One trained model, one held-out episode (drift drawn from the eval-only pool). <strong>Top:</strong> the concentration field right now \u2014 the hidden peak (\u25bc, blue) the agent cannot see, and the agent (\u25b2, green) that senses only its own cell. <strong>Below:</strong> the spacetime raster (time \u2193); the green path is the agent, the blue dashed path is the peak it chases. Press Play or scrub: the agent climbs onto the drifting peak, then tracks it \u2014 the cumulative-reward line steepens once tracking begins. Scoreboard numbers are for this single rollout; the bar table reports the held-out mean.</figcaption>';
    }

    root.innerHTML = scoreboard() + '<div class="ctrl-stage">' + staticSvg() + readout() + "</div>" + controls() + caption();
    var elField = root.querySelector(".ctrl-field"), elHl = root.querySelector(".ctrl-hl");
    var elStep = root.querySelector(".ctrl-step-v"), elSense = root.querySelector(".ctrl-sense"), elAction = root.querySelector(".ctrl-action"), elCum = root.querySelector(".ctrl-cum");
    var elLive = root.querySelector(".ctrl-cum-live"), elDot = root.querySelector(".ctrl-cum-dot");
    var elScrub = root.querySelector(".ctrl-scrub"), elPlay = root.querySelector('[data-act="play"]'), elStepBtn = root.querySelector('[data-act="step"]');
    var t = 0, playing = false, timer = null;
    function atEnd() { return t >= T - 1; }
    function update(fromScrub) {
      elField.innerHTML = fieldGroup(t);
      elHl.setAttribute("y", (rTop + t * rh).toFixed(1));
      elStep.textContent = t;
      var lvl = tok[t], conc = field[t] && field[t][pos[t]];
      elSense.innerHTML = (lvl != null ? "level " + lvl + " / " + (L - 1) : "\u2014") + ' <span class="ctrl-dim">(' + (conc != null ? conc.toFixed(2) : "?") + ")</span>";
      if (t < action.length) elAction.innerHTML = '<span class="ctrl-arrow">' + ACT_ARROW[action[t]] + "</span> " + ACT[action[t]];
      else elAction.textContent = "episode end";
      elCum.textContent = cum[t].toFixed(2);
      elLive.setAttribute("points", cumPts(t));
      elDot.setAttribute("cx", sx(t).toFixed(1)); elDot.setAttribute("cy", sy(cum[t]).toFixed(1));
      if (!fromScrub) elScrub.value = t;
      elPlay.textContent = playing ? "Pause" : "Play";
      elStepBtn.disabled = atEnd();
    }
    function step() { if (!atEnd()) { t += 1; update(); } }
    function loop() {
      clearTimeout(timer);
      if (playing && !atEnd()) timer = setTimeout(function () { if (playing) { step(); loop(); } }, 420);
      else if (atEnd()) { playing = false; update(); }
    }
    root.addEventListener("click", function (e) {
      var b = e.target.closest("[data-act]"); if (!b) return;
      var a = b.getAttribute("data-act");
      if (a === "step") { playing = false; clearTimeout(timer); step(); update(); }
      else if (a === "play") { if (atEnd()) t = 0; playing = !playing; update(); loop(); }
      else if (a === "reset") { playing = false; clearTimeout(timer); t = 0; update(); }
    });
    root.addEventListener("input", function (e) { if (e.target.classList.contains("ctrl-scrub")) { playing = false; clearTimeout(timer); t = parseInt(e.target.value, 10) || 0; update(true); } });
    update();
  }

  // ══ Interactive-demo helpers (shared by the two runnable model demos) ═════
  // The byte race + the cursor chase both run the parity-validated JS model
  // layer live (loaded as classic scripts before this file). They share a
  // transport (Step / Play / Reset on a timer) and number formatting; factored
  // here per the rule of two. The model layer attaches to window.SmolModels /
  // window.SmolDemos.
  function humanFlops(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
    return "" + n;
  }
  // Byte -> a display glyph: printable ASCII + Latin-1 supplement as the char,
  // space/newline/tab as marks, control/C1 bytes as a middot.
  function byteGlyph(b) {
    if (b === 32) return "\u2423";
    if (b === 10) return "\u21b5";
    if (b === 9) return "\u21e5";
    if (b === 13) return "\u23ce";
    if (b < 32 || (b >= 127 && b <= 160)) return "\u00b7";
    return String.fromCharCode(b);
  }
  // Wire Step/Play/Reset on `root` (buttons carry data-act). `h`:
  //   step(), reset(), atEnd():bool, interval:ms, onState(playing):void.
  // `atEnd` lets the byte race stop at the stream end; the (endless) cursor
  // chase returns false. Returns { stop } (used on edit/teardown).
  function wireTransport(root, h) {
    var playing = false, timer = null;
    var reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function setState(p) { playing = p; if (h.onState) h.onState(p); }
    function hardStop() { playing = false; clearTimeout(timer); if (h.onState) h.onState(false); }
    function loop() {
      clearTimeout(timer);
      if (playing && !h.atEnd()) {
        timer = setTimeout(function () {
          if (!playing) return;
          h.step();
          if (h.atEnd()) { setState(false); return; }
          loop();
        }, h.interval || 200);
      } else if (h.atEnd()) { setState(false); }
    }
    root.addEventListener("click", function (e) {
      var b = e.target.closest("[data-act]"); if (!b) return;
      var a = b.getAttribute("data-act");
      if (a === "step") { playing = false; clearTimeout(timer); if (!h.atEnd()) h.step(); setState(false); }
      else if (a === "play") { if (reduced || h.atEnd()) return; playing = !playing; setState(playing); loop(); }
      else if (a === "reset") { playing = false; clearTimeout(timer); h.reset(); setState(false); }
    });
    // prefers-reduced-motion: no auto-play timer — fall back to Step-only (hide Play).
    if (reduced) { var pb = root.querySelector('[data-act="play"]'); if (pb) { pb.hidden = true; pb.style.display = "none"; pb.setAttribute("aria-hidden", "true"); } }
    return { stop: hardStop };
  }

  // ══ AutocompleteRace (live byte loss-per-FLOP race) ═══════════════════════
  // Three byte models predict the SAME editable enwik8 stream live: each runs
  // predict-then-learn (create once, then step per byte — fold the revealed
  // byte, adapt online, read the next-byte probabilities). Per model: top-k
  // next-char bars, cumulative bpb (= Σ −log2 p(nextByte) / bytes; byte 0 = 8
  // bits), top-1 hit-rate, and the HUD FLOPs/byte — so the ~1000× transformer
  // cost shows against the ~10k-FLOP online learners. Mirrors parity.mjs's byte
  // loop; the editable text IS the stream (latin-1: text.charCodeAt(i)&0xff ==
  // the enwik8 byte). Shell built once; only the panels/context mutate per step.
  function mountAutocompleteRace(root) {
    var SM = window.SmolModels, SD = window.SmolDemos;
    var d = readData(root), defs = d.models || [], seedText = d.seedText || "";
    if (!SM || !SD) { root.innerHTML = '<p class="demo-err">interactive model layer not loaded</p>'; return; }

    var stream = [];
    function deriveStream(text) { var n = text.length, a = new Array(n); for (var i = 0; i < n; i++) a[i] = text.charCodeAt(i) & 0xff; return a; }
    function fresh(def) {
      var opts = {};
      if (def.config) opts.config = def.config;
      if (def.weights) opts.weights = def.weights;
      return { def: def, model: SM[def.id], state: SM[def.id].create(opts), pred: null, totalBits: 0, folded: 0, hits: 0, hitDenom: 0 };
    }
    var runs = [], pos = 0;
    function atEnd() { return pos >= stream.length; }
    function reset() {
      stream = deriveStream(root.querySelector(".race-text").value);
      runs = defs.map(fresh); pos = 0; update();
    }
    function step() {
      if (atEnd()) return;
      var k = pos, b = stream[k], last = (k === stream.length - 1);
      for (var i = 0; i < runs.length; i++) {
        var r = runs[i], p, am;
        if (r.pred === null) { p = 1 / 256; am = -1; } // byte 0: uniform prior, no top-1 credit
        else { p = r.pred[b]; am = SD.argmax(r.pred); }
        r.totalBits += -Math.log2(p);
        if (k >= 1) { r.hitDenom++; if (am === b) r.hits++; }
        r.folded = k + 1; // bytes SCORED so far — the bpb denominator
        // Prequential folds n−1 times: skip folding the final byte (its prediction
        // would never be scored) so executed model.steps == scored predictions.
        if (!last) { var out = r.model.step(r.state, b, k); r.state = out.state; r.pred = out.probs; }
      }
      pos = k + 1; update();
    }

    function topk(pred, n) {
      var ranked = []; for (var b = 0; b < 256; b++) ranked.push({ b: b, p: pred[b] });
      ranked.sort(function (a, c) { return c.p - a.p; });
      return ranked.slice(0, n);
    }
    function panelHtml(r) {
      var def = r.def, color = ROLE_COLOR[def.role] || ROLE_COLOR.neutral;
      var bpb = r.folded > 0 ? r.totalBits / r.folded : 0;
      var hit = r.hitDenom > 0 ? (r.hits / r.hitDenom * 100) : 0;
      var truth = atEnd() ? -1 : stream[pos];
      var s = '<div class="race-phead"><span class="race-swatch" style="background:' + color + '"></span><span class="race-name">' + def.label + '</span>';
      s += '<span class="race-flop" title="' + (def.flopsNote ? def.flopsNote.replace(/"/g, "&quot;") : "steady-state per byte") + '">' + humanFlops(def.flopsPerByte) + ' <span class="race-flop-l">FLOPs/byte</span></span></div>';
      s += '<div class="race-metrics"><div class="race-metric"><span class="race-metric-v" style="color:' + color + '">' + bpb.toFixed(3) + '</span><span class="race-metric-l">cumulative bpb</span></div>';
      s += '<div class="race-metric"><span class="race-metric-v">' + hit.toFixed(0) + '%</span><span class="race-metric-l">top-1 hit-rate</span></div>';
      s += '<div class="race-metric"><span class="race-metric-v race-dim">' + (def.params ? def.params.toLocaleString() : "0") + '</span><span class="race-metric-l">params</span></div></div>';
      s += '<p class="race-pk">p(next byte) \u2014 top 6</p>';
      if (r.pred) {
        var top = topk(r.pred, 6), maxP = top[0].p || 1;
        top.forEach(function (t) {
          var isT = t.b === truth;
          s += '<div class="race-bar' + (isT ? " istrue" : "") + '"><span class="race-bar-ch">' + byteGlyph(t.b) + '</span><span class="race-bar-track"><span class="race-bar-fill" style="width:' + (t.p / maxP * 100).toFixed(1) + '%;background:' + color + '"></span></span><span class="race-bar-p">' + (t.p * 100).toFixed(1) + '%</span></div>';
        });
      } else {
        s += '<p class="race-flat">uniform prior \u2014 press Step or Play to read the stream</p>';
      }
      if (truth >= 0) {
        var pend = r.pred ? (-Math.log2(r.pred[truth])).toFixed(2) : "8.00";
        s += '<p class="race-pending">next is <code>' + byteGlyph(truth) + '</code> \u2192 pays <strong style="color:' + color + '">' + pend + '</strong> bits</p>';
      }
      return s;
    }
    function contextHtml() {
      var lo = Math.max(0, pos - 44), s = lo > 0 ? '<span class="race-ctx-pre">\u2026</span>' : "";
      for (var i = lo; i < pos; i++) s += '<span class="race-ctx-seen">' + byteGlyph(stream[i]) + '</span>';
      if (!atEnd()) s += '<span class="race-ctx-cur">\u25ae</span>';
      return s;
    }
    function update() {
      var ctx = root.querySelector(".race-context"); if (ctx) ctx.innerHTML = contextHtml();
      var ps = root.querySelector(".race-pos"); if (ps) ps.textContent = "byte " + pos + " / " + stream.length;
      for (var i = 0; i < runs.length; i++) {
        var el = root.querySelector('.race-panel[data-model="' + runs[i].def.id + '"]');
        if (el) el.innerHTML = panelHtml(runs[i]);
      }
    }
    function onState(playing) {
      var pb = root.querySelector('[data-act="play"]'); if (pb) { pb.textContent = playing ? "Pause" : "Play"; pb.disabled = atEnd() && !playing; }
      var sb = root.querySelector('[data-act="step"]'); if (sb) sb.disabled = atEnd();
    }
    var online = defs.filter(function (m) { return !m.params; }).map(function (m) { return m.label; }).join(" + ");
    var fl = defs.map(function (m) { return m.flopsPerByte; });
    var ratio = Math.round(Math.max.apply(null, fl) / Math.min.apply(null, fl)).toLocaleString();
    var caption = 'Each model runs <strong>predict-then-learn</strong> on the same bytes: it scores the next byte at \u2212log\u2082\u00a0p (byte\u00a00 = 8 bits), then folds it and adapts. The two online learners (<strong>' + online + '</strong>) carry <strong>zero pretrained parameters</strong> and warm up live; the transformer is pretrained. On this seed stream, the transformer reaches the lowest bpb \u2014 but spends \u2248<strong>' + ratio + '\u00d7</strong> the FLOPs per byte of the cheapest online mixer. That gap is the <a href="' + LINK("concepts/loss-per-flop-and-scaling-laws") + '">loss-per-FLOP</a> tension the whole compendium measures. Edit the seed to feed the models your own bytes.';

    root.innerHTML =
      '<div class="race-head"><div class="race-title">The loss-per-FLOP race \u2014 three byte models read the same stream, live</div>' +
      '<div class="demo-transport"><button type="button" data-act="step">Step</button>' +
      '<button type="button" class="primary" data-act="play">Play</button>' +
      '<button type="button" data-act="reset">Reset</button><span class="race-pos"></span></div></div>' +
      '<label class="race-seed">editable seed \u2014 enwik8 @ 2,000,000 (the shared byte stream)' +
      '<textarea class="race-text" rows="3" spellcheck="false" aria-label="editable byte stream"></textarea></label>' +
      '<div class="race-context" aria-label="revealed context and cursor"></div>' +
      '<div class="race-grid">' + defs.map(function (def) { return '<div class="race-panel" data-model="' + def.id + '"></div>'; }).join("") + '</div>' +
      '<figcaption class="figcaption">' + caption + '</figcaption>';
    root.querySelector(".race-text").value = seedText;
    var transport = wireTransport(root, { step: step, reset: reset, atEnd: atEnd, interval: 140, onState: onState });
    var inputTimer = null;
    root.addEventListener("input", function (e) {
      if (!e.target.classList.contains("race-text")) return;
      clearTimeout(inputTimer);
      inputTimer = setTimeout(function () { transport.stop(); reset(); }, 300);
    });
    reset(); onState(false);
  }

  // ══ CursorChase (interactive chemotaxis cursor-follow) ════════════════════
  // A concentration field whose PEAK drifts toward the cursor's x at the env's
  // own rate (≤1 cell/tick, ChemoEnv's drift — so the peak never jumps out of an
  // organism's local sensing range; SmolDemos.concentration on the W-cell ring).
  // Each control model drives an organism that senses only
  // its own cell, quantized to a level, and chemotaxes to chase the peak: fold the
  // concentration token, take the greedy action argmax(logits[levels..levels+N)),
  // fold the action token, move, accrue reward = concentration at the new cell —
  // exactly parity.mjs's control tape (EVEN pos = sense, ODD = action). All three
  // step the same field every tick so the reward race is comparable; the toggles
  // only show/hide a marker. The HUD makes the FLOP/step contrast (66 vs ~10k) and
  // the honest tracking gap visible. Built once; the canvas redraws per tick.
  function mountCursorChase(root) {
    var SM = window.SmolModels, SD = window.SmolDemos;
    var d = readData(root), defs = d.models || [], env = d.env || { width: 16, levels: 8, sigma: 2 };
    if (!SM || !SD) { root.innerHTML = '<p class="demo-err">interactive model layer not loaded</p>'; return; }
    var W = env.width, L = env.levels, cfgF = { width: W, sigma: env.sigma }, N_ACT = SD.N_ACTIONS;
    var START = Math.floor(W / 2);
    var BLUE = ROLE_COLOR.reference, AMBER = "232,181,77";
    var CW = 720, CH = 210, PADL = 10, PADR = 10, FTOP = 30, FBOT = 138, LANE0 = 150, LANEH = 13, AXIS = 188;
    var iw = CW - PADL - PADR, cw = iw / W;
    function cellCX(x) { return PADL + (x + 0.5) * cw; }

    var shown = {};
    function fresh(def) {
      var opts = {};
      if (def.config) opts.config = def.config;
      if (def.weights) opts.weights = def.weights;
      return { def: def, model: SM[def.id], state: SM[def.id].create(opts), pos: 0, agentX: START, cumR: 0, steps: 0, lastA: 1 };
    }
    var runs = [], peakX = W - 2, cursorX = peakX, steps = 0, ctx = null, canvas = null;
    var announcedLeader = null, announcedAt = -999; // aria-live throttle state
    var RATE = 0.3, driftPhase = 0; // ChemoEnv drift pace (fixtures use 0.2–0.3): the integer peak jumps ±1 toward the cursor only ~every 1/RATE ticks, staying on a cell and within a local climber's sensing range
    function ringDelta(target, cur) { var dd = ((target - cur) % W + W) % W; if (dd > W / 2) dd -= W; return dd; } // shortest signed ring distance
    function atEnd() { return false; } // the chase is endless; Reset zeroes it
    function reset() { peakX = W - 2; cursorX = peakX; driftPhase = 0; runs = defs.map(fresh); steps = 0; var sl = root.querySelector(".chase-cursor"); if (sl) sl.value = cursorX; draw(); updateHud(); }
    // ChemoEnv order (smolml/envs/chemotaxis.py, parity.mjs): sense at the CURRENT
    // field, act, move the agent — THEN drift the shared peak once — THEN reward at
    // the new cell under the new field. (One shared cursor-driven peak, three racers.)
    function senseActMove(r) {
      var raw = SD.concentration(r.agentX, peakX, cfgF), level = SD.quantizeLevel(raw, L);
      var out = r.model.step(r.state, level, r.pos); r.state = out.state;
      var a = SD.argmax(out.logits, L, L + N_ACT) - L;
      r.pos++; out = r.model.step(r.state, L + a, r.pos); r.state = out.state; r.pos++;
      r.agentX = ((r.agentX + SD.ACTION_DELTAS[a]) % W + W) % W; r.lastA = a;
    }
    function step() {
      steps++;
      for (var i = 0; i < runs.length; i++) senseActMove(runs[i]);   // sense (current field) + act + move
      // THEN drift the peak toward your cursor with ChemoEnv's exact mechanic: an
      // integer peak that jumps ±1 only when a phase accumulator (RATE/tick) crosses
      // 1 — keeps the peak on a cell and within a local climber's sensing range.
      var tgt = ((Math.round(cursorX) % W) + W) % W, dd = ringDelta(tgt, peakX);
      if (dd !== 0) { driftPhase += RATE; if (driftPhase >= 1) { driftPhase -= 1; peakX = ((peakX + (dd > 0 ? 1 : -1)) % W + W) % W; } }
      else driftPhase = 0;
      for (var j = 0; j < runs.length; j++) {                         // THEN reward at the new cell under the new field
        var r = runs[j]; r.cumR += SD.concentration(r.agentX, peakX, cfgF); r.steps++;
      }
      draw(); updateHud();
    }

    function tri(cx, cy, r, up, color) {
      ctx.fillStyle = color; ctx.beginPath();
      if (up) { ctx.moveTo(cx, cy); ctx.lineTo(cx - r, cy + r * 1.3); ctx.lineTo(cx + r, cy + r * 1.3); }
      else { ctx.moveTo(cx, cy + r * 1.3); ctx.lineTo(cx - r, cy); ctx.lineTo(cx + r, cy); }
      ctx.closePath(); ctx.fill();
    }
    function draw() {
      if (!ctx) return;
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = "#16130e"; ctx.fillRect(0, 0, CW, CH);
      // cell tints — the quantized concentration each organism actually senses
      for (var x = 0; x < W; x++) {
        var c = SD.concentration(x, peakX, cfgF), a = Math.pow(Math.max(0, Math.min(1, c)), 0.55);
        ctx.fillStyle = "rgba(" + AMBER + "," + a.toFixed(3) + ")";
        ctx.fillRect(PADL + x * cw + 1, FTOP, cw - 2, FBOT - FTOP);
      }
      // continuous field curve (aligned to cell centers; wraps on the ring)
      ctx.beginPath();
      for (var s = 0; s <= W * 8; s++) {
        var xx = -0.5 + (s / 8) * 1.0; // -0.5 .. W-0.5
        var cc = SD.concentration(((xx % W) + W) % W, peakX, cfgF);
        var px = PADL + (xx + 0.5) * cw, py = FBOT - cc * (FBOT - FTOP);
        if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = "rgba(" + AMBER + ",0.9)"; ctx.lineWidth = 1.6; ctx.stroke();
      // the field peak (drifts toward your cursor at the env's rate)
      var pkx = PADL + (peakX + 0.5) * cw;
      ctx.strokeStyle = BLUE; ctx.lineWidth = 1.3; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(pkx, FTOP - 9); ctx.lineTo(pkx, FBOT); ctx.stroke(); ctx.setLineDash([]);
      tri(pkx, FTOP - 10, 6, false, BLUE);
      // your cursor target (faint) while the peak is still catching up
      if (Math.abs(ringDelta(cursorX, peakX)) > 0.4) {
        var cgx = PADL + (cursorX + 0.5) * cw;
        ctx.strokeStyle = "rgba(154,142,118,0.7)"; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);
        ctx.beginPath(); ctx.moveTo(cgx, FTOP - 9); ctx.lineTo(cgx, FBOT); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.fillStyle = "#9a8e76"; ctx.font = "11px ui-monospace, monospace"; ctx.textAlign = "left";
      ctx.fillText("peak \u25bc drifts toward your cursor", PADL + 2, 16);
      // organisms, one lane each
      for (var i = 0; i < runs.length; i++) {
        var r = runs[i]; if (shown[r.def.id] === false) continue;
        var col = ROLE_COLOR[r.def.role] || ROLE_COLOR.neutral, ly = LANE0 + i * LANEH;
        tri(cellCX(r.agentX), ly, 6, true, col);
      }
      // axis ticks
      ctx.fillStyle = "#9a8e76"; ctx.textAlign = "center";
      [0, 4, 8, 12, 15].forEach(function (cc2) { ctx.fillText("" + cc2, cellCX(cc2), AXIS); });
      ctx.fillText("ring cell 0\u201315 (wraps) \u2014 organisms (\u25b2) chase the peak (\u25bc)", CW / 2, CH - 6);
    }
    function hudRow(def) {
      var col = ROLE_COLOR[def.role] || ROLE_COLOR.neutral;
      return '<div class="chase-row" data-model="' + def.id + '">' +
        '<label class="chase-toggle"><input type="checkbox" data-show="' + def.id + '" checked><span class="chase-swatch" style="background:' + col + '"></span><span class="chase-name">' + def.label + '</span></label>' +
        '<span class="chase-cell"><span class="chase-cv">' + def.params.toLocaleString() + '</span><span class="chase-cl">params</span></span>' +
        '<span class="chase-cell" title="' + (def.flopsNote ? def.flopsNote.replace(/"/g, "&quot;") : "per control step") + '"><span class="chase-cv">' + def.flopsPerStep.toLocaleString() + '</span><span class="chase-cl">FLOPs/step</span></span>' +
        '<span class="chase-cell"><span class="chase-cv chase-mean" style="color:' + col + '">0.000</span><span class="chase-cl">mean reward \u2191</span></span>' +
        '<span class="chase-cell"><span class="chase-cv chase-cum">0.0</span><span class="chase-cl">cumulative</span></span></div>';
    }
    function updateHud() {
      var best = -1, bestId = null;
      runs.forEach(function (r) { var m = r.steps ? r.cumR / r.steps : 0; if (m > best) { best = m; bestId = r.def.id; } });
      runs.forEach(function (r) {
        var row = root.querySelector('.chase-row[data-model="' + r.def.id + '"]'); if (!row) return;
        row.querySelector(".chase-mean").textContent = (r.steps ? r.cumR / r.steps : 0).toFixed(3);
        row.querySelector(".chase-cum").textContent = r.cumR.toFixed(1);
        row.classList.toggle("dim", shown[r.def.id] === false);
        row.classList.toggle("lead", r.def.id === bestId && steps > 0);
      });
      var st = root.querySelector(".chase-steps"); if (st) st.textContent = steps + " steps";
      // aria-live summary: announce on leader change or every 30 steps (no SR spam)
      var live = root.querySelector(".chase-status");
      if (live && (bestId !== announcedLeader || steps - announcedAt >= 30 || steps === 0)) {
        announcedLeader = bestId; announcedAt = steps;
        if (steps === 0) { live.textContent = "Ready \u2014 press Play or Step; drag the peak-target slider or move your cursor."; }
        else {
          var leadDef = null; for (var di = 0; di < defs.length; di++) if (defs[di].id === bestId) leadDef = defs[di];
          var parts = runs.map(function (r) { return r.def.label + " " + (r.steps ? r.cumR / r.steps : 0).toFixed(2); });
          live.textContent = "After " + steps + " steps, leader: " + (leadDef ? leadDef.label + " (" + best.toFixed(2) + " mean reward, " + leadDef.flopsPerStep.toLocaleString() + " FLOPs/step)" : "\u2014") + ". Mean reward \u2014 " + parts.join(", ") + ".";
        }
      }
    }

    root.innerHTML =
      '<div class="chase-head"><div class="chase-title">Chase the cursor \u2014 three controllers, one peak you drive</div>' +
      '<div class="demo-transport"><button type="button" data-act="step">Step</button>' +
      '<button type="button" class="primary" data-act="play">Play</button>' +
      '<button type="button" data-act="reset">Reset</button><span class="chase-steps"></span></div></div>' +
      '<canvas class="chase-canvas" width="' + CW + '" height="' + CH + '" role="img" aria-label="Concentration field on a 16-cell ring; the peak drifts toward your cursor (or the peak-target slider) and the three organisms chase it"></canvas>' +
      '<label class="chase-cursor-l">peak target (ring cell)<input type="range" class="chase-cursor" min="0" max="' + W + '" step="0.5" value="' + (W - 2) + '" aria-label="peak target ring cell \u2014 drag or use arrow keys to move the peak"></label>' +
      '<p class="chase-hint">Move your cursor across the field \u2014 or drag the <strong>peak-target</strong> slider (arrow keys work) \u2014 and the peak (\u25bc) drifts toward it at the environment\u2019s own pace. Each organism (\u25b2) senses only the concentration at its own cell and chemotaxes to climb. Press <strong>Play</strong> (or Step) and watch who tracks.</p>' +
      '<div class="chase-hud">' + defs.map(hudRow).join("") + '</div>' +
      '<p class="chase-status" aria-live="polite"></p>' +
      '<figcaption class="figcaption">The minimal organism <strong>chemotaxis_min</strong> (5 params, <strong>66 FLOPs/step</strong>) typically tracks tightest; the reservoir family is <strong>~150\u00d7 heavier</strong> per step and \u2014 read honestly \u2014 a weaker tracker here, the loss-per-FLOP discipline carried from byte prediction to <a href="' + LINK("concepts/in-context-control") + '">control</a>. All three race the same field; the checkboxes just show or hide a marker.</figcaption>';
    canvas = root.querySelector(".chase-canvas"); ctx = canvas.getContext("2d");
    function setCursorFromEvent(e) {
      var rect = canvas.getBoundingClientRect();
      var clientX = e.touches && e.touches[0] ? e.touches[0].clientX : e.clientX;
      var xb = (clientX - rect.left) / rect.width * CW; // backing-store x
      var px = (xb - PADL) / cw - 0.5;
      cursorX = ((px % W) + W) % W;
      var sl = root.querySelector(".chase-cursor"); if (sl) sl.value = cursorX.toFixed(2);
      draw();
    }
    canvas.addEventListener("pointermove", setCursorFromEvent);
    canvas.addEventListener("touchmove", function (e) { setCursorFromEvent(e); e.preventDefault(); }, { passive: false });
    root.addEventListener("input", function (e) {
      if (!e.target.classList.contains("chase-cursor")) return;
      var v = parseFloat(e.target.value); if (isNaN(v)) return;
      cursorX = ((v % W) + W) % W; draw();
    });
    root.addEventListener("change", function (e) {
      var cb = e.target.closest("[data-show]"); if (!cb) return;
      shown[cb.getAttribute("data-show")] = cb.checked; draw(); updateHud();
    });
    wireTransport(root, { step: step, reset: reset, atEnd: atEnd, interval: 110, onState: function (p) { var pb = root.querySelector('[data-act="play"]'); if (pb) pb.textContent = p ? "Pause" : "Play"; } });
    reset();
  }

  // ── dispatch ──────────────────────────────────────────────────────────────
  var MOUNTERS = {
    chart: mountChart,
    prequential: function (r) { mountStream(r, "prequential"); },
    contextmixing: function (r) { mountStream(r, "contextmixing"); },
    fastweight: mountFastWeight,
    codelength: mountCodeLength,
    scaling: mountScaling,
    sourceiv: mountSourceIv,
    controlrollout: mountControlRollout,
    autocompleterace: mountAutocompleteRace,
    cursorchase: mountCursorChase
  };
  function init() {
    var nodes = document.querySelectorAll("[data-widget]");
    for (var i = 0; i < nodes.length; i++) {
      var root = nodes[i];
      if (root.getAttribute("data-mounted")) continue;
      var fn = MOUNTERS[root.getAttribute("data-widget")];
      if (fn) { root.setAttribute("data-mounted", "1"); try { fn(root); } catch (e) { console.error("[compendium] mount failed", e); } }
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
