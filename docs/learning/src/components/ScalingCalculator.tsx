/** @jsxImportSource preact */
import { useState } from "preact/hooks";

// "Needs billions" is arithmetic, not psychology: C ~= 6 * N * D FLOPs, and the
// wall-clock at a fixed device throughput follows. Drag N and D (log scale),
// pick hardware, and watch a laptop-century appear. Presets anchor the extremes.

const HARDWARE: { label: string; flops: number }[] = [
  { label: "laptop GPU (~2\u00d710\u00b9\u00b3)", flops: 2e13 },
  { label: "A100 (~3\u00d710\u00b9\u2074)", flops: 3.1e14 },
  { label: "H100 (~1\u00d710\u00b9\u2075)", flops: 1e15 },
];

const PRESETS: { label: string; n: number; d: number }[] = [
  { label: "GPT-3", n: 175e9, d: 300e9 },
  { label: "smolml d32 baseline", n: 32928, d: 1.5e6 },
];

const SUP: Record<string, string> = {
  "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074",
  "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079",
  "-": "\u207b", "+": "",
};
const sci = (x: number) => {
  if (x === 0) return "0";
  const e = Math.floor(Math.log10(x));
  const m = x / 10 ** e;
  const exp = String(e).split("").map((c) => SUP[c] ?? c).join("");
  return `${m.toFixed(2)}\u00d710${exp}`;
};

function humanTime(seconds: number): string {
  const units: [string, number][] = [
    ["year", 365.25 * 86400],
    ["day", 86400],
    ["hour", 3600],
    ["minute", 60],
    ["second", 1],
  ];
  for (const [name, s] of units) {
    if (seconds >= s) {
      const v = seconds / s;
      return `${v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(1)} ${name}${v >= 1.5 ? "s" : ""}`;
    }
  }
  return `${seconds.toExponential(1)} s`;
}

// log10 slider plumbing
const fromLog = (l: number) => 10 ** l;
const toLog = (v: number) => Math.log10(v);

export default function ScalingCalculator() {
  const [logN, setLogN] = useState(toLog(175e9));
  const [logD, setLogD] = useState(toLog(300e9));
  const [hw, setHw] = useState(0);

  const N = fromLog(logN);
  const D = fromLog(logD);
  const C = 6 * N * D;
  const seconds = C / HARDWARE[hw].flops;
  const gpt3C = 6 * 175e9 * 300e9;
  const frac = C / gpt3C;

  return (
    <figure class="sc">
      <div class="sc-head">
        <div class="sc-eq">
          C = 6 &middot; N &middot; D = <strong>{sci(C)}</strong> FLOPs
        </div>
        <div class="sc-time">
          &asymp; <strong>{humanTime(seconds)}</strong> at 100% util on {HARDWARE[hw].label}
        </div>
      </div>

      <label class="sc-row">
        <span class="sc-name">N — parameters</span>
        <input type="range" min={3} max={12} step={0.05} value={logN}
          onInput={(e) => setLogN(parseFloat((e.target as HTMLInputElement).value))} />
        <span class="sc-val">{sci(N)}</span>
      </label>
      <label class="sc-row">
        <span class="sc-name">D — tokens seen</span>
        <input type="range" min={4} max={12} step={0.05} value={logD}
          onInput={(e) => setLogD(parseFloat((e.target as HTMLInputElement).value))} />
        <span class="sc-val">{sci(D)}</span>
      </label>
      <label class="sc-row">
        <span class="sc-name">hardware</span>
        <select value={hw} onChange={(e) => setHw(parseInt((e.target as HTMLSelectElement).value))}>
          {HARDWARE.map((h, i) => (
            <option value={i}>{h.label}</option>
          ))}
        </select>
        <span class="sc-val">{sci(HARDWARE[hw].flops)} FLOP/s</span>
      </label>

      <div class="sc-bar" aria-hidden="true">
        <div class="sc-bar-fill" style={{ width: `${Math.max(1, Math.min(100, (toLog(Math.max(C, 1)) / toLog(gpt3C)) * 100))}%` }} />
        <span class="sc-bar-label">{frac >= 1 ? `${frac.toFixed(1)}\u00d7 GPT-3` : `${(frac * 100).toExponential(1)}% of GPT-3`}</span>
      </div>

      <div class="sc-presets">
        <span>presets:</span>
        {PRESETS.map((p) => (
          <button type="button" onClick={() => { setLogN(toLog(p.n)); setLogD(toLog(p.d)); }}>
            {p.label}
          </button>
        ))}
      </div>

      <figcaption class="figcaption">
        The transformer "fits on a napkin", but the <em>training</em> cost is set by C = 6&middot;N&middot;D.
        GPT-3 (N&asymp;1.75&times;10&sup1;&sup1;, D&asymp;3&times;10&sup1;&sup1;) lands at &asymp;3&times;10&sup2;&sup3; FLOPs —
        centuries on a laptop. Simplicity of the kernel &ne; cheapness of training; that category
        error is the whole trap.
      </figcaption>

      <style>{`
        .sc { margin: 1.8em 0; border: 1px solid var(--line); border-radius: var(--radius-lg);
          background: var(--ink-1); padding: 1.1rem 1.2rem; }
        .sc-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.5rem;
          padding-bottom: 0.8rem; margin-bottom: 0.8rem; border-bottom: 1px solid var(--line);
          font-family: var(--font-mono); }
        .sc-eq { font-size: 1.05rem; color: var(--paper); }
        .sc-eq strong { color: #e8b54d; }
        .sc-time { font-size: 0.92rem; color: var(--paper-dim); }
        .sc-time strong { color: #f0913e; }
        .sc-row { display: grid; grid-template-columns: 9rem 1fr 8.5rem; align-items: center;
          gap: 0.7rem; margin: 0.5rem 0; }
        .sc-name { font-size: 0.86rem; color: var(--paper-dim); }
        .sc-row input[type=range] { width: 100%; accent-color: #e8b54d; }
        .sc-row select { background: var(--ink-2); color: var(--paper); border: 1px solid var(--line);
          border-radius: 5px; padding: 0.25rem 0.4rem; font-family: var(--font-body); }
        .sc-val { font-family: var(--font-mono); font-size: 0.82rem; color: var(--paper); text-align: right; }
        .sc-bar { position: relative; height: 1.5rem; margin: 1rem 0 0.4rem; background: var(--ink-2);
          border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }
        .sc-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent-deep), var(--accent)); }
        .sc-bar-label { position: absolute; inset: 0; display: grid; place-items: center;
          font-family: var(--font-mono); font-size: 0.74rem; color: #16130e; mix-blend-mode: screen; }
        .sc-presets { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem;
          margin-top: 0.7rem; font-family: var(--font-mono); font-size: 0.78rem; color: var(--muted); }
        .sc-presets button { background: var(--ink-2); color: var(--accent); border: 1px solid var(--line);
          border-radius: 5px; padding: 0.25rem 0.6rem; cursor: pointer; font-family: inherit; font-size: 0.78rem; }
        .sc-presets button:hover { border-color: var(--accent-deep); background: var(--ink-3); }
        @media (max-width: 560px) { .sc-row { grid-template-columns: 1fr; gap: 0.2rem; } .sc-val { text-align: left; } }
      `}</style>
    </figure>
  );
}
