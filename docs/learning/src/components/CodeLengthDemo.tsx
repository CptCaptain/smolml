/** @jsxImportSource preact */
import { useState } from "preact/hooks";

// Compression = prediction, made tangible: a probability the model assigns to
// the *true* next byte maps to a code length of -log2(p) bits. Drag the
// confidence and watch the bill. Below p = 1/256 you spend MORE than 8 bits —
// a model worse than the uniform "no model".

const W = 560;
const Hh = 240;
const PAD = { l: 46, r: 16, t: 16, b: 34 };
const bits = (p: number) => -Math.log2(p);

export default function CodeLengthDemo() {
  const [p, setP] = useState(0.5);
  const cost = bits(p);
  const saved = 8 - cost; // vs the 8-bit uniform "no model"

  // -log2(p) curve over p in (0,1]
  const x0 = PAD.l;
  const x1 = W - PAD.r;
  const y0 = PAD.t;
  const y1 = Hh - PAD.b;
  const CAP = 12; // y-axis caps at 12 bits
  const px = (pp: number) => x0 + pp * (x1 - x0);
  const py = (b: number) => y1 - (Math.min(b, CAP) / CAP) * (y1 - y0);

  const path: string[] = [];
  for (let i = 1; i <= 200; i++) {
    const pp = i / 200;
    path.push(`${px(pp).toFixed(1)},${py(bits(pp)).toFixed(1)}`);
  }

  return (
    <figure class="cld">
      <div class="cld-grid">
        <div class="cld-readout">
          <div class="cld-bignum">
            <span class="cld-val">{cost.toFixed(3)}</span>
            <span class="cld-unit">bits / byte</span>
          </div>
          <p class="cld-eq">
            &minus;log&#8322;({p.toFixed(2)}) = {cost.toFixed(3)}
          </p>
          <label class="cld-label">
            model&rsquo;s probability on the <em>true</em> next byte
            <input
              type="range"
              min="0.01"
              max="0.99"
              step="0.01"
              value={p}
              onInput={(e) => setP(parseFloat((e.target as HTMLInputElement).value))}
            />
            <output>p = {p.toFixed(2)}</output>
          </label>
          <p class={saved >= 0 ? "cld-note good" : "cld-note bad"}>
            {saved >= 0
              ? `${saved.toFixed(2)} bits cheaper than the 8-bit "no model".`
              : `${(-saved).toFixed(2)} bits worse than no model — being confidently wrong costs.`}
          </p>
        </div>

        <svg viewBox={`0 0 ${W} ${Hh}`} class="cld-svg" role="img" aria-label="Code length minus log2 p versus probability">
          {/* 8-bit no-model reference */}
          <line x1={x0} x2={x1} y1={py(8)} y2={py(8)} class="cld-ref" />
          <text x={x1 - 4} y={py(8) - 5} class="cld-reftext" text-anchor="end">
            8 bits — no model
          </text>
          {/* axes */}
          <line x1={x0} x2={x1} y1={y1} y2={y1} class="cld-axis" />
          <line x1={x0} x2={x0} y1={y0} y2={y1} class="cld-axis" />
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <text x={px(t)} y={y1 + 18} class="cld-tick" text-anchor="middle">
              {t}
            </text>
          ))}
          {[0, 4, 8, 12].map((b) => (
            <text x={x0 - 6} y={py(b) + 4} class="cld-tick" text-anchor="end">
              {b}
            </text>
          ))}
          <text x={(x0 + x1) / 2} y={Hh - 4} class="cld-axislabel" text-anchor="middle">
            p(true next byte)
          </text>
          {/* curve */}
          <polyline points={path.join(" ")} fill="none" stroke="#5ea8e6" stroke-width="2.2" />
          {/* current point */}
          <line x1={px(p)} x2={px(p)} y1={py(cost)} y2={y1} class="cld-drop" />
          <circle cx={px(p)} cy={py(cost)} r="6" fill="#e8b54d" stroke="#16130e" stroke-width="1.5" />
        </svg>
      </div>
      <figcaption class="figcaption">
        Arithmetic coding stores the true next byte in <strong>&minus;log&#8322;&nbsp;p</strong> bits.
        Confident and right &rarr; nearly free; surprised &rarr; expensive; confidently wrong (small
        p) &rarr; worse than the 8-bit uniform prior. Sum this over a stream &divide; bytes = bpb.
      </figcaption>

      <style>{`
        .cld { margin: 1.8em 0; border: 1px solid var(--line); border-radius: var(--radius-lg);
          background: var(--ink-1); overflow: hidden; }
        .cld-grid { display: grid; grid-template-columns: 14rem 1fr; gap: 1rem; padding: 1.1rem 1.2rem; }
        .cld-readout { display: flex; flex-direction: column; gap: 0.5rem; }
        .cld-bignum { display: flex; align-items: baseline; gap: 0.4rem; }
        .cld-val { font-family: var(--font-mono); font-size: 2.3rem; font-weight: 700; color: #e8b54d; line-height: 1; }
        .cld-unit { font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted); }
        .cld-eq { font-family: var(--font-mono); font-size: 0.82rem; color: var(--paper-dim); margin: 0; }
        .cld-label { font-size: 0.86rem; color: var(--paper-dim); display: flex; flex-direction: column; gap: 0.35rem; }
        .cld-label em { color: var(--accent); font-style: normal; }
        .cld-label input { width: 100%; accent-color: #e8b54d; }
        .cld-label output { font-family: var(--font-mono); font-size: 0.8rem; color: var(--paper); }
        .cld-note { font-family: var(--font-mono); font-size: 0.76rem; margin: 0.2rem 0 0; }
        .cld-note.good { color: #5cc46a; }
        .cld-note.bad { color: #f0913e; }
        .cld-svg { width: 100%; height: auto; font-family: var(--font-mono); }
        .cld-axis { stroke: #5a5240; stroke-width: 1.2; }
        .cld-ref { stroke: #6f6650; stroke-dasharray: 3 4; stroke-width: 1.2; }
        .cld-reftext { fill: #9a8e76; font-size: 11px; }
        .cld-tick { fill: #9a8e76; font-size: 11px; }
        .cld-axislabel { fill: #c8bca4; font-size: 12px; }
        .cld-drop { stroke: #e8b54d; stroke-width: 1; stroke-dasharray: 2 3; opacity: 0.6; }
        @media (max-width: 560px) { .cld-grid { grid-template-columns: 1fr; } }
      `}</style>
    </figure>
  );
}
