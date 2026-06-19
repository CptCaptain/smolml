/** @jsxImportSource preact */
import { useState } from "preact/hooks";

// Fast-weight associative memory, hands-on. A d x V store M holds key->value
// associations written by a single gradient-free outer product (M <- decay*M +
// key (x) e_value) and read by a matvec (q @ M -> logits -> softmax). Write an
// association and it is recalled instantly; write a *similar* key and watch
// crosstalk; turn down decay and watch old associations fade. No backprop.

const D = 6;
const BYTES = ["e", "a", "t", "o", "n"];
const V = BYTES.length;

interface Key {
  label: string;
  hint: string;
  vec: number[];
}
function norm(v: number[]): number[] {
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
}
const KEYS: Key[] = [
  { label: "ctx \u03b1", hint: "distinct", vec: norm([1, 0.15, 0, 0, 0.1, 0]) },
  { label: "ctx \u03b2", hint: "distinct", vec: norm([0.1, 1, 0.1, 0, 0, 0]) },
  { label: "ctx \u03b3", hint: "distinct", vec: norm([0, 0, 1, 0.15, 0, 0.12]) },
  { label: "ctx \u03b2\u2032", hint: "looks like \u03b2 \u2192 crosstalk", vec: norm([0.18, 0.92, 0, 0.32, 0, 0]) },
];

const zeros = () => Array.from({ length: D }, () => Array.from({ length: V }, () => 0));

function readDist(M: number[][], q: number[]): number[] {
  const logits = Array.from({ length: V }, (_, j) => {
    let s = 0;
    for (let i = 0; i < D; i++) s += q[i] * M[i][j];
    return s;
  });
  const beta = 3.0;
  const mx = Math.max(...logits);
  const ex = logits.map((l) => Math.exp(beta * (l - mx)));
  const z = ex.reduce((a, b) => a + b, 0);
  return ex.map((e) => e / z);
}

export default function FastWeightDemo() {
  const [M, setM] = useState<number[][]>(zeros);
  const [decay, setDecay] = useState(0.92);
  const [selKey, setSelKey] = useState(0);
  const [selVal, setSelVal] = useState(2); // 't'
  const [query, setQuery] = useState(0);
  const [log, setLog] = useState<string[]>([]);

  const write = (ki: number, vi: number) => {
    const k = KEYS[ki].vec;
    setM((prev) => {
      const next = prev.map((row) => row.map((x) => x * decay));
      for (let i = 0; i < D; i++) next[i][vi] += k[i];
      return next;
    });
    setLog((l) => [`write  ${KEYS[ki].label} \u2192 '${BYTES[vi]}'`, ...l].slice(0, 5));
  };
  const fade = () => {
    setM((prev) => prev.map((row) => row.map((x) => x * decay ** 6)));
    setLog((l) => [`fade   \u00d7 decay\u2076 (forgetting)`, ...l].slice(0, 5));
  };
  const reset = () => {
    setM(zeros());
    setLog([]);
  };
  const seed = () => {
    let next = zeros();
    const seq: [number, number][] = [[0, 2], [1, 0], [2, 4]]; // a->t, b->e, g->n
    for (const [ki, vi] of seq) {
      next = next.map((row) => row.map((x) => x * decay));
      for (let i = 0; i < D; i++) next[i][vi] += KEYS[ki].vec[i];
    }
    setM(next);
    setLog(["seed   \u03b1\u2192't', \u03b2\u2192'e', \u03b3\u2192'n'"]);
  };

  const dist = readDist(M, KEYS[query].vec);
  const argmax = dist.indexOf(Math.max(...dist));
  const maxAbs = Math.max(0.4, ...M.flat().map((x) => Math.abs(x)));

  const cellColor = (x: number) => {
    const a = Math.min(1, Math.abs(x) / maxAbs);
    return x >= 0
      ? `rgba(232,181,77,${a.toFixed(3)})`
      : `rgba(94,168,230,${a.toFixed(3)})`;
  };

  return (
    <figure class="fw">
      <div class="fw-top">
        {/* write controls */}
        <div class="fw-controls">
          <p class="fw-h">write an association</p>
          <div class="fw-keys">
            {KEYS.map((k, i) => (
              <button type="button" class={`fw-key ${selKey === i ? "on" : ""}`} onClick={() => setSelKey(i)}>
                {k.label}
                <span class="fw-key-hint">{k.hint}</span>
              </button>
            ))}
          </div>
          <p class="fw-sub">value (next byte)</p>
          <div class="fw-vals">
            {BYTES.map((b, i) => (
              <button type="button" class={`fw-val ${selVal === i ? "on" : ""}`} onClick={() => setSelVal(i)}>
                {b}
              </button>
            ))}
          </div>
          <div class="fw-actions">
            <button type="button" class="primary" onClick={() => write(selKey, selVal)}>
              Write &nbsp;{KEYS[selKey].label} &rarr; &lsquo;{BYTES[selVal]}&rsquo;
            </button>
          </div>
          <label class="fw-decay">
            memory decay&nbsp;<output>{decay.toFixed(2)}</output>
            <input type="range" min="0.80" max="1.00" step="0.01" value={decay}
              onInput={(e) => setDecay(parseFloat((e.target as HTMLInputElement).value))} />
          </label>
          <div class="fw-actions">
            <button type="button" onClick={seed}>Seed 3</button>
            <button type="button" onClick={fade}>Let it fade</button>
            <button type="button" onClick={reset}>Reset</button>
          </div>
          {log.length > 0 && (
            <ul class="fw-log">
              {log.map((l) => <li>{l}</li>)}
            </ul>
          )}
        </div>

        {/* memory matrix heatmap */}
        <div class="fw-matrix">
          <p class="fw-h">memory M &nbsp;<span class="fw-dim">(d&times;V = {D}&times;{V})</span></p>
          <div class="fw-grid" style={{ gridTemplateColumns: `1.4rem repeat(${V}, 1fr)` }}>
            <span class="fw-corner" />
            {BYTES.map((b) => <span class="fw-col-label">{b}</span>)}
            {M.map((row, i) => (
              <>
                <span class="fw-row-label">d{i}</span>
                {row.map((x) => (
                  <span class="fw-cell" style={{ background: cellColor(x) }} title={x.toFixed(2)} />
                ))}
              </>
            ))}
          </div>
          <p class="fw-mathnote">
            write: <code>M &larr; decay&middot;M + key &otimes; e_byte</code> &middot; read:{" "}
            <code>q @ M &rarr; softmax</code>
          </p>
        </div>
      </div>

      {/* read panel */}
      <div class="fw-read">
        <div class="fw-read-head">
          <span class="fw-h">read: query with</span>
          <div class="fw-keys inline">
            {KEYS.map((k, i) => (
              <button type="button" class={`fw-key sm ${query === i ? "on" : ""}`} onClick={() => setQuery(i)}>
                {k.label}
              </button>
            ))}
          </div>
        </div>
        <div class="fw-bars">
          {BYTES.map((b, i) => (
            <div class={`fw-bar ${i === argmax ? "best" : ""}`}>
              <span class="fw-bar-ch">{b}</span>
              <span class="fw-bar-track"><span class="fw-bar-fill" style={{ width: `${dist[i] * 100}%` }} /></span>
              <span class="fw-bar-p">{(dist[i] * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
        <p class="fw-recall">
          recalled &rarr; <strong>&lsquo;{BYTES[argmax]}&rsquo;</strong> at {(dist[argmax] * 100).toFixed(0)}% confidence
        </p>
      </div>

      <figcaption class="figcaption">
        The write is one outer product &mdash; instant, <strong>O(1)</strong>, no gradient. Querying
        &alpha;/&beta;/&gamma; recalls their stored byte; querying <strong>&beta;&prime;</strong>
        (which points almost the same way as &beta;) recalls a blurred mix &mdash; that is the
        <em> crosstalk</em> of superposing associations in one matrix. Drop the decay and write more
        to watch old associations <em>forget</em>.
      </figcaption>

      <style>{`
        .fw { margin: 1.8em 0; border: 1px solid var(--line); border-radius: var(--radius-lg);
          background: var(--ink-1); padding: 1.1rem 1.2rem; }
        .fw-top { display: grid; grid-template-columns: 1fr 1fr; gap: 1.4rem; }
        .fw-h { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.12em;
          text-transform: uppercase; color: var(--muted); margin: 0 0 0.6rem; }
        .fw-dim { color: var(--faint); text-transform: none; letter-spacing: 0; }
        .fw-sub { font-family: var(--font-mono); font-size: 0.7rem; color: var(--muted); margin: 0.7rem 0 0.4rem; }
        .fw-keys { display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .fw-keys.inline { display: inline-flex; }
        .fw-key { display: flex; flex-direction: column; align-items: flex-start; font-family: var(--font-mono);
          font-size: 0.78rem; cursor: pointer; background: var(--ink-2); color: var(--paper-dim);
          border: 1px solid var(--line); border-radius: 6px; padding: 0.35rem 0.55rem; }
        .fw-key.sm { flex-direction: row; }
        .fw-key-hint { font-size: 0.6rem; color: var(--faint); }
        .fw-key.on { border-color: var(--c-fast); color: #f0913e; }
        .fw-vals { display: flex; gap: 0.35rem; }
        .fw-val { font-family: var(--font-mono); width: 2rem; height: 2rem; cursor: pointer;
          background: var(--ink-2); color: var(--paper-dim); border: 1px solid var(--line); border-radius: 6px; }
        .fw-val.on { border-color: var(--accent); color: var(--accent); }
        .fw-actions { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.7rem; }
        .fw-actions button, .fw-decay { font-family: var(--font-mono); font-size: 0.76rem; }
        .fw-actions button { cursor: pointer; background: var(--ink-2); color: var(--paper);
          border: 1px solid var(--line); border-radius: 5px; padding: 0.35rem 0.7rem; }
        .fw-actions button.primary { background: rgba(240,145,62,0.14); color: #f0913e; border-color: #b6692a; }
        .fw-actions button:hover { border-color: var(--accent); }
        .fw-decay { display: flex; align-items: center; gap: 0.4rem; margin-top: 0.8rem; color: var(--paper-dim); }
        .fw-decay input { flex: 1; accent-color: #f0913e; }
        .fw-decay output { color: #f0913e; }
        .fw-log { list-style: none; margin: 0.7rem 0 0; padding: 0; font-family: var(--font-mono);
          font-size: 0.7rem; color: var(--muted); }
        .fw-log li { padding: 0.05rem 0; }
        .fw-grid { display: grid; gap: 2px; }
        .fw-corner, .fw-col-label, .fw-row-label { font-family: var(--font-mono); font-size: 0.7rem; color: var(--muted);
          display: grid; place-items: center; }
        .fw-col-label { color: var(--paper-dim); }
        .fw-cell { aspect-ratio: 1; border-radius: 2px; border: 1px solid rgba(58,51,38,0.6);
          background: var(--ink); min-height: 1.5rem; }
        .fw-mathnote { font-family: var(--font-mono); font-size: 0.68rem; color: var(--muted); margin: 0.7rem 0 0; line-height: 1.5; }
        .fw-read { margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid var(--line); }
        .fw-read-head { display: flex; flex-wrap: wrap; align-items: center; gap: 0.7rem; margin-bottom: 0.6rem; }
        .fw-read-head .fw-h { margin: 0; }
        .fw-bars { display: flex; flex-direction: column; gap: 0.25rem; }
        .fw-bar { display: grid; grid-template-columns: 1.4rem 1fr 2.6rem; align-items: center; gap: 0.5rem; }
        .fw-bar-ch { font-family: var(--font-mono); color: var(--paper-dim); text-align: center; }
        .fw-bar-track { height: 0.8rem; background: var(--ink-2); border-radius: 3px; overflow: hidden; }
        .fw-bar-fill { display: block; height: 100%; background: #8a6a3a; }
        .fw-bar.best .fw-bar-fill { background: #f0913e; }
        .fw-bar.best .fw-bar-ch { color: #f0913e; font-weight: 700; }
        .fw-bar-p { font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted); text-align: right; }
        .fw-recall { font-family: var(--font-mono); font-size: 0.82rem; color: var(--paper-dim); margin: 0.6rem 0 0; }
        .fw-recall strong { color: #f0913e; }
        @media (max-width: 620px) { .fw-top { grid-template-columns: 1fr; } }
      `}</style>
    </figure>
  );
}
