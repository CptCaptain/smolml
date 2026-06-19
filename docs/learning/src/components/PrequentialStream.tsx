/** @jsxImportSource preact */
import { useEffect, useRef, useState } from "preact/hooks";

// The prequential loop, alive: a *real* online model (order-0 + order-1 byte
// counts, Laplace-smoothed) predicts the next byte BEFORE it is revealed, pays
// -log2 p bits, then folds the truth into its counts and moves on. The running
// bpb visibly drops as the model learns the stream — no held-out split, and a
// memorizer can't cheat the future because every prediction precedes its reveal.

const STREAM = "the cat sat on the mat. the cat ate the rat. ";
const ALPHA = [...new Set(STREAM.split(""))].sort();
const A = ALPHA.length;
const show = (c: string) => (c === " " ? "\u2423" : c);

interface State {
  pos: number;
  c0: Record<string, number>;
  c1: Record<string, Record<string, number>>;
  cumBits: number;
  history: number[]; // per-byte bits
}

function fresh(): State {
  const c0: Record<string, number> = {};
  for (const ch of ALPHA) c0[ch] = 0;
  return { pos: 0, c0, c1: {}, cumBits: 0, history: [] };
}

function predict(s: State): { ch: string; p: number }[] {
  const prev = s.pos > 0 ? STREAM[s.pos - 1] : null;
  const tot0 = Object.values(s.c0).reduce((a, b) => a + b, 0);
  const row = prev != null ? s.c1[prev] : undefined;
  const tot1 = row ? Object.values(row).reduce((a, b) => a + b, 0) : 0;
  return ALPHA.map((ch) => {
    const p0 = (s.c0[ch] + 1) / (tot0 + A);
    const p1 = row ? (((row[ch] ?? 0) + 1) / (tot1 + A)) : 1 / A;
    return { ch, p: 0.5 * p0 + 0.5 * p1 };
  }).sort((a, b) => b.p - a.p);
}

function advance(s: State): State {
  const truth = STREAM[s.pos];
  const dist = predict(s);
  const pTrue = dist.find((d) => d.ch === truth)!.p;
  const bitsPaid = -Math.log2(pTrue);
  const prev = s.pos > 0 ? STREAM[s.pos - 1] : null;
  const c0 = { ...s.c0, [truth]: s.c0[truth] + 1 };
  const c1 = { ...s.c1 };
  if (prev != null) {
    c1[prev] = { ...(c1[prev] ?? {}), [truth]: (c1[prev]?.[truth] ?? 0) + 1 };
  }
  return {
    pos: s.pos + 1,
    c0,
    c1,
    cumBits: s.cumBits + bitsPaid,
    history: [...s.history, bitsPaid],
  };
}

export default function PrequentialStream() {
  const [s, setS] = useState<State>(fresh);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  const done = s.pos >= STREAM.length;
  const dist = done ? [] : predict(s);
  const truth = done ? null : STREAM[s.pos];
  const bytes = s.pos;
  const bpb = bytes > 0 ? s.cumBits / bytes : 0;
  const nextBits = truth ? -Math.log2(dist.find((d) => d.ch === truth)!.p) : 0;

  useEffect(() => {
    if (!playing) return;
    if (done) {
      setPlaying(false);
      return;
    }
    timer.current = window.setTimeout(() => setS((prev) => advance(prev)), 520);
    return () => clearTimeout(timer.current);
  }, [playing, s, done]);

  const top = dist.slice(0, 6);
  const maxP = top.length ? top[0].p : 1;

  return (
    <figure class="ps">
      {/* stream tape */}
      <div class="ps-tape" aria-label="prediction stream">
        {STREAM.split("").map((ch, i) => (
          <span
            class={
              i < s.pos ? "ps-ch seen" : i === s.pos ? "ps-ch cursor" : "ps-ch future"
            }
          >
            {i === s.pos && !done ? "\u25af" : show(ch)}
          </span>
        ))}
      </div>

      <div class="ps-main">
        {/* predicted distribution */}
        <div class="ps-dist">
          <p class="ps-h">
            {done ? "stream complete" : "model\u2019s p(next byte) \u2014 before reveal"}
          </p>
          {top.map((d) => {
            const isTrue = d.ch === truth;
            return (
              <div class={isTrue ? "ps-bar istrue" : "ps-bar"}>
                <span class="ps-bar-ch">{show(d.ch)}</span>
                <span class="ps-bar-track">
                  <span class="ps-bar-fill" style={{ width: `${(d.p / maxP) * 100}%` }} />
                </span>
                <span class="ps-bar-p">{(d.p * 100).toFixed(1)}%</span>
              </div>
            );
          })}
          {!done && (
            <p class="ps-pending">
              truth is <code>{show(truth!)}</code> &rarr; pays{" "}
              <strong>{nextBits.toFixed(2)}</strong> bits
            </p>
          )}
        </div>

        {/* readout */}
        <div class="ps-readout">
          <div class="ps-metric">
            <span class="ps-metric-v">{bpb.toFixed(3)}</span>
            <span class="ps-metric-l">running bpb</span>
          </div>
          <div class="ps-sub">
            <span>{s.cumBits.toFixed(1)} bits</span>
            <span>&divide; {bytes} bytes</span>
          </div>
          {/* per-byte bits history sparkline */}
          <svg viewBox="0 0 220 56" class="ps-spark" role="img" aria-label="bits paid per byte">
            <line x1="0" y1="55" x2="220" y2="55" stroke="#3a3326" />
            {s.history.map((b, i) => {
              const w = 220 / STREAM.length;
              const h = Math.min(b, 9) / 9 * 50;
              return <rect x={i * w} y={55 - h} width={Math.max(1, w - 0.6)} height={h} fill="#5ea8e6" />;
            })}
          </svg>
          <p class="ps-spark-cap">bits paid per byte (it drops as the model learns)</p>
        </div>
      </div>

      <div class="ps-controls">
        <button type="button" onClick={() => setS((p) => (p.pos < STREAM.length ? advance(p) : p))} disabled={done}>
          Step
        </button>
        <button type="button" class="primary" onClick={() => setPlaying((p) => !p)} disabled={done}>
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={() => { setPlaying(false); setS(fresh()); }}>
          Reset
        </button>
      </div>

      <figcaption class="figcaption">
        Every prediction is made <em>before</em> the byte is revealed, so a model that memorizes the
        past cannot cheat the future — an honest generalization measure with no held-out split. The
        cumulative bits equal the compressed length, tying straight back to
        <a href="/concepts/compression-equals-prediction"> compression = prediction</a>.
      </figcaption>

      <style>{`
        .ps { margin: 1.8em 0; border: 1px solid var(--line); border-radius: var(--radius-lg);
          background: var(--ink-1); padding: 1.1rem 1.2rem; }
        .ps-tape { font-family: var(--font-mono); font-size: 1.15rem; letter-spacing: 0.04em;
          background: var(--ink); border: 1px solid var(--line); border-radius: var(--radius);
          padding: 0.7rem 0.8rem; line-height: 1.8; word-break: break-all; }
        .ps-ch.seen { color: var(--paper); }
        .ps-ch.future { color: var(--faint); }
        .ps-ch.cursor { color: #e8b54d; background: rgba(232,181,77,0.18); border-radius: 3px; padding: 0 1px; }
        .ps-main { display: grid; grid-template-columns: 1fr 13rem; gap: 1.2rem; margin-top: 1rem; }
        .ps-h { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.12em;
          text-transform: uppercase; color: var(--muted); margin: 0 0 0.5rem; }
        .ps-bar { display: grid; grid-template-columns: 1.4rem 1fr 3rem; align-items: center; gap: 0.5rem; margin: 0.25rem 0; }
        .ps-bar-ch { font-family: var(--font-mono); color: var(--paper-dim); text-align: center; }
        .ps-bar-track { height: 0.85rem; background: var(--ink-2); border-radius: 3px; overflow: hidden; }
        .ps-bar-fill { display: block; height: 100%; background: #4c7da8; }
        .ps-bar-p { font-family: var(--font-mono); font-size: 0.74rem; color: var(--muted); text-align: right; }
        .ps-bar.istrue .ps-bar-fill { background: #5cc46a; }
        .ps-bar.istrue .ps-bar-ch { color: #5cc46a; font-weight: 700; }
        .ps-pending { font-family: var(--font-mono); font-size: 0.78rem; color: var(--paper-dim); margin: 0.5rem 0 0; }
        .ps-pending strong { color: #e8b54d; }
        .ps-readout { border-left: 1px solid var(--line); padding-left: 1rem; }
        .ps-metric { display: flex; flex-direction: column; }
        .ps-metric-v { font-family: var(--font-mono); font-size: 2rem; font-weight: 700; color: #e8b54d; line-height: 1; }
        .ps-metric-l { font-family: var(--font-mono); font-size: 0.7rem; color: var(--muted); }
        .ps-sub { display: flex; justify-content: space-between; font-family: var(--font-mono);
          font-size: 0.74rem; color: var(--paper-dim); margin: 0.5rem 0; }
        .ps-spark { width: 100%; height: auto; margin-top: 0.4rem; }
        .ps-spark-cap { font-family: var(--font-mono); font-size: 0.66rem; color: var(--muted); margin: 0.2rem 0 0; line-height: 1.3; }
        .ps-controls { display: flex; gap: 0.5rem; margin-top: 1rem; }
        .ps-controls button { font-family: var(--font-mono); font-size: 0.8rem; cursor: pointer;
          background: var(--ink-2); color: var(--paper); border: 1px solid var(--line);
          border-radius: 5px; padding: 0.4rem 0.9rem; }
        .ps-controls button.primary { background: var(--accent-glow); color: var(--accent); border-color: var(--accent-deep); }
        .ps-controls button:hover:not(:disabled) { border-color: var(--accent); }
        .ps-controls button:disabled { opacity: 0.4; cursor: default; }
        @media (max-width: 560px) { .ps-main { grid-template-columns: 1fr; }
          .ps-readout { border-left: none; padding-left: 0; border-top: 1px solid var(--line); padding-top: 0.8rem; } }
      `}</style>
    </figure>
  );
}
