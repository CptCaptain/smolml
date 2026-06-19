/** @jsxImportSource preact */
import { useState } from "preact/hooks";
import StreamScaffold from "./StreamScaffold.tsx";

// The prequential loop, alive: a *real* online model (order-0 + order-1 byte
// counts, Laplace-smoothed) predicts the next byte BEFORE it is revealed, pays
// -log2 p bits, then folds the truth into its counts and moves on. The running
// bpb visibly drops as the model learns the stream — no held-out split, and a
// memorizer can't cheat the future because every prediction precedes its reveal.
// Chrome (tape, readout, transport) is the shared StreamScaffold.

const STREAM = "the cat sat on the mat. the cat ate the rat. ";
const ALPHA = [...new Set(STREAM.split(""))].sort();
const A = ALPHA.length;
const show = (c: string) => (c === " " ? "\u2423" : c);

interface State {
  pos: number;
  c0: Record<string, number>;
  c1: Record<string, Record<string, number>>;
  cumBits: number;
  history: number[];
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
    const p1 = row ? ((row[ch] ?? 0) + 1) / (tot1 + A) : 1 / A;
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
  if (prev != null) c1[prev] = { ...(c1[prev] ?? {}), [truth]: (c1[prev]?.[truth] ?? 0) + 1 };
  return { pos: s.pos + 1, c0, c1, cumBits: s.cumBits + bitsPaid, history: [...s.history, bitsPaid] };
}

export default function PrequentialStream() {
  const [s, setS] = useState<State>(fresh);
  const [playing, setPlaying] = useState(false);

  const done = s.pos >= STREAM.length;
  const dist = done ? [] : predict(s);
  const truth = done ? null : STREAM[s.pos];
  const nextBits = truth ? -Math.log2(dist.find((d) => d.ch === truth)!.p) : 0;
  const top = dist.slice(0, 6);
  const maxP = top.length ? top[0].p : 1;

  const step = () => setS((p) => (p.pos < STREAM.length ? advance(p) : p));

  return (
    <StreamScaffold
      stream={STREAM}
      pos={s.pos}
      cumBits={s.cumBits}
      history={s.history}
      playing={playing}
      onStep={() => { if (done) setPlaying(false); else step(); }}
      onPlay={() => setPlaying((p) => !p)}
      onReset={() => { setPlaying(false); setS(fresh()); }}
      caption={
        <>
          Every prediction is made <em>before</em> the byte is revealed, so a model that memorizes
          the past cannot cheat the future — an honest generalization measure with no held-out
          split. The cumulative bits equal the compressed length, tying straight back to{" "}
          <a href="/concepts/compression-equals-prediction">compression = prediction</a>.
        </>
      }
    >
      <p class="ps-h">{done ? "stream complete" : "model\u2019s p(next byte) \u2014 before reveal"}</p>
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
          truth is <code>{show(truth!)}</code> &rarr; pays <strong>{nextBits.toFixed(2)}</strong> bits
        </p>
      )}

      <style>{`
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
      `}</style>
    </StreamScaffold>
  );
}
