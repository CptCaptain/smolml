/** @jsxImportSource preact */
import { useState } from "preact/hooks";
import StreamScaffold from "./StreamScaffold.tsx";

// Online context mixing, alive: K order-k byte specialists (smoothed conditional
// frequency tables), their predictions blended by logistic mixing weights that
// learn online by SGD on the cross-entropy of each revealed byte. Watch the
// weights re-allocate toward whichever order is paying off, and the running bpb
// fall as the higher orders lock onto the repetition. Math mirrors the
// context-mixing concept page exactly.

const STREAM = "mississippi river, mississippi river. ";
const ALPHA = [...new Set(STREAM.split(""))].sort();
const V = ALPHA.length;
const ORDERS = [0, 1, 2];
const K = ORDERS.length;
const LR = 0.03;
const show = (c: string) => (c === " " ? "\u2423" : c);
const ln2 = Math.log(2);

interface State {
  pos: number;
  // counts[k]: context-string -> (char -> count)
  counts: Record<string, Record<string, number>>[];
  w: number[];
  cumBits: number;
  history: number[];
}

function fresh(): State {
  return {
    pos: 0,
    counts: ORDERS.map(() => ({})),
    w: ORDERS.map(() => 1 / K),
    cumBits: 0,
    history: [],
  };
}

interface SpecView {
  order: number;
  seen: boolean;
  s: number[]; // log p_k over ALPHA
  top: { ch: string; p: number };
}

function specialists(st: State): SpecView[] {
  return ORDERS.map((k, ki) => {
    const ctx = st.pos < k ? null : STREAM.slice(st.pos - k, st.pos);
    const row = ctx != null ? st.counts[ki][ctx] : undefined;
    const tot = row ? Object.values(row).reduce((a, b) => a + b, 0) : 0;
    const seen = tot > 0;
    const p = ALPHA.map((ch) => (seen ? ((row![ch] ?? 0) + 1) / (tot + V) : 1 / V));
    const s = p.map((x) => Math.log(x));
    let bi = 0;
    for (let j = 1; j < V; j++) if (p[j] > p[bi]) bi = j;
    return { order: k, seen, s, top: { ch: ALPHA[bi], p: p[bi] } };
  });
}

function mixed(specs: SpecView[], w: number[]): number[] {
  const z = ALPHA.map((_, b) => specs.reduce((acc, sp, ki) => acc + w[ki] * sp.s[b], 0));
  const mx = Math.max(...z);
  const ex = z.map((v) => Math.exp(v - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((e) => e / sum);
}

function advance(st: State): State {
  const truth = STREAM[st.pos];
  const yi = ALPHA.indexOf(truth);
  const specs = specialists(st);
  const P = mixed(specs, st.w);
  const bits = -Math.log(P[yi]) / ln2;

  // logistic-mixing SGD: dL/dw_k = sum_b (P[b] - 1{b=y}) * s_k[b]
  const w = st.w.map((wk, ki) => {
    let g = 0;
    for (let b = 0; b < V; b++) g += (P[b] - (b === yi ? 1 : 0)) * specs[ki].s[b];
    return Math.max(0, Math.min(3, wk - LR * g));
  });

  // fold the byte into every order-k count table
  const counts = st.counts.map((tbl, ki) => {
    const ord = ORDERS[ki];
    const ctx = st.pos < ord ? null : STREAM.slice(st.pos - ord, st.pos);
    if (ctx == null) return tbl;
    const next = { ...tbl, [ctx]: { ...(tbl[ctx] ?? {}), [truth]: (tbl[ctx]?.[truth] ?? 0) + 1 } };
    return next;
  });

  return { pos: st.pos + 1, counts, w, cumBits: st.cumBits + bits, history: [...st.history, bits] };
}

export default function ContextMixingDemo() {
  const [st, setSt] = useState<State>(fresh);
  const [playing, setPlaying] = useState(false);

  const done = st.pos >= STREAM.length;
  const specs = done ? [] : specialists(st);
  const truth = done ? null : STREAM[st.pos];
  const P = done ? [] : mixed(specs, st.w);
  const yi = truth ? ALPHA.indexOf(truth) : -1;
  const bits = yi >= 0 ? -Math.log(P[yi]) / ln2 : 0;
  const maxW = Math.max(0.01, ...st.w);

  // top-5 mixed predictions
  const ranked = P.map((p, i) => ({ ch: ALPHA[i], p, i })).sort((a, b) => b.p - a.p).slice(0, 5);
  const maxP = ranked.length ? ranked[0].p : 1;

  const step = () => setSt((s) => (s.pos < STREAM.length ? advance(s) : s));

  return (
    <StreamScaffold
      stream={STREAM}
      pos={st.pos}
      cumBits={st.cumBits}
      history={st.history}
      playing={playing}
      onStep={() => { if (done) setPlaying(false); else step(); }}
      onPlay={() => setPlaying((p) => !p)}
      onReset={() => { setPlaying(false); setSt(fresh()); }}
      caption={
        <>
          Each specialist is a smoothed order-k frequency table; the mixer is one-layer logistic
          regression on their stretched (log-prob) outputs, learned online by SGD. Watch the
          weights shift toward the higher orders once the repetition appears — that is the entire
          learning algorithm of the <a href="/concepts/context-mixing">context-mixing reference</a>.
        </>
      }
    >
      <p class="cm-h">{done ? "stream complete" : `${K} specialists \u2192 online logistic mix`}</p>
      {!done && (
        <div class="cm-specs">
          {specs.map((sp, ki) => (
            <div class={`cm-spec ${sp.seen ? "" : "abstain"}`}>
              <div class="cm-spec-top">
                <span class="cm-order">order-{sp.order}</span>
                <span class="cm-pred">
                  {sp.seen ? <>&rarr; &lsquo;{show(sp.top.ch)}&rsquo; {(sp.top.p * 100).toFixed(0)}%</> : "abstains"}
                </span>
              </div>
              <div class="cm-wrow">
                <span class="cm-wlabel">w</span>
                <span class="cm-wtrack"><span class="cm-wfill" style={{ width: `${(st.w[ki] / maxW) * 100}%` }} /></span>
                <span class="cm-wval">{st.w[ki].toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {!done && <p class="cm-h">mixed P(next byte)</p>}
      {ranked.map((r) => (
        <div class={`cm-bar ${r.i === yi ? "istrue" : ""}`}>
          <span class="cm-bar-ch">{show(r.ch)}</span>
          <span class="cm-bar-track"><span class="cm-bar-fill" style={{ width: `${(r.p / maxP) * 100}%` }} /></span>
          <span class="cm-bar-p">{(r.p * 100).toFixed(1)}%</span>
        </div>
      ))}
      {!done && (
        <p class="cm-pending">
          truth is <code>{show(truth!)}</code> &rarr; pays <strong>{bits.toFixed(2)}</strong> bits
        </p>
      )}

      <style>{`
        .cm-h { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.12em;
          text-transform: uppercase; color: var(--muted); margin: 0 0 0.5rem; }
        .cm-specs { display: grid; gap: 0.4rem; margin-bottom: 0.9rem; }
        .cm-spec { border: 1px solid var(--line); border-radius: 6px; padding: 0.4rem 0.55rem; background: var(--ink-2); }
        .cm-spec.abstain { opacity: 0.55; }
        .cm-spec-top { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.76rem; }
        .cm-order { color: var(--c-reference); }
        .cm-pred { color: var(--paper-dim); }
        .cm-wrow { display: grid; grid-template-columns: 1rem 1fr 2.4rem; align-items: center; gap: 0.4rem; margin-top: 0.25rem; }
        .cm-wlabel { font-family: var(--font-mono); font-size: 0.68rem; color: var(--muted); }
        .cm-wtrack { height: 0.5rem; background: var(--ink); border-radius: 3px; overflow: hidden; }
        .cm-wfill { display: block; height: 100%; background: #5ea8e6; }
        .cm-wval { font-family: var(--font-mono); font-size: 0.7rem; color: var(--paper-dim); text-align: right; }
        .cm-bar { display: grid; grid-template-columns: 1.4rem 1fr 3rem; align-items: center; gap: 0.5rem; margin: 0.22rem 0; }
        .cm-bar-ch { font-family: var(--font-mono); color: var(--paper-dim); text-align: center; }
        .cm-bar-track { height: 0.8rem; background: var(--ink-2); border-radius: 3px; overflow: hidden; }
        .cm-bar-fill { display: block; height: 100%; background: #4c7da8; }
        .cm-bar.istrue .cm-bar-fill { background: #5cc46a; }
        .cm-bar.istrue .cm-bar-ch { color: #5cc46a; font-weight: 700; }
        .cm-bar-p { font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted); text-align: right; }
        .cm-pending { font-family: var(--font-mono); font-size: 0.78rem; color: var(--paper-dim); margin: 0.4rem 0 0; }
        .cm-pending strong { color: #e8b54d; }
      `}</style>
    </StreamScaffold>
  );
}
