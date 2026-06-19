/** @jsxImportSource preact */
import type { ComponentChildren } from "preact";
import { useEffect, useRef } from "preact/hooks";

// Shared chrome for the predict-before-reveal stream demos (rule of two:
// PrequentialStream and ContextMixingDemo both want a stream tape, a running-bpb
// readout with a per-byte bits sparkline, and Step/Play/Reset transport with a
// play timer). The scaffold owns the chrome and the auto-play loop; each demo
// owns its model state and supplies its prediction panel as children.

const showCh = (c: string) => (c === " " ? "\u2423" : c);

interface Props {
  stream: string;
  pos: number;
  cumBits: number;
  history: number[];
  playing: boolean;
  onStep: () => void;
  onPlay: () => void;
  onReset: () => void;
  /** prediction panel. */
  children: ComponentChildren;
  caption?: ComponentChildren;
  barColor?: string;
  stepMs?: number;
}

export default function StreamScaffold({
  stream,
  pos,
  cumBits,
  history,
  playing,
  onStep,
  onPlay,
  onReset,
  children,
  caption,
  barColor = "#5ea8e6",
  stepMs = 520,
}: Props) {
  const done = pos >= stream.length;
  const bytes = pos;
  const bpb = bytes > 0 ? cumBits / bytes : 0;
  const onStepRef = useRef(onStep);
  onStepRef.current = onStep;

  useEffect(() => {
    if (!playing || done) return;
    const id = window.setTimeout(() => onStepRef.current(), stepMs);
    return () => clearTimeout(id);
  }, [playing, pos, done, stepMs]);

  return (
    <figure class="strm">
      <div class="strm-tape" aria-label="prediction stream">
        {stream.split("").map((ch, i) => (
          <span class={i < pos ? "strm-ch seen" : i === pos ? "strm-ch cursor" : "strm-ch future"}>
            {i === pos && !done ? "\u25af" : showCh(ch)}
          </span>
        ))}
      </div>

      <div class="strm-main">
        <div class="strm-panel">{children}</div>

        <div class="strm-readout">
          <div class="strm-metric">
            <span class="strm-metric-v">{bpb.toFixed(3)}</span>
            <span class="strm-metric-l">running bpb</span>
          </div>
          <div class="strm-sub">
            <span>{cumBits.toFixed(1)} bits</span>
            <span>&divide; {bytes} bytes</span>
          </div>
          <svg viewBox="0 0 220 56" class="strm-spark" role="img" aria-label="bits paid per byte">
            <line x1="0" y1="55" x2="220" y2="55" stroke="#3a3326" />
            {history.map((b, i) => {
              const w = 220 / stream.length;
              const h = (Math.min(b, 9) / 9) * 50;
              return <rect x={i * w} y={55 - h} width={Math.max(1, w - 0.6)} height={h} fill={barColor} />;
            })}
          </svg>
          <p class="strm-spark-cap">bits paid per byte (it drops as the model learns)</p>
        </div>
      </div>

      <div class="strm-controls">
        <button type="button" onClick={onStep} disabled={done}>Step</button>
        <button type="button" class="primary" onClick={onPlay} disabled={done}>
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={onReset}>Reset</button>
      </div>

      {caption && <figcaption class="figcaption">{caption}</figcaption>}

      <style>{`
        .strm { margin: 1.8em 0; border: 1px solid var(--line); border-radius: var(--radius-lg);
          background: var(--ink-1); padding: 1.1rem 1.2rem; }
        .strm-tape { font-family: var(--font-mono); font-size: 1.15rem; letter-spacing: 0.04em;
          background: var(--ink); border: 1px solid var(--line); border-radius: var(--radius);
          padding: 0.7rem 0.8rem; line-height: 1.8; word-break: break-all; }
        .strm-ch.seen { color: var(--paper); }
        .strm-ch.future { color: var(--faint); }
        .strm-ch.cursor { color: #e8b54d; background: rgba(232,181,77,0.18); border-radius: 3px; padding: 0 1px; }
        .strm-main { display: grid; grid-template-columns: 1fr 13rem; gap: 1.2rem; margin-top: 1rem; }
        .strm-readout { border-left: 1px solid var(--line); padding-left: 1rem; }
        .strm-metric { display: flex; flex-direction: column; }
        .strm-metric-v { font-family: var(--font-mono); font-size: 2rem; font-weight: 700; color: #e8b54d; line-height: 1; }
        .strm-metric-l { font-family: var(--font-mono); font-size: 0.7rem; color: var(--muted); }
        .strm-sub { display: flex; justify-content: space-between; font-family: var(--font-mono);
          font-size: 0.74rem; color: var(--paper-dim); margin: 0.5rem 0; }
        .strm-spark { width: 100%; height: auto; margin-top: 0.4rem; }
        .strm-spark-cap { font-family: var(--font-mono); font-size: 0.66rem; color: var(--muted); margin: 0.2rem 0 0; line-height: 1.3; }
        .strm-controls { display: flex; gap: 0.5rem; margin-top: 1rem; }
        .strm-controls button { font-family: var(--font-mono); font-size: 0.8rem; cursor: pointer;
          background: var(--ink-2); color: var(--paper); border: 1px solid var(--line);
          border-radius: 5px; padding: 0.4rem 0.9rem; }
        .strm-controls button.primary { background: var(--accent-glow); color: var(--accent); border-color: var(--accent-deep); }
        .strm-controls button:hover:not(:disabled) { border-color: var(--accent); }
        .strm-controls button:disabled { opacity: 0.4; cursor: default; }
        @media (max-width: 560px) { .strm-main { grid-template-columns: 1fr; }
          .strm-readout { border-left: none; padding-left: 0; border-top: 1px solid var(--line); padding-top: 0.8rem; } }
      `}</style>
    </figure>
  );
}
