/** @jsxImportSource preact */
import { useState } from "preact/hooks";

// The Source-(iv) screen as a toggle: pick the reasons a backprop-free candidate
// claims to be worth scouting, and see whether any of them actually move the
// loss-per-FLOP scoreboard. Only (iv) does. Everything else is parked.

interface Source {
  id: string;
  name: string;
  claim: string;
  impact: "high" | "low" | "zero";
  note: string;
}
const SOURCES: Source[] = [
  { id: "i", name: "cheaper credit assignment", claim: "\u201cskip the backward pass\u201d", impact: "low",
    note: "Backward is only ~2\u00d7 forward, so the ceiling is ~3\u00d7 \u2014 and it is usually spent back. Modest." },
  { id: "ii", name: "locality \u2192 parallelism / async", claim: "\u201cno global sync\u201d", impact: "zero",
    note: "Wall-clock & scaling only. Scores ZERO on a fixed FLOP budget \u2014 out of scope." },
  { id: "iii", name: "no activation storage", claim: "\u201cfits in memory\u201d", impact: "zero",
    note: "Memory only; barely touches FLOPs." },
  { id: "iv", name: "better learning dynamics", claim: "\u201creduces loss faster per FLOP\u201d", impact: "high",
    note: "The update rule itself extracts more loss-reduction per FLOP. The only thing that moves our scoreboard." },
];
const IMPACT_PCT: Record<string, number> = { high: 100, low: 28, zero: 4 };

export default function SourceIvScreen() {
  const [on, setOn] = useState<Set<string>>(new Set());

  const toggle = (id: string) => {
    const next = new Set(on);
    next.has(id) ? next.delete(id) : next.add(id);
    setOn(next);
  };

  const hasIv = on.has("iv");
  const hasAny = on.size > 0;
  const verdict = !hasAny ? "empty" : hasIv ? "scout" : "park";

  return (
    <figure class="siv">
      <p class="siv-q">
        Before any candidate earns a GPU-hour: <em>is there a plausible reason this reduces loss
        faster <strong>per FLOP</strong> — or is it just avoiding a cheap backward pass / buying
        parallelism we don&rsquo;t reward?</em>
      </p>

      <div class="siv-chips">
        {SOURCES.map((s) => (
          <button
            type="button"
            class={`siv-chip ${on.has(s.id) ? "on" : ""} imp-${s.impact}`}
            onClick={() => toggle(s.id)}
            aria-pressed={on.has(s.id)}
          >
            <span class="siv-chip-id">({s.id})</span> {s.name}
          </button>
        ))}
      </div>

      <div class="siv-rows">
        {SOURCES.filter((s) => on.has(s.id)).map((s) => (
          <div class="siv-row">
            <div class="siv-row-head">
              <span class="siv-row-name">({s.id}) {s.claim}</span>
              <span class={`siv-tag imp-${s.impact}`}>
                {s.impact === "high" ? "moves the metric" : s.impact === "low" ? "barely" : "scores zero"}
              </span>
            </div>
            <div class="siv-meter">
              <div class={`siv-meter-fill imp-${s.impact}`} style={{ width: `${IMPACT_PCT[s.impact]}%` }} />
            </div>
            <p class="siv-note">{s.note}</p>
          </div>
        ))}
        {!hasAny && <p class="siv-empty">Toggle the reasons the candidate claims &uarr;</p>}
      </div>

      <div class={`siv-verdict v-${verdict}`}>
        {verdict === "empty" && <span>awaiting a claim&hellip;</span>}
        {verdict === "scout" && (
          <span>
            <strong>Scout it.</strong> There is a real per-FLOP story (iv) — the only admissible
            reason to spend the compute.
          </span>
        )}
        {verdict === "park" && (
          <span>
            <strong>Parked.</strong> Only (i)/(ii)/(iii) — these buy speed, parallelism, or memory,
            none of which the fixed-FLOP metric rewards. Not forbidden as inspiration; just not a win here.
          </span>
        )}
      </div>

      <div class="siv-presets">
        <span>try:</span>
        <button type="button" onClick={() => setOn(new Set(["i", "ii"]))}>Forward-Forward</button>
        <button type="button" onClick={() => setOn(new Set(["iv"]))}>a real (iv) candidate</button>
        <button type="button" onClick={() => setOn(new Set())}>clear</button>
      </div>

      <figcaption class="figcaption">
        <em>Forward-Forward</em> replaces the backward pass with a second forward pass — but two
        forwards &asymp; one forward+backward, so its (i) saving is &asymp;0 and its real selling
        point is (ii) locality. No distinct (iv) story &rarr; parked here.
      </figcaption>

      <style>{`
        .siv { margin: 1.8em 0; border: 1px solid var(--line); border-radius: var(--radius-lg);
          background: var(--ink-1); padding: 1.1rem 1.2rem; }
        .siv-q { font-size: 0.95rem; color: var(--paper-dim); margin: 0 0 1rem; }
        .siv-q em { font-style: italic; color: var(--paper); }
        .siv-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .siv-chip { font-family: var(--font-mono); font-size: 0.8rem; cursor: pointer;
          background: var(--ink-2); color: var(--paper-dim); border: 1px solid var(--line);
          border-radius: 999px; padding: 0.35rem 0.8rem; transition: all .15s ease; }
        .siv-chip-id { color: var(--muted); }
        .siv-chip:hover { border-color: var(--faint); }
        .siv-chip.on { color: #16130e; }
        .siv-chip.on.imp-high { background: var(--accent); border-color: var(--accent); }
        .siv-chip.on.imp-low { background: #b89255; border-color: #b89255; }
        .siv-chip.on.imp-zero { background: #6f6650; border-color: #6f6650; color: var(--paper); }
        .siv-rows { min-height: 2rem; }
        .siv-row { margin: 0.7rem 0; }
        .siv-row-head { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }
        .siv-row-name { font-family: var(--font-mono); font-size: 0.82rem; color: var(--paper); }
        .siv-tag { font-family: var(--font-mono); font-size: 0.68rem; text-transform: uppercase;
          letter-spacing: 0.08em; padding: 0.1rem 0.45rem; border-radius: 4px; }
        .siv-tag.imp-high { color: #16130e; background: var(--accent); }
        .siv-tag.imp-low { color: #16130e; background: #b89255; }
        .siv-tag.imp-zero { color: var(--paper); background: var(--faint); }
        .siv-meter { height: 0.6rem; background: var(--ink-2); border-radius: 4px; overflow: hidden; margin: 0.3rem 0; }
        .siv-meter-fill { height: 100%; }
        .siv-meter-fill.imp-high { background: linear-gradient(90deg, var(--accent-deep), var(--accent)); }
        .siv-meter-fill.imp-low { background: #b89255; }
        .siv-meter-fill.imp-zero { background: var(--faint); }
        .siv-note { font-size: 0.82rem; color: var(--muted); margin: 0.2rem 0 0; }
        .siv-empty { color: var(--faint); font-family: var(--font-mono); font-size: 0.82rem; }
        .siv-verdict { margin-top: 1rem; padding: 0.8rem 1rem; border-radius: var(--radius);
          border: 1px solid var(--line); font-size: 0.92rem; }
        .siv-verdict.v-scout { border-left: 3px solid var(--c-transformer); background: rgba(92,196,106,0.08); }
        .siv-verdict.v-park { border-left: 3px solid var(--c-fast); background: rgba(240,145,62,0.08); }
        .siv-verdict.v-empty { color: var(--muted); font-family: var(--font-mono); }
        .siv-presets { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem;
          margin-top: 0.9rem; font-family: var(--font-mono); font-size: 0.78rem; color: var(--muted); }
        .siv-presets button { background: var(--ink-2); color: var(--accent); border: 1px solid var(--line);
          border-radius: 5px; padding: 0.25rem 0.6rem; cursor: pointer; font-family: inherit; font-size: 0.78rem; }
        .siv-presets button:hover { border-color: var(--accent-deep); }
      `}</style>
    </figure>
  );
}
