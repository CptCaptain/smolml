/** @jsxImportSource preact */
import { useMemo, useState } from "preact/hooks";
import type { Series } from "../data/curves";

// The recurring interactive visualization: validation bits-per-byte (y, linear)
// vs total FLOPs (x, log10). One shared, parameterized component re-used by the
// loss-per-FLOP, 0.1, 0.2, context-mixing-reference, and first-finding pages —
// the rule-of-two workhorse. Lower-left is better.

// Role colors mirror src/styles/global.css and the embedded matplotlib plots so
// the interactive re-renders read as the same instrument. Hardcoded (not CSS
// vars) to keep the island self-contained and SSR/hydration-stable.
const ROLE_COLOR: Record<string, string> = {
  reference: "#5ea8e6",
  fast_weight: "#f0913e",
  transformer: "#5cc46a",
  free: "#cf8be0",
  neutral: "#9a8e76",
};

export interface ChartAnnotation {
  flops: number;
  bpb: number;
  text: string;
  /** label offset in px from the anchor point. */
  dx?: number;
  dy?: number;
}

interface Props {
  series: Series[];
  /** vertical dashed marker at an equal-FLOP budget. */
  budgetLine?: number;
  budgetLabel?: string;
  /** horizontal dashed line at 8 bpb (uniform "no model"). */
  noModelLine?: boolean;
  annotations?: ChartAnnotation[];
  /** force y-domain; otherwise auto from the data. */
  yMin?: number;
  yMax?: number;
}

const VB_W = 780;
const VB_H = 480;
const M = { top: 30, right: 26, bottom: 58, left: 66 };
const PX0 = M.left;
const PX1 = VB_W - M.right;
const PY0 = M.top;
const PY1 = VB_H - M.bottom;

const SUP: Record<string, string> = {
  "0": "\u2070",
  "1": "\u00b9",
  "2": "\u00b2",
  "3": "\u00b3",
  "4": "\u2074",
  "5": "\u2075",
  "6": "\u2076",
  "7": "\u2077",
  "8": "\u2078",
  "9": "\u2079",
};
const sup = (n: number) =>
  String(n)
    .split("")
    .map((c) => SUP[c] ?? c)
    .join("");

export default function BpbFlopChart({
  series,
  budgetLine,
  budgetLabel = "equal-FLOP budget",
  noModelLine = false,
  annotations = [],
  yMin,
  yMax,
}: Props) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<{ sid: string; i: number } | null>(null);

  const visible = series.filter((s) => !hidden.has(s.id));

  const { lx0, lx1, ylo, yhi } = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const s of visible) {
      for (const p of s.points) {
        xs.push(p.flops);
        ys.push(p.bpb);
      }
    }
    if (budgetLine) xs.push(budgetLine);
    if (noModelLine) ys.push(8);
    if (xs.length === 0) {
      xs.push(1e6, 1e10);
      ys.push(4, 8);
    }
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const lo = Math.floor(Math.log10(minX) * 2) / 2 - 0.15;
    const hi = Math.ceil(Math.log10(maxX) * 2) / 2 + 0.15;
    const dataYlo = yMin ?? Math.min(...ys) - 0.3;
    const dataYhi = yMax ?? Math.max(...ys) + 0.3;
    return { lx0: lo, lx1: hi, ylo: dataYlo, yhi: dataYhi };
  }, [visible, budgetLine, noModelLine, yMin, yMax]);

  const xScale = (f: number) => PX0 + ((Math.log10(f) - lx0) / (lx1 - lx0)) * (PX1 - PX0);
  const yScale = (b: number) => PY1 - ((b - ylo) / (yhi - ylo)) * (PY1 - PY0);

  // x ticks: integer decades within [lx0, lx1]
  const xTicks: number[] = [];
  for (let k = Math.ceil(lx0); k <= Math.floor(lx1); k++) xTicks.push(k);

  // y ticks: nice ~1.0 steps
  const yTicks: number[] = [];
  const step = (yhi - ylo) / 6 > 0.75 ? 1 : 0.5;
  const startY = Math.ceil(ylo / step) * step;
  for (let v = startY; v <= yhi + 1e-9; v += step) yTicks.push(Math.round(v * 10) / 10);

  return (
    <figure class="chart">
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        role="img"
        aria-label="Bits-per-byte versus total FLOPs; lower-left is better."
        class="chart-svg"
      >
        {/* gridlines */}
        {xTicks.map((k) => (
          <line x1={xScale(10 ** k)} x2={xScale(10 ** k)} y1={PY0} y2={PY1} class="grid" />
        ))}
        {yTicks.map((v) => (
          <line x1={PX0} x2={PX1} y1={yScale(v)} y2={yScale(v)} class="grid" />
        ))}

        {/* axes */}
        <line x1={PX0} x2={PX1} y1={PY1} y2={PY1} class="axis" />
        <line x1={PX0} x2={PX0} y1={PY0} y2={PY1} class="axis" />

        {/* x tick labels */}
        {xTicks.map((k) => (
          <text x={xScale(10 ** k)} y={PY1 + 20} class="tick" text-anchor="middle">
            10{sup(k)}
          </text>
        ))}
        <text x={(PX0 + PX1) / 2} y={VB_H - 10} class="axis-label" text-anchor="middle">
          total FLOPs (log scale) &rarr;
        </text>

        {/* y tick labels */}
        {yTicks.map((v) => (
          <text x={PX0 - 10} y={yScale(v) + 4} class="tick" text-anchor="end">
            {v.toFixed(step < 1 ? 1 : 0)}
          </text>
        ))}
        <text
          class="axis-label"
          text-anchor="middle"
          transform={`translate(16 ${(PY0 + PY1) / 2}) rotate(-90)`}
        >
          &larr; validation bits-per-byte
        </text>

        {/* no-model reference */}
        {noModelLine && yhi >= 7.9 && (
          <g>
            <line x1={PX0} x2={PX1} y1={yScale(8)} y2={yScale(8)} class="ref-line" />
            <text x={PX1 - 6} y={yScale(8) - 6} class="ref-text" text-anchor="end">
              8.0 bpb &mdash; uniform &ldquo;no model&rdquo;
            </text>
          </g>
        )}

        {/* equal-FLOP budget marker */}
        {budgetLine && (
          <g>
            <line x1={xScale(budgetLine)} x2={xScale(budgetLine)} y1={PY0} y2={PY1} class="budget-line" />
            <text x={xScale(budgetLine) - 7} y={PY0 + 14} class="budget-text" text-anchor="end">
              {budgetLabel}
            </text>
          </g>
        )}

        {/* series */}
        {visible.map((s) => {
          const color = ROLE_COLOR[s.role] ?? ROLE_COLOR.neutral;
          const pts = [...s.points].sort((a, b) => a.flops - b.flops);
          const path = pts.map((p) => `${xScale(p.flops)},${yScale(p.bpb)}`).join(" ");
          return (
            <g>
              {s.kind === "curve" && pts.length > 1 && (
                <polyline
                  points={path}
                  fill="none"
                  stroke={color}
                  stroke-width="2"
                  stroke-dasharray={s.dashed ? "7 5" : undefined}
                  stroke-linejoin="round"
                />
              )}
              {pts.map((p, i) => {
                const isHover = hover?.sid === s.id && hover?.i === i;
                const r = s.kind === "point" ? 8 : 5;
                return (
                  <g
                    tabIndex={0}
                    role="button"
                    aria-label={`${s.label}: ${p.bpb.toFixed(4)} bpb at ${p.flops.toExponential(2)} FLOPs${p.tag ? ` \u2014 ${p.tag}` : ""}`}
                    class="chart-mark"
                    onMouseEnter={() => setHover({ sid: s.id, i })}
                    onMouseLeave={() => setHover(null)}
                    onFocus={() => setHover({ sid: s.id, i })}
                    onBlur={() => setHover(null)}
                    style={{ cursor: "pointer" }}
                  >
                    {/* generous invisible hit area */}
                    <circle cx={xScale(p.flops)} cy={yScale(p.bpb)} r={16} fill="transparent" />
                    {s.kind === "point" ? (
                      <rect
                        x={xScale(p.flops) - r}
                        y={yScale(p.bpb) - r}
                        width={r * 2}
                        height={r * 2}
                        transform={`rotate(45 ${xScale(p.flops)} ${yScale(p.bpb)})`}
                        fill={color}
                        stroke={isHover ? "#fff7e8" : "none"}
                        stroke-width="1.5"
                      />
                    ) : (
                      <circle
                        cx={xScale(p.flops)}
                        cy={yScale(p.bpb)}
                        r={isHover ? r + 2 : r}
                        fill={color}
                        stroke={isHover ? "#fff7e8" : "#16130e"}
                        stroke-width="1.5"
                      />
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* annotations */}
        {annotations.map((a) => {
          const ax = xScale(a.flops);
          const ay = yScale(a.bpb);
          const tx = ax + (a.dx ?? 14);
          const ty = ay + (a.dy ?? -16);
          return (
            <g>
              <line x1={ax} y1={ay} x2={tx} y2={ty} class="anno-leader" />
              <text x={tx} y={ty} class="anno-text" text-anchor={a.dx && a.dx < 0 ? "end" : "start"}>
                {a.text}
              </text>
            </g>
          );
        })}

        {/* tooltip */}
        {hover &&
          (() => {
            const s = series.find((x) => x.id === hover.sid);
            const p = s?.points[hover.i];
            if (!s || !p) return null;
            const px = xScale(p.flops);
            const py = yScale(p.bpb);
            const lines = [
              s.label,
              `bpb ${p.bpb.toFixed(p.bpb >= 1 ? 4 : 4)}`,
              `${p.flops.toExponential(2)} FLOPs`,
            ];
            if (p.tag) lines.push(p.tag);
            const w = 168;
            const h = 16 + lines.length * 15;
            let bx = px + 14;
            if (bx + w > VB_W) bx = px - w - 14;
            let by = py - h - 10;
            if (by < PY0) by = py + 12;
            return (
              <g pointer-events="none">
                <rect x={bx} y={by} width={w} height={h} rx="6" class="tip-box" />
                <rect x={bx} y={by} width="4" height={h} rx="2" fill={ROLE_COLOR[s.role]} />
                {lines.map((ln, i) => (
                  <text
                    x={bx + 12}
                    y={by + 17 + i * 15}
                    class={i === 0 ? "tip-title" : "tip-line"}
                  >
                    {ln}
                  </text>
                ))}
              </g>
            );
          })()}
      </svg>

      {/* legend / series toggles */}
      <div class="legend">
        {series.map((s) => {
          const off = hidden.has(s.id);
          return (
            <button
              type="button"
              class={off ? "legend-item off" : "legend-item"}
              onClick={() => {
                const next = new Set(hidden);
                if (off) next.delete(s.id);
                else next.add(s.id);
                setHidden(next);
              }}
              aria-pressed={!off}
            >
              <span class="legend-swatch" style={{ background: ROLE_COLOR[s.role] }} />
              {s.label}
              {s.reconstructed && <span class="legend-recon"> (reconstructed x)</span>}
            </button>
          );
        })}
      </div>

      <style>{`
        .chart { margin: 0; }
        .chart-svg { width: 100%; height: auto; display: block;
          font-family: "JetBrains Mono Variable", ui-monospace, monospace; }
        .chart-mark { outline: none; }
        .chart-mark:focus-visible rect, .chart-mark:focus-visible circle:last-of-type {
          stroke: #fff7e8; stroke-width: 2.5; }
        .chart-mark:focus-visible { outline: 2px solid #e8b54d; outline-offset: 2px; border-radius: 3px; }
        .grid { stroke: rgba(180,160,120,0.10); stroke-width: 1; }
        .axis { stroke: #5a5240; stroke-width: 1.3; }
        .tick { fill: #9a8e76; font-size: 13px; }
        .axis-label { fill: #c8bca4; font-size: 13px; letter-spacing: 0.04em; }
        .ref-line { stroke: #6f6650; stroke-width: 1.3; stroke-dasharray: 3 4; }
        .ref-text { fill: #9a8e76; font-size: 12px; }
        .budget-line { stroke: #e8b54d; stroke-width: 1.4; stroke-dasharray: 5 5; opacity: 0.8; }
        .budget-text { fill: #e8b54d; font-size: 12px; }
        .anno-leader { stroke: #cf8be0; stroke-width: 1.2; }
        .anno-text { fill: #e7cdf0; font-size: 12.5px; font-weight: 600; }
        .tip-box { fill: #221d15; stroke: #3a3326; stroke-width: 1; }
        .tip-title { fill: #fff7e8; font-size: 12.5px; font-weight: 700; }
        .tip-line { fill: #c8bca4; font-size: 12px; }
        .legend { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem;
          padding: 0.6rem 0.2rem 0.1rem; font-family: "JetBrains Mono Variable", ui-monospace, monospace; }
        .legend-item { display: inline-flex; align-items: center; gap: 0.45rem;
          background: none; border: 1px solid transparent; border-radius: 5px;
          padding: 0.18rem 0.45rem; color: #ece3d2; font-size: 0.78rem; cursor: pointer;
          font-family: inherit; transition: opacity .15s ease, border-color .15s ease; }
        .legend-item:hover { border-color: #3a3326; }
        .legend-item.off { opacity: 0.4; text-decoration: line-through; }
        .legend-swatch { width: 0.85em; height: 0.85em; border-radius: 2px; flex: none; }
        .legend-recon { color: #9a8e76; }
      `}</style>
    </figure>
  );
}
