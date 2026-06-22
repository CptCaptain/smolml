// Canonical chart datasets — the single source of truth for every bpb-vs-FLOP
// chart on the site. Numbers are transcribed verbatim from the researchers'
// experiment notes under docs/learning/experiments/ so the interactive
// re-renders cannot drift from the harness-produced tables/plots.
//
// Provenance is recorded per dataset. Where a coordinate was NOT reported by
// the harness (e.g. the per-eval x-positions of the amortized smoke trajectory),
// it is flagged `reconstructed` and the chart caption says so. We never present
// a reconstructed coordinate as a measured one.

export type SeriesRole =
  | "reference" // context-mixing reference ceiling (matplotlib tab:blue)
  | "fast_weight" // fast-weight hybrid (matplotlib tab:orange)
  | "transformer" // transformer baseline (matplotlib tab:green / orange in solo plot)
  | "free" // the free online unigram floor
  | "pc_refine" // surprise-gated predictive-coding refinement (B.1, matplotlib tab:orange; site rose so it never reads as fast-weight)
  | "warm" // warm_mix: warmed online context-mixer (B.2; harness tab:orange, site vermilion --c-warm so it never reads as fast-weight)
  | "neutral";

export interface CurvePoint {
  /** total FLOPs (x). */
  flops: number;
  /** validation bits-per-byte (y). */
  bpb: number;
  /** short marker annotation, e.g. "order 0..3" or "4×10¹⁰". */
  tag?: string;
}

export interface Series {
  id: string;
  label: string;
  role: SeriesRole;
  /** "curve" = connected markers; "point" = a single lone marker. */
  kind: "curve" | "point";
  dashed?: boolean;
  points: CurvePoint[];
  /** true when x-coordinates were reconstructed, not reported. */
  reconstructed?: boolean;
}

// ── First finding: three-way Pareto (synthetic text8 clone, final 512 B tail) ──
// Source: experiments/first-finding-pareto.md + experiments/unified-leaderboard.md
// (the precise table). All three models on the same prequential protocol, now
// at the canonical seq_len=128 (matches the 0.2 / A.1 runs after the master
// regeneration), so the transformer/fast-weight numbers agree across all pages.
export const firstFinding: Series[] = [
  {
    id: "context_mixing",
    label: "context-mixing reference",
    role: "reference",
    kind: "point",
    points: [{ flops: 4.283e6, bpb: 4.7779, tag: "free, ~4.3×10⁶ FLOPs" }],
  },
  {
    id: "transformer",
    label: "transformer baseline",
    role: "transformer",
    kind: "curve",
    dashed: true,
    points: [
      { flops: 1.727e8, bpb: 8.0003, tag: "b0" },
      { flops: 1.571e9, bpb: 7.6914, tag: "b2×10⁹" },
      { flops: 9.962e9, bpb: 6.0125, tag: "b10¹⁰" },
      { flops: 3.933e10, bpb: 4.2059, tag: "b4×10¹⁰" },
    ],
  },
  {
    id: "fast_weight",
    label: "fast-weight hybrid",
    role: "fast_weight",
    kind: "curve",
    dashed: true,
    points: [
      { flops: 2.052e8, bpb: 7.4095, tag: "b0" },
      { flops: 1.604e9, bpb: 7.3953, tag: "b2×10⁹" },
      { flops: 9.995e9, bpb: 5.9130, tag: "b10¹⁰" },
      { flops: 3.936e10, bpb: 4.4017, tag: "b4×10¹⁰" },
    ],
  },
];

// ── Context-mixing reference ceiling (bundled English sample, final 800 B) ──
// Source: experiments/index.md (the 2026-06-18 reference-ceiling entry).
// Distinct run from `firstFinding` (800 B eval stream, order sweep), so the
// reference bpb differs (4.17 here vs 4.78 on the 512 B clone tail).
export const contextMixingReference: Series[] = [
  {
    id: "context_mixing",
    label: "context-mixing reference (order sweep)",
    role: "reference",
    kind: "curve",
    dashed: true,
    points: [
      { flops: 2.66e6, bpb: 4.7066, tag: "order 0..0" },
      { flops: 4.28e6, bpb: 4.3922, tag: "order 0..1" },
      { flops: 5.74e6, bpb: 4.2284, tag: "order 0..2" },
      { flops: 7.02e6, bpb: 4.1733, tag: "order 0..3" },
    ],
  },
  {
    id: "transformer_untrained",
    label: "untrained transformer (contrast)",
    role: "transformer",
    kind: "point",
    points: [{ flops: 3.48e9, bpb: 7.9786, tag: "~500× the FLOPs" }],
  },
];

// ── 0.2: first prequential transformer baseline curve (text8 clone, 512 B) ──
// Source: experiments/0.2-prequential-baseline.md. Four total-FLOP budgets;
// the model is frozen during eval (adaptation FLOPs = 0).
export const prequentialBaseline: Series[] = [
  {
    id: "transformer",
    label: "transformer (frozen, prequential)",
    role: "transformer",
    kind: "curve",
    dashed: true,
    points: [
      { flops: 1.7e8, bpb: 8.0, tag: "0 pretrain — “no model”" },
      { flops: 1.6e9, bpb: 7.69, tag: "2×10⁹" },
      { flops: 1.0e10, bpb: 6.01, tag: "10¹⁰" },
      { flops: 3.9e10, bpb: 4.21, tag: "4×10¹⁰" },
    ],
  },
];

// ── 0.1: amortized baseline smoke run (bundled 5 KB sample) ──
// Source: experiments/0.1-baseline-harness-smoke.md.
// MEASURED: final bpb at the shared 5×10¹⁰ budget (d32=3.37, d64=3.82) and the
// per-run step counts (220 / 46). RECONSTRUCTED: the x-positions of the
// intermediate eval checkpoints — only the bpb *sequence* and the final budget
// were reported, so intermediate FLOPs are placed at a uniform eval cadence
// (perStep × stepAtEval). Endpoints are exact; the shape is the documented
// takeaway. Flagged `reconstructed` and called out in the chart caption.
const D32_PER_STEP = 5e10 / 220;
const D64_PER_STEP = 5e10 / 46;
export const AMORTIZED_BUDGET = 5e10;
export const amortizedBaseline: Series[] = [
  {
    id: "d32",
    label: "d32 — 32,928 params (220 steps)",
    role: "transformer",
    kind: "curve",
    dashed: true,
    reconstructed: true,
    points: [
      { flops: D32_PER_STEP * 5, bpb: 8.0 },
      { flops: D32_PER_STEP * 30, bpb: 4.86 },
      { flops: D32_PER_STEP * 70, bpb: 4.23 },
      { flops: D32_PER_STEP * 130, bpb: 3.98 },
      { flops: D32_PER_STEP * 220, bpb: 3.37, tag: "3.37 @ 5×10¹⁰" },
    ],
  },
  {
    id: "d64",
    label: "d64 — 164,288 params (46 steps)",
    role: "fast_weight", // reuse the orange role purely as a distinct second color
    kind: "curve",
    dashed: true,
    reconstructed: true,
    points: [
      { flops: D64_PER_STEP * 2, bpb: 8.0 },
      { flops: D64_PER_STEP * 14, bpb: 4.38 },
      { flops: D64_PER_STEP * 30, bpb: 3.9 },
      { flops: D64_PER_STEP * 46, bpb: 3.82, tag: "3.82 @ 5×10¹⁰" },
    ],
  },
];

// The free online unigram floor (A.1): predict proportional to Laplace counts,
// then increment, scored prequentially on the same 512 B clone tail.
// Source: experiments/A.1-fast-weight-memory.md (~1.3×10⁵ FLOPs, 5.33 bpb).
export const freeUnigram: Series = {
  id: "free_unigram",
  label: "free online unigram (floor)",
  role: "free",
  kind: "point",
  points: [{ flops: 1.3e5, bpb: 5.33, tag: "~10⁵ FLOPs — the honest floor" }],
};

// The uninformed "no-model" anchor: uniform over 256 bytes = 8 bits/byte.
export const NO_MODEL_BPB = 8.0;

// ── B.1: surprise-gated predictive-coding refinement (synthetic text8 clone, 1200 B) ──
// Source: experiments/B.1-surprise-gated-pc-refinement.md (the four-entrant table).
// All four ran on the IDENTICAL 1200-byte prequential eval stream (seed 0, CPU).
// The three pretrained entrants share a bit-identical frozen transformer core, so
// they land at near-identical total FLOPs (~2.31e11) and are distinguished only on
// the y-axis; context-mixing sits ~22,000× to the left. Each is a lone measured
// point (not a budget sweep), so kind:"point". The gated−uniform matched-FLOP
// lever (−0.0045 bpb) is below the resolution of a log-FLOP axis — it lives in the
// page's result table, not the chart.
export const surpriseGatedPc: Series[] = [
  {
    id: "transformer",
    label: "transformer (core only)",
    role: "transformer",
    kind: "point",
    points: [{ flops: 2.311e11, bpb: 4.1992, tag: "frozen core — cheapest correct predictor" }],
  },
  {
    id: "pc_gated",
    label: "pc_refine — gated (surprise)",
    role: "pc_refine",
    kind: "point",
    points: [{ flops: 2.312e11, bpb: 4.2288, tag: "−0.0045 bpb vs uniform at matched FLOPs" }],
  },
  {
    id: "pc_uniform",
    label: "pc_refine — uniform K (control)",
    role: "neutral",
    kind: "point",
    points: [{ flops: 2.312e11, bpb: 4.2333, tag: "fixed settling depth, same eval FLOPs" }],
  },
  {
    id: "context_mixing",
    label: "context-mixing reference",
    role: "reference",
    kind: "point",
    points: [{ flops: 1.036e7, bpb: 4.4637, tag: "free, ~10⁷ FLOPs — the per-FLOP ceiling" }],
  },
];

// ── B.2 Phase 1: warm_mix vs transformer (real enwik8, 4 MB slice, 32 k eval) ──
// Source: experiments/B.2-warmed-mixing.md (Phase-1 table). The project's first move
// onto real text (ADR-0004 enwik8 carve); prior/eval disjoint, ALL FLOPs counted.
// warm_mix is the context-mixer with one new idea — a stateful prior→eval warm-start:
// at warmup 0 it is bit-identical to the cold context-mixing reference (rendered here
// as a separate `reference` marker, drawn over the warm curve's cold start, so the
// "cold == reference" identity is visually explicit), then warming drops bpb cheaply.
// warm_mix strictly dominates the transformer: lower bpb at ~94× fewer total FLOPs —
// the project's first genuine per-FLOP win. (The transformer is badly undertrained at
// this tiny budget; its windowed-recompute eval alone is ~9.5e11 FLOPs.)
export const warmedMixing: Series[] = [
  {
    id: "transformer",
    label: "transformer baseline",
    role: "transformer",
    kind: "point",
    points: [{ flops: 9.71e11, bpb: 5.5453, tag: "badly undertrained on real enwik8" }],
  },
  {
    id: "warm_mix",
    label: "warm_mix (warmed)",
    role: "warm",
    kind: "curve",
    dashed: true,
    points: [
      { flops: 3.05e8, bpb: 3.2106, tag: "warmup 0 — bit-identical to the cold reference" },
      { flops: 1.3e9, bpb: 2.8805, tag: "warmed @1e9" },
      { flops: 1.03e10, bpb: 2.77, tag: "warmed @1e10 — strictly dominates the transformer" },
    ],
  },
  {
    id: "context_mixing",
    label: "context-mixing reference (cold)",
    role: "reference",
    kind: "point",
    points: [{ flops: 3.05e8, bpb: 3.2106, tag: "warm_mix @ warmup 0" }],
  },
];

// ── B.2 Phase 2: gated_mix vs fixed-order warm_mix (real enwik8, warmed @1e9) ──
// Source: experiments/B.2-warmed-mixing.md (Phase-2 table). The fixed-order warm_mix
// curve (orders 2..6) is the frontier; gated_mix holds orders 0..K but escalates
// cheapest-first and stops on a pre-reveal `1 − max p` gate, charging FLOPs only for
// the orders evaluated (thresholds 0.7 / 0.5 / 0.3 / 0.1). Every gated point is
// dominated by a fixed-order point (≤ bpb AND ≤ FLOPs): the gate recomputes a
// confidence softmax per escalation (O(depth·V)) on top of an already-cheap O(K·V)
// mix, so it costs more than it saves — honestly Pareto-hollow, NOT a win.
export const gatedMix: Series[] = [
  {
    id: "warm_mix_fixed",
    label: "warm_mix — fixed order (frontier)",
    role: "warm",
    kind: "curve",
    dashed: true,
    points: [
      { flops: 1.229e9, bpb: 3.2482, tag: "order 2" },
      { flops: 1.304e9, bpb: 2.8805, tag: "order 3" },
      { flops: 1.351e9, bpb: 2.7096, tag: "order 4" },
      { flops: 1.422e9, bpb: 2.6666, tag: "order 5" },
      { flops: 1.477e9, bpb: 2.6552, tag: "order 6 — best fixed" },
    ],
  },
  {
    id: "gated_mix",
    label: "gated_mix (escalating gate)",
    role: "neutral",
    kind: "curve",
    dashed: true,
    points: [
      { flops: 1.335e9, bpb: 3.1179, tag: "thr 0.7 (aggressive)" },
      { flops: 1.463e9, bpb: 2.8274, tag: "thr 0.5" },
      { flops: 1.538e9, bpb: 2.7037, tag: "thr 0.3" },
      { flops: 1.57e9, bpb: 2.6698, tag: "thr 0.1 (near-full)" },
    ],
  },
];

// ── B.3: bounded (hashed) order-6 tables on the FULL enwik8 ADR carve ─────────
// Source: experiments/B.3-hashed-mix-full-corpus.md (the full-carve table). The
// engineering unlock: B.2's order-6 win used unbounded dict count tables that OOM
// (~58 GB) on a full-95 MB warmup; hashed_mix bounds the high orders (k ≥
// hash_min_order=4) to a fixed 2^table_bits = 2^20-slot hashed table (collisions
// accepted), so memory is fixed regardless of corpus size. On the REAL 5 MB ADR
// eval stream (first ~95 MB = prior), the order-6 advantage survives the bounding:
// it beats order-3 per FLOP in ≤4.3 GiB — the first end-to-end full-carve run.
// All points landed (run complete). The hashed_o6 curve: cold -> ~7 MB -> full-95 MB warmup — full
// warmup did NOT saturate the 2^20 table (it kept helping: 2.11 -> 2.02). The transformer
// (5.4770 @ 1.46e14) is off-scale, so it stays out of the plot (table only). hashed_o6 uses --c-warm;
// the order-3 cold reference uses the reference blue. NOTE: unlike B.2, the blue point is a DIFFERENT,
// cheaper model (order-3), not the cold start of the order-6 curve — the page caption says so.
export const hashedMixFull: Series[] = [
  {
    id: "reference_cold",
    label: "context-mix order-3 (cold)",
    role: "reference",
    kind: "point",
    points: [{ flops: 4.74e10, bpb: 2.6224, tag: "order-3, no warmup — peak 0.7 GiB" }],
  },
  {
    id: "hashed_o6",
    label: "hashed order-6 (bounded)",
    role: "warm",
    kind: "curve",
    dashed: true,
    points: [
      { flops: 7.73e10, bpb: 2.257, tag: "cold, no warmup — peak 2.3 GiB" },
      { flops: 1.78e11, bpb: 2.1111, tag: "warmed ~7 MB — peak 4.3 GiB" },
      { flops: 1.478e12, bpb: 2.0157, tag: "full 95 MB warmup — peak 5.0 GiB" },
    ],
  },
];


// ── B.4: delta_mix matched-FLOP kill-test (real enwik8, 4 MB slice, total ≈1.07e10) ──
// Source: experiments/B.4-delta-mix.md (the matched-FLOP kill-test table). The first
// NON-Pareto-hollow Space-B (learning-rule) result: one online delta-rule (LMS, error-
// correcting) fast-weight stream on a sparse signed feature-hashed key, added as one more
// raw-logit row in the warmed hashed context-mixer. The kill-test plots the cheap count
// baselines first and demands the candidate beat BOTH at matched total FLOPs:
//   (a) counts_only      — the cheap hashed order-6 ladder at budget      → reference (blue)
//   (b) delta            — counts + the delta stream (the candidate)      → fast_weight (orange)
//   (c) counts_more_warm — the SAME FLOPs spent on more warm count bytes  → warm (vermilion)
// (b) beats both (a) and (c); the binding pair is (b) vs (c) at matched total FLOPs
// (1.074e10 vs 1.072e10): −0.0146 bpb. Three lone MEASURED points (kind:"point"), clustered
// tightly in x (a hair apart on a log-FLOP axis — matched FLOPs forces this, as in B.1) and
// separated in y by ~0.017 bpb, so a tight yMin/yMax carries the story and the matched (b,c)
// pair is marked in the page caption/annotations. delta is colored fast_weight (orange) because
// it IS a fast-weight associative memory — the delta-rule flavor, the A.1 family done right:
// A.1 (also orange) was the transformer bolt-on that collapsed to the byte marginal; B.4 is the
// same family that finally clears the cheap-baseline bar. The two count entrants reuse B.3's
// vermilion (warmed hashed counts) and the reference blue (the cheap count ladder it must beat).
// The full-5 MB-ADR-carve headline (delta_o6_warmfull) vs the 2.0157 bpb @ 1.48e12 bar was STILL
// RUNNING when the note was written, and runs/full/leaderboard.md carries no delta_o6_warmfull
// row, so it is PENDING — NOT plotted here (no invented coordinate); it lives as a pending row.
export const deltaMix: Series[] = [
  {
    id: "counts_only",
    label: "counts_only (hashed order-6)",
    role: "reference",
    kind: "point",
    points: [{ flops: 1.05e10, bpb: 2.4353, tag: "the cheap ladder at budget" }],
  },
  {
    id: "delta",
    label: "delta (counts + delta stream)",
    role: "fast_weight",
    kind: "point",
    points: [{ flops: 1.074e10, bpb: 2.4181, tag: "candidate — beats both baselines" }],
  },
  {
    id: "counts_more_warm",
    label: "counts_more_warm (hashed order-6)",
    role: "warm",
    kind: "point",
    points: [{ flops: 1.072e10, bpb: 2.4327, tag: "same FLOPs, all on more warm counts" }],
  },
];
