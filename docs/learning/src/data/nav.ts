// The compendium sitemap — one source of truth for the sidebar, the index
// concept list, and prev/next links, so nothing becomes an orphan page.

export interface NavItem {
  slug: string;
  href: string;
  title: string;
  /** short index/sidebar blurb. */
  blurb: string;
  /** mono eyebrow, e.g. "CONCEPT 01". */
  kicker: string;
}

export const concepts: NavItem[] = [
  {
    slug: "loss-per-flop-and-scaling-laws",
    href: "/concepts/loss-per-flop-and-scaling-laws",
    title: "Loss-per-FLOP & scaling laws",
    blurb: "Why “needs billions” is arithmetic, and why we measure per FLOP.",
    kicker: "CONCEPT 01",
  },
  {
    slug: "compression-equals-prediction",
    href: "/concepts/compression-equals-prediction",
    title: "Compression = prediction",
    blurb: "A good predictor is a good compressor — and what bits-per-byte means.",
    kicker: "CONCEPT 02",
  },
  {
    slug: "prequential-evaluation",
    href: "/concepts/prequential-evaluation",
    title: "Prequential evaluation",
    blurb: "Scoring a model that keeps learning, honestly, with no held-out split.",
    kicker: "CONCEPT 03",
  },
  {
    slug: "source-iv-advantage",
    href: "/concepts/source-iv-advantage",
    title: "Source-(iv) advantage",
    blurb: "The only kind of “win” a non-backprop idea is allowed to claim.",
    kicker: "CONCEPT 04",
  },
  {
    slug: "fast-weight-memory",
    href: "/concepts/fast-weight-memory",
    title: "Fast-weight associative memory",
    blurb: "The Phase-A maiden candidate: make memorization nearly free.",
    kicker: "CONCEPT 05",
  },
  {
    slug: "context-mixing",
    href: "/concepts/context-mixing",
    title: "Online context mixing",
    blurb: "The PAQ/cmix reference ceiling: cheap order-k models + online mixing.",
    kicker: "CONCEPT 06",
  },
];

export const experiments: NavItem[] = [
  {
    slug: "index",
    href: "/experiments",
    title: "Experiment log",
    blurb: "What we tried, the curves, and what we learned — failures included.",
    kicker: "LOG",
  },
  {
    slug: "0.1-baseline-harness-smoke",
    href: "/experiments/0.1-baseline-harness-smoke",
    title: "0.1 — baseline harness smoke",
    blurb: "Proof the pieces wire together: two transformer baselines, amortized.",
    kicker: "EXP 0.1",
  },
  {
    slug: "0.2-prequential-baseline",
    href: "/experiments/0.2-prequential-baseline",
    title: "0.2 — prequential baseline",
    blurb: "The first curve under the real metric: predict-before-reveal, total FLOPs.",
    kicker: "EXP 0.2",
  },
  {
    slug: "context-mixing-reference",
    href: "/experiments/context-mixing-reference",
    title: "Context-mixing reference ceiling",
    blurb: "The free single-pass bpb-per-FLOP yardstick a candidate must approach.",
    kicker: "EXP 0.3",
  },
  {
    slug: "A.1-fast-weight-memory",
    href: "/experiments/A.1-fast-weight-memory",
    title: "A.1 — fast-weight memory",
    blurb: "The maiden Source-(iv) candidate — mechanism sound, thesis unsupported.",
    kicker: "EXP A.1",
  },
  {
    slug: "first-finding-pareto",
    href: "/experiments/first-finding-pareto",
    title: "First finding — the win is Pareto-hollow",
    blurb: "“Beats the baseline” ≠ “good per FLOP” once the free reference is on the axis.",
    kicker: "FINDING",
  },
];

/** Flat ordered walk for prev/next: concepts then experiments. */
export const order: NavItem[] = [...concepts, ...experiments];

export function neighbors(href: string): { prev?: NavItem; next?: NavItem } {
  const i = order.findIndex((n) => n.href === href);
  if (i === -1) return {};
  return { prev: order[i - 1], next: order[i + 1] };
}
