# docs-builder process log

My working memory across sessions. Read `CONTRIBUTING.md` first (the charter), then this. I own
`docs/learning/` only; I never touch harness/research code. The published site is **MDX built with
Astro** (ADR 0005); the plain-`.md` notes under `concepts/` and `experiments/` are the researchers'
prose source — I keep them and author the interactive pages under `src/`.

## How the site is wired

- **Stack:** Astro 6 + `@astrojs/mdx` + `@astrojs/preact` (lightweight islands). KaTeX via
  `remark-math` + `rehype-katex`, configured through the Astro 6 `markdown.processor = unified({…})`
  API (MDX inherits it). Fonts self-hosted via `@fontsource-variable` (no runtime CDN).
- **Build:** `npm run build` from `docs/learning/` is the PR gate. `npm run dev` to author.
- **Routing:** file-based under `src/pages/`. Concept pages → `/concepts/<slug>`, experiments →
  `/experiments/<slug>`, log index → `/experiments`.
- **Single source of truth for numbers:** `src/data/curves.ts`. Every chart reads from it; numbers
  are transcribed verbatim from the experiment notes with provenance comments. Never hardcode a
  datapoint in a page.
- **Sitemap:** `src/data/nav.ts` drives the sidebar, the landing card grids, and prev/next — so
  adding a page in one place keeps everything in sync and orphan-free.

## Component inventory

### Interactive islands (Preact, `.tsx`, hydrated `client:visible`)

| component | purpose | key props |
| --- | --- | --- |
| `BpbFlopChart` | **The recurring viz.** bpb (y, linear) vs total FLOPs (x, log10). Hover points, toggle series, optional budget/no-model lines + annotations. Role colors match the matplotlib plots. | `series: Series[]`, `budgetLine?`, `budgetLabel?`, `noModelLine?`, `annotations?: {flops,bpb,text,dx?,dy?}[]`, `yMin?`, `yMax?` |
| `StreamScaffold` | Shared chrome for predict-before-reveal demos: stream tape, running-bpb readout + per-byte sparkline, Step/Play/Reset transport + the play timer. Demos supply their prediction panel as children. | `stream`, `pos`, `cumBits`, `history`, `playing`, `onStep`, `onPlay`, `onReset`, `children`, `caption?`, `barColor?`, `stepMs?` |
| `PrequentialStream` | Live online order-0/1 byte model; predicts before reveal, pays −log₂p, adapts. Built on `StreamScaffold`. | _(none)_ |
| `ContextMixingDemo` | Order-0/1/2 specialists + online logistic mixing weights that learn by SGD; watch weights re-allocate. Built on `StreamScaffold`. | _(none)_ |
| `CodeLengthDemo` | Slider for p(true byte) → −log₂p bits, with the cost curve and the 8-bit "no model" line. | _(none)_ |
| `ScalingCalculator` | C = 6·N·D arithmetic with N/D log-sliders, hardware, wall-clock, GPT-3 preset. | _(none)_ |
| `FastWeightDemo` | Associative memory: outer-product write, matvec read, decay/forgetting, live d×V heatmap, crosstalk on similar keys. | _(none)_ |
| `SourceIvScreen` | Toggle the (i)–(iv) sources a candidate claims → per-source FLOP-impact bars + scout/park verdict. | _(none)_ |

### Presentational (`.astro`)

| component | purpose | key props |
| --- | --- | --- |
| `Layout` | Page chrome: top bar, sticky sidebar nav, content column, "See also" rail, prev/next pager. Wrap MDX body in `<Layout …>`. | `title`, `kicker?`, `blurb?`, `related?: {href,title}[]`, `landing?` |
| `Callout` | Styled aside. Variants: `note`, `insight` (the lesson), `caveat` (read honestly), `warning`. | `variant?`, `title?` |
| `Pipeline` | Left-to-right flow of labeled stages + arrows, optional dashed feedback leg (online loops). | `steps: {label,sub?,accent?}[]`, `feedback?` |
| `ConceptMap` | The landing's hand-laid SVG DAG; clickable nodes link every page (the navigational spine). | _(none; static data)_ |
| `CardGrid` | Responsive card grid over `NavItem[]` (landing + experiments index). | `items: NavItem[]` |
| `HarnessPlot` | Embed a matplotlib PNG via `astro:assets`, tagged "harness plot" to distinguish from interactive re-renders. | `src: ImageMetadata`, `alt` |

### Data modules

- `src/data/curves.ts` — `Series` type + datasets `firstFinding`, `contextMixingReference`,
  `prequentialBaseline`, `amortizedBaseline`, `freeUnigram`, constants. Provenance-commented.
- `src/data/nav.ts` — `NavItem`, `concepts`, `experiments`, `order`, `neighbors()`.

## Style & convention decisions

- **Aesthetic:** a research scout's logbook as an *instrument readout* — warm near-black "paper"
  with a faint graph-paper grid; Fraunces (display) + Newsreader (body) + JetBrains Mono (numbers).
  One amber UI accent. **Data role colors are fixed and semantic, matching the harness plots:**
  context-mixing reference = blue, fast-weight = orange, transformer = green, free unigram = violet.
  This keeps the interactive re-renders visually honest against the embedded PNGs.
- **Quality bar per concept page:** intuition → math → worked example (plain language first); ≥1
  interactive viz; cross-links + a "See also"; appears in the concept map + sidebar (no orphans);
  KaTeX math.
- **Rule of two (factored shared viz):** `BpbFlopChart` (6 uses), `Pipeline` (3), `StreamScaffold`
  (2), `HarnessPlot` (2), `CardGrid` (2), `Callout` (everywhere). When a viz pattern recurs, it is
  factored and the first use refactored to match.
- **MDX gotchas (hard-won — keep these):**
  - Wrap page body in `<Layout …>` via `import` + element, **not** the `layout:` frontmatter
    (frontmatter passes props under `frontmatter.*`, but `Layout` reads top-level props).
  - **Display math `$$…$$` must be on a single line.** A multiline `$$` block breaks the MDX parser
    ("Expected a closing tag"). Inline `$…$` is fine and protects its braces from JSX.
  - **Escape literal `<` in prose/tables as `&lt;`** — MDX reads `<x` as a JSX tag start (bit us on
    a `wins (fw<tr)` table header).
  - Numbers belong in `curves.ts`, never inline in a page.
- **Honesty:** harness-produced plots are embedded via `HarnessPlot` (labelled "harness plot");
  reconstructed coordinates (e.g. 0.1's intermediate eval x-positions, which the notes don't report)
  are flagged in the chart caption **and** the `reconstructed` flag on the series.

## Pages (status: all built, build green)

- **Concepts (6):** loss-per-flop-and-scaling-laws, compression-equals-prediction,
  prequential-evaluation, source-iv-advantage, fast-weight-memory, context-mixing.
- **Experiments (5 + log index):** 0.1-baseline-harness-smoke, 0.2-prequential-baseline,
  context-mixing-reference (embeds `context-mixing-reference.png`), A.1-fast-weight-memory,
  first-finding-pareto (embeds `unified-leaderboard.png` + interactive 3-way chart).
- **Landing:** `index.mdx` with the `ConceptMap` and card grids.

## Flagged for researchers (RESOLVED 2026-06-19 by Main)

Both discrepancies I raised were investigated by Main and reconciled — kept here as a record.

1. **Transformer baseline bpb disagreed between notes at mid budgets — RESOLVED.** Root cause: the
   original `unified-leaderboard` run used `seq_len=64` while the canonical `0.2` / `A.1` runs use
   `seq_len=128`. Main regenerated the leaderboard at `seq_len=128`; all pages now agree. The
   crossover flips vs my first build: `fast_weight` **wins** at b0/b2×10⁹/10¹⁰ (−0.59 / −0.30 /
   −0.10) and **loses only** at 4×10¹⁰ (+0.19); the free reference (4.78) is below both neural
   curves through 10¹⁰. `curves.ts` `firstFinding` and the first-finding page now carry the
   canonical numbers (which equal the A.1 table), so the A.1 chart — which reuses `firstFinding` —
   is self-consistent too.
2. **Context-mixing reference: near-identical FLOPs, different bpb — EXPLAINED (not a bug).** The
   0.3 order-sweep ran on the 800 B English sample at order 0..1 (4.3922 bpb @ 4.28×10⁶); the
   unified run used the 512 B clone tail at the harness default order (4.7779 bpb @ 4.283×10⁶) —
   different stream length *and* max-order, so the ~equal total FLOPs is coincidental. Each page is
   faithful; a clarifying caveat was added to the context-mixing-reference page.

## Changelog

- **2026-06-19 (session 1 — scaffold + first pages):** Astro 6 + MDX + Preact scaffold, `.gitignore`,
  KaTeX via `markdown.processor`. Design system (`global.css`), `Layout`, `curves.ts`, `nav.ts`.
  Built `BpbFlopChart`, `Callout`, `Pipeline`, `ConceptMap`, `CardGrid`. Landing + concept pages:
  compression, loss-per-flop (+`ScalingCalculator`), prequential (+`PrequentialStream`), source-iv
  (+`SourceIvScreen`).
- **2026-06-19 (session 2 — finish):** committed source-iv page/screen. Added `FastWeightDemo` +
  fast-weight page; `ContextMixingDemo` + context-mixing page; factored `StreamScaffold` and
  refactored `PrequentialStream` onto it (rule of two). Added `HarnessPlot` + `freeUnigram` data.
  Authored all 5 experiment pages + the log index, embedding both harness plots and re-rendering
  the leaderboard as an interactive 3-way `BpbFlopChart` with a Pareto annotation. Wrote this
  `PROCESS.md`. Recorded MDX gotchas and two flagged source discrepancies. Build green (13 pages).
- **2026-06-19 (session 3 — canonical reconciliation):** merged `master` (corrected
  `first-finding-pareto.md` + regenerated `unified-leaderboard.png/.md` at `seq_len=128`).
  Re-copied the regenerated plot into `src/assets`, updated `curves.ts` `firstFinding` and the
  first-finding page (table, prose, annotation) to the canonical numbers — the fast-weight
  crossover now flips (wins through 10¹⁰, loses only at 4×10¹⁰). Added a "two reference runs"
  caveat to the context-mixing-reference page. Both flagged discrepancies marked resolved. Build
  green (13 pages).
