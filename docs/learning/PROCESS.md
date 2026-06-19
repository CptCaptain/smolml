# docs-builder process log

My working memory across sessions. Read `CONTRIBUTING.md` first (the charter), then this. I own
`docs/learning/` only; I never touch harness/research code. The published site is **MDX built with
Astro** (ADR 0005); the plain-`.md` notes under `concepts/` and `experiments/` are the researchers'
prose source — I keep them and author the interactive pages under `src/`.

## How the site is wired

**HARD CONSTRAINT (user requirement): the built site must work opened directly from disk
(`file://`), with no server.** That shapes the whole toolchain below.

- **Stack:** Astro 6 + `@astrojs/mdx` (no UI framework — see interactivity). KaTeX via `remark-math`
  + `rehype-katex` through the Astro 6 `markdown.processor = unified({…})` API (MDX inherits it).
  Fonts self-hosted via `@fontsource-variable` (no runtime CDN).
- **Interactivity = classic scripts, not islands.** Browsers block ES-module (`type="module"`)
  hydration from a `file://` origin, so there are **no framework islands**. Every interactive viz
  is a thin `data-widget` marker (`src/components/*.astro`) mounted by ONE classic script,
  `public/js/compendium.js` (vanilla JS, no imports/no fetch, included once via `Layout`). State
  lives in a closure; interaction re-renders via `innerHTML` with listeners **delegated on the
  widget root** (so they survive re-render); the chart updates its tooltip in place to preserve
  keyboard focus.
- **CSS inlined** (`build.inlineStylesheets: "always"`) — no external stylesheet, so no CSS 404 on
  `file://`. Widget CSS lives in `global.css` (global scope, because runtime-built DOM never
  receives Astro-scoped styles).
- **Relative paths:** `scripts/relativize.mjs` (chained in the `build` script) rewrites every
  absolute asset/link and `url(/…)` in the built HTML to a page-relative path; links built at
  runtime by `compendium.js` use the `data-root` prefix that `Layout` sets per page. So `file://`
  resolves everything (no server root, no directory-index magic).
- **Build:** `npm run build` (= `astro build && node scripts/relativize.mjs`) is the PR gate.
  `npm run dev` to author (relative paths work over http too).
- **Single source of truth for numbers:** `src/data/curves.ts` (provenance-commented). The chart
  marker serializes the relevant dataset into an inline JSON `<script>` the widget reads.
- **Sitemap:** `src/data/nav.ts` drives sidebar, landing cards, and prev/next — orphan-free.
- **To view the build:** open `dist/index.html` (or any `dist/**/index.html`) directly in a
  browser. No server.

## Component inventory

### Interactive widgets (vanilla, mounted by `public/js/compendium.js`)

Each is a thin `.astro` marker `<div data-widget="…">`; the chart also embeds an inline JSON
`<script>` of its build-time data. `compendium.js` auto-mounts every `[data-widget]` on load.
Only the chart takes data; the rest are self-contained.

| widget (`data-widget`) | purpose | data passed |
| --- | --- | --- |
| `chart` (`BpbFlopChart.astro`) | **The recurring viz.** bpb (y, linear) vs total FLOPs (x, log10). Hover/keyboard-focus points (tooltip in place, focus preserved), toggle series, optional budget/no-model lines + annotations. Role colors match the matplotlib plots. | JSON: `series`, `budgetLine?`, `budgetLabel?`, `noModelLine?`, `annotations?`, `yMin?`, `yMax?` |
| `prequential` (`PrequentialStream.astro`) | Live online order-0/1 byte model; predict-before-reveal, pays −log₂p, adapts. Shares the stream scaffold (tape/readout/sparkline/transport) inside `compendium.js`. | _(none)_ |
| `contextmixing` (`ContextMixingDemo.astro`) | Order-0/1/2 specialists + online logistic mixing weights (SGD); shares the stream scaffold. | _(none)_ |
| `codelength` (`CodeLengthDemo.astro`) | Slider for p(true byte) → −log₂p bits, cost curve + 8-bit "no model" line. | _(none)_ |
| `scaling` (`ScalingCalculator.astro`) | C = 6·N·D arithmetic with N/D log-sliders, hardware, wall-clock, GPT-3 preset. | _(none)_ |
| `fastweight` (`FastWeightDemo.astro`) | Associative memory: outer-product write, matvec read, decay/forgetting, live d×V heatmap, crosstalk on similar keys. | _(none)_ |
| `sourceiv` (`SourceIvScreen.astro`) | Toggle the (i)–(iv) sources a candidate claims → FLOP-impact bars + scout/park verdict. | _(none)_ |

The shared **stream scaffold** and **chart** logic live as functions in `compendium.js` (rule of
two: one chart routine for 8 chart instances; one stream scaffold for both stream demos).

### Presentational (`.astro`)

| component | purpose | key props |
| --- | --- | --- |
| `Layout` | Page chrome: top bar, sticky sidebar nav, content column, "See also" rail, prev/next pager. Wrap MDX body in `<Layout …>`. | `title`, `kicker?`, `blurb?`, `related?: {href,title}[]`, `landing?` |
| `Callout` | Styled aside. Variants: `note`, `insight` (the lesson), `caveat` (read honestly), `warning`. | `variant?`, `title?` |
| `Pipeline` | Left-to-right flow of labeled stages + arrows, optional dashed feedback leg (online loops). | `steps: {label,sub?,accent?}[]`, `feedback?` |
| `ConceptMap` | The landing's hand-laid SVG DAG; clickable nodes link every page (the navigational spine). Edges are smooth cubic-Bézier `<path>`s (orientation-aware control points, Δ=0.4·span) with arrowheads; the experiment "bus" stays orthogonal. | _(none; static data)_ |
| `CardGrid` | Responsive card grid over `NavItem[]` (landing + experiments index). | `items: NavItem[]` |

### Data modules

- `src/data/curves.ts` — `Series` type (roles incl. `pc_refine`, `warm`) + datasets `firstFinding`,
  `contextMixingReference`, `prequentialBaseline`, `amortizedBaseline`, `freeUnigram`,
  `surpriseGatedPc` (B.1), `warmedMixing` + `gatedMix` (B.2), constants. Provenance-commented.
- `src/data/nav.ts` — `NavItem`, `concepts`, `experiments`, `order`, `neighbors()`.

## Style & convention decisions

- **Aesthetic:** a research scout's logbook as an *instrument readout* — warm near-black "paper"
  with a faint graph-paper grid; Fraunces (display) + Newsreader (body) + JetBrains Mono (numbers).
  One amber UI accent. **Data role colors are fixed and semantic, matching the harness plots:**
  context-mixing reference = blue, fast-weight = orange, transformer = green, free unigram = violet,
  predictive-coding refinement = rose (`--c-pc`; B.1 — the harness plots `pc_refine` in tab:orange,
  but orange is reserved here for fast-weight, so PC got its own token + chart/map/swatch color and
  never reads as the memory).
  warm_mix = warm **vermilion** (`--c-warm` #e0654d; B.2 — same precedent as PC: the harness plots
  `warm_mix` in tab:orange (fast-weight's color), and gold/amber would collide with the UI accent,
  so it earned its own vermilion token; its cold point renders in the `reference` blue, drawn over
  the warm curve's cold start, so "cold == the context-mixing reference" reads visually).
  This keeps the interactive re-renders visually honest against the embedded PNGs.
- **Quality bar per concept page:** intuition → math → worked example (plain language first); ≥1
  interactive viz; cross-links + a "See also"; appears in the concept map + sidebar (no orphans);
  KaTeX math.
- **Rule of two (factored shared viz):** `BpbFlopChart` (11 uses), `Pipeline` (6), `CardGrid` (2),
  `Callout` (everywhere); inside `compendium.js`, one chart routine serves all 11 chart instances and
  one stream scaffold serves both stream demos. When a viz pattern recurs it is factored.
- **MDX gotchas (hard-won — keep these):**
  - Wrap page body in `<Layout …>` via `import` + element, **not** the `layout:` frontmatter
    (frontmatter passes props under `frontmatter.*`, but `Layout` reads top-level props).
  - **Display math `$$…$$` must be on a single line.** A multiline `$$` block breaks the MDX parser
    ("Expected a closing tag"). Inline `$…$` is fine and protects its braces from JSX.
  - **Escape literal `<` in prose/tables as `&lt;`** — MDX reads `<x` as a JSX tag start (bit us on
    a `wins (fw<tr)` table header).
  - **Use literal Unicode glyphs (→ — “ ” − ₂ Σ ÷ …), never `\uXXXX` escapes, in `src/`.** JSX
    plain-string attributes and MDX prose do **not** interpret JS `\uXXXX` (only `{}` expressions
    and `.ts`/frontmatter do), so an escape in an attribute or prose renders literally as text.
    Glyphs are safe in every context (JS strings, JSX attrs, prose, KaTeX). `public/js/*.js` is
    exempt — classic JS interprets `\u` fine.
  - Numbers belong in `curves.ts`, never inline in a page.
- **Honesty & one plot per page:** each chart page shows exactly **one** plot — the interactive
  `BpbFlopChart` (no duplicate static PNG beside it). The harness-produced PNGs still live under
  `docs/learning/experiments/` as research artifacts but are not embedded. Reconstructed
  coordinates (e.g. 0.1's intermediate eval x-positions, which the notes don't report) are flagged
  in the chart caption **and** the `reconstructed` flag on the series.
- **Accessibility (WCAG AA):** `--faint` (#6f6650, ~3:1 on the ink bg) is **decoration only**
  (borders, gridlines, swatch fills) — never small text; use `--muted` (#9a8e76, ~5.7:1) or lighter
  for any actual text. Interactive chart marks (`BpbFlopChart`) are keyboard-focusable
  (`tabIndex`, `role="button"`, per-point `aria-label`, focus mirrors hover so the tooltip opens on
  focus) with a visible focus ring — fixed once in the shared component, so all chart uses benefit.

## Pages (status: all built, build green)

- **Concepts (8):** loss-per-flop-and-scaling-laws, compression-equals-prediction,
  prequential-evaluation, source-iv-advantage, fast-weight-memory, context-mixing,
  predictive-coding, online-warmup.
- **Experiments (7 + log index):** 0.1-baseline-harness-smoke, 0.2-prequential-baseline,
  context-mixing-reference, A.1-fast-weight-memory, B.1-surprise-gated-pc-refinement,
  B.2-warmed-mixing, first-finding-pareto. Each renders the shared interactive `BpbFlopChart`
  (one plot per page; B.2 is the lone two-plot page — Phase 1 + Phase 2 are distinct experiments;
  harness PNGs stay as artifacts under `experiments/`, not embedded — session 6).
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
- **2026-06-19 (session 4 — cross-vendor review fixes, Codex APPROVE-WITH-FIXES):** (MAJOR 1) added
  the 5 individual experiment pages to `ConceptMap` as a connector-bus of leaf nodes off the log
  node — every experiment is now reachable from the map, not just the sidebar/cards. (MAJOR 2)
  made `BpbFlopChart` marks keyboard-focusable (tabIndex/role/aria-label, focus mirrors hover,
  visible focus ring) — once in the shared component. (MAJOR 3) routed all `--faint` *text* to
  `--muted` for WCAG AA (`.fw-dim`, `.fw-key-hint`, `.siv-empty`, `.strm-ch.future`, nav/rail
  labels); `--faint` is now decoration-only. (MINOR) removed the broken `check` npm script;
  reworded first-finding's imprecise "1000× cheaper" to "40×–2300× cheaper through 10¹⁰". Build
  green (13 pages); nothing changed outside docs/learning.
- **2026-06-19 (session 5 — file:// re-platform, no server):** new hard requirement: the build must
  work opened directly from disk. ES-module island hydration is blocked from `file://`, so I
  removed `@astrojs/preact`/`preact` and the island system entirely and re-implemented all 7
  interactive widgets as one **classic vanilla script** `public/js/compendium.js` (no
  imports/fetch), auto-mounting `data-widget` markers (the former `.tsx` islands became thin
  `.astro` markers; their CSS moved to `global.css`). Set `build.inlineStylesheets:"always"` (no
  CSS 404s), added `scripts/relativize.mjs` (chained into `build`) to rewrite every absolute
  asset/link/`url()` to page-relative, and a `data-root` prefix on `<html>` for links the widget
  script builds at runtime. Verified from `file://` in a headless browser: index + a concept page +
  first-finding load fully styled, sidebar/map links navigate, every widget renders **and** responds
  (slider, stream step, legend toggle, fast-weight recall, source-iv verdict), both harness plots
  load, math renders, **zero console errors / no ERR_FILE_NOT_FOUND**. Build green (13 pages).
- **2026-06-19 (session 6 — user polish):** (1) one plot per page — removed the duplicate static
  `HarnessPlot` PNG embed sitting beside the interactive `BpbFlopChart` on first-finding and
  context-mixing-reference; deleted `HarnessPlot.astro`, the dead PNG asset imports, and the
  `src/assets/` copies (the originals stay under `experiments/` as artifacts). (2) Swept every
  literal `\uXXXX` escape in `src/` (32 across 7 files) to actual glyphs — JSX attrs/MDX prose
  don't interpret `\u`, so they were rendering as text (e.g. fast-weight's `key\u2192byte`). Build
  green (13 pages); verified from `file://`: no rendered text contains a `\u` escape, and the two
  pages show exactly one (interactive) chart with no `<img>`. `git diff` confined to docs/learning.
- **2026-06-19 (session 7 — curved concept-map edges):** converted the `ConceptMap` DAG edges from
  straight `<line>`s to smooth cubic-Bézier `<path>`s via a new `edgePath()` helper — control points
  offset along the dominant axis (Δ=0.4·span) so each curve departs/arrives perpendicular to the
  node edge (no kinks, no bowing into boxes); arrowheads (`orient="auto-start-reverse"`) and dashed
  feedback/log edges preserved; the experiment bus left orthogonal. Build green; verified from
  `file://` (zero console errors) and screenshotted the landing map to confirm clean curves.
- **2026-06-19 (session 8 — B.1 surprise-gated PC pages):** authored two pages from the researcher
  notes — concept `predictive-coding` (intuition → free-energy/settling KaTeX → worked example →
  surprise-gating; `Pipeline` loop + reused interactive `BpbFlopChart`) and experiment
  `B.1-surprise-gated-pc-refinement` (mirrors A.1: status/bet/setup/FLOP-honesty incl. the
  cross-vendor undercharge fix/4-way result table/verdict/learnings). Added a new semantic data role
  `pc_refine` + `--c-pc` rose token across `curves.ts`, `compendium.js` `ROLE_COLOR`, `global.css`
  (swatch) and `ConceptMap` (`role-pc`) so PC never reads as fast-weight's orange; added the
  `surpriseGatedPc` dataset (4 entrants on the identical 1200 B stream). Wired both into `nav.ts`
  (CONCEPT 07, EXP B.1), the `ConceptMap` (new `Predictive coding` node off source-iv + dashed leg to
  the log; B.1 leaf on the experiment bus, re-spaced to 6 leaves at `LEAF_W=150`), the experiments
  index (auto via nav), and reciprocal See-also links on source-iv / prequential / loss-per-flop. No
  new component factored — rule of two not triggered (reused `BpbFlopChart` / `Callout` / `Pipeline`
  / `ConceptMap` / `CardGrid`); `BpbFlopChart` now 8 uses, `Pipeline` 5. Build green (15 pages);
  verified from `file://` (zero console errors): map shows the rose PC node + B.1 leaf with no
  overlaps, the B.1 chart renders all 4 marks/legend/annotations, KaTeX renders, no `\u` leaks, all
  cross-links resolve. **Viz note:** the gated−uniform matched-FLOP lever (−0.0045 bpb) is sub-pixel
  on a log-FLOP axis (the three pretrained entrants share x), so the chart carries the macro Pareto
  story and the result table carries the lever (flagged in both captions). Researcher note found
  internally consistent — no science flagged.
- **2026-06-19 (session 9 — B.2 warmed-mixing pages):** authored two pages from the researcher notes
  — concept `online-warmup` (amortized vs transductive vs warm-start intuition → smoothed-frequency
  KaTeX → why it's nearly free → why it's a legit Source-(iv) move; `Pipeline` + reused interactive
  `BpbFlopChart`) and experiment `B.2-warmed-mixing` (transductive-handicap framing; Phase 1 —
  `warm_mix` strictly dominates the transformer per FLOP, the project's FIRST genuine per-FLOP win,
  on real enwik8, headline `insight` Callout; the order curve; Phase 2 — `gated_mix` honestly
  Pareto-hollow, `caveat` Callout spelling out WHY [gate overhead on an already-cheap mix];
  FLOP-honesty/verdict/learnings). Added a new semantic data role `warm` + `--c-warm` vermilion token
  across `curves.ts`, `compendium.js` `ROLE_COLOR`, `global.css` (swatch) and `ConceptMap`
  (`role-warm`) — same precedent as `pc_refine`. Added the `warmedMixing` (Phase 1: transformer point
  + warm_mix curve + a `reference`-blue cold marker drawn over the warm curve's cold start) and
  `gatedMix` (Phase 2: fixed-order warm_mix frontier + dominated `neutral`-gray gated curve) datasets.
  Wired both into `nav.ts` (CONCEPT 08, EXP B.2), the `ConceptMap` (new `Online warm-start` node off
  context-mixing; B.2 leaf on the experiment bus, re-spaced to 7 leaves at `LEAF_W=128`), the
  experiments index (auto via nav), and reciprocal See-also links on context-mixing / source-iv /
  prequential / loss-per-flop / context-mixing-reference / first-finding. No new component factored —
  rule of two not triggered (reused `BpbFlopChart` / `Callout` / `Pipeline` / `ConceptMap` /
  `CardGrid`); `BpbFlopChart` now 11 uses, `Pipeline` 6. Build green (17 pages); verified from
  `file://` (zero console errors): both charts mount, the Phase-1 blue cold dot sits at the warm
  curve's origin, every Phase-2 gated point reads as dominated, KaTeX renders (18 nodes on B.2), no
  `\u` leaks, map shows the warm node + B.2 leaf with no overlaps, all cross-links resolve. **Viz
  note:** Phase 2's real FLOP span is narrow (1.23–1.57×10⁹, one log decade) so the chart shows a
  single x tick — faithful to the data; the domination is carried by the vertical separation + the
  caption. Researcher note found internally consistent — no science flagged.
