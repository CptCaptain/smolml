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
| `chart` (`BpbFlopChart.astro`, `RegretFlopChart.astro`) | **The recurring viz.** A y-metric vs total FLOPs (x, log10) scatter; lower-left = better. Hover/keyboard-focus points (tooltip in place, focus preserved), toggle series, optional budget/no-model lines + annotations. y defaults to **bpb** (linear); an optional `metric` config (`yKey`/`yLabel`/`tipLabel`/`valDecimals`/`yTickStep`/`aria`) repoints it — `RegretFlopChart` uses it for **regret** (C.A control rung). Role colors match the matplotlib plots. | JSON: `series`, `annotations?`, `yMin?`, `yMax?`, `budgetLine?`, `budgetLabel?`, `noModelLine?`, `metric?` |
| `prequential` (`PrequentialStream.astro`) | Live online order-0/1 byte model; predict-before-reveal, pays −log₂p, adapts. Shares the stream scaffold (tape/readout/sparkline/transport) inside `compendium.js`. | _(none)_ |
| `contextmixing` (`ContextMixingDemo.astro`) | Order-0/1/2 specialists + online logistic mixing weights (SGD); shares the stream scaffold. | _(none)_ |
| `codelength` (`CodeLengthDemo.astro`) | Slider for p(true byte) → −log₂p bits, cost curve + 8-bit "no model" line. | _(none)_ |
| `scaling` (`ScalingCalculator.astro`) | C = 6·N·D arithmetic with N/D log-sliders, hardware, wall-clock, GPT-3 preset. | _(none)_ |
| `fastweight` (`FastWeightDemo.astro`) | Associative memory: outer-product write, matvec read, decay/forgetting, live d×V heatmap, crosstalk on similar keys. | _(none)_ |
| `sourceiv` (`SourceIvScreen.astro`) | Toggle the (i)–(iv) sources a candidate claims → FLOP-impact bars + scout/park verdict. | _(none)_ |
| `controlrollout` (`ControlRollout.astro`) | **In-context control / chemotaxis rung.** Scrubbable instrument over one trained held-out `ChemoEnv` rollout: regret/reward scoreboard, an unrolled W-cell field bar (concentration heat + hidden-peak ▼ / agent ▲ markers), a spacetime raster (time ↓) tracing the agent path (green) chasing the peak path (blue dashed, wrap-broken at the ring seam), and a cumulative-reward spark vs perfect/random reference lines. Step/Play/Reset/scrub; built once then mutated in place (the 33×16 raster never re-renders). | inline JSON: the full `icl_control_rollout.json` (read from `public/` at build time by the marker) |
| `autocompleterace` (`AutocompleteRace.astro`) | **Live byte loss-per-FLOP race — consumes the model layer.** Three byte models (`context_mixing` blue, `delta_mix` orange, `transformer` green) run predict-then-learn on the SAME editable enwik8 stream: per-model top-6 next-char bars, cumulative bpb (Σ −log₂ p ÷ bytes; byte 0 = 8 bits), top-1 hit-rate, and HUD FLOPs/byte (~9.5k … ~11.5M). Editable seed (latin-1 → bytes via `charCodeAt&0xff`), Step/Play/Reset; shell built once, panels mutate per step. | inline JSON: `seedText` + each model's `config`/`weights` (byte fixtures + `transformer.weights.json`, read at build time) merged with HUD facts from `curves.ts` |
| `cursorchase` (`CursorChase.astro`) | **Interactive chemotaxis cursor-follow — consumes the model layer.** A `<canvas>` concentration field whose peak **drifts toward your cursor at ≤1 cell/tick** (ChemoEnv's drift); three controllers (`chemotaxis_min` green, `reservoir` orange, `reservoir_plastic` rose) sense only their own cell and chemotax to chase it (parity tape: even = sense, odd = action). HUD: params + FLOPs/step (66 vs ~10k) + live mean/cumulative reward (leader-highlighted); checkboxes toggle each marker. Step/Play/Reset; canvas redraws per tick. | inline JSON: `env` + each controller's `config`/`weights` (chemotaxis fixture + reservoir `*.weights.json`) merged with HUD facts from `curves.ts` |

The shared **stream scaffold** and **chart** logic live as functions in `compendium.js` (rule of
two: one chart routine for **17** chart instances — 13 bpb (`BpbFlopChart`) + 4 regret (`RegretFlopChart`); one stream scaffold for both stream demos; one `wireTransport` helper — Step/Play/Reset on a timer, + a shared `.demo-transport` button style — for the two live model demos).

### Presentational (`.astro`)

| component | purpose | key props |
| --- | --- | --- |
| `Layout` | Page chrome: top bar, sticky sidebar nav, content column, "See also" rail, prev/next pager. Wrap MDX body in `<Layout …>`. The `scripts?: string[]` prop injects extra classic scripts (deferred, in document order) into `<head>` **before** `compendium.js` — the two model-demo pages use it to load the interactive-demo model layer (`/js/models/*.js`) so the widget runtime sees `SmolModels`/`SmolDemos` at mount. | `title`, `kicker?`, `blurb?`, `related?: {href,title}[]`, `landing?`, `scripts?` |
| `Callout` | Styled aside. Variants: `note`, `insight` (the lesson), `caveat` (read honestly), `warning`. | `variant?`, `title?` |
| `Pipeline` | Left-to-right flow of labeled stages + arrows, optional dashed feedback leg (online loops). | `steps: {label,sub?,accent?}[]`, `feedback?` |
| `ConceptMap` | The landing's hand-laid SVG DAG; clickable nodes link every page (the navigational spine). Edges are smooth cubic-Bézier `<path>`s (orientation-aware control points, Δ=0.4·span) with arrowheads; the experiment "bus" stays orthogonal. | _(none; static data)_ |
| `CardGrid` | Responsive card grid over `NavItem[]` (landing + experiments index). | `items: NavItem[]` |
| `HBars` | Horizontal labeled value bars: a row per bar, fill width ∝ a precomputed `pct`, value printed in the fill, colored by an `accent` class (`.genbar-fill.<accent>` in `global.css`). Caller sets the scale + says (in the caption) whether longer = better. Used by B.4 (delta-vs-abstain bpb, lower=shorter) + the C.A within-episode reward bars (higher=longer). | `bars: {label,pct,text,accent}[]` |

### Data modules

- `src/data/curves.ts` — `Series` type (roles incl. `pc_refine`, `warm`, and the control roles
  `reservoir` / `reservoir_plastic` / `chemotaxis` / `forage_min`; `CurvePoint` carries optional `bpb?` **or**
  `regret?`) + datasets `firstFinding`, `contextMixingReference`, `prequentialBaseline`,
  `amortizedBaseline`, `freeUnigram`, `surpriseGatedPc` (B.1), `warmedMixing` + `gatedMix` (B.2),
  `hashedMixFull` (B.3), `deltaMix` + `deltaFull` (B.4), `controlCandidates` / `reservoirControl`
  / `chemotaxisControl` (C.A chemotaxis) / `forageControl` (C.A.4 forage) — all regret-vs-FLOP — and `demoByteModels` + `demoControlModels` (the two
  live-demo HUD fact tables — params / FLOPs-per-step / role / refBpb, from the model-layer README),
  constants. Provenance-commented.
- `src/data/nav.ts` — `NavItem`, `concepts`, `experiments`, `order`, `neighbors()`.
- `public/icl_control_rollout.json` — a sample **trained, held-out** `ChemoEnv` rollout the harness
  writes (fields: `width, levels, horizon, mu[], pos[], conc_token[], reward[], action[], field[][],
  mean_reward, regret`). `ControlRollout.astro` reads it at build time (anchored on `process.cwd()`,
  not `import.meta.url` — the latter points at the bundled chunk under `dist/` at prerender) and
  serializes it into the inline JSON `<script>` the `controlrollout` widget reads.

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
  Control-rung candidates (C.A) extend the same precedent: reservoir family = **indigo**
  (`--c-reservoir` #8f86d6, frozen C.A.1) with a lighter **lavender** sibling (`--c-plastic` #b3a4e8,
  online-plastic C.A.1b — same hue = same family), chemotaxis_min = **teal** (`--c-chemo` #3aa890),
  and forage_min = **magenta** (`--c-forage` #cf72b3, C.A.4 — the per-type contingency tracker; magenta
  sits in the otherwise-unused ~300–340° gap between rose and violet, and never co-occurs with either on a
  chart or adjacent on the map). All four are otherwise-unused hues (no collision with
  blue/orange/green/violet/rose/vermilion/amber), flat semantic mark colors (no gradient/glow — not the
  cyan-on-dark slop pattern), and light enough to host dark bar-fill text. They appear on the **regret**
  charts only (no bpb-axis collision) and as `ConceptMap` leaf borders + `.swatch`es. C.A.4's memory-parity
  control `forage_reservoir` **reuses** the C.A.1b reservoir-family lavender (`--c-plastic`) — it IS that
  `reservoir_plastic` mechanism ported to forage, so no new token (KISS).
- **Interactive model demos (C.A) reuse existing role tokens — no new color.** The byte race keeps
  `context_mixing` = reference blue, `delta_mix` = fast_weight orange (it IS a delta-rule fast weight, the
  B.4 lineage), `transformer` = transformer green. The cursor chase keeps the in-context-control page's
  `ControlRollout` legend — peak/cursor = reference blue, concentration heat = amber accent — and gives the
  three controllers `chemotaxis_min` = transformer green (the "agent", as in `ControlRollout`),
  `reservoir` = fast_weight orange, `reservoir_plastic` = pc rose. (The static regret charts use the new
  indigo/lavender/teal control tokens above; the live demos deliberately reuse existing tokens — a known
  cross-viz color divergence for the same models, left for a future harmonization pass.) Both demos
  **consume** the engineer's parity-gated model layer (`public/js/models/*.js`, `public/data/demos/*`)
  read-only; the markers inline `config`/`weights`/`seed` at build time (escaping `<` so the seed's `<!--`
  cannot break the inline JSON) exactly like `ControlRollout`. The live transformer is the *trained* demo
  export (low bpb, huge FLOPs) — flagged distinct from the *untrained* transformer at ~8.0 bpb in the
  context-mixing curve.
- **Quality bar per concept page:** intuition → math → worked example (plain language first); ≥1
  interactive viz; cross-links + a "See also"; appears in the concept map + sidebar (no orphans);
  KaTeX math.
- **Rule of two (factored shared viz):** `BpbFlopChart` (13 uses) + `RegretFlopChart` (4 uses) are two
  thin markers over **one** `mountChart` routine (parametrized by an optional `metric` config; absent ⇒
  the bpb defaults, so all 13 bpb charts are byte-unchanged). `HBars` (4 uses — factored from B.4's inline
  `.genbars` two-bar, first use refactored, + the three C.A reward ladders C.A.1 / C.A.2 / C.A.4),
  `Pipeline` (8), `CardGrid` (2), `Callout` (everywhere); inside `compendium.js`, one chart routine serves all 17 chart instances, one
  stream scaffold serves both stream demos, and one `wireTransport` helper (+ shared `.demo-transport`
  button style) serves the two live model demos. The older bespoke transports in
  `mountStream`/`mountControlRollout` were left as-is (their scrub/loop semantics differ; not
  retrofitted, to avoid churning shipped parity-adjacent widgets). When a viz pattern recurs it is factored.
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

- **Concepts (9):** loss-per-flop-and-scaling-laws, compression-equals-prediction,
  prequential-evaluation, source-iv-advantage, fast-weight-memory, context-mixing,
  predictive-coding, online-warmup, in-context-control.
- **Experiments (12 + log index):** 0.1-baseline-harness-smoke, 0.2-prequential-baseline,
  context-mixing-reference, A.1-fast-weight-memory, B.1-surprise-gated-pc-refinement,
  B.2-warmed-mixing, B.3-hashed-mix-full-corpus, B.4-delta-mix, C.A.1-reservoir-control,
  C.A.2-chemotaxis-min-control, C.A.4-forage-local-learners, first-finding-pareto. The bpb pages render the
  shared interactive `BpbFlopChart`; the three C.A control pages render `RegretFlopChart` (same routine, regret on y).
  (One plot per page; B.2 is the lone two-plot page — Phase 1 + Phase 2 are distinct experiments;
  harness PNGs stay as artifacts under `experiments/`, not embedded — session 6.)
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

**New (2026-06-23, C.A control candidates) — minor, flagged not fixed:** the findings note states the
transformer "can push regret to 0.141 only by spending 2.96×10¹² FLOPs (**16 OOM more**)", but
2.96×10¹² ÷ 2.70×10⁵ ≈ 1.1×10⁷, i.e. **~7 OOM**, not 16 (the cheapest-point gap is ~6 OOM, also stated and
correct). The C.A.2 page uses the arithmetically-correct **~7 OOM**; the "16 OOM" wording was flagged to
Main rather than reproduced. No other numbers touched.

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
- **2026-06-19 (session 10 — B.3 hashed-mix full-corpus page):** authored the experiment page
  `B.3-hashed-mix-full-corpus` from the researcher note, mirroring B.2's layout/frontmatter/imports.
  Tells the engineering-unlock story: the OOM blocker (~58 GB unbounded order-6 dicts on a full-95 MB
  warmup), the fixed-memory hashed-table fix (PAQ/cmix-style $2^{20}$-slot table, `hash_min_order=4`;
  a behavior-preserving refactor that kept the cold reference bit-identical, plus the cross-vendor
  halve-on-overflow charge + order-≤8 guard), the full-ADR-carve table (3 LANDED points + 2 rows
  clearly marked *computing*), the interpretation, and the what-we-learned. Headline `insight` Callout
  (the order-6 win scales under bounded memory — the unlock was engineering, not a new mechanism) and a
  `caveat` Callout spelling out that the two pending rows carry no numbers. **No new component, role,
  or token** — rule of two not triggered: reused `BpbFlopChart` / `Callout` / `ConceptMap`, the
  existing `--c-warm` (vermilion) for the bounded order-6 `hashed_mix` family and the `reference` blue
  for the order-3 cold point. Added the `hashedMixFull` dataset to `curves.ts` (3 landed points only;
  the 2 pending entrants are NOT plotted — they live as *computing* table rows, no invented
  coordinates). Wired into `nav.ts` (EXP B.3, between B.2 and first-finding), the experiments index
  (auto via nav), the `ConceptMap` (8th experiment leaf `B.3 hashed mix` on the bus, re-spaced 7→8 at
  `LEAF_W=112`), and reciprocal See-also links (B.2 ↔ B.3 in both the rail and prose; online-warmup →
  B.3). `BpbFlopChart` now 12 uses. Build green (18 pages); verified from `dist/`: B.3 page built, 27
  KaTeX spans, chart JSON carries both series, zero `\u` leaks, *computing* cells render literally, and
  B.3 resolves from the concept map / experiments index / B.2 / online-warmup. **Viz note:** unlike
  B.2 (where the blue dot was warm_mix's own cold start), here the blue `reference` point is a
  *different, cheaper* model (order-3) — not the cold start of the order-6 curve; the caption says so
  explicitly so the "order-6 beats order-3 per FLOP" reading stays honest. The 3 landed points span
  &lt;1 FLOP decade ($4.74\times10^{10}$–$1.78\times10^{11}$), wider than B.2 Phase 2, so the log-x
  separation reads cleanly; peak RAM (a new dimension) stays in the table, not on the axes (flagged in
  the caption). Researcher note found internally consistent — no science flagged.
- **2026-06-19 (session 11 — C.A.0 in-context-control concept page):** authored the concept
  `in-context-control` from the researcher note — the chemotaxis "control" rung of the graded ICL
  suite. Intuition (bacterial run-and-tumble) → `ChemoEnv` math (Gaussian bump on a `W=16` ring,
  `L=8` levels, drifting peak, local-only sensing, obs ≡ reward, held-out disjoint drift pools) →
  the action/feedback token tape → the metric (regret-vs-oracle headline per total FLOP at fixed
  params + a `caveat` that world-model bits are policy-conditional) → Algorithm Distillation →
  the interactive rollout → the bar table (148,608-param baseline; regret falls / reward rises with
  distillation steps; `insight` Callout) → an honest-limitation `caveat` (v1 distills a *stationary
  reactive* source, so the rung demonstrates climb-and-track on held-out dynamics but does **not yet
  isolate** in-context drift-rate inference — do not overclaim). Two `Pipeline`s (env loop +
  distillation), reused `Callout`. **New single-use widget `ControlRollout`** (`controlrollout`) +
  `mountControlRollout` in `compendium.js` + a `.ctrl-*` block in `global.css`; it reuses the
  existing role colors (transformer green = agent, reference blue = hidden peak, amber heat =
  concentration) — **no new data role or token** (the rung adds no `BpbFlopChart` curve; the bar is
  a markdown table, not a chart, so regret never mislabels the bpb axis). Rule of two **not**
  triggered for the rollout (one use); `Pipeline` 6→8. Wired into `nav.ts` (CONCEPT 09), the
  `ConceptMap` (new `In-context control` node off `preq` + `lpf`, left-bottom open space, no
  overlaps; caption updated), the landing card grid (auto via nav), and reciprocal See-also links on
  prequential / loss-per-flop / source-iv. Build green (19 pages); verified from `file://`: widget
  mounts (546 raster rects, wrap-broken agent path, scoreboard 0.220/0.622), Step/scrub update step
  + cumulative reward + highlight band, **zero console errors**, KaTeX renders (8 spans), no `\u`
  leaks, map node + all cross-links resolve. **Viz note:** the scoreboard shows this single
  rollout's regret (0.220), louder than the held-out-mean 0.141 in the bar table — flagged in both
  the baked figcaption and the page so the two never read as contradictory. The rollout JSON's
  `reward[t]` is the concentration at the agent's *new* cell after transition t (`field[t+1][pos[t+1]]`),
  so cumulative reward at frame t is `Σ reward[0..t-1]`. Researcher note found internally
  consistent — no science flagged.

- **2026-06-22 (session 11 — B.4 delta_mix page):** authored the experiment page `B.4-delta-mix` from
  the researcher note, mirroring B.3/B.2 layout/frontmatter/imports. Tells the first **non-Pareto-hollow
  Space-B** story: the bet (a count table is a degenerate one-hot-key Hebbian store — zero generalization,
  only `K` global mixer weights), the mechanism (one online **delta-rule (LMS)** fast-weight stream on a
  sparse **signed feature-hashed** key, `O(sV)` feasibility crux, ~8.2k FLOPs/byte on the ~15.7k bar), the
  (iv) story (linear-in-a-fixed-feature-map + convex loss ⇒ exact gradient is a rank-1 outer product,
  zero backward), the **matched-FLOP kill-test RESULT** (a/b/c on a `BpbFlopChart`; (b) delta 2.4181 beats
  counts_only 2.4353 and counts_more_warm 2.4327; binding (b vs c) = −0.0146 bpb), the two diagnostics
  (+0.8595 mixer weight `insight` Callout; the **generalization two-bar** — delta-only 3.73 vs the
  abstaining count 8.0 on 20,051 unseen contexts), the recurring-reflex `insight` Callout ("beat the cheap
  baseline at matched FLOPs", B.4 the first Space-B to clear it), and the honest framing (`caveat`: CI win
  real but modest, full-carve **pending**).
  - **Full-carve headline = PENDING marker.** `runs/full/leaderboard.md` carried no `delta_o6_warmfull`
    row at build time, so the $1.48	imes10^{12}$ / 2.0157-bar row is a *pending* table row + caveat — NOT
    plotted (no invented coordinate), per the note. Used the pending marker, not a real number.
  - **No new role/token (KISS).** `delta` is colored `fast_weight` (orange) because it IS a fast-weight
    associative memory — the delta-rule flavor, the A.1 family done right (A.1 was the transformer Hebbian
    bolt-on that collapsed; both read orange, the lineage). The two count entrants reuse `reference` blue
    (`counts_only`, the cheap ladder) and `warm` vermilion (`counts_more_warm`). Added the `deltaMix`
    dataset to `curves.ts` (3 measured points only).
  - **Wiring (exactly as B.3):** `nav.ts` (EXP B.4 between B.3 and first-finding), experiments index (auto),
    `ConceptMap` (9th leaf `B.4 delta mix`, role `fast_weight`; re-spaced the bus 8→9 at `LEAF_W=100` and
    shortened four long leaf labels to fit; added the delta-lineage edge **`warm` → `fwm`** so the warmed
    ladder now feeds fast-weight memory). Extended the **fast-weight-memory** concept with a "two lineages"
    section (Hebbian bolt-on vs error-correcting delta-rule / DeltaNet / signed feature-hashing) + related/
    See-also. Reciprocal links A.1↔B.4 and B.3↔B.4.
  - **Shared-component change (self-adapt):** `BpbFlopChart` y-tick **fallback** in `compendium.js` — the
    coarse 0.5/1.0-only step yields **zero** ticks on a tight range (B.4 is 2.40–2.45), so when the
    existing logic produces 0 ticks it now falls back to a 1-2-5 nice step with step-derived decimals.
    Verified isolated: simulated every shipped chart's range — each keeps an identical tick set (none hit
    0 ticks); only tight ranges refine. B.4 renders clean `2.40 / 2.42 / 2.44` labels.
  - **New static viz (rule of two NOT yet triggered):** the generalization **two-bar** (`.genbars` /
    `.genbar-row` / `.genbar-label` / `.genbar-track` / `.genbar-fill{.delta,.abstain}` in `global.css`) —
    delta `--c-fast` orange vs the abstaining count `--c-neutral`, widths = bpb/8.0. First occurrence, so
    inlined in the page (not factored into a `.astro` component yet); CSS lives in `global.css` for reuse —
    factor on the second occurrence. `BpbFlopChart` now 13 uses.
  - Build green (**19 pages**); verified from `file://` (zero console errors): the B.4 chart mounts (3 marks,
    y-ticks 2.40/2.42/2.44, both annotations, 3-series legend), the two-bar renders (46.6% / 100% with the
    right role classes), 25 KaTeX spans, no `\u` leaks, and the landing map shows the `B.4 delta mix` leaf,
    the new `warm`→`fwm` edge, A.1 + B.4 both orange (the lineage), with all three B.4 links resolving.
    Researcher note found internally consistent — no science flagged.
- **2026-06-23 (session 13 — C.A interactive model demos):** built the two runnable browser demos that
  **consume** the engineer's parity-gated JS model layer (`public/js/models/*.js`, `public/data/demos/*`,
  README HUD table) — making the loss-per-FLOP contrast *felt*, not just plotted. **(1) `AutocompleteRace`**
  (`autocompleterace`) on the **context-mixing** concept ("Try it live: the loss-per-FLOP race"):
  `context_mixing` / `delta_mix` / `transformer` run predict-then-learn on the same editable 2,048-byte
  enwik8 seed (latin-1 text → bytes via `charCodeAt&0xff`, verified bit-identical to the fixture `stream`);
  per model — top-6 next-char bars, live cumulative bpb (matches parity to ~1e-8), top-1 hit-rate, FLOPs/byte
  (~9.5k / ~18k / ~11.5M → ≈1,217× gap). **(2) `CursorChase`** (`cursorchase`) on the **in-context-control**
  concept ("Try it live: chase the cursor"): a `<canvas>` field whose peak **drifts toward your cursor at
  ≤1 cell/tick** (ChemoEnv's own drift — keeps the peak inside the organism's local sensing range, so a fast
  cursor never strands the reactive climber: faithful AND robust), with the three controllers chasing via the
  exact parity control tape (even = sense, odd = action, greedy `argmax(logits[L..L+3))`, move, reward = conc
  at the new cell); HUD shows params + FLOPs/step (66 vs ~10k) + live mean/cumulative reward (leader-
  highlighted) and per-organism toggles.
  - **Shared (rule of two):** new `wireTransport` helper (Step/Play/Reset on a timer) + `.demo-transport`
    style serve both new demos; both reuse existing role tokens (no new color) and `ControlRollout`'s
    build-time inline-JSON pattern. The older `mountStream`/`mountControlRollout` transports were left as-is
    (different scrub/loop semantics; not retrofitted, to avoid churning shipped parity-adjacent widgets).
  - **Plumbing:** added a `Layout` `scripts?: string[]` prop that injects the model-layer classic scripts
    into `<head>` deferred-in-order **before** `compendium.js` (so `SmolModels`/`SmolDemos` exist at mount);
    `relativize.mjs` rewrites the new `/js/models/*` srcs for file:// unchanged. Added
    `demoByteModels`/`demoControlModels` HUD-fact tables to `curves.ts` (params/FLOPs/role/refBpb, from the
    README HUD table — single source of truth); the markers read `config`/`weights`/`seed` from the
    fixtures/weights JSON at build time and inline them (escaping `<` so the seed's `<!--` can't break the
    inline `<script>`; transformer weights ~428 KB inline → context-mixing page ~592 KB, expected).
  - **Wiring (no orphans):** both demos live on existing concept pages — **no new page / nav entry /
    ConceptMap node**. Reciprocal See-also added: context-mixing ↔ in-context-control ↔ loss-per-flop.
    `BpbFlopChart` still 13 uses; `Pipeline` 8.
  - **Honesty:** a `caveat` Callout on context-mixing flags that the live transformer is the *trained* demo
    export (distinct from the *untrained* ~8.0-bpb transformer in the curve above); a `caveat` on
    in-context-control flags that the chase's live rewards are an interactive field, NOT the held-out
    distillation regret in the bar. No science flagged.
  - Build green (**20 pages**); headless `file://` smoke (both pages): **zero console errors**,
    `SmolModels`/`SmolDemos` load via the `scripts` prop, both widgets mount. Byte race: bpb advances live
    (online learners warm from 8 → ~4.17, trained transformer ~2.70; hit-rates 32 / 33 / 49 %; 6 top-k bars
    each; editable + Step/Play/Reset). Cursor chase: the organism chased the cursor cell 8 → 2 → 14 with
    `chemotaxis_min` leading (mean ~0.89 at **66 FLOPs/step**) over `reservoir` (0.80) and `reservoir_plastic`
    (honestly weak — heavier ≠ better), toggle dims a marker. Re-ran the engineer's parity gate read-only:
    **ALL 6 PORTS PARITY-GREEN** (model layer untouched). Did NOT commit, did NOT modify the model layer /
    Python, did NOT run project-wide gates; `git diff` confined to docs/learning (5 files modified + 2 new
    components).
  - **Cross-vendor review round (codex, same day):** fixes after the frontend review.
    - **Cursor-chase tick ordering (fidelity).** Reordered `step()` to ChemoEnv's real phase: for each
      organism sense (CURRENT field) → act → move; THEN drift the shared peak once; THEN reward =
      `concentration(new agentX, new peak)` and `steps++`. The drift now uses ChemoEnv's **exact integer
      phase-accumulator** (`smolml/envs/chemotaxis.py`): the integer peak jumps ±1 toward the cursor only
      when a phase accumulator at the env's `RATE` (0.3 — the fixtures' drift_rate is 0.2–0.3) crosses 1,
      so the peak stays on a cell and inside a local climber's sensing range. (The earlier 1-cell/tick
      continuous drift, masked by the unfaithful drift-before-sense order, stranded the run-and-tumble
      climber once the faithful order made its sensed token one drift-step stale — the drift rate, not the
      ordering, was the bug.) Verified in node + browser: a continuously-swept cursor is tracked
      (chemotaxis_min mean ~0.86 leads reservoir ~0.59 / plastic ~0.27) and a target teleport re-acquires
      (agent 8→2→14 as the peak drifts over).
    - **Byte-race EOF (fidelity).** `step()` now skips the model.step **fold** of the final byte (its
      prediction is never scored), so executed steps == scored predictions (prequential folds n−1 times).
      bpb is unchanged (still parity-exact); stepping past a short edited seed caps cleanly at `byte n / n`.
    - **Accessibility.** CursorChase gained a keyboard-operable **peak-target range slider** (arrow keys
      move the peak; synced with the pointer) and an `aria-live="polite"` status line announcing the
      current leader + per-model mean reward (throttled to leader-change / every 30 steps to avoid SR
      spam). `wireTransport` now gates the Play auto-timer on `prefers-reduced-motion: reduce` — Play is
      hidden, Step-only fallback (verified: Play hidden, Step still advances).
    - **Caption honesty.** Softened the two widget captions so they cannot read false after interaction:
      "**On this seed stream**, the transformer reaches the lowest bpb…" and "chemotaxis_min **typically**
      tracks tightest…".
    - Rebuilt green (**20 pages**); parity re-run **ALL 6 PORTS PARITY-GREEN** (loops only, model layer
      untouched); headless re-smoke of both demos = **zero console errors**. Still uncommitted.
- **2026-06-23 (session 12 — C.A in-context-control candidates):** authored two experiment pages from the
  researcher findings note and extended the `in-context-control` concept. **`C.A.1-reservoir-control`** —
  the reservoir family as honest negatives: C.A.1 (frozen echo-state core + distilled linear readout, 0
  backward; regret 0.494→0.371→0.278 over 150/600/1500 steps, **caps above the bar's 0.229**) and C.A.1b
  (same core + an online reward-modulated **plastic** readout at ~0 distillation; clears the random floor
  with genuine Source-(iv) dynamics — within-episode 0.460&gt;0.404, ~243× cheaper than the bar — but regret
  0.501, ~2.2× the bar). **`C.A.2-chemotaxis-min-control`** — the lone winner: 5 hand-coded run-and-tumble
  scalars, **0.180 regret @ 2.70×10⁵ FLOPs**, lower than the bar at ~6 OOM fewer FLOPs — framed
  *prominently* as a **FLOP-floor** win on a **stationary** rung (the documented C.A.0 limitation), not a
  general result; includes the counter-intuitive "distillation raises regret" mini-table
  (0.180→0.191→0.251) and the within-episode climb-then-track reward bars (0.844&gt;0.662). Extended the
  concept with a "**the candidates**" section: the cross-candidate **regret-vs-FLOP landscape** chart + an
  "only the floor beats the bar" `insight`, and updated the closing stationarity caveat to link C.A.2.
  - **New shared component `RegretFlopChart` (the control analog of `BpbFlopChart`).** Rather than
    duplicate the ~140-line chart routine, I **generalized `mountChart`** with an optional `metric` config
    (`yKey`/`yLabel`/`tipLabel`/`valDecimals`/`yTickStep`/`aria`); absent ⇒ exact bpb behaviour, so all 13
    bpb charts are byte-unchanged (verified: y-tick sets identical). `RegretFlopChart` is a thin marker
    that sets `metric` to plot `regret` (0.1 tick step). `CurvePoint` gained optional `bpb?`/`regret?`.
    3 uses (concept + C.A.1 + C.A.2); `mountChart` now serves **16** instances.
  - **New shared component `HBars`** (rule of two): factored B.4's inline `.genbars` two-bar into a
    parameterized `bars: {label,pct,text,accent}[]` component and **refactored B.4's first use** onto it
    (DOM/classes/widths byte-identical — coordinated with `DocsBuilderB4`, who held B.4 byte-stable;
    verified `>3.73 bpb</span>` unchanged). Reused for the C.A within-episode reward bars.
  - **3 new semantic data roles + tokens** (same precedent as `pc_refine`/`warm`): `reservoir` indigo
    (`--c-reservoir` #8f86d6), `reservoir_plastic` lavender (`--c-plastic` #b3a4e8, same-hue sibling =
    "reservoir family"), `chemotaxis` teal (`--c-chemo` #3aa890). Added across `curves.ts` `SeriesRole`,
    `compendium.js` `ROLE_COLOR`, `global.css` (`.swatch.*` + `.genbar-fill.{neutral,reservoir_plastic,
    chemotaxis}`), and `ConceptMap` (`role-reservoir`/`role-chemotaxis` borders).
  - **Wiring:** `nav.ts` (EXP C.A.1, C.A.2 between B.4 and first-finding), experiments index (auto via
    nav), `ConceptMap` (re-spaced the bus **9→11 leaves** at `LEAF_W=78`, shortened labels to fit; new
    `reservoir`/`chemotaxis` leaf borders + a new dashed **`ctrl → exp`** edge so the control rung's
    experiments are logged like every other concept's), reciprocal See-also/related on `in-context-control`
    ↔ both pages and C.A.1 ↔ C.A.2.
  - **Honesty:** used ONLY the findings' numbers; the chemotaxis 100/400-step regrets (no reported FLOPs)
    live in a mini-table, **not** plotted (no invented coordinate). Flagged a minor arithmetic slip in the
    findings ("16 OOM" should be ~7) to Main; did not reproduce it.
  - Build green (**22 pages**); verified from `file://` (zero console errors): all three regret charts
    mount (8/7/4 marks, correct teal/indigo/lavender/green fills, regret y-axis + 0.1 ticks, legends,
    separated annotations), reward bars render clean (no stray whitespace), KaTeX renders (476/447/437
    spans), no `\u` leaks, the map shows 11 non-overlapping leaves (min gap 4px) + the `ctrl→exp` edge, and
    B.4's two-bar is byte-identical post-refactor. Researcher findings found internally consistent — only
    the "16 OOM" wording flagged.
  - **Frontend review round (codex, 2026-06-23):** fixed 1 must-fix — the C.A.2 chart caption no longer
    overclaims "lower regret than the entire bar curve" / "nothing the transformer can afford gets below it"
    (now: best **regret-per-FLOP**; lower than the bar's *cheapest* point (0.229) at ~6 OOM fewer FLOPs; the
    bar reaches lower *absolute* regret 0.171/0.141 only at ~7 OOM more FLOPs). Minor: relabeled the
    in-context-control bar-table x-axis "training FLOPs" → "total FLOPs" (same total-FLOP numbers the chart
    plots; matches "all FLOPs counted"); clamped `HBars` `pct` to [0,100] (defensive). **Deferred
    low-severity (current data valid, build green):** (a) `CurvePoint`'s `bpb?`/`regret?` are both optional,
    so the shared chart reads `p[yKey]` with no compile-time guarantee the active key exists — a mismatched
    future series would render NaN coords; all current series carry the right key, but the markers should
    validate if the contract is reused. (b) Pre-existing across **all** charts: marks are `role="button"` +
    `tabindex` but have no Enter/Space activation handler (toggle is click / legend only) — an a11y gap not
    introduced here.
- **2026-06-26 (session 14 — C.A.4 forage local-learners page):** authored the experiment page
  `C.A.4-forage-local-learners` from the researcher note, mirroring C.A.2's layout/frontmatter/imports
  (`Layout` + `Callout` + `RegretFlopChart` + `HBars`): intuition (the forage rung as a contextual bandit
  over K=3 cue types — find which type pays from your eat-outcomes, then camp it) → math (a per-type value
  vector reset each episode + a local delta rule `v[t] += lr·(r − v[t])`, optimistic-init exploration, a
  distilled-scalar softmax policy `logit(EAT) = g·v[t] + b_eat`) → worked example (one episode, K=3,
  optimistic v=[+0.3,…], lr=0.8, g=8 — poison flips a type negative, g climbs, it camps) → result.
  - **Lead Callouts:** an `insight` (`forage_min` 0.047 regret @ $2.66\times10^{5}$ FLOPs / 8 params beats
    the swept transformer bar's cheapest 0.113 @ $6.39\times10^{11}$ / 148,672 params — ~6 OOM fewer FLOPs,
    and beats `win_stay_lose_shift` too) and a `caveat` (reflex-proof rung + a genuine within-episode
    learner ⇒ the (iv) thesis, not a strawman; but the rung's optimum is cheaply learnable by a local rule;
    **structure** — exact per-cue-type credit assignment — **not capacity** is the per-FLOP lever).
  - **Viz:** the regret-vs-FLOP `RegretFlopChart` (`forageControl`: `forage_min` lower-left, transformer
    bar upper-right, `forage_reservoir` far up-right; every candidate curve *rises* with distillation — more
    FLOPs do not buy lower regret here), the distillation mini-table (0.047/0.056/0.161), a structure-not-
    capacity `forage_reservoir` section, a robustness `note` (seeds 1–7: 0.0435 ± 0.009, max 0.058), verdict
    + learnings + a `reproduce` note. **HBars = the reference-policy reward ladder** (oracle +0.96,
    `win_stay_lose_shift` +0.85, `random` −0.11, `always_eat` −0.33 in `neutral` gray + `forage_min` +0.91
    in magenta; scaled on the eat reward range [−1,+1]) — shows reflex-proof (both fixed reflexes below
    zero) and `forage_min` edging `wsls` toward the oracle. `forage_min`'s reward is **derived** (oracle
    0.96 − regret 0.047 ≈ +0.91), flagged transparently in the caption (the note states regret, not the
    absolute reward).
  - **Data:** added `forageControl` to `curves.ts` (`forageBar` transformer + `forageMinCurve` +
    `forageReservoirCurve`, all *reported-FLOP* 3-point curves — unlike C.A.2's chemo point, every forage
    point has a FLOP coordinate, so each is a curve; provenance-commented). `forage_reservoir` **reuses**
    the `reservoir_plastic` role (lavender) — it IS that C.A.1b mechanism ported to forage, so no new token.
  - **One new role + token (4th control hue):** `forage_min` = **magenta** `--c-forage` #cf72b3, in the
    otherwise-unused ~300–340° gap (rose↔violet), added across `curves.ts` `SeriesRole`, `compendium.js`
    `ROLE_COLOR`, `global.css` (`--c-forage` + `.swatch.forage_min` + `.genbar-fill.forage_min`), and
    `ConceptMap` (`Role` + `.role-forage`). Same per-candidate-hue precedent as reservoir/plastic/chemo;
    chosen over reusing chemo teal so the map's two control-winner leaves don't read as the same model.
  - **Wiring (no orphans):** `nav.ts` (EXP C.A.4 between C.A.2 and first-finding → auto experiments-index
    card + prev/next), `ConceptMap` (12th bus leaf `C.A.4 forage` inserted between `ec2` and `eff`; `eff`
    moved 880→962 and the viewBox widened 940→1024 so the existing 11 leaves + labels stay byte-unchanged;
    magenta `role-forage` border; linked to `in-context-control` via the existing `ctrl → exp` bus, sibling
    of C.A.2 — same structure as C.A.1/C.A.2, no direct leaf→concept edge), and reciprocal cross-links on
    `in-context-control` (related rail + a bridging "sibling rung" paragraph in the honest-framing section +
    See-also). No new component factored — reused `RegretFlopChart` (now 4 uses), `HBars` (now 4 uses),
    `Callout`, `ConceptMap`; `mountChart` now serves **17** chart instances.
  - Build green (**23 pages**); verified from `file://` (zero console errors): the C.A.4 chart mounts (9
    marks across 3 series — green/magenta/lavender confirmed; legend toggle 9→6→9 survives re-render), the
    reward ladder renders (5 rows, `forage_min` +0.91 magenta @ 95.7%), 32 KaTeX spans, no `\u` leaks, the
    concept-map leaf sits at clean 4px gaps with a magenta border (no overlap/clipping; viewBox 1024), and
    every cross-link (siblings, concepts, first-finding, in-context-control) + the experiments-index card
    resolve. `git diff` confined to `docs/learning/`; did NOT commit/PR, did NOT run repo gates.
  - **Flagged for researcher confirmation (not "fixed"):** (1) `forage_min`'s absolute mean reward isn't in
    the note — I **derived** it (oracle 0.96 − regret 0.047 ≈ +0.91) for the reward ladder, transparently
    captioned; confirm the oracle baseline + forage_min's reward. (2) The note reports only within-episode
    *deltas* (2nd−1st: +0.16 forage_min, +0.31/+0.18 reservoir), not absolute 1st/2nd-half rewards, so —
    unlike C.A.1/C.A.2 — there is no within-episode absolute-reward HBars; I used the reference-policy
    reward ladder instead and put the +0.16 within-episode signal in prose. (3) The shared-context
    `forage_reservoir` curve has 3 points including (6.23e10, 0.514 @50 steps) that the note's *table* omits
    (the table lists only @0 and @200); I used the shared-context 3-point set per instruction. (4) No model
    file paths / driver command in the note (only `runs/forage/leaderboard.png` + `render_rollout`), so the
    `reproduce` note cites the run dir + model names, not invented `.py` paths. No science relitigated.
