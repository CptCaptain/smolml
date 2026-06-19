# 0005 — The learning compendium is an MDX (Astro) site

- Status: accepted
- Date: 2026-06-17

## Context

`docs/learning/` is a first-class, ever-growing compendium that wants **interactive**
visualizations (animated prequential streams, draggable scaling-law curves, fast-weight
read/write demos) that Markdown cannot express. Options weighed: keep Markdown (no
interactivity), hand-written static HTML (zero toolchain, but poor component reuse and
boilerplate-heavy authoring), or MDX (Markdown prose + embedded components).

## Decision

Author the compendium in **MDX, built with Astro**. Astro has first-class MDX, ships a static
site (dependency-free to *view* once built), and supports interactive islands. The JS/Node
toolchain is **confined to `docs/learning/`** and never touches the research loop.

## Consequences

- The repo becomes **polyglot**: a small Node/Astro project lives under `docs/learning/`.
- "**Docs build stays green**" becomes a PR gate for any docs change.
- Concepts get **reusable interactive components** (define `<PrequentialDemo/>` once, reuse with
  props) — the right primitive for a growing concept web, and the reason MDX beats raw HTML here.
- **Constraint: the built site must open from `file://` with no server.** This rules out
  ES-module island hydration (browsers block module scripts from a `file://` origin), so
  interactivity is delivered as a single **classic** script (no imports/fetch) plus a post-build
  step that **relativizes** every asset/link and **inlines CSS**. Astro/MDX stays the authoring
  layer; `npm run build` emits a directly-openable `dist/`.

## Alternatives considered

- **Static HTML:** zero toolchain, but every page re-implements its viz; rejected for weaker
  consistency/DX on an ever-growing site. (Fallback if the toolchain ever becomes a real drag.)
- **Plain Markdown:** no interactivity; rejected — the whole point is richer visualization.
