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

## Alternatives considered

- **Static HTML:** zero toolchain, but every page re-implements its viz; rejected for weaker
  consistency/DX on an ever-growing site. (Fallback if the toolchain ever becomes a real drag.)
- **Plain Markdown:** no interactivity; rejected — the whole point is richer visualization.
