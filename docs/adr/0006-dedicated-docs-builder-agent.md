# 0006 — A dedicated docs-builder sub-agent owns the compendium

- Status: accepted
- Date: 2026-06-17

## Context

The compendium is first-class, ever-growing, and must stay **consistent** to be useful. If every
researcher/implementer hand-writes their own learning pages, we get inconsistent style,
re-invented visualizations, and context-switching away from research. Frontend (MDX/Astro) work
is also a different skill from the research loop.

## Decision

Split the roles:

- **Researchers** generate concepts and experiments and **explain** them in prose; they confirm
  accuracy. They never edit `docs/learning/` directly.
- A **dedicated docs-builder sub-agent** (`claude_code`, persistent session titled `docs-builder`)
  owns `docs/learning/` exclusively. It is **self-adaptive**: it maintains a reusable **component
  library** and a **process log** (`docs/learning/PROCESS.md`) recording its conventions and
  decisions, so output stays consistent across invocations. Its operating manual is
  `docs/learning/CONTRIBUTING.md`.

Workflow per concept: researcher explains → docs-builder drafts an MDX page (reusing/extending
components) → researcher/human confirms it's accurate and helpful → iterate → docs-builder opens
its PR (cross-reviewed by a different vendor; build green).

## Consequences

- Consistent, compounding docs quality; researchers stay on research.
- A persistent session lets the builder accumulate context (its own library + process memory).
- Adds a coordination hop (explain → confirm); accepted as the cost of consistency.
- Docs PRs get two checks: **content** confirmation (researcher/human) and **code** cross-review
  (different vendor).

## Alternatives considered

- **Each implementer writes its own docs:** rejected — inconsistent and off-task.
- **Orchestrator hand-writes docs:** rejected — it is frontend code, and it neither scales nor
  self-adapts.
