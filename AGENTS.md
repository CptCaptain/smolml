# AGENTS.md — operating manual for smolml

Read this before doing anything. It applies to every agent (Claude Code, Codex, Pi, the
orchestrator) working in this repo.

## What this project is

A **scout project**: search for a *learning algorithm* with better **loss-per-FLOP** than the
transformer baseline, at tiny scale. We are doing the legitimate research question hiding
inside "replicate GPT-3 on a laptop" — *are there mechanisms that learn more per FLOP?* — not
training a giant model. Background and the math are in `README.md`.

## The locked decisions (do not silently relitigate)

- **Metric** — validation **bits-per-byte (bpb)** at a fixed **total-FLOP** budget on tiny
  `enwik8`. Lower bpb at equal FLOPs = win. (`docs/adr/0001`)
- **Eval protocol** — **prequential / online**: predict each byte *before* it is revealed,
  then may adapt; score cumulative bpb vs. total FLOPs (pretraining + inference + adaptation
  all counted). Data carve: final 5 MB of enwik8 is the fixed eval stream; first ~95 MB is a
  freely-usable prior corpus. (`docs/adr/0004`)
- **Source-(iv) filter** — a non-backprop candidate qualifies **only** if its learning
  *dynamics* extract more loss-reduction per FLOP. "Avoids backprop" and "is parallel" do
  **not** qualify (parallelism is out of scope under the metric). (`docs/adr/0003`)
- **Stack** — PyTorch, device auto-detect cuda>mps>cpu; **no hand C/CUDA during the search**.
  `uv` for env, `ruff` format+check, `pytest`. (`docs/adr/0002`)
- Shared language lives in `CONTEXT.md` (glossary — keep it implementation-free). The candidate
  pipeline and statuses live in `docs/candidates.md`. The roadmap is `docs/plan.md`.

## STANDING DIRECTIVE — grow the learning compendium (via the docs-builder)

This project is a **learning experience**. `docs/learning/` is an ever-growing **interactive MDX
site (Astro)** — the canonical learning compendium and a first-class deliverable. Format rationale
is ADR 0005; the role split is ADR 0006; the docs-builder's operating manual is
`docs/learning/CONTRIBUTING.md`.

**Division of labor (ADR 0006) — read this before touching docs:**

- **Researchers** = implementers, the orchestrator, and the human. They generate concepts and run
  experiments. They **do NOT hand-write** anything under `docs/learning/`. When a researcher
  introduces a concept or finishes an experiment, they produce a clear **written explanation**
  (intuition, the math, a worked example, and what is worth visualizing) and hand it to the
  docs-builder via the orchestrator — then **confirm** the resulting page is accurate and helpful
  (a short back-and-forth).
- **The docs-builder** = a dedicated, persistent sub-agent (`claude_code`, title `docs-builder`)
  that does nothing but build and grow the site. It owns `docs/learning/`, maintains a reusable
  **component library**, keeps a **process log** (`docs/learning/PROCESS.md`) so output stays
  consistent across sessions, and **self-adapts** (factors out a shared component whenever a viz
  pattern recurs). It does not touch harness/research code.

PR gates for any docs change: the docs site **build stays green**, content is **confirmed by a
researcher/human**, and the code is **cross-reviewed by a different vendor** (it is frontend code).

If a PR introduces a concept or an experiment without updating `docs/learning/`, it is
incomplete. Keep entries lightweight but real; favor intuition + a picture + an example over
walls of text.

## Workflow (how changes land)

- Each task gets its own branch + worktree and opens **its own PR**. Do **not** merge — a human
  merges after review.
- Every PR is reviewed by a **different vendor** than the implementer (cross-review). Reviewers
  report; they do not edit.
- Run the gates and paste results in the PR: `uvx ruff format --check`, `uvx ruff check`,
  `uv run pytest`, plus the relevant smoke run / leaderboard output.

## Coding conventions

- KISS — smallest change that satisfies the task. **Hard cutover, no backward-compat shims.**
- Python 3.12, modern typing (`list[str]`, `X | None`); type hints everywhere; 4-space indent;
  lines ≤100. Use `uv run` to execute things, never bare `python`.
- Keep deps minimal (torch, numpy; matplotlib only for plots). Reproducible seeds.
