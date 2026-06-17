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

## STANDING DIRECTIVE — grow the learning compendium

This project is also a **learning experience**. `docs/learning/` is an ever-growing hypertext
document and is a **first-class deliverable, not optional**.

Whenever you encounter, introduce, or rely on a non-trivial concept (a metric, an algorithm, a
piece of theory, a trick), you MUST:
1. Add or update a concept page under `docs/learning/concepts/<concept>.md` with: a plain
   explanation, at least one **visualization** (a Mermaid diagram inline, and/or a link to a
   generated plot the harness wrote), and a concrete **worked example**.
2. **Cross-link** it: link to/from related concept pages and from `docs/learning/index.md` so
   the web stays connected (this is hypertext — no orphan pages).
3. When you run an experiment, log it in `docs/learning/experiments/` (hypothesis, setup, the
   bpb-vs-FLOP result + plot, and what we learned — including failures, which are the point).

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
