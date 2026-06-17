# Context & Glossary

The shared language for this project. Glossary only — no implementation details, no specs.

## Terms

### Scout project
A search for a *fundamentally* more compute-efficient learning mechanism, as opposed to
following the crowd's "scale transformers" pheromone trail. Coverage of the idea space
matters more than polish on any single idea.

### Loss-per-FLOP (the metric)
The project's single north-star: **validation bits-per-byte at a fixed training-FLOP
budget** on a fixed tiny corpus. The only definition of "faster" that counts. Deliberately
*not* sample efficiency, parameter efficiency, wall-clock time, or inference speed — those
are distinct, trading-off axes we are not optimizing.

### Bits-per-byte (bpb)
Cross-entropy on held-out text, normalized to bits per byte. Tokenizer-independent, which
is why it's the comparison currency rather than per-token loss.

### Fixed FLOP budget
A constant training-compute allowance every candidate gets, so comparisons are fair across
machines and frameworks. FLOPs, not seconds — hardware- and language-agnostic.

### Baseline
A clean, honestly-counted transformer trained under the same harness and FLOP budget. The
bar every candidate must clear. "Beating the baseline" means lower bpb at equal FLOPs.

### Candidate / mechanism
A specific learnable system entered into the search (a sequence-mixing architecture, a
learning rule, a conditional-compute scheme, …). Tracked in `docs/candidates.md`.

### Capability claims
Statements like "it answers questions." Explicitly *out of scope* until a candidate has
already won on loss-per-FLOP. Treated as marketing, not evidence, until then.

### Source (iv) — the only qualifying advantage
The single admissible reason a Space-B (non-backprop) candidate is worth scouting: its
learning *dynamics* extract **more loss-reduction per FLOP**. Distinguished from (i) cheaper
credit assignment (~3× ceiling), (ii) locality/parallelism (wall-clock only — out of scope),
and (iii) no activation storage (memory only). See ADR 0003.

### Amortized vs. transductive
*Amortized* learning trains a reusable model once, then answers many queries (transformers).
*Transductive* learning re-learns per input stream (classic online context-mixing
compressors). Which of these counts as a valid candidate is a project boundary still under
discussion.
