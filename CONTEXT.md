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
compressors). RESOLVED: **hybrids are first-class** (an amortized model that keeps adapting).
Pure transductive single-stream compression is out-of-scope as a *target* (not forbidden as
inspiration / as a per-FLOP ceiling).

### Continual / online learning (in scope, a goal)
A qualifying artifact may keep learning as it sees data, rather than freezing after training.
This is desirable in its own right (a reusable model that adapts), not just a means to lower
loss.

### Total-FLOP accounting (inference counted)
The FLOP budget counts **all** compute: pretraining + any inference/test-time-learning +
prediction. Test-time adaptation is never free — a hybrid that learns at inference pays for
those FLOPs in the same budget, so it cannot game the metric by hiding compute at eval.

### Prequential evaluation (the protocol)
The model predicts each unit of the stream *before* it is revealed (log-loss in bits), then
may adapt on it. Score = cumulative bpb over the evaluation stream at a fixed *total*-FLOP
budget, reported as a curve. The one definition of "win" now. See ADR 0004.

### Compression = prediction
Not a distinction: cumulative one-step-ahead log-loss equals compressed length (arithmetic
coding). "Lower bpb" and "better predictive model" are the same objective. The thing we
actually want beyond low bpb is *reusability + continual adaptation*, not the avoidance of
compression.
