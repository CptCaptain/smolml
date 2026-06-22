# Task B.3 — bounded (hashed) count tables → full-corpus mixing

- Status: DONE + merged to `main` (d0083dc), live. Full ADR-carve run complete: bounded order-6 on
  the real 5 MB eval reaches cold 2.2570 / ~7MB-warm 2.1111 / **full-95MB-warm 2.0157** bpb vs order-3
  2.6224, in fixed ≤5.0 GiB (no OOM); transformer 5.4770 @ 1.46e14. Full warmup did NOT saturate the
  2²⁰ table (kept helping). 108 tests green; cross-vendor reviewed.
- Enabler for the **full enwik8 ADR carve** (95 MB prior warmup / 5 MB eval) that B.2 could not run.

## Why (the measured blocker)

B.2 ran on a 4 MB slice. The full ADR carve OOMs: `warm_mix`/`gated_mix` store per-order counts in
**unbounded Python dicts**, and the order-6 config that *won* (2.66 bpb) explodes to **~58 GB** on a
full-95 MB warmup (measured: ~1.0 M contexts at 2 MB, ~0.5 new contexts/byte and still climbing).
order-3 saturates (~3 GB) but is the weaker config. To validate the order-6 win at full ADR scale we
need **fixed-memory count tables** — the standard PAQ/cmix technique.

## Mechanism — `hashed_mix`

`@register_model("hashed_mix") class HashedMix(WarmMix)`. High orders use a **fixed-size hashed count
table** instead of a growing dict; low orders stay exact (they're small).

- Per hashed order `k ≥ hash_min_order`: a table of `T = 2**table_bits` slots, each a 256-wide
  `uint16` count vector (halve-on-overflow). Context → `hash(context) % T` → slot. **Collisions are
  accepted** (multiple contexts share a slot — bounded memory at the cost of mixing noise).
- Orders `k < hash_min_order` keep the exact dict store (order-0..3 are small).
- Everything else — logistic mixing, online mixer-SGD, the warm prior→eval handoff (inherited from
  `WarmMix`), prequential `step` — is unchanged; only the **count store** swaps.

### FLOP honesty

A hashed lookup is charged exactly like the dict gather it replaces (`gather_flops`); the hash
computation is `pointwise_flops` (a few ops/byte); mixing/Laplace/update costs are unchanged. The
per-byte charge stays dynamic and is summed by the harness — **no compute hides** in the hashing.

### Memory

`T = 2**21` (2 M) × 256 × 2 B = 1 GB per hashed order; bounding orders 4–6 ⇒ ~3 GB total, **fixed**
regardless of corpus size (the whole point). `table_bits` is the memory↔collision knob.

### Honest expectation (report straight)

At full-corpus order-6 the table is overloaded (~30–50 M contexts into 2 M slots ⇒ heavy collisions),
so the order-6 advantage **may erode toward order-3** at feasible memory. Whether the order-6 win
survives bounding is the open question this task answers — a real result either way.

## Config

`table_bits` (hashed-table size, default e.g. 21), `hash_min_order` (orders ≥ this are hashed,
default e.g. 4); reuse `max_order`, `alpha`, `lr`, and the `WarmMix` warmup. Validate in
`__post_init__`.

## Acceptance

- Gates green: `uvx ruff format --check .`, `uvx ruff check .`, `uv run pytest`.
- `tests/test_hashed_mix.py`: `hash_min_order > max_order` ⇒ bit-identical to fixed-order `warm_mix`
  (no order hashed); a hashed order with a large `table_bits` (collision-free on a tiny fixture)
  reproduces the exact-dict predictions; **memory is bounded** — the hashed table's slot count is
  independent of how many distinct contexts are folded (assert table size constant after folding a
  long vs short stream); per-byte FLOP charge matches the exact-store charge for collision-free case;
  reproducible seed; registration.
- **Full ADR carve run** (committed runner, detached — multi-hour): `hashed_mix` order-6 warmed on
  the full 95 MB prior + the cold `context_mixing` reference + `gated_mix` (hashed) + a `transformer`
  anchor, all on the **full 5 MB ADR eval stream**. Plot bpb-vs-total-FLOP; report peak RAM, the
  collision load, and whether the order-6 win survives full-corpus bounding. Honest either way.

## Out of scope

GB-scale faithful cmix (nibble counts, checksums, run-length state machines) and a GPU transformer
path — the remaining ADR endpoints. This task delivers fixed-memory mixing sufficient to run the
full carve and answer whether bounded order-6 holds its per-FLOP win.
