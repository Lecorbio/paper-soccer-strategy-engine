# Compact value BFM challenger

`compact_value_bfm` is a standalone, value-only complete-turn best-first
minimax bot for the 8x10 CodinGame rules. It is isolated from the maintained
Rank-4 bot and does not replace it. The checked-in `model.hpp` is an explicit
zero-valued build fixture, identified as `bootstrap-zero-not-qualified`; it is
not a strength artifact and cannot authorize an upload.

## Runtime contract

The engine uses the compact representation retained from the local arena
prototype: 316 used-edge bits, a byte-sized ball and mover, terminal winner,
and immutable 105-vertex/316-edge topology tables. The topology is numbered in
the independent research encoder's canonical order so the sparse inputs match
`tools/jacek_replay_features.py` exactly.

The evaluator has exactly 6,301 bias-free inputs. Inputs 0--315 are used edges.
Each canonical vertex selects one of 57 categories: category 56 means true-turn
distance at least seven or unreachable; otherwise the category is
`8 * distance + clamp(free_degree - 1, 0, 7)`. Player Two rotates 180 degrees.
Active indices are sorted.

Eligible shapes are `6301->8->8->1`, `6301->8->16->1`, and
`6301->12->8->1`. W1, W2 and W3 are input-major signed three-bit two's
complement codes packed least-significant-bit first. The loader checks exact
dimensions, finite positive float32 scales, byte count, SHA-256, zero tail
padding, and rejects code `100` (-4). W1 accumulates as integers and scales
once; W2/W3 have a fixed scalar float32 order. Full and sibling-delta inference
share that exact path.

## Search contract

One tree edge is one legal rebound-complete turn. Generation retains at most
250 unique full boundaries, uses nine LIFO extractions followed by one FIFO
extraction, mover-canonical seeded neighbour shuffling, 4,000 root and 512
non-root partial-path caps, exact attacking/own-goal paths, and a separate
64-path tactical witness pass. Global priority is immediate win, forced
cutoff, safe handoff, opponent immediate goal, then own goal. Boundary aliases
retain the mover-canonical lexicographically smallest transcript.

Every new nonsolved child receives a value. Terminal and tactical proofs use
complete-turn mate distance. Ancestors are overwritten by negamax/minimax;
there is no rollout or average return. Selection is
`value + C*sqrt(log(parentVisits)/childVisits)`, with `value+FPU` when
unvisited. Final root selection is `value + lambda*log(max(1, visits))`.
Proof status and traversal exhaustion are separate: a truncated expanded node
becomes non-selectable once every child it actually retained is closed, while
remaining unsolved unless it has proof evidence. This guarantees that the
allocation-free selector and progressive-widening prefix cannot stall on a
permanent dead end.
Defaults are `C=.95`, `FPU=.5`, `lambda=1`, 80,000 nodes, 2,000,000 expansion
safety cap, one thread, and 800/155 ms clocks. A deterministic complete turn is
prepared before the clock starts.

The CTest `papersoccer_codingame_compact_value_bfm_search_variant_parity`
compiles the baseline, no-feature-sort-only, single-pass-selection-only, and
combined implementations separately, then feeds every executable the same
frozen 24-state legal corpus. It requires byte-identical fixed-work evidence
from a zero-valued tie-breaking profile and a deterministic nonzero
`6301->12->8->1` three-bit profile. The evidence includes input and successor
state identities, feature identity, every legal root successor, the selected
transcript, bit-exact values and final scores, the ordered root transcript with
visits, selection visits and generation order, proof flags, and all exposed
search counters.

`COMPACT_VALUE_BFM_STATE_EVALUATION_CACHE_V1` enables the first independent
search-throughput experiment. Its per-search, direct-mapped table reuses a
value only after an exact used-edge, ball, mover, winner, and ply match; the
complete-turn mover is the evaluation perspective. Collisions replace entries
and cannot return a mismatched value. Search stats expose cache probes, hits,
and misses. The dedicated
`papersoccer_codingame_compact_value_bfm_state_evaluation_cache_parity_<base>`
CTests require cache activity and remove only those three counters before
demanding a byte-identical semantic trace from the corresponding uncached
implementation. This mode is off by default and is not yet a selected campaign
variant.

`COMPACT_VALUE_BFM_PROGRESSIVE_WIDENING_V1` is the second independent
search-throughput experiment. For each expanded node it makes the first
`min(open, 8 + 2 * floor(sqrt(max(1, selectionVisits))))` open children in
the existing deterministic generation order eligible for ordinary UCT
selection. It gradually admits more children without changing generation,
evaluations, score formulas, or tie comparisons. Search stats expose widening
probes, restricted selections, eligible children, and deferred children. Its
invariance CTest requires deterministic repeated output, exact root action,
tactical-class, initial-value-bit, and generation-order agreement with the
uncached combined baseline, legal selected turns in both modes, exercised
restriction counters, and an actual allocation change. This mode is also off
by default; compiling it together with state-evaluation-cache-v1 is rejected.

`COMPACT_VALUE_BFM_SUBTREE_REUSE_V1` is the final independent throughput
experiment. A direct-mapped exact-state index recognizes a node at the same
complete-turn depth and mover perspective whose deterministic expansion used
the full configured action cap. A hit clones that one-ply subtree's ordered
states and exact initial values into fresh nodes with independent visits and
backups, avoiding repeated complete-turn generation and inference without
creating a shared-node DAG. Hash collisions, depth/state mismatches, and
near-cap expansions are rejected. Search stats expose reuse probes, hits,
misses, rejections, and reused-child counts. Its invariance CTest requires
two byte-identical reuse runs and exact baseline agreement for state identity,
all legal root successors, selected turns, score/tie results, inference bits,
visits, generation order, and logical search accounting, while also requiring
all reuse outcomes and strictly less measured generator work. The mode is off
by default and cannot be combined with either earlier intervention.

CMake expands every intervention across baseline, no-feature-sort-only,
single-pass-selection-only, and combined controls. A header-only negative
compile test rejects every pair of intervention macros. The
`papersoccer_codingame_compact_value_bfm_source_compaction_parity` test also
requires the readable modular engine and all four generated standard variants
to emit the same trace, then checks its normalized semantic SHA-256 against the
golden trace from pre-intervention commit `43d9ec9`.

## Generation and checks

```sh
python3 submissions/codingame/bots/compact_value_bfm/export_model.py \
  --runtime results/compact_value_bfm/compact-value-bfm-20260831-v1/selected/<sha256>.runtime.json
python3 submissions/codingame/bots/compact_value_bfm/export_submission.py
python3 submissions/codingame/bots/compact_value_bfm/export_submission.py --check
```

The model exporter accepts only a content-addressed runtime with the exact
architecture, activation, quantization, selection seed/epoch, payload, and body
hash contracts. Both exporters write through a same-directory temporary file,
fsync it, and atomically replace the destination. Generated source must be
ASCII and at most 95,000 characters.

## Exact Rank-4 gate harness

`rank4_gate.cpp` is a separate local executable and is never amalgamated into
the submission. It compiles the maintained Rank-4 source in an isolated
namespace and refuses to run unless the on-disk source SHA-256 is exactly
`5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`.
The Rank-4 `GameState` is authoritative; a compact state is replayed in
lockstep and checked after every opening and played complete turn.

Banks are LF-only TSV files with header `opening_id<TAB>transcript`. Each
transcript is slash-separated, legal, live, and contains at least 12 physical
edge plies. `--pair-offset` and `--pair-count` select disjoint row ranges; each
row is played once with the candidate as each color.

```sh
cmake --build build --target papersoccer_codingame_compact_value_bfm_rank4_gate

# Fixed-work development screen
build/papersoccer_codingame_compact_value_bfm_rank4_gate \
  --bank results/compact_value_bfm/compact-value-bfm-20260831-v1/openings/development.tsv \
  --expected-bank-sha256 BANK_SHA256 \
  --candidate-source submissions/codingame/bots/compact_value_bfm/submission.cpp \
  --expected-candidate-sha256 CANDIDATE_SHA256 \
  --rank4-source submissions/codingame/bots/rank_4/submission.cpp \
  --pair-offset 0 --pair-count 100 --mode fixed-work \
  --candidate-c .95 --candidate-fpu .5 --candidate-lambda 1 \
  --candidate-actions 250 --candidate-root-partial-paths 4000 \
  --candidate-nonroot-partial-paths 512 --candidate-nodes 80000 \
  --candidate-expansions 2000000 --rank4-nodes 3000000 \
  --max-turns 320 --output screen.json

# Actual-clock gate; clocks are fixed internally at candidate 800/155 ms and
# exact Rank-4 800/165 ms.
build/papersoccer_codingame_compact_value_bfm_rank4_gate \
  --bank development-clock.tsv --expected-bank-sha256 BANK_SHA256 \
  --expected-candidate-sha256 CANDIDATE_SHA256 \
  --pair-offset 0 --pair-count 200 --mode actual-clock \
  --max-turns 320 --minimum-candidate-wins 211 \
  --minimum-wins-per-color 104 --output clock-gate.json
```

The JSON binds candidate, model body/payload, bank, Rank-4/opponent source,
configuration, every paired game, per-decision timings, deadline/headroom/hard
timeout counters, and categorized failures. Both engine clocks stop after the
complete-turn text is constructed and before the arena applies it or performs
the lockstep audit, so harness verification cannot count against either bot.
The protected-final tooling may
create and consume a five-pair-sharded bank only after immutable development
selection, a clean candidate commit, and the complete local preflight. No
protected bank has been created by the checked-in bootstrap fixture.

## Resumable development runner

`development_runner.py` screens an entire selected runtime family without
changing `model.hpp` or `submission.cpp`. It validates the six content-addressed
opening-bank manifests from `compact_value_bfm_openings.py`, atomically derives
canonical two-column gate TSVs, renders each model source into the ignored run
directory, compiles a source-specific Rank-4 gate binary, and records every run
as a content-addressed receipt. `--resume` reuses only receipts and binaries
whose request, source, runtime, bank-manifest, derived-TSV, Rank-4, and binary
hashes all still match.

```sh
python3 submissions/codingame/bots/compact_value_bfm/development_runner.py \
  --artifact-root results/compact_value_bfm/compact-value-bfm-20260831-v1/training \
  --selection SELECTED_PRIMARY_SEARCH.selection.json \
  --selection SELECTED_PRIMARY_TEACHER.selection.json \
  --selection SELECTED_NEUTRAL_SEARCH.selection.json \
  --selection SELECTED_NEUTRAL_TEACHER.selection.json \
  --selection SELECTED_RANK4_CONTROL.selection.json \
  --bank model_screen=MODEL_SCREEN.opening-bank.json \
  --bank tuple_screen=TUPLE_SCREEN.opening-bank.json \
  --bank tuple_confirmation=TUPLE_CONFIRMATION.opening-bank.json \
  --bank profile_screen=PROFILE_SCREEN.opening-bank.json \
  --bank profile_confirmation=PROFILE_CONFIRMATION.opening-bank.json \
  --bank actual_clock=ACTUAL_CLOCK.opening-bank.json \
  --output-root results/compact_value_bfm/compact-value-bfm-20260831-v1/development \
  --development-output results/compact_value_bfm/compact-value-bfm-20260831-v1/development/development-input.json \
  --resume
```

The adaptive sequence is exact: all eligible arms plus the nondeployable
Rank-4 control on 100 pairs; the complete eight-tuple roster for the retained
three; best-two-plus-default 250-pair confirmation; three work profiles on 100
pairs; best-two-plus-default profile confirmation on 250 pairs; then the
selected tuple/profile on the disjoint 200-pair actual-clock gate. The final
development input is validated by `compact_value_bfm_campaign.py` before it is
published atomically.

## Public design sources and limitations

This is an independent adaptation of public ideas, not a copy or claim about
unpublished source, weights, corpora, or competition settings. Public sources
are Jacek Dermont's [Paper Soccer article](https://github.com/jdermont/playground-kvhfh5iv/blob/5f369d6e64870690d100f0c3fe2f8e75c447dac3/papersoccer.md),
[QtPaperSoccer commit 366d530](https://github.com/jdermont/QtPaperSoccer/tree/366d5304c09c2c820bd3ef4ea94624c034b8d955),
[Best-First Minimax Search with UCT](https://www.codingame.com/playgrounds/55004/best-first-minimax-search-with-uct),
and [Paper Soccer neural inputs](https://www.codingame.com/playgrounds/157341/inputs-for-neural-networks-for-the-board-games/paper-soccer).

The earlier large 6,301-input pilot improved a 20 ms development panel but
remained below Rank 4 and never passed the protected game gate. Likewise,
earlier native and arena BFM campaigns produced useful engineering evidence
but were rejected as promotion candidates. Local paired games cannot establish
live strength: runner speed, opponents, queue composition, and platform
behavior differ. This namespace makes no Rank-4, Rank-1, or upload claim until
the separately frozen strict gate, CI, one-shot attestation, and 90-game window
are complete.
