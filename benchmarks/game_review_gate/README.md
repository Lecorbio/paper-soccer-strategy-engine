# Game Review strength and calibration gate

This directory is the separate preregistered evidence pipeline for the three
fixed-node DeepTurnSearch candidates used by Game Review. It does not modify,
rerun, reinterpret, or replace `benchmarks/flagship_study`.

No strength result is recorded here until the frozen development, validation,
latency, and one-shot test stages have completed. In particular, the presence
of a Deep review profile is not an Expert claim. The playable label
**Expert — DeepTurnSearch** is permitted only when the final paired 95% lower
bound is strictly above 50% against both frozen references. A failed gate is
reported as **strength unresolved** and leaves Expert out of the selectors.

## Frozen contract

The immutable inputs are in `manifest.json` and `opening_identities.json`:

- candidates: DeepTurnSearch at 100,000, 200,000, and 400,000 nodes;
- references: Rank5DerivedBot's fixed 50,000-node demo profile and the selected
  20,000-node JacekInspiredBot profile;
- opening depths: 4, 8, 12, and 20 physical edges, including rebound edges;
- development: 25 color-swapped pairs per depth, reference, and candidate
  (1,200 games);
- validation: 50 pairs per depth, reference, and candidate (2,400 games);
- frozen test: 100 pairs per depth and reference for the locked candidate
  (1,600 games, exactly once);
- bootstrap: 10,000 opening-depth-stratified resamples of whole color-swapped
  pairs;
- Wasm limits: p95 at most 400 ms and maximum at most 750 ms.

The one-percentage-point strength band is formed from the validation leader
across all three candidates before applying the Wasm limits. If no profile in
that band meets both limits, selection stops rather than falling back to a
materially weaker candidate.

The twelve gate banks contain 700 openings. Generation used new domain-specific
seeds and rejected exact-state and horizontal-reflection equivalence against
all current and retired flagship banks and every earlier gate bank. Verify both
the committed bytes and a fresh regeneration with the existing native tool:

```sh
python3 benchmarks/game_review_gate/run_gate.py validate \
  --opening-tool build/native/papersoccer_opening_bank \
  --verify-regeneration
```

The regeneration command rebuilds each gate bank in manifest order, adding
the preceding regenerated banks to the exclusion set, and requires byte-for-
byte equality. It never writes into the flagship directory.

## Framework checks

From the repository root:

```sh
./scripts/build-and-test.sh
python3 -m unittest discover -s tests/game_review_gate -p 'test_*.py'
node --check benchmarks/game_review_gate/measure_wasm_latency.mjs
python3 benchmarks/game_review_gate/run_gate.py validate \
  --opening-tool build/native/papersoccer_opening_bank
```

The native arena used as evidence must be a sanitizer-free Release binary
configured from the exact current commit with an otherwise clean source tree.
The runner checks the embedded commit, build type, `NDEBUG`, sanitizer state,
and dirty flag. If a preserved untracked `matches.json` makes CMake provenance
dirty, build the arena in a clean auxiliary worktree at the same commit and
pass its absolute path; do not delete or move the replay file.

## Development and validation

First commit the manifest, banks, identities, gate code, probe bridge, and bot
implementation. Configure the evidence arena only after that commit exists:

```sh
cmake -S . -B build/game-review-gate \
  -DCMAKE_BUILD_TYPE=Release \
  -DPAPERSOCCER_ENABLE_SANITIZERS=OFF
cmake --build build/game-review-gate --parallel --target papersoccer_arena
build/game-review-gate/papersoccer_arena provenance

python3 benchmarks/game_review_gate/run_gate.py run \
  --phase development --arena build/game-review-gate/papersoccer_arena
python3 benchmarks/game_review_gate/run_gate.py aggregate --phase development

python3 benchmarks/game_review_gate/run_gate.py run \
  --phase validation --arena build/game-review-gate/papersoccer_arena
python3 benchmarks/game_review_gate/run_gate.py aggregate --phase validation
```

Each phase can be split safely across independent processes. For example, run
four commands with `--shard-count 4` and distinct `--shard-index` values 0–3,
then aggregate once all shards exist. A rerun resumes only a shard with the
same manifest, run, unit, arena, and profile identities. Raw arena JSON remains
under the ignored `results/game_review_gate/<manifest-sha256>/` tree. Curated
phase results retain raw-shard hashes and every whole-pair score, allowing all
summaries and bootstrap intervals to be recomputed without trusting totals.
Every raw shard, compact phase result, latency artifact, selection lock, and
test-once marker also carries one canonical SHA-256 identity over the tracked
rules, search, arena, Wasm bridge, and gate-runner sources. Live tracked edits
are rejected before every phase, and development, validation, latency, and
test must all reproduce that same competition-source identity.

## Wasm latency and native parity

Build the dedicated single-file artifact with Emscripten 6.0.2. The build must
use a fixed 64 MiB initial heap, disabled memory growth, and the five restricted
`ps_analysis_probe_*` exports in addition to the review ABI:

```sh
emcmake cmake -S . -B build/wasm-review -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm-review --parallel \
  --target update_papersoccer_analysis_wasm
```

The measurement harness replays committed validation TSV moves through the
authoritative Wasm `Match`, rejects mid-rebound openings, constructs a fresh
analyzer for every search, and times only `ps_analysis_probe_run` with
`performance.now`. It uses the same 20 accepted possession boundaries at each
of 4/8/12/20 for all candidates, after eight untimed warmups per candidate:

```sh
node benchmarks/game_review_gate/measure_wasm_latency.mjs \
  --module web/papersoccer-analysis-wasm.js \
  --emscripten-version 6.0.2 \
  --output results/game_review_gate/latency-pass-1.json
```

For every timed sample, the harness finds the candidate-to-move game in the
corresponding native validation shard and compares the complete recommended
action, root score, and deterministic search counters. The artifact retains
the normalized Wasm transcript plus action/transcript SHA-256 values. The
Python validator recomputes those hashes and reconstructs the native transcript
from the hash-locked raw shard. Thus `parity_failures: 0` is checked evidence,
not a trusted handwritten count. This parity claim applies to the 240 sampled
profile/position combinations only; it is not a claim about every reachable
position or every JavaScript runtime.

## Stabilize and freeze the C++ lock

The linked analysis module consumes only the selected node budget and the two
profile-specific calibration identities and coefficients. It does not embed a
selection-lock or latency-artifact SHA, avoiding a module-hash feedback loop.

Use the first measurement as a draft input to render the C++ header. This does
not create the official selection lock:

```sh
python3 benchmarks/game_review_gate/run_gate.py render-lock \
  --latency-input results/game_review_gate/latency-pass-1.json
cmake --build build/wasm-review --parallel \
  --target update_papersoccer_web update_papersoccer_analysis_wasm
```

Rebuild both Wasm artifacts, measure the analysis artifact to a new file, and
check whether selection is stable. The gameplay rebuild is required because a
positive held-out result may later expose the locked DeepTurnSearch bot:

```sh
node benchmarks/game_review_gate/measure_wasm_latency.mjs \
  --module web/papersoccer-analysis-wasm.js \
  --emscripten-version 6.0.2 \
  --output results/game_review_gate/latency-pass-2.json
python3 benchmarks/game_review_gate/run_gate.py check-lock \
  --latency-input results/game_review_gate/latency-pass-2.json
```

If the check differs, render from pass 2, rebuild, remeasure to pass 3, and
repeat. Once it is stable, promote only the final measurement and create the
official lock:

```sh
python3 benchmarks/game_review_gate/run_gate.py record-latency \
  --input results/game_review_gate/latency-pass-2.json
python3 benchmarks/game_review_gate/run_gate.py lock-selection
python3 benchmarks/game_review_gate/run_gate.py check-lock
cmake --build build/wasm-review --parallel \
  --target check_papersoccer_web check_papersoccer_analysis_wasm
```

The selection lock contains two mappings fitted exclusively on validation
decisions:

- Fast uses fresh Rank5Derived reference searches because its immutable
  complete-turn settings are identical to the Fast 50k analysis settings; its
  analysis profile ID remains distinct from the Rank5Derived bot ID.
- Deep uses only fresh decisions from the selected Deep candidate.

Scores are oriented to the player making the possession before fitting. A
nonpositive slope is rejected. The lock stores standardized-logistic
coefficients and the C++ header uses the algebraically equivalent raw-score
form `a - b*mean/scale` and `b/scale`; coefficients are never hand-edited.

Commit the development and validation results, final latency measurement,
selection/calibration lock, generated C++ header, and final single-file module
before authorizing the test. Reconfigure and rebuild the native arena from that
exact clean commit so its embedded provenance matches.

## One-shot frozen test and report

Start the test only after all prerequisites above are tracked and clean:

```sh
python3 benchmarks/game_review_gate/run_gate.py run \
  --phase test --arena build/game-review-gate/papersoccer_arena
python3 benchmarks/game_review_gate/run_gate.py aggregate --phase test
python3 benchmarks/game_review_gate/run_gate.py report
python3 benchmarks/game_review_gate/run_gate.py validate --require-complete
```

The first test command atomically creates a run marker bound to the manifest,
official selection lock, and arena binary. Interrupted shards may resume that
same unfinished run. A completed marker or existing curated test result blocks
a second evaluation. Aggregation itself may be repeated because it performs no
new games and must reproduce identical bytes.

`compact_results.json`, `REPORT.md`, and `web/game-review-gate.js` are created
only from completed evidence. The last file is a direct-file-safe generated
decision artifact: the browser adds the Expert option to all three opponent
selectors only when its frozen-test decision is positive.
They report opponent-specific whole-pair scores and intervals, operational
counts, the conditional Expert decision, selected profile identity, and the
two calibration mapping hashes. They do not report an invented overall
accuracy number or alter any authentic `rank_5` or flagship-study claim.
