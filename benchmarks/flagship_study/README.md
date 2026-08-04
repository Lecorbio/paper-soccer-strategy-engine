# Flagship bot study

This directory contains the preregistered, manifest-driven comparison of
**Tactical MctsBot**, **Hand-evaluated AlphaBetaBot**,
**Neural alpha-beta (JacekInspiredBot)**, and
**Rank5DerivedBot — fixed 50k demo profile**. The study code uses only the
Python standard library; bot execution uses the optimized native arena.

Rank5DerivedBot adapts search code from the rank 5/206 CodinGame submission to
different demo rules and a fixed-work profile. These measurements do not
evaluate the authentic ranked submission.

`analysis_contract.md` is the human-readable statistical contract. The frozen
`manifest.json`, opening banks, selection lock, curated data, SVG charts, and
`REPORT.md` appear here as the study advances. Raw per-unit arena reports stay
under the ignored `results/` tree, keyed by manifest hash.

## Framework and CI checks

From the repository root:

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
python3 -m unittest discover -s tests/flagship_study -p 'test_*.py'
python3 tests/flagship_study/run_smoke.py \
  --arena build/release/papersoccer_arena \
  --manifest tests/fixtures/flagship_study/smoke_manifest.json
```

The smoke study is deliberately tiny and writes only ignored smoke outputs. It
is not evidence about comparative strength or latency.

## Freeze the full study

Manifest generation is allowed only from a clean, committed framework tree.
For v4 it verifies and reuses the development and never-accessed test banks
byte-for-byte, creates four fresh validation banks disjoint from every v3
opening, validates every replay, and refuses to replace an existing artifact.

```sh
python3 benchmarks/flagship_study/prepare_manifest.py \
  --opening-tool build/release/papersoccer_opening_bank \
  --source-commit "$(git rev-parse HEAD)" \
  --fresh-validation-keep-frozen-test
python3 benchmarks/flagship_study/run_study.py validate
```

Commit `manifest.json` and `openings/` before running configuration selection.
The manifest hash is the namespace for every raw result.

The v4 flag is specific to the audited v3 stop: v3 completed validation but
stopped before test when a scale-dependent calibration guard falsely rejected
a finite fit. V4 uses new validation seeds, bot/bootstrap/analysis/calibration
seeds, and filenames. The test banks and their hashes remain unchanged. See
`V3_VALIDATION_FAILURE.md` and `superseded/README.md`.

## Development and validation

There are 36 deterministic units in each tuning phase (nine configurations by
four opening depths). A shard index owns units whose stable position modulo
`--shard-count` equals that index. Reissuing the same command safely resumes
completed unit files.

```sh
# First run the preregistered depth-4/depth-20 projection units on the frozen
# manifest machine and arena: two exact-budget units for every candidate.
for index in 0 3 4 7 8 11 12 15 16 19 20 23 24 27 28 31 32 35; do
  python3 benchmarks/flagship_study/run_study.py run \
    --phase development --arena build/release/papersoccer_arena \
    --shard-count 36 --shard-index "$index"
done
python3 benchmarks/flagship_study/run_study.py project-runtime --write
# Then complete the depth-8/depth-12 development units on that same machine.
for index in 1 2 5 6 9 10 13 14 17 18 21 22 25 26 29 30 33 34; do
  python3 benchmarks/flagship_study/run_study.py run \
    --phase development --arena build/release/papersoccer_arena \
    --shard-count 36 --shard-index "$index"
done
python3 benchmarks/flagship_study/run_study.py aggregate --phase development

# Run all validation indices serially on the frozen machine under the gate conditions.
for index in $(seq 0 35); do
  python3 benchmarks/flagship_study/run_study.py run \
    --phase validation --arena build/release/papersoccer_arena \
    --shard-count 36 --shard-index "$index"
done
python3 benchmarks/flagship_study/run_study.py aggregate --phase validation
python3 benchmarks/flagship_study/run_study.py lock-selection
```

Full validation refuses to run unless the gate machine is on AC power with Low
Power Mode disabled. If any tunable family has no configuration at or below the
50 ms validation p95 limit, selection stops before test.

Review and commit `runtime_projection.json`, the development/validation curated
data, and `selection_lock.json` before any test command. The lock freezes
selected IDs, validation metrics, opening and manifest hashes, validation-only
calibration mappings, and the validation Pareto classification.

## Frozen test and publication

The test schedule has 24 units (six matchups by four depths) and exactly 4,800
games. A committed manifest, every committed bank, and a committed selection
lock are hard prerequisites. An incomplete run resumes the same identity; a
completed marker rejects a second evaluation.

```sh
for index in $(seq 0 23); do
  python3 benchmarks/flagship_study/run_study.py run \
    --phase test --arena build/release/papersoccer_arena \
    --shard-count 24 --shard-index "$index"
done
python3 benchmarks/flagship_study/run_study.py aggregate --phase test
python3 benchmarks/flagship_study/run_study.py analyze-test
```

Do not use `--destructive-test-override` for this study. Any truncation is an
operational defect, never a draw or half-point, and blocks publication. The
analysis command derives all three deterministic SVGs and the report from the
curated development, validation, and frozen test artifacts.
