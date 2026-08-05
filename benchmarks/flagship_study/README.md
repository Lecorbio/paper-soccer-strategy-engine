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
`REPORT.md` are retained here. `REPORT.md` is the performance-focused result;
raw per-unit arena reports stay under the ignored `results/` tree.

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

## Validate the published study

The checked-in manifest is the immutable identity used by every published
result. Validate its current files from the repository root:

```sh
python3 benchmarks/flagship_study/run_study.py validate
```

## Published artifacts

Start with the live [benchmark overview](https://lecorbio.github.io/paper-soccer-strategy-engine/benchmarks/)
for a concise presentation, or [`REPORT.md`](REPORT.md) for the full technical
analysis. The machine-readable development, validation, and test results are
under `data/`; `selection_lock.json` records the selected configurations, and
`charts/` contains the generated figures.

The website reads the checked-in `web/benchmarks/benchmark-results.js`
snapshot. It publishes bot-performance results only and excludes individual
games, execution environments, timestamps, and hashes. Verify or intentionally
regenerate it from the frozen study inputs with:

```sh
python3 benchmarks/flagship_study/web_summary.py --check
python3 benchmarks/flagship_study/web_summary.py --write
```

The completed study cannot be rerun from a later framework checkout. Its exact
execution framework and retired audit attachments remain available in the
annotated `flagship-study-v4-record` tag.
