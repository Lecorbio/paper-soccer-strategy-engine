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

## Frozen record and release assets

The annotated
[`flagship-study-v4-record`](https://github.com/Lecorbio/paper-soccer-strategy-engine/tree/flagship-study-v4-record)
tag preserves the exact completed-study framework, immutable manifest and
selection lock, three phase-data files, and retired audit attachments. Validate
that historical record from a detached checkout:

```sh
git fetch --tags origin
git checkout --detach flagship-study-v4-record
python3 benchmarks/flagship_study/run_study.py validate
```

The release packager, executive abstract, and compact summaries were added
later on `main`; they are intentionally not present in the historical tag. The
frozen manifest, selection lock, and decision-level development, validation,
and test JSON files on current `main` remain byte-identical to the tag. Build
and verify the upload-ready assets from current `main`:

```sh
git switch main
git pull --ff-only origin main
python3 benchmarks/flagship_study/run_study.py validate
python3 benchmarks/flagship_study/release_summary.py --check
python3 benchmarks/flagship_study/package_release.py build
python3 benchmarks/flagship_study/package_release.py check
```

The commands write only ignored files under
`results/releases/flagship-study-v4/`:

- `flagship-study-v4-core.zip` contains the manifest, selection lock, promoted
  report, release notes, three charts, and compact `summary.json`,
  `pairwise.csv`, and `configurations.csv` files under
  `benchmarks/flagship_study/summary/`.
- `flagship-study-v4-decision-data.zip` contains the three decision-level phase
  JSON files separately, so the small core artifact remains convenient.
- `SHA256SUMS` authenticates both ZIP assets.

The packager requires the tracked summaries to be byte-for-byte fresh, fixes
ZIP member order, timestamps, Unix modes, and compression settings, then
verifies every archived member against its source and checks the immutable
inputs against the tag. A GitHub Release made from these assets must target the
existing `flagship-study-v4-record` tag; do not recreate or move the tag to
current `main`.

The compact summary records both protected identities:

- Authentic rank-5 submission SHA-256:
  `f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`.
- Neural model: `models/jacek_article_value_model.json`, SHA-256
  `57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084`.

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

The completed study cannot be rerun from a later framework checkout. Current
`main` supports validation, presentation, and deterministic packaging of the
already-frozen record; it does not replace the tag's execution framework.
