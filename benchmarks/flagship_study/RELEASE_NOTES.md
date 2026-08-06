# Flagship study v4

This frozen, preregistered study compared four competitive Paper Soccer bots
under standard 8×10 demo rules and a 50 ms validation p95 latency constraint.

## Headline results

- **4,800 decisive test games:** 2,400 color-swapped pairs across four opening
  depths, with zero truncations.
- **Neural versus hand alpha-beta:** the selected neural entrant scored 60.4%
  (pair-clustered 95% CI 57.4%–63.5%), supporting the conclusion that it is
  stronger under this study design.
- **Neural versus Rank5DerivedBot:** the selected neural entrant scored 51.4%
  (pair-clustered 95% CI 48.2%–54.5%), so this comparison remains statistically
  unresolved.

Rank5DerivedBot adapts search code from the rank 5/206 CodinGame submission to
different demo rules and a fixed-work profile. These measurements do not
evaluate the authentic ranked submission.

## Artifact identities

- Authentic rank-5 submission SHA-256:
  `f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`.
- Frozen neural model: `models/jacek_article_value_model.json`, SHA-256
  `57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084`.

## Assets

- `flagship-study-v4-core.zip` contains the frozen manifest, selection lock,
  report, release notes, three SVG charts, and compact JSON/CSV summaries.
- `flagship-study-v4-decision-data.zip` contains the decision-level
  development, validation, and test JSON files.
- `SHA256SUMS` authenticates both ZIP files.

## Validate the frozen record

```sh
git fetch --tags origin
git checkout --detach flagship-study-v4-record
python3 benchmarks/flagship_study/run_study.py validate
```

## Rebuild and verify the release assets

The packager and promoted presentation live on current `main`; the immutable
manifest, selection lock, and three decision-level JSON files remain
byte-identical to the tag.

```sh
git switch main
git pull --ff-only origin main
python3 benchmarks/flagship_study/run_study.py validate
python3 benchmarks/flagship_study/release_summary.py --check
python3 benchmarks/flagship_study/package_release.py build
python3 benchmarks/flagship_study/package_release.py check
cd results/releases/flagship-study-v4
shasum -a 256 -c SHA256SUMS
```

The GitHub Release must target the existing `flagship-study-v4-record` tag.
