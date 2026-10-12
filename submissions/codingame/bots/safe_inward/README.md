# Safe inward ordering

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Prefer nonterminal inward rebound edges near the mover’s own goal.

The implementation ID is `safe_inward`. This stable name is used by the build,
generators, and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

The candidate was rejected by its development gate. It remains a reproducible
comparator, without a live promotion claim.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs safe_inward --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_safe_inward_submission \
  papersoccer_codingame_safe_inward_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_safe_inward_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

### Design notes

This candidate starts from the immutable `rank_5` bot and changes only move ordering. In
the mover's own goal shell, a nonterminal inward edge receives a small priority when it
lands on a rebound point and therefore continues the turn.

The frozen move-ordering hypothesis and result are recorded in `EXPERIMENTS.md`. The
candidate was rejected by the development gate and must not be submitted.
