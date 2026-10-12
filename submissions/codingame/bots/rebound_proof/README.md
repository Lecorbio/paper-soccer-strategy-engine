# Rebound-component proof

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Prove current-turn scoring paths and blocked rebound-component losses.

The implementation ID is `rebound_proof`. This stable name is used by the build,
generators, and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

The candidate was rejected by the predeclared validation color guard despite its local
strength. It was not promoted.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs rebound_proof --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_rebound_proof_submission \
  papersoccer_codingame_rebound_proof_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_rebound_proof_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

### Design notes

This candidate reproduces the historical pure exact-proof patch on top of the immutable
`rank_5` bot. It reconstructs a current-turn route to the attacking goal when one exists
entirely through rebound vertices, and proves a closed component loss under the contest
blocked-mover rule.

The candidate is exact and locally strong, but it was rejected by the predeclared
validation color guard. See `EXPERIMENTS.md`; do not submit it.
