# Topology ordering

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Prioritize edges that interrupt short rebound-goal corridors.

The implementation ID is `topology`. This stable name is used by the build, generators,
and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

Retained as a local comparator under the historical promotion protocol. Its experiment
log and frozen local leaderboard are the relevant evidence; no live promotion result is
claimed.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs topology --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_topology_submission \
  papersoccer_codingame_topology_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_topology_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

### Design notes

This candidate starts byte-for-byte from the immutable `rank_5` engine. Its single
experiment is capped rebound-goal connectivity used only for move ordering. Near a goal
shell, when the opponent can score with zero or one fresh landing, edges that break that
corridor are searched before edges that preserve it. The replay book, learned value
anchor, evaluation scores, clocks, mate claims, and general alpha-beta search remain
unchanged.

Promotion evidence is owned by `../../promotion/`. A candidate is submission- worthy
only after the initial sentinel, disjoint development/validation/test banks,
generated-artifact tests, timing distribution, and source-size checks all pass the
frozen manifest. Raw wins against `rank_5` alone are diagnostic, not promotion evidence
unless they come from the predeclared paired, source-game-clustered elite banks and
clear every stage threshold.
