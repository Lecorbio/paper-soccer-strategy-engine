# Handoff-frontier proof

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Add a small safe-handoff evaluation feature after an inconclusive exact proof.

The implementation ID is `frontier_proof`. This stable name is used by the build,
generators, and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

The retained evidence is an exposed-bank sensitivity comparison. It does not establish
promotion.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs frontier_proof --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_frontier_proof_submission \
  papersoccer_codingame_frontier_proof_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_frontier_proof_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

### Design notes

This candidate starts from `all_depth_proof`. Its exact rebound-component Win/Loss
checks still run at every complete-turn search boundary. When the depth-zero check
returns `Unknown`, the same scan also supplies the number of unique fresh handoff
endpoints at the component frontier.

The evaluator adds `mover_sign * frontier_count * 10`. The weight is one half of the
existing mobility weight. Endpoint uniqueness makes the feature invariant under
horizontal reflection, while mover sign makes it antisymmetric under the 180-degree
player rotation. Proven Win/Loss scores and pruning are unchanged; the term applies only
to previously heuristic `Unknown` leaves.

This directory is an experimental candidate, not a promoted bot. Its retained evidence
is an exposed-bank sensitivity screen documented in `EXPERIMENTS.md`. That result does
not imply a passed fresh-bank comparison.
