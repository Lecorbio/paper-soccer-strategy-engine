# Conservative frontier proof

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Use half the handoff-frontier evaluation weight while preserving exact proofs.

The implementation ID is `conservative_frontier_proof`. This stable name is used by the
build, generators, and historical evidence.

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
node submissions/codingame/tools/generate_submission.mjs conservative_frontier_proof --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_conservative_frontier_proof_submission \
  papersoccer_codingame_conservative_frontier_proof_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_conservative_frontier_proof_(submission_current|submission_test|protocol_smoke_test)$' \
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

The evaluator adds `mover_sign * frontier_count * 5`. Endpoint uniqueness makes the
feature invariant under horizontal reflection, while mover sign makes it antisymmetric
under the 180-degree player rotation. Proven Win/Loss scores and pruning are unchanged;
the term applies only to previously heuristic `Unknown` leaves.

This is the conservative member of the frontier experiment family. It remains an
experimental candidate rather than a promoted bot. Its exposed-bank sensitivity results
are recorded in `EXPERIMENTS.md`.

Regenerate and verify the paste-ready artifact with:

```sh
node submissions/codingame/tools/generate_submission.mjs conservative_frontier_proof
node submissions/codingame/tools/generate_submission.mjs conservative_frontier_proof --check
```

The focused CMake targets are
`papersoccer_codingame_conservative_frontier_proof_submission`,
`papersoccer_codingame_conservative_frontier_proof_submission_test`,
`papersoccer_codingame_conservative_frontier_proof_timing_probe`, and
`papersoccer_codingame_conservative_frontier_proof_comparison_gate`.
