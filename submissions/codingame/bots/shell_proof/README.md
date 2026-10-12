# Goal-shell rebound proof

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Apply exact leaf proofs near the attacking goal shell.

The implementation ID is `shell_proof`. This stable name is used by the build,
generators, and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

A frozen local comparator with source, focused tests, timing probe, and comparison
harness. No CodinGame promotion result is claimed.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs shell_proof --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_shell_proof_submission \
  papersoccer_codingame_shell_proof_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_shell_proof_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

This source is retained in the frozen local comparison roster. Its focused checks and
the published tournament preserve the reproducible comparison. Frozen source snapshots
and reports keep their recorded identities.

### Design notes

**Status:** archived experimental comparator, not a production or platform-rank claim.
It remains buildable and participates in the frozen local CodinGame-rules leaderboard.

This candidate starts from the immutable `rank_5` engine. It keeps the exact
current-possession rebound-goal proof at the root and runs the same Win/Loss component
analysis at depth-zero boundaries only when the mover is near the attacking goal shell.
`Unknown` positions continue through the original evaluation and complete-turn
alpha-beta search unchanged.

The generated submission is 96,619 characters with SHA-256
`857f27ceeb5bb9cb514622eddb0b326b6fd2b1d55b6ed54e20869879f45fcce6`. Its checked-in
evidence is the source, focused tests, comparison harness, timing probe, and frozen
local leaderboard result; no CodinGame promotion result is claimed.

Generate and verify it from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs shell_proof --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --target papersoccer_codingame_shell_proof_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_shell_proof_submission_test$' \
  --output-on-failure
```
