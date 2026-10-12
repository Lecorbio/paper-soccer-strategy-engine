# All-depth rebound proof

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Apply exact rebound proofs at every complete-turn search boundary.

The implementation ID is `all_depth_proof`. This stable name is used by the build,
generators, and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

An experimental candidate with exposed-bank comparison evidence. The separate historical
promotion manifest binds this bot; that binding is not a platform-rank result.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs all_depth_proof --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_all_depth_proof_submission \
  papersoccer_codingame_all_depth_proof_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_all_depth_proof_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

### Design notes

This candidate is the immutable `rank_5` bot plus the exact rebound-component proofs
from `rebound_proof`, applied at every non-root complete-turn search boundary. A proof
can terminate a subtree before complete actions are enumerated; `Unknown` positions
continue through the original search without an evaluation change.

At depth zero the proof runs before evaluation. At greater depth the transposition table
is probed first, and the proof runs only if no valid cached cutoff applies. Exact proof
scores are deliberately not stored because their mate-distance value depends on the
current turn ply.

This directory is an experimental candidate, not a promoted bot. See `EXPERIMENTS.md`
for the exposed-bank evidence and the required gates.
