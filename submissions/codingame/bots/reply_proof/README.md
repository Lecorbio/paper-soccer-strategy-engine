# First-reply proof

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Extend exact rebound proofs to the opponent’s immediate reply.

The implementation ID is `reply_proof`. This stable name is used by the build,
generators, and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

The experiment log records exposed development evidence and prospective validation
constraints. No sealed final-test result is claimed.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs reply_proof --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_reply_proof_submission \
  papersoccer_codingame_reply_proof_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_reply_proof_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

### Design notes

This first-reply candidate starts from the exact `rebound_proof` implementation. It
keeps the unconditional current-turn root proof and depth-zero proof, then adds one
TT-first proof at the first complete-turn boundary below each root action (`turn_ply ==
1`).

The placement asks a semantic tactical question: does this candidate action hand the
opponent a forced rebound-component Win or Loss? It has no board coordinate, phase,
degree, or learned threshold. `Unknown` positions continue through the original search
unchanged.

The experiment was frozen for prospective validation. Its exposed-bank evidence and
decision constraints remain in `EXPERIMENTS.md`; no sealed final-test result is claimed
here.

Regenerate and verify the paste-ready artifact with:

```sh
node submissions/codingame/tools/generate_submission.mjs reply_proof
node submissions/codingame/tools/generate_submission.mjs reply_proof --check
```
