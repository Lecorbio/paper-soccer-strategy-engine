# First-exchange proof

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Extend exact rebound proofs through the opponent’s reply and the root player’s
counterturn.

The implementation ID is `exchange_proof`. This stable name is used by the build,
generators, and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

The experiment log records exposed model-selection evidence. No new frozen-bank or live
promotion result is claimed.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs exchange_proof --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_exchange_proof_submission \
  papersoccer_codingame_exchange_proof_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_exchange_proof_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

### Design notes

This first-exchange candidate keeps the exact current-turn root and depth-zero proofs
from `rebound_proof`, then checks the first two complete-turn boundaries below each root
action. After the transposition-table probe, exact component proof runs only when
`turn_ply == 1 || turn_ply == 2`.

Those boundaries cover the opponent's immediate reply and the root player's counterturn:
one full exchange. The placement has no board coordinate, tier, phase, score, or learned
threshold. `Unknown` positions continue through the original search unchanged.

The comparison runner reports separate ply-one and ply-two probes, Win hits, Loss hits,
and exact cutoffs. In wall-clock mode it also enforces the declared node budget as a
hard safety cap.

The first-exchange experiment was frozen for a prospective gate. Its exposed
model-selection evidence remains in `EXPERIMENTS.md`; no result on a new frozen bank is
claimed.

Regenerate and verify the paste-ready artifact with:

```sh
node submissions/codingame/tools/generate_submission.mjs exchange_proof
node submissions/codingame/tools/generate_submission.mjs exchange_proof --check
```
