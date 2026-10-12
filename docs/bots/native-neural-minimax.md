# Native neural minimax

[Bot catalog](../../submissions/codingame/bots/README.md) ·
[Standalone C++20 source](../../submissions/codingame/bots/jacek_native_bfm/submission.cpp)

## Design

Complete legal turns are the search actions. Best-first minimax allocates work
with a UCT-style score and evaluates positions with a compact neural value
network. Exact terminal rules and tactical rebound proofs remain separate from
the learned evaluation.

The implementation ID is `jacek_native_bfm`. Its engine and training lineage
are independent of the repository's incumbent bots and replay tables. The
teacher-residual alpha-beta bot is used as an external comparison opponent.

## Usage

The [standalone source](../../submissions/codingame/bots/jacek_native_bfm/submission.cpp)
is a self-contained C++20 program. Its maintained inputs, generated model,
comparison harness, and focused tests live in the
[bot directory](../../submissions/codingame/bots/jacek_native_bfm/).
Use it to reproduce local comparisons; the historical rank below belongs to a
specific checkpoint, rather than every later source in this lineage.

## Results

The strongest recorded live checkpoint was history 62, trained with seed
`20260822`. Its completed window reached rank 5: **63–27 raw**, or **44–27**
after removing opponent forfeits, with zero failures attributable to the bot.
Later diagnostics and rollbacks did not establish rank-4 parity.

The retained source includes instrumentation added after that historical
checkpoint. Its source identity and each live window are recorded separately
in the [frozen bot record](../../submissions/codingame/bots/jacek_native_bfm/README.md#completed-codingame-diagnostics).
The [local leaderboard](../../benchmarks/codingame_leaderboard/README.md)
uses a different measurement from platform rank.

## Testing

From the repository root after [project setup](../development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs jacek_native_bfm --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_jacek_native_bfm_submission \
  papersoccer_codingame_jacek_native_bfm_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_jacek_native_bfm_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [original design and checkpoint record](../../submissions/codingame/bots/jacek_native_bfm/README.md)
contains model identities, training provenance, tactical guarantees, timing
checks, and completed live diagnostics. The
[experiment log](../../submissions/codingame/bots/jacek_native_bfm/EXPERIMENTS.md)
retains the chronological hypotheses and decisions. Those frozen records keep
their original bytes; this overview provides a shorter route through them.
