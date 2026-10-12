# Fresh-corpus neural minimax

[Bot catalog](../../submissions/codingame/bots/README.md) ·
[Standalone C++20 source](../../submissions/codingame/bots/jacek_arena_bfm/submission.cpp)

## Design

This independent bot combines complete-turn generation, best-first minimax,
and a compact neural value network. Equivalent successor positions are
deduplicated before search; generation, move ordering, and sparse inference
use mover-relative coordinates.

The implementation ID is `jacek_arena_bfm`. Its selected offline model was
trained from random initialization on an **8,000-game scratch corpus**. It
does not inherit another bot's source, checkpoint, replay book, or labels.

## Usage

The [standalone source](../../submissions/codingame/bots/jacek_arena_bfm/submission.cpp)
is the retained offline artifact. Its engine, model, comparison harness, and
tests are in the [bot directory](../../submissions/codingame/bots/jacek_arena_bfm/).
The selected offline source was not uploaded; the earlier live collection
source has its own separate safety record.

## Results

The selected offline artifact remained below the frozen history-62 neural
minimax reference. Its
[comparison record](../../submissions/codingame/bots/jacek_arena_bfm/README.md#current-canonical-offline-source)
documents the individual source versions and local measurements.

One earlier 90-game live collection window contained **three bot timeouts**,
74 clean rule-terminal games, and 13 opponent operational failures. That
failed the safety requirement, and none of the collection games entered
training. The [live safety record](../../submissions/codingame/bots/jacek_arena_bfm/README.md#live-safety-snapshot)
also records the separate emergency rollback. No platform improvement is
claimed for this bot.

## Testing

From the repository root after [project setup](../development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs jacek_arena_bfm --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_jacek_arena_bfm_submission \
  papersoccer_codingame_jacek_arena_bfm_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_jacek_arena_bfm_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [original design and results record](../../submissions/codingame/bots/jacek_arena_bfm/README.md)
covers corpus provenance, model identities, symmetry tests, offline
comparisons, and live safety. The
[experiment log](../../submissions/codingame/bots/jacek_arena_bfm/EXPERIMENTS.md)
retains the detailed decisions. The original records and bot directory remain
unchanged; this overview links their useful entry points.
