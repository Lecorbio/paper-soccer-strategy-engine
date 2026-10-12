# Replay-guided alpha-beta

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Complete-turn alpha-beta with a compact replay-value model and verified replay book.

The implementation ID is `rank_5`. This stable name is used by the build, generators,
and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

History version 26 ranked 5 of 206 with score 42.42773147296124. The generated source is
93,005 ASCII characters.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs rank_5 --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_rank_5_submission \
  papersoccer_codingame_rank_5_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_rank_5_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

<details>
<summary>Detailed design and comparison notes</summary>

**Status:** immutable verified predecessor and the contest source behind the separately
named `Rank5DerivedBot` demo adapter. The later teacher-residual snapshot is
[`rank_4`](../rank_4/README.md); this directory is retained for provenance,
reproduction, and historical comparisons.

This folder originally replaced the production reference maintained in `../alpha_beta/`.

`bot.cpp` is the maintained implementation and `submission.cpp` is the single
paste-ready C++20 submission. The generated file is 93,005 ASCII characters, below
CodinGame's 100,000-character limit. Search uses 650 ms on the first execution and 130
ms afterward, leaving margin below the contest's 1,000 ms and 200 ms limits.

At its recorded promotion point, the maintained source was the strongest completed arena
candidate: history version 26, agent `6561779`, submission `41015554`, rank 5 of 206
with score `42.42773147296124`. Its SHA-256 is
`f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`.

The candidate retains complete-turn alpha-beta search and its safe fallback. Its static
position evaluator is generated from `replay_value_model.json`, an 88-game public
rank-leading replay corpus; the model generator is part of the same freshness check as
the submission. The candidate also adds a generated exact-history replay book: a
response is eligible only when the player ID and the complete slash-delimited game
history match a publicly verified path. The proposed full turn is applied to a state
copy and used only if every edge is legal and the rebound sequence is complete.
Unmatched, ambiguous, or illegal entries fall back to normal search.


Useful analysis gates:

```sh
python3 submissions/codingame/tools/screen_replay_book.py \
  AGENT_ID submissions/codingame/bots/rank_5/replay_book.json
python3 submissions/codingame/tools/analyze_arena.py AGENT_ID --pretty
./build/native/papersoccer_codingame_rank_5_timing_probe
```

`replay_book.json` is the auditable source of replay paths and provenance;
`generate_replay_book.mjs` rejects invalid actions, parity errors, and conflicting
responses at an identical full history. `replay_value_model.json` is the corresponding
auditable evaluator artifact. Do not hand-edit any generated C++ files or headers. Arena
evidence and rejected experiments are recorded in `EXPERIMENTS.md`.

</details>
