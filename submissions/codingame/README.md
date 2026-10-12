# CodinGame bots

Standalone C++20 bots for [Paper Soccer on CodinGame](https://www.codingame.com/multiplayer/bot-programming/paper-soccer),
from complete-turn alpha-beta search to neural tree search and exact rebound
proofs. Each bot has its own source, tests, and design notes.

Start with **[Quantized alpha-beta](bots/quantized_alpha_beta/README.md)**, the
bot behind the best recorded result: **rank 2 of 212 players**, observed on
**9 October 2026**. Its [standalone source](bots/quantized_alpha_beta/submission.cpp)
is 99,032 ASCII characters and uses a four-bit value network with complete-turn
alpha-beta search. The earlier
[27 September record](bots/quantized_alpha_beta/history/2026-09-27/README.md)
contains the same source. These are dated platform observations; they do not
promise a current rank.

**[Browse all 24 bots](bots/README.md)** ·
**[Local comparison results](../../benchmarks/codingame_leaderboard/README.md)** ·
**[Generation and protocol tools](tools/README.md)**

## Layout

```text
submissions/codingame/
├── bots/       standalone bots, tests, design notes, and source history
├── promotion/  frozen comparison protocol and historical evidence
└── tools/      source generation, protocol checks, and replay utilities
```

Most bots pair maintained implementation files with a generated
`submission.cpp`. Quantized alpha-beta is an exact source snapshot with an
identity check instead of a generator. See the
[bot directory guide](bots/README.md#working-with-a-bot) for the distinction.

## Build and test

Run these commands from the repository root after the
[project setup](../../docs/development.md#build-and-test):

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_quantized_alpha_beta_submission
ctest --test-dir build/native -R quantized_alpha_beta --output-on-failure
```

Each bot page gives its own focused checks. For the complete project suite,
run `PAPERSOCCER_BUILD_JOBS=2 ./scripts/build-and-test.sh`.

## Reading the results

The [bot catalog](bots/README.md) distinguishes historical platform ranks,
local match results, and experiments that did not pass their comparison gates.
The frozen local leaderboard compares 22 bots under the same contest rules;
Quantized alpha-beta and Compact value minimax are additional buildable bots,
outside that roster. Adding a bot to the project does not add a result to the
frozen tournament.

The browser's `Rank5DerivedBot` is an adapter of the historical replay-guided
alpha-beta bot. Its rules and work limits differ from CodinGame, so platform
ranks do not transfer to the demo.

## Historical evidence

Source snapshots, frozen reports, seeds, and recorded game identities retain
their original bytes. Bot-specific records are linked from each bot page;
cross-tool research records remain at their original paths. The
[history guide](../../docs/history.md) explains those boundaries, and the
[campaign catalog](../../docs/campaigns.md) records completed work.
