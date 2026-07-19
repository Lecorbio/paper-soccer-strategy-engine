# Goal-block strategy experiment

This experiment tests a counter to the leading CodinGame bot's goal-line trap.
The leader's public article describes a compact neural evaluator driven by used
edges and rebound-aware distances. Public Arena replays show the same pattern:
the bot repeatedly opens north, builds a lane around its own goal, and uses the
opponent's forced rebounds to finish the enclosure.

## Accepted candidate

`train_replay_value_model.py` downloads the leader's public replay history and
reconstructs every complete-turn boundary. It trains an 1156-8-8-1 value model
from 316 used-edge flags and, for each of 105 vertices, a one-hot capped
rebound-aware distance from zero through seven. Player 1 positions are rotated
180 degrees so both colors share one representation.

The retained model used 88 games (86 wins and 2 losses). Its game-level holdout
contained 970 positions and reached 98.76% classification accuracy after int8
weight quantization. The C++ inference result on the initial position differed
from the Python reference by less than 0.000001.

The production search blends 15% of this learned value with 85% of the existing
hand-written evaluation. It won all 6 full games against the preceding bot
under the real 650 ms/130 ms response budgets. A 30% blend won 5 of 6 locally,
but its live score fell from 41.65 to 41.07 despite splitting its two games with
the leader. Equal-node screens showed that 50% and 75% blends won both colors at
30,000, 100,000, and 300,000 nodes, but their extra evaluation cost made the
15% blend the safer timed choice.

The 15% blend finished the live Arena at rank 8 of 206 with a score of 42.32,
up from 41.65. Its 90-game batch included a win against the leader.

`replay_value_model.json` is the quantized training artifact.
`generate_replay_value_header.mjs` converts it to the compact production header
at `../../replay_value_model.hpp`. `opening_book_gate.cpp` is the paired-game
screen used for blend, replay-correction, and independently configured timing
comparisons.

## Replay-derived counters

After the learned evaluator reached rank 8, losses against the leading Arena
bots were compared with public games in which an elite bot beat the same
opponent from the same color. The retained candidate copies 12 complete
winning response paths, keyed by player ID and the hash of the exact transcript,
and preserves the previously accepted late-game correction. This produces 275
exact response entries without changing search on any unknown position.

`submission_test.cpp` reconstructs every retained replay from the initial
position, checks every table hit against the expected action, and verifies that
the resulting complete turn is legal. Corrections also retain the runtime
state-copy legality guard, so a malformed or colliding entry cannot be applied
to the live state.

## Rejected alternatives

- A direct linear bonus for rebound-aware goal distance tied at low weights and
  lost both colors at larger weights.
- Copying the second-ranked bot's first response (`2`) to the leader's north
  opening reproduced one public win, but lost all three local games as Player 1
  against the production baseline. The replay-specific opening book is not in
  production.
- Increasing the response budgets to 900 ms and 180 ms won a small local
  self-play screen but regressed in the live Arena. Production remains at the
  measured 650 ms and 130 ms budgets.
- A 100% learned-value blend became unstable at deeper node budgets. The model
  remains a guide for AlphaBeta rather than replacing its evaluator.

## Reproduce

Run the trainer with Python and NumPy, then regenerate and verify the submission:

```sh
python3 submissions/codingame/bots/alpha_beta/experiments/goal_block_strategy/train_replay_value_model.py
node submissions/codingame/bots/alpha_beta/experiments/goal_block_strategy/generate_replay_value_header.mjs
node submissions/codingame/tools/generate_submission.mjs alpha_beta
cmake --build build
ctest --test-dir build --output-on-failure
```
