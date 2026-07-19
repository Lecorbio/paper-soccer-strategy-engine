# CodinGame Paper Soccer submission

`paper_soccer_alpha_beta.cpp` is the paste-ready C++20 submission. Copy the
entire file into the CodinGame editor, select C++, and run it in the arena.

The maintained bot uses complete-turn AlphaBeta V2. One search ply is one
complete legal turn, including every mandatory rebound; there is no fixed cap
on the number of edges in a turn. Its evaluation blends 85% of the original
tactical score with 15% from a compact quantized 1156-8-8-1 value model trained
from public Arena replays. The model represents all 316 edges plus
rebound-aware distances to all 105 vertices and rotates Player 1 positions so
both colors share one representation. Its internal response budgets are 650 ms
on the first execution and 130 ms afterward, leaving margin below CodinGame's
1000 ms and 200 ms limits.

The live submission finished rank 8 of 206 with a score of 42.32, improving the
preceding production score of 41.65. Its Arena batch included a win against the
rank-1 bot whose goal-line strategy supplied the training data.

One independently validated replay correction is included for Player 0 after
the exact transcript
`7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474`: action
`42474176`. The bot matches the full slash-delimited transcript, applies the
correction to a state copy, and commits it only when the action is legal.
Unknown transcripts, the wrong player, or an illegal correction use untouched
V2 search. The three replay corrections that failed their frozen gates are not
included.

The paste-ready file is generated from the maintained sources listed in
`alpha_beta.sources`; production does not depend on the experiment tree. The
training and paired-gate evidence is retained in
`experiments/goal_block_strategy/`. Do not hand-edit the generated file.
Rebuild and verify it from the repository root with:

```sh
node submissions/codingame/generate_submission.mjs
node submissions/codingame/generate_submission.mjs --check
cmake --build build
ctest --test-dir build --output-on-failure
```

The generator removes repository-local includes, combines duplicate standard
headers, rejects unexpected dependencies, and enforces the platform-wide
100,000-character source limit. The tests compile the generated file by
itself; verify direction codes, contest rules, atomic complete-turn parsing,
rebound-complete search fallbacks, and interrupted-search legality; exercise
the sole accepted replay correction and every exclusion/fallback path; and
smoke-test the first protocol exchange as both player IDs.
