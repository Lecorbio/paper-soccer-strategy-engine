# CodinGame Paper Soccer submission

**Status:** retained historical production baseline. The current production
snapshot is [`rank_4`](../rank_4/README.md); this artifact remains buildable for
reproduction, regression tests, and the local CodinGame-rules leaderboard.

`submission.cpp` is the paste-ready C++20 submission. Copy the
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

The last fully measured baseline finished rank 8 of 206 with a score of 42.32,
improving the preceding production score of 41.65. Its Arena batch included a
win against the rank-1 bot whose goal-line strategy supplied the training data.

The retained artifact also contains exact responses copied from 12 public
winning replays against opponents that beat the baseline. Its 275-entry table
covers every retained response along those complete continuations, plus one
independently screened late-game correction. A correction requires both the
expected player ID and the hash of the complete slash-delimited transcript.
The bot applies the proposed action to a state copy and commits it only when
the whole action is legal. Unknown transcripts, hash misses, the wrong player,
or an illegal action fall back to untouched V2 search.

The paste-ready file is generated from the maintained sources listed in
`sources.txt`; the generated submission does not depend on the experiment
tree. The training and paired-gate evidence is retained in
`experiments/goal_block_strategy/`. Do not hand-edit `submission.cpp`.
Rebuild and verify it from the repository root with:

```sh
node submissions/codingame/tools/generate_submission.mjs alpha_beta
node submissions/codingame/tools/generate_submission.mjs alpha_beta --check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The generator removes repository-local includes, combines duplicate standard
headers, strips source comments and redundant blank lines, rejects unexpected
dependencies, and enforces the platform-wide 100,000-character source limit.
The tests compile the generated file by itself; verify direction codes,
contest rules, atomic complete-turn parsing, rebound-complete search fallbacks,
and interrupted-search legality; reconstruct all 12 copied replays and check
every retained response for an exact legal state transition; exercise the
late-game correction and exclusion/fallback paths; and smoke-test the first
protocol exchange as both player IDs.
