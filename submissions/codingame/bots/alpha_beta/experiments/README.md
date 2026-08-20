# Paper Soccer experiment archive

This directory intentionally keeps only the compact historical evidence behind
the former alpha-beta production bot. Current production is documented in
[`rank_4`](../../rank_4/README.md). Raw harnesses, generated binaries, caches,
and rejected or superseded experiment implementations were not retained.

## Recorded production result

The retained CodinGame submission finished rank 8 of 206 with a score of
42.32. It adds the replay-trained value model documented in
`goal_block_strategy/` and improves the preceding production score of 41.65.
The earlier pure V2 submission finished rank 13 with 37.67.

The preceding historical artifact was named
`paper_soccer_v2_replay_book.cpp`, contained 64,818 characters, and had SHA-256
`e79390a1833d9b9b28a22d7ff8a662bdb32d77c57b6d24152e0a888aef0cb66c`.
That byte-exact source is omitted because it contained an attribution banner
disallowed by the repository's public-file policy. The retained source
supersedes it while preserving its accepted replay correction.

The accepted correction was exact and narrow: Player 0, transcript
`7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474`, action
`42474176`. It improved its frozen continuation screen from 0/8 wins to 8/8.
Three other proposed corrections were rejected. The correction did not occur
in the five public Arena replays inspected after submission, so the rank result
is evidence for the combined bot rather than causal evidence for activation.

## Rejected follow-ups

- PVS scored 31-29 across 60 color-swapped games. Its early/middle/late scores
  were 45%/55%/55%, but the paired 95% interval was 41.67%-61.67%. It passed
  the challenger gate and failed the production-replacement confidence gate.
- Mining the two new Player-0 losses produced two candidate corrections. The
  first fell from the recorded action's 4/8 wins to 1/8. The second improved
  from 4/8 to 5/8 but missed the frozen 6/8 and improvement requirements. Both
  were rejected.
- At the Laars loss state, all 15 legal complete actions were enumerated and 12
  alternatives were screened across eight frozen cells each. Actions `5`,
  `13`, and `123` were best at 6/8 wins, below the required 7/8 and three-win
  improvement over the 4/8 recorded baseline. No action was accepted.

## Retained evidence

`v2_replay_correction_gate/` contains the frozen gate criteria, the complete
64-game report, the concise gate evaluation, preflight and benchmark summaries,
and the recorded live Arena result. Its README lists the exact compact archive.
