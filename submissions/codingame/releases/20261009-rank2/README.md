# Paper Soccer — rank 2 of 212

The strongest recorded public result is **rank 2 of 212 CodinGame Paper Soccer
players**, observed on **9 October 2026 at 06:39:43 UTC**. This is a dated
leaderboard snapshot; ranks and the player count change over time.

![CodinGame Paper Soccer status showing rank 2 of 212](leaderboard.png)

The image is a crop of the archived platform status bar. The displayed rank
and player count are unchanged; the game board, editor and game transcripts
are omitted. Its provenance and source identity are in [manifest.json](manifest.json).

## Use the exact bot

Paste the entire [submission.cpp](submission.cpp) into the CodinGame C++ editor.
This self-contained C++20 source is **99,032 ASCII characters**, below the
100,000-character platform limit. SHA-256:

```text
78e731a81ce7cc702a200d0832a6c75e967a11b12452051724ab1387c226e4b8
```

Build from the repository root:

```sh
mkdir -p build/codingame
clang++ -std=c++20 -O3 -DNDEBUG \
  submissions/codingame/releases/20261009-rank2/submission.cpp \
  -o build/codingame/rank2
cmake -E sha256sum submissions/codingame/releases/20261009-rank2/submission.cpp
```

The bot combines complete-turn alpha-beta search, a bias-free
**6,301 → 12 → 8 → 1 value network**, four-bit weights, corrected proof ranking
and flat goal-distance traversal. Search budgets are **550 ms initially and
140 ms afterward**. Heuristic values remain separate from terminal proofs.

## Result and provenance

These are the same source bytes as the [27 September rank-3 release](../20260927-rank3/README.md).
That release records the completed calibration and independent local
confirmation. The subsequent rank-2 observation involved no new upload and
does not establish a new formal qualification block or a new game win rate.
The saved editor identity audit matched this source; the platform does not
expose a remote source digest.

This immutable release is separate from the maintained `bots/` registry and
the browser's `Rank5DerivedBot` adapter. The 22-bot local leaderboard remains a
different benchmark. Preserve these source bytes rather than regenerating or
reformatting them. The ongoing 8/8 experiments have not replaced this release.
