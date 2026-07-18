# CodinGame Paper Soccer submission

`paper_soccer_alpha_beta.cpp` is the paste-ready C++20 submission. Copy the
entire file into the CodinGame editor, select C++, and run it in the arena.

The adapter translates CodinGame's direction digits into engine moves, applies
the opponent's complete turn, and prints one complete turn of its own. It uses
the contest's own-goal rules: Player 0 scores in the north goal, Player 1 scores
in the south goal, and entering the other goal loses. The search is
limited to 850 ms on the bot's first execution and 150 ms afterward, leaving
margin below CodinGame's 1000 ms and 200 ms limits.

The paste-ready file is generated from the maintained engine sources listed in
`alpha_beta.sources` plus `alpha_beta_adapter.cpp`. Do not hand-edit the
generated file. Rebuild and verify it from the repository root with:

```sh
node submissions/codingame/generate_submission.mjs
node submissions/codingame/generate_submission.mjs --check
cmake --build build
ctest --test-dir build --output-on-failure
```

The generator removes repository-local includes, combines duplicate standard
headers, rejects unexpected dependencies, and enforces the platform-wide
100,000-character source limit. The tests compile the generated file by itself, verify the
eight direction codes and rebound turn boundaries, replay composed actions,
and smoke-test the first protocol exchange as both player IDs.
