# CodinGame Paper Soccer submission

`paper_soccer_alpha_beta.cpp` is the paste-ready C++20 submission. Copy the
entire file into the CodinGame editor, select C++, and run it in the arena.

The maintained bot uses complete-turn AlphaBeta V2. One search ply is one
complete legal turn, including every mandatory rebound; there is no fixed cap
on the number of edges in a turn. Its internal response budgets are 650 ms on
the first execution and 130 ms afterward, leaving margin below CodinGame's
1000 ms and 200 ms limits.

One independently validated replay correction is included for Player 0 after
the exact transcript
`7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474`: action
`42474176`. The bot matches the full slash-delimited transcript, applies the
correction to a state copy, and commits it only when the action is legal.
Unknown transcripts, the wrong player, or an illegal correction use untouched
V2 search. The three replay corrections that failed their frozen gates are not
included.

The paste-ready file is generated from the maintained sources listed in
`alpha_beta.sources`; production does not depend on the experiment tree. Do
not hand-edit the generated file. Rebuild and verify it from the repository
root with:

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
