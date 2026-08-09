# Rank-4 exact-exchange experiment

`rank_4_exchange` is an experimental, non-promoted successor to the live
`rank_4` bot. It keeps rank 4's replay book, replay-value anchor, 100% compact
teacher residual, mover-relative evaluation, and complete-turn alpha-beta
decision path. It adds an exact graph proof for forced rebound goals and
blocked losses at the root, at depth-zero leaves, and across the first full
reply/counterturn exchange.

The proof traversal uses generation marks instead of clearing a board-sized
array for every probe. A cached leaf evaluation is also treated as proof that
the same state's earlier rebound analysis returned `Unknown`; this skips a
repeat traversal without changing any score, cache replacement, node count,
or fixed-work action. Exact win/loss leaves are never stored as heuristic
evaluations, so the shortcut cannot hide a mate.

The generated paste-ready source is **91,259 characters**, below CodinGame's
100,000-character cap, with SHA-256:

```text
e72bb4bdc3377d0a4602fd807a31083683677c4516f318d75352b60d63355f20
```

This source has not been submitted. It beat the live rank-4 source 58-48 in
106 paired games at both 5,000 and 30,000 nodes, but lost 51-55 at the
intermediate 10,000-node horizon. The 5,000-node result had balanced 29/29
candidate wins and no screen had unfinished games. The isolated equal-clock
800/165 ms screen was 8-10, so the horizon-sensitive experiment does not
satisfy a performance-promotion gate. Rank 4 remains the maintained live bot.

Generate and verify from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs rank_4_exchange
node submissions/codingame/tools/generate_submission.mjs rank_4_exchange --check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build -R rank_4_exchange --output-on-failure
```

Run the direct fixed-work comparison against the actual rank-4 source:

```sh
./build/papersoccer_codingame_rank_4_exchange_comparison_gate 12 30000 200
```

The hypotheses, complete ablation results, timing caveats, and no-submission
decision are in `EXPERIMENTS.md`.
