# Rank-4 full-turn best-first minimax experiment

`rank_4_fullturn_bfm` is an isolated search experiment built around the exact
quantized evaluator, replay corrections, rules, and 800/165 ms deployment
clock of `../rank_4`. It does not replace or modify the maintained rank-4 bot.

The candidate makes complete legal turns its explicit tree actions. It samples
at most 250 actions with a deterministic mixed DFS/BFS deque walk, assigns a
value to every newly created complete-turn child, selects a frontier with an
article-style UCT score, and backs values up with minimax rather than Monte Carlo
averaging. Exact attacking-goal and own-goal BFS probes bracket a
path-sensitive, 32-extraction handoff probe, reserving prioritized results
before ordinary capped sampling so an immediate goal cannot disappear solely
because of traversal order. Handoff-probe extractions count toward fixed work.
Capped sampling uses a 64-bit canonical seed and direction frame across
identity, horizontal reflection, 180-degree player rotation, and their
composition. Rebound-goal BFS uses canonical direction order and directly
prefers a legal center-goal edge.
Every retained handoff is classified on its actual result position; exact
forced-cutoff and opponent-immediate-win classifications propagate as solved
mate-distance children. The precise guarantees and bounded limits are frozen
in the experiment ledger.

These symmetry measures make the focused transformed-state fixtures pass, but
they are not a universal stabilizer-equivariance proof. In particular, cap-one
completed-turn retention need not return a symmetry-fixed action for every
self-symmetric position. That residual limitation is recorded as a no-submit
consideration in the ledger.

Tree expansion commits a retained child batch atomically. If the batch cannot
fit a hard work or node cap, the only partial-commit exception is one generated
proven win for the mover—a terminal win or exact forced cutoff—when one work
unit and child slot still remain.

The CodinGame program emits the selected complete turn as one encoded output
line. It does not cache or emit primitive segments across protocol rounds.
Received opponent turns and replay corrections are applied atomically to a
copy before the authoritative state is updated.

This directory owns its generated CodinGame source, focused tests, timing
probe, comparison harness, clock-screen recorder, replay decision auditor,
strict offline decision-audit analyzer, and experiment ledger. Generate and
check it from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs rank_4_fullturn_bfm
cmake -S . -B build
cmake --build build -j4 --target \
  papersoccer_codingame_rank_4_fullturn_bfm_submission_test \
  papersoccer_codingame_rank_4_fullturn_bfm_timing_probe \
  papersoccer_codingame_rank_4_fullturn_bfm_comparison_gate
ctest --test-dir build --output-on-failure \
  -R 'rank_4_fullturn_bfm'
```

The comparison harness uses only its five public in-source openings and
deterministically generated seed batches; it has no promotion-bank or
`matches.json` input. Mixed even/odd opening depths and explicit search-policy
knobs support small 800/165 ms hypothesis screens. `run_clock_screen.py`
records the exact command, source and gate hashes, clock maxima, and operational
counters in a write-once JSON report. Fixed work remains a reproducibility and
profiling tool, not a strength criterion.

The synchronized generated source is 99,955 ASCII characters, SHA-256
`c268286020ae841e4b5442e3578107ab32ded151e460a5bbe0495cb9b0b19d87`.
Regenerate and recheck both values after any source change.
Its exploratory policy uses final visit weight `0`, the full 250-action cap at
root and deeper nodes, and normal BFM expansion. Root-only and narrower-deeper
settings remain explicit diagnostic hooks; both were rejected in clock screens.

The exact formulas, tactical guarantees, frozen promotion gates, commands,
and results are recorded in [EXPERIMENTS.md](EXPERIMENTS.md). The final
equal-clock result was 24-82 over 106 games, with both deterministic seed
batches and both color thresholds failing. That remains a reproducible
promotion rejection, but it no longer forbids exploratory live submissions.
An asserted `dd1192...` artifact was later tested publicly as agent `6606663`,
submission `41119120`; public APIs cannot prove the editor bytes, so replay
evidence retains `source_binding_status=asserted-not-api-verified`.
The current `c268...` artifact was then copy-back verified in the authenticated
editor immediately before exploratory upload as history version 60. Public
battle metadata identifies it as agent `6608659`, submission `41121957`; this
is diagnostic only and retains the same conservative public-API binding label.

`collect_arena_batch.py` archives complete public batches behind a frozen
ID-only exclusion registry, and `papersoccer_fullturn_replay_decision_auditor`
compares observed actions with raw BFM, replay-book control flow, rank 4, and a
deterministic root action set. The current continuation results and rejected
search/evaluator ablations are appended to the ledger rather than rewriting
the frozen promotion result.
