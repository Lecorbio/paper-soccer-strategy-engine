# Rebound-proof challenger

[Bot catalog](../README.md) · [Standalone C++20 source](submission.cpp)

## Design

Alpha-beta with exact rebound-component proofs and narrowly matched replay corrections.

The implementation ID is `challenger`. This stable name is used by the build,
generators, and historical evidence.

## Usage

[`submission.cpp`](submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live beside it. Use the source for local comparisons; the
results below describe its measured status.

## Results

A principal-variation-search ablation reached rank 8 and was reverted after real-clock
regressions. The retained source and chronological comparisons are recorded in the
experiment log.

See the [frozen local
leaderboard](../../../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../../../../docs/development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs challenger --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_challenger_submission \
  papersoccer_codingame_challenger_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_challenger_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

<details>
<summary>Detailed design and comparison notes</summary>

This folder contains a distinct challenger derived from the immutable `../rank_5/`
reference. It retains the verified replay book, 88-game evaluator, complete-turn
alpha-beta search, safe fallback, and 650/130 ms search budgets.

The challenger adds exact rebound-component analysis at the root and at every depth-zero
turn boundary. It traverses only unused edges whose intermediate landing points already
grant a rebound (visited vertices or the boundary). A reachable attacking goal is
therefore a proven complete winning turn, while a closed component with no safe handoff
is a proven mover loss. The root can return a reconstructed legal scoring action
immediately; frontier proofs give alpha-beta an exact mate value before the expensive
static evaluator. This is especially useful defensively when a candidate handoff would
give the opponent a multi-edge goal on the next turn.

The later selective goal-zone extension was removed after multiple arena batches
underperformed the simpler proof-only candidate. Unknown depth-zero components therefore
return directly to the cached evaluator.

Two exact replay branches retain wins against jacek. Normal search remains in control
after `0/0`; only the player-0 suffix after `0/0/1/6` selects `7`, matching candidate
v1's public win over jacek. Challenger v9's replacement `43` lost both matching games in
its 0-6 jacek batch, while restoring `7` changes those two losses and no completed v9
win. The recurrent player-1 `.../5330` position selects `2722`. Both paths fall back to
normal search on any opponent deviation.

Three further exact branches repair v3's reproducible arena horizon failures: `61` after
`6/7/5` as player 1, `614347` after the EricSMSO `.../0527271` prefix as player 1, and
`5` after the exact player-0 EricSMSO turn-14 prefix. Together they would have changed
twelve v3 losses and one win; their actions are backed by repeated completed wins and
remain exact-prefix only.

Two later replay choices are deliberately narrow. Player 1 selects `0617271` after exact
derjack prefix `6/7/5/61/44/53/0`; this state has a 10-0 recovered record for that
action versus 5-5 for the replaced action, and the path stops after a further exact
opponent reply and 10-0 action `42` to avoid later collisions. Against Snekkers, player
0 selects `2` only after the exact alternate prefix ending `.../31/4167`. Disproven
broad jacek eligibility and unexercised player-1 Snekkers/jacek replacements were
removed.

Principal variation search passed deterministic equivalence and improved fixed-node
gates, yet version 35 showed harmful real-clock player-0 drift and settled rank 8. The
rejected PVS and TT-retention changes were removed, so the production search path and
transposition replacement again match v4 exactly.

`comparison_gate.cpp` compiles this maintained engine and `rank_5/bot.cpp` in separate
namespaces. It compares them with identical hard node caps from curated recurrent
histories and deterministic random complete-turn openings, then swaps colors for every
starting state. It also prints shared-position actions, completed depths, node counts,
scores, timings, and proof counts.

Generate and verify from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs challenger
node submissions/codingame/tools/generate_submission.mjs challenger --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
./build/native/papersoccer_codingame_challenger_comparison_gate 8 30000 200
./build/native/papersoccer_codingame_challenger_timing_probe
```

The generated `submission.cpp` is the only paste-ready artifact. Do not edit it
directly. Current local and arena evidence is recorded in `EXPERIMENTS.md`.

</details>
