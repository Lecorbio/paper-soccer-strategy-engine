# Paper Soccer Strategy Engine

This repository contains a deterministic C++20 baseline for paper soccer with:

- A pure rules engine (`papersoccer_core`)
- A terminal CLI for human and bot play (`papersoccer_cli`)
- Seeded `RandomBot`, Monte Carlo Tree Search (`MctsBot`), and deterministic
  alpha-beta (`AlphaBetaBot`) opponents
- A native arena for paired strength matches and position throughput measurements
- A JSON replay exporter for bot self-play (`papersoccer_replay_export`)
- A static browser game powered by the same C++ engine through WebAssembly
- A generated, paste-ready [CodinGame entry](submissions/codingame/README.md)
- Dependency-free tests integrated with CTest (`papersoccer_tests`)

The rules engine remains the single source of truth for human play, bots, replays, and
browser sessions. The first complete MCTS implementation is available in native builds and
uses the existing `GameState` and rule APIs without replacing them with a separate model.

## Rules Implemented (Kurnik-style defaults)

- Board coordinates are zero-based: `x in [0,8]`, `y in [0,12]`
- Field points exclude the goals: `x in [0,8]`, `y in [1,11]`
- Start point: `(4,6)`
- 8-direction moves to neighboring points
- Segments are undirected and cannot be reused
- Movement along the outer boundary lines is forbidden
- Player 1 attacks north goal row `y=0`, Player 2 attacks south goal row `y=12`
- North goal nodes: `(3,0)`, `(4,0)`, `(5,0)`
- South goal nodes: `(3,12)`, `(4,12)`, `(5,12)`
- Goal entries are legal only from the 3-point mouth on each side
- Goal-post segments still count as walls, so straight entry from the side mouth points is illegal
- Legal north-goal entries are:
  - from `(3,1)` only to `(4,0)`
  - from `(4,1)` to `(3,0)`, `(4,0)`, `(5,0)`
  - from `(5,1)` only to `(4,0)`
- Entering opponent goal wins immediately
- Extra turn when landing on:
  - Any previously visited point
  - Any wall boundary point; the open center of each goal mouth is excluded
- If the player to move has zero legal moves at turn start, that player loses

## Alpha-Beta Bot

`AlphaBetaBot` is a deterministic adversarial-search opponent. It complements MCTS: instead of
estimating moves from sampled games, it enumerates a bounded game tree and assumes that both
players choose their strongest reply. Alpha-beta pruning skips branches that cannot change the
root decision, while a transposition table reuses positions reached through different move
orders.

Paper soccer cannot use a conventional "one edge equals one turn" depth rule. A rebound leaves
the same player on move, so the search keeps the same maximizing or minimizing perspective and
does not consume a turn-depth unit until possession actually changes. Within the configured
physical-ply horizon, a depth-one search therefore follows same-player rebound combinations
through a goal, trap, or handoff.
Terminal scores are exact and Player-1-oriented: a Player 1 win is near `+1,000,000`, a Player 2
win is near `-1,000,000`, and the physical move count rewards faster wins and delayed losses.

Non-terminal leaves use a transparent hand-written evaluator. Positive values favor Player 1;
negative values favor Player 2. Its initial feature weights are:

| Feature | Weight | Meaning |
| --- | ---: | --- |
| Direct goal available | 50,000 | Strongly prefer a goal on the current turn |
| Unused-edge goal-distance difference | 120 | Compare each side's shortest remaining route to goal |
| Vertical ball progress | 80 | Reward moving toward Player 1's goal and penalize the reverse |
| Possession-preserving choices | 35 | Value available rebound or boundary continuations |
| Legal mobility | 20 | Value options for the player currently in possession |
| Forward choices | 10 | Reward legal moves aimed toward that player's goal |
| Center alignment | 6 | Prefer useful central access for the player in possession |
| Tempo | 15 | Give a small value to retaining the move |

The goal-distance feature runs breadth-first search over currently unused edges. The combined
heuristic is clamped to `[-100,000, 100,000]`, leaving a wide, unambiguous band between heuristic
positions and proven wins. These weights are a readable starting point, not trained constants;
the arena is the intended place to measure and tune them.

Search uses iterative deepening, so every completed shallower result remains usable if the next
depth reaches the configured node or optional wall-clock budget. A zero `max_time_ms`, including
the default, disables the wall-clock cutoff and preserves deterministic fixed-work searches.
The default configuration searches up to six possession
handoffs, visits at most 100,000 nodes across the complete decision, stores 65,536 compact
transposition entries, and applies a soft horizon after ten physical edges. At that horizon the
search evaluates positions with multiple choices, but continues a forced single-move line as far
as the absolute 512-edge recursion guard. This bounds unusually large same-turn rebound trees,
still proves long forced combinations, and covers the nine-edge forcing sequence found in the
human replay regression. If even depth one cannot finish, the bot returns a deterministic legal
move ordered by immediate outcomes, rebounds, progress, center access, and child mobility.

The compact search position maintains a two-word incremental key over used edges, visited
vertices, ball location, side to move, and game status. A table entry stores its searched depth,
score bound, and best move. The same make/unmake position is shared with MCTS's tactical search,
while the public rules engine remains authoritative.

Native usage:

```cpp
papersoccer::AlphaBetaConfig config{
    .max_turn_depth = 6,
    .max_nodes = 100'000,
    .transposition_table_entries = 65'536,
    .max_search_plies = 10,
    .max_time_ms = 0,
};
papersoccer::AlphaBetaBot bot(config);
papersoccer::Move move = bot.choose_move(state);
const papersoccer::AlphaBetaSearchStats &stats = bot.last_search_stats();
```

`AlphaBetaSearchStats` reports attempted and completed depth, visited and evaluated nodes,
terminal nodes, alpha-beta cutoffs, transposition probes/hits/cutoffs/stores, physical-horizon
cutoffs, maximum physical ply, the root score, whether the budget interrupted the final
iteration, a principal variation, and the searched root alternatives. Each root alternative is
tagged `Exact`, `Lower`, or `Upper`, because a branch eliminated by the narrowed root window may
produce a bound rather than an exact value. The selected root score is exact for the last
completed bounded depth.

### Initial smoke evaluation — July 16, 2026

With the defaults above, an eight-position shared-state check and a ten-pair color-swapped match
against Tactical `MctsBot` at 2,000 iterations produced:

| Check | Alpha-beta | MCTS reference |
| --- | ---: | ---: |
| Shared-position median decision time | 3.457 ms | 56.075 ms |
| Shared-position p95 decision time | 41.954 ms | 63.333 ms |
| Match median decision time | 2.460 ms | 41.978 ms |
| Match p95 decision time | 7.043 ms | 77.758 ms |
| Ten-pair record | 13 wins / 7 losses | 7 wins / 13 losses |

There were no illegal moves, truncations, or alpha-beta node-budget exhaustions. The paired 95%
bootstrap interval was wide at 40%-90%, so the 65% score is promising smoke evidence, not a
strength conclusion. A larger fixed-seed tournament and evaluator ablations are still required
before choosing a default opponent or tuning the weights around this result. Timings are
machine-specific.

## Monte Carlo Tree Search Bot

Monte Carlo Tree Search (MCTS) estimates which move is strongest by repeatedly playing out
possible futures. It is a good fit for paper soccer because positions can offer many legal
continuations, games eventually terminate under the authoritative rules, and random samples
can guide the search without trying to enumerate the entire game tree in advance.

One search iteration has four parts:

1. **Selection:** follow promising moves already in the tree. MCTS uses UCT, which combines a
   move's average result with a bonus for moves that have received less attention.
2. **Expansion:** add one previously unexplored legal move to the tree.
3. **Frontier evaluation:** optionally try the bounded tactical proof search, then use the
   configured rollout policy until the rules engine reports a win if no winner was proven.
4. **Backpropagation:** add that result to every tree node visited during the iteration.

The UCT comparison is:

```text
mean value + exploration * sqrt(log(parent visits) / child visits)
```

The mean value is exploitation: prefer a move whose sampled games have gone well. The second
term is exploration: occasionally investigate a less-visited move in case it is better than
the current favorite. The default exploration value is approximately `sqrt(2)`
(`1.4142135623730951`).

A Player 1 win is stored as `+1` and a Player 2 win as `-1`. Every node keeps this same
Player-1-oriented value. Player 1 decision nodes prefer larger means; Player 2 decision nodes
prefer smaller means. This is especially important after a rebound, because landing on a
boundary or previously visited point keeps the same player on move. The search follows
`GameState::to_move` rather than assuming that players alternate after every move.

Before sampling, both `MctsBot` and `MctsSearch` take an immediate legal winning move when one
is available; a solved root therefore reports zero sampled iterations. Otherwise, the final
root move is chosen primarily by visit count, which is more stable than the temporary UCT score
used while searching. Visit ties are resolved by mean result from the root player's perspective
and then by the engine's existing legal-move order.

The default `Tactical` rollout policy takes immediate wins, follows forced moves without drawing
an unnecessary random number, avoids terminal self-traps, and avoids handing the opponent an
immediate win when a safe alternative exists. It also enables MCTS-Solver propagation: terminal
winners are proven back through fully explored branches, proven losing children are avoided, and
a search stops early when the root winner is proven. `Uniform` retains the original uniformly
random rollout and selection behavior as a reproducible comparison policy.

The leaf policy is a separate setting. `MctsLeafPolicy::RolloutOnly` remains the engine and
website default, preserving the current Tactical rollout-only bot exactly. The arena keeps that
bot as its frozen reference. `MctsLeafPolicy::TacticalQuiescence` is an experimental opt-in that
runs a bounded adversarial proof probe at each non-terminal MCTS frontier before falling back to
the unchanged Tactical rollout. Tactical quiescence requires the `Tactical` rollout policy. The
probe is deterministic, consumes no rollout random numbers, and its work does not replace any of
the configured new MCTS iterations.

The probe extends only tactically noisy positions. Noise includes immediate goals and terminal
traps, a single legal move, a same-player rebound, a move that leaves one forced legal reply, and
defensive or escape contexts identified by one-ply reply lookahead. In that lookahead, an
opponent direct tactic is an immediate terminal goal or trap, an already forced single move, or a
move that creates one forced reply; an otherwise unconstrained opponent rebound alone does not
trigger the defensive extension. Once a position is noisy, every legal move is searched in the
authoritative rules order so that a quiet escape is never omitted. Rebounds use the compact
position's actual `to_move` value at every level; search depth alone never changes player
perspective.

Proof results are absolute winners: Player 1, Player 2, or Unknown. At a position belonging to a
player, one child proven for that player is enough to prove a win. A loss is proven only when
every legal move is proven for the opponent. A quiet position, a depth or node cutoff, or any
unresolved legal alternative therefore returns Unknown and can never create a false proof. The
compact position is made and unmade exactly around every probe. A proven frontier supplies the
exact terminal reward and participates in the existing solver propagation; an Unknown frontier
continues with the existing Tactical rollout.

Search simulations use a compact internal board: vertices and edges are indexed, used edges and
visited vertices are bitsets, adjacency is precomputed, and moves are made and unmade in place.
The public `GameState` and rules functions remain the authoritative API for games and replays.

### Configuration and reproducibility

`MctsConfig` defaults to a seed of `RandomBot::default_seed()`, `2,000` new iterations per move,
the exploration value above, tactical rollouts, tree reuse, a 65,536-node tree bound, and the
`RolloutOnly` leaf policy. The experimental quiescence limits default to depth `8` and `256`
searched tactical nodes per probe. Both limits must be positive; depth is capped at `64` and the
node budget at `1,000,000`. The iteration count must be positive, exploration must be finite and
non-negative, and the MCTS tree bound must be at least two.

The search uses a fixed iteration budget rather than a time limit. A complete match is
reproducible from the same starting state, bot configurations, seeds, and played moves on the
same implementation. Fixed work also makes tests and comparisons independent of machine speed.
`MctsBot` re-roots its retained tree through moves that were explored previously, compacts away
unreachable branches, and adds the configured amount of new work. It rebuilds deterministically
when the supplied state is unrelated or the played branch was not expanded. At the node bound,
iterations continue as rollouts without allocating more tree nodes.

Standalone `MctsSearch` objects remain position-based and incremental. Incremental calls preserve
the same random sequence, so one call of 2,000 iterations is equivalent to, for example, twenty
calls of 100 iterations. The legacy seed/exploration constructor selects `Uniform`; pass a full
`MctsConfig` to use tactical rollouts or experimental quiescence.

The native API supports both a one-shot bot and an incremental search:

```cpp
papersoccer::MctsConfig config{
    .seed = 12345,
    .iterations = 2000,
    .rollout_policy = papersoccer::MctsRolloutPolicy::Tactical,
    .reuse_tree = true,
    .max_nodes = 65536,
    .leaf_policy = papersoccer::MctsLeafPolicy::TacticalQuiescence,
    .quiescence_max_depth = 8,
    .quiescence_max_nodes = 256,
};
papersoccer::MctsBot bot(config);
papersoccer::Move move = bot.choose_move(state);
papersoccer::SearchStats stats = bot.last_search_stats();

papersoccer::MctsSearch search(state, 12345);
search.run_iterations(100);
search.run_iterations(100);
papersoccer::Move incremental_move = search.best_move();
```

`SearchStats` reports completed iterations, allocated nodes, rollout plies, total and reused root
visits, maximum tree depth, proven nodes and winner, rebuild count, node-bound saturation, and the
root value estimate. `simulated_plies` counts rollout moves only, while `root_value` is always
from Player 1's perspective. Tactical work is reported separately as `tactical_probes`,
`tactical_nodes`, `tactical_solved_positions`, `tactical_depth_cutoffs`,
`tactical_node_cutoffs`, and `max_tactical_depth`. These are deterministic counters; timing is
deliberately measured by the arena rather than stored in `SearchStats`. Arena summaries also
derive `tactical_solution_rate` from solved positions divided by probes.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

You can also run the test binary directly:

```bash
./build/papersoccer_tests
```

When Node.js 18 or newer is available, CTest also instantiates the compiled WebAssembly
module and tests the browser-to-C++ command boundary.
It can be run directly with:

```bash
node --test tests/web/web_wasm_test.mjs
```

## Measure Bot Performance

The native arena compares fresh bot instances on deterministic seed pairs and swaps their colors
for the second game in every pair. Its defaults compare the experimental Tactical +
TacticalQuiescence candidate with the frozen Tactical + RolloutOnly reference. Both reuse their
trees and use the same 2,000-iteration budget, exploration value, and MCTS node bound:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
./build/release/papersoccer_arena positions --positions 16 > positions.json
./build/release/papersoccer_arena matches --pairs 20 > quiescence-equal-iterations-20-pairs.json
```

Both modes emit `papersoccer.arena.v1` JSON. Match reports contain per-game configurations,
lossless string seeds, outcomes, plies, per-decision timings and search counters, color splits,
throughput, and a deterministic pair-bootstrap 95% confidence interval. Position reports measure
both bots on the same generated non-terminal states. Use `--help` for policy, reuse, iteration,
node-bound, exploration, board, and sampling options. Quiescence is controlled independently for
the two entrants with:

- `--candidate-leaf-policy` and `--reference-leaf-policy`, accepting `rollout-only` or
  `tactical-quiescence`.
- `--candidate-quiescence-max-depth` and `--reference-quiescence-max-depth`.
- `--candidate-quiescence-max-nodes` and `--reference-quiescence-max-nodes`.

The same arena can compare alpha-beta with the frozen MCTS reference:

```bash
./build/release/papersoccer_arena positions \
  --positions 32 \
  --candidate-kind alpha-beta \
  --candidate-alpha-beta-depth 6 \
  --candidate-alpha-beta-max-nodes 100000 \
  --reference-kind mcts \
  --reference-iterations 2000 \
  > alpha-beta-positions.json

./build/release/papersoccer_arena matches \
  --pairs 200 \
  --candidate-kind alpha-beta \
  --candidate-alpha-beta-depth 6 \
  --candidate-alpha-beta-max-nodes 100000 \
  --reference-kind mcts \
  --reference-iterations 2000 \
  > alpha-beta-vs-mcts.json
```

Alpha-beta reports keep their diagnostics separate from MCTS: completed/attempted depth, node
and leaf counts, terminal nodes, pruning and transposition counters, physical-horizon cutoffs,
root scores and bounds, principal variations, and budget exhaustion. Arena timing summaries
also include median nodes per second for either search family.

An explicit equal-iteration preliminary comparison can be reproduced with:

```bash
./build/release/papersoccer_arena matches \
  --pairs 20 \
  --seed 828927513140 \
  --candidate-policy tactical \
  --candidate-leaf-policy tactical-quiescence \
  --candidate-quiescence-max-depth 8 \
  --candidate-quiescence-max-nodes 256 \
  --candidate-reuse true \
  --candidate-iterations 2000 \
  --reference-policy tactical \
  --reference-leaf-policy rollout-only \
  --reference-reuse true \
  --reference-iterations 2000 \
  > quiescence-equal-iterations-20-pairs.json
```

If that result is positive and operationally healthy, change `--pairs 20` to `--pairs 200` for
the full 400-game comparison. Equal iteration counts do not imply equal latency: every candidate
iteration can do additional bounded tactical work. First compare the entrants' `median_ns` and
`p95_ns` on shared positions, then repeat the match with a measured candidate iteration budget
whose median is reasonably close to the reference (`1,250` on the evaluation machine):

```bash
./build/release/papersoccer_arena positions \
  --positions 32 \
  --candidate-iterations 2000 \
  --reference-iterations 2000 \
  > quiescence-latency-tuning.json

./build/release/papersoccer_arena matches \
  --pairs 20 \
  --candidate-iterations 1250 \
  --reference-iterations 2000 \
  > quiescence-latency-matched-20-pairs.json
```

Larger tactical depth and node limits can prove longer combinations, but add work to every noisy
frontier. A successful proof may compensate by skipping a rollout and solving retained tree
nodes; an Unknown probe adds overhead before the normal rollout. Report both equal-iteration and
approximately equal-latency results rather than treating either one alone as decisive.

### Experimental quiescence evaluation — July 16, 2026

The staged evaluation keeps TacticalQuiescence experimental. Timings below are
machine-specific; deterministic records, counters, and confidence intervals can be reproduced
with the commands above.

| Evaluation | Result |
| --- | --- |
| Correctness gate | Passed: Release, ASan/UBSan, browser/Wasm, compact-rule parity, restoration, cutoff, proof-soundness, reproducibility, and native/Wasm arena smoke tests; zero illegal moves or unexpected truncations |
| 20-pair equal-iteration record | 19 wins, 21 losses at 2,000 vs 2,000 iterations; 47.5% score; paired 95% interval 37.5%-57.5% |
| Equal-iteration fixed work | Candidate 4,093,635 completed iterations, 3,947,878 probes, 65,477 solved positions (1.659%); reference 4,342,896 completed iterations and zero probes |
| Equal-iteration operations | Maximum cumulative rebuild count 7 candidate / 5 reference; zero saturated searches, illegal moves, or truncations |
| Native match median/p95 | Candidate 36.757/66.853 ms; reference 22.683/45.968 ms |
| Native shared-position median/p95 | Candidate 49.862/61.204 ms; reference 33.049/36.004 ms |
| Wasm shared-position median/p95 | Candidate 56.088/71.932 ms; reference 37.384/41.799 ms; deterministic counters matched native exactly |
| 20-pair latency-matched record | 11 wins, 29 losses at 1,250 vs 2,000 iterations; 27.5% score; paired 95% interval 12.5%-42.5% |
| Latency-matched work and timing | Candidate 2,433,972 iterations, 2,341,396 probes, 42,004 solved (1.794%), median/p95 23.459/43.570 ms; reference 4,004,291 iterations, median/p95 23.608/47.535 ms; rebuild maximum 10/6 and zero saturation |
| Full 200-pair/400-game comparison | Not run: the equal-iteration preliminary result did not favor quiescence, so the strength gate failed |

The probe solved real frontier positions and stayed operationally healthy, but its low solution
rate did not offset its extra work. Reducing iterations to match latency made the candidate much
weaker. These results are not evidence for promotion; `RolloutOnly` remains the engine and
website default.

The tactical regression fixture derived from arena pair 103 preserves a real escape: before the
critical move, Player 2 at `(6,2)` has five legal defenses, and `(5,3)` is the only one the bounded
forcing search does not refute. The frozen Tactical bot chose `(7,3)` and allowed a same-player
rebound attack to reach goal. The probe must keep the root Unknown because of the surviving
escape, while proving representative tempting alternatives losing. This fixture came from the
arena baseline; no `papersoccer.human-match.v1` export was supplied or committed for this work.

TacticalQuiescence remains experimental unless its correctness fixtures pass, supplied human
exploit fixtures improve without regression, the full paired arena result has a 95% lower bound
above 50%, native and Wasm latency remain acceptable, and truncation, rebuild, and saturation
rates remain healthy. Even if those gates pass, the website default should remain RolloutOnly
unless human-game evidence is strong enough to justify promotion.

For historical context, the current Tactical RolloutOnly baseline was promoted on July 14, 2026
after a 389-11 result over 200 color-swapped seed pairs against the older Uniform, non-reusing
reference, with no illegal moves or truncations and a 95.5% to 98.75% paired bootstrap interval.
Those measurements established the frozen reference; they are not evidence for experimental
quiescence.

The full promotion tournament is intentionally not part of CTest. CTest runs low-budget support
and CLI smoke tests, parses both report modes as JSON, and verifies paired accounting without
making normal tests depend on machine speed.

For the equivalent Emscripten `positions` report, use the arena built with the pinned toolchain:

```bash
emcmake cmake -S . -B build/wasm-arena -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm-arena --target papersoccer_arena
node build/wasm-arena/papersoccer_arena.js positions --positions 32 > wasm-positions.json
node build/wasm-arena/papersoccer_arena.js matches --pairs 20 \
  > wasm-quiescence-equal-iterations-20-pairs.json
```

The checked-in browser engine's default-path timing can also be probed with:

```bash
node benchmarks/wasm_mcts.mjs 2000 9
```

The historical Tactical RolloutOnly Wasm initial-position median was 57.700 ms on the gate
machine, below its previous 174.120 ms baseline. These figures do not measure quiescence. Use the
Wasm arena report when policy configuration and completed MCTS and tactical counters need to
match the native `positions` benchmark exactly.

## Run CLI

```bash
./build/papersoccer_cli
```

CLI commands:

- `<index>`: play the move with that index
- `b`: print the current ASCII board
- `a`: toggle automatic board printing
- `h`: help
- `q`: quit

The CLI auto-prints the board each turn by default.
On startup, the CLI lets you choose:

1. Human vs Human
2. Human vs seeded `RandomBot` (choose the human side)
3. Seeded `RandomBot` vs seeded `RandomBot`
4. Human vs `MctsBot` (choose the human side)
5. Seeded `RandomBot` vs `MctsBot` (choose the MCTS side)
6. `MctsBot` vs `MctsBot`
7. Human vs `AlphaBetaBot` (choose the human side)
8. Seeded `RandomBot` vs `AlphaBetaBot` (choose the alpha-beta side)
9. `MctsBot` vs `AlphaBetaBot` (choose the alpha-beta side)
10. `AlphaBetaBot` vs `AlphaBetaBot`

Modes involving two different controller types ask which side uses which controller. RandomBot
modes include a RandomBot base-seed prompt. When MCTS is selected, the CLI also asks for an MCTS
base seed and a single iteration budget applied to every MCTS move; pressing Enter accepts the
default of `2,000` iterations. Alpha-beta modes ask for a possession-handoff depth and node
budget, defaulting to `6` and `100,000` respectively; alpha-beta itself uses no random seed.
Player 1 uses the relevant base seed and Player 2 uses `base_seed + 1`. After every MCTS move,
the CLI prints new iterations, tree size, rollout plies, total and reused root visits, maximum
depth, tactical probe work, proof/rebuild information, saturation, and the Player-1-oriented root
value estimate. After every alpha-beta move, it prints completed/attempted depth, nodes, leaf and
terminal counts, cutoffs, transposition hits, physical-horizon cutoffs, maximum physical ply,
root score, principal-variation length, and whether the node budget was reached.

Rule examples above use the engine's `Point{x, y}` order. User-facing coordinates in the
CLI and renderer are shown as `(row, column)`, which corresponds to `(y, x)`. For example,
the initial engine point `(4,6)` is displayed as `(6,4)`. Every coordinate is zero-based.

`RulesConfig.width` and `RulesConfig.height` describe the field span, not the largest board
coordinate. The default height remains `10`, while the field boundary rows are `1` and `11`
and the goal rows are `0` and `12`.

## Run Web Game and Replay Viewer

Open the static app directly:

```bash
open web/index.html
```

The app starts in **Play vs bot** mode:

1. Choose `RandomBot`, `MctsBot`, or `AlphaBetaBot` as the opponent.
2. Choose Player 1 to move first and attack the top goal, or Player 2 to let the bot
   move first and attack the bottom goal.
3. Enter the opponent seed. MCTS games also expose the fixed number of new simulations per
   bot move; alpha-beta games expose possession-handoff depth. Alpha-beta is deterministic and
   ignores the seed when choosing moves, although the shared replay metadata still records it.
   Reusing the same settings and human moves reproduces the bot's choices.
4. Select **Start**, then click any highlighted destination on the board. Destinations
   marked with `↻` grant another move.

Once a game starts, its opponent, side, seed, and search setting are locked. The active
configuration remains visible beside the latest MCTS search counters when MCTS is selected.
Select **Change settings** to stop the current game without discarding it, or **Export game** to
save a `papersoccer.human-match.v1` document containing the standard replay and every recorded
bot search. Exported human matches can be opened through the normal replay loader.

Select **Watch replay** to inspect the built-in game or generate a new bot-vs-bot match.
Player 1 and Player 2 each have an independent bot and seed, plus an MCTS simulation setting or
alpha-beta handoff-depth setting when applicable.
Generation advances one move at a time so the controls can update between searches, stops after
512 moves if the game has not finished, reports progress during longer matches, and opens the
result directly in the replay viewer.
Use **Open replay** to load another replay file. The app remains fully static and works without
a server.

The browser does not contain a second implementation of the game. C++ owns the live
state, legal moves, rebounds, win detection, bot RNG, and replay history. The small
JavaScript adapter sends versioned move commands to the compiled C++ WebAssembly module;
the remaining JavaScript handles canvas drawing, controls, accessibility, and animation.

### Rebuild the C++ WebAssembly module

The generated single-file module is checked in, so playing the web game does not require
a compiler or local server. It is currently pinned to Emscripten 6.0.2 for byte-reproducible
builds. After changing C++ rules, either bot, or the web-session layer used by the browser,
rebuild it with:

```bash
emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm --target update_papersoccer_web
```

This regenerates `web/papersoccer-wasm.js`. Its WebAssembly bytes are embedded so opening
`web/index.html` directly still works. The generated file is marked as generated for
repository language statistics.

To verify that the checked-in module exactly matches the C++ sources without updating it:

```bash
cmake --build build/wasm --target check_papersoccer_web
```

The versioned command/snapshot boundary keeps both browser bots in C++; JavaScript only schedules
their turns and renders the returned session snapshots. Memory growth remains disabled because
growable Wasm buffers are not compatible with every direct-file browser.

Generate a new bot-vs-bot replay as JSON:

```bash
./build/papersoccer_replay_export 12345 512 > replay.json
```

Arguments are optional:

- `base-seed`: defaults to `RandomBot::default_seed()`
- `max-plies`: defaults to `512`

The replay exporter remains a `RandomBot` vs `RandomBot` tool; adding MCTS to the native CLI
does not change its arguments, player metadata, or replay schema.

Use the viewer's `Open replay` button to load the generated file.

The exporter writes the `papersoccer.replay.v2` schema, whose points use the zero-based board
coordinates described above. The viewer also accepts legacy `papersoccer.replay.v1` files;
it translates every v1 point one row down (`y + 1`) while loading, so old replays keep the
same geometry.

## Project Layout

Public C++ includes remain stable under `include/papersoccer`; implementation and tests are
grouped by responsibility:

```text
.
├── include/papersoccer/        Public C++ API
│   ├── arena.hpp               Arena configuration and report entrypoints
│   ├── bot.hpp                 Bot interface and search configurations
│   ├── debug.hpp               Text rendering API
│   ├── geometry.hpp            Board geometry helpers
│   ├── match.hpp               Stateful match and move history
│   ├── rules.hpp               Legal moves and state transitions
│   ├── types.hpp               Shared game types and hashing
│   └── web_game.hpp            Versioned browser-session API
├── src/
│   ├── core/                   Rules, geometry, matches, and debug rendering
│   ├── bots/                   Random, MCTS, alpha-beta, and tactical search
│   │   └── mcts_internal.hpp   Private compact search position/topology
│   ├── arena/
│   │   ├── main.cpp            Native/Wasm command-line entrypoint
│   │   ├── runner.cpp          Match execution and statistical summaries
│   │   ├── report.cpp          Stable JSON report serialization
│   │   └── internal.hpp        Private data shared by runner and reporter
│   ├── cli/main.cpp            Interactive terminal game
│   ├── replay/main.cpp         Seeded replay exporter
│   └── web/
│       ├── web_game.cpp        C++ browser sessions and snapshots
│       └── wasm_bridge.cpp     Minimal Emscripten C ABI
├── web/
│   ├── index.html              Static application shell
│   ├── game-engine.js          Client for the compiled C++ session API
│   ├── app-support.js          Timers, configuration, and replay helpers
│   ├── app.js                  Application/session and control orchestration
│   ├── board-view.js           Canvas rendering and board interaction
│   ├── styles.css              Game and replay presentation
│   └── papersoccer-wasm.js     Generated single-file WebAssembly module
├── tests/
│   ├── core/                   Rules and match behavior
│   ├── bots/                   Random, MCTS, and alpha-beta behavior
│   ├── arena/                  Arena API smoke and CLI integration tests
│   ├── web/                    C++ session, Wasm, and web-support tests
│   ├── replay_export_test.mjs  Replay exporter integration test
│   └── test_main.cpp           Native test entrypoint
├── benchmarks/wasm_mcts.mjs    WebAssembly MCTS timing probe
└── CMakeLists.txt              Build targets and complete test registration
```

The frontends (`cli`, `replay`, `arena`, and `web`) depend on the same public API and core
library. Browser rules and bots remain authoritative in C++; the JavaScript layer only manages
sessions, presentation, and input.
