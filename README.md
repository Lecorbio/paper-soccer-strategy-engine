# Paper Soccer Strategy Engine

This repository contains a deterministic C++20 baseline for paper soccer with:

- A pure rules engine (`papersoccer_core`)
- A terminal CLI for human and bot play (`papersoccer_cli`)
- Seeded `RandomBot` and Monte Carlo Tree Search (`MctsBot`) opponents
- A native arena for paired strength matches and position throughput measurements
- A JSON replay exporter for bot self-play (`papersoccer_replay_export`)
- A static browser game powered by the same C++ engine through WebAssembly
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

## Monte Carlo Tree Search Bot

Monte Carlo Tree Search (MCTS) estimates which move is strongest by repeatedly playing out
possible futures. It is a good fit for paper soccer because positions can offer many legal
continuations, games eventually terminate under the authoritative rules, and random samples
can guide the search without trying to enumerate the entire game tree in advance.

One search iteration has four parts:

1. **Selection:** follow promising moves already in the tree. MCTS uses UCT, which combines a
   move's average result with a bonus for moves that have received less attention.
2. **Expansion:** add one previously unexplored legal move to the tree.
3. **Simulation:** use the configured rollout policy until the rules engine reports a win.
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

Search simulations use a compact internal board: vertices and edges are indexed, used edges and
visited vertices are bitsets, adjacency is precomputed, and moves are made and unmade in place.
The public `GameState` and rules functions remain the authoritative API for games and replays.

### Configuration and reproducibility

`MctsConfig` defaults to a seed of `RandomBot::default_seed()`, `2,000` new iterations per move,
the exploration value above, tactical rollouts, tree reuse, and a 65,536-node tree bound. The
iteration count must be positive, exploration must be finite and non-negative, and the node
bound must be at least two.

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
`MctsConfig` to use tactical rollouts.

The native API supports both a one-shot bot and an incremental search:

```cpp
papersoccer::MctsConfig config{
    .seed = 12345,
    .iterations = 2000,
    .rollout_policy = papersoccer::MctsRolloutPolicy::Tactical,
    .reuse_tree = true,
    .max_nodes = 65536,
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
from Player 1's perspective. Timing is deliberately measured by the arena rather than stored in
these deterministic statistics.

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
node --test tests/web_wasm_test.mjs
```

## Measure Bot Performance

The native arena compares fresh bot instances on deterministic seed pairs and swaps their colors
for the second game in every pair. Its defaults compare tactical, tree-reusing MCTS against the
uniform, non-reusing reference at the same 2,000-iteration budget:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
./build/release/papersoccer_arena positions --positions 16 > positions.json
./build/release/papersoccer_arena matches --pairs 200 > matches.json
```

Both modes emit `papersoccer.arena.v1` JSON. Match reports contain per-game configurations,
lossless string seeds, outcomes, plies, per-decision timings and search counters, color splits,
throughput, and a deterministic pair-bootstrap 95% confidence interval. Position reports measure
both bots on the same generated non-terminal states. Use `--help` for policy, reuse, iteration,
node-bound, exploration, board, and sampling options.

The tactical default passed its promotion gate on July 14, 2026: 389 wins and 11 losses over 200
color-swapped seed pairs, with no illegal moves or truncations and a deterministic paired 95%
bootstrap interval of 95.5% to 98.75%. On the same machine, median tactical decision latency was
30.066 ms across the match and 50.073 ms on the native initial-position probe, below the previous
171.558 ms default baseline. The optimized uniform reference took 3.03 ms natively and 3.353 ms
in Wasm on the initial position, over 50 times the prior native and Wasm throughput. These
machine-specific results are evidence for the checked-in default, not portable timing thresholds.

The full promotion tournament is intentionally not part of CTest. CTest runs low-budget support
and CLI smoke tests, parses both report modes as JSON, and verifies paired accounting without
making normal tests depend on machine speed.

For the equivalent Emscripten `positions` report, use the arena built with the pinned toolchain:

```bash
emcmake cmake -S . -B build/wasm-arena -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm-arena --target papersoccer_arena
node build/wasm-arena/papersoccer_arena.js positions --positions 16 > wasm-positions.json
```

The checked-in browser engine's default-path timing can also be probed with:

```bash
node benchmarks/wasm_mcts.mjs 2000 9
```

The promoted Wasm initial-position median was 57.700 ms on the gate machine, below its previous
174.120 ms baseline. Use the Wasm arena report when policy configuration and completed MCTS
counters need to match the native `positions` benchmark exactly.

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

The original three modes and their prompts remain unchanged. Modes involving two different
controller types ask which side uses which controller. RandomBot modes retain the existing
RandomBot base-seed prompt. When MCTS is selected, the CLI also asks for an MCTS base seed and
a single iteration budget applied to every MCTS move; pressing Enter accepts the default of
`2,000` iterations.
Player 1 uses the relevant base seed and Player 2 uses `base_seed + 1`. After every MCTS move,
the CLI prints new iterations, tree size, rollout plies, total and reused root visits, maximum
depth, proof/rebuild information, saturation, and the Player-1-oriented root value estimate.

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

1. Choose `RandomBot` or `MctsBot` as the opponent.
2. Choose Player 1 to move first and attack the top goal, or Player 2 to let the bot
   move first and attack the bottom goal.
3. Enter the opponent seed. MCTS games also expose the fixed iteration budget used for
   each move. Reusing the same settings and human moves reproduces the bot's choices.
4. Select **Start**, then click any highlighted destination on the board. Destinations
   marked with `↻` grant another move.

Select **Watch replay** to inspect the built-in game or generate a new bot-vs-bot match.
Player 1 and Player 2 each have an independent bot, seed, and MCTS iteration setting.
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

- `include/papersoccer/types.hpp` - core types and hashing
- `include/papersoccer/geometry.hpp` - geometry and adjacency helpers
- `include/papersoccer/bot.hpp` - bot interface, `RandomBot`, and native MCTS API
- `include/papersoccer/arena.hpp` - paired-match and position-measurement API
- `include/papersoccer/match.hpp` - authoritative state plus played-move history
- `include/papersoccer/rules.hpp` - game rules API
- `include/papersoccer/web_game.hpp` - versioned browser-session command API
- `src/bot.cpp` - shared bot naming and factory implementation
- `src/mcts.cpp` - Monte Carlo Tree Search implementation
- `src/mcts_internal.hpp` - compact search-only topology and mutable position
- `src/arena.cpp` - arena execution, statistics, and JSON reports
- `src/arena/main.cpp` - native/Wasm arena command-line entrypoint
- `src/random_bot.cpp` - seeded `RandomBot` implementation
- `src/geometry.cpp` - geometry implementation
- `src/match.cpp` - match orchestration and replay metadata
- `src/rules.cpp` - state transitions and legal move logic
- `src/web_game.cpp` - C++ browser-session snapshots and command validation
- `src/web/wasm_bridge.cpp` - minimal C ABI compiled by Emscripten
- `src/cli/main.cpp` - terminal game loop
- `src/replay/main.cpp` - bot self-play JSON replay exporter
- `web/index.html` - static browser game and replay shell
- `web/papersoccer-wasm.js` - generated single-file C++ WebAssembly module
- `web/game-engine.js` - thin JavaScript client for the C++ module
- `web/app.js` - browser play, canvas rendering, and replay controls
- `web/styles.css` - game and replay styling
- `tests/match_test.cpp` - match history and state ownership checks
- `tests/replay_export_test.mjs` - replay seed precision check
- `tests/web_game_session_test.cpp` - C++ web-session and stale-command checks
- `tests/web_wasm_test.mjs` - real WebAssembly/browser-client integration checks
- `tests/bot_test.cpp` - RandomBot determinism and legality checks
- `tests/mcts_test.cpp` - MCTS determinism, legality, and strategy checks
- `tests/arena_smoke_test.cpp` - paired arena and measurement report smoke tests
- `tests/arena_cli_test.mjs` - arena CLI option-routing and JSON integration tests
- `benchmarks/wasm_mcts.mjs` - initial-position WebAssembly timing probe
- `tests/test_main.cpp` - test entrypoint
- `tests/rules_test.cpp` - rule correctness scenarios
