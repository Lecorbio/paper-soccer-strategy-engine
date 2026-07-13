# Paper Soccer Strategy Engine (Base v0)

This repository contains a deterministic C++20 baseline for paper soccer with:

- A pure rules engine (`papersoccer_core`)
- A terminal CLI for human and bot play (`papersoccer_cli`)
- A seeded bot layer for automated play experiments
- A JSON replay exporter for bot self-play (`papersoccer_replay_export`)
- A static browser game powered by the same C++ engine through WebAssembly
- Dependency-free tests integrated with CTest (`papersoccer_tests`)

The current version intentionally focuses on core game correctness so minimax/MCTS bots can be added on top without refactoring game state logic.

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
  - Any boundary point of the field rectangle
- If the player to move has zero legal moves at turn start, that player loses

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
On startup, the CLI now lets you choose:

- human vs human
- human vs seeded `RandomBot`
- seeded `RandomBot` vs seeded `RandomBot`

If a bot is involved, the CLI prompts for a base seed. Player 1 uses the base seed and
Player 2 uses `base_seed + 1`, which makes bot games reproducible.

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

1. Choose Player 1 to move first and attack the top goal, or Player 2 to let the bot
   move first and attack the bottom goal.
2. Enter a bot seed. Reusing the same side, seed, and human moves reproduces the bot's
   choices.
3. Select **Start**, then click any highlighted destination on the board. Destinations
   marked with `↻` grant another move.

Select **Watch replay** to inspect the built-in RandomBot game, or use **Open replay**
to load another replay file. The app remains fully static and works without a server.

The browser does not contain a second implementation of the game. C++ owns the live
state, legal moves, rebounds, win detection, bot RNG, and replay history. The small
JavaScript adapter sends versioned move commands to the compiled C++ WebAssembly module;
the remaining JavaScript handles canvas drawing, controls, accessibility, and animation.

### Rebuild the C++ WebAssembly module

The generated single-file module is checked in, so playing the web game does not require
a compiler or local server. It is currently pinned to Emscripten 6.0.2 for byte-reproducible
builds. After changing C++ rules, bots, or the web-session layer, rebuild it with:

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

The versioned command/snapshot boundary is also the scaling seam. New C++ bots can use the
same browser integration without porting their logic to JavaScript. A computationally
expensive minimax or MCTS bot should eventually run the same module in a Web Worker so its
search does not block canvas rendering; the UI-facing protocol does not need to change.
Those bots may also need a deliberately larger fixed Emscripten memory budget. Memory growth
is disabled because growable Wasm buffers are not compatible with every direct-file browser.

Generate a new bot-vs-bot replay as JSON:

```bash
./build/papersoccer_replay_export 12345 512 > replay.json
```

Arguments are optional:

- `base-seed`: defaults to `RandomBot::default_seed()`
- `max-plies`: defaults to `512`

Use the viewer's `Open replay` button to load the generated file.

The exporter writes the `papersoccer.replay.v2` schema, whose points use the zero-based board
coordinates described above. The viewer also accepts legacy `papersoccer.replay.v1` files;
it translates every v1 point one row down (`y + 1`) while loading, so old replays keep the
same geometry.

## Project Layout

- `include/papersoccer/types.hpp` - core types and hashing
- `include/papersoccer/geometry.hpp` - geometry and adjacency helpers
- `include/papersoccer/bot.hpp` - bot interface and seeded `RandomBot`
- `include/papersoccer/match.hpp` - authoritative state plus played-move history
- `include/papersoccer/rules.hpp` - game rules API
- `include/papersoccer/web_game.hpp` - versioned browser-session command API
- `src/bot.cpp` - bot implementation
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
- `tests/bot_test.cpp` - bot determinism and legality checks
- `tests/test_main.cpp` - test entrypoint
- `tests/rules_test.cpp` - rule correctness scenarios
