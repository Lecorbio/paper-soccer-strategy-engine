# Paper Soccer Strategy Engine (Base v0)

This repository contains a deterministic C++20 baseline for paper soccer with:

- A pure rules engine (`papersoccer_core`)
- A terminal CLI for human and bot play (`papersoccer_cli`)
- A seeded bot layer for automated play experiments
- A JSON replay exporter for bot self-play (`papersoccer_replay_export`)
- A static browser replay viewer (`web/index.html`)
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

## Run Web Replay Viewer

Open the static viewer directly:

```bash
open web/index.html
```

The viewer includes a built-in RandomBot replay, so it works without a server.

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
- `include/papersoccer/rules.hpp` - game rules API
- `src/bot.cpp` - bot implementation
- `src/geometry.cpp` - geometry implementation
- `src/rules.cpp` - state transitions and legal move logic
- `src/cli/main.cpp` - terminal game loop
- `src/replay/main.cpp` - bot self-play JSON replay exporter
- `web/index.html` - static web replay viewer
- `web/app.js` - canvas rendering and replay controls
- `web/styles.css` - viewer styling
- `tests/bot_test.cpp` - bot determinism and legality checks
- `tests/test_main.cpp` - test entrypoint
- `tests/rules_test.cpp` - rule correctness scenarios
