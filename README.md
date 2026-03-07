# Paper Soccer Strategy Engine (Base v0)

This repository contains a deterministic C++20 baseline for paper soccer with:

- A pure rules engine (`papersoccer_core`)
- A terminal CLI for human-vs-human play (`papersoccer_cli`)
- Dependency-free tests integrated with CTest (`papersoccer_tests`)

The current version intentionally focuses on core game correctness so minimax/MCTS bots can be added on top without refactoring game state logic.

## Rules Implemented (Kurnik-style defaults)

- Field points: `x in [0,8]`, `y in [0,10]`
- Start point: `(4,5)`
- 8-direction moves to neighboring points
- Segments are undirected and cannot be reused
- Movement along the outer boundary lines is forbidden
- Player 1 attacks north goal row `y=-1`, Player 2 attacks south goal row `y=11`
- Goal nodes per side: `(3,±1)`, `(4,±1)`, `(5,±1)` (north uses `-1`, south uses `11`)
- Goal entries are legal only from the 3-point mouth on each side
- From a mouth point, any adjacent goal node is legal
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

User-facing coordinates in the CLI and renderer are shown as `(row, column)`.
Internally, the engine still stores points as `Point{x, y}`.

## Project Layout

- `include/papersoccer/types.hpp` - core types and hashing
- `include/papersoccer/geometry.hpp` - geometry and adjacency helpers
- `include/papersoccer/rules.hpp` - game rules API
- `src/geometry.cpp` - geometry implementation
- `src/rules.cpp` - state transitions and legal move logic
- `src/cli/main.cpp` - terminal game loop
- `tests/test_main.cpp` - test entrypoint
- `tests/rules_test.cpp` - rule correctness scenarios
