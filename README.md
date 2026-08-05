# Paper Soccer Strategy Engine

A deterministic C++20 playground for playing, studying, and building game AI
for paper soccer.

[![CI and Pages](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/workflows/pages.yml/badge.svg)](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/workflows/pages.yml)
[![MIT License](https://img.shields.io/github/license/Lecorbio/paper-soccer-strategy-engine)](LICENSE)

**[Play in your browser](https://lecorbio.github.io/paper-soccer-strategy-engine/)**
· [View benchmark results](https://lecorbio.github.io/paper-soccer-strategy-engine/benchmarks/)
· [Learn how it works](docs/algorithms.md)
· [Explore the documentation](#documentation)

[![Paper Soccer browser game with the Benchmark results link and Rank5DerivedBot selected](docs/assets/demo-rank5-derived.jpg)](https://lecorbio.github.io/paper-soccer-strategy-engine/)

## What is this?

Paper soccer is a pencil-and-paper strategy game played by drawing unused
segments on a grid. Rebounds can extend a turn, and a match ends when a player
scores or the player to move has no legal move. Those simple rules create a
compact but surprisingly rich search problem.

This project puts one authoritative rules engine behind every way of exploring
the game: the browser demo, terminal play, search bots, replay tools, and the
experiment arena. The included opponents range from seeded random play to MCTS,
alpha-beta search, neural-evaluated search, and a contest-derived search adapter.

The project grew out of the [CodinGame Paper Soccer
contest](https://www.codingame.com/multiplayer/bot-programming/paper-soccer).
Competing in its arena and iterating on its match results played a major role in
improving the bot's search and evaluation. The preserved submissions and their
development history are documented in [CodinGame
submissions](submissions/codingame/README.md); benchmark results and research
notes remain in their own documentation so they can be studied without
crowding this overview.

## Try the game

The [live demo](https://lecorbio.github.io/paper-soccer-strategy-engine/) needs
no installation. You can play against a bot, inspect its search diagnostics,
undo moves, or use **Watch replay** to generate a bot match or open an existing
replay. **[Benchmark results](https://lecorbio.github.io/paper-soccer-strategy-engine/benchmarks/)**
opens the frozen four-bot study.

To run the checked-in browser build locally, open `web/index.html` in a modern
browser. No local server is required.

## Build and test

The native build needs CMake 3.20 or newer and a C++20 compiler. Node.js 18 or
newer and Python 3 are needed to run the complete test suite.

```bash
./scripts/build-and-test.sh
```

The script creates a release build in `build/native`, compiles the native
tools, and runs the registered test suite. Afterward, start the terminal game
with:

```bash
./build/native/papersoccer_cli
```

See [Reproducibility](docs/reproducibility.md) for manual build commands,
sanitizer configurations, WebAssembly verification, and research dependencies.

## What is included?

- A shared C++ rules and match engine used by native and WebAssembly frontends.
- Random, MCTS, alpha-beta, neural-evaluated, and contest-derived opponents.
- A terminal game, browser game, replay exporter, and paired experiment arena.
- Deterministic seeds, fixed-work search options, diagnostics, and artifact
  checks for repeatable investigation.
- Preserved CodinGame submissions and the tooling that generates them.

## Documentation

- [Demo, CLI, and replays](docs/demo-and-replays.md) explains how to play,
  configure bots, import or export games, and use the native tools.
- [Rules and algorithms](docs/algorithms.md) covers the game contract and the
  design of each search opponent.
- [Architecture](docs/architecture.md) maps the C++ API, rules engine,
  frontends, WebAssembly boundary, and repository layout.
- [Experiments and benchmark evidence](docs/experiments.md) records the arena
  methodology, historical results, promotion decisions, and limitations. The
  [benchmark overview](https://lecorbio.github.io/paper-soccer-strategy-engine/benchmarks/)
  presents the current result interactively; the [flagship study
  report](benchmarks/flagship_study/REPORT.md) contains the full analysis.
- [Reproducibility](docs/reproducibility.md) contains build variants, artifact
  verification, hashes, and clean-environment checks.
- [CodinGame submissions](submissions/codingame/README.md) documents the
  preserved contest bots, provenance, and generation workflow.
- [Neural model card](models/README.md) describes the learned evaluator,
  training data, validation, and model contract.

## License

Owner-authored source and documentation are available under the [MIT
License](LICENSE). See [NOTICE](NOTICE.md) for attribution and the status of
generated or external material.
