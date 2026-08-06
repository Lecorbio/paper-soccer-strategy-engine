# Paper Soccer Strategy Engine

A deterministic C++20/WebAssembly game-AI engine whose authentic CodinGame submission ranked **5th of 206**, backed by a preregistered **4,800-game** evaluation.

[![CI and Pages](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/workflows/pages.yml/badge.svg)](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/workflows/pages.yml)
[![MIT License](https://img.shields.io/github/license/Lecorbio/paper-soccer-strategy-engine)](LICENSE)

**[Play the live demo](https://lecorbio.github.io/paper-soccer-strategy-engine/)**
· **[Explore the benchmark](https://lecorbio.github.io/paper-soccer-strategy-engine/benchmarks/)**
· [Read the frozen study](benchmarks/flagship_study/REPORT.md)

- **[5/206](submissions/codingame/bots/rank_5/README.md):** verified authentic CodinGame result, tied to an immutable generated submission and SHA-256.
- **[4,800 decisive test games](benchmarks/flagship_study/REPORT.md#frozen-test-results):** 2,400 color-swapped opening pairs with zero truncations.
- **[60.4% neural-versus-hand score](benchmarks/flagship_study/REPORT.md#pairwise-results):** 95% CI 57.4%–63.5%; the selected neural profile measured **[35.7 ms validation p95](benchmarks/flagship_study/REPORT.md#development-and-validation-findings)** on the study machine.

> **Provenance matters:** `rank_5` is the authentic ranked CodinGame artifact. `Rank5DerivedBot` adapts its search code to different demo rules and a fixed 50,000-node profile; demo measurements do not evaluate the ranked submission.

[![Paper Soccer browser game with Rank5DerivedBot selected and search diagnostics available](docs/assets/demo-rank5-derived.jpg)](https://lecorbio.github.io/paper-soccer-strategy-engine/)

## Why this project?

Paper soccer is a compact adversarial game in which rebounds can extend one possession across several edges. That makes a conventional edge-by-edge search horizon misleading: the engine instead searches complete turns, while deterministic work budgets make bot comparisons replayable.

One authoritative rules model powers native play, WebAssembly, replay tooling, and the experiment arena. The project combines MCTS, possession-aware alpha-beta, hand-crafted and neural evaluation, compact make/unmake search state, diagnostics, and frozen statistical evidence.

## Architecture

```mermaid
flowchart LR
    Browser["Browser UI"] --> Wasm["Versioned WebAssembly bridge"]
    CLI["CLI + replay tools"] --> API["Shared C++20 API"]
    Wasm --> API
    Arena["Paired experiment arena"] --> API
    API --> Core["Rules + match state"]
    API --> Search["MCTS + alpha-beta + neural search"]
    Search --> Core
    Arena --> Evidence["Frozen study artifacts"]
```

C++ owns move legality, rebounds, terminal detection, bot state, and replay history. JavaScript only schedules commands and renders complete snapshots. See the [full architecture](docs/architecture.md) and [algorithm notes](docs/algorithms.md).

## Evaluation design

The flagship study compared four competitive demo-rule entrants: tactical MCTS, hand-evaluated alpha-beta, neural alpha-beta, and the fixed Rank5Derived profile. It froze disjoint development, validation, and test openings before test evaluation; played both colors from every opening; selected configurations under a 50 ms validation-p95 gate; and resampled whole opening pairs 10,000 times.

The strongest supported head-to-head finding was neural alpha-beta over hand alpha-beta. Neural alpha-beta versus Rank5Derived remained statistically unresolved at 51.4% with a 95% CI of 48.2%–54.5%—a negative result reported rather than tuned away. The [manifest](benchmarks/flagship_study/manifest.json), [selection lock](benchmarks/flagship_study/selection_lock.json), [curated data](benchmarks/flagship_study/data/test.json), and [report](benchmarks/flagship_study/REPORT.md) are checked in.

## Build and test

Requires CMake 3.20+, a C++20 compiler, Node.js 18+, and Python 3.

```bash
./scripts/build-and-test.sh
```

This creates a Release build in `build/native`, compiles the native tools, verifies generated artifacts, and runs the registered tests. Then launch the terminal game with `./build/native/papersoccer_cli`.

## Documentation

- [Demo, CLI, and replays](docs/demo-and-replays.md)
- [Rules and algorithms](docs/algorithms.md)
- [Architecture](docs/architecture.md)
- [Experiments and benchmark evidence](docs/experiments.md)
- [Reproducibility and artifact hashes](docs/reproducibility.md)
- [CodinGame submission archive](submissions/codingame/README.md)
- [Neural model card](models/README.md)

## License

Owner-authored source and documentation are available under the [MIT License](LICENSE). See [NOTICE](NOTICE.md) for attribution and the status of generated or external material.
