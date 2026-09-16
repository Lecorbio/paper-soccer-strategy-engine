# Paper Soccer Strategy Engine

A deterministic C++20 and WebAssembly engine for playing paper soccer, studying
search algorithms, and testing standalone CodinGame bots. It includes a browser
game, possession-by-possession Game Review, native arenas, and reproducible
model experiments.

[![CI and Pages](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/workflows/pages.yml/badge.svg)](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/workflows/pages.yml)
[![MIT License](https://img.shields.io/github/license/Lecorbio/paper-soccer-strategy-engine)](LICENSE)

**[Play the demo](https://lecorbio.github.io/paper-soccer-strategy-engine/)**
· **[Bot leaderboard](https://lecorbio.github.io/paper-soccer-strategy-engine/leaderboard/)**
· **[Benchmark results](https://lecorbio.github.io/paper-soccer-strategy-engine/benchmarks/)**
· **[Documentation](docs/README.md)**

![A late-game Paper Soccer match against Expert DeepTurnSearchBot at move 33 with live search diagnostics](docs/assets/expert-game.png)

## Recorded results

| Result | What the evidence establishes |
| --- | --- |
| [Rank 4/208 on CodinGame](submissions/codingame/bots/rank_4/README.md) | Historical version 56: 66–24 over 90 games, score `44.29750553418035`. |
| [4,800-game frozen study](benchmarks/flagship_study/REPORT.md) | Four selected bots, 2,400 color-swapped opening pairs, and disjoint selection/test data under the demo rules. |
| [Accepted local large teacher](benchmarks/large_teacher_campaign/REPORT.md) | Four strength panels and retention passed; uncontended maximum 981.945875 ms. This is a local teacher result. |
| [Closed compact-training experiment](benchmarks/compact_value_bfm/TRAINED_V2_OUTCOME.md) | Four pilots and bounded follow-up research produced no newly trained candidate passing the original gates. |

The [22-bot local leaderboard](benchmarks/codingame_leaderboard/README.md) reports
990 protocol-faithful games. Local ratings, historical public ranks, offline
prediction accuracy, and live qualification are separate measurements. The
[campaign catalog](docs/campaigns.md) explains the status and evidence for each
research family.

## Quick start

Install CMake 3.20+, a C++20 compiler, Node.js 18+, Python 3.12–3.14, and Chrome
or Chromium. Set up the small research dependency used by the full test suite:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-research.txt
PAPERSOCCER_BUILD_JOBS=2 ./scripts/build-and-test.sh
./build/native/papersoccer_cli
```

The build script uses the local `.venv` automatically. Tests use small fixed
fixtures; they do not start a production campaign. The checked-in Wasm modules
also let you open `web/index.html` directly in a browser (`open web/index.html`
on macOS). Rebuilding those modules requires pinned Emscripten 6.0.2.

See [Reproducibility](docs/reproducibility.md) for individual targets, compiler
settings, browser discovery, generated-artifact checks, and optional local
forensics.

## Repository map

| Location | Purpose |
| --- | --- |
| `include/`, `src/` | Public C++ API, rules, bots, search, arenas, and browser bridges. |
| `web/` | Static game, Game Review, leaderboard, and benchmark pages. |
| `tests/`, `cmake/` | Correctness and integration checks; grouped build/test registration. |
| `tools/`, `scripts/` | Research workflows, artifact validators, and build entry points. |
| `submissions/codingame/` | Standalone contest bots, generators, and historical records. |
| `models/`, `benchmarks/` | Reviewed model artifacts and curated, reproducible results. |
| `results/` | Mostly ignored local evidence; a few explicitly tracked historical collections. |
| `docs/` | Architecture, workflows, campaign status, and provenance guidance. |

Development starts from `main`. Evidence-bearing historical worktrees are kept
at their original paths because archived receipts can bind absolute paths.
They are not active development checkouts. See [Development](docs/development.md)
before starting another experiment.

## Documentation

- [Documentation index](docs/README.md): choose a reading path.
- [Architecture](docs/architecture.md) and [algorithms](docs/algorithms.md).
- [Demo, CLI, Game Review, and replays](docs/demo-and-replays.md).
- [Development and testing](docs/development.md).
- [Tooling guide](docs/tooling.md) and [campaign catalog](docs/campaigns.md).
- [Experiment methodology](docs/experiments.md) and [reproducibility](docs/reproducibility.md).
- [Historical evidence](docs/history.md), [contest artifacts](submissions/codingame/README.md), and [model provenance](models/README.md).

## License

Owner-authored source and documentation are available under the [MIT License](LICENSE).
See [NOTICE](NOTICE.md) for attribution and the status of generated or external material.
