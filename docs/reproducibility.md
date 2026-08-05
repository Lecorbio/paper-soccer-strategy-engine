# Reproducibility

The repository treats determinism, generated-artifact freshness, and evidence
provenance as separate checks. The fast path builds and tests the maintained
project. Additional commands verify the checked-in WebAssembly module, trained
model header, and immutable CodinGame submission without rerunning long
tournaments.

## Prerequisites

For the default native build and complete local test registration:

- CMake 3.20 or newer;
- a C++20 compiler (GCC or Clang);
- Node.js 18 or newer for JavaScript, browser/Wasm, replay, arena-CLI, and
  submission-freshness tests; and
- Python 3 for generated-model freshness tests.

Training and research scripts additionally use Python 3.12–3.14 and the pinned
package in `requirements-research.txt`. The C++ engine, CLI, checked-in models,
and live browser game have no Python or NumPy runtime dependency.

Emscripten is needed only to rebuild the checked-in browser module. Exact
artifact verification requires Emscripten 6.0.2.

## One-command native build and test

From a clean clone, run:

```bash
./scripts/build-and-test.sh
```

This is the canonical configure-build-CTest entrypoint used by both local
instructions and CI. It configures the default native build, builds all normal
targets, and runs CTest with failures visible. CI selects GCC, Clang, or
sanitizer settings around the same script rather than maintaining a different
test recipe.

The equivalent manual sequence is useful while developing a single target:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the core binary directly when isolating a C++ failure:

```bash
./build/papersoccer_tests
```

When Node.js 18+ is available, CTest registers browser/Wasm, UI support, replay
exporter, arena CLI, generated submission, and protocol tests. The checked-in
Wasm boundary can also be exercised directly:

```bash
node --test tests/web/web_wasm_test.mjs
```

Full competitive tournaments are deliberately outside CTest. Tests use small,
fixed budgets to validate legality, deterministic counters, report schemas,
paired accounting, and command integration without depending on machine speed.

## Research environment

Create an isolated environment and validate that the pinned training dependency
imports without generating a model or dataset:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-research.txt
python tools/train_jacek_neural.py --help >/dev/null
```

`requirements-research.txt` intentionally contains only the dependency imported
by research/training code; application and test runtime dependencies are not
inflated to mirror a developer machine.

## Generated CodinGame submission

The accepted historical artifact is
`submissions/codingame/bots/rank_5/submission.cpp`. It is generated from an
ordered maintained-source manifest, the replay book, and the replay-trained
value model. Do not edit the generated file, generated headers, replay book,
model, or verified source while reorganizing documentation.

Verify generator freshness without writing it:

```bash
node submissions/codingame/tools/generate_submission.mjs rank_5 --check
```

Verify the immutable generated source identity:

```bash
cmake -E sha256sum submissions/codingame/bots/rank_5/submission.cpp
```

Expected SHA-256:

```text
f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29
```

The hash belongs to `submission.cpp`, not the smaller maintained `bot.cpp`.
The exact artifact corresponds to CodinGame history version 26, agent
`6561779`, submission `41015554`, and its completed historical rank 5/206
result. The [rank-5 record](../submissions/codingame/bots/rank_5/README.md)
documents source size, arena evidence, replay provenance, and timing margins.

The full native test suite also compiles each submission, runs per-bot tests and
protocol smokes, and invokes every generator in `--check` mode. Shared tooling
and the directory contract are documented in the
[CodinGame archive](../submissions/codingame/README.md).

## WebAssembly artifact

`web/papersoccer-wasm.js` embeds the Wasm bytes, allowing direct-file use and
static GitHub Pages deployment. The build pins Emscripten 6.0.2, uses a fixed
32 MiB initial heap, disables memory growth, and emits a single file for browser
and Node environments.

After changing C++ rules, bots, or the browser-session boundary, regenerate it:

```bash
emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm --target update_papersoccer_web
```

To prove that the checked-in module matches current sources without updating
the repository:

```bash
emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm --target check_papersoccer_web
```

CMake rejects another Emscripten version before producing an artifact. The
generated JavaScript is expected to be marked as generated for repository
language statistics; the C++ and hand-written JavaScript around it remain
maintained source.

## Benchmark website snapshot

`web/benchmarks/benchmark-results.js` is a compact, deterministic publication
of the frozen flagship study. It is generated from the study manifest,
selection lock, and curated test data, and contains bot-performance results
only. It deliberately excludes individual games, execution environments,
timestamps, and hashes.

Check that the published snapshot is current without changing it:

```bash
python3 benchmarks/flagship_study/web_summary.py --check
```

After intentionally changing the source benchmark artifacts or the publication
contract, regenerate the snapshot with:

```bash
python3 benchmarks/flagship_study/web_summary.py --write
```

The generator validates that the study is frozen, complete, and tied to the
selected configurations before it publishes anything.

## Jacek-inspired model

The checked-in model JSON and generated C++ header are bound by hashes and a
generator freshness test:

```bash
python3 tools/generate_jacek_neural_model.py --check
```

The full corpus and retraining workflow is intentionally not part of the normal
build. It is deterministic from versioned code, seeds, rules, and budgets:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target papersoccer_jacek_training_data
mkdir -p results/research/jacek_article

./build/papersoccer_jacek_training_data \
  results/research/jacek_article/training_v2.jsonl 1024 73194721 4000

python3 tools/train_jacek_neural.py \
  results/research/jacek_article/training_v2.jsonl \
  --output results/research/jacek_article/value_model.json \
  --seed 20260723 \
  --epochs 50
```

This writes local research output under `results/` instead of overwriting the
reviewed checkpoint. To intentionally update the maintained model, compare the
new artifact and metrics first, then follow the canonical generation procedure
in [models/README.md](../models/README.md). That record contains the exact
training corpus, trainer, model, and header hashes and notes that an independent
clean retraining reproduced the reviewed artifacts byte-for-byte.

Several generators bind the trainer script itself by SHA-256. Do not edit a
hashed trainer merely to redirect an ad-hoc run. Pass an explicit output under
`results/`, reproduce into `results/repro/` when checking an exact artifact,
and compare that output with the tracked canonical file.

## Experiment outputs

Use `results/` for raw local reports, generated corpora, downloaded batches,
and exploratory models:

```text
results/
├── arena/
├── models/
└── codingame/
```

Create subdirectories as needed before redirecting output. Examples throughout
[Experiments](experiments.md) write raw arena and Wasm JSON under `results/`.
Research scripts use the same convention, including paths such as:

- `results/codingame/selfplay_nn/neural_model.json`;
- `results/codingame/selfplay_nn_v2/arena_batch_<AGENT_ID>.json`; and
- `results/codingame/alpha_beta/goal_block_strategy/replay_value_model.json`.

The root `results/` directory is ignored. Do not use it for compact evidence
that supports a committed claim, verified contest artifacts, frozen regression
fixtures, or model provenance. Reviewed benchmark summaries stay under
`benchmarks/`; source-owned experiment records stay beside the relevant
CodinGame bot.

## What is deterministic

The following are intended to reproduce exactly when inputs and implementation
match:

- legal moves and game transitions;
- fixed-seed RandomBot and MCTS random streams;
- fixed-iteration or fixed-node search choices and deterministic counters;
- paired opening banks and bootstrap resampling from recorded seeds;
- generated submission/model headers; and
- checked-in Wasm bytes when using the pinned toolchain.

Wall-clock timings are machine-specific. Arena win records are meaningful only
with the recorded rules, opponents, color swaps, seeds, budgets, and stopping
criteria. A confidence interval that crosses the promotion threshold is not a
positive gate. Rank5Derived uses different rules and work limits from the
CodinGame entrant, so its deterministic demo results never inherit the original
rank.

## Pre-release checklist

Before publishing documentation or code that changes evidence paths:

1. Run `./scripts/build-and-test.sh`.
2. Run the rank-5 generator `--check` and verify the expected SHA-256.
3. With Emscripten 6.0.2, run `check_papersoccer_web` if browser C++ changed.
4. Run `web_summary.py --check` if benchmark inputs or presentation changed.
5. Run the Python import validation and model generator check if research code
   or dependencies changed.
6. Confirm raw outputs remain under ignored `results/` and curated reports
   remain tracked under `benchmarks/`.
7. Recheck Markdown links from their containing file; paths inside `docs/`
   require `../` to reach repository-root artifacts.
8. Review claims against [Experiments](experiments.md) and the specialized
   source records, preserving limitations and bot provenance.
