# Reproducibility

The repository treats determinism, generated-artifact freshness, and evidence
provenance as separate checks. The fast path builds and tests the maintained
project. Additional commands verify the checked-in gameplay and analysis
WebAssembly modules, trained model header, and immutable CodinGame artifacts
without rerunning long tournaments.

## Prerequisites

For the default native build and complete local test registration:

- CMake 3.20 or newer;
- a C++20 compiler (GCC or Clang);
- Node.js 18 or newer for JavaScript, browser/Wasm, replay, arena-CLI, and
  submission-freshness tests;
- Chrome or Chromium for the registered browser smoke tests, discoverable
  automatically or supplied through `PAPERSOCCER_CHROME`; and
- Python 3 for generated-model freshness tests.

Training and research scripts additionally use Python 3.12–3.14 and the pinned
package in `requirements-research.txt`. The C++ engine, CLI, checked-in models,
and live browser game have no Python or NumPy runtime dependency.

Emscripten is needed only to rebuild the checked-in browser modules. Exact
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
Wasm and Game Review boundaries can also be exercised directly:

```bash
node --test tests/web/web_wasm_test.mjs
node --test tests/web/web_review_wasm_test.mjs
node --test tests/web/game_review_client_test.mjs
node --test tests/web/game_review_ui_test.mjs
```

Full competitive tournaments are deliberately outside CTest. Tests use small,
fixed budgets to validate legality, deterministic counters, report schemas,
paired accounting, and command integration without depending on machine speed.
The review registrations also cover native/Wasm transcript parity,
long-trail fixed-heap stress, classic-worker and direct-file fallback
protocols, cancellation and stale sessions, and authoritative try-line
sandboxing.

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

## Generated CodinGame submissions

The current production snapshot is
`submissions/codingame/bots/rank_4/submission.cpp`. It combines complete-turn
search, the retained replay book and replay-value anchor, and the generated
teacher-residual model. Verify its generators without writing any artifact:

```bash
node submissions/codingame/tools/generate_submission.mjs rank_4 --check
```

Verify the maintained generated source identity:

```bash
cmake -E sha256sum submissions/codingame/bots/rank_4/submission.cpp
```

Expected SHA-256:

```text
5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9
```

The local snapshot corresponds to CodinGame history version 56, agent
`6604719`, submission `41114327`, and its completed rank 4/208 result. Because
CodinGame does not expose a remote source digest, its record documents an
authenticated history/source fingerprint rather than claiming that the local
SHA-256 was measured remotely. See the
[rank-4 record](../submissions/codingame/bots/rank_4/README.md).

The immutable historical `rank_5` artifact remains a separate required input
because `Rank5DerivedBot` adapts its search implementation. Verify it too:

```bash
node submissions/codingame/tools/generate_submission.mjs rank_5 --check
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

Do not hand-edit either generated submission, its generated headers, replay
data, model data, or verified maintained source while reorganizing
documentation.

The full native test suite also compiles each submission, runs per-bot tests and
protocol smokes, and invokes every generator in `--check` mode. Shared tooling
and the directory contract are documented in the
[CodinGame archive](../submissions/codingame/README.md).

## CodinGame-rules bot leaderboard

The leaderboard is an offline, reviewed benchmark rather than a browser arena.
Its explicit roster is
`benchmarks/codingame_leaderboard/roster.json`. The manifest covers all 22
CMake-registered bot directories, binds each generated submission by SHA-256,
and maps every directory to its own executable entrant, including
`rank_4_jacek_hybrid` and `jacek_arena_bfm`. The website accepts no uploads and
cannot run a bot; it only renders the checked-in summary.

First configure a Release build and compile the native referee and submission
executables:

```bash
cmake -S . -B build/leaderboard -DCMAKE_BUILD_TYPE=Release
cmake --build build/leaderboard
```

Run the fast contract validation independently of a full tournament:

```bash
python3 benchmarks/codingame_leaderboard/leaderboard.py validate
```

An intentional full refresh runs all 990 games serially. It starts fresh
persistent bot processes for each match, uses the standard player-ID and
complete-turn protocol, and enforces 1,000 ms for each bot's first response and
200 ms thereafter:

```bash
python3 benchmarks/codingame_leaderboard/leaderboard.py run \
  --referee build/leaderboard/papersoccer_codingame_referee \
  --build-dir build/leaderboard \
  --checkpoint build/leaderboard/leaderboard-checkpoint.json \
  --output benchmarks/codingame_leaderboard/tournament.json
```

If a run is interrupted, repeat the same command with `--resume`. Resume is
fail-closed: it refuses a checkpoint whose roster, tournament contract, or
runtime fingerprint differs from the current invocation. Referee and
process-launch failures abort the run. Bot timeouts, crashes, malformed output,
illegal actions, incomplete rebounds, and output after a terminal edge remain
visible in the artifact as ordinary scored forfeits.

The schedule is fixed by SplitMix64 seed `20260813`: two complete
color-swapped round robins plus three seeded color-swapped perfect-matching
rounds. Each entrant plays 90 games, 45 as each player, and four or six games
against every opponent. Rating follows decisive 1v1 TrueSkill with `mu=25`,
`sigma=25/3`, `beta=25/6`, `tau=25/300`, and zero draw probability. The
conservative leaderboard value is the full-precision `mu - 3 sigma`; forfeits
update it as ordinary losses. The UI calls this **Local CodinGame-style score**
because CodinGame does not publish the full ranking parameters or matchmaking
contract.

Validate a completed raw artifact, then derive the compact static snapshot:

```bash
python3 benchmarks/codingame_leaderboard/leaderboard.py validate \
  --artifact benchmarks/codingame_leaderboard/tournament.json \
  --referee build/leaderboard/papersoccer_codingame_referee

python3 benchmarks/codingame_leaderboard/leaderboard.py publish \
  --input benchmarks/codingame_leaderboard/tournament.json \
  --output web/leaderboard/leaderboard-results.js \
  --referee build/leaderboard/papersoccer_codingame_referee
```

The raw `papersoccer.codingame-leaderboard-tournament.v1` artifact retains the
complete action transcripts, outcomes, timings, hashes, schedule and rating
contract, source-tree identity, and compiler/OS/CPU provenance. The derived
`papersoccer.codingame-leaderboard-summary.v1` classic-script snapshot contains
only standings and pairwise summaries. Check byte-for-byte publication
freshness without rewriting either file:

```bash
python3 benchmarks/codingame_leaderboard/leaderboard.py publish \
  --input benchmarks/codingame_leaderboard/tournament.json \
  --output web/leaderboard/leaderboard-results.js \
  --referee build/leaderboard/papersoccer_codingame_referee \
  --allow-historical-sources --check
```

The checked-in 22-bot tournament retains the exact source provenance of its
completed run. Later non-gameplay documentation and compatibility maintenance
changed the surrounding source tree, so `--allow-historical-sources` is
required when replaying and publishing this artifact. The flag still enforces
the current roster and schedule, replays every transcript, recomputes standings,
and checks exact snapshot bytes. Omit it for a newly generated tournament from
current sources.

The manual tournament workflow uploads both generated artifacts for maintainer
review. They become site evidence only through a normal pull request. Regular
pull-request CI runs contract, unit, smoke, and snapshot-freshness checks; it
does not spend 990 wall-clock-limited games on every change. Changes to the
roster, generated submissions, referee, rules, schedule, or rating contract
therefore make the checked-in publication stale instead of silently rerating
old games.

## WebAssembly artifacts

`web/papersoccer-wasm.js` embeds the live-game Wasm bytes, allowing direct-file
use and static GitHub Pages deployment. The build pins Emscripten 6.0.2, uses a
fixed 32 MiB initial heap, disables memory growth, and emits a single file for
browser and Node environments.

`web/papersoccer-analysis-wasm.js` is a separate lazy-loaded single-file
artifact for Game Review. It has its own fixed 64 MiB initial heap, disables
memory growth, and supports browser, classic-worker, and Node environments. On
hosted pages `web/game-review-worker.js` owns a separate instance. Under
`file://`, failure to create or load that worker creates a second main-thread
analysis instance and schedules one synchronous possession step per event-loop
task. Neither path shares state with the 32 MiB live-game module.

After changing C++ rules, bots, review code, or either browser boundary,
regenerate both artifacts:

```bash
emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm --target update_papersoccer_web
cmake --build build/wasm --target update_papersoccer_analysis_wasm
```

To prove that the checked-in modules match current sources without updating
the repository:

```bash
emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm --target check_papersoccer_web
cmake --build build/wasm --target check_papersoccer_analysis_wasm
```

CMake rejects another Emscripten version before producing an artifact. The
generated JavaScript is expected to be marked as generated for repository
language statistics; the C++, worker, and hand-written JavaScript clients
around it remain maintained source. The analysis C ABI and deterministic export
contract are documented in
[Architecture](architecture.md#browser-and-analysis-boundaries).

## Documentation screenshot

The README screenshot is a reproducible documentation capture rather than a
benchmark artifact. Open the local `web/index.html` at a 1440×900 viewport,
select **Expert — DeepTurnSearch**, choose **Move second**, and start the game.
At each human turn, choose these displayed row/column destinations in order:

```text
r6 c3, r6 c2, r5 c4, r5 c3, r4 c4,
r4 c2, r5 c2, r6 c1, r5 c2, r5 c1
```

The fixed `deep-400k` opponent then finishes move 33 and returns control to the
human. Scroll the control panel to its top so the selected Expert profile,
active-game diagnostics, move number, status, and highlighted legal choices are
visible. Capture that exact state as `docs/assets/expert-game.png`.

The PNG replaces the retired replay screenshot; do not keep both images or
recapture a different game/profile under the same name.

## Game Review gate

The frozen Game Review gate uses fresh opening banks disjoint from every
flagship bank. It evaluates 100k, 200k, and 400k DeepTurnSearch candidates
against the fixed Rank5Derived 50k and selected JacekInspired 20k references at
4, 8, 12, and 20 opening plies. Development contains 25 color-swapped pairs per
cell, validation 50, and the one-shot selected-profile test 100. Calibration is
fit on validation decisions only. The exact contract is
`benchmarks/game_review_gate/manifest.json`.

This section records the original source-bound execution workflow. The frozen
evidence binds competition source identity
`a021dd00845ed6cc4708852820e0f40157f3c57f4926a2143b1fcb35295a1a42`
at commit `cacfc0b2ef031fd19ef727ad258ba0a4ed417ef2`. Full
`run_gate.py validate` also hashes the ignored raw shards under
`results/game_review_gate/`.
Consequently it is not a clean-clone validation command and correctly rejects
later `main` source identities. Current checkouts can run the gate unit tests
and inspect the checked-in compact result and report; replaying the complete
validation chain requires the exact historical source snapshot plus its
preserved raw result tree.

In that archival environment, first build an optimized native arena and verify
all frozen opening identities:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release

python3 benchmarks/game_review_gate/run_gate.py validate \
  --opening-tool build/release/papersoccer_opening_bank \
  --verify-regeneration
```

Run and aggregate development and validation. The runner is resumable and
stores raw reports below the manifest-bound `results/game_review_gate/` path:

```bash
python3 benchmarks/game_review_gate/run_gate.py run \
  --phase development --arena build/release/papersoccer_arena
python3 benchmarks/game_review_gate/run_gate.py aggregate --phase development

python3 benchmarks/game_review_gate/run_gate.py run \
  --phase validation --arena build/release/papersoccer_arena
python3 benchmarks/game_review_gate/run_gate.py aggregate --phase validation
```

For parallel execution, invoke every distinct `--shard-index` from zero through
`--shard-count - 1` with the same shard count, then aggregate once. A shard
will reuse an already validated raw unit instead of overwriting it.

The selected profile and calibration are linked into the analysis module, so
stabilize that generated module before promoting its latency. First measure the
pre-lock probe artifact and render a draft lock from that measurement:

```bash
emcmake cmake -S . -B build/wasm -DCMAKE_BUILD_TYPE=Release

node benchmarks/game_review_gate/measure_wasm_latency.mjs \
  --module web/papersoccer-analysis-wasm.js \
  --output results/game_review_gate/wasm-latency-draft-1.json \
  --emscripten-version 6.0.2

python3 benchmarks/game_review_gate/run_gate.py render-lock \
  --latency-input results/game_review_gate/wasm-latency-draft-1.json
cmake --build build/wasm \
  --target update_papersoccer_web update_papersoccer_analysis_wasm
```

Remeasure the linked artifact to a new draft, then check that the same lock is
selected:

```bash
node benchmarks/game_review_gate/measure_wasm_latency.mjs \
  --module web/papersoccer-analysis-wasm.js \
  --output results/game_review_gate/wasm-latency-draft-2.json \
  --emscripten-version 6.0.2

python3 benchmarks/game_review_gate/run_gate.py check-lock \
  --latency-input results/game_review_gate/wasm-latency-draft-2.json
cmake --build build/wasm \
  --target check_papersoccer_web check_papersoccer_analysis_wasm
```

If that check differs, render from draft 2, rebuild, and measure to a fresh
draft path; repeat until `check-lock --latency-input` succeeds. Never overwrite
or hand-edit a draft. Promote only the stable linked-module measurement, then
create and verify the official selection lock. The example below assumes draft
2 passed; substitute the last fresh draft after any additional iteration:

```bash
python3 benchmarks/game_review_gate/run_gate.py record-latency \
  --input results/game_review_gate/wasm-latency-draft-2.json
python3 benchmarks/game_review_gate/run_gate.py lock-selection
python3 benchmarks/game_review_gate/run_gate.py check-lock
```

During the original run, maintainers reviewed and committed the manifest,
banks, development and validation summaries, latency measurement, final
analysis module, selection/calibration lock, and generated C++ lock before
test. The Release arena embeds its source commit, and the test runner refuses
an arena from another or dirty source state. They rebuilt the native Release
arena from that exact clean commit, then ran the test exactly once:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --target papersoccer_arena

python3 benchmarks/game_review_gate/run_gate.py run \
  --phase test --arena build/release/papersoccer_arena
python3 benchmarks/game_review_gate/run_gate.py aggregate --phase test
python3 benchmarks/game_review_gate/run_gate.py report
python3 benchmarks/game_review_gate/run_gate.py validate --require-complete
```

The test runner writes a manifest-, selection-, and arena-bound one-shot marker
before starting and refuses a second completed evaluation. Do not delete or
work around that marker. The final report records whether each paired 95% lower
bound from 10,000 whole-pair bootstrap resamples clears 50%; only both positive
results permit the **Expert — DeepTurnSearch** selector. Deep review itself does
not depend on that strength claim.

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
exploratory models, and explicitly allowlisted curated campaign records:

```text
results/
├── arena/                       ignored local output
├── codingame/                   ignored local output
├── research/                    ignored local output
├── rank_4_jacek_hybrid/         tracked campaign evidence
└── jacek_arena_bfm/             mixed; selected evidence is tracked
```

Create subdirectories as needed before redirecting output. Examples throughout
[Experiments](experiments.md) write raw arena and Wasm JSON under `results/`.
Research scripts use the same convention, including paths such as:

- `results/codingame/selfplay_nn/neural_model.json`;
- `results/codingame/rank_4/arena_batch_<AGENT_ID>.json`; and
- `results/codingame/alpha_beta/goal_block_strategy/replay_value_model.json`.

Raw `results/` output is ignored by default. The `.gitignore` allowlist makes
two deliberate exceptions: the rank-4/Jacek hybrid campaign is tracked there,
and selected fresh-arena exclusions, plans, attestations, derivations, models,
reports, comparisons, arena records, and scratch manifests/producers are
tracked under `results/jacek_arena_bfm/`. Other compact benchmark summaries stay
under `benchmarks/`; source-owned experiment records stay beside the relevant
CodinGame bot.

## What is deterministic

The following are intended to reproduce exactly when inputs and implementation
match:

- legal moves and game transitions;
- fixed-seed RandomBot and MCTS random streams;
- fixed-iteration or fixed-node search choices and deterministic counters;
- possession partitioning, exact endgame proofs, Game Review grades, and
  complete-action diagnostics for a locked profile and calibration;
- paired opening banks and bootstrap resampling from recorded seeds;
- leaderboard scheduling, rating from recorded decisive games, canonical raw
  serialization, and compact snapshot publication;
- generated submission/model headers; and
- checked-in Wasm bytes when using the pinned toolchain.

Wall-clock timings and fresh time-limited bot decisions are machine-specific.
Leaderboard results are therefore frozen with their executable hashes and
runtime provenance; the schedule and rating are reproducible exactly from the
recorded games even when another machine's live rematch differs. Arena win
records are meaningful only with the recorded rules, opponents, color swaps,
seeds, budgets, and stopping criteria. A confidence interval that crosses the
promotion threshold is not a positive gate. Rank5Derived uses different rules
and work limits from the CodinGame entrant, so its deterministic demo results
never inherit the original rank.

## Pre-release checklist

Before publishing documentation or code that changes evidence paths:

1. Run `./scripts/build-and-test.sh`.
2. Run both the rank-4 and rank-5 generators in `--check` mode and verify their
   expected SHA-256 values.
3. With Emscripten 6.0.2, run `check_papersoccer_web` and
   `check_papersoccer_analysis_wasm` if shared or browser C++ changed.
4. Run `web_summary.py --check` if benchmark inputs or presentation changed.
5. Run the leaderboard `validate` and `publish --check` commands if its roster,
   referee, rules, submissions, tournament contract, artifact, or page changed.
6. Run the Python import validation and model generator check if research code
   or dependencies changed.
7. Confirm raw outputs remain in ignored paths and that only the explicit
   `.gitignore` allowlists add curated campaign evidence under `results/`;
   general benchmark reports remain tracked under `benchmarks/`.
8. Recheck Markdown links from their containing file; paths inside `docs/`
   require `../` to reach repository-root artifacts.
9. Review claims against [Experiments](experiments.md) and the specialized
   source records, preserving limitations and bot provenance.
10. Treat the completed Game Review gate as a frozen record. Run full
    `validate --require-complete` only with its exact source snapshot and
    preserved raw shards; on current `main`, run the gate unit tests and review
    the checked-in lock, compact result, and report instead.
