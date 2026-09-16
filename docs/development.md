# Development workflow

Use `main` as the starting point for new work. The repository contains a working
engine and historical scientific evidence; their maintenance requirements differ.

## Build and test

Follow the [quick start](../README.md#quick-start) to install the pinned research
dependency. The [reproducibility guide](reproducibility.md) lists all prerequisites.

```sh
PAPERSOCCER_BUILD_JOBS=2 ./scripts/build-and-test.sh
ctest --test-dir build/native --output-on-failure
python tools/check_documentation_links.py
```

For a small edit, select the relevant existing CTest target or Python/Node test
module first. Run the complete suite for changes to shared build configuration,
runtime contracts, or integration boundaries. CI uses GCC, Clang, sanitizers,
leaderboard contracts, and replay-training contracts. Generated sources and
model headers are checked against their generators.

Build configuration is grouped into research checks, contest targets, browser
checks, and Wasm targets under `cmake/`. Includes retain the original root scope
and execution order. Add tests to the relevant group instead of growing the
top-level configuration. Keep names, working directories, timeouts, environment
settings, and native/Wasm conditions explicit.

Some recovery-forensics tests need private local artifacts and skip when those
artifacts are absent. Portable contract tests and generated-artifact checks
remain required in a fresh clone. Absence of local evidence is never evidence
of a model passing a gate.

## Source and artifact boundaries

- Edit handwritten source and run the corresponding generator when generated
  output changes. Do not hand-edit deployed submissions or model payloads.
- Keep raw runs under ignored output directories. Publish concise reviewed
  metrics and hashes through a campaign's publication path.
- Treat frozen manifests, report hashes, source snapshots, seeds, exclusions,
  and spent execution claims as immutable. A correction needs new evidence and
  a stated relationship to the original result.
- Keep machine-local configuration such as `AGENTS.md` untracked and locally
  excluded. Do not copy local absolute paths into public documentation.

## Historical worktrees

Old branch names are retired after their committed work is integrated. Some
detached worktrees remain as evidence stores at their original paths. Their
model files, compilers, environments, or source files can be bound by receipts.
The unfinished hybrid work is privately archived, including separate staged,
unstaged, and untracked contents, and remains outside the active codebase.

Before deleting a worktree or cache, establish that its commits are preserved
and its files are not required by any retained receipt. Git status alone cannot
prove that ignored results are disposable. Do not use blanket `git clean -fdx`
on an experiment worktree.

## Start the next experiment

1. Read the [campaign catalog](campaigns.md) and the predecessor's stopping decision.
2. Define the hypothesis, input split, budget, success gate, and stop rule before execution.
3. Freeze a new source/input identity and use a fresh output namespace. Retain
   protected-data exclusions and prior spent claims.
4. Choose explicit worker and memory limits; official timing stays uncontended.
5. Start heavy work explicitly. Build, test, import, and login must not start a
   production campaign. Existing retired login jobs remain disabled.
6. On completion, save the result, verify worker/lock cleanup, and retire the
   monitor or startup entry. A stopped experiment stays stopped.

The rebuild runner's ordinary `run --resume` reads a verified terminal outcome.
Its optional `--replay-completed` performs the expensive computational audit;
see [rebuild lifecycle](jacek-replay-rebuild.md#completed-runs-and-explicit-replay).
