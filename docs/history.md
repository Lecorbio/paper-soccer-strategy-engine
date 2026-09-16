# Historical evidence

The current guides organize the repository; frozen records describe what was
actually run. Do not rewrite an old result to fit a newer implementation or
interpret a diagnostic as a passing qualification.

## Where evidence lives

- `benchmarks/` contains curated reports, compact data, and reproducibility tools.
- `submissions/codingame/` contains contest source artifacts and bot-specific
  ledgers. Historical public ranks belong to those exact artifacts.
- Most `results/` output is ignored local data. The tracked hybrid and arena
  evidence collections are explicit exceptions to that rule.
- Detached historical worktrees preserve local models, source snapshots,
  environments, and binaries at paths referenced by receipts.
- A verified private cleanup bundle preserves every pre-cleanup branch tip and
  detached commit. Unfinished hybrid work has separate staged, unstaged, and
  untracked backups; it was not integrated into active source.

The [campaign catalog](campaigns.md) is the starting point for each result and
its implementation guide. Published metrics and hashes can be inspected in a
fresh clone; full numerical reproduction may require retained local inputs.

## Frozen flagship-study v3 links

The [v3 failure record](../benchmarks/flagship_study/V3_VALIDATION_FAILURE.md)
is itself hash-bound by the v4 study manifest. Three original relative links
refer to large predecessor datasets removed from the active checkout. Their
original bytes remain in the published `flagship-study-v4-record` snapshot:

- [Development data](https://github.com/Lecorbio/paper-soccer-strategy-engine/blob/db163d4d44570cb5acac8a7de953d4a31a0346ac/benchmarks/flagship_study/superseded/v3-data/development.json)
- [Validation data](https://github.com/Lecorbio/paper-soccer-strategy-engine/blob/db163d4d44570cb5acac8a7de953d4a31a0346ac/benchmarks/flagship_study/superseded/v3-data/validation.json)
- [Runtime projection](https://github.com/Lecorbio/paper-soccer-strategy-engine/blob/db163d4d44570cb5acac8a7de953d4a31a0346ac/benchmarks/flagship_study/superseded/v3-data/runtime_projection.json)

These are predecessor evidence, not v4 test inputs or new results. The report
retains its exact original bytes. The documentation checker recognizes only
these three exact source/target pairs while the source hash remains unchanged;
all other local links and maintained Markdown anchors must resolve normally.

## Continue from main

Historical branch names in a frozen protocol are evidence identifiers, not a
list of active branches. Use the recorded source commit for an audit and a
new source/input plan for a new experiment. A completed-run metadata check does
not claim a fresh numerical replay. See [development](development.md) and the
[rebuild lifecycle](jacek-replay-rebuild.md#completed-runs-and-explicit-replay).
