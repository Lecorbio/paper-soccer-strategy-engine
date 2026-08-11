# Shared CodinGame tools

This directory contains strategy-independent utilities. Bot implementations,
models, replay books, timing probes, and experiment evidence belong under
`../bots/<name>/`.

## Submission generation

```sh
node submissions/codingame/tools/generate_submission.mjs BOT_NAME [--check]
```

The generator reads `bots/BOT_NAME/submission.json` and `sources.txt`, runs any
configured bot-local generators, removes repository-local includes, combines
standard headers, strips comments and redundant blank lines, rejects unexpected
dependencies, and enforces the configured source-size limit.

## Protocol smoke test

`protocol_smoke_test.mjs` starts a compiled submission as each player ID and
checks its first protocol response. CTest invokes it for every registered bot.

## Arena replay analysis

```sh
python3 submissions/codingame/tools/analyze_arena.py AGENT_ID --pretty
python3 submissions/codingame/tools/screen_replay_book.py \
  AGENT_ID submissions/codingame/bots/BOT_NAME/replay_book.json
```

The first command downloads completed public battles and summarizes opponents
and same-color divergences. The second measures which completed games a replay
book would change first.

## Arena batch diagnostics

`collect_arena_batch.py` freezes a complete public battle window for one exact
agent, submission, source artifact, and repository commit:

```sh
python3 submissions/codingame/tools/collect_arena_batch.py \
  --agent-id AGENT_ID --submission-id SUBMISSION_ID \
  --source /path/to/exact-editor-source.cpp \
  --source-sha256 EXPECTED_SHA256 \
  --exclusion-registry /path/to/frozen-exclusion-registry.json \
  --exclusion-registry-sha256 APPROVED_REGISTRY_SHA256 \
  --run-id descriptive-unique-id --expected-games 90 \
  --export-auditor-tsv /path/to/validated-games.tsv
```

The default archive is `results/codingame_arena_diagnostics`, which is kept
separate from every training corpus. It is append-only and content addressed:
exact source bytes, API responses, request receipts, replay payloads, per-game
records, and the final batch manifest are never replaced. For every fetched
replay, the manifest embeds the rules-validated transcript, frozen opponent
rank/score, player color, result, and operational classification. Games against
unranked opponents are retained. Protected and already-known game IDs are
counted as window-accounted records without requesting or claiming replay
details; accepted and clean-terminal coverage are reported separately.

The collector requires an already-frozen ID-only exclusion registry and its
explicitly approved SHA-256. It never builds one itself or traverses protected
evidence, so collection cannot open a protected replay or promotion payload
while discovering exclusions. The approved hash is an external completeness
contract; a merely schema-valid empty or stale registry is not sufficient.

Operational endings are derived from frames, stdout, stderr, and exact rules
replay. The unreliable public `agents[].valid` value is deliberately ignored,
so opponent forfeits remain distinct from clean strength games and from the
focus bot's own failures.

`--export-auditor-tsv` provides the strict four-column input consumed by the
full-turn replay decision auditor. It exports accepted, clean, rule-terminal
games only; protected, rejected, incomplete, timeout, crash, illegal-output,
and opponent-forfeit records are retained in the manifest but excluded from
the TSV. The export is also write-once and refuses to replace different bytes.
Use `--export-existing-manifest MANIFEST --export-auditor-tsv OUTPUT
--exclusion-registry-sha256 APPROVED_REGISTRY_SHA256` to recreate a
provenance-bearing auditor input from an archived manifest without network
access. Offline export still requires the caller-approved registry hash and
fails closed unless the content-addressed manifest, archived source, game
records, and exclusion registry all match their cross-references. An accepted
game that appears in the approved registry is rejected rather than exported.

Verify an archive without networking:

```sh
python3 submissions/codingame/tools/collect_arena_batch.py \
  --data-root results/codingame_arena_diagnostics --check
```
