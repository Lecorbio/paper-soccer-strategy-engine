# Jacek-native live-loss restart pipeline

The live-loss restart producer turns one explicitly named collector
`clean-auditor.tsv` into new native self-play games. It never treats an
observed arena move as a correct move, a policy target, or a value target. The
arena transcript is used only to reconstruct a nonterminal starting boundary;
both players' moves after that boundary are generated afresh by the native
complete-turn BFM engine, and only the fresh game's final result supplies the
primary mover-relative target.

## Closed input and provenance contract

`tools/jacek_native_restart_round2.py` accepts exactly one `--input` file. It
does not search result directories, fetch replays, or open match banks. Paths
whose names identify matches, protected, sealed, prospective, or final banks
fail closed. The TSV must contain exactly these leading metadata fields:

- `agent_id`
- `arena_manifest_sha256`
- `asserted_source_sha256`
- `asserted_submission_id`
- `collector_sha256`
- `exclusion_registry_sha256`
- `repository_commit`
- `run_id`
- `source_binding_status`

The header is exactly
`game_id<TAB>candidate_player<TAB>winner<TAB>turns`. Metadata must precede the
header, game IDs must be unique, and no extra columns or metadata are accepted.
The caller must separately provide the expected agent, submission, source,
arena-manifest, and exclusion-registry identities; every identity must match
before generation begins. An asserted source binding stays asserted and is
never upgraded by this pipeline.

Before any output file is opened, the C++ producer independently replays every
complete transcript with `OwnGoalsAllowed + MoverLoses`. It rejects illegal
directions, incomplete rebound chains, bytes after a handoff or terminal
result, nonterminal transcripts, duplicate games, and winner mismatches. This
structural validation supplements the collector's clean-export guarantee; the
four-column format intentionally contains no second operational-status field.

The run directory archives the exact TSV bytes and SHA-256, the metadata map,
the source/manifest/exclusion identities, runtime checkpoints, exact producer
binary, compiler identity, source hashes, shard hashes, and a canonical
manifest. The strict corpus loader verifies all of them again.

## Deterministic prefix and continuation schedule

Only candidate losses are eligible. For each loss, the selector considers
noninitial complete-turn boundaries where the candidate is to move. It retains
up to `--prefixes-per-loss` boundaries evenly across that ordered list, then
deduplicates identical native feature states in collector order. The complete
selected-prefix plan is stored in the manifest and recomputed by the loader;
removing or cherry-picking a prefix invalidates the run.

For a cheap pipeline pilot, `--max-selected-prefixes N` applies a frozen,
evenly spaced cap after the full loss-prefix plan is built and deduplicated;
`N=1` selects its middle entry. The cap and resulting plan are manifest-bound,
so a pilot subset is deterministic rather than hand-picked. Zero (the
default) retains the full plan.

Each prefix receives an even number of continuations. The two player
checkpoints swap roles on odd continuations. Seeds derive only from the frozen
run seed and global continuation index. Temperature is restart-relative, so
late live boundaries can still receive a short exploratory phase. All
continuations originating from the same arena game share one split group,
preventing their related states from crossing whole-game dataset splits.

Optional reanalysis uses the native 30k/100k teacher pair. Exact results
override outcomes; a nonexact auxiliary value is eligible only when there is no
deadline interruption, the selected action is stable, and value delta is at
most 0.05. Generator/proof sampling truncation counters and planned-work
exhaustion are retained as separate diagnostics, not mislabeled as clock
failures.

Shards run concurrently. `--parallel N` selects between one and `--shards`
workers; when omitted, the producer uses the smaller of the available CPU-core
count and the shard count. Parallelism is execution-only: it is excluded from
the canonical manifest, as are wall-clock timings. Shard reports are sorted by
shard index before serialization, so identical runs made with `--parallel 1`
and `--parallel 2` produce byte-identical shards and manifests.

## Example

```sh
python3 tools/jacek_native_restart_round2.py \
  --input /explicit/path/clean-auditor.tsv \
  --output-dir /explicit/path/native-restart-run \
  --expected-agent-id 6600000 \
  --expected-submission-id 41100000 \
  --expected-source-sha256 "$SOURCE_SHA256" \
  --expected-manifest-sha256 "$ARENA_MANIFEST_SHA256" \
  --expected-exclusion-registry-sha256 "$EXCLUSION_REGISTRY_SHA256" \
  --player-one-checkpoint /explicit/path/current.runtime \
  --player-two-checkpoint /explicit/path/previous.runtime \
  --teacher-checkpoint /explicit/path/teacher.runtime \
  --prefixes-per-loss 4 \
  --max-selected-prefixes 0 \
  --continuations-per-prefix 2 \
  --work 4096 --reanalysis-work 30000 --verification-work 100000 \
  --shards 14 --parallel 14
```

Validate all manifest-declared shards by naming them explicitly:

```sh
python3 tools/jacek_native_restart_corpus_round2.py \
  /explicit/path/native-restart-run/shard-00-of-14.jsonl \
  /explicit/path/native-restart-run/shard-01-of-14.jsonl \
  /explicit/path/native-restart-run/shard-02-of-14.jsonl \
  /explicit/path/native-restart-run/shard-03-of-14.jsonl \
  /explicit/path/native-restart-run/shard-04-of-14.jsonl \
  /explicit/path/native-restart-run/shard-05-of-14.jsonl \
  /explicit/path/native-restart-run/shard-06-of-14.jsonl \
  /explicit/path/native-restart-run/shard-07-of-14.jsonl \
  /explicit/path/native-restart-run/shard-08-of-14.jsonl \
  /explicit/path/native-restart-run/shard-09-of-14.jsonl \
  /explicit/path/native-restart-run/shard-10-of-14.jsonl \
  /explicit/path/native-restart-run/shard-11-of-14.jsonl \
  /explicit/path/native-restart-run/shard-12-of-14.jsonl \
  /explicit/path/native-restart-run/shard-13-of-14.jsonl
```

The additive round-two trainer accepts these shards through its explicit
restart path group. Restart lineage remains separate in the model report and
records `observed_move_policy_labels: 0`; ordinary procedural games are never
relabelled as live restarts, or vice versa.

Focused verification:

```sh
cmake --build build -j8 --target \
  papersoccer_jacek_native_restart_round2 \
  papersoccer_jacek_native_restart_round2_test
ctest --test-dir build --output-on-failure -R jacek_native_restart_round2
python3 -m unittest -v \
  tests/codingame/test_jacek_native_restart_round2.py
```
