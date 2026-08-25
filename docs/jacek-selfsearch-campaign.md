# Jacek replay BFM self-search campaign

This noncanonical campaign tests whether a frozen replay-BFM model can improve
its successor by guiding a substantially deeper fixed-work search.  Raw model
predictions are never copied as labels.  The search teacher backs values up
through complete-turn minimax, records explicit proof status, and blends the
result with the terminal outcome using the existing 75/25 target policy.

The workflow starts only after the 5,000-game R0/R1/R2 evaluation and its
uncontended latency audit validate.  It selects the game-strongest round,
freezes that runtime as the actor/teacher, and first runs a matched pilot:

- 2,000 successful league games and at most 24 suffix positions per game;
- identical position IDs labelled by both incumbent-guided BFM and Rank-4;
- paired shallow labels and paired deep relabeling of the hardest 25%;
- search-label and Rank-4-label `6301 -> 192 -> 32 -> 1` students trained from
  the same seeds, feature rows, canonical anchors, and common adjudicator;
- fresh games against each other, the incumbent, Rank-4, and `jace_nn`.

A failed pilot stops.  A passing search-label student becomes the actor and
teacher for one 10,000-game iteration with at most 20 positions per game.
Final publication remains local and requires fresh 980 ms gates; no external
upload or replacement of Rank-4 is automatic.

## Data contracts

Frozen position TSV columns are:

```text
position_id  root_group_id  group_id  source  split  winner  mover  prefix
```

`papersoccer.jacek-replay-search-teacher.v2` stores the direct mover-relative
`teacher_value`, the exact model/source/configuration identity, full search
statistics, `root_solved`, and an optional absolute `proven_winner`.  Unsolved
values must be finite in `[-1,1]`; solved values are emitted as exactly `+1`
or `-1`.  The corpus parser deliberately rejects applying Rank-4's
`tanh(root_score/12000)` transform to these values.

Version 2 records the explicit search termination reason, progressive-widening
count, every complete-turn generation stop class, frontier health counters,
and the maximum sampled frontier width.  Accepted rows must terminate with
`root-solved` or `fixed-work-cap`; deadline and prematurely closed-frontier
counters fail closed.  The teacher's separate `--audit-terminations` mode
emits only `papersoccer.jacek-replay-search-termination-audit.v1` diagnostics
and cannot be consumed as teacher labels.

Every generated game, position, label chunk, merge, pack, seed checkpoint,
gate, and publication has an atomic receipt.  A receipt binds producer and
model hashes, configuration, inputs, outputs, counts, and deterministic chunk
identity.  Resume reuses only evidence that revalidates recursively.

The common adjudicator excludes every canonical fingerprint already present in
the cumulative R0/R1/R2 shards, prior self-search shards, or the current train
split before choosing the lowest-hash validation IDs.  New packs also consume
all earlier train/validation/test manifests as split-isolation evidence.  This
prevents a rotated or reflected training state from reappearing in model
selection.

Training anchors are the three distinct R0, R1, and R2 train shards—not merely
the latest round's shard.  The regression gate likewise combines all three
canonical validation shards.  Their nine-manifest ancestry must exactly match
the source-shard list in the selected canonical Round-2 model.

## Resource policy

- Fixed-work generation and labels: ten workers.
- Training: two seeds concurrently.
- Timed strength games: four calibrated workers.
- Authoritative latency: one uncontended worker.
- Required launch health: AC power and at least 12 GiB free disk.

Strength panels are split into deterministic five-pair shards and scheduled
from one global four-worker queue.  This keeps calibrated concurrency occupied
as opponents finish at different rates and gives every shard an independent
resume receipt.  Merged panel reports are ordered by pair offset.  The latency
audit remains sequential.

The pilot is expected to take roughly 13–21 hours after producers freeze.  A
passing pilot followed by the approximately 200,000-position full iteration
is expected to add roughly 30–45 hours.

## Frozen launch

The campaign refuses binaries that are not from one clean, unsanitized Release
build of the requested commit.  After the full test matrix passes, create the
build receipt with:

```text
python tools/jacek_selfsearch_workflow.py write-build-manifest \
  --repository REPOSITORY --expected-commit COMMIT \
  --continuation-generator BUILD/papersoccer_jacek_replay_continuations \
  --search-teacher BUILD/papersoccer_jacek_replay_search_teacher \
  --rank4-teacher BUILD/papersoccer_jacek_replay_rank4_position_teacher \
  --comparison BUILD/papersoccer_jacek_replay_comparison \
  --pack-tool REPOSITORY/tools/jacek_replay_pack.py \
  --trainer REPOSITORY/tools/jacek_replay_train.py \
  --output CAMPAIGN/release-build.json
```

The `run` command takes the same producer paths plus
`--build-manifest CAMPAIGN/release-build.json`.  It rechecks the commit, CMake
cache, executable hashes, embedded source closures, prerequisite evaluation,
canonical ancestry, power, disk, and all previously completed stage artifacts
before and after each stage.  The supervisor lock is inherited by child
processes, so an orphan worker prevents a duplicate supervisor from starting.
