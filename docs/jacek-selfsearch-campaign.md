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

`papersoccer.jacek-replay-search-teacher.v4` stores the direct mover-relative
`teacher_value`, the exact model/source/configuration identity, full search
statistics, `root_solved`, and an optional absolute `proven_winner`.  Unsolved
values must be finite in `[-1,1]`; solved values are emitted as exactly `+1`
or `-1`.  The corpus parser deliberately rejects applying Rank-4's
`tanh(root_score/12000)` transform to these values.

`papersoccer.jacek-replay-teacher.v3` stores the self-search Rank-4
fixed-work result without changing the canonical Rank-4 engine. A row may
finish inside depth one only when it consumed the exact node cap and completed
at least one searched root action. Every row records `root-solved` or
`fixed-work-cap`, distinguishes node and depth caps, and rejects deadline,
early-stop, zero-action, missing-lineage, and unsupported-proof results. The
historical v1 Rank-4 schema retains its positive completed-depth requirement.

Both fixed-work teachers record `max_time_ms: 0`: wall-clock time cannot alter
a training label. A separate 900-second no-output watchdog supervises each
streaming chunk process. If a producer hangs or crashes, its temporary output
is deleted and no receipt is written; the watchdog never turns partial work
into a label.

Version 4 records the explicit search termination reason, resumable-frontier
and progressive-widening counts, every complete-turn generation stop class,
frontier health counters, and the maximum sampled frontier width. Complete-turn
pages preserve a bounded deterministic DFS cursor across both action and
partial-work caps; no partial turn is dropped or replayed. Accepted rows must
terminate with `root-solved` or `fixed-work-cap`; deadline and prematurely
closed-frontier counters fail closed. The teacher's separate
`--audit-terminations` mode
emits only `papersoccer.jacek-replay-search-termination-audit.v2` diagnostics
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
the latest round's shard.  Retention selection combines all three canonical
validation shards, while the post-selection regression gate combines all three
canonical test shards.  Their nine-manifest ancestry must exactly match the
source-shard list in the selected canonical Round-2 model.

Retention-safe mixed training starts both matched student arms from the exact
frozen phase actor; the three seeds change deterministic row order, never the
initial weights.  Every 256-row batch contains 64 new rows and 192 canonical
anchor rows in both pilot and full phases.  New and anchor weighted-Huber
losses are normalized separately and combined with fixed 0.25/0.75
coefficients.  The learning rate is capped at 6e-5 for at most 50,000 packed
new train rows and scales inversely with the actual cumulative new-train
optimizer-step count above that.  This keeps the first-pass AdamW update and
decoupled weight-decay budgets stable when the full phase has several times
more rows, while still consuming every new row before its first selectable
epoch.  The exact base, reference count, actual post-dedup count, scale,
effective rate, and both step counts are receipt-bound and must match between
the search and Rank-4 arms.  The new stream receives a complete fresh
permutation each epoch, while the anchor cursor continues across epochs without
repeating before a full permutation is consumed.

Checkpoint selection has two independent validation objectives.  The common
phase adjudicator (4,000 rows in the pilot and 8,000 in the full phase) must
improve over epoch zero, and the three canonical validation shards must remain
within 0.5 percentage points of the actor's sign accuracy and 2% of its
weighted-Huber error.  An early qualifying checkpoint may be preserved, but
training and patience cannot stop before one complete anchor pass.  If no
trained checkpoint qualifies, the exact actor bytes are retained.  Canonical
test shards are never exposed to training or seed/epoch selection; stage 17
uses their 121,052 rows for the post-selection retention gate.  The full
decision applies noninferiority against both the pilot actor and the original
campaign incumbent, preventing two phases from compounding their allowed
slack.

`tools/jacek_replay_retention.py` additionally provides a training-incompatible
blind-holdout format for procedural post-selection audits.  It freezes whole
root groups before model selection, opens only strict Rank-4 400k fixed-work
labels afterwards, and requires both point and root-cluster noninferiority.
The standalone holdout strengthens evidence; it is never an input to the
trainer and does not replace the canonical test or game gates.

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
