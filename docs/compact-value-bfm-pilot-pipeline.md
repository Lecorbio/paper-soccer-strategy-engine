# Compact Value-BFM teacher-phase pipeline

`tools/compact_value_bfm_pilot_pipeline.py` adapts a frozen
Rank-4-teacher-challenger **pilot or full** phase to the existing continuation
and teacher executables. It generates data and labels only: it never trains,
opens a protected test split, runs a gate, or uploads.

## Frozen preparation

`prepare` validates the campaign and phase references and derives the campaign
ID, attempt, phase, game count, quotas, and three-column
`game_ordinal`, `actor_mode`, `base_seed` schedule from them. The caller passes
a base output directory; artifacts are routed below
`attempt-NNN/pilot` or `attempt-NNN/full`, so receipts from the two phases
cannot collide.

Student/prior runtimes, the initial checkpoint, canonical roots, producer
binaries, build manifest, and dynamic exclusions come exclusively from the
sealed phase plan. Optional CLI paths only assert equality; they cannot replace
a frozen artifact. A full phase uses the admitted pilot runtime as its student.

Preparation also resolves the phase build manifest in either initial-bundle
`route` form or later-attempt ledger `path` form. It requires the current Git
HEAD to equal that manifest's clean commit, compares every Python dependency
and every native-producer source in `PIPELINE_REQUIRED_BUILD_SOURCES` to the
manifest, and compares all four executable producer hashes to the same build.
The normalized manifest, source subset, producer records, and a closure hash
are stored in the pipeline plan. Every load and resume recomputes this check;
a tool edit cannot become self-consistent merely by being recorded in a new
pipeline receipt.

The roots manifest is authoritative before any game is generated. Test and
protected roots are removed, while train/validation assignments remain frozen
at whole-root granularity. Preparation also freezes the student, prior
student, accepted `6301x192x32x1` teacher, maintained Rank-4 source, producer
binaries, prior CSR shards, and original compact training-bundle manifest.
Rich action labels bind `source_bundle_body_sha256` to that original compact
training bundle (the identity required by `compact_value_bfm_train`); the
challenger input-document identity is recorded separately.

The native continuation generator accepts compact runtimes only when every row
has a student actor. Because each governance logical shard mixes student and
accepted-teacher games, `run-games` executes each of the ten logical shards as
two schedule-preserving profiles:

- compact student/self-play/Rank-4/prior-incumbent rows, with both compact
  runtimes supplied;
- accepted-teacher-vs-Rank-4 rows, without compact runtime flags.

The profiles merge back into exact global game-ordinal order. At most eight
workers run concurrently, and every worker receives one-thread BLAS/OpenMP
settings.

Before a produced game can inherit a root's split, the adapter rederives its
attempt seed and native game ID, verifies the manifest's root row and root
transcript hash, and checks that the exact frozen root transcript is the game
prefix. A claimed root ID therefore cannot move a continuation across the
whole-root train/validation boundary.

## Fingerprint-only exclusions

Runtime filtering accepts only self-contained fingerprint artifacts. In
particular, the governance `mixed-six` evidence document is not an executable
exclusion source because it contains paths back into the recovery worktree.
Before freezing a campaign, materialize its six explicit unprotected bank
copies into one standalone artifact:

```sh
python3 tools/compact_value_bfm_pilot_pipeline.py build-fingerprint-set \
  --classification mixed-development \
  --output-directory /absolute/frozen/fingerprints \
  --source /absolute/copy/model-screen.opening-bank.json \
  --source /absolute/copy/tuple-screen.opening-bank.json \
  --source /absolute/copy/tuple-confirmation.opening-bank.json \
  --source /absolute/copy/profile-screen.opening-bank.json \
  --source /absolute/copy/profile-confirmation.opening-bank.json \
  --source /absolute/copy/actual-clock.opening-bank.json
```

Opening-bank inputs are replayed and converted to canonical sparse-feature
hashes; their older state-serialization hashes are not mixed with that domain.
The command also accepts validated historical replay-BFM exclusion TSVs for
the mixed-development set, so all earlier development openings remain excluded
without following paths back into an old worktree.
The same command accepts standard unprotected CSR shard manifests for
`prior-train` and `prior-validation`; it loads and validates the adjacent NPZ
and hashes each row's four-way canonical active-feature representation. The
result contains sorted hashes only and stays valid after its source copies are
removed. Freeze these artifacts under the exact input names
`mixed-development-fingerprints`, `prior-train-fingerprints`, and
`prior-validation-fingerprints`. A protected canonical-fingerprint artifact
and every frozen live fingerprint artifact are also mandatory inputs when
applicable.

Before labeling, each candidate position is canonicalized across exact,
rotate, reflect, and rotate-reflect states in both sparse-feature and opening
state-serialization domains. It is rejected on intersection with any
prior-train, prior-validation, mixed-development, protected, live, or already
selected position hash. No protected label, metric, or transcript is read.
Fingerprint-only exclusions accumulated after a failed final gate or rejected
live window are required by the phase reference and enter this same fail-closed
state-domain audit; the protected source artifacts themselves never enter the
pipeline.

## Resumable stages and outputs

The execution order is:

1. `run-games`
2. `freeze-positions`
3. `run-shallow-labels`
4. `run-rank4-labels`
5. `select-hard`
6. `run-deep-labels`
7. `finalize`

The combined `run` command performs the same sequence. `--resume` reuses only
phase-bound receipts whose inputs and outputs remain byte-identical. It also
revalidates the frozen build/source closure before reading any stage receipt.

Production uses the single campaign-derived base
`CAMPAIGN/phase-executions`, deriving `attempt-NNN/{pilot,full}` beneath it.
Every heavy stage holds the shared nonblocking lease
`CAMPAIGN/.rank4-teacher-heavy-stage.lock`, and receipts bind that path and the
frozen source closure through the execution authority. Alternate roots and all
injected contexts, producers, predictors, or packers are rejected in
production; nonproduction tests must authorize such hooks explicitly.

Each retained game contributes equal-sized, disjoint opening, tactical, late,
and decisive strata, never more than 20 positions. Tactical states are ranked
by immediate goal, rebound, constraint, and target-goal-distance evidence;
decisive states are within the terminal horizon of the generated game. All
positions receive a 64k fixed-work complete-turn action group and an independent
maintained-Rank-4 label. Hard selection is global across the phase, with every
non-exhaustive shallow group first and then teacher/student action regret,
action, Rank-4, and terminal-outcome disagreement. Standard phases replace the
hardest 25% at 500k nodes. The attribution-only `hardest-5pct-2m-v1` profile
requires the full 20-position roster and replaces the hardest 5% at two million
nodes.

Action-group analysis has an offline-only exhaustive-root mode: it drains the
complete-turn generator in deterministic 250-action pages before ordinary tree
descent, so a root wider than deployed search still publishes every canonical
successor with a value backed up by the same fixed-work search. Deployed and
game-generating searches retain their 250-action cap. If the fixed node budget
cannot even hold the exhaustive root set, the row remains non-exhaustive and
finalization rejects it.

Finalization emits:

- a content-addressed rich complete-turn successor-label document directly
  accepted by the compact trainer's ranking-label validator;
- ordinary scalar targets using 75% search-backed root value and 25% terminal
  outcome;
- separate standard train and validation
  `papersoccer.jacek-replay-csr-shard.v1` NPZ/manifests.

No fake test shard is created. Existing canonical anchor shards remain
separate, and canonical row intersections against prior train or validation
anchors fail closed rather than being blended into the new corpus.

Use `--help` on `build-fingerprint-set`, `prepare`, or a stage for exact CLI
arguments. Production commands require the continuation generator,
complete-turn action teacher, and Rank-4 position teacher built from the same
checkout.
