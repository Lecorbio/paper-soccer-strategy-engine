# Mandatory-training Compact Value-BFM campaign

This campaign starts from main commit
`231b5ce0af9171670d68fe5fe022a67f86122cfd`. Its success condition is a newly
trained 6301→12→8→1 bias-free, signed-three-bit scalar candidate completing
90 clean live games, finishing calibration, and scoring strictly above
44.29750553418035. Report rank separately. Offline qualification alone does
not complete the campaign.

The original campaign and recovery evidence remain immutable. V2 has its own
contract, input copies, append-only hash-linked ledger, and per-command
receipts. Protected and live transcripts are not training inputs. The previous
corrected live reference is verified before its legal boundaries are reduced
to fingerprint-only exclusions. A verified operational ending before any legal
turn contributes the empty-board fingerprint only.

## Native data path

`tools/compact_value_bfm_campaign_v2.py` implements the fresh campaign data
adapter. The native continuation producer's opt-in
`--root-policy fresh-complete-v2` plays from the full supplied legal root.
A `-` transcript denotes the empty board in this mode only. The mode retains
short completed games. Default legacy generation is unchanged.

The frozen schedule matches both actor and root-depth marginals. A quarter of
games start empty, for training only. The remaining games start from freshly
rules-generated complete-turn roots with exactly 8, 12, or 20 drawn edges.
Fixed board boundaries do not count. Nonempty canonical roots are assigned
80/20 before continuation generation. Each native game binds the scheduled
root, seed, actor, complete prefix, terminal result, and exact producer/runtime.

Position selection uses absolute drawn-edge progress and tactical evidence,
with at most 20 samples per game. Before teacher search, a rules-only DFS
checks every legal complete-turn successor. A limit or forbidden intersection
rejects the whole group. Validation is processed first. The narrow overlap
exception applies independently to each train parent and successor at at most
six drawn edges, for prior-train/live overlap only. Protected, validation, and
development exclusions never receive that exception. Repeated train examples
are deduplicated; shared prefixes are not held-out evidence.

Teacher labels use the accepted large network at 64,000 fixed-work nodes.
The hardest quarter (rounded upward when necessary) is replaced at 500,000
nodes using teacher/student regret and action/Rank-4/terminal disagreement.
The existing rich-label validator and scalar CSR writer preserve mover-relative
values, exhaustive successor identities, proof evidence, and 75/25
search-backed/terminal targets.

## Execution and resume

Use the research Python environment and a Release native build containing
`papersoccer_jacek_replay_continuations`,
`papersoccer_jacek_replay_search_teacher`, and
`papersoccer_jacek_replay_rank4_position_teacher`.
All campaign outputs belong under the ignored `results/compact_value_bfm/`
directory. Commands require an explicit `--root`.

```sh
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN freeze \
  --previous PREVIOUS_CAMPAIGN --build NATIVE_BUILD
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN games \
  --phase smoke-064 --games 64 --workers 8
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN positions \
  --phase smoke-064
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN labels \
  --phase smoke-064 --workers 8
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN train-smoke \
  --phase smoke-064
```

Completed native commands are reused only when their command, input bindings,
and output hashes match. Unprotected producer work without a completion receipt
may restart deterministically; this policy does not apply to protected gates.
A nonblocking file lock excludes concurrent heavy campaign stages. The original
worker limits are eight for generation/labels, two for training seeds, and one
for uncontended timing. A separate explicit authorization can raise future
training and permit the bounded concurrent equivalence experiment described
below. Numerical runtimes use one thread each.

Smoke runs all three loss recipes with one seed and the actual warm-up plus
four-epoch QAT producer. Production pilot training requires all three approved
seeds. Smoke data and trained smoke candidates are ineligible for qualification.
Extend the smoke to 128 games if needed for 100 eligible validation groups;
insufficient early coverage must be reported, never scored as a passing gate.

## Qualification still required

The executable smoke path is an implementation checkpoint, not production
admission. Before pilot/full release, complete and verify:

- source-bound upload/maintained-engine comparison with identical weights and
  settings; their differing closure semantics mean this is not a pure speed A/B;
- the 2,000-game pilot, all nine training runs, canonical retention, overall and
  early regret/flip/coverage checks, and the 105/200 Rank-4 screen;
- admitted 10,000-game training and the six-opponent early/late state-capable
  evaluation suite, with frozen deployed controls and paired root bootstrap;
- uncontended profiling, individual search interventions and required strength,
  throughput, latency, and invariant checks;
- the fresh 1,000-game development gate, exact-source CI and preflights, two
  independent protected 1,000-game gates, and source-bound live evaluation.

Diagnostic uploads are already authorized when tied to a useful concrete
experiment and appropriate offline checks. Identical-source reuploads to chase
score variation are forbidden. No recurring automation is assumed.

## Lossless ranking storage

The native smoke exposed substantial rich-JSON memory overhead. The separate
`compact_value_bfm_ranking_store.py` adapter validates every original rich row
with the maintained validator, then stores all successor features, exact
teacher values, identities, canonical transcripts, visit counts, proofs, and
termination evidence in read-only memory-mapped arrays. Group metadata retains
source bindings and work budgets. Nothing is removed from exhaustive groups.

The trainer's source binding distinguishes the v2 storage-index schema from
the existing v1 aggregate schema. Existing v1 loading and validation retain
their original behavior. The runtime schema and network are unchanged. Tests
compare eager and mapped groups, scalar targets, losses, and gradients exactly;
altered cache bytes are rejected. The smoke's 341,685 successors occupy about
154 MB in this representation.

`compact_value_bfm_parallel_store_v2.py` is an explicit opt-in converter for
future, unstarted stores. The ordinary preparation/reconstruction path still
uses the serial builder; this tool does not replace or resume an active, partial,
or legacy store. It validates whole rich rows in spawned workers, then merges
in source, line, and successor order with exact offset rebasing and global
identity/duplicate checks. Fixtures establish byte-identical arrays and identical
index semantics apart from declared producer provenance and artifact paths.

After a corpus-specific memory review, invoke it from one immutable source:

```sh
python tools/compact_value_bfm_parallel_store_v2.py \
  --root ROOT --output ROOT/FRESH_STORE --bundle-manifest BUNDLE_MANIFEST \
  --source TRAIN_JSONL --source VALIDATION_JSONL_GZ --workers 4
```

The default is four workers, each with one numerical thread, and at most four
pending chunks. The builder owns the global heavy-stage lease and rejects mixed
source namespaces. Default limits are 8 MiB per ordinary chunk, 64 MiB per whole
group, and 128 MiB per converted shard. Oversized input fails closed without
splitting or dropping a group; an explicit larger limit requires operator review
and a fresh output. Interrupted claims cannot retry in place.

The worst-case pending scratch bound is 768 MiB at four workers and 1,536 MiB
at eight. It excludes merged arrays, the corpus-sized group index/seen set, and
parsed JSON expansion. Recorded worker peak RSS values are not simultaneous
resident-memory measurements. Neither worker count is approved for a real corpus
by the fixture tests: review actual memory for that corpus and worker count before
use, and measure eight-worker capacity separately. No real-corpus conversion or
speedup has been established by these tests.

`compact_value_bfm_state_adapter.cpp` provides direct state initialization for
all six local opponents. Empty-board games are explicitly routed to the process
referee. Compile the four `turn_action_v2` opponents with
`CAMPAIGN_OPPONENT_REPLAY_CORRECTION` to retain their source's replay-correction
check; other opponents use their normal complete-turn chooser. Qualification
uses 800/155 ms internal clocks. Reduced-clock adapter tests establish legal
state handling only.

## First native smoke outcome

The first 64-game smoke completed generation, independent isolation preflight,
large-teacher labeling, real training, export, and native checks. It retained
994 groups (829 train, 165 validation) and replaced 249 groups at the deeper
budget. All 64 generated trajectories were distinct; 18 short completed games
were retained. There were 148 comparable validation groups, including 12 early groups. That early sample is small, and the smoke
supports no pilot-admission conclusion.

Each loss recipe ran one float warm-up and four QAT epochs from the frozen
initialization. All three master tensors changed with finite nonzero norms.

| Ranking weight | Changed W1 codes | Changed W2 codes | Changed W3 codes | Canonical sign accuracy |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,070 | 0 | 1 | 85.68% |
| 0.10 | 1,274 | 1 | 1 | 85.39% |
| 0.25 | 937 | 1 | 1 | 85.48% |

Every export is 92,165 ASCII bytes, with 2,835 bytes spare. Exact exports passed
native unit tests and 4,096-state inference parity each. Six true empty-board
process-referee checks, covering both colors for all three exports, completed
without forfeits under 1,000/200 ms external deadlines.

All three smoke models failed canonical retention. None is qualified, no live
success is claimed, and these diagnostic models do not count as completed
production-attempt failures. The 2,000-game pilot must use fresh data, the
original float initialization, all three approved seeds, and its own admission
evidence. `compact_value_bfm_pilot_v2.py --root CAMPAIGN games` freezes that
pilot's context and runs the native generation stage while preserving smoke
exclusions and the campaign-global heavy-stage lock.

## Streaming pilot execution

`compact_value_bfm_stream_v2.py` checks positions in per-game process workers,
with a complete validation barrier before training-root checks. Validation
fingerprints use read-only binary indexes, and each completed game has an
immutable receipt and compressed successor census. The phase index retains
lightweight metadata rather than every successor dictionary.

`compact_value_bfm_labels_v2.py` runs source-bound native label chunks, uses the
maintained rich validators, checks the full preflight successor census, and
preserves canonical rich rows in deterministic gzip files. Source identities
are independently reconstructed in CMake's declared order. Hard selection is
global; incomplete shallow groups are mandatory deep replacements, and no
incomplete group can reach training. Final merging is streamed, and the mapped
ranking store accepts both plain and compressed rich input.

Pilot model selection is frozen before training. Seed selection reuses the
maintained validation key, and both overall and early strata must have at
least 100 comparable groups with at least 80% coverage. Each ranking candidate
must reduce regret by 10% in both strata, stay within the action-flip margin,
preserve canonical accuracy, and retain the source reserve. Selection remains
conditional on a separate 105/200, zero-failure Rank-4 screen.

`compact_value_bfm_pilot_selection_v2.py --root CAMPAIGN --context CONTEXT
--phase PHASE --wait-for-labels train` is a one-shot sequential continuation:
it waits on the shared campaign lock, verifies completed labels, runs all nine
trainings, and freezes offline model selection. It is not a recurring
monitor and does not claim pilot admission or live success.

The pilot screen bridge generates its bank only after offline model selection,
filters candidate roots against prior and current campaign fingerprints, and
uses the unchanged historical Rank-4 gate format. The exact source/runtime,
compiler, bank, command, native result, and color roster are bound together.
An interrupted claimed screen is spent, and offline selection alone cannot
admit a pilot. `--wait-for-selection UPSTREAM_PID` provides a one-shot dependency
wait that exits if upstream training stops without completing selection.

For multi-opponent qualification, the state adapter preserves each frozen
opponent's own source clock constants. Candidate clocks remain800/155 ms.
Reduced-clock structural checks remain explicitly ineligible as strength
measurements.

`compact_value_bfm_full_v2.py` scales only an actually admitted pilot. It reopens
the bound Rank-4 result, verifies the exact model payload and offline eligibility,
and requires the real 105/200, zero-failure screen. Full data generation uses the
admitted pilot student, fresh roots, and accumulated pilot fingerprints. Full
training keeps the original float initialization, the admitted ranking weight,
a scalar control, and all three seeds. The streamed 10,000-game workflow still
ends awaiting search/evaluation and final qualification; it does not authorize
live success from training alone.

## Source-bound continuation and evaluation

`compact_value_bfm_full_selection_v2.py` selects the scalar control and admitted
ranking recipe from all six completed full-round seeds. It reopens actual pilot
admission, original initialization, input audits, seed references, tensor updates,
and exact exports. Canonical retention and source reserve remain mandatory.
Full-round regret and flip measurements are diagnostics; the pilot admission
threshold is not imposed a second time. The result is
`CONTEXT/PHASE/full-model-selection.json`, not a strength qualification.

`compact_value_bfm_opponent_suite_v2.py` executes the paired six-opponent screen
or confirmation against the immutable deployed upload. Each opponent receives
32 screen pairs or 100 confirmation pairs, equally divided among 8, 12, 20, and
40 drawn-edge roots. Canonical states are distinct across opponents. Candidate
and control share every root and both colors; confirmation excludes screen roots
and all played continuation boundaries. Each suite binds its selection, bank,
compiler, complete source dependencies, binaries, commands, and raw games.
Independent replay checks winner, root progress, and both actors' clocks before
the existing paired bootstrap assessment. A four-worker execution limit and
180-second native process watchdog preserve partial evidence on failure. A
claimed interrupted suite is spent. Reduced-clock adapter checks are explicitly
ineligible as strength evidence.

The suite requires a validated `search/search-selection.json`, including the
source-specific ablation and profiling evidence, even when the baseline wins.
Search-strength trajectories from every variant are excluded before suite roots
are chosen. Resumed and completed banks recheck isolation against the immutable
corpus and all earlier played states.

The separate native `rank4_gate_trajectories.cpp` offers `--include-trajectories`.
The historical `rank4_gate.cpp` retains its exact release-pinned bytes. The new output
contains each actual root and accepted complete-turn trajectory, and a sidecar
preserves finished games if the process is interrupted. The maintained default
output is unchanged. V2 screens require independent trajectory validation and
carry every played postroot boundary, including terminal feature fingerprints,
into full training and subsequent attempts.

`compact_value_bfm_pilot_v2.py --root CAMPAIGN --attempt 2 games` may start a
second isolated pilot only after a verified completed rejection of the first
trained attempt. Pilot closure reopens all nine seed references, reproduces
canonical retention and early metrics, and carries exhaustive censuses and played
states. A full-round rejection additionally requires
`compact_value_bfm_full_outcome_v2.py --root CAMPAIGN --attempt 1 record`.
That bridge revalidates actual pilot admission, all six full-round seeds, and the
completed rejecting search, suite, or development evidence. Pilot plus full
training count as one attempt; both corpora and all unprotected trajectories are
carried forward. Incomplete stages never count as completed model failures.
Smoke does not count, and attempts after two still require the approved
unprotected attribution and intervention binding.

These commands resume from source-bound artifacts. Optional scheduling is a
separate user-authorized task follow-up; it does not alter the frozen experiment
policy or substitute for successful stage evidence.

## Search, development, and CI bridges

`compact_value_bfm_search_v2.py` freezes the maintained 2×2 feature-construction
and descendant-selection variants from the same full-trained payload. It runs
three sequential fixed-work/clocked profiling passes on a frozen unprotected root
corpus. Standard fixed-work traces, complete root-action legality, and full/delta
inference must agree. Each retained optimization needs at least 10% throughput
gain, at most 5% p95 regression, and independently supported actual-clock strength.
The historical upload/maintained-source comparison is not used as a speed A/B.

`compact_value_bfm_search_strength_v2.py` supplies the full-round actual-clock
comparison: all variants use one newly isolated 500-pair bank, one worker, explicit
shuffle seed1, and complete played trajectories. Native configuration is checked
before deployment of the runner; a mismatched seed cannot silently spend the
whole A/B batch. Claims bind compiler, source, bank, exact command and output paths.
Claimed or orphaned native output cannot be rerun automatically.

`compact_value_bfm_category_profile_v2.py` supplies native category attribution
after ordinary profiling, also available through the search command
`category-profile`. It instruments diagnostic derivatives of the four exact
sources and repeats every frozen fixed-work run. Seven exclusive categories
reconcile to the full instrumented interval: feature construction, first layer,
dense evaluation, complete-turn generation, tree traversal, state application,
and residual search. Derivative bytes, function anchors, and model identity are
reconstructed during validation. Decisions, complete traces and counters must
match the corresponding original run exactly.

The report includes observed instrumentation overhead and timing variation.
Instrumented times are attribution evidence only; the 10% throughput and 5% p95
gates continue to use original source measurements. Missing or invalid category
evidence keeps search assessment incomplete. Successful validation permits final
source selection without rewriting any earlier incomplete receipt.
Two fresh structural roots have verified native derivative/original trace,
decision, counter and inference parity. Campaign profiling still requires the
actual admitted full model and its complete frozen workload.
Optional cache/widening/reuse treatments also remain disabled for selection until
their separate source-bound native invariant evidence is integrated.

`compact_value_bfm_development_v2.py` follows actual passing screen and confirmation
results. It freezes 500 fresh canonical pairs, runs four disjoint 125-pair native
shards, and requires 550 wins, 265 per color, a paired 95% lower win-rate bound
above 50%, and zero failures. All played search, suite and development states
remain in fingerprint-only exclusions. A passing development result still awaits
exact-source preflight/CI and two independent protected gates.

`compact_value_bfm_ci_v2.py` binds a validated selected source to clean committed
bytes and an actual GitHub workflow run on the experimental branch. Its supported
publication paths are the maintained `compact_value_bfm/submission.cpp` and a
separate `compact_value_bfm/trained_v2_candidate.cpp`. The latter receives explicit
native executable, unit-test, feature-parity and inference-probe targets when
present. Harnesses are copied with only their standalone source include changed;
historical release-pinned harness bytes and deployment artifacts remain intact.
The helper verifies the committed CMake mapping and preserves the raw GitHub response. It
requires all five distinct jobs to succeed for the exact commit. An arbitrary
tracked C++ copy is insufficient, and green repository CI alone never qualifies
a candidate. Historical fixed-branch CI validators remain unchanged.

For an isolated local check of a source before publication, configure
`PAPERSOCCER_COMPACT_VALUE_BFM_RELEASE_SOURCE` to its absolute path. The normal
CI configuration uses `trained_v2_candidate.cpp` when that committed file exists.
The candidate targets have been exercised with a real smoke-trained source:
native unit checks, 4,096-state feature parity, and 4,096-state runtime inference
parity passed. These establish the execution path, not candidate qualification.

## Final release and live evidence

`compact_value_bfm_release_v2.py` stages only an actually development-qualified
source. Publication binds clean committed bytes and exact green CI. Its fresh
local preflight builds GCC, Clang and sanitizer panels, verifies native tests,
4,096-state feature/scalar/full-delta parity and decoded model identity, then
checks sequential timing and both-color empty-board process-referee games.
The resulting `release/freeze.json` binds the source, runtime, commit, CI,
development result, toolchain and complete preflight evidence.

`compact_value_bfm_protected_v2.py` materializes no protected roots before that
freeze validates. Each independent OS-generated 256-bit seed produces 500 pairs.
The two gates run sequentially, each as 100 five-pair shards on four single-thread
workers. Gate B excludes Gate A proposals and played boundaries irrespective of
the score. Each gate needs 527 wins, 260 per color and zero failures. Completed
shards revalidate before reuse; an incomplete claimed shard cannot retry.
Both release preflight games also contribute their initial, played and terminal
state fingerprints before protected roots are generated.

`compact_value_bfm_live_v2.py` provides the normal qualified-release path. It
creates a new v2 authorization and explicit identity-only compatibility records
for the maintained exact-90 collector, preserving every historical authorization.
An editor copyback must match the frozen source. `start-submit` revalidates actual
protected qualification and claims at most one UI submission for that source;
the command itself does not click Submit. After the UI action, attestation binds
the new server agent/submission/session to the same owner and rejects stale or
inconsistent observations. Ambiguity permits observation, never another click.

Start its durable `watch` command immediately after attestation. The first exact
90 matching game IDs are frozen before filtering any later metadata. Every game
and replay session must agree with that identity. Once collection completes,
the same watcher immediately captures calibration, polls while it remains
incomplete, and runs the final source-bound assessment under the campaign lease.
Its timeout covers collection and calibration together. A timed-out run resumes
the same frozen window; completed score observations are never refreshed.
The final check requires 90
operationally clean games and completed calibration (`percentage=100`,
`inProgress=false`). Opponent forfeits are not clean strength evidence. Rank is
reported separately.

Public leaderboard scores are rounded. The current adapter uses a conservative
plus/minus 0.01 interval and requires its lower bound to exceed
44.29750553418035. Values too close to resolve, such as a displayed 44.30, remain
precision-inconclusive pending authoritative finer-precision evidence; they do
not finalize success or failure. The first completed-calibration observation is
immutable, so later favorable score fluctuations cannot replace it. Useful
diagnostic uploads retain their existing user authorization, but their separate
experiment-specific adapter is not supplied by this qualified-release CLI.

After two verified completed unsuccessful trained attempts,
`compact_value_bfm_attribution_v2.py` freezes only unprotected diagnostics and one
recommended intervention category. Pilot and full training count together. It
does not itself launch attempt three. `compact_value_bfm_intervention_v2.py`
can bind the existing `refined-adaptive-scales-v1` recipe only when the verified
recommendation is QAT/scales. A fresh third pilot must carry both failed attempts'
exclusions and retain the original float initialization, teacher, architecture,
one warmup epoch and four QAT epochs. Any admitted full phase inherits the same
exact recipe. Teacher-ranking and search recommendations still require their
own concrete experiment integration; attempt four remains unavailable.

`compact_value_bfm_terminal_outcome_v2.py` closes a completed protected rejection
or a completed calibrated live rejection. It keeps terminal proof separate from
the unprotected metrics used for attribution, and carries fingerprints from the
protected proposals, played games, release preflight and live replays. An
interrupted gate, ambiguous submission or clean window with an inconclusive
score cannot count as a completed failed attempt. A completed calibrated window
with operational failures already misses the clean-game requirement and can
close irrespective of score precision.

Training backpropagation skips first-layer scatter rows whose computed gradient
is exactly zero. Dense reductions and the order and multiplicity of every
retained scatter remain unchanged. Bitwise gradient and optimizer checks cover
signed zeros, duplicate features and float/QAT batches. Existing active training
continues from its immutable earlier snapshot.

Future training phases may explicitly freeze `training_executor` as
`{"mode":"spawn-v2","maximum_workers":2}` through pilot preparation's
`--training-executor spawn-v2` option. An already frozen phase cannot change
executor on resume. This uses two persistent spawned
processes, each with one seed thread and a separately enforced numerical thread
limit. The coordinator retains exports and aggregate receipts. Worker input
identities and seed bindings must reproduce the audited training corpus and
anchor filtering. Standard phases keep the existing thread executor. Production
activation still requires a real smoke equivalence run and memory/performance
measurement when campaign training slots are free; synthetic checks alone do
not establish production equivalence or a speedup.

`compact_value_bfm_seed_process_check_v2.py prepare --root ROOT --output OUTPUT`
freezes that experiment without reconstructing the large core datasets or
starting training. `OUTPUT` must be a fresh child of
`ROOT/diagnostics/seed-process-equivalence`. Prepare and execute from the same
immutable source snapshot; `inspect --plan OUTPUT/plan.json` verifies the plan.
The harness preserves the historical smoke's separate prepared and retained
position files and reproduces its exact input binding.

Once the campaign lease is free, `run --plan OUTPUT/plan.json` trains all three
recipes with two seeds each, first through threads and then through two spawned
processes. Those stages run sequentially and compare complete deterministic
receipts, scale trials, metrics, float checkpoints, runtimes and source exports.
For the first seed of each recipe, the new thread execution must also match the
original completed smoke receipt and artifacts. Historical results supply that
comparison baseline while both new executor stages perform fresh training.
The report includes process-tree memory samples, per-process resource records
and elapsed times. Its single fixed order is explicit; an observed elapsed ratio
does not establish an order-independent speedup. Both children and complete
memory observations are required before considering production activation.
No result qualifies a model or enables the production executor automatically.

An explicit user authorization can raise future training to four spawned seed
workers without changing the historical campaign policy or existing phase
receipts. New pilot and full contexts select this with
`--training-executor spawn-v2 --training-workers 4` and
`--training-resource-authorization AUTHORIZATION`. Each process keeps one
numerical thread. The four-worker scheduler dispatches the complete recipe/seed
roster together, then collects receipts and exports in the frozen roster order.
Full training may select four workers after a pilot trained with the original
executor. Existing contexts reject an executor or resource change on resume.

`compact_value_bfm_training_acceleration_v2.py prepare` binds a separate
four-worker equivalence experiment to the explicit authorization, immutable
source snapshot and exact identities of the running pilot and its two unspent
continuation waiters. Preparation starts no training. Retire those waiters
before running the prepared context: timed matches, qualification and full
training must remain paused throughout validation. The controller uses a
separate training-validation lock and supervises only its own child process
group; it preserves the existing pilot and stops validation if its guards fail.
The temporary ceiling is two existing pilot workers plus four validation
workers, each with one numerical thread.

The four-worker experiment runs all three fixed seeds for each of the three
recipes through fresh thread and spawn stages, retaining the three historical
first-seed comparisons. Concurrent elapsed times are explicitly contended and
cannot establish a speedup. Failed or interrupted stages keep their claims and
cannot silently retry. After verified equivalence, restore source-bound
campaign continuation and select four workers only for newly frozen training
contexts. The campaign operator or heartbeat enforces that activation review;
the resource authorization alone records worker capacity and does not certify
equivalence. Representative full-size memory observations remain necessary; this
smoke experiment does not establish full-size memory headroom or qualify a
candidate.

Prepare an actual corpus for a capacity check before starting any seeds:

```sh
python tools/compact_value_bfm_campaign_v2.py --root PHASE_CONTEXT \
  prepare-training-inputs --phase PHASE
```

This uses the ordinary campaign lease and the same preparation helper as
training. It builds or verifies the ranking store, deduplicated scalar shards,
anchor filtering and sealed input audit. It does not load model initialization,
measure baseline models or train seeds. Preparation-only samples and deduplication
sets are released before subsequent seed work; array contents and ordering stay
unchanged.

From one immutable source snapshot, use
`compact_value_bfm_training_capacity_v2.py prepare --root ROOT --context
PHASE_CONTEXT --phase PHASE --output OUTPUT` to freeze a load-only measurement
under `ROOT/diagnostics/training-capacity`. Preparation binds the already audited
corpus, source files, Python runtime and explicit four-worker authorization.
`run --plan OUTPUT/plan.json` requires the ordinary heavy-stage lease. It holds
the coordinator's reconstructed inputs and four independent spawned children's
inputs together while collecting process memory observations. Numerical runtimes
remain limited to one thread. No seed, optimizer step or candidate artifact is
created; a spent execution cannot silently retry.

The report distinguishes mapped file extent from sampled resident memory.
Mappings are not pinned or guaranteed fully prefaulted, and shared pages may
appear in several RSS totals. This load-only probe does not measure preparation
temporaries or peak training-batch workspaces. Review those additional costs for
the actual corpus before enabling four-worker production; the report never
grants automatic activation or model qualification.

For a new position-preflight plan, `compact_value_bfm_stream_v2.py positions`
accepts `--packed-base-exclusions`. This builds shared, read-only indexes for
the prior exclusion history before starting workers. Each role/domain retains
its full 256-bit fingerprint union and its original iteration order, preserving
rejection reasons, early-training exceptions and whole-group filtering. Workers
map the binary arrays instead of rebuilding large Python sets of strings.

The new position plan binds the index, ordered source artifacts, producer and
array hashes. The validation barrier remains separate and unchanged. A frozen
plan cannot change exclusion mode or producer on resume; active historical
plans must continue with their original source. The opt-in does not alter
teacher labels, training math, clocks or qualification thresholds.

When load-only measurements leave batch memory uncertain,
`compact_value_bfm_training_workspace_v2.py prepare --capacity-plan CAPACITY_PLAN
--output OUTPUT` prepares a separate allocation probe. It uses the exact engine
source bound by the completed capacity plan, with the newer probe source bound
separately. The current probe supports the frozen `fa012e7` engine and rejects
other engine versions. Run and validate it through its standalone CLI so an already
loaded, different `tools` package cannot substitute for the frozen engine.

The probe reconstructs the actual corpus and holds four workers' real mapped
views and forward/backward buffers together using the existing capacity controls.
It uses the original checkpoint and allocation-only gradients, starts no
optimizer steps and writes no trained candidates. A scoped return hook may
retain named backward arrays beyond their normal lifetime without modifying the
numerical function. The report distinguishes that conservative retained overlap,
individual process peaks, expression temporaries and file pages that were not
touched. Such measurements inform resource review; they do not automatically
authorize production or establish model quality.

## Reusing float decisions during scale search

Scale selection evaluates many quantized candidates against the same float
parameters. The trainer can reuse each validation group's exact float-best
successor within one initial or adaptive scale-search invocation. Parent-frame
conversion and successor-ID tie-breaking remain unchanged. Each invocation gets
a fresh private cache; parameter contents, architecture, group order and backing
arrays are checked before reuse. Mutable eager groups receive content checks.
The cache retains decisions and small guards, not feature or prediction tensors.
Quantized inference, teacher comparisons, reductions, candidate order, optimizer
updates and numerical receipt fields retain their existing behavior.

`compact_value_bfm_validation_cache_check_v2.py` supplies the separate real-data
equivalence gate. Its `prepare` command accepts `--root`, an explicit immutable
`--candidate-snapshot`, a fresh `--output` below
`ROOT/diagnostics/validation-cache-equivalence`, and `--after-driver-launch` for
the third-pilot data/workspace driver. Preparation binds the original native
64-game smoke inputs without starting training. The reference engine remains
the frozen `fa012e7` snapshot.

After that data/workspace driver has completed and exited, `run --plan PLAN`
holds the ordinary heavy-stage lease and executes fresh reference and candidate
cohorts sequentially. Each uses at most four spawned processes, one numerical
thread per seed, the refined adaptive profile, all three ranking weights and all
three fixed seeds. Every complete training binding, deterministic receipt,
trial report, checkpoint, runtime and generated source must match. A separately
bound read-only observer must also show that the mapped cache was used and
repeated float-ranking forwards decreased while other forward work stayed equal.
Incomplete claimed runs cannot silently retry. These checks provide numerical
equivalence evidence; they neither claim a wall-time speedup nor activate or
qualify a production model. Existing active snapshots remain immutable.
