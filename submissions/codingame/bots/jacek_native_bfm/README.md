# Native Jacek-style best-first minimax candidate

`jacek_native_bfm` is a clean challenger built around complete-turn generation,
best-first minimax search with UCT-style allocation, and a compact neural value
function. It is intentionally independent of every incumbent bot, replay table,
and historical action label in this repository. The established `rank_4` bot is
used only as an external opponent in `comparison_gate.cpp`.

The checkpoint, training, clock, verification, and live-gate ledger is retained
in this README and the adjacent [experiment log](EXPERIMENTS.md).
The round-two checkpoint is a reproducible research candidate. Seed `20260822`
passed the previous-native-champion gates, was activated from an immutable
deployment descriptor, and completed an exact-source 90-game CodinGame
diagnostic as history version `62`. The raw result was 63-27 and the
forfeit-clean result was 44-27, with zero candidate operational failures. It
finished 0-9 in clean games against jacek. Rank 5 and the aggregate clean score
are exploratory evidence, not a strength, promotion, superiority, or Rank 4
parity claim. The prior history-61 round-one upload remains frozen below as an
explicit historical baseline.

This is an auditable adaptation of public ideas, not a claim that Jacek
Dermont's unpublished CodinGame source, weights, training corpus, or exact
competition settings have been reproduced. The public reference is
[QtPaperSoccer commit
`366d5304c09c2c820bd3ef4ea94624c034b8d955`](https://github.com/jdermont/QtPaperSoccer/tree/366d5304c09c2c820bd3ef4ea94624c034b8d955),
dated 2026-03-08. The accompanying public explanations are [Best-First
Minimax Search with UCT](https://www.codingame.com/playgrounds/55004/best-first-minimax-search-with-uct)
and [Paper Soccer neural
inputs](https://www.codingame.com/playgrounds/157341/inputs-for-neural-networks-for-the-board-games/paper-soccer).

## Provenance contract

| Decision | Direct user contract | Public article | Pinned public code | Candidate choice |
| --- | --- | --- | --- | --- |
| Search unit | Search complete turns | Describes the evaluator inputs, not action generation | Emits complete paths ending at a goal, block, or handoff | One node is one legal, rebound-complete turn |
| Candidate cap | At most 250 actions | Not disclosed | `moveLimit = 250` | Hard cap of 250 generated actions per expanded node |
| Deque schedule | Nine LIFO pops, then one FIFO pop | Not disclosed | Fifteen LIFO pops, then one FIFO pop | **9:1**, deliberately different from the pinned desktop code |
| Duplicate actions | Deduplicate the full resulting boundary state | Not disclosed | Suppresses path cycles and repeated blocked paths, but does not disclose boundary-state deduplication | Keep one path per full handoff `PositionKey`; the final ball vertex alone is insufficient |
| Neighbor order | Deterministic tests and reproducible training | Not disclosed | Shuffles neighbors | Seeded shuffling, with the seed explicit in fixed-work tests |
| Search family | Native best-first minimax, not the incumbent search | Explains BFM/UCT generally | BFM tree allocation and player-relative minimax backup | Single-thread BFM/UCT adapted to the CodinGame clock |
| Neural input | Follow the disclosed CodinGame representation | 316 used-edge flags followed by 105 eight-way true-turn-distance buckets; rotate Player 2 by 180 degrees | Contains multiple newer desktop schemas; it does not identify the unpublished CodinGame checkpoint | Exact 1,156-input article schema, frozen by integration tests |
| Neural shape | Train a new native model | Discloses `1156 -> 32 -> 32 -> 1` for the CodinGame bot | Public `NetworkDeep` supplies sparse evaluation and partial reevaluation machinery | Bias-free `1156 -> 32 -> 32 -> 1`, independently trained; artifact schema and hash are recorded below |
| Runtime | Judge by CodinGame time, not node count | Article reports about 200 ms on one thread | Desktop application is threaded and is not a CodinGame timing contract | One thread; 800 ms first decision and 155 ms later decisions, with construction-inclusive 900/180 ms pre-upload ceilings |
| Comparison | Rank 4 is an opponent only | Not applicable | Not applicable | No incumbent source, model, replay, or labels enter the candidate artifact |

The 9:1 schedule and full-boundary deduplication are direct requirements for
this experiment. They must not be described as behavior copied from the pinned
QtPaperSoccer revision.

The exact deployed search profile is also frozen: 250 retained actions, 4,000
root partial paths, 512 non-root partial paths, 80,000 tree nodes, a maximum of
2,000,000 expansions, `C = 0.95`, `FPU = 0.5`, and final ordering by
`value + log(visits)`. The full 50,000 partial-path and 120,000-node ceilings
are diagnostic limits, not upload defaults. `comparison_gate --equal-clock`
uses the deployed 4,000/512/80,000 profile; larger limits require explicit
flags and cannot be reported as evidence for the uploaded bot.

## Bounded tactical witness pass

Before the ordinary 9:1 sampler, each expanded boundary runs a separate FIFO
pass over at most 64 partial paths. It keeps the lexicographically smallest
complete-turn witness it discovers for each of `ForcedCutoff`, `SafeHandoff`,
and `OpponentImmediateGoal`. Discovered witnesses are exact classifications of
their resulting boundary states and enter the same tactical, cap-prioritized
top-k as other actions. Separate exhaustive rebound-component reachability
retains an attacking-goal path and an own-goal path when either is reachable.

This is a bounded discovery guarantee, not an exhaustive claim for the three
witness classes. If the 64-path queue or the decision deadline truncates the
pass, an undiscovered class may still exist. The native harnesses expose this
explicitly as `tactical_proof_paths`, `tactical_classes_found`, and
`tactical_proof_truncations`; a found witness remains exact even when the pass
reports truncation.

## Frozen neural input order

The input vector is mover-relative. Player 1 uses the board's canonical
orientation. Player 2 is rotated 180 degrees before either feature family is
indexed.

1. Indices `0..315` are the 316 canonical undirected playable-edge flags in
   the order emitted by the candidate's checked geometry table. A value is one
   exactly when that edge is already used.
2. Indices `316..1155` are 105 consecutive blocks of eight. Board vertices
   use the same canonical vertex order as the candidate's checked geometry
   table. Within vertex block `v`, bucket `d` is at `316 + 8*v + d` and is one
   exactly when the true-turn distance from the ball is `d`; distances at or
   beyond seven use bucket seven.
3. No incumbent score, replay identity, move history, player name, or arena
   result is an input.

`submission_test.cpp` freezes the edge table, vertex rotation, feature-family
boundaries, one-hot distance layout, and Player 1/Player 2 rotational
equivalence. Changing the ordering is a model-format change, not a harmless
refactor.

## Model artifact

The checked model header is generated from a versioned JSON artifact. Its
metadata must declare the input schema, dimensions, activation family,
quantization/storage format, training seed, and source JSON SHA-256. Generation
fails closed when those declarations disagree with the engine. The retained
artifact hash and final generated-submission hash are added only after the
reproducible training and source-generation checks have passed; see
`EXPERIMENTS.md` for the evidence record.

The frozen feature-schema identifier is
`canonical-edges316-onehot-true-turn-distance105x8-v1`. The model schema is
`jacek_native_model/v1`. It has no biases. Hidden layer one uses `x*x` for
nonnegative preactivations and `0.01*x` for negative preactivations; hidden
layer two uses `x` and `0.01*x`; the output uses `tanh`. Weights are stored in
row-major `w1`, `w2`, `w3` order with symmetric per-layer 3-bit quantization
and signed two's-complement values packed least-significant-bit first. These
are artifact-format requirements and are checked before header generation.

Round-two training keeps that runtime format but no longer lets one max-abs
outlier choose a layer's only scale. It forms deterministic lower-rank
percentile candidates through `p995`, chooses the three scales by two-pass
held-out coordinate search on the exact combined target, and freezes those
scales during QAT. The selected tensors are normalized to exact float32
`q * scale` values and must survive the existing max-derived exporter
byte-for-byte before they can enter a checkpoint. Per-epoch evidence includes
the old dynamic-max baseline, scale and W1 row/level coverage, plus turn-binned
prediction standard deviation, range and quantiles. Exact solved targets are
also honored during checkpoint selection rather than being overridden again
by recorded self-play outcomes.

No upstream network checkpoint or unpublished weight is present. See
`UPSTREAM_NOTICE.md` and `APACHE-2.0.txt` for provenance and license scope.

## Ground-up build and corpus provenance

The bootstrap workflow compiles its self-play producer from a frozen nine-file
source list and writes one canonical `build-provenance.json` beside every shard
set. That compact, sorted, newline-terminated record binds the ordered source
hashes and their recomputed producer hash; canonical path-independent build
arguments; compiler executable, binary, and version hashes; and the archived
`selfplay-binary` hash. The archived executable is mode-checked before any
shard starts.

The build-provenance SHA-256 is mandatory in every game record. The corpus
validator requires the sibling record, checks canonical encoding, source order,
producer agreement, safe repository-relative paths, and per-game agreement,
then carries the deduplicated full contracts and hashes into the corpus summary
and model artifact. Training additionally verifies the current source bytes,
archived producer binary, and compiler identity. Later portable reads retain
the structural and cryptographic checks without requiring the original build
machine to remain installed.

The hardened ground-up bootstrap uses only the frozen untrained native seed on
both sides, opening depth zero, final mover-relative outcomes, and no incumbent,
protected replay, live-game, or action-policy labels. Nonzero deterministic
prefixes are tested separately; only subsequent native league rounds may use
opening depths 0, 4, 8, and 12. The frozen run produced 10,000 games across 14
shards and completed its build, corpus, training, export, and source-generation
checks before the identities below were recorded.

## Retained round-one checkpoint identity

The retained checkpoint is a **hardened ground-up bootstrap**, not evidence that
the value function is competitive. All 10,000 self-play games use opening depth
zero and the same frozen untrained native runtime on both sides, so the initial
checkpoint cannot depend on a prior trained model. There are no eligible stable
or exact reanalysis targets; the effective labels are final mover-relative
outcomes only. Its frozen identity is:

| Artifact | Frozen value |
| --- | --- |
| Self-play manifest/report | SHA-256 `dce7fb5017b0dec93f6b69dca7f2b7aa4e4e06a02592cd5b2df4d74931a032b9`; `bc8db8bbfe3954dd27d44f50f2bde57a7f79d3442450ecb1c7775d17bc934100` |
| Build/producer/binary | SHA-256 `19e5b3d73b2dd2345fb647c8836177b906544dbb17e78ddae5dd5dafc6796919`; `8d53e59681d7ae9fa2eb0d8cc279945fbf78e49cf40dc29ac139d272eab24542`; `6eb2b47c658f50c4c1d08d76e79a770e2775cce73c0c0eb9c28278981d04e07d` |
| Untrained runtime/corpus | SHA-256 `6773c3b76c8df3e5b824d524bed938b45263215bd3198295f3bed1d082c8c6c2`; `c0200bd13c300081187b544098f4dcf2823f907a39a6769c58cfd8b5af04c9e6` |
| Training seed | `20260813` (selected from `20260811,20260812,20260813`) |
| JSON | 3,516,698 bytes; SHA-256 `19f954092bea404ab18ccc7aaec8b7f6627f0b459017a7f83b6d666b6bb03acc` |
| Packed weights | 14,268 bytes; SHA-256 `7125339d76ade22b0d8e3de249876927b99611372ff81396994c074522394218` |
| Generated header | 21,736 characters; SHA-256 `b9e6e5765bfc6f69e18a968c06e2f92825f91dfd3732176d94f7cd43608af43f` |
| Schema | SHA-256 `dd36c1b2800620fab1d5dc88afe95fcbb13864d581a18f01d26b3e1c3a4a6dfd` |
| Historical history-61 live submission | 94,528 characters; SHA-256 `3bda271b35695292324c4e1943062211d102d66b0bb69f43615ba7a0b89e6e20` |

That historical source leaves 471 characters of explicit project-cap headroom.
The round-two successor is 94,771 characters, SHA-256
`ac63ab602e6b837032fd2e88e2d8ca07e56ebabde956587e005270f45fcaad93`,
and leaves 228 characters. Its later model activation changes the generated
source identity to the exact uploaded hash recorded below. The
independent initial-state golden value is `-0.000181343639`; C++ inference must
match it within `2e-6`. The integration test also freezes rotational and
partial/full evaluation parity and rejects a one-byte packed-payload mutation.

## Activated round-two checkpoint identity

Round two is selected and activated through the immutable
deployment descriptor. The training JSON remains evidence rather than a
mutable deployment pointer: its `chosen_seed` is still `null`, its provisional
minimum-validation-loss seed is `20260823`, and the actual-clock gates selected
seed `20260822`. This is a **native-checkpoint promotion over the prior native
champion**. Its subsequent CodinGame batch is a diagnostic, not a live-strength
or Rank 4 parity result.

| Artifact | Activated round-two value |
| --- | --- |
| Cumulative corpus | 22,238 games; SHA-256 `87cf43fe841dfe7d00fc98ff8d560dfea10a9c3a1832b19eaf092fd0e07edf47` |
| Corpus composition | 12,000 strict-current native league games, 10,000 archived round-one games, and 238 continuations from 119 clean live-loss prefixes |
| Whole-game split | 17,779/2,230/2,229 train/validation/test games; 1,755,307/198,858/197,724 retained rows after 0/21,961/23,469 overlap removals |
| Label boundary | `observed_move_policy_labels: 0`; observed live moves construct restart states only |
| Training artifact | 2,550,520 bytes; SHA-256 `b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14` |
| Retained seeds | `20260821,20260822,20260823`; provisional seed `20260823`; `chosen_seed: null` |
| Selected checkpoint | seed `20260822`; checkpoint SHA-256 `efe76094835fe727e546e5c31c5d4717b796086e636e02a4b05c5aa13b84769a` |
| Selection sidecar | promotion, eligible; 4,494 bytes; SHA-256 `5597e4228850cd44aac4adc5f11e3d6533e5528e3e04c51700d2f04b2cbe2cef`; payload SHA-256 `3b8afae23304fbdb9505b6646ea8f7339ad70e652109aefefd057806ce83f529` |
| Selected runtime | 19,308 bytes; SHA-256 `17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3`; packed SHA-256 `e2304195d491d7b2d5ae1334a8341b38d67d315073accc37915885ede3c6a2cb` |
| Deployment descriptor | 5,325 bytes; SHA-256 `88092ac6601faac0f3da31bdaa1e2a5eca15bdb762b18810d450b33ee0d6ef2f` |
| Activated header | 21,736 characters; SHA-256 `3c1a8ef97f6dc14b9eed64679bd698939380db6fb72181d0b45d1aea74bd3458` |
| Uploaded history-62 source | 94,771 characters; SHA-256 `653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`; 228 characters below the 94,999 project cap |

The cumulative restart lineage binds collector TSV SHA-256
`d5cea44b03a340f220fcb5d2f4864c59151bfd25ad659302bf4c0ead1768b79b`,
arena manifest SHA-256
`0328bded1916af5bd34554bbd315577cc346b7ba2e32b83f34ef3ef0e30351cf`,
exclusion-registry SHA-256
`ac5d335a8e084e782be93f9c53635896f16344f08e9164481dc7b54eaf923a60`,
and the historical asserted source identity. That lineage discloses the live
origin without converting an observed action into a policy or value label.

## Verification boundary

Local fixed-work matches are diagnostic: they make regressions reproducible
but do not establish CodinGame strength. The historical checkpoint and
external promotion gates below used equal-clock 800/165 ms games. The uploaded
exploratory source instead uses 800/155 ms for additional operational margin,
both colors, complete batches, and exact-source binding. Early unoptimized
submissions are legitimate experiments when their source hash and
configuration are recorded; they are not promoted from a single noisy batch.

The timing probe and comparison harness build their openings in memory. They
do not read `matches.json`, replay banks, chronological loss suites, or any
other protected data. Once a live batch influences development it is
development evidence and cannot serve as a later promotion holdout.

Round-two seed promotion uses the separate immutable
[actual-clock selection contract](ROUND2_SELECTION.md). The training JSON
keeps `chosen_seed: null`; complete content-addressed gate evidence selects an
exact runtime without rewriting the model. A separately generated deployment
descriptor binds that model, sidecar, runtime, decision kind, and cumulative
native checkpoint files, plus the exact baseline and content-addressed gate
reports/stdout. Loading it recomputes the selection from those transcripts and
verifies recursive checkpoint ancestry before any production generation.
Merely copying a header or setting `chosen_seed` is not activation. The tracked
descriptor is now deliberately installed, and both the Node and CMake
production generators resolve seed `20260822` through the selection-aware
activation path. The descriptor, not a mutation of the training JSON, is the
production switch.

Source-bound public live games can be diagnosed with the separate, read-only
[replay decision auditor](REPLAY_DECISION_AUDITOR.md). Its output is diagnostic
evidence only and never enters the candidate model or promotion holdout.
The history-61 batch already influenced round-two development, and the current
history-62 batch now informs the next diagnosis. Neither batch nor any derived
fixed-work counterfactual may be reused as independent promotion evidence.

The timing probe measures model construction, search construction, action
selection, and application together. It must remain below 900 ms for a first
response and 180 ms later, leaving margin before CodinGame's 1,000/200 ms
operational limits. Player 0 and Player 1 are measured by separate fresh
process invocations so one color cannot inherit a warmed model or allocator
from the other. For the historically uploaded 800/155 ms artifact, the fresh-process Release
measurements were P0 425.910/157.571 ms with 81,461,248-byte peak RSS and a
76,497,304-byte peak footprint, and P1 397.385/157.921 ms with 91,504,640-byte
peak RSS and a 78,791,088-byte peak footprint. Both pass 900/180; node and
expansion counts remain diagnostics only.

Before upload, the activated round-two source passed its post-activation fresh-process timing
probe: P0 measured 424.894/157.944 ms and P1 measured 419.405/157.479 ms.
Both construction-inclusive processes clear the 900/180 ms pre-upload ceilings.
The independent focused post-activation checks also pass: descriptor
validation, transitive purity, exact-source freshness, three GCC activation
tests, the AppleClang 21 Release build and seven focused tests, and four exact
ASan/UBSan tests. The intentionally focused sanitizer scope excludes unrelated
unbuilt CTest targets. These checks established pre-upload operational
readiness; the exact-source history-62 archive below supplies the separate live
identity and result.

A separate 24-game actual-clock safety screen against canonical `rank_4`
finished 8-16, with candidate colors 5/3. It recorded zero unfinished games,
headroom failures, or operational failures; the candidate's maximum later
response was 166.726 ms. The score is not strength evidence. It only supports
using this exact source as a live diagnostic. For historical context, the
earlier 800/165 ms decisive bootstrap reached 179.918 ms later, and the old
external Rank 4 source reached 180.513 ms, producing one headroom failure.

A subsequent seed-fidelity diagnostic ran the same 24-game actual-clock setup
with the deployed constant shuffle seed and with per-game varied shuffle
seeds. The candidate scored 4-20 and 5-19 respectively. Both variants remained
weak, so a deployment-versus-harness seed mismatch is not the main explanation
for the strength gap.

## Build and verification

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 \
  submissions/codingame/bots/jacek_native_bfm/check_purity.py
node submissions/codingame/tools/generate_submission.mjs jacek_native_bfm
node submissions/codingame/tools/generate_submission.mjs jacek_native_bfm --check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 --target \
  papersoccer_codingame_jacek_native_bfm_submission_test \
  papersoccer_codingame_jacek_native_bfm_timing_probe \
  papersoccer_codingame_jacek_native_bfm_comparison_gate \
  papersoccer_jacek_native_selfplay \
  papersoccer_jacek_native_model_gate \
  papersoccer_jacek_native_search_ab_gate \
  papersoccer_jacek_native_search_ab_gate_test \
  papersoccer_jacek_native_replay_decision_auditor \
  papersoccer_jacek_native_replay_decision_auditor_test
ctest --test-dir build --output-on-failure -R jacek_native
```

The corpus/exporter contract tests that do not need NumPy run under ordinary
Python. To run all trainer and quantization tests locally, use the repository's
pinned research environment (or the bundled Python selected for the workspace):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.codingame.test_jacek_native_training
```

CMake registers the dedicated numerical trainer test only when its configured
Python can import NumPy. The broader Python discovery test remains safe on a
minimal CI interpreter because numerical cases explicitly skip there.
The final candidate also passes the native AddressSanitizer and
UndefinedBehaviorSanitizer suite.

The submission configuration reserves explicit source headroom with a
94,999-character project limit. The exact generated length and SHA-256 belong
in the experiment record and must be refreshed after every production change.

## Checkpoint promotion gate

The exporter can materialize any retained seed as a compact runtime checkpoint
without recompiling the engine:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 \
  submissions/codingame/tools/generate_jacek_native_model.py \
  --model models/jacek_native_bootstrap_model.json --seed 20260813 \
  --output build/candidate-model.hpp \
  --runtime-output build/candidate.runtime
```

`papersoccer_jacek_native_model_gate` loads two such files, validates their
model and packed hashes, and prints the SHA-256 of each complete runtime file.
It plays every procedural opening in both colors with the exact production
250/4,000/512/80,000 search profile and the deployed constant shuffle seed.
`--vary-shuffle-seed` is diagnostic only. Checkpoint promotion is frozen as:

```sh
# Fast screen: 1,000 games.
build/papersoccer_jacek_native_model_gate \
  --candidate-checkpoint build/candidate.runtime \
  --baseline-checkpoint build/baseline.runtime \
  --pairs 500 --first-ms 50 --later-ms 10 \
  --opening-turns 0,4,8,12 --minimum-candidate-wins 530

# Decisive clock: 212 games, with a color floor.
build/papersoccer_jacek_native_model_gate \
  --candidate-checkpoint build/candidate.runtime \
  --baseline-checkpoint build/baseline.runtime \
  --pairs 106 --first-ms 800 --later-ms 155 \
  --opening-turns 0,4,8,12 --minimum-candidate-wins 112 \
  --minimum-wins-per-color 50
```

Both commands fail on any unfinished game, official operational timeout, or
construction-inclusive response at or above 900/180 ms. A fixed-work self-play
score is not a substitute for either actual-clock gate.

### Same-runtime search A/B gate

Search experiments use a separate gate so weight changes cannot masquerade as
search improvements. `papersoccer_jacek_native_search_ab_gate` loads one exact
runtime once, plays every procedural opening in both colors, gives both sides
the same clocks and deployment-constant shuffle seed, and varies only the
declared candidate/baseline search profiles. Pair parity reverses which color
runs first: even pairs run the candidate as Player 1 and then Player 2, while
odd pairs run Player 2 and then Player 1. This balances process warm-up and
time drift across profiles:

```sh
# Generated from the canonical deployment descriptor by the CMake runtime target.
build/papersoccer_jacek_native_search_ab_gate \
  --checkpoint build/jacek-native-current.runtime \
  --pairs 64 --first-ms 800 --later-ms 155 \
  --opening-turns 0,4,8,12 --seed 6510615555426900575 \
  --candidate-tree-nodes 120000 --baseline-tree-nodes 80000 \
  --candidate-c 0.95 --baseline-c 0.95 \
  --candidate-fpu 0.5 --baseline-fpu 0.5 \
  --candidate-final value-log-visits \
  --baseline-final value-log-visits
```

The supported final selectors are the deployed `value-log-visits`,
`value-only`, `value-log-visits-plus3`, and
`value-log-selection-visits-plus3`. The gate records both complete profiles,
the one runtime/model/packed identity, color results, tree/deadline pressure,
formula overrides, elapsed maxima, and safety failures. Its explicit timing
scope is `search-through-apply`: checkpoint/model loading happens before the
timer, so this search A/B does not replace the construction-inclusive final
production timing probe required before upload. Search-through-apply still
fails closed at the 900/180 ms headroom limits. The gate never imports the
Rank-4 engine and does not change production constants or generated source.
Run it through `tools/jacek_native_search_ab_record.py` for immutable evidence;
the recorder holds a serial actual-clock lock and writes the complete stdout
and canonical report under their SHA-256 names.

The selected seed-20260813 runtime is
`877ee8d0afdb20cf3466bee4c09f654d33c6ac4ecc230b8022f570a31e60f93d`.
A diagnostic against sibling seed 20260812 runtime
`d6236e108fcb563d630db73c8e0c41a009efb8dbd3fc2f7e05f529daae685c4f`
lost 484-516 at 50/10 ms (candidate colors 244/240), with zero unfinished
games or timing failures. This is a failed sibling diagnostic, not a
previous-champion promotion gate.

The official bootstrap baseline is the frozen untrained runtime
`6773c3b76c8df3e5b824d524bed938b45263215bd3198295f3bed1d082c8c6c2`.
Against it, the retained runtime passed both preregistered screens:

- 50/10 ms: 689-311 over 1,000 games, colors 363/326, zero unfinished or
  timing failures; candidate maxima 51.1998/12.6256 ms.
- 800/165 ms: 144-68 over 212 games, colors 76/68, zero unfinished or
  timing failures; candidate maxima 451.303/179.918 ms. The later maximum
  passed by only 0.082 ms.

These results show learning over the untrained seed. They do not establish
parity with the incumbent or safety on CodinGame.

## External Rank 4 gates

Checkpoint promotion and external comparison answer different questions. The
checkpoint gate above compares weights with the engine held fixed. The
external gate treats canonical `rank_4` only as an opponent and always uses
actual equal clocks.

The early development screen is 53 paired openings, hence 106 games. It must
score at least 58 wins and is repeated with a different deterministic seed
before the result influences promotion:

```sh
build/papersoccer_codingame_jacek_native_bfm_comparison_gate \
  --equal-clock --pairs 53 --opening-turns 8 \
  --seed 6510615555426900575 --maximum-turns 384 \
  --minimum-candidate-wins 58
```

The final parity declaration is deliberately larger: 212 paired openings and
424 games. With four opening depths, `--pairs 53` means 53 openings at each
depth. A pass requires the candidate's 95% Wilson lower confidence bound to be
at least 50% (233 wins when all 424 games finish), at least 48% wins in each
color (`ceil(0.48 * 212) = 102`), and zero unfinished games, timing failures,
or operational timeouts:

```sh
build/papersoccer_codingame_jacek_native_bfm_comparison_gate \
  --equal-clock --pairs 53 --opening-turns 0,4,8,12 \
  --seed 6510615555426900575 --minimum-wilson-lower 0.5 \
  --minimum-wins-per-color 102
```

The harness prints the paired-opening/game counts, raw color split, win rate,
Wilson lower bound, exact search limits, maximum first/later response times,
and pass/fail state for every enabled threshold. “Paired opening” always means
the same opening played once in each color; it is never used as a synonym for
one game.

The exact generated source
`8e67a0c795809e17490f719b2130d172c8aea2fd8df51ad0d44ca2d97614c1e3`
failed the first external screen 35-71 over 106 games, with candidate colors
16/19. Its win rate was 0.330189 and the 95% Wilson lower bound was 0.24798.
There were no unfinished games or operational timeouts, but one candidate
later response reached 180.513 ms and failed the pre-upload ceiling. The
58-win development threshold failed, so the second-seed repeat, 424-game
parity gate, and CodinGame upload were not run.

## Completed CodinGame diagnostics

### Historical history-61 baseline

The round-one 800/155 ms source was uploaded from Git commit
`8cf6005aace930016b86ac05de2ac8743447612c`. The editor contents were copied
back and matched the 94,528-character generated source SHA-256
`3bda271b35695292324c4e1943062211d102d66b0bb69f43615ba7a0b89e6e20`.
No API source verification was performed in this flow, so the binding is
recorded precisely as **asserted, not API-verified**. The live identity is
agent `6609056`, submission `41123817`, history version `61`,
model SHA-256
`19f954092bea404ab18ccc7aaec8b7f6627f0b459017a7f83b6d666b6bb03acc`,
and packed-weight SHA-256
`7125339d76ade22b0d8e3de249876927b99611372ff81396994c074522394218`.

All 90 historical arena games completed. The raw score was 52-38, with colors 29-15 and
23-23, rank 9, and score 39.54. Eleven wins were opponent forfeits: seven
illegal actions and four timeouts. After removing those forfeits before
analysis, 79 clean games remained at 41-38, split 21-15 and 20-23 by color.
The candidate recorded zero operational failures. Against the frozen cohorts,
the clean results were 4-25 versus the top 5, 6-32 versus the top 10, and
22-37 versus the top 20. This is a completed exploratory batch, not evidence
of parity or a checkpoint promotion.

The provenance-safe archive is bound by manifest SHA-256
`0328bded1916af5bd34554bbd315577cc346b7ba2e32b83f34ef3ef0e30351cf`
and clean auditor TSV SHA-256
`d5cea44b03a340f220fcb5d2f4864c59151bfd25ad659302bf4c0ead1768b79b`.
The frozen 30,000-work replay audit input has SHA-256
`7f06835b8cfc0e4a8a51ff02195aed12d06a729af70a66cf0ddced0cafd86fee`;
its canonical summary has SHA-256
`4d9d56bc1c66c8cac6366c64b4b2c2683bdd5a9f0302c45591c76a57a672972b`.
Across 1,918 decisions in the 79 clean games, the audit classified 1,070 as
BFM override, 702 as match, 136 as initial-evaluator ordering, six as generator
omission, and four as operational failure. These labels compare the deployed
choice with a fixed-work counterfactual; they do not establish that the
counterfactual would have won. The very small omission count directs the next
iteration toward evaluator/reanalysis and BFM allocation rather than broader
action generation. This batch remains explicitly historical; it is not the
current deployed checkpoint.

### Current round-two history-62 diagnostic

The selected seed-`20260822` source was uploaded from repository commit
`e1ae4c7c66a03d9a2c3b82ddf79adafcb7e0c661` as agent `6609905`, submission
`41124914`, history version `62`. The editor read-back matched the exact
94,771-character source SHA-256
`653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`.
No API source verification was performed, so the binding remains precisely
**asserted, not API-verified**. The deployed model SHA-256 is
`b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14`;
its packed-weight SHA-256 is
`e2304195d491d7b2d5ae1334a8341b38d67d315073accc37915885ede3c6a2cb`.

All 90 games completed and the full window was accounted for. The raw result
was 63-27, split 30-13 and 33-14 by candidate color, at frozen rank 5 and score
42.68. The candidate recorded zero operational failures. Nineteen wins were
opponent forfeits: ten illegal actions and nine timeouts. Removing them leaves
71 clean rule-terminal games at 44-27, split 18-13 and 26-14 by color. Clean
frozen-cohort results were 11-14 against the top 5, 26-22 against the top 10,
and 34-24 against the top 20.

The named clean results with at least three games were: jacek 0-9, Deltaspace
6-1, Laars 5-2, Waffle3z 4-0, derjack 1-3, EricSMSO 3-3, and YurkovAS 7-0.
The 0-9 result against jacek is the clearest disclosed remaining gap. Rank 5,
the aggregate clean record, and individual opponent results do not establish
Rank 4 parity.

The provenance-safe archive is bound by manifest SHA-256
`bb5aaf7340ddee174ddf916aee2bfbcd4b3afb01f5d53bbc5c7d728bf610b4cf`
and clean auditor TSV SHA-256
`4a25768aed11e7c4bc368e63bce8c335e420f02fee7e131d2b56dfa97ab048e2`.
The fixed-30,000-work replay audit JSONL has SHA-256
`9eb8d1741bbd11390c3d0b1c15ea0cac35f73931e6695635c2f5778d7c7ff8f1`;
its summary has SHA-256
`79d88d6f608d2a78e86dd1f6abca8a5406a9866a417489dc7b2d99beb17cdd46`.
Across 1,811 decisions in all 71 clean games, it classified 1,015 as BFM
override, 715 as match, 76 as initial-evaluator ordering, four as generator
omission, and one as operational failure; boundary-equivalent and tactical-miss
counts were both zero. These are deterministic
counterfactual classifications, not correct-move labels, winning alternatives,
or independent promotion evidence.

Round-two native league, reanalysis, restart training, and actual-clock
selection are complete. Seed `20260822` passed 803-197 at 50/10 ms and 146-66
at 800/155 ms, with decisive colors 84/62 and zero unfinished games, headroom
failures, or operational timeouts for either side. The checkpoint is activated
locally through deployment descriptor SHA-256
`88092ac6601faac0f3da31bdaa1e2a5eca15bdb762b18810d450b33ee0d6ef2f`.
That exact activated checkpoint is the current history-62 live diagnostic
described above. It remains development-contaminated as soon as it informs the
next iteration and does not replace the unrun 424-game Rank 4 parity gate.
