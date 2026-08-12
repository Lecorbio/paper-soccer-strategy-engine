# Native Jacek-style best-first minimax candidate

`jacek_native_bfm` is a clean challenger built around complete-turn generation,
best-first minimax search with UCT-style allocation, and a compact neural value
function. It is intentionally independent of every incumbent bot, replay table,
and historical action label in this repository. The established `rank_4` bot is
used only as an external opponent in `comparison_gate.cpp`.

The checkpoint, training, clock, verification, and live-gate ledger is retained
in this README and the adjacent [experiment log](EXPERIMENTS.md).
The strongest completed live result remains the round-two seed-`20260822`
history-62 checkpoint: 63-27 raw, 44-27 after removing opponent forfeits, rank
5, and zero candidate operational failures. Histories 63 and 65 tested a new
model and `C=0.80`; both were rejected after complete windows. Exact
history-62 rollback history 66 completed at rank 6. A later first-decision
20,000-node hypothesis passed two isolated local gates, but history 67 fell to
rank 9 at 62-28 raw and 52-28 clean. Its live opening behavior also failed to
reproduce its deterministic fixed-20k prediction, so the hypothesis was
rejected. Commit `a7dd201dbaf32b98f6d661fe4b076c4c769e1815` restored the exact
94,771-character history-62 source and launched it as history 68. That new
rollback window is not yet reported as complete. History 62 therefore remains
the strongest completed result under the campaign ordering. None of these live
batches is a strength, superiority, or Rank 4 parity claim. The prior
history-61 round-one upload remains frozen below as an explicit historical
baseline.

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

## Round-three campaign candidate and local rollback

Round three retained the native architecture and trained on 26,402 cumulative
games: the archived 10,000-game round-one corpus, 12,000 existing native league
games, 4,000 new balanced native league games, 238 history-61 restart
continuations, and 164 history-62 restart continuations. The corpus SHA-256 is
`f58768b6090eb60968334310bd4ea7e62d9977b7a8318ba382803bdcbc1d3130`.
Its whole-game split contains 21,111/2,645/2,646
train/validation/test games and 2,030,276/231,690/228,884 retained samples after
0/25,003/26,579 canonical overlap removals. Training used seeds
`20260831,20260832,20260833`, phase weights `3.0/1.5/1.0` for turns
`0-11/12-23/24+`, and applied the weights to the combined target after exact
outcome override while retaining the 25% stable-reanalysis auxiliary term.

| Artifact | Round-three identity |
| --- | --- |
| Training JSON | 2,664,686 bytes; SHA-256 `a13b86accd168ae51cecf5df85092642ba8edb21d9ef944ac7c002bfa6a5d19a` |
| Selected checkpoint | seed `20260832`; checkpoint SHA-256 `fedbab01a0f44e0612575aeee867bb846da2ec6eed7cd4383327dd67112213bd` |
| Selection sidecar | 4,985 bytes; SHA-256 `84cc694cf390fc6a0ddc19c5e61d5770a84172c31618d9d14576365e7e16f635`; payload SHA-256 `6abb8884a7c8a8a72dd75360d0a7d8f1f0255159c8b9b6f3098a4f841b24e45c` |
| Runtime / packed weights | SHA-256 `0aaff836c1e96b949713c2a25b88456ca1f60f4599d5ce303f4f46fcc1ed7b52`; `d8e1110caddf4bbd65a8c7e7d387c979c1a15807f29e5ab908d471e3159bab94` |
| Deployment descriptor | 7,137 bytes; SHA-256 `a872281d7d3c458df38507578e01c3e5d663ac487a9cad3128685615b15b817a` |
| Generated header | 21,737 characters; SHA-256 `205a5a7c8ec67fa86df768024e7c6458d29e15a6d846079ef15a31cb4ed9cb47` |
| Uploaded history-63 source | 94,772 characters; SHA-256 `5e628e5552da4a22f6dd3c73064d8f2670d13d3e3e6f23ce3ee93bdd79fb306a` |

Only seed `20260832` passed both local model gates: 562-438 at 50/10 ms
(colors 252/310) and exactly 112-100 at 800/155 ms (56/56). Seed `20260831`
scored 610-390 and 123-89 but failed the decisive second-color floor at 78/45;
seed `20260833` scored 540-460 and 108-104 and failed at 75/33. All six runs
had zero unfinished games, headroom failures, or operational timeouts. The
complete report and stdout hashes are recorded in `EXPERIMENTS.md`.

The isolated same-runtime search campaign retained the original history-62
profile. Every row had zero unfinished games, headroom failures, and
operational timeouts:

| Candidate versus exact history-62 search | Result; colors | Clock | Decision | Report / stdout SHA-256 |
| --- | --- | --- | --- | --- |
| 120,000 versus 80,000 nodes | 65-63; 32/33 | 800/155 | reject: total below 70 | `8bcc269aba0e34543d7ca896a2d8846c6674decc60c39b4444a51986dc7b25c8`; `8c9c2036c44921a4ba596c7d4256f45017a75413f681f87c2d05103420a02f14` |
| `value + log(selectionVisits + 3)` | 63-65; 38/25 | 50/10 | reject: total and color | `45dcef903263a8b07de38403b201c5d583f88e80250cb32ab679cbe052dec52a`; `bcc4157eb1820cca753d9983c14a2c0d7225d919c35359595765051d85d530b0` |
| `C=0.65` | 73-55; 30/43 | 50/10 | reject: first-color floor 31 | `0fa284cefc82048c5135e1ce60ca9c541aa185b19fb4be40875c58b9b4d24bf7`; `1afb59cadda61ccaed4c77100ce4b920d89eebb0a5b6e5b6adb859fba8b3082c` |
| `C=0.65` decisive | 55-51; 34/21 | 800/155 | reject: total floor 58 and color floor 25 | `6c9db5729bd4f284f6b736c19077519eec25db9bb9176b96c6115f6a27d4631d`; `e0f754254ec7e1b6fec7ef7aef5b3a42b74998ce61287a8259641fb366897d86` |
| `C=0.80` screen | 65-63; 34/31 | 50/10 | below 70 screen total; retained as final replay-grounded extension | `c714f44f01bc09b3fe2995cdf3154166e516e7038f4cb1c3c88baba2b65a80ba`; `0ae1aadcb83778f8ecaca41177fda4faddf1d27c2f6022f52d24e184f7da2182` |
| `C=0.80` decisive | 62-44; 37/25 | 800/155 | pass: total 58 and color 25 floors | `4ae2508f5ab80af715588b0d21543bd2df5c4ea2af48be03d06d6c7cddf87515`; `26889d86fa47ec164e265c75ca83b124901c875c6f382749b658de1f3f35d116` |
| `C=1.25` | 43-85; 24/19 | 50/10 | reject | `0e241235e32b09f3cb6617ed77b4a22eabfb0c0cb026c1ee5e2da31fde110d29`; `10a77b8ec0b426140d9095fd4eb28a014ba0f65eb04334f4af06511afed09411` |
| `FPU=0.25` | 57-71; 40/17 | 50/10 | reject | `9c8ee163c1902db53a6b8d088049bd32bc6d9c8194bdcbf7fd79753340116732`; `8e925f4bca5680cbf4402737a3dd5c3f4190a7d30f89b77ca469830b6c9eed47` |
| `FPU=0.75` screen | 87-41; 41/46 | 50/10 | advance | `80047c77d5b5f146ab45b247027e554723fe7667441741beb9d3fed4b8f7846b`; `aa5d798a891db547129de45f9fef092077e35ae3b78ce2e12edc867a3e850564` |
| `FPU=0.75` decisive | 55-51; 37/18 | 800/155 | reject: total 58 and color 25 floors | `6a5d0c7a55c3a3514faa321fddc5933848232b33993ca106185d6228f679c35f`; `2dbd6553e56155be23bea562dbf20f3436ef30e6554082b6291bab398547a08c` |

The completed history-63 window then rejected this local promotion. The final
local state is the exact history-62 seed-`20260822` model with the original
80,000-node, `C=0.95`, `FPU=0.5`, `value + log(visits)` search. Immutable v2
reactivation descriptor SHA-256
`31772c68cd9da04503e2fa760926ccbd825820d62d45584da3c857d2e6b26aa6`
regenerates header SHA-256
`3c1a8ef97f6dc14b9eed64679bd698939380db6fb72181d0b45d1aea74bd3458`
and source SHA-256
`653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`
exactly. Commit `0f3a95f50b42c8135fc929a0a8fe8ccf6756d4f2` first restored this
identity, and its CI and Pages run completed successfully. After the rejected
history-65 experiment, commit `07317cebf680eb4394ca424801ebb0cfb644002a`
restored the same bytes again and the editor accepted them as history 66. The
history-64 and history-66 rollbacks are not new model identities or independent
promotion gates.

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
The history-61 batch influenced round-two development, and the history-62 and
history-63 batches informed the round-three campaign and rollback diagnosis.
History 64 verifies the exact rollback remotely and is now also development
evidence. None of those batches or any derived fixed-work counterfactual may be
reused as independent promotion evidence.

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

### Retained round-two history-62 diagnostic

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
That exact activated checkpoint produced the retained history-62 live
diagnostic described above. It became development evidence as soon as it
informed round three and does not replace the unrun 424-game Rank 4 parity
gate.

### Rejected round-three history-63 diagnostic

Round-three seed `20260832` was uploaded from commit
`cf3800e35dbb3dc870e23450c34d66484bc953a8` as agent `6611653`, submission
`41127173`, history version `63`. Editor read-back matched the exact
94,772-character source SHA-256
`5e628e5552da4a22f6dd3c73064d8f2670d13d3e3e6f23ce3ee93bdd79fb306a`.
The binding remains `asserted-not-api-verified`. Its model and packed-weight
SHA-256 identities are respectively
`a13b86accd168ae51cecf5df85092642ba8edb21d9ef944ac7c002bfa6a5d19a`
and `d8e1110caddf4bbd65a8c7e7d387c979c1a15807f29e5ab908d471e3159bab94`.

The full 90-game window was accounted for with zero candidate operational
failures. It scored 59-31 raw, with colors 32-20 and 27-11, at frozen rank 6
and score 42.09. Four wins were opponent forfeits: one illegal action and three
timeouts. The 86 clean rule-terminal games scored 55-31, with colors 31-20 and
24-11. Clean cohorts were 7-16 against the top 5, 23-27 against the top 10,
and 41-31 against the top 20. Named clean results with at least three games
were jacek 0-4, Marchete 1-3, Deltaspace 2-5, Laars 3-1, Snekkers 1-3,
Waffle3z 5-3, EricSMSO 3-4, derjack 4-3, YurkovAS 4-1, trictrac 5-4,
ILove47 3-0, and Spoonboy82 4-0.

| Evidence | Archive path | SHA-256 |
| --- | --- | --- |
| Complete-batch manifest | `results/codingame_arena_diagnostics/manifests/6611653/41127173/5e628e5552da4a22f6dd3c73064d8f2670d13d3e3e6f23ce3ee93bdd79fb306a/d57434f808654b25b78999bc1a9e9d1e97754225c51327675e2172084e863620.json` | `d57434f808654b25b78999bc1a9e9d1e97754225c51327675e2172084e863620` |
| Clean auditor TSV | `results/codingame_arena_diagnostics/runs/jacek-native-round3-41127173-20260812/clean-auditor.tsv` | `bf349361300b30063eb8e0d3f28802f3585ef9d556b227f6c75607663d80d95c` |
| Fixed-30k decision audit | `results/codingame_arena_diagnostics/runs/jacek-native-round3-41127173-20260812/native-decisions-fixed30k.jsonl` | `201d1343037a9104ffea2a68ffc4573a056b7319272646642e946960736a5a87` |

The fixed-30k audit covers 2,075 decisions: 1,094 `bfm-override`, 861
`match`, 115 `initial-evaluator-ordering`, two `generator-omission`, and three
diagnostic `operational-failure`; `tactical-miss` is zero and actual-boundary
retention is 99.904%. There are no search or diagnostic-root deadline events,
while the search reaches its 30,000-node audit cap on 91.13% of decisions,
almost identical to history 62. The decisive regression is at the first own
decision: the observed history-63 choice has mean/median initial evaluator rank
5.837/7 and is top-five only 40.7% of the time, versus 3.211/3 and 94.4% for
history 62. This, together with the worse clean top-five result, points to an
early evaluator/search-allocation regression rather than action coverage,
tactical classification, deadlines, or a larger tree budget. History 63 is
rejected and remains diagnostic development evidence only.

### Exact history-62 rollback history-64 diagnostic

Commit `0f3a95f50b42c8135fc929a0a8fe8ccf6756d4f2` restored the exact
history-62 model and production search profile. The corresponding CI and Pages
run [31614166711](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/runs/31614166711)
completed successfully. The byte-identical 94,771-character source SHA-256
`653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`
was uploaded as agent `6611801`, submission `41127479`, history version `64`.
The source binding remains `asserted-not-api-verified`.

All 90 games completed and the full window was accounted for with zero
candidate operational failures. The raw result was 64-26, with candidate
colors 31-13 and 33-13, at frozen rank 6 and score 41.67. Eleven wins were
opponent forfeits: six timeouts and five illegal actions. Removing them leaves
79 clean rule-terminal games at 53-26, with colors 26-13 and 27-13. Clean
cohort results were 6-10 against the top 5, 17-18 against the top 10, and
31-23 against the top 20. Named clean results were jacek 0-4, Marchete 2-3,
Deltaspace 4-0, Laars 0-2, and Snekkers 0-1. The raw jacek result was 1-4
because its additional candidate win was a jacek timeout.

| Evidence | Archive path | SHA-256 |
| --- | --- | --- |
| Complete-batch manifest | `results/codingame_arena_diagnostics/manifests/6611801/41127479/653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90/3edfab1632e9dcfca326c2d58f2951982352837e8f86e0ac9ab6085dfd817026.json` | `3edfab1632e9dcfca326c2d58f2951982352837e8f86e0ac9ab6085dfd817026` |
| Clean auditor TSV | `results/codingame_arena_diagnostics/runs/jacek-native-history64-41127479-20260812/clean-auditor.tsv` | `cd82d1ea00ca7061f7b7f95a9a62b289d40df38128d346b4b7f3da2e64ff4d2c` |
| Fixed-30k decision audit | `results/codingame_arena_diagnostics/runs/jacek-native-history64-41127479-20260812/native-decisions-fixed30k.jsonl` | `a2a804efe90d21a20af34530f833535c8f27eb5773d34687f8a3c0e1d3b1fbf8` |
| Fixed-30k summary | `results/codingame_arena_diagnostics/runs/jacek-native-history64-41127479-20260812/native-decisions-fixed30k-summary.json` | `e477ff03154320e1dafe5b8f77fa6dca3af18e500dd09358df7409d7b346dde1` |

The fixed-30k audit joined 1,996 decisions across all 79 clean games. It
classified 1,135 `bfm-override` (56.86%), 777 `match` (38.93%), 75
`initial-evaluator-ordering` (3.76%), seven `generator-omission` (0.35%), and
two diagnostic `operational-failure` (0.10%); `tactical-miss` was zero.
Actual-boundary retention was 99.55%, and search reached its 30,000-node cap on
91.28% of decisions. On the first own decision, 59/79 were `bfm-override`,
19/79 were `match`, and one was `initial-evaluator-ordering`; the observed
action's initial rank had mean 2.684, median 3, and was top-five in 78/79
games. Every first-decision audit reached the tree cap. This reproduces the
history-62 evaluator profile and confirms that history 63's opening regression
was model-specific. History 64 was the then-deployed exact history-62 source,
but its batch is development evidence rather than an independent promotion
result.

The replay-grounded extension first retested `C=0.65` against the exact
history-62 runtime for 106 serial actual-clock games at 800/155 ms, opening
seed `2026081203`. It scored 55-51 with colors 34/21, below both the required
58 total wins and 25 wins per color. Candidate maxima were 468.290/162.010 ms
and baseline maxima were 487.884/165.192 ms; both sides had zero unfinished
games, headroom failures, and operational timeouts. The report and stdout
SHA-256 identities are respectively
`6c9db5729bd4f284f6b736c19077519eec25db9bb9176b96c6115f6a27d4631d`
and `e0f754254ec7e1b6fec7ef7aef5b3a42b74998ce61287a8259641fb366897d86`.
The candidate was rejected and not uploaded.

### Rejected `C=0.80` history-65 diagnostic and history-66 rollback

The terminal isolated search experiment changed only exploration from
`C=0.95` to `C=0.80`. Its 128-game 50/10 ms screen scored 65-63 with colors
34/31: balanced, but below the original 70-win screen threshold. Retained as a
final replay-grounded extension, it then passed the 106-game 800/155 ms gate
62-44 with colors 37/25, exactly meeting the weaker-color floor. Candidate
maximum first/later search-through-apply times were 452.398/164.421 ms versus
464.665/162.460 ms for the baseline, with zero unfinished games, headroom
failures, or operational timeouts for either side.

Commit `38367fe5eca723d8db7a15a03ace46a57d0594ba` generated the exact
94,771-character candidate source SHA-256
`1aba496a561b86453c95208ee6ea45596d385c2847647398df608e3192f90043`.
Editor copy-back matched that identity. Purity, exact-source-current, and
protocol checks passed; GCC and Clang each passed all five focused tests, and
the focused ASan/UBSan build passed all three tests. Fresh-process timing was
429.425/158.818 ms for Player 0 and 426.090/158.454 ms for Player 1, below the
900/180 ms pre-upload ceilings.

The source was uploaded as agent `6611839`, submission `41127785`, history
version `65`. All 90 games completed at 55-35 raw, colors 26-19 and 29-16,
frozen rank 9 and score 40.53, with zero candidate operational failures. Nine
wins were opponent forfeits. Removing them leaves 81 clean rule-terminal games
at 46-35, colors 24-19 and 22-16. Clean cohort results were 6-14 against the
top 5, 14-29 against the top 10, and 30-33 against the top 20. Named clean
results were jacek 0-6, Marchete 0-4, Deltaspace 1-0, Laars 4-2, and Snekkers
1-2.

| Evidence | Archive path | SHA-256 |
| --- | --- | --- |
| Complete-batch manifest | `results/codingame_arena_diagnostics/manifests/6611839/41127785/1aba496a561b86453c95208ee6ea45596d385c2847647398df608e3192f90043/54371f7a59f6bd86f845db839d433124f3dd2f141adcb490f0c2cc7dc1b00595.json` | `54371f7a59f6bd86f845db839d433124f3dd2f141adcb490f0c2cc7dc1b00595` |
| Clean auditor TSV | `results/codingame_arena_diagnostics/runs/jacek-native-history65-41127785-20260812/clean-auditor.tsv` | `86b96c96952609251c794ebb83228e9e0ef054ead06bc67bdb4118ff10fdb2ae` |
| `C=0.80` fixed-30k audit | `results/codingame_arena_diagnostics/runs/jacek-native-history65-41127785-20260812/native-decisions-c080-fixed30k.jsonl` | `bdc98b538464670b7dfdde8bd0dc6798366836e0cf00f310f5b5bcfff7d30b16` |
| `C=0.95` fixed-30k counterfactual | `results/codingame_arena_diagnostics/runs/jacek-native-history65-41127785-20260812/native-decisions-c095-fixed30k.jsonl` | `da76f15c3263cec97ea4085f674f69b5eefeb7366258f9795a95e7cd25bfa3a5` |
| Paired audit summary | `results/codingame_arena_diagnostics/runs/jacek-native-history65-41127785-20260812/native-decisions-c080-vs-c095-fixed30k-summary.json` | `2ebe87f8ac054be1666489dd163eda36d8113e3a93697d16a748adefcbaae57d` |

The paired audit aligned 1,986 decisions and changed the fixed-30k choice on
615. On the 81 first decisions, `C=0.80` classified 8 `bfm-override`, 43
`initial-evaluator-ordering`, and 30 `match`, whereas counterfactual `C=0.95`
classified 23, 0, and 58 respectively. Overall, `C=0.80` classified 1,032
`bfm-override`, 178 `initial-evaluator-ordering`, 771 `match`, four
`generator-omission`, and one diagnostic `operational-failure`; `C=0.95`
classified 973, 133, 876, four, and zero. Both retained the observed boundary
on 99.748% of decisions, had no search or diagnostic-root deadlines, and
reached the tree cap on 91.843% and 91.994% respectively. These labels are
counterfactual diagnostics, not correct-move or causal winning labels. Together
with the rank, cohort, and named-opponent regressions, they reject `C=0.80`.

Commit `07317cebf680eb4394ca424801ebb0cfb644002a` reverts the experiment
and restores the exact 94,771-character history-62 source SHA-256
`653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`.
Editor copy-back matched, the platform Play check won a legal game in seven
turns, and the safe source was uploaded as history 66.

#### Exact history-62 history-66 full-window result

History 66 is agent `6611921`, submission `41127895`; its source binding
remains `asserted-not-api-verified`. All 90 matching-submission games completed
and the full window was accounted for. The raw result was 60-30, colors 34-15
and 26-15, at frozen rank 6 and score 41.46. The candidate recorded zero
operational failures. Eight wins were opponent forfeits: six timeouts and two
illegal actions. Removing them leaves 82 clean rule-terminal games at 52-30,
colors 29-15 and 23-15. Clean cohort results were 8-11 against the top 5,
16-22 against the top 10, and 36-29 against the top 20. Named clean results
were jacek 0-6, Marchete 2-1, Deltaspace 1-1, Laars 4-0, and Snekkers 1-3.

| Evidence | Archive path | SHA-256 |
| --- | --- | --- |
| Complete-batch manifest | `results/codingame_arena_diagnostics/manifests/6611921/41127895/653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90/eca45a7047c03c267194fc52048c67bccde29ec501452e42f77bd49bd95a8abc.json` | `eca45a7047c03c267194fc52048c67bccde29ec501452e42f77bd49bd95a8abc` |
| Clean auditor TSV | `results/codingame_arena_diagnostics/runs/jacek-native-history66-41127895-20260812/clean-auditor.tsv` | `95f03256b2cd760416816de3171c16918f4105af6e9d212eaf9a865d9c58f74e` |
| Fixed-30k decision audit | `results/codingame_arena_diagnostics/runs/jacek-native-history66-41127895-20260812/native-decisions-fixed30k.jsonl` | `261f88ed661a43521dd5ef8dea8496804e7623daa37ae6be40a7059dc4f576eb` |
| Fixed-30k summary | `results/codingame_arena_diagnostics/runs/jacek-native-history66-41127895-20260812/native-decisions-fixed30k-summary.json` | `4a18611592c46b21a151f46af27defd7df2081f6e7497c80c3302e5db063890c` |

The retained fixed-30k evidence is an auditor rebuild and rerun explicitly
bound to history-62 model SHA-256
`b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14`.
An initial invocation exposed a stale build and was discarded before evidence
was recorded; no hash for that invalid output is documented. The valid rerun
joined 2,058 decisions and classified 1,132 `bfm-override`, 818 `match`, 87
`initial-evaluator-ordering`, 16 `generator-omission`, and five diagnostic
`operational-failure`; `tactical-miss` was zero. Actual-boundary retention was
99.0768%, the tree-cap rate was 91.3994%, and neither search nor the diagnostic
root recorded a deadline. On 82 first own decisions, 56 were `bfm-override`,
25 `match`, and one `initial-evaluator-ordering`; initial rank had mean 2.744,
median 3, and was top-five in 81/82, with every first decision tree-capped. The
five diagnostic operational rows do not contradict the manifest's zero live
candidate failures.

The eight-hour campaign therefore terminated with `C=0.80` rejected and the exact
history-62 source deployed as history 66 at rank 6. The historical history-62
batch remains the best recorded rank at 5; rank 1 was not reached.

### Rejected first-decision 20k history-67 diagnostic and history-68 rollback

Commit `b7b8a41e8f9512008da0c5442b6b401dc89479fd` added a phase-aware
same-runtime gate and a production schedule that could use a separate tree cap
for the first own decisions. The gate binds opening and later caps, the number
of opening own decisions, and opening/later decision, deadline, tree-cap, time,
and maximum-tree telemetry. The legacy global tree-cap interface remains
supported.

Four exact-history-62 actual-clock comparisons isolated the hypothesis. All
used 800/155 ms clocks, balanced colors, constant deployment shuffling,
`C=0.95`, `FPU=0.5`, and the production final rule:

| Gate | Candidate-reference | Colors | Result | Report / stdout SHA-256 |
| --- | ---: | ---: | --- | --- |
| Global 20k versus global 80k, 106 games | 52-54 | 26 / 26 | reject: 58-win total floor | `ce0742c581c668d1ef896dd60a176419f58423b4b2d10e131936052c8a907fbe` / `c9ce3110f5601c2e30da3a6b7f64810a7bf53f1818837afec0e66d4bd674f308` |
| First four own decisions 20k, then 80k, 106 games | 50-56 | 28 / 22 | reject: total and color floors | `04dd63b485774cfa14e84072b6a79ce1deb0c03e0a990728291d714910e8caf3` / `1c9af275d3df029ac41f1c6449185216a73a32190ded613c50783aa89a49bd1a` |
| First own decision 20k, then 80k, 106 games | 67-39 | 34 / 33 | pass | `e8e2c4d2212a067651f2db0a72083534e6cccbefad8dffa3e29ce2ca16c73af8` / `e5ed4a6c92338f96f493a771fa3e3412ed661c56f2b65f9351993e1c6f95af3e` |
| First-decision depth-0 replication, 32 games | 29-3 | 13 / 16 | pass | `86ce7d9ccd1c2d0c6c33c67851c1f45a7ba0ad53c3d626430d7dd4e34c099a9c` / `8c533a23062f659f7c299184ffdc712fbe9152b9a881e8bfa9506e8eda84ea97` |

The isolated first-decision result and its depth-0 replication justified an
experimental upload despite the rejected broader schedules. The exact
94,942-character source SHA-256
`10a00c74e65866e84be3086427b6b14c6b9fb1b50be8451bb382d12cec36bf10`
was uploaded as agent `6612628`, submission `41128698`, history 67. All 90
matching-submission games completed with zero candidate operational failures.
It scored 62-28 raw, colors 34-11 and 28-17, at frozen rank 9 and score 40.05.
Ten wins were opponent forfeits: six illegal actions and four timeouts. The 80
clean rule-terminal games scored 52-28, colors 29-11 and 23-17. Clean cohorts
were 3-11 against the top 5, 11-17 against the top 10, and 32-20 against the
top 20. Named clean results were jacek 0-2, Marchete 1-4, Deltaspace 1-1,
Laars 1-3, and Snekkers 0-1.

| Evidence | Archive path | SHA-256 |
| --- | --- | --- |
| Complete-batch manifest | `results/codingame_arena_diagnostics/manifests/6612628/41128698/10a00c74e65866e84be3086427b6b14c6b9fb1b50be8451bb382d12cec36bf10/a3dbbb8a9fcd4f11f584fe80ea109b7a19335c6b6ea1748824eb4d655524003d.json` | `a3dbbb8a9fcd4f11f584fe80ea109b7a19335c6b6ea1748824eb4d655524003d` |
| Clean auditor TSV | `results/codingame_arena_diagnostics/runs/jacek-native-first1-41128698-20260812/clean-auditor.tsv` | `7434010bf57293b831a81f395399503df2eed2ba07a38c72665d7b8a6d52ebc3` |
| Fixed-30k decision audit | `results/codingame_arena_diagnostics/runs/jacek-native-first1-41128698-20260812/native-decisions-fixed30k.jsonl` | `154acf50fd0e613ec9c0f18235b0c2787c0f3fa1526f15d116e6cee2ebd77727` |
| Fixed-30k summary | `results/codingame_arena_diagnostics/runs/jacek-native-first1-41128698-20260812/native-decisions-fixed30k-summary.json` | `37cb7d27cf5610fc0d8be73fb8aeb0f523aab57ebc47c68f5304c510849bf25e` |
| First-decision fixed-20k audit | `results/codingame_arena_diagnostics/runs/jacek-native-first1-41128698-20260812/native-first1-fixed20000.jsonl` | `33e0d90fefbdb381424b102032ac73500ebd6530896ae8053c44bb8bedc6a6a2` |
| First-decision fixed-80k audit | `results/codingame_arena_diagnostics/runs/jacek-native-first1-41128698-20260812/native-first1-fixed80000.jsonl` | `fe7e1e51e26dc7a8cc00e2f2af4abb722741fc60e7fc033ef2df25163b2f14e7` |
| First-decision 20k-versus-80k summary | `results/codingame_arena_diagnostics/runs/jacek-native-first1-41128698-20260812/native-first1-20k-vs80k-summary.json` | `8c8410fb173959395ee7a6465a2df107383a5675d54538e507d2d0b7e763681b` |

The valid fixed-30k audit covers 2,011 decisions: 1,109 `bfm-override`, 803
`match`, 91 `initial-evaluator-ordering`, six `generator-omission`, and two
diagnostic `operational-failure`; `tactical-miss` is zero. Actual-boundary
retention is 99.453%, the tree-cap rate is 91.596%, and neither search nor the
diagnostic root recorded a deadline.

The first-decision counterfactual aligned all 80 clean games and changed 49
choices. Live actions matched fixed 20k on 31/80 decisions and fixed 80k on
56/80. Player 0 repeated one identical initial state 40 times: history 67
played `0` 34 times, `7` five times, and `1` once, although fixed 20k always
selected `7` and fixed 80k always selected `0`. Those choices match only 5/40
for fixed 20k and 34/40 for fixed 80k. Player-1 replies instead matched fixed
20k 26 times and fixed 80k 22 times.

This disproves the intended stabilization mechanism. The evidence cannot
distinguish production clock/compiler/environment divergence from an asserted
source that was not the arena source, because the collector binding is
explicitly `asserted-not-api-verified`. The next foundational enabler is
API-independent deployed-source attestation or deterministic work telemetry in
production, not another round of cap fishing. History 67 is rejected on its
rank, clean elite result, and failed mechanism check.

Commit `a7dd201dbaf32b98f6d661fe4b076c4c769e1815` restored the exact
history-62 schedule and regenerated the 94,771-character source SHA-256
`653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`.
It was launched as agent `6612745`, submission `41128812`, history 68. This is
the current remote rollback, but its arena window was still running when this
ledger entry was frozen; no result is claimed. Rank 1 was not reached.
