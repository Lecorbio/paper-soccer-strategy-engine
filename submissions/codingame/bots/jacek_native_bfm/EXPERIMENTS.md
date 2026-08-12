# Native Jacek-style BFM experiment record

## Objective and isolation

This track tests whether a clean implementation of Jacek Dermont's publicly
described complete-turn BFM/UCT and neural-input architecture transfers to the
CodinGame Paper Soccer league. It does not start from the repository's best
bot. Production purity is enforced mechanically: incumbent engines, replay
books, replay-trained evaluators, residual labels, and alpha-beta search are
forbidden from the candidate source manifest and generated submission.

The canonical `rank_4` source appears only in `comparison_gate.cpp` as an
external benchmark. It is not a source of actions, features, weights, search
code, or training examples.

## Public facts, explicit adaptations, and unknowns

The reference revision is [QtPaperSoccer
`366d5304c09c2c820bd3ef4ea94624c034b8d955`](https://github.com/jdermont/QtPaperSoccer/tree/366d5304c09c2c820bd3ef4ea94624c034b8d955).

Publicly inspectable facts include:

- a 250-action generation limit;
- shuffled primitive neighbors;
- a deque that uses fifteen back/LIFO extractions followed by one front/FIFO
  extraction;
- current worker defaults `alpha = 0.35`, `FPU = 0.5`, `C = 0.95`, root
  exploration `Croot = 1.0`, and virtual-loss scale `0.5`;
- player-relative minimax backup over complete-turn children;
- partial first-layer neural reevaluation; and
- a final public-code ordering term of `heuristic + log(games + 3)`.

The public neural-input article discloses a separate CodinGame design:
316 used-edge flags, 105 eight-way true-turn-distance buckets, a 180-degree
Player 2 rotation, two 32-unit hidden layers, one thread, and an approximately
200 ms setting. The pinned desktop code has evolved to support other, richer
input schemas. Therefore the current desktop source is not evidence for the
unpublished competition feature revision or checkpoint.

This experiment deliberately adapts the public behavior:

- the user-specified deque schedule is 9:1, not the public desktop 15:1;
- generated actions deduplicate by complete boundary `PositionKey`, a stronger
  equivalence than a shared endpoint vertex and behavior not disclosed by the
  pinned generator;
- one thread replaces the desktop worker pool and virtual-loss coordination;
- seeded randomization makes fixed-work runs reproducible;
- repository-neutral rules and position types replace Qt application classes;
- a separate bounded 64-path tactical witness pass precedes ordinary 9:1
  sampling and reports its own truncation;
- 800/155 ms is the current exploratory deployment clock profile, tested
  construction-inclusively against 900/180 ms pre-upload ceilings (inside
  1,000/200 ms operational limits); the frozen historical gates below remain
  recorded at their actual 800/165 ms setting; and
- the network is trained from scratch without upstream weights or protected
  replay data.

Unknown and therefore unreproducible details include Jacek's CodinGame source
revision, exact weights, corpus, target construction, optimizer, augmentation,
random seeds, quantization, competition-time constants, and any private
post-processing. No local result can justify calling this an exact reproduction.

## Bounded tactical-witness guarantee

Every expanded boundary first receives a FIFO witness pass capped at 64 partial
paths. For each class it reaches, the generator retains the lexicographically
smallest complete-turn `ForcedCutoff`, `SafeHandoff`, or
`OpponentImmediateGoal` witness and feeds it through the cap-prioritized
tactical top-k. Attacking-goal and own-goal paths are handled separately by
exhaustive rebound-component reachability.

This contract is deliberately one-sided: every retained witness is exact, but
absence is not proof that the class is absent when the proof queue or deadline
truncates. `tactical_proof_paths`, `tactical_classes_found`, and
`tactical_proof_truncations` make that boundary visible in self-play,
comparison, and replay-auditor output. Ordinary 9:1 sampler work is counted
separately.

## Build and corpus provenance contract

The native workflow compiles the self-play producer from an ordered, frozen
nine-file source contract. Before generation it creates canonical compact JSON
with exactly one trailing newline and archives it as `build-provenance.json`
beside the shards. The report contains:

- the nine repository-relative source paths and SHA-256 values plus the
  recomputed ordered producer hash;
- path-independent canonical build arguments;
- the compiler basename, compiler-binary hash, full version text, and version
  hash; and
- the archived `selfplay-binary` name and SHA-256. Both workflow copies are
  checked executable, and the archived copy is hash-checked before generation.

Every game must declare the exact build-provenance and producer SHA-256 values
from its directory. The corpus validator rejects a missing or noncanonical
record, reordered or unsafe source paths, inconsistent producer hashes,
game/report disagreement, and forbidden dependency or machine-path text. It
deduplicates full contracts by content hash and carries both hashes and full
contracts through the corpus summary into the model artifact.

Training uses the strict local mode: it also rehashes the current nine source
files and archived producer and verifies the current compiler binary and
version. Portable validation of an immutable archived corpus always preserves
the canonical structural/hash checks, while intentionally not requiring that a
future checkout still have the original compiler installed. The model and
purity validators then bind and revalidate the complete contract and native
checkpoint ancestry; an arbitrary claimed SHA is never trusted.

The initial hardened run is 10,000 ground-up self-play games at opening depth
zero with the same frozen untrained native checkpoint on both sides. It has no
incumbent source, replay data, policy actions, or live/arena labels. Depths 0,
4, 8, and 12 are reserved for later native checkpoint leagues, whose declared
ancestry must lead back to that seed. The frozen generation command was:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/jacek_native_workflow.py generate \
  --output-dir results/jacek_native_bfm/bootstrap-v1-10k-hardened-seed-73194721 \
  --binary build/jacek_native_selfplay-bootstrap-v1-hardened \
  --compiler clang++ --games 10000 --seed 73194721 --work 2048 \
  --samples-per-game 100 --shards 14 --parallel 14 --temperature 3.0 \
  --temperature-turns 12 --opening-depths 0 --reanalysis-work 0 \
  --max-complete-turns 384 \
  --player-one-checkpoint models/jacek_native_untrained_seed.runtime \
  --player-two-checkpoint models/jacek_native_untrained_seed.runtime
```

The manifest records the deterministic configuration and all 14 shard hashes.
The corpus validator then rechecked the local build before three-seed training;
the retained model was exported only after those checks passed:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/jacek_native_corpus.py \
  results/jacek_native_bfm/bootstrap-v1-10k-hardened-seed-73194721/shard-*.jsonl \
  --report results/jacek_native_bfm/bootstrap-v1-10k-hardened-seed-73194721/corpus-report.json \
  --verify-local-build
PYTHONDONTWRITEBYTECODE=1 python3 tools/train_jacek_native.py \
  results/jacek_native_bfm/bootstrap-v1-10k-hardened-seed-73194721/shard-*.jsonl \
  --output results/jacek_native_bfm/bootstrap-v1-10k-hardened-seed-73194721/trained-model.json \
  --seeds 20260811,20260812,20260813 --epochs 50 --patience 8 \
  --batch-size 256 --learning-rate 0.001 --weight-decay 0.00001 \
  --auxiliary-weight 0.25 --qat-epochs 4
```

## Model reproducibility ledger

The retained artifact is a 10,000-game **hardened bootstrap**. It establishes a
ground-up data, trainer, quantizer, exporter, C++ inference, and provenance
baseline; it does not by itself establish playing strength. No sample supplied
a stable or exact reanalysis auxiliary target, so the effective labels in this
run are final mover-relative outcomes only.

| Item | Retained evidence |
| --- | --- |
| Feature/model schema | `canonical-edges316-onehot-true-turn-distance105x8-v1`; `jacek_native_model/v1`; schema SHA-256 `dd36c1b2800620fab1d5dc88afe95fcbb13864d581a18f01d26b3e1c3a4a6dfd` |
| Dimensions/activation | Bias-free `1156 -> 32 -> 32 -> 1`; square/leaky, leaky-ReLU, fast `tanh` |
| Self-play run | 10,000 games in 14 shards, seed `73194721`, 522.476 s, 19.1396 games/s; manifest SHA-256 `dce7fb5017b0dec93f6b69dca7f2b7aa4e4e06a02592cd5b2df4d74931a032b9`; report SHA-256 `bc8db8bbfe3954dd27d44f50f2bde57a7f79d3442450ecb1c7775d17bc934100` |
| Build identity | Build-provenance SHA-256 `19e5b3d73b2dd2345fb647c8836177b906544dbb17e78ddae5dd5dafc6796919`; producer `8d53e59681d7ae9fa2eb0d8cc279945fbf78e49cf40dc29ac139d272eab24542`; archived binary `6eb2b47c658f50c4c1d08d76e79a770e2775cce73c0c0eb9c28278981d04e07d` |
| Seed/runtime identity | Untrained runtime SHA-256 `6773c3b76c8df3e5b824d524bed938b45263215bd3198295f3bed1d082c8c6c2`; source model `bd0cb5d0e85c8237d20e080e7bc8c6695ba9aa493242b52f95da6fb53d44fb09`; packed seed `d63676c280ebdad85d8663edf6c70bfc9015020ff96b7e0b6db4b88a564efec2` |
| Corpus provenance | 10,000 depth-zero games, 1,106,236 augmented samples; corpus SHA-256 `c0200bd13c300081187b544098f4dcf2823f907a39a6769c58cfd8b5af04c9e6`; no protected/live replay actions or incumbent labels |
| Whole-game split | 8,000/999/1,001 train/validation/test games; winner counts 4,020/3,980, 502/497, and 503/498; 884,764/100,260/100,931 retained rows after 0/10,112/10,169 cross-split overlap removals |
| Training | AdamW, batch 256, learning rate `0.001`, weight decay `1e-5`, maximum 50 epochs, patience 8, four QAT epochs; seeds `20260811,20260812,20260813`; seed `20260813` selected by minimum quantized validation MSE |
| Pipeline contracts | Workflow SHA-256 `66300709838982dce4c3eb2ce0b5677653ddf61fc974dc69b566671f5f27d6f2`; trainer `924ae40597377f4dd3823c7e4df8369756cb7a3713006c6cc576087af47b7089`; corpus validator `8ec6cf38bfb9a998464fb72eb1c7438b042d1f4b5fe1deea2970023777c089fb` |
| JSON artifact | 3,516,698 bytes; SHA-256 `19f954092bea404ab18ccc7aaec8b7f6627f0b459017a7f83b6d666b6bb03acc` |
| Packed/header artifact | 14,268 packed bytes, SHA-256 `7125339d76ade22b0d8e3de249876927b99611372ff81396994c074522394218`; 21,736-character header, SHA-256 `b9e6e5765bfc6f69e18a968c06e2f92825f91dfd3732176d94f7cd43608af43f` |
| Inference parity | Independent initial-state golden `-0.000181343639`; C++ tolerance `2e-6`; rotation and partial/full evaluation tests pass |
| Quantized metrics | Train MSE `0.918911994`, sign `0.5638170`, mean `0.008104`; validation MSE `0.914289117`, sign `0.5619489`, mean `0.009275`; test MSE `0.917874098`, sign `0.5681208`, mean `0.008869` |
| Training resources | Total process 1,539.28 s; measured training-loop throughput 28,380.18 examples/s; peak RSS 7,829,962,752 bytes; peak footprint 8,256,755,952 bytes |
| Search telemetry | 559,786 searches; 25,713,027 expansions; 810,204,372 child evaluations; 2,214,423,957 completed actions; 1,336,772,058 ordinary partial paths; 1,498,776 generator truncations; 524,131 tree-cap searches |
| Tactical telemetry | 526,604,342 proof paths; 33,781,768 witness classes found; 4,019,650 proof truncations |
| Submission artifact | 94,528 characters, SHA-256 `3bda271b35695292324c4e1943062211d102d66b0bb69f43615ba7a0b89e6e20`; 471 characters below the 94,999 project cap |

The metrics are recorded for reproducibility, not advertised as holdout proof.
The train/validation/test values are similar and near a low-information
starting point; playing strength still requires the actual-clock gates below.

Opening depth zero is deliberate for this initial bootstrap: neither side can
depend on a previously trained checkpoint. The self-play executable separately
constructs and records deterministic complete-turn prefixes, and its nonzero
depth path is covered by byte-determinism and sanitizer tests. Later league
rounds use depths 0, 4, 8, and 12; those future settings must not be attributed
to this bootstrap corpus.

## Test ladder

1. **Rules and protocol:** direction round trips, own-goal handling,
   mover-loses blocking, atomic full-turn parsing, and mandatory rebound
   completion.
2. **Generator invariants:** no illegal or truncated action, maximum 250
   outputs, exact 9:1 extraction schedule, path-cycle safety, deterministic
   seeded order, uniqueness by full resulting boundary state, and a separate
   bounded tactical pass whose exact witnesses survive ordinary-path
   truncation while its own truncation remains observable.
3. **Feature and inference invariants:** frozen 1,156-index order, one active
   distance bucket per vertex, Player 2 rotation, artifact-schema checks, and
   cross-language inference parity.
4. **Search invariants:** legal complete action, deterministic fixed-work
   result, hard work/action caps, player-relative backup, and a legal fallback
   when the time budget is exhausted.
5. **Build/data lineage:** canonical source/compiler/binary/build records,
   mandatory per-game hashes, whole-game splits, checkpoint ancestry, and
   adversarial rejection of tampered or stale artifacts.
6. **Runtime:** both colors at the current 800 ms first decision and 155 ms
   later decisions, each in a fresh process; fail at 900/180 ms
   construction-inclusive wall time. Node counts are diagnostics, not the pass
   criterion.
7. **External comparison:** procedural in-memory openings, both colors, fixed
   work for CI reproducibility, then equal-clock 800/155 against canonical
   `rank_4` for the deployment decision.
8. **Live experiment:** upload the exact generated source even if preliminary
   local strength is imperfect, wait for a complete arena batch, archive the
   source hash and configuration, analyze replay decisions, and change one
   hypothesis at a time.

The read-only [replay decision auditor](REPLAY_DECISION_AUDITOR.md) compares a
recorded candidate action with fresh native alternatives without turning that
diagnostic replay into training or promotion data. Its output includes search
and root witness counts/truncation, candidate coverage and ordinal, alternative
scores, and the source/checkpoint identities needed to interpret the result.

## Frozen Release clock evidence

The production timing probe includes model construction, search construction,
complete-turn selection, and application. Each color runs in a separate fresh
process. On the recorded local Release build:

| Process | First response | Later response | Maximum RSS | Peak footprint |
| --- | ---: | ---: | ---: | ---: |
| P0 | 425.910 ms | 157.571 ms | 81,461,248 bytes | 76,497,304 bytes |
| P1 | 397.385 ms | 157.921 ms | 91,504,640 bytes | 78,791,088 bytes |

Both processes pass the construction-inclusive 900/180 ms pre-upload ceilings.
The official operational limits remain 1,000/200 ms. These measurements are a
local safety gate, not a substitute for CodinGame execution or strength data.
A 24-game actual-clock safety screen against canonical `rank_4` finished 8-16,
with candidate colors 5/3, zero unfinished games, zero headroom failures, zero
operational failures, and a 166.726 ms candidate maximum later response. Its
small score is not strength evidence; it is operational evidence for an
exploratory upload of the exact 800/155 ms source.

The earlier 800/165 ms batches remain historical evidence: the decisive
bootstrap gate reached 179.918 ms, only 0.082 ms below the ceiling, and the old
external Rank 4 source reached 180.513 ms and recorded one headroom failure.

The post-upload seed-fidelity check used the same 24-game actual-clock setup
twice. The deployed constant shuffle seed scored 4-20; explicitly varying the
candidate shuffle seed scored 5-19. This one-game difference does not support
seed mismatch as the main cause of the live and local strength gap.

## Frozen checkpoint-to-checkpoint gates

Native model iterations must first beat the prior runtime checkpoint with the
same engine and production search limits. The runtime loader validates and
prints the complete-file, model-artifact, and packed-weight SHA-256 values.

1. Fast screen: 500 paired openings (1,000 games), 50/10 ms, at least 530 wins.
2. Decisive screen: 106 paired openings (212 games), 800/155 ms, at least 112
   wins overall and at least 50 wins in each color.
3. Both screens require zero illegal/empty actions, unfinished games,
   operational timeouts, and 900/180 ms headroom failures.

These are actual-clock gates. Fixed-work self-play chooses data and diagnoses
regressions, but cannot promote a checkpoint.

The retained seed-20260813 runtime SHA-256 is
`877ee8d0afdb20cf3466bee4c09f654d33c6ac4ecc230b8022f570a31e60f93d`.
The frozen untrained bootstrap baseline is
`6773c3b76c8df3e5b824d524bed938b45263215bd3198295f3bed1d082c8c6c2`.
Both official bootstrap gates passed:

1. Fast: 689-311 in 1,000 games, candidate colors 363/326, zero unfinished,
   headroom failures, or operational timeouts. Candidate maxima were
   51.1998/12.6256 ms and baseline maxima 50.7695/13.4069 ms. Elapsed time was
   468.88 s and maximum RSS was 39,845,888 bytes.
2. Decisive: 144-68 in 212 games, candidate colors 76/68, zero unfinished,
   headroom failures, or operational timeouts. Candidate maxima were
   451.303/179.918 ms and baseline maxima 460.284/174.228 ms; the candidate
   reached the frozen 80,000-node tree limit. Elapsed time was 1,637.20 s and
   maximum RSS was 418,840,576 bytes.

A separate 50/10 ms diagnostic compared the selected runtime to sibling
seed-20260812 runtime
`d6236e108fcb563d630db73c8e0c41a009efb8dbd3fc2f7e05f529daae685c4f`.
It lost 484-516, with candidate colors 244/240, zero unfinished games or timing
failures, candidate maxima 51.1276/11.3426 ms, and sibling maxima
50.7535/11.2645 ms. This is a failed and inconclusive sibling diagnostic, not a
previous-champion promotion gate. Together the runs show that training improved
over the untrained seed but that validation-MSE seed selection did not pick the
stronger sibling at the fast clock.

## Frozen external Rank 4 gates

External comparisons use canonical `rank_4` solely as the opponent and keep
both engines at the current 800/155 ms deployment profile. The terms are
intentionally explicit:

1. Development screen: 53 paired openings = 106 games, at least 58 candidate
   wins. Repeat it under a second deterministic opening seed before treating
   the result as a stable development signal.
2. Final parity declaration: 212 paired openings = 424 games. Require a 95%
   Wilson lower bound of at least 50% for total candidate win rate (233 wins
   when all games finish) and at least 102 wins in each color (48% of the 212
   games played in that color, rounded up).
3. Both gates require zero unfinished games, headroom failures at 900/180 ms,
   or operational timeouts at 1,000/200 ms. The final command uses 53 openings
   at each of depths 0, 4, 8, and 12.

`comparison_gate` computes the Wilson interval and enforces total-win,
per-color, and confidence-bound thresholds directly. It also reports the
exact production 250/4,000/512/80,000 candidate profile and maximum response
times; fixed-work clocks of zero are exempt from clock classification and are
not promotion evidence.

The exact submission source
`8e67a0c795809e17490f719b2130d172c8aea2fd8df51ad0d44ca2d97614c1e3`
failed the development screen 35-71. Candidate colors were 16/19, total win
rate 0.330189, and Wilson 95% lower bound 0.24798. All games finished and no
operational timeout occurred, but the candidate had one 180.513 ms later
headroom failure; the reference maximum was 167.592 ms. First-response maxima
were zero because every opening began at depth eight.

Candidate totals were 1,301,714 expansions, 44,991,248 child evaluations,
131,434,625 completed actions, 61,492,440 duplicate boundaries, 75,276,791
ordinary partial paths, 28,556,626 tactical proof paths, 72,447 generator
truncations, a 65,113-node maximum tree, and 315,248 ms. The reference used
261,779,951 nodes and 306,140 ms. The process took 621.45 s, with maximum RSS
401,637,376 bytes and peak footprint 105,136,800 bytes.

The run failed both the 58-win threshold and the zero-headroom-failure
condition. Per the frozen ladder, that phase stopped: there was no second seed,
no 424-game parity gate, and no CodinGame upload of that historical source.

## Historical history-61 live diagnostic baseline

The user authorized this exploratory CodinGame diagnostic even though local
strength is not established. The only production change from the historical
artifact is the later search budget reduction from 165 ms to 155 ms. The
uploaded generated source is 94,528 characters with SHA-256
`3bda271b35695292324c4e1943062211d102d66b0bb69f43615ba7a0b89e6e20`;
the model, packed weights, and generated header identities are unchanged.

The exact source passes generated-source compilation, protocol and native
tests, ASan+UBSan, the fresh-process timing probe above, and the 24-game
operational safety screen. It was uploaded from Git commit
`8cf6005aace930016b86ac05de2ac8743447612c` as agent `6609056`, submission
`41123817`, history version `61`. The editor contents were copied back and
matched the generated-source hash, but no API source verification was
performed in this flow. The source binding is therefore
`asserted-not-api-verified`, not stronger.

The complete 90-game batch scored 52-38 raw, with colors 29-15 and 23-23,
rank 9, and score 39.54. The candidate had zero operational failures. Eleven
wins were opponent forfeits, comprising seven illegal-action failures and four
timeouts. Removing them left 79 clean games at 41-38, with colors 21-15 and
20-23. Clean frozen-cohort results were 4-25 against the top 5, 6-32 against
the top 10, and 22-37 against the top 20.

The provenance identities are:

| Evidence | Archive path | SHA-256 |
| --- | --- | --- |
| Complete-batch manifest | `results/codingame_arena_diagnostics/manifests/6609056/41123817/3bda271b35695292324c4e1943062211d102d66b0bb69f43615ba7a0b89e6e20/0328bded1916af5bd34554bbd315577cc346b7ba2e32b83f34ef3ef0e30351cf.json` | `0328bded1916af5bd34554bbd315577cc346b7ba2e32b83f34ef3ef0e30351cf` |
| Clean auditor TSV | `results/codingame_arena_diagnostics/runs/jacek-native-41123817-20260811/clean-auditor.tsv` | `d5cea44b03a340f220fcb5d2f4864c59151bfd25ad659302bf4c0ead1768b79b` |
| Frozen 30k-work decision audit | `results/codingame_arena_diagnostics/runs/jacek-native-41123817-20260811/native-decisions-fixed30k.jsonl` | `7f06835b8cfc0e4a8a51ff02195aed12d06a729af70a66cf0ddced0cafd86fee` |
| Canonical audit summary | `results/codingame_arena_diagnostics/runs/jacek-native-41123817-20260811/native-decisions-fixed30k-summary.json` | `4d9d56bc1c66c8cac6366c64b4b2c2683bdd5a9f0302c45591c76a57a672972b` |

The fixed-30k audit covers 1,918 decisions in all 79 clean games: 1,070
`bfm-override`, 702 `match`, 136 `initial-evaluator-ordering`, six
`generator-omission`, and four `operational-failure`. The labels describe a
comparison with a deterministic fixed-work search; they are not labels of the
correct move and do not show that an alternative would have won. Generator
omissions are rare in this sample. The next native iteration therefore focuses
on evaluator signal, reanalysis, and BFM allocation rather than treating action
coverage as the primary weakness.

This entire history-61 batch is development-contaminated as soon as it informs that
choice. It is excluded from promotion evidence and training-policy labels. No
promotion, parity, or superiority claim follows from the raw rank or clean
score.

The behavior-preserving diagnostic successor to the history-61 source was 94,771
characters with SHA-256
`ac63ab602e6b837032fd2e88e2d8ca07e56ebabde956587e005270f45fcaad93`.
It short-circuits duplicate-boundary tactical classification and adds
root/non-root truncation, depth, leader-change, entropy, margin, and cap
telemetry. This pre-activation artifact was local research code; the later
seed-`20260822` activation produced the distinct exact source uploaded as
history 62 below.

The historical corrected round-two pipeline pilot validated 32 paired-schedule games and
1,430 augmented samples. Of 122 reanalysed boundaries, 21 passed the frozen
30k-to-100k stability contract, including 17 non-exact auxiliary labels; zero
had a deadline interruption. A deterministic two-epoch train/export smoke also
produced byte-identical repeated JSON artifacts and a loadable 14,268-byte
runtime. These were pipeline checks, not strength results. The full league,
restart corpus, three-seed training round, and actual-clock selection described
below supersede the pilot's pending status.

Before the full round, a trainer audit reproduced the round-one quantization
failure mechanism: max-abs scaling gave `w1` a `0.5258222222` step and retained
only 250 of 36,992 weights. Round two now excludes max-abs from eligible scale
candidates, uses deterministic lower-rank `p800,p900,p950,p975,p990,p995`
clipping candidates, performs two-pass per-layer validation coordinate search,
then keeps the chosen scales fixed throughout straight-through QAT. Float,
pre-QAT, QAT and provisional seed selection all minimize combined-target MSE,
so exact solved labels retain their specified override. The final parameters
are exact exporter-idempotent float32 `q * scale` tensors. An adversarial unit
test gives `w1` one `100.0` outlier among `0.02` weights and verifies that the
outlier cannot become an eligible scale, all 1,156 input rows remain represented,
two scale searches are identical, and exporter round-trip is exact.

The same 32-game pilot then compared one seed (`20260821`), two float epochs and
two QAT epochs under the old and new quantizers. This deliberately small A/B is
a trainer diagnostic, not playing-strength evidence:

| Pilot quantizer | Validation outcome / combined MSE | Turns 0-11 MSE / prediction std | W1 row coverage / nonzero weights | W1 scale |
| --- | ---: | ---: | ---: | ---: |
| Dynamic max with dynamic-scale QAT | `1.013657 / 1.012237` | `0.984105 / 0.209616` | `100% / 48.108%` | `0.0285402592` |
| Robust coordinate scales with fixed-scale QAT | `0.949965 / 0.948797` | `0.955827 / 0.140179` | `100% / 70.737%` | `0.0151568940` |

Two independent robust runs produced the identical model SHA-256
`8637b2a2b1958602949fef3b66df15455f6c2a63ecea2b7cdb78fc86138f6824`;
its 14,268-byte packed payload is
`778c9b342afcef30a1fed616f17c634bd53f65402e45f8e1e10dfb2e48a38b24`.
The robust run retained its pre-QAT checkpoint; QAT could not replace it without
a strict combined-MSE improvement. The pilot also exposed that none of its 24
selected turns 0-11 reanalyses were stable or exact, so early strength still
depends on the larger diverse outcome corpus and must be judged by the recorded
early-bin metrics and actual-clock matches. Dataset materialization now stores
uint16 feature indices and releases Python samples while packing each split,
reducing the steady sparse representation and avoiding simultaneous full native
and NumPy copies at the intended million-scale corpus size.

## Round-two corpus, model, activation, and live diagnostic

The completed cumulative corpus contains 22,238 games and has SHA-256
`87cf43fe841dfe7d00fc98ff8d560dfea10a9c3a1832b19eaf092fd0e07edf47`.
Its whole-game split contains 17,779/2,230/2,229 train/validation/test games and
1,755,307/198,858/197,724 retained samples after 0/21,961/23,469 canonical
cross-split overlap removals. The corpus records
`observed_move_policy_labels: 0`; a live transcript can construct a restart
position, but neither the observed candidate move nor the opponent move becomes
a policy or value label.

| Lineage component | Games | Frozen identity |
| --- | ---: | --- |
| Strict-current native league | 12,000 | manifests `4b5fd9d85c94cd6090e322326d6d85e36eb5032b3fb39782e115825e41d8b0ae`, `a11f22c251329ffdc36c9a8e6928c610dfe41829f8ec5350c73a63b57f918c83`, `5d980ffdf49c8ed41ada20c54bd1d5a186417bc85448061a00ae11f2dd2f3158`, `a87c9c0f0d5ad41ff5140a7021de3924b64075ab1b9c0eda60eb3ddda8b102c3`; build `c031aeffc9bbb6805c2956b43c90e5a87dc06b51a9b51b17c1bffacf1a104b7c`; binary `9224b4fd1597f442f9f4764b5787144e620b3380c81ca5f52efd5586e1af88ae` |
| Archived round one | 10,000 | manifest `dce7fb5017b0dec93f6b69dca7f2b7aa4e4e06a02592cd5b2df4d74931a032b9`; build `19e5b3d73b2dd2345fb647c8836177b906544dbb17e78ddae5dd5dafc6796919`; binary `6eb2b47c658f50c4c1d08d76e79a770e2775cce73c0c0eb9c28278981d04e07d` |
| Live-loss restart continuations | 238 from 119 prefixes | manifest `b9aeb2e6b903a2e8be42f2553b7cca6d64aca00ad2ea7846849e936dcc5bb192`; build `3c241eaa6f446c0fd0bd950bc03f1644c311108047588d570341d05ba41aff5b`; binary `dfd79e108b28946fa1968cd43d0ca27c401c2b499ae86c9afb7ef758a84619f2` |

The restart component is additionally bound to collector TSV
`d5cea44b03a340f220fcb5d2f4864c59151bfd25ad659302bf4c0ead1768b79b`,
arena manifest
`0328bded1916af5bd34554bbd315577cc346b7ba2e32b83f34ef3ef0e30351cf`,
exclusion registry
`ac5d335a8e084e782be93f9c53635896f16344f08e9164481dc7b54eaf923a60`,
and historical asserted source
`3bda271b35695292324c4e1943062211d102d66b0bb69f43615ba7a0b89e6e20`.
This is disclosed development lineage, not independent live promotion evidence.

The immutable 2,550,520-byte model JSON has SHA-256
`b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14`.
It retains seeds `20260821,20260822,20260823`, provisional seed `20260823`,
and `chosen_seed: null`. Actual-clock selection chose seed `20260822`. Its
quantized validation outcome/combined MSE is `0.918977857/0.906705831`, sign
accuracy `0.568647`; test outcome/combined MSE is
`0.924110413/0.912629976`, sign accuracy `0.571918`. The validation turns 0-11
bin remains difficult at `0.999271512` outcome MSE and `0.503871` sign
accuracy. These metrics describe the selected value model; the promotion
decision comes from the actual-clock matches, not validation loss.

All six gate transcripts use the frozen constant-shuffle-seed profiles and
record zero unfinished games, candidate/baseline headroom failures, and
candidate/baseline operational timeouts:

| Seed / profile | Result and colors | Candidate / baseline maxima | Status | Report / stdout SHA-256 |
| --- | --- | --- | --- | --- |
| `20260821` screen | 688-312; 421/267 | 51.1642/15.7647; 51.4621/11.3183 ms | pass | `2baee0a80f18b045357aef2686d58bacc89a1cf147e3699b1538be3e6c788ee2`; `8eeadd7083536d26e87d0971422767cb4d7ee025af1b1df146439acd1b1886e8` |
| `20260821` decisive | 130-82; 86/44 | 468.073/164.123; 465.607/162.886 ms | fail: second-color floor | `b44c1cec78c5c86421ea693af329662451ac9301665dd8fd3db0997721e3854a`; `56791ba0eaed563c9774b0b8eac3e070f27aa842320367d8f2bbc4a943b450d1` |
| `20260822` screen | 803-197; 419/384 | 51.4645/14.5919; 51.0783/14.9628 ms | pass | `41c8dc7c19f3ccacaf74de7e318fee129a4eedc7735ce5d0d2e940f2e3e9a983`; `bbdc5056c6aeb1d3e51362a6303b384179a758c628d5d70a9e3cf74bb25cd62c` |
| `20260822` decisive | 146-66; 84/62 | 472.797/162.355; 451.195/163.006 ms | pass | `28138eab2218e7a574dc4b7379fd19f8219a5efdbca5f107f507540f0761a98d`; `4e956927891e51b6380d63b10e4bcd487b7201c595c21e821cf8f490c7e34091` |
| `20260823` screen | 672-328; 415/257 | 51.2916/11.2645; 51.1338/13.5631 ms | pass | `662122aba29156f3400c34f0b4dc4b25068c9f05b5027583f8eb8efdbfe73f19`; `ac5b58aca9ae8635de89489e498da52e08adb9f1010072926b763a6d7740dd67` |
| `20260823` decisive | 133-79; 89/44 | 474.352/163.964; 458.947/169.484 ms | fail: second-color floor | `93ac04d0d020015d270738785b75225960d88280b4a4c62bd438d47f549db938`; `a23a6390b3e32e6fe858bcae6a2f524cb6635c53cc602d972de0c4e8d01a7f1d` |

Only seed `20260822` passed both gates. The promotion-eligible selection sidecar
has SHA-256
`5597e4228850cd44aac4adc5f11e3d6533e5528e3e04c51700d2f04b2cbe2cef`
and payload SHA-256
`3b8afae23304fbdb9505b6646ea8f7339ad70e652109aefefd057806ce83f529`.
It binds exact tested/deployed runtime
`17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3`
and packed weights
`e2304195d491d7b2d5ae1334a8341b38d67d315073accc37915885ede3c6a2cb`.
Deployment descriptor
`88092ac6601faac0f3da31bdaa1e2a5eca15bdb762b18810d450b33ee0d6ef2f`
activates that selection locally. The generated header/source identities are
`3c1a8ef97f6dc14b9eed64679bd698939380db6fb72181d0b45d1aea74bd3458`
(21,736 characters) and
`653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`
(94,771 characters). Pre-upload post-activation fresh-process timing measured P0 at
424.894/157.944 ms and P1 at 419.405/157.479 ms. Both pass the 900/180 ms
construction-inclusive pre-upload ceilings. This is local operational evidence
that preceded the live result below. Focused verification also passed immutable activation,
transitive purity, exact-source freshness, three GCC activation tests, the
AppleClang 21 Release build and seven focused tests, and four exact ASan/UBSan
tests. The sanitizer statement refers only to those four built targets; a broad
pattern that also names unrelated unbuilt targets is not evidence and is not
reported as a failure.

### Retained round-two history-62 live diagnostic

The exact selected source was uploaded from repository commit
`e1ae4c7c66a03d9a2c3b82ddf79adafcb7e0c661` as agent `6609905`, submission
`41124914`, history version `62`. The 94,771-character source SHA-256 is
`653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`,
the model SHA-256 is
`b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14`,
and the packed-weight SHA-256 is
`e2304195d491d7b2d5ae1334a8341b38d67d315073accc37915885ede3c6a2cb`.
Editor read-back matched the generated source, but the platform API did not
verify it, so the binding is `asserted-not-api-verified`.

The complete 90-game window scored 63-27 raw, with candidate colors 30-13 and
33-14, frozen rank 5, and score 42.68. The candidate had zero operational
failures. Nineteen wins were opponent forfeits: ten illegal actions and nine
timeouts. Removing them left 71 clean rule-terminal games at 44-27, with
colors 18-13 and 26-14. Clean cohort results were 11-14 against the top 5,
26-22 against the top 10, and 34-24 against the top 20. Named clean results
with at least three games were jacek 0-9, Deltaspace 6-1, Laars 5-2, Waffle3z
4-0, derjack 1-3, EricSMSO 3-3, and YurkovAS 7-0.

| Evidence | Archive path | SHA-256 |
| --- | --- | --- |
| Complete-batch manifest | `results/codingame_arena_diagnostics/manifests/6609905/41124914/653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90/bb5aaf7340ddee174ddf916aee2bfbcd4b3afb01f5d53bbc5c7d728bf610b4cf.json` | `bb5aaf7340ddee174ddf916aee2bfbcd4b3afb01f5d53bbc5c7d728bf610b4cf` |
| Clean auditor TSV | `results/codingame_arena_diagnostics/runs/jacek-native-round2-41124914-20260812/clean-auditor.tsv` | `4a25768aed11e7c4bc368e63bce8c335e420f02fee7e131d2b56dfa97ab048e2` |
| Fixed-30k decision audit | `results/codingame_arena_diagnostics/runs/jacek-native-round2-41124914-20260812/native-decisions-fixed30k.jsonl` | `9eb8d1741bbd11390c3d0b1c15ea0cac35f73931e6695635c2f5778d7c7ff8f1` |
| Decision-audit summary | `results/codingame_arena_diagnostics/runs/jacek-native-round2-41124914-20260812/native-decisions-fixed30k-summary.json` | `79d88d6f608d2a78e86dd1f6abca8a5406a9866a417489dc7b2d99beb17cdd46` |

The fixed-30k audit covers 1,811 decisions in all 71 clean games: 1,015
`bfm-override`, 715 `match`, 76 `initial-evaluator-ordering`, four
`generator-omission`, and one `operational-failure`; `boundary-equivalent` and
`tactical-miss` were both zero. Those counterfactual
classifications are diagnostic, not correct-move labels or evidence that an
alternative would have won. The 0-9 clean result against jacek remains a direct
live gap; neither rank 5 nor the aggregate result establishes Rank 4 parity.

## Eight-hour round-three campaign and rollback

### Cumulative training and exact checkpoint selection

The campaign added four balanced 1,000-game native league blocks and 164 fresh
continuations from 82 evenly selected history-62 loss prefixes. The four league
manifest identities are
`09aedcab65d0bbfc7c3f387245b3bab7bdc0ee3b10c5c87249d77d7398aa392b`,
`e99484ff0e81cc6f53f51778e0da5d02dfddba977e2fd6a1f9bafd344061959a`,
`56e5599004fc8331a4f6fe032ff88ed4f148385f07c9de84a1594c4a2941ef13`,
and `1246303233718cae60154851475281f69b09d4f3083329753a7c7c6b4541247f`;
the restart manifest is
`7e5fbd070a9c2863bd26ee4436de8e73615d1125c1cf56b06cad439d7149d8f2`.
Observed live moves constructed restart states only and remained ineligible as
policy or value labels.

The cumulative corpus contains 26,402 games: 10,000 archived round-one games,
12,000 existing native league games, the 4,000 new league games, 238
history-61 restart continuations, and the 164 new history-62 continuations. Its
SHA-256 is
`f58768b6090eb60968334310bd4ea7e62d9977b7a8318ba382803bdcbc1d3130`.
The whole-game split is 21,111/2,645/2,646 games and
2,030,276/231,690/228,884 samples for train/validation/test after
0/25,003/26,579 canonical overlap removals. Trainer SHA-256 is
`062c06047b3a8211ddd815ba46c666d40938e6b686eb71090ba533b805227daf`.
Seeds `20260831,20260832,20260833` used the existing robust fixed-scale
three-bit QAT contract. Phase weights `3.0`, `1.5`, and `1.0` apply to turns
`0-11`, `12-23`, and `24+` after the exact-outcome override has formed the
combined target; stable native reanalysis retains auxiliary weight `0.25`.

The immutable 2,664,686-byte model JSON has SHA-256
`a13b86accd168ae51cecf5df85092642ba8edb21d9ef944ac7c002bfa6a5d19a`.
The exact same-runtime gates against retained history-62 seed `20260822` were:

| Seed / profile | Result and colors | Status | Report / stdout SHA-256 |
| --- | --- | --- | --- |
| `20260831` screen | 610-390; 368/242 | pass | `42997bb3a91afe12731b0f0717d9ae12f8e086047206458246c9db4424076681`; `f269cfa837365bd909a808dec2afef0e86ea92e9d563b4dbd254a31953042096` |
| `20260831` decisive | 123-89; 78/45 | fail: second-color floor 50 | `2b15d07d4485b7ccb9d822f6be210390d8ba4e51b161d82b8ba3d5c7f347ba65`; `0c4b1da3429d9a1c77ee011d15ac722af72048a90aed4f304b167b25ca787a68` |
| `20260832` screen | 562-438; 252/310 | pass | `93909ad9e2e610e9078063b5c3d115fd4d72e5245b6a23158772cf553a4312bb`; `8446e45d57b3348a7ee006b7a518d5212d2ccf4adae1b1a334ab566fec5c5e39` |
| `20260832` decisive | 112-100; 56/56 | pass exactly | `b69278c7c843fbcefc44b79c1ba4e62cbe4ba54d78e83af3f988208af3b6399b`; `7adcdd2c084d3e34b88db8806945d9db7d1844bedf534d63404222c2452d4afa` |
| `20260833` screen | 540-460; 351/189 | pass | `274b6cf2ae28a122115fceec509071bccdeb244de7a141cb627b7c429f260c83`; `fdf5e37b12cbffcb95d3679a7b8470451b904e666249d7d702b4e6b47e015c95` |
| `20260833` decisive | 108-104; 75/33 | fail: total and second-color floors | `31b1a2400ca563878c06e981fe4a3e4a9b749fb02569e3138144b59e23a82152`; `4eaf793a6e7165e1b2e3aff3f0b25ca444e32210848a2bed8739055a279b5d4f` |

Every gate had zero unfinished games, candidate/baseline headroom failures,
and candidate/baseline operational timeouts. Seed `20260832` was the sole
passing seed. Its checkpoint SHA-256 is
`fedbab01a0f44e0612575aeee867bb846da2ec6eed7cd4383327dd67112213bd`,
runtime SHA-256 is
`0aaff836c1e96b949713c2a25b88456ca1f60f4599d5ce303f4f46fcc1ed7b52`,
and packed-weight SHA-256 is
`d8e1110caddf4bbd65a8c7e7d387c979c1a15807f29e5ab908d471e3159bab94`.
The 4,985-byte selection sidecar has SHA-256
`84cc694cf390fc6a0ddc19c5e61d5770a84172c31618d9d14576365e7e16f635`
and payload SHA-256
`6abb8884a7c8a8a72dd75360d0a7d8f1f0255159c8b9b6f3098a4f841b24e45c`.
The 7,137-byte immutable deployment descriptor has SHA-256
`a872281d7d3c458df38507578e01c3e5d663ac487a9cad3128685615b15b817a`.
It generated a 21,737-character header with SHA-256
`205a5a7c8ec67fa86df768024e7c6458d29e15a6d846079ef15a31cb4ed9cb47`
and a 94,772-character source with SHA-256
`5e628e5552da4a22f6dd3c73064d8f2670d13d3e3e6f23ce3ee93bdd79fb306a`.

### History-63 full-window live result

The exact round-three source was uploaded from commit
`cf3800e35dbb3dc870e23450c34d66484bc953a8` as agent `6611653`, submission
`41127173`, history version `63`. Editor read-back matched the generated source;
the platform API did not verify it, so its binding is
`asserted-not-api-verified`.

All 90 games completed and the full window was accounted for. The raw result
was 59-31, with candidate colors 32-20 and 27-11, frozen rank 6, and score
42.09. The candidate recorded zero operational failures. Four wins were
opponent forfeits: one illegal action and three timeouts. Removing those wins
left 86 clean rule-terminal games at 55-31, with colors 31-20 and 24-11.
Clean cohort results were 7-16 against the top 5, 23-27 against the top 10,
and 41-31 against the top 20. Named clean results with at least three games
were jacek 0-4, Marchete 1-3, Deltaspace 2-5, Laars 3-1, Snekkers 1-3,
Waffle3z 5-3, EricSMSO 3-4, derjack 4-3, YurkovAS 4-1, trictrac 5-4,
ILove47 3-0, and Spoonboy82 4-0.

| Evidence | Archive path | SHA-256 |
| --- | --- | --- |
| Complete-batch manifest | `results/codingame_arena_diagnostics/manifests/6611653/41127173/5e628e5552da4a22f6dd3c73064d8f2670d13d3e3e6f23ce3ee93bdd79fb306a/d57434f808654b25b78999bc1a9e9d1e97754225c51327675e2172084e863620.json` | `d57434f808654b25b78999bc1a9e9d1e97754225c51327675e2172084e863620` |
| Clean auditor TSV | `results/codingame_arena_diagnostics/runs/jacek-native-round3-41127173-20260812/clean-auditor.tsv` | `bf349361300b30063eb8e0d3f28802f3585ef9d556b227f6c75607663d80d95c` |
| Fixed-30k decision audit | `results/codingame_arena_diagnostics/runs/jacek-native-round3-41127173-20260812/native-decisions-fixed30k.jsonl` | `201d1343037a9104ffea2a68ffc4573a056b7319272646642e946960736a5a87` |

The strict analyzer joined all 2,075 audited decisions to all 86 clean games.
It classified 1,094 `bfm-override` (52.723%), 861 `match` (41.494%), 115
`initial-evaluator-ordering` (5.542%), two `generator-omission` (0.096%), and
three diagnostic `operational-failure` (0.145%); `tactical-miss` and
`boundary-equivalent` were zero. Actual-boundary retention was 99.904%. No
search or diagnostic-root deadline occurred. Search reached its 30,000-node
fixed-work cap on 91.13% of decisions, almost unchanged from history 62's
91.22%, so this audit does not support increasing the production node cap.

The strongest diagnostic is the opening evaluator shift. On the first own
decision, history 63's observed action had mean/median initial evaluator rank
5.837/7 and was top-five only 40.7% of the time; history 62 measured 3.211/3
and 94.4%. History 63's first fixed-30k choice matched the observed production
choice in 63/86 games (73.3%), versus 15/71 (21.1%) for history 62. Combined
with the clean top-five regression from 11-14 to 7-16, this identifies an early
evaluator/search-allocation regression. Generator widening, tactical patches,
and 120,000 nodes are not supported by these replays. The round-three live
candidate is rejected.

### Same-runtime search screens

All screens used exact history-62 weights on both sides, balanced pair-parity
colors, deployment-constant shuffling, and zero unfinished games, headroom
failures, or operational timeouts. Only the stated search field changed:

| Candidate | Candidate-baseline; colors | Profile and threshold result | Report / stdout SHA-256 |
| --- | --- | --- | --- |
| 120,000 versus 80,000 nodes | 65-63; 32/33 | 800/155; reject below 70 total | `8bcc269aba0e34543d7ca896a2d8846c6674decc60c39b4444a51986dc7b25c8`; `8c9c2036c44921a4ba596c7d4256f45017a75413f681f87c2d05103420a02f14` |
| Final `value + log(selectionVisits + 3)` | 63-65; 38/25 | 50/10; reject below 70 total and 31 second color | `45dcef903263a8b07de38403b201c5d583f88e80250cb32ab679cbe052dec52a`; `bcc4157eb1820cca753d9983c14a2c0d7225d919c35359595765051d85d530b0` |
| `C=0.65` | 73-55; 30/43 | 50/10; reject below 31 first color | `0fa284cefc82048c5135e1ce60ca9c541aa185b19fb4be40875c58b9b4d24bf7`; `1afb59cadda61ccaed4c77100ce4b920d89eebb0a5b6e5b6adb859fba8b3082c` |
| `C=0.65` decisive | 55-51; 34/21 | 800/155; reject below 58 total and 25 second color | `6c9db5729bd4f284f6b736c19077519eec25db9bb9176b96c6115f6a27d4631d`; `e0f754254ec7e1b6fec7ef7aef5b3a42b74998ce61287a8259641fb366897d86` |
| `C=1.25` | 43-85; 24/19 | 50/10; reject | `0e241235e32b09f3cb6617ed77b4a22eabfb0c0cb026c1ee5e2da31fde110d29`; `10a77b8ec0b426140d9095fd4eb28a014ba0f65eb04334f4af06511afed09411` |
| `FPU=0.25` | 57-71; 40/17 | 50/10; reject | `9c8ee163c1902db53a6b8d088049bd32bc6d9c8194bdcbf7fd79753340116732`; `8e925f4bca5680cbf4402737a3dd5c3f4190a7d30f89b77ca469830b6c9eed47` |
| `FPU=0.75` screen | 87-41; 41/46 | 50/10; pass | `80047c77d5b5f146ab45b247027e554723fe7667441741beb9d3fed4b8f7846b`; `aa5d798a891db547129de45f9fef092077e35ae3b78ce2e12edc867a3e850564` |
| `FPU=0.75` decisive | 55-51; 37/18 | 800/155; reject below 58 total and 25 second color | `6a5d0c7a55c3a3514faa321fddc5933848232b33993ca106185d6228f679c35f`; `2dbd6553e56155be23bea562dbf20f3436ef30e6554082b6291bab398547a08c` |

The 71-state first-decision clock panel also showed that 80,000 and 120,000
nodes changed 53 choices, but both completed all 71 by their node caps with no
deadline events. The paired 120,000-node score above did not qualify, so the
production cap remains 80,000. The apparent fast-clock `FPU=0.75` advantage did
not transfer to the decisive clock and failed badly by color. None of the
search hypotheses changes production constants.

### Exact history-62 local reactivation

The strongest completed operationally safe source is therefore retained
history 62. Immutable v2 deployment descriptor
`models/jacek_native_history62_reactivated_v2_deployment.json` has SHA-256
`31772c68cd9da04503e2fa760926ccbd825820d62d45584da3c857d2e6b26aa6`.
It selects seed `20260822`, model
`b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14`,
runtime
`17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3`,
and packed weights
`e2304195d491d7b2d5ae1334a8341b38d67d315073accc37915885ede3c6a2cb`.
The local production profile is exactly 80,000 nodes, `C=0.95`, `FPU=0.5`,
and `value + log(visits)`. It regenerates the original 21,736-character header
SHA-256
`3c1a8ef97f6dc14b9eed64679bd698939380db6fb72181d0b45d1aea74bd3458`
and 94,771-character source SHA-256
`653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`
byte-for-byte. Commit `0f3a95f50b42c8135fc929a0a8fe8ccf6756d4f2`
restored this identity; CI and Pages run
[31614166711](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/runs/31614166711)
completed successfully. The exact rollback is now both locally active and
remotely deployed as history 64.

### History-64 exact rollback full-window live result

The exact 94,771-character history-62 source SHA-256
`653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90`
was uploaded from commit `0f3a95f50b42c8135fc929a0a8fe8ccf6756d4f2` as
agent `6611801`, submission `41127479`, history version `64`. Editor read-back
matched the generated source. The platform API did not verify source bytes, so
the binding remains `asserted-not-api-verified`.

All 90 games completed and the full window was accounted for. The raw result
was 64-26, with candidate colors 31-13 and 33-13, frozen rank 6, and score
41.67. The candidate recorded zero operational failures. Eleven wins were
opponent forfeits: six timeouts and five illegal actions. Removing those wins
left 79 clean rule-terminal games at 53-26, with colors 26-13 and 27-13.
Clean cohort results were 6-10 against the top 5, 17-18 against the top 10,
and 31-23 against the top 20. Named clean results were jacek 0-4, Marchete
2-3, Deltaspace 4-0, Laars 0-2, and Snekkers 0-1. The raw jacek result was
1-4 because one candidate win was a jacek timeout.

| Evidence | Archive path | SHA-256 |
| --- | --- | --- |
| Complete-batch manifest | `results/codingame_arena_diagnostics/manifests/6611801/41127479/653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90/3edfab1632e9dcfca326c2d58f2951982352837e8f86e0ac9ab6085dfd817026.json` | `3edfab1632e9dcfca326c2d58f2951982352837e8f86e0ac9ab6085dfd817026` |
| Clean auditor TSV | `results/codingame_arena_diagnostics/runs/jacek-native-history64-41127479-20260812/clean-auditor.tsv` | `cd82d1ea00ca7061f7b7f95a9a62b289d40df38128d346b4b7f3da2e64ff4d2c` |
| Fixed-30k decision audit | `results/codingame_arena_diagnostics/runs/jacek-native-history64-41127479-20260812/native-decisions-fixed30k.jsonl` | `a2a804efe90d21a20af34530f833535c8f27eb5773d34687f8a3c0e1d3b1fbf8` |
| Fixed-30k summary | `results/codingame_arena_diagnostics/runs/jacek-native-history64-41127479-20260812/native-decisions-fixed30k-summary.json` | `e477ff03154320e1dafe5b8f77fa6dca3af18e500dd09358df7409d7b346dde1` |

The strict analyzer joined all 1,996 decisions to all 79 clean games. It
classified 1,135 `bfm-override` (56.86%), 777 `match` (38.93%), 75
`initial-evaluator-ordering` (3.76%), seven `generator-omission` (0.35%), and
two diagnostic `operational-failure` (0.10%); `tactical-miss` was zero.
Actual-boundary retention was 99.55%, and the search reached its fixed
30,000-node cap on 91.28% of decisions. The two diagnostic operational rows do
not contradict the manifest's zero live candidate operational failures.

The first-decision audit had 59/79 `bfm-override`, 19/79 `match`, and one
`initial-evaluator-ordering`. The observed action's initial rank had mean
2.684, median 3, and was top-five in 78/79 games; every first-decision search
reached the node cap. This is consistent with history 62's first-decision mean
3.211, median 3, and 94.4% top-five rate, and sharply unlike history 63's
5.837/7 and 40.7%. The exact rollback therefore restores the stronger opening
evaluator profile. Its 56.86% BFM-override rate and 91.28% tree-cap rate keep
search allocation as a diagnostic target, but this replay-conditioned batch
cannot independently promote a new search constant.

The final replay-grounded extension tested `C=0.65` against exact history-62
at the 800/155 ms production clock, opening seed `2026081203`, for 53 paired
openings and 106 games. It scored 55-51 with colors 34/21 and therefore failed
both the required 58 total wins and the 25-win per-color floor. Candidate
maximum first/later responses were 468.290/162.010 ms; baseline maxima were
487.884/165.192 ms. Both sides recorded zero unfinished games, headroom
failures, and operational timeouts. The content-addressed report has SHA-256
`6c9db5729bd4f284f6b736c19077519eec25db9bb9176b96c6115f6a27d4631d`;
its stdout has SHA-256
`e0f754254ec7e1b6fec7ef7aef5b3a42b74998ce61287a8259641fb366897d86`.
The search change is rejected and was not uploaded.

## Results ledger

| Evaluation | Candidate | Reference | Candidate colors | Unfinished / headroom / operational | Clock |
| --- | ---: | ---: | ---: | ---: | --- |
| Seed13 vs seed12 sibling diagnostic | 484 | 516 | 244 / 240 | 0 / 0 / 0 | 50/10 ms |
| Bootstrap fast vs untrained | 689 | 311 | 363 / 326 | 0 / 0 / 0 | 50/10 ms |
| Bootstrap decisive vs untrained | 144 | 68 | 76 / 68 | 0 / 0 / 0 | 800/165 ms |
| External Rank 4 development | 35 | 71 | 16 / 19 | 0 / 1 / 0 | equal 800/165 ms |
| Exploratory operational screen vs Rank 4 | 8 | 16 | 5 / 3 | 0 / 0 / 0 | candidate 800/155 ms |
| Constant-seed fidelity diagnostic | 4 | 20 | not a gate | 0 / 0 / 0 | candidate 800/155 ms |
| Varied-seed fidelity diagnostic | 5 | 19 | not a gate | 0 / 0 / 0 | candidate 800/155 ms |
| Round-two seed21 screen vs bootstrap | 688 | 312 | 421 / 267 | 0 / 0 / 0 | 50/10 ms |
| Round-two seed21 decisive vs bootstrap | 130 | 82 | 86 / 44 | 0 / 0 / 0; color floor failed | 800/155 ms |
| Round-two seed22 screen vs bootstrap | 803 | 197 | 419 / 384 | 0 / 0 / 0 | 50/10 ms |
| Round-two seed22 decisive vs bootstrap | 146 | 66 | 84 / 62 | 0 / 0 / 0 | 800/155 ms |
| Round-two seed23 screen vs bootstrap | 672 | 328 | 415 / 257 | 0 / 0 / 0 | 50/10 ms |
| Round-two seed23 decisive vs bootstrap | 133 | 79 | 89 / 44 | 0 / 0 / 0; color floor failed | 800/155 ms |
| Round-three seed31 screen vs history 62 | 610 | 390 | 368 / 242 | 0 / 0 / 0 | 50/10 ms |
| Round-three seed31 decisive vs history 62 | 123 | 89 | 78 / 45 | 0 / 0 / 0; color floor failed | 800/155 ms |
| Round-three seed32 screen vs history 62 | 562 | 438 | 252 / 310 | 0 / 0 / 0 | 50/10 ms |
| Round-three seed32 decisive vs history 62 | 112 | 100 | 56 / 56 | 0 / 0 / 0 | 800/155 ms |
| Round-three seed33 screen vs history 62 | 540 | 460 | 351 / 189 | 0 / 0 / 0 | 50/10 ms |
| Round-three seed33 decisive vs history 62 | 108 | 104 | 75 / 33 | 0 / 0 / 0; total/color failed | 800/155 ms |
| Search 120k vs 80k | 65 | 63 | 32 / 33 | 0 / 0 / 0; total failed | 800/155 ms |
| Search final selection-visits +3 | 63 | 65 | 38 / 25 | 0 / 0 / 0; total/color failed | 50/10 ms |
| Search `C=0.65` | 73 | 55 | 30 / 43 | 0 / 0 / 0; color failed | 50/10 ms |
| Search `C=0.65` decisive | 55 | 51 | 34 / 21 | 0 / 0 / 0; total/color failed | 800/155 ms |
| Search `C=1.25` | 43 | 85 | 24 / 19 | 0 / 0 / 0 | 50/10 ms |
| Search `FPU=0.25` | 57 | 71 | 40 / 17 | 0 / 0 / 0 | 50/10 ms |
| Search `FPU=0.75` screen | 87 | 41 | 41 / 46 | 0 / 0 / 0 | 50/10 ms |
| Search `FPU=0.75` decisive | 55 | 51 | 37 / 18 | 0 / 0 / 0; total/color failed | 800/155 ms |
| External final parity | not run | not run | not run | stopped | 424-game design |
| Historical history-61 live batch, raw | 52 | 38 | 29 / 23 | our operational 0 | CodinGame 90 games |
| Historical history-61 live batch, clean | 41 | 38 | 21 / 20 | 11 opponent forfeits removed | CodinGame 79 games |
| Retained history-62 live batch, raw | 63 | 27 | 30 / 33 | our operational 0 | CodinGame 90 games |
| Retained history-62 live batch, clean | 44 | 27 | 18 / 26 | 19 opponent forfeits removed | CodinGame 71 games |
| Rejected history-63 live batch, raw | 59 | 31 | 32 / 27 | our operational 0 | CodinGame 90 games |
| Rejected history-63 live batch, clean | 55 | 31 | 31 / 24 | 4 opponent forfeits removed | CodinGame 86 games |
| Current history-64 exact rollback, raw | 64 | 26 | 31 / 33 | our operational 0 | CodinGame 90 games |
| Current history-64 exact rollback, clean | 53 | 26 | 26 / 27 | 11 opponent forfeits removed | CodinGame 79 games |

The bootstrap passed both comparisons with its untrained ancestor, but its
historical 800/165 ms source lost decisively to the external Rank 4 reference
and breached local timing headroom once. The historically uploaded 800/155 ms
source has better measured timing margin and is an auditable exploratory
baseline, not a promotion, parity, or superiority result. That history-61 batch
is diagnostic development evidence and cannot serve as an independent holdout.
Round-two seed `20260822` remains the strongest completed native live source.
Round three passed its local checkpoint gate but regressed from rank 5 to rank
6 and from 11-14 to 7-16 against the clean top-five cohort. All isolated search
alternatives also failed their frozen total or color gates. The final local
and remote state is therefore the exact reactivated history-62 source, now
uploaded as history 64. All three completed history-62/63/64 batches are
development-contaminated and none establishes Rank 4 parity, particularly
given their combined 0-17 clean record against jacek. The campaign terminates
without uploading the failed `C=0.65` extension: history 64 remains deployed at
frozen rank 6, history 63 was the best new campaign candidate at rank 6, and
the historical history-62 batch remains the best recorded rank at 5. The
rank-1 target was not met.
