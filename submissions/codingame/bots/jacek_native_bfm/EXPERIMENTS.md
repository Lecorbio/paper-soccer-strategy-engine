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
- 800/165 ms is the deployed clock profile, tested construction-inclusively
  against 900/180 ms pre-upload ceilings (inside 1,000/200 ms operational
  limits); and
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
| Submission artifact | 94,528 characters, SHA-256 `8e67a0c795809e17490f719b2130d172c8aea2fd8df51ad0d44ca2d97614c1e3`; 471 characters below the 94,999 project cap |

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
6. **Runtime:** both colors at 800 ms first decision and 165 ms later decisions,
   each in a fresh process; fail at 900/180 ms construction-inclusive wall
   time. Node counts are diagnostics, not the pass criterion.
7. **External comparison:** procedural in-memory openings, both colors, fixed
   work for CI reproducibility, then equal-clock 800/165 against canonical
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
| P0 | 400.968 ms | 167.306 ms | 81,739,776 bytes | 76,382,616 bytes |
| P1 | 405.716 ms | 168.314 ms | 91,881,472 bytes | 78,610,840 bytes |

Both processes pass the construction-inclusive 900/180 ms pre-upload ceilings.
The official operational limits remain 1,000/200 ms. These measurements are a
local safety gate, not a substitute for CodinGame execution or strength data.
The later actual-clock batches were less comfortable: the decisive bootstrap
gate reached 179.918 ms, only 0.082 ms below the ceiling, and the external Rank
4 screen reached 180.513 ms and recorded one headroom failure. The standalone
probe therefore cannot be treated as robust upload safety evidence.

## Frozen checkpoint-to-checkpoint gates

Native model iterations must first beat the prior runtime checkpoint with the
same engine and production search limits. The runtime loader validates and
prints the complete-file, model-artifact, and packed-weight SHA-256 values.

1. Fast screen: 500 paired openings (1,000 games), 50/10 ms, at least 530 wins.
2. Decisive screen: 106 paired openings (212 games), 800/165 ms, at least 112
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
both engines at 800/165 ms. The terms are intentionally explicit:

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
condition. Per the frozen ladder, development stopped: there was no second
seed, no 424-game parity gate, and no CodinGame upload.

## Results ledger

| Evaluation | Candidate | Reference | Candidate colors | Unfinished / headroom / operational | Clock |
| --- | ---: | ---: | ---: | ---: | --- |
| Seed13 vs seed12 sibling diagnostic | 484 | 516 | 244 / 240 | 0 / 0 / 0 | 50/10 ms |
| Bootstrap fast vs untrained | 689 | 311 | 363 / 326 | 0 / 0 / 0 | 50/10 ms |
| Bootstrap decisive vs untrained | 144 | 68 | 76 / 68 | 0 / 0 / 0 | 800/165 ms |
| External Rank 4 development | 35 | 71 | 16 / 19 | 0 / 1 / 0 | equal 800/165 ms |
| External final parity | not run | not run | not run | stopped | 424-game design |
| Exact-source live batch | not run | field | not run | stopped | upload withheld |

The bootstrap passed both comparisons with its untrained ancestor, but lost
decisively to the external Rank 4 reference and breached the local timing
headroom once. The supported conclusion is negative: this is an auditable
starting checkpoint, not a promotion, parity, or superiority result. The
ASan+UBSan verification suite passed, but functional safety does not override
the failed strength and timing gates.
