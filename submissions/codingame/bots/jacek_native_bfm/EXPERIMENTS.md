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

## Exploratory live diagnostic baseline

The user authorized a new exploratory CodinGame diagnostic even though local
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

This entire live batch is development-contaminated as soon as it informs that
choice. It is excluded from promotion evidence and training-policy labels. No
promotion, parity, or superiority claim follows from the raw rank or clean
score.

The behavior-preserving diagnostic successor to the uploaded source is 94,771
characters with SHA-256
`ac63ab602e6b837032fd2e88e2d8ca07e56ebabde956587e005270f45fcaad93`.
It short-circuits duplicate-boundary tactical classification and adds
root/non-root truncation, depth, leader-change, entropy, margin, and cap
telemetry. It is local research code, not a second live submission.

The corrected round-two pipeline pilot validated 32 paired-schedule games and
1,430 augmented samples. Of 122 reanalysed boundaries, 21 passed the frozen
30k-to-100k stability contract, including 17 non-exact auxiliary labels; zero
had a deadline interruption. A deterministic two-epoch train/export smoke also
produced byte-identical repeated JSON artifacts and a loadable 14,268-byte
runtime. These are pipeline checks, not strength results. The full league,
restart corpus, three-seed training round, and actual-clock selection remain
pending.

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
| External final parity | not run | not run | not run | stopped | 424-game design |
| Exact-source live batch, raw | 52 | 38 | 29 / 23 | our operational 0 | CodinGame 90 games |
| Exact-source live batch, clean | 41 | 38 | 21 / 20 | 11 opponent forfeits removed | CodinGame 79 games |

The bootstrap passed both comparisons with its untrained ancestor, but its
historical 800/165 ms source lost decisively to the external Rank 4 reference
and breached local timing headroom once. The current 800/155 ms source has
better measured timing margin and is an auditable exploratory baseline, not a
promotion, parity, or superiority result. The completed live batch is now
diagnostic development evidence and cannot serve as an independent holdout.
