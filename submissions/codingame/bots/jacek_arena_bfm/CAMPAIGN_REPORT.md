# Final campaign evidence report

This report freezes the collection decision, immediate safety action, rollback
audit, fresh scratch/model expansion, offline source rejections, first CI
result, and measured live conclusion for the five-hour
`jacek_arena_bfm` campaign.

## Fresh construction and scratch evidence

- Campaign boundary: `2026-08-13T10:12:52Z`.
- Initial random-bootstrap corpus: 2,000 games, 30,659 value rows, manifest
  `72eb091180fc0191e679b97d1b789a05a13420bc8f661123f11069728344b60c`.
- Interim scratch continuation: 2,000 games, 76,562 value rows, manifest
  `3b77cdb7adf5936fe7358f92555475fc4a5fb624886f8a3ad177c275ce9fcab4`.
- Final pre-cutoff continuation: 4,000 games, 169,096 value rows, manifest
  `07aae233762fdbc86a8ef958c45dfc7200002102a6415779403d0d31a8164b50`.
- Cumulative valid fresh scratch corpus: 8,000 games and 276,317 value rows,
  balanced at 2,000 games for each opening depth `0,4,8,12`.  Its nine local
  dense row payloads total 853,559,862 bytes.
- Historical model weights, corpora, actions, labels, and replay content used:
  none.  Arena games used by the scratch continuation: none.

## First live candidate

| Field | Exact value |
| --- | --- |
| Model | `fresh-32x32-s101-7f23a40ba6ca` |
| Source SHA-256 | `3883f4c3f29a32c039492adc6151e94b5dfd84653ce0dfb2383356e7f5e3c9f8` |
| Source size | 88,664 ASCII bytes |
| Commit | `dbce4dec8ca2f31ef7790992dcddda7948eef132` |
| Agent | `6615613` |
| Submission | `41130787` |
| Upload | `2026-08-13T10:55:59Z` |
| Exact-window manifest | `8b8b3e5e59cda6817e54d17da7ef531a75bfd49ae5d0d7cefb1b650fb5795de0` |

The exact matching-submission window completed at 90 games: 74 were clean,
unambiguous rule terminals; three were timeouts attributable to this bot; 13
contained opponent operational failures.  The focus timeout game IDs are
`898882047`, `898882199`, and `898882273`.

At exact completion the candidate ranked 24. Its 74 clean games were 41-33
overall: 20-14 as player 0 and 21-19 as player 1. It had no clean top-five or
Jacek match before the rollback. These results cannot override the three
operational failures.

Any focus operational failure rejects the whole window.  Therefore all 90
games, including the 74 otherwise clean terminals, are excluded from training.
The exact fresh-arena usage report is:

| Use | Games |
| --- | ---: |
| State value | 0 |
| Opponent action ranking | 0 |
| Arena validation | 0 |
| Final live holdout | 0 |

The immutable rejection record is
`results/jacek_arena_bfm/reports/48224d76f1e42ea453bf5afe1ab96abfbcd9cf8081a017f15f936b2f573fe9a6.json`.

## Safety rollback

The three own timeouts disqualified the fresh candidate and immediately
triggered the authorized safe rollback:

| Field | Exact value |
| --- | --- |
| Source SHA-256 | `d9d96f83197f13b7212e7b652851097053ee7f1662845e06dd722d1c0bc24f71` |
| Source size | 99,810 ASCII bytes |
| Runtime | `C=0.95`, 80,000 nodes, 800/155 ms |
| Agent | `6615714` |
| Submission | `41130866` |
| Upload | `2026-08-13T11:13:22Z` |

The editor paste/copy-back matched exactly and Play My Code produced legal
stdout with expected H62 telemetry before submission.  The H62 artifact is an
evaluation/rollback source only; it is not imported into the new engine,
model, scratch games, actions, labels, or training lineage.

The exact rollback accounting window completed at 90 games under manifest
`44530074995e48754100ace955d4d75c16021947194b7e2c3e87a802b7de7cb9`:
76 clean rule terminals, 14 opponent operational failures, and zero H62
operational failures. Its rollback-only derivation is
`bc7336d949ec947740856c953e8f58660a90e43073891fe43cb3a19a723a82c9`.
The two-window campaign sequence validates as sequential and complete under
report `595e759d72eb6ee9ca75a4dc17a54a94cff3f1ab3affd03bad34ed55cb7edaa0`.
At exact completion the rollback ranked 9. Its 76 clean games were 49-27
overall, 4-17 against the frozen top five, and 0-5 against Jacek; the color
split was 20-14 as player 0 and 29-13 as player 1. The content-addressed
outcome summary is
`539905832f487bb8f23ef0e90b46640038eeada1be9ee31e8b54d3e4a7e1ffe1`.

CodinGame upload bytes are editor-attested by exact copy-back equality.  The
public CodinGame API does not expose the editor source bytes, so the attestation
cannot be replaced by an API readback.

## Mixture and selection disposition

The 25%, 40%, and 55% arena-exposure mixtures and their required two seeds
were not run.  No eligible fresh arena training window exists after the
whole-window rejection, so those exposure levels cannot be realized.
Scratch-only retrains may be evaluated separately, but they must not be named
or selected as arena mixtures.

Two otherwise identical one-epoch, random-initialized scratch-only retrains
were run on the then-cumulative 4,000-game corpus. Seed 101
(`fresh-32x32-s101-ad87209e4c6c`) was the interim offline selection over seed
1701 on frozen validation MSE (`0.883150` versus
`0.891634`) and sign accuracy (`0.627034` versus `0.618694`). The
content-addressed decision is
`4617ca6c044709110be852ae02e09aa76ee40091ebc0f6682cf439c9e40c9d5a`.
It remained offline and did not supersede the live H62 rollback.

The first checked-in cumulative namespace source combined that exact selected
model with the deadline-hardened engine:

| Field | Exact value |
| --- | --- |
| Model identity | `fresh-32x32-s101-ad87209e4c6c` |
| Model-header SHA-256 | `d3260f825e9bdfce5c56d579f1a693e9de4ae41a4fe1786f3ea8001ae91e67f8` |
| Training lineage | 4,000 fresh scratch games; 107,221 value rows |
| Arena value/ranking rows | 0 / 0 |
| Effective arena exposure | 0.0 |
| Generated-source SHA-256 | `88683044c0600d363d1d584e78af1edf59fa87e764f7a2d3889df0d7e503046b` |
| Generated-source size | 92,686 ASCII bytes |
| Exact source commit | `46a36220f9d1a00081f7b697f42b7692043620c4` |
| CodinGame deployment | Offline only; never uploaded |
| Frozen-H62 fast screen | 291-709; 1,000/1,000 clean; 500/500 colors; zero failures; fail |
| Actual-clock 212-game gate | Not run; mandatory beats-H62 gate failed |
| First repository CI | [Run 31697888489](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/runs/31697888489): GCC, Clang, ASan/UBSan pass |

That source failed the mandatory strength gate and did not replace the
deployed H62 rollback.  The immutable earlier pending-source snapshot remains at
`results/jacek_arena_bfm/reports/e82aeb1fc5a80987764be9243721ccd6ebc726a0463831aa954c7dec9ec6b5c3.json`;
its completed source report is
`results/jacek_arena_bfm/reports/a56c89b5de57bc0af4193e4b6621b5e435ed9413f92d10f81e24f34aa40973dd.json`.

The canonical 4,000-game search baseline kept the same evaluator and added
mover-relative ordering for generator hashes, edge/action
tie-breaks, beam classes, and sparse-feature accumulation.  It is 94,195
ASCII bytes with SHA-256
`1e4961e1347a3e12a9a7d5af500d22fbd44e4a851630461d8893e74e8f788591`
at `C=0.95`.  New exact tests require bit-equal color-swapped evaluation,
ordered generator equality under rotation, and equal fixed-node search
boundaries.  This canonical baseline remains an offline comparison and has
never replaced H62.

The predecessor's local exact-source generator gates retained every exact
goal, own-goal, block,
and forced witness with unique-boundary recall 1.000, zero illegal actions,
and zero rotational inconsistencies. Fixed-250, tactical-progressive, and
priority-beam p99 latency was 254/223/220 us under GCC and 705/497/494 us under
Clang. Dedicated uncontended construction-inclusive first/later maxima were
795.511/147.511 ms under GCC and 794.368/147.247 ms under Clang, below both
900/180 ms decision targets and 990/198 ms same-runtime maxima. Source
ASCII/size/purity, protocol, legality, GCC, and Clang gates passed; the linked
CI run also passed ASan/UBSan for that exact predecessor commit.  A final CI
identity for the canonical source awaits the final evidence commit.

Postmortem replay of the three exact pre-timeout states isolated unbounded
complete-turn generation across the deadline. The exact uploaded source
reproduced local maxima of `159.912`, `161.361`, and `162.114` ms against its
155 ms later-turn budget. The offline fix checks deadlines during generation,
reserves 6 ms for finalization, retains a deterministic legal emergency turn,
and uses a 128 ms later budget; its corresponding maxima were `124.844`,
`126.253`, and `126.734` ms. This diagnostic evidence is explicitly
nontraining and is bound by report SHA
`acb532751a214ffcdfaa02f55bb57df7d38be29d8615e8d6bd00d909f447c78b`.

The earlier offline-hardened source carrying the 2,000-game collection model
ran a separate exact 1,000-game fast screen against frozen H62, balanced
500/500 colors. All 1,000 games were operationally clean, but that source lost
`57-943`. Its isolated aggregate report has SHA
`ea8882f89199a09859c951d4da240e8b4f65d768909d42a606866470951587df`.

The cumulative 4,000-game source's distinct eight-worker screen completed
1,000 clean games and lost 291-709. Candidate first-decision p99 was 782.905 ms
and maximum was 1275.943 ms; later-decision p99 was 123.536 ms and maximum was
133.390 ms. These concurrent-screen values satisfy that harness's 1500/300 ms
hard limits but are not the dedicated uncontended 990/198 ms qualification
gate above. Strength failure blocks qualification, so the 212-game
actual-clock gate was not run. The exact aggregate report is
`results/jacek_arena_bfm/comparisons/final-cumulative-scratch-fast-1000.json`;
its SHA-256 is
`1785c1a7a53bd14a32b98f0c0d83c964d1a2eccb08e91a0c9f292290a67ba2cc`.

## 8,000-game random-init retrain and rejection

The full 8,000-game corpus was trained from random initialization only.  The
primary one-epoch 32-wide candidates were:

| Seed | Identity | Validation MSE | Sign accuracy | Small H62 screen |
| ---: | --- | ---: | ---: | ---: |
| 101 | `fresh-32x32-s101-778156217cbd` | 0.908276 | 0.601280 | 0-20 at `C=0.95`; advanced after search sweep |
| 314159 | `fresh-32x32-s314159-b7182ce143a7` | 0.908306 | 0.601400 | 2-18 at `C=0.95`; rejected |
| 1701 | `fresh-32x32-s1701-d31a201fc773` | 0.913370 | 0.597900 | 0-20 at `C=0.95`; rejected |

Seed-101 two- and three-epoch runs worsened validation to MSE
`0.915605` and `0.924737`; learning rates `.0001` and `.0003` also worsened
validation.  The seed-1701 `.0003` run was likewise worse.  The primary exact
model audit is
`5272ab83cf1a9e1324e6791e817f777a915947f597b9fbf9159c163a1a6344de`;
all lineage checks pass, with zero checkpoint, arena, historical, or H62
training inputs.

The 48-wide random-init run had validation MSE `0.914324` and sign accuracy
`0.599451`, and its 82,393-byte packed header yields a 121,383-byte canonical
source.  A 64-wide run was not repeated because the existing fresh-shape
packed header is already 109,589 bytes before the non-model body; with the
canonical body it would be 148,579 bytes.  Both widths are rejected by the
99,999-character contract independently of their metrics.

The 8k seed-101 evaluator advanced with the canonical engine and `C=0.25` in
a 94,194-byte source, SHA-256
`9373f392ffc426c8e6d61843277d7612fe3536cc0abd59baf88f68972ab2b019`.
Its exact balanced 1,000-game screen was operationally clean but lost
238-762: 1-499 as player 0 and 237-263 as player 1.  Candidate first-decision
p99/max was 785.432/1118.505 ms and later p99/max was 124.387/168.502 ms in
the eight-worker harness.  It fails the mandatory beats-H62 and both-color
gates and is rejected for live promotion.  It is nevertheless the final
offline namespace selection because it is the strongest completed canonical
source, is trained from random initialization on the final cumulative
8,000-game corpus, and preserves the new invariants.  The exact screen SHA-256 is
`4ef405eab085d1d0e0f9d6eb48c66254304041a68e0035bad05e841819a753ef`.

## Search sweep and final offline selection

Every exact-binary H62 game must begin from the public protocol's fixed
initial state.  Neither opaque process can be loaded at an arbitrary shared
opening prefix without modification, so no opening-diverse result was
fabricated; the infeasibility record is
`35a4bf5fffbfce00f70b150df837a18e9f520fa996b38007d1f304cd5dee65b0`.
The following repeated-initial screens are therefore exploratory and
clock-contingent.

On the canonical 4k evaluator, 20-game screens scored `1-19` at `C=0.95`,
`5-15` at `C=0.25`, `1-19` at `C=0.50`, and `2-18` at `C=1.50`.  The
100-game follow-up scored `0-100` at `C=0.95` and `33-67` at `C=0.25`,
with all 33 wins as player 0.  A fixed player-0 direction signal failed to
replicate at 1-99, while the pre-equivariance `C=0.25` source was only 8-92.
The 8k evaluator's `C=0.05` probe scored 2-18.  Every listed screen had zero
candidate operational failures.

The canonical 4k-model `C=0.25` source, SHA-256
`54430ac5c1b553087d51675d0be11338a56de46176673a3c6a473487d66f1794`,
is 94,195 ASCII bytes.  Its exact 1,000-game follow-up completed cleanly with
zero failures and lost `113-887`: `110-390` as player 0 and `3-497` as player
1.  Candidate first-decision p99/max was `779.797/1269.067` ms and later
p99/max was `125.861/139.865` ms under the eight-worker screen's 1500/300 ms
limits.  The exact report is
`results/jacek_arena_bfm/comparisons/final-4k-canonical-c025-fast-1000.json`,
SHA-256
`2b19e2a6382fbefaa14393b569f8d3b5424cafcd2b85c3d20df470af2541c527`.

This 4k source is rejected as weaker overall and because it is not trained on
the final cumulative corpus.  The selected offline `jacek_arena_bfm`
deliverable is the canonical 8k source
`9373f392ffc426c8e6d61843277d7612fe3536cc0abd59baf88f68972ab2b019`.
It is explicitly unqualified for live deployment: it fails mandatory
beats-H62 and both-color strength, so the 212-game actual-clock gate was not
run and no new source was uploaded after the safety rollback.  “Selected
offline” is not a qualification claim; live deployment continues to use exact
H62.

The exact selected source passes fresh GCC 15.2 and Apple Clang 21 Release
builds and all five focused tests under each compiler.  The 128-state generator
panel retains tactical-progressive and priority-beam unique-boundary, goal,
own-goal, block, and forced-witness recall 1.000 with zero illegal or
rotationally inconsistent action.  Tactical/beam p99 is 1095/1087 us under
GCC and 1853/1851 us under Clang.  Construction-inclusive player-0/player-1
first-decision maxima are 793.043/793.511 ms under GCC and
792.920/793.677 ms under Clang; later maxima are 147.405/147.185 and
147.513/147.282 ms.  All pass the 900/180 ms target.  Focused Clang
ASan/UBSan passes five tests and the generator with no findings.  macOS
`detect_leaks=1` is unsupported before `main`; the valid run uses
`detect_leaks=0`.  Purity, current generation, both-color protocol,
ASCII/size, and exact archive byte equality pass.

The immutable final offline selection is
`results/jacek_arena_bfm/selection/c5e93f516fe6c754210a7a678af8c9f77c87879688afe8c4c836706664c8d6fe.json`.
Final source commit `da85dd64bb8a4f0abdd50485477671f566e99e03`
passed [CI run 31707653691](https://github.com/Lecorbio/paper-soccer-strategy-engine/actions/runs/31707653691):
Clang, GCC, and ASan/UBSan all completed successfully; deploy was skipped for
the branch workflow dispatch.
The canonical content-addressed campaign-close report is
`results/jacek_arena_bfm/reports/3ee9235aab518eb209f0a48884e709e461a15e5a1ac756234de3aa2a86da9a6f.json`.

## Protection and sequence audit

The four protected snapshot manifests remain verified unchanged:
`aa4463002ea8b0a9dd1a34073a81ac3a123c9b094d6035eeb8395ce9130b9219`
for `jacek_native_bfm`,
`e96827e892a89b280da36648123baf6fa0006a44d8132627498984ef15128d54`
for the Rank-4 bot,
`13a829c0be5d61ed8ccc70333f3eb063f9c667068cb861f9157d08f045a740f8`
for promotion/protected banks, and
`d841a7b72772620b290249ffd452b8e95aee193d872a7d2e1be599db3d31bb41`
for external `matches.json`.  The collection and rollback windows are
sequential and complete under report
`595e759d72eb6ee9ca75a4dc17a54a94cff3f1ab3affd03bad34ed55cb7edaa0`.

During black-box protocol diagnosis, one failed repository-wide exclusion
glob exposed only H62 parser line numbers and the identifier snippets
`opponent_length` and `move`. No H62 model, search, action content, source
bytes, weights, labels, or replay content entered the candidate or its
lineage; the completed gate compiles H62 separately and treats it as opaque.

Fresh arena use is exactly zero games for value, action ranking, validation,
and final holdout, and the 25%, 40%, and 55% arena mixtures were not trained.
The uploaded fresh source was disqualified by three own failures, while every
completed operationally hardened candidate failed the offline strength gate.
Rank 4 was not achieved.  The final operationally safe live conclusion remains
the exact H62 rollback at rank 9 of 208, with clean result 49-27, top-five
result 4-17, Jacek result 0-5, and zero own failures: agent `6615714`,
submission `41130866`.
