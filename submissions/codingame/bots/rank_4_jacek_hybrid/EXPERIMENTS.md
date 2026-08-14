# Rank-4/Jacek hybrid experiment record

## Behavior-neutral scaffold

The initial candidate is a byte-for-byte copy of Rank-4's engine and generated
model/replay headers. Its JSON model and replay inputs are also byte-identical.
The only generated-source transformation is leading-indentation removal.

Exact source identities, size reduction, and fixed-work parity results are
frozen below. Semantic changes must be isolated as later ablations; this
baseline must remain reproducible.

- Canonical Rank-4 source: 98,624 characters, SHA-256
  `5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`.
- Compacted scaffold source: 86,930 ASCII characters, SHA-256
  `b2ba9adc171bdbd2c52b9fcd76e3d1a8a024a695b396bfe4d5ace4ff795e95d0`.
- Exact transformation check: the compact source equals every canonical
  Rank-4 source line after `trimStart`, including line order and contents.
- Source reduction: 11,694 characters; remaining allowance under 99,999:
  13,069 characters.
- Engine source SHA-256 shared with Rank-4:
  `9276c258cd613b6b78948aeb8aa2649851d226947d419f84a351968a9035c0ad`.
- Replay book JSON/header SHA-256:
  `835f273e37b43e79c21269169f2ea68570c610ff6a974a235a82c8c97e8551dc` /
  `1a23a0425628e4bc903f34f1923f8e5393ced9326a31e210b202d22715a52a2c`.
- Replay-value JSON/header SHA-256:
  `1687475728c04f6d255a6fdcbeefce8a5255f7db01f63c3e2a8a74a5c4fcf7b5` /
  `36ac226b60f0cc5a5fac9f2dc35017320ebd00c304847f83a8d07ef1eef7f2bd`.
- Teacher-residual JSON/header SHA-256:
  `db120a0e12c64671fdc2361a3e48a2e61ac39700d65e52140835e7b5ccee0cd8` /
  `e2034019126d54c2e0fd33922d7304c09979a64007a8d55e20823eda46834e42`.
- Fixed-work parity: 32/32 decisions matched canonical Rank-4, including
  encoded action, resulting boundary state, and every search-stat field, over
  eight opening/developed/tactical positions at 1, 251, 2,000, and 10,000
  nodes.
- Release tests: submission semantics, generator freshness, protocol smoke,
  dual-engine parity, and both-color timing all passed under Apple Clang 21.
  GCC 15 also compiled the flattened source and passed submission, freshness,
  protocol, and parity gates. Focused Apple Clang ASan/UBSan submission and
  parity tests passed.
- Observed local timing: player 0 800.073/165.107 ms and player 1
  800.075/165.094 ms for first/later decisions. These are local observations,
  not a same-runtime CodinGame qualification claim.

## Ablation 1: mover-relative horizontal tie

Rank-4's heuristic move ordering already makes vertical progress relative to
the mover, but its final equal-score tie always chooses smaller physical `x`.
After a 180-degree board rotation and player swap, smaller `x` maps to larger
`x`; this made ordering, deadline fallbacks, and partial fixed-work searches
color-dependent.

The isolated candidate changes only this expression in `bot.cpp`:

```cpp
return mover == Player::One ? left.move.to.x < right.move.to.x
                            : left.move.to.x > right.move.to.x;
```

The read-only audit harness
`tests/codingame/rank4_rotation_equivariance_audit.cpp` generated 1,000 legal
states with a fixed xorshift seed and paired each with its 180-degree
player/color rotation. Rank-4 baseline and the temporary one-line-patched
source were compiled with Apple Clang 21, C++20, and `-O2`.

- Scalar evaluation: 0/1,000 sign-negation mismatches both before and after
  the patch. The 24 mover-relative floating features agreed within
  `8.642674e-7`; 427/1,000 pairs were not bit-identical because symmetric
  edge accumulation visits addends in a different physical order.
- Ordered edge lists: Rank-4 baseline 185/1,000 mismatches; patched 0/1,000.
- Complete fallback actions: Rank-4 baseline 85/1,000 mismatches; patched
  0/1,000.
- Fixed-node root actions over 200 paired states at node budgets
  `1/16/64/256/1024`: baseline mismatches `17/9/9/9/10`; patched mismatches
  `0/0/0/0/0`.
- Fixed-node root scores at the same budgets: baseline mismatches
  `0/8/4/5/8`; patched mismatches `0/0/0/0/0`.
- Full tracked search-stat tuples at the same budgets: patched mismatches
  `0/0/0/0/0`.
- Completed fixed-depth root actions and scores remained rotation-equivalent:
  0/60 mismatches at depth 1 and 0/23 at depth 2. A few traversal-count
  differences remain after the tie patch (2/60 and 1/23), consistent with
  physical-index transposition-table bucket hashing. That larger cache-key
  change is deliberately excluded from this ablation.

Permanent regressions freeze the `6/1` one-node fallback pair (`47` and
rotated `03`) plus 16- and 64-node pairs from transcript `4/3/6/4/3/0`.
Actions rotate exactly, scores negate, and tracked work statistics match.

The original 32-decision compact-scaffold panel now reports 19 exact
action/stat matches, 13 traversal-only deltas, and zero action deltas. A
separate frozen rotated tie witness proves the intentional behavior change:
Rank-4 selects `05`, while the hybrid selects canonical counterpart `03`, at
the same one-node work and value semantics. Global Rank-4 action parity is no
longer a valid requirement for this candidate.

- Generated source at this historical ablation step: 86,988 ASCII characters,
  SHA-256
  `a092d879a53092b0c5a9c24bf43194226faf38be2cb4b4babc0b4c2c7666f394`.
- Engine source SHA-256 at that step:
  `1fafc0ed02bd2475723708c55701971ea0a0f4a324a36ace1c74ece8dd9c305c`.
- Remaining source allowance: 13,011 characters.
- Generator freshness and ASCII checks passed. The 14/14 submission suite,
  bounded-delta parity gate, and both-color protocol smoke passed with Apple
  Clang 21 and GCC 15.2. Focused Apple Clang ASan/UBSan submission and parity
  binaries also passed.
- Observed local construction-inclusive timing: player 0 800.096/165.094 ms
  and player 1 800.089/165.095 ms for first/later decisions. These are local
  deadline observations, not a same-runtime CodinGame qualification claim.

## Ablation 2: toggleable exact rebound exchange proof

The hybrid selectively adapts the exact, safe rebound-component analysis from
`rank_4_exchange`. The later `SearchConfig::exact_proof_mask` supersedes the
original Boolean toggle; mask zero preserves the proof-off control and the
historical implementation-step `choose_complete_turn` path enabled every safe
scope. The later DEVELOPMENT selection below supersedes that temporary mask
15 choice with operational mask 7.

The enabled proof performs only these classifications:

- At the root, return an attacking-goal route through the current rebound
  component when one is reachable.
- At a depth-zero boundary, replace the heuristic only when the component is
  exactly a mover Win or Loss.
- At reply and counterturn boundaries (`turn_ply` 1 and 2), run the same proof
  only after an adequate transposition-table entry has had its cutoff chance.

Any state with at least one safe fresh-vertex handoff is `Unknown` and falls
through to the unchanged alpha-beta/evaluator path. An attacking route into
the mover's own goal is never a Win proof. Rebound BFS adjacency is traversed
in mover-relative canonical direction order so the concrete returned goal
route rotates exactly with player/color swap.

Development evidence (not a validation or promotion-bank result):

- Disabled parity: 32/32 decisions matched the default-off mode in encoded
  action and every search-stat field across eight positions and node budgets
  1, 251, 2,000, and 10,000; all proof counters remained zero.
- Exact fixtures: both-color root attacking goals, fresh-mouth Unknown,
  own-goal rejection, depth-zero proof, reply Win/Loss, and counterturn
  Win/Loss all passed with exact mate scores and legal full-turn replay.
- Symmetry: crafted root proof routes rotate exactly; reply and counterturn
  outcome/score fixtures pass for both movers.
- A clean detached release build passed the 17/17 hybrid submission suite and
  the Rank-4 bounded-delta parity gate.
- Observed local construction-inclusive deadline timing with proof enabled:
  player 0 800.077/165.110 ms and player 1 800.094/165.104 ms for first/later
  decisions; each remained below the 1000/200 ms hard gate. These are local
  observations, not a same-runtime CodinGame qualification claim.
- Source after this isolated ablation: 92,830 ASCII characters, SHA-256
  `2ca2280f533bffc1750732ae55926b8353b16299a40c456e0b58a4cb8d426468`;
  remaining allowance under 99,999: 7,169 characters.

No protected validation/final bank was run for this isolated implementation
step. Whole-game fixed-work and clock evidence belongs to the separate hybrid
comparison gate and must decide whether this proof is retained.

## Ablation 3: independently selectable exact-proof scopes

The exact proof is split into four orthogonal `SearchConfig::exact_proof_mask`
bits: root attacking-goal return (`1`), depth-zero boundary value (`2`), reply
boundary at ply one (`4`), and counterturn boundary at ply two (`8`). Mask
zero remains the tie-only control. The initial candidate at this historical
implementation step used mask `15`; the subsequent DEVELOPMENT matrix selected
operational mask `7`. Unknown mask bits are rejected at construction.

Root and leaf scans now have separate probe/Win/Loss counters. Ply-one and
ply-two counters remain separate, and tests assert that their sum equals the
existing aggregate rebound counters. The comparison gate accepts independent
candidate and hybrid-control masks and prints all four scopes for both sides.
The old Boolean candidate option remains only as a `0`/`15` compatibility
alias. All rejected root-scout switches and counters were removed from the
gate.

Local structural evidence:

- Mask zero retained the frozen 32-decision tie-only bounded-delta result:
  19 exact action/stat matches, 13 intentional traversal-only deltas, and no
  action delta against the compact reference panel.
- All 16 masks were exercised on six tactical/search witnesses and their
  player/color rotations. All 96 paired cases returned legal complete turns,
  actions rotated exactly, scores negated, disabled scopes did no work, and
  per-scope counters summed exactly to the aggregate counters.
- Isolated root, leaf, ply-one Win/Loss, and ply-two Win/Loss fixtures pass for
  both movers. The submission suite passes 18/18 and the lightweight gate
  schema/SHA tests pass 2/2.
- Generated source at this historical ablation step: 94,004 ASCII characters,
  SHA-256
  `6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f`;
  remaining allowance under 99,999: 5,995 characters.

No whole-game match was run during this implementation step, and no protected
validation or final bank was opened. At that historical point, the planned
actual-clock development matrix used one complete preregistered development
bank and four nested hybrid-control comparisons: masks `1 vs 0`, `3 vs 1`,
`7 vs 3`, and `15 vs 7`. The later DEVELOPMENT scope-selection section records
the completed matrix and supersedes that proposal.

## DEVELOPMENT scope selection: mask 7

The preregistered nested depth-20 clock matrix measured each scope's marginal
in search order. The results were `1 vs 0: 38-38`, `3 vs 1: 40-36`, `7 vs 3:
39-37`, and `15 vs 7: 38-38`. On that DEVELOPMENT evidence, mask `7` was
selected: root attacking-goal return, exact depth-zero boundary value, and
ply-one proof are enabled; ply-two proof is disabled.

The selected mask then advanced through both complete DEVELOPMENT comparisons,
covering all four preregistered development depths and 306 paired-color games:

- Mask 7 beat the same-binary mask-zero control 166-140: 84-69 when the
  candidate occupied physical color 0 and 82-71 as physical color 1.
- Mask 7 beat canonical Rank-4 169-137: 83-70 as physical color 0 and 86-67
  as physical color 1.
- Both gates completed with zero failed games.

The frozen selection receipt is
`results/rank_4_jacek_hybrid/gates/full_development_clock/selection/1b6736186006b6820021dc0315faab50dcba97db719ea5bbfe6768a7e2a243d3.json`
(SHA-256
`1b6736186006b6820021dc0315faab50dcba97db719ea5bbfe6768a7e2a243d3`).
These are DEVELOPMENT results, not heldout validation, final qualification, or
live-arena evidence.

## Ablation 4: null-action proof fast path (rejected)

The null-action fast path specialized exact rebound-component analysis so
non-root proof probes could avoid route reconstruction and ordering work. Its
authoritative actual-clock gate compared mask-7 fast-path code with the
archived pre-fast-path mask-7 algorithm over the 76-game depth-20 DEVELOPMENT
bank, with equal 3,000,000-node caps and 800/165 ms clocks.

The fast-path candidate lost 36-40. Its physical-color splits were 19-19 as
color 0 and 17-21 as color 1. All 76 games finished with zero failures,
illegal actions, exceptions, operational violations, or hard timeouts. Timing
passed its frozen thresholds: candidate first/later p99 was 800.164/165.155 ms
and first/later maximum was 800.164/165.190 ms. Search progress also passed:
candidate average completed depth was 3.894 versus 3.865, and average nodes
were 213,863.751 versus 200,862.635 for the control.

The frozen noninferiority gate nevertheless required at least 38 total wins
and at least 19 wins in each physical color. The candidate missed both the
total-win threshold and the color-1 threshold, so the optimization is rejected
and the same attempt identity may not be rerun. The authoritative report is
`results/rank_4_jacek_hybrid/gates/null_fastpath_clock/fa4596213c69a782976a41fd362eb61cc0f484f2e51598a5838eeef8fbabfa59.json`
(SHA-256
`fa4596213c69a782976a41fd362eb61cc0f484f2e51598a5838eeef8fbabfa59`).
The mandatory-revert decision receipt is
`results/rank_4_jacek_hybrid/gates/null_fastpath_clock/selection/27de96bac5b2ea6c43613ee8b9f5c64f16a33505bfcfde1872d33e3b3c2268bb.json`
(SHA-256
`27de96bac5b2ea6c43613ee8b9f5c64f16a33505bfcfde1872d33e3b3c2268bb`).

The targeted rollback removed the null-action fast path and restored the
archived pre-fast-path proof algorithm. It retained the selected operational
mask 7 and the test-only rebound-component audit hook. Regeneration assigned
new production identities rather than reusing the archived control hashes:

- `bot.cpp`: 63,107 bytes, SHA-256
  `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`.
- `submission.cpp`: 94,312 ASCII characters, SHA-256
  `2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
- `submission_test.cpp`: 39,137 bytes, SHA-256
  `ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697`.

The scaffold, tie-only, mask-15, archived pre-fast-path, and rejected
null-fast-path identities above remain historical evidence; none identified
the then-current rollback bytes. At this rollback boundary, no heldout or
final bank had yet been opened for the rollback candidate and no live upload
had occurred.

## Ablation 5: sole-legal-edge ordering bypass (rejected)

When `ordered_moves` sees exactly one legal primitive edge, no ordering
decision exists. The candidate returns that edge immediately with a neutral
score instead of computing progress, center, goal, block, and mobility terms
that every caller then ignores. Search values, legal actions, proof logic,
tables, and pruning are unchanged.

The isolated rollback-based prototype used 1,000 deterministic procedural
states plus four tactical states, both exact-proof masks 0 and 7, and node
budgets 1, 16, 64, 256, and 1,024. All 10,040 paired decisions matched in
encoded action, root score, and every emitted `SearchStats` field. Eighteen
roots had one legal edge; their control heuristic scores were nonzero while
the candidate scores were zero, proving the branch executed.

Alternating construction-plus-50,000-node microbenchmarks used 30 warm-up and
300 measured pairs. With production table sizes, the forced-heavy panel
improved 1.424% at median and 0.596% at p99. The mixed panel changed +0.082%
at median and improved 0.983% at p99. Result signatures were exact. The same
panels with small tables also stayed inside the frozen 0.5% regression limit.

A both-color singleton witness asserted exactly one legal move, neutral bypass
score, and legal output. Focused submission, parity, generator, protocol, and
cheap replay tests passed 5/5. The evaluated artifacts were:

- `bot.cpp`: 63,350 bytes, SHA-256
  `16a4358680cfc69e830136d4e0c2e6e45371139a02ca09ecb9bf1f9e239d3b2b`.
- `submission.cpp`: 94,527 ASCII characters, SHA-256
  `d18c49c7cc149d8b48a69a03ebb13dd4fc49ae8927c1324515ba1ae197822b15`.
- `submission_test.cpp`: 40,103 bytes, SHA-256
  `0823299900cf0d31730c73ccb91a3a55c7a7ef351949e583a40a7c66a43f5e88`.

The sole preregistered DEVELOPMENT depth-20 clock comparison was valid and
finished 38-38. The candidate went 20-18 as physical color 0 and 18-20 as
physical color 1, with zero unfinished, failed, illegal, operational,
exception, or hard-timeout results. It passed timing limits and the progress
gate via 182,468.772 versus 181,258.376 average nodes, but failed the frozen
minimum of 19 wins in each color by one color-1 win. The optimization is
therefore rejected and its exact identity may not be rerun.

The authoritative report is
`results/rank_4_jacek_hybrid/gates/sole_legal_edge_clock/1d472a43a2f7ce0bc314e2c3619a912fb759b2a2eba489ca451e59a16094f315.json`.
The rejection receipt is
`results/rank_4_jacek_hybrid/gates/sole_legal_edge_clock/selection/e36314e33d9ca66f9c901c0e99dc613e10b43e1941668cc59ad5e6b3d8a0b5af.json`.
Production was restored to the 94,312-character rollback source
`2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
At this ablation's rejection boundary, heldout validation and final banks had
not yet been opened and no live upload had occurred.

## Ablation 6: private PositionKey component cache (rejected)

The candidate replaces repeated 128-bit component mixing during compact
make/unmake with immutable per-topology values. Categories and XOR transition
points are unchanged: used edge, visited vertex, ball, player, status, and
turn-boundary ball. A hybrid-private `hybrid_detail` header prevents any
effect on Rank 4 or other bots; the shared MCTS header remains byte-identical.

The exact candidate is 95,750 ASCII characters, SHA-256
`47f44e8e62d3aaa2a48f6eea6fca4d17cfbbfd3ff9a5ac01ca84b1e0bf4cca03`.
It passed 399,450 direct key observations, a 20-configuration dimensions/rules
matrix with 2,560 make/unmake operations, and all 10,040 fixed-work decisions
with identical actions, root values, and statistics. The full 511-prefix
replay audit also passed with zero displaced or capped decisions.

In its single locked production-table microbenchmark, forced-heavy median/p99
ratios were 0.976901/0.981884 and mixed ratios were 0.989095/0.987205. Both
panels cleared the frozen requirement of at least 1% median improvement and no
more than 0.5% p99 regression. The immutable preintegration receipt is
`results/rank_4_jacek_hybrid/position_key_components_prototype/PASS.md`.

The sole preregistered DEVELOPMENT depth-20 actual-clock comparison was valid
and finished 37-39. The candidate went 21-17 as physical color 0 and 16-22 as
physical color 1. All 76 games completed with zero unfinished, failed,
illegal, operational, exception, or hard-timeout results. Exact mask-7 proof
accounting reconciled, including a zero ply-two scope.

Timing and progress both passed. Candidate first/later p99 was
800.199/165.170 ms and first/later maximum was 800.199/165.198 ms. Average
completed depth was 3.845 versus 3.839, and average nodes were 189,229.157
versus 185,645.754 for the archived rollback control. The frozen selection
rule nevertheless required at least 38 total wins and at least 19 wins in
each physical color. The candidate missed the total floor by one and the
color-1 floor by three, so it is rejected and the exact attempt identity may
not be rerun.

The authoritative report is
`results/rank_4_jacek_hybrid/gates/position_key_cache_clock/423af75c4cab30cbb600bd0633bbcba1a58e163de9cf3f2d984874b61c8b1e8d.json`.
The mandatory-rollback decision receipt is
`results/rank_4_jacek_hybrid/gates/position_key_cache_clock/selection/37a0872d67356ba9aaf2c6419b6d4e7cecba0a0ff187d548b0f48d6f825f0a9a.json`.
Production was restored to mask-7 `bot.cpp` SHA-256
`34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`
and 94,312-character source SHA-256
`2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
At this ablation's rejection boundary, validation and final banks had not yet
been opened and no live upload had occurred.

## Ablation 7: safe-handoff frontier width (rejected at Stage 1)

The last semantic hypothesis registered before held-out qualification reused
the exact leaf-boundary proof scan to count unique reachable fresh endpoints
at which the opponent retains a legal reply after the incoming edge. Exact
`Win` and `Loss` outcomes keep their mate scores. Only `Unknown` leaves receive
`player_sign(to_move) * count * 10 * (100 - replay_blend) / 100`, after which
the adjusted value is stored in the existing evaluation cache. Weight 10 is
the only registered value.

The first draft was refrozen before timing or games because unconditional
endpoint marking also shortened duplicate scans in root and ply-one proof
calls. The final candidate marks and counts only when the leaf caller requests
the feature; null-output root/ply scans retain the rollback boolean path. A
mask-1/mask-5 isolation panel covered 10,040 decisions with zero action,
score, or `SearchStats` deltas.

The exact Stage-0 candidate is:

- `bot.cpp`: 64,521 bytes, SHA-256
  `408adc5288674550cc08274aec74380074117e32ad8f6915c7e39badc8dfba98`;
- `submission.cpp`: 95,272 ASCII characters, SHA-256
  `08d0c0859ef8a197f8bfdd89afb048bec41c3a888228433b85991cd937882550`;
- source headroom: 4,727 characters below the 99,999-character limit.

The maintained submission, parity, current-source, both-color protocol,
cheap replay, and full 511-prefix replay tests pass. The dedicated frontier
suite passes 8/8 and covers an independent graph oracle, rotation, endpoint
deduplication, literal weight registration, exact mate bypass, active teacher
residual interaction, cache reuse, TT reuse, and fixed-work determinism. The
submission, parity, and frontier suites also pass under fresh Apple-Clang
ASan/UBSan with leak detection disabled on macOS.

The immutable packet and sequential stop-on-failure thresholds are under
`results/rank_4_jacek_hybrid/frontier_semantic_prototype/`. Under that
historical plan, Stage 3 would have executed the depth-20 clock bank once and
Stage 4 would have reused that immutable receipt while running only depths 4,
8, and 12. The Stage-1 failure below meant neither stage was authorized. No
timing benchmark or whole game had been run when the packet was frozen.

The sole locked Stage-1 run used 512 public-rule/tactical leaf fixtures, seven
warm-up pairs, and 31 measured alternating AB/BA pairs. Its control/candidate
median times were 2,497,209/2,534,500 ns; paired median ratio 1.003703 passed
the 1.010 ceiling. The paired p99 ratio was 1.106654, above the frozen 1.020
ceiling, despite a lower standalone candidate p99. The conjunctive gate
therefore failed and the sequential plan forbids any retry, optimization, or
whole-game screen. The raw report is
`results/rank_4_jacek_hybrid/gates/frontier_semantic_timing/bb883a55fcc3ab0e1992d9135f683fcb8aa1aacde913e4fd22f06bce83c5d4bf.json`.
The canonical rejection receipt is
`results/rank_4_jacek_hybrid/gates/frontier_semantic_timing/selection/f85a74985e56e3ad67d3602a44d712e5f511a4a491d313a160583bd764e9be89.json`.
The candidate is rejected, not heldout-qualified, not uploaded, and not final;
the working hybrid source was restored at that boundary to the exact
94,312-character mask-7 rollback.

## Authoritative held-out qualification of the restored rollback

The restored mask-7 rollback subsequently completed its one-shot held-out
qualification. VALIDATION passed 61-45, with 34 wins as physical color 0 and
27 as physical color 1; the frozen floors were 54 total and 26 in each color.
FINAL finished 104-108, with 48 wins as color 0 and 56 as color 1. The
candidate missed the 108-win total floor and the 53-win color-0 floor. Safety,
timing, proof and sweep accounting, input and compiler stability, and source,
admin, binding, and portability provenance all remained clean.

The canonical reports are
`results/rank_4_jacek_hybrid/gates/heldout_qualification/binding_recovery_v1/reports/validation/e0b5ed9bd6c77ce90317cc363ab19679e01216172de9ebb01b5eb05d2c6bc5cc.json`
and
`results/rank_4_jacek_hybrid/gates/heldout_qualification/binding_recovery_v1/reports/final/19e0d5e692d5afce2f9a83ef2247bcf53816bf7d66c6a60aa0bcef9205b4c271.json`.
The terminal decision is
`results/rank_4_jacek_hybrid/gates/heldout_qualification/binding_recovery_v1/decisions/9c12b44cc2ffa475e55e1e166c637f725e8107736677c432b03ea31ef376997f.json`.
It records `final_qualification=false` and arena authorization false. This
exact qualification cannot be retried, and no live upload occurred.

## Post-heldout DEVELOPMENT mask-3 removal (rejected at Stage 1)

After the held-out decision closed, a separately preregistered
DEVELOPMENT-only removal campaign tested the pre-heldout mask-`3` fallback.
Its first stage directly compared mask 3 with the same-binary mask-`7` control
over the existing depth-4, -8, -12, and -20 DEVELOPMENT banks. This ordering
was frozen before the new DEVELOPMENT result and did not use held-out score,
color, timing, or opening details to tune the candidate.

Mask 3 finished `152-154`. Its physical-color records were `85-68` as color 0
and `67-86` as color 1. The frozen selection rule required at least 160 wins
overall and at least 77 wins in each color, so the candidate missed the total
floor by eight and the color-1 floor by ten. Those were the only threshold
errors. All 306 games completed with zero unfinished, failed, illegal,
operational, exception, or hard-timeout results.

Candidate first/later p99 and maxima were `800.166/800.185 ms` and
`165.167/165.237 ms`; the mask-7 reference values were
`800.203/800.260 ms` and `165.170/165.335 ms`. All remained below the strict
`990/198 ms` maxima. Proof-scope identities and sums, exact engine-work and
bank/color aggregation, process safety, chronology, and before/after
source/binary/admin/environment evidence all replayed cleanly.

The canonical report is
`results/rank_4_jacek_hybrid/gates/mask3_removal_clock/reports/64623834951fd4a00484ef0aa1a890127fe9b6a19539b65b3fb874fcf4794725.json`.
The terminal decision is
`results/rank_4_jacek_hybrid/gates/mask3_removal_clock/decisions/894442b0d0a81418d591469bb1b8c1d34cc6c0a8ed13371812a66a21d1e5bc48.json`.
It records a terminal DEVELOPMENT rejection: Stages 2 and 3 were never
claimed or opened, and no retry, source activation, fresh held-out campaign,
or arena action is authorized. Canonical Rank 4 remains the incumbent; the
mask-7 rollback remains only historical/control evidence.
