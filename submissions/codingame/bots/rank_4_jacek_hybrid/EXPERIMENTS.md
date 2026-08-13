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

- Current generated source: 86,988 ASCII characters, SHA-256
  `a092d879a53092b0c5a9c24bf43194226faf38be2cb4b4babc0b4c2c7666f394`.
- Current engine source SHA-256:
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
operational `choose_complete_turn` path explicitly enables every safe scope.

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
zero remains the tie-only control; the operational candidate uses mask `15`.
Unknown mask bits are rejected at construction.

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
- Generated source: 94,004 ASCII characters, SHA-256
  `6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f`;
  remaining allowance under 99,999: 5,995 characters.

No whole-game match was run during this implementation step, and no protected
validation or final bank was opened. The smallest proposed actual-clock
development matrix uses one complete preregistered development bank and four
nested hybrid-control comparisons: masks `1 vs 0`, `3 vs 1`, `7 vs 3`, and
`15 vs 7`. This measures each scope's conditional marginal in search order.
A cheap fixed-node screen should exercise all 16 masks first; only the selected
mask should advance against mask zero and Rank-4 on the full development bank.
