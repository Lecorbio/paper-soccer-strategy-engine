# Rotation-equivariant table-placement ablation

Status: **design only; not implemented or selected**.  This is the single
next search hypothesis after an exact-proof mask is selected.  It must not be
combined with another search or evaluator change in the same ablation.

## Why this change

The mover-relative horizontal tie fixed edge ordering, fallback actions, and
fixed-node player/color rotation.  One known asymmetry remains: the fixed-depth
audit still found traversal-stat differences in 2/60 depth-one pairs and 1/23
depth-two pairs.  The experiment ledger attributes those differences to the
physical-index transposition/evaluation-table hashes.

That diagnosis matches the code.  `position_key_component(category, index)`
currently gives the two key words unrelated salts over the same physical
index.  `TranspositionTable::bucket_index` and
`EvaluationTable::combined_key` then combine the ordered words.  A state and
its 180-degree player/color rotation therefore encounter unrelated bucket
collisions, victims, hints, and cutoffs even after their move order agrees.

This is a better risk-adjusted next experiment than the nearby alternatives:

- the shared rebound-outcome cache already preserved fixed-work play but lost
  deadline throughput;
- iterative-deepening reuse and aspiration windows already lost paired gates;
- boundary PVS improved deterministic node gates but materially harmed live
  player-zero play; and
- tactical leaf extensions either aliased incompatible table modes or cost
  roughly 25--35% of throughput.

## Exact mapping and key contract

Do not modify shared `src/bots/mcts_internal.hpp`, because that would change
the canonical Rank-4 build.  Fork its current contents into a private hybrid
internal header, replace only this bot's `sources.txt` entry and include, and
leave the shared file and `rank_4` byte identities unchanged.  Moving the
rotation maps already built by `CompleteTurnSearch` into the private
`SearchTopology` adds no second rotation map; only the bounded precomputed
word arrays declared under Resource bounds are new topology storage.

For rules with width `W` and height `H`, define the involution:

```
R(Point{x,y}) = Point{W-x, H+2-y}
R(edge {a,b}) = edge {R(a),R(b)}
R(Player::One) = Player::Two
R(Player::Two) = Player::One
R(Status::InProgress) = Status::InProgress
R(Status::WonByOne) = Status::WonByTwo
R(Status::WonByTwo) = Status::WonByOne
```

Every topology vertex and edge gets its exact rotated index, and both maps
must be involutions.  Keep the existing two-word `PositionKey` calculation
and exact-key equality byte-for-byte.  In particular, let its existing first
word use

```
Z1(category,index) = mix_position_key(
    ((category << 56) ^ index) ^ 0x243f6a8885a308d3)
```

Add one non-identity accumulator to `SearchPosition`:

```
rotated_first = XOR Z1(category, R(component_index))
```

Categories retain their current meanings: used edge `1`, visited vertex `2`,
ball `3`, player to move `4`, status `5`, and the extra boundary-ball tag `6`.
The rotated accumulator uses the rotated edge/vertex/ball index, opponent
player, or rotated status as applicable.  Make/unmake XORs remain incremental.
Precompute the first-word values for the rotated edge and vertex categories in
the topology, so a search-node update adds only array loads and XORs rather
than another mixer call.

The lookup object is the unchanged exact identity key plus this placement-only
word.  Including boundary category `6`, it has the exact invariant

```
lookup(state).physical_first = lookup(R(state)).rotated_first
lookup(state).rotated_first = lookup(R(state)).physical_first
```

The existing independently salted second identity word is deliberately not
repurposed.  This retains the incumbent's 128-bit exact-entry collision
strength instead of reducing symmetry-related identity to one 64-bit
constraint.

Use one symmetric, dispersed table-placement word for both tables and for the
evaluation table's victim-way choice:

```
lo = min(lookup.physical_first, lookup.rotated_first)
hi = max(lookup.physical_first, lookup.rotated_first)
placement = mix_position_key(lo ^ rotl64(hi, 23))
```

The two-way bucket is `2 * (placement % bucket_count)` and the evaluation
victim way is `(placement >> 32) & 1`.  Do not use `first ^ second`: because
the state key itself is an XOR, that would discard each component's
orientation independently and create structured collisions.

Most importantly, **placement is canonical but identity is not**.  Entries
continue to store and compare the unchanged full `PositionKey`; the extra
rotated word is never stored as identity.  `best_move` stays in the physical
frame and is returned only after that exact ordered-key match.  Never replace
equality with `min/max`, never let a rotated state hit the original entry, and
never transform a cached move in this ablation.  The only intended semantic
delta is which exact entries collide and are evicted under finite table/work
limits.  Full-depth minimax values and all exact-proof claims are unchanged.

## Required exact tests

1. For every topology vertex and edge, prove `R(R(index)) == index`; prove the
   player and all three status mappings above.
2. On at least 1,000 new legal procedural states plus every tactical proof
   fixture, require the physical/rotated placement words to swap exactly after
   rotation, and require make/unmake to restore the unchanged exact identity
   key and rotated accumulator exactly.
3. On deliberately asymmetric fixtures, require the unchanged exact keys of
   `state` and `R(state)` to differ while their TT bucket, evaluation bucket,
   and evaluation victim way agree.
4. Store distinct scores and physical best moves for an asymmetric state and
   its rotation in the same two-way bucket.  Both must coexist and each lookup
   must return only its own score/move.  Storing one and querying the other
   must miss.  This test distinguishes canonical placement from forbidden
   semantic merging.
5. Repeat paired insertion/probe/replacement sequences with table sizes
   `2,4,8` and the production sizes.  The rotated sequence must have identical
   hit/store/cutoff/victim statistics and rotated legal actions.
6. Extend the existing rotation audit: fixed-node actions rotate, scores
   negate, and the complete tracked statistics match at every budget.  The
   prior 2/60 and 1/23 fixed-depth traversal mismatches must become zero, with
   the selected proof mask enabled as well as disabled.
7. At completed depth with ample work, TT-on and TT-off must retain the same
   root score and legal action under the existing deterministic tie rules.
   Every exact Win/Loss fixture, own-goal rejection, protocol, purity,
   GCC/Clang, and focused ASan/UBSan test remains mandatory.

Any cross-frame cache hit, non-involutive map, proof-score change, illegal
action, or new rotation mismatch rejects the hypothesis before a game gate.

## Resource bounds

- Expected generated-source delta: 700--1,400 ASCII characters, because the
  existing rotation-map implementation moves rather than duplicates and the
  exact identity table entries do not grow.
- Hard source-delta cap: 2,000 characters over the exact selected-proof
  control; total generated source must remain at most 99,999 characters.
- No additional TT/evaluation entries, larger table entries, per-node
  allocation, or second rotation map.  One 64-bit live-position accumulator
  and at most 8 KiB of topology-owned precomputed placement words are allowed.
- Alternating same-state microbenchmark median slowdown at most 0.5% and p99
  slowdown at most 1.0%.  Construction-inclusive fresh-process timing must
  remain below 900/180 ms, same-runtime maxima below 990/198 ms, and the hard
  1000/200 ms limits.

Exceeding any source, allocation, or timing bound is an immediate rollback;
do not try to compensate by changing clocks, table sizes, proof scope, or
node allocation.

## Development-only ablation

Freeze the selected proof mask, model/replay inputs, clocks, table sizes,
node cap, move ordering, and exact pre-change source SHA before implementation.
Compile a frozen physical-table control and the canonical-placement candidate
as separate gate engines.  A compile-time gate switch is acceptable; do not
carry a per-node runtime branch into a finalist.  Do not open validation or
final banks for this experiment.

Run sequentially on the already assigned development banks:

1. A full 306-game fixed-500-node paired-color safety run.  Require zero
   unfinished/illegal/operational failures and at least 150 candidate wins.
   Record TT/evaluation probes, hits, cutoffs, stores, victims, completed
   depth, proof counters, and both physical colors.
2. If clean, a 76-game `development_d20` 800/165 ms screen with 3,000,000
   node ceilings.  Require at least 40--36 overall, at least 19 candidate wins
   in each physical color, zero failures, and all timing bounds above.
3. If clean, the complete 306-game development actual-clock comparison.
   Require at least 160--146 overall, at least 77 candidate wins in each
   physical color, and zero failures.  Then rerun the candidate against exact
   Rank-4 on the same development set; it may not lose more than two wins from
   the frozen selected-proof baseline and must remain positive in both colors.

The thresholds are frozen before observing this ablation.  Failure at any
stage restores the exact selected-proof control bytes and records the reject
reason.  Passing only authorizes the existing candidate-freeze/validation
process; it does not itself authorize an arena upload or make the development
bank a promotion result.
