# Rank-4 exact-exchange experiment record

## Predeclared question and boundary

The question was whether the previously validated exact rebound proof could
improve the active rank-4 residual bot without changing its neural model or
opening/replay knowledge. The canonical `rank_4` directory stayed
byte-identical. All comparisons used the real rank-4 source as reference,
paired colors, deterministic openings, the same residual weight and replay
corrections, and no sealed prospective/final evidence.

Promotion required a clear equal-clock improvement with no operational
failure. Fixed-work gains alone were not sufficient. No CodinGame submission
was authorized by the evidence below.

## Retained implementation

The retained experiment proves immediate wins and forced blocked losses in a
rebound-connected component:

- once before iterative deepening, returning an exact legal winning action;
- at depth-zero leaves before heuristic evaluation; and
- at turn plies 1 and 2, after an adequate transposition entry gets the first
  chance to cut off.

The proof is player-rotation symmetric. Generation-mark visitation removes a
full-array clear per probe. An existing cached heuristic leaf safely skips a
repeat proof because such an entry is created only after the proof returned
`Unknown`. The final generated artifact is 91,259 characters, SHA-256
`e72bb4bdc3377d0a4602fd807a31083683677c4516f318d75352b60d63355f20`.

## Retained evidence

Direct candidate-versus-rank-4 results:

| Work profile | Games | Candidate-reference | Color split | Unfinished |
|---|---:|---:|---:|---:|
| 5,000 nodes | 106 | 58-48 | 29 / 29 | 0 |
| 5,000 nodes, widened | 306 | 158-148 | 85 / 73 | 0 |
| 30,000 nodes | 106 | 58-48 | 32 / 26 | 0 |
| Equal clock, 800/165 ms | 18 | 8-10 | 4 / 4 | 0 |

At 5,000 nodes the final behavior-neutral implementation reproduced the same
58-48 actions and search counts as the initial proof integration. It searched
9,671,264 candidate nodes versus 9,838,979 reference nodes and made 4,574,199
rebound probes. The leaf-cache shortcut avoids repeat proof work without
altering fixed-work decisions.

The fixed-work gain is repeatable and useful progress, but the equal-clock
result is not a promotion result. Timed screens varied from 9-9 in an earlier
run to the final isolated 8-10; neither is a clear improvement. The candidate
therefore remains an experiment and rank 4 remains live.

## Rejected ablations

All ablations below used the same deterministic paired bank.

| Variant | 5,000 nodes | 30,000 nodes | Equal clock | Decision |
|---|---:|---:|---:|---|
| Full exchange, residual 75% | 59-47 | 52-54 | not run | Reject: reverses deep |
| Full exchange, residual 50% | 55-51 | not run | not run | Reject: weaker |
| Reply only (ply 1) | 59-47 | 55-51 | 9-9 | Reject: weaker deep, no clock gain |
| Add ply 3 | 58-48 | 54-52 | not run | Reject: extra work, weaker deep |
| Root/leaf only | 60-46 | 54-52 | 9-9 | Reject: horizon-sensitive |

A shared rebound-outcome cache was also rejected. It preserved the complete
58-48 fixed-work result and found 323,800 duplicate probes in 106 games, but
its hash/table overhead slightly reduced deadline throughput. The final code
uses only the cheaper leaf-cache implication and generation marks.

## Verification and disposition

The merged submission tests cover exact/symmetric rebound wins and losses,
depth-zero proofs, first-reply and second-exchange proofs, residual rotation,
determinism, replay legality, interruption safety, and node/time hard caps.
Generator freshness, protocol smoke, comparison smoke, and timing checks are
part of the focused CTest set.

Disposition: **do not submit**. The experiment establishes a reproducible
fixed-work improvement and identifies proof overhead/horizon sensitivity as
the next bottleneck, but it does not clear equal-clock promotion evidence.
