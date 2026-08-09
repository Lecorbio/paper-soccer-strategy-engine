# All-depth exact rebound proof ledger

## T5: exact proof at every turn boundary

The candidate starts from `rebound_proof` (T3). Its only search change is an
eight-line exact Win/Loss check after a non-cutting transposition-table probe
and before `search_actions` at every positive-depth, non-root turn boundary.
Depth-zero and root proof behavior, evaluator, replay data, move ordering,
clock behavior, and transposition storage are unchanged.

This is a sound subtree shortcut under `BlockedRule::MoverLoses`: Win means the
side to move can score during the current mandatory rebound action; Loss means
every reachable continuation stays in a closed rebound component with no safe
fresh handoff. `Unknown` never prunes. Proof returns are not stored, avoiding a
mate-distance/transposition-key mismatch.

## Exposed-bank screen

Against immutable `rank_5` at 5,000 nodes, the candidate won 54-42 on the
48-opening development bank and 57-39 on the exposed 48-opening T4 validation
bank. An independent retired rank-one bank screen was also positive at 74-62.
The role-relative control uplift was positive for both colors in all three
screens.

Budget transfer is unresolved. At 30,000 nodes the candidate won 53-43 on
both development and exposed T4 validation, while T3 won 55-41 on each under
the same local runner. The candidate completed slightly deeper searches, so
this is compatible with sound pruning but is still a promotion warning: node
caps can change horizon and move-order trajectories even when every completed
fixed-depth minimax value is preserved.

The TT-first placement reproduced proof-first actions, scores, and completed
depths at 5,000 and 30,000 nodes on both exposed banks while avoiding 3,496 and
3,443 proof scans at 5,000 nodes, and 26,897 and 26,192 at 30,000 nodes.

## Decision rule

Do not use a fresh sealed test or submit this bot until it passes the normal
development and prospective-validation gates at both configured node budgets,
including control-relative role, stratum, throughput, and timing checks. The
30,000-node regression versus T3 must be treated as evidence, not hidden by the
strong 5,000-node result.

## T10 prospective result

The candidate passed the exposed development gate at both budgets (54-42 at
5,000 nodes and 53-43 at 30,000 nodes), but failed prospective T10 validation.
The fixed 30,000-node profile was directionally strong at 79-65, cluster mean
0.561 and lower bound 0.480; it missed retention (0.750 versus 0.800) and the
smallest stratum (0.429 versus 0.480). The construction-inclusive 130 ms profile
tied 72-72 and failed cluster mean/lower bound, physical-color uplift,
retention, and elite transfer. The sealed T10 final remained unopened.

This isolates the next problem: exact component proofs improve fixed-budget
search but perturb iterative-deepening trajectories under the deployment clock.
The next experiment should target root-action stability across completed depths
or allocate proof work selectively from a completed-iteration uncertainty
signal; it should not add another coordinate-specific heuristic or tune away a
failed cohort after the fact.
