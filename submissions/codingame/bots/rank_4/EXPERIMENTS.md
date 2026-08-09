# Rank-4 production and improvement record

## Production promotion

The canonical source is the retained `selfplay_nn_v2` 800/165 ms artifact:
98,624 characters, SHA-256
`5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`.
The original model training, version 42-44 live batches, and chronological loss
regressions remain under `../selfplay_nn_v2`; moving them would invalidate
frozen path-and-hash evidence manifests.

Authenticated CodinGame inspection on 2026-08-09 identified history version
56 as the same implementation by its 800/165 ms clock, 100% residual weight,
and unique model coefficients. Public battle metadata bound the completed
batch to agent `6604719` and submission `41114327`:

- rank 4 of 208, score `44.29750553418035`, 66 wins and 24 losses;
- 5-6 against Deltaspace, 9-2 against derjack, and 6-2 against EricSMSO;
- 0-7 against jacek and 0-3 against Marchete;
- 6-0 against Laars and 5-1 against Snekkers; and
- no incomplete arena games or observed operational failures.

A public leaderboard recheck on 2026-08-10 still reported agent `6604719` at
rank 4 of 208, score 44.3 (rounded by that endpoint), with arena processing
complete.

The previous `rank_5` reference was rank 5 of 206 with score
`42.42773147296124` and a 57-33 record. The new batch is +1.86977406121911
score and +10 percentage points of raw win rate, but the two arena schedules
were not paired and the field changed. Deterministic local comparisons supply
the stronger causal evidence: the same rank-4 source beat `rank_5` 178-128 at
5,000 nodes over 306 games, 58-48 at 30,000 nodes over 106 games, and 13-5 in
the retained-profile timed screen. The timed screen gave rank 4 its production
800/165 ms profile and `rank_5` its 650/130 ms profile, so it is not an
equal-clock result.

## Can it be improved?

Yes, but the evidence argues for improving search reliability before making
the residual larger or adding another broad training run.

The first candidate worth testing is rank 4 plus the already implemented
first-full-exchange exact rebound proof from `exchange_proof`. That proof beat
`rank_5` 77-67 at the 130 ms deployment profile and was positive across its
exposed fixed-work screens. Combining the two mechanisms is still an untested
interaction, not an assumed gain. A behavior-neutral removal of leading
indentation reduces the current 98,624-character artifact to about 86,930
characters; that creates enough room to test the proof while remaining below
CodinGame's 100,000-character cap. The canonical rank-4 artifact itself stays
byte-identical and uncompacted.

The second credible direction is an uncertainty-gated residual with explicit
corridor, goal-line parity, escape-route, and safe-handoff topology features.
The present linear residual improves teacher-score error only slightly and can
change alpha-beta pruning enough to select a worse completed-depth action.
Any successor should down-weight corrections when deeper teachers disagree and
must beat both the broad paired bank and a frozen elite-loss/deeper-teacher
agreement gate.

Previously rejected directions should not be repeated unchanged:

- 800/130 ms and 900/180 ms clock changes redistributed the unstable
  `0/6/33` versus `0/6/5` horizon family without robustly improving play;
- iterative-deepening reuse and aspiration windows searched deeper but lost
  their paired gates;
- exact-history imitation patches failed when elite opponents deviated from
  the copied continuation; and
- a much larger teacher corpus produced mixed gates and did not establish
  stronger arena performance by itself.

The version-56 battles are post-selection diagnostics. They should be frozen
as held-out regression evidence, not fitted and then reused as promotion proof.
No sealed prospective or final bank should be opened for an exploratory
rank-4 successor.
