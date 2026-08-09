# Topology challenger experiment ledger

## Immutable reference

`../rank_5/submission.cpp` is the accepted incumbent and is not edited by this
experiment. Its expected SHA-256 is
`f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`.

## Hypothesis T1: capped rebound-goal connectivity ordering

**Frozen before implementation.** When the ball is in the outer three rows and
the opponent has a rebound-aware path to goal using at most one fresh landing,
searching path-breaking edges first will expose the defensive continuation one
iteration earlier than the race-oriented incumbent ordering. This changes only
ordering: fully completed minimax scores, the evaluator, mate claims, replay
paths, and clocks remain unchanged.

The alternative low-exit macro solver was rejected before implementation. The
three recurrent elite threats have 11, 10, and 32 frontier exits, so a selective
solver capped at two exits would not activate where the evidence says it must.

The generated candidate is 95,289 characters with SHA-256
`8a4ff025ec9074dbe72533eb779ce3bf84987a11d2efe5dde765d4abbad5e6dc`.
Twelve generated-artifact tests pass, including north/south symmetry, midfield
non-activation, state preservation, and activation on the three frozen elite
threat families.

Promotion requires the frozen gate in `../../promotion/manifest.json`. The
experiment is rejected if any operational, generated-source, timing, validation
confidence, color-balance, or locked-test requirement fails. No live submission
is allowed on development evidence alone.

## Harness negative control

Before T1 was evaluated, an exact `rank_5` clone (93,005 characters, incumbent
SHA-256) ran the initial sentinel and all 96 development pairs. Every pair split
1-1, producing 96-96 games and a paired mean and 95% interval of exactly
`0.5000`. The initial sentinel passed its non-regression requirement, while the
development controller explicitly returned `REJECT` for failing `0.52` and for
not earning more wins than the incumbent. Eight fixed-node shards completed in
about five seconds locally; the old sequential gate took roughly twenty seconds
for only 74 comparable low-node games.

## T1 result: rejected at development

T1 passed preflight, all 12 generated-artifact tests, the source limit, and the
initial sentinel. On the 96-pair / 192-game development bank at 5,000 nodes per
decision it scored **94-98**, with two 2-0 pairs, 90 splits, and four 0-2 pairs.
The opening-pair mean was `0.489583`. After clustering multiple openings from
the same trajectory, the 59-source-game estimate was `0.496469` with a 95%
interval of `[0.468927, 0.523323]`, below the frozen `0.52` requirement. The
candidate also failed the independently required game-win check.

The experiment did not collapse search depth: candidate and incumbent mean
completed depth were `2.3253` and `2.3211`, respectively. It was nevertheless
far too broad inside full games, recording 920,673 active ordering nodes and
418,564 path-breaking candidates. In the final run it consumed 18,530 ms of
measured search time versus 14,619 ms for the incumbent at nearly equal node
counts, a 26.8% time penalty; throughput was only `0.7872x` incumbent (about
`0.78x` across repeated runs), also failing the predeclared `0.90x` throughput
floor. All three changed elite/rank-one pairs regressed 0-2, while both 2-0
gains came from field games.
The regression therefore falsifies both the strategic targeting and the
runtime selectivity of the ordering hypothesis.

Per the staged protocol, validation, the newer rank-one locked test, full timing
distribution, and live Arena submission were not run. `rank_5` remains the
incumbent unchanged. The final decision is an explicit `REJECT` for candidate
SHA-256 `8a4ff025ec9074dbe72533eb779ce3bf84987a11d2efe5dde765d4abbad5e6dc`
under manifest SHA-256
`f5f4ad355a88465e5bfff9a6c90b014d0cac248817f93fc9e155657c55d19a28`.

## Next hypothesis T2: safe inward own-goal rebounds

The next experiment should be a narrow ordering prior for the behavior visible
in rank one's layered defense. In the mover's own-goal shell (Player One at
`y=9..11`, Player Two at `y=1..3`), add about `+250` only when a move travels
toward the mover's own goal, lands on a visited or boundary vertex so the turn
continues, is not a goal, and leaves a nonterminal state. This offsets the
incumbent's roughly 200-point bias against the inward direction without changing
evaluation, proof claims, clocks, or completed minimax values.

The signal replicated observationally: when such a move was available, rank
one selected it in 10/19 development cases versus 7/18 for other elites, and
10/16 validation cases versus 12/25. Rank one also had higher own-shell and
horizontal-shell edge shares in both splits. Because those development and
validation actions were inspected to formulate T2, they are now contaminated
for that hypothesis. T2 must start with newly frozen, source-game-disjoint
development and validation banks from unused games; the uninspected locked test
must remain sealed until all preceding gates pass.
