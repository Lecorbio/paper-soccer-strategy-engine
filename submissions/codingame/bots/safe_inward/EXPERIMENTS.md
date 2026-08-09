# Safe-inward experiment ledger

## Immutable reference

The reference is `../rank_5/submission.cpp`, 93,005 characters, SHA-256
`f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`.

## T2: safe inward own-goal rebounds

The hypothesis was frozen before implementation. In the mover's own goal
shell (Player One at `y=9..11`, Player Two at `y=1..3`), move ordering adds
`+250` only to an edge that travels toward the mover's own goal, grants an
extra turn, does not enter a goal, and leaves at least one legal continuation.
Evaluation, proof claims, transposition semantics, replay data, and clocks are
unchanged.

The generated candidate is 93,737 characters with SHA-256
`a95d66673d46b33f5d25b5175818f78240bed8da8504c1a7bb1cc38a2b052575`.
Thirteen focused artifact tests pass, including north/south symmetry, boundary
rebounds, fresh and trapped negatives, exact complete-action replay, and input
state preservation.

The experiment used a newly frozen, source-game-disjoint corpus. All 298 prior
raw game IDs were excluded from 90 current-bot games, leaving 87. The first 50
formed development; a retired rank-one holdout became validation; and the
latest 37 games remain the sealed final test. Winner tier is preserved as a
separate gate dimension.

## Result: rejected at development

T2 passed preflight, source-size, artifact, protocol, initial-sentinel, and
throughput checks. On 48 color-swapped development pairs at 5,000 nodes it
lost **46-50**: three 2-0 pairs, 40 splits, and five 0-2 pairs. The
26-source-game clustered score was `0.472756`, with 95% interval
`[0.402244, 0.541667]`, below the frozen `0.52` requirement.

The elite-winner cohort scored only `0.409091`; the incumbent-winner cohort
scored `0.500000`. Throughput was `1.0121x` the incumbent, so runtime was not
the cause. The strategy signal itself failed on fresh play. Validation, the
new sealed test, timing, and live Arena submission were not run.

The decision is `REJECT` for candidate SHA-256
`a95d66673d46b33f5d25b5175818f78240bed8da8504c1a7bb1cc38a2b052575`
under manifest SHA-256
`6ae69f354b5d72b1fa8bff08938692d52e51ec811993a37ffd2c55c899a20f25`.
