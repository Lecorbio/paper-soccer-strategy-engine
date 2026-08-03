# Flagship study analysis contract

Version: 1.1.0

This document fixes the study estimands, selection logic, and analysis before
the frozen test tournament is opened. The machine-readable manifest is the
authoritative configuration. This document explains the definitions used by
the runner and report generator.

## Scope and entrants

The study compares four demo-rule entrant families: **Tactical MctsBot**,
**Hand-evaluated AlphaBetaBot**, **Neural alpha-beta (JacekInspiredBot)**, and
**Rank5DerivedBot — fixed 50k demo profile**. The first three families are
selected from their preregistered grids on validation data. The
**Rank5DerivedBot — fixed 50k demo profile** entrant is a fixed comparator and
is never tuned.

Rank5DerivedBot adapts search code from the rank 5/206 CodinGame submission to
different demo rules and a fixed-work profile. These measurements do not
evaluate the authentic ranked submission.

The opening generator is a seeded, uniform legal-edge data-generation
mechanism. It is not an entrant, opponent, strength baseline, latency result,
or charted configuration. The authentic `rank_5` source is a protected
provenance artifact and is not executable in this study.

## Rules, openings, and outcome semantics

All games use the standard 8x10 demo rules: entering only the opponent goal
wins, and a blocked player to move loses. One opening ply means one applied
physical edge, including an edge on which the same player retains possession
because of a rebound. The 512-ply safety limit counts the opening transcript.

Every bank stores endpoint transcripts and is replayed through the game engine.
The uniform generator samples a legal-edge index with unbiased rejection
sampling. Terminal candidates, duplicate states, and horizontal-reflection
canonical equivalents are rejected. Development, validation, and test banks
are disjoint and immutable after their hashes enter the manifest.

The engine has no draw status. Every legal move consumes a previously unused
edge, the board has 316 playable edges, and goal or blocked termination names a
winner. Results therefore report wins, losses, and truncations. Any truncation
is an operational defect: it is never worth half a point and blocks strength,
Bradley–Terry, and calibration publication until investigated.

For a color-swapped pair, the left entrant scores 1.0 for two wins, 0.5 for a
1–1 split, and 0.0 for two losses. A split is two decisive games, not two
draws.

## Development and validation design

Each of the nine tunable configurations faces the same
**Rank5DerivedBot — fixed 50k demo profile** entrant at 4, 8, 12, and 20
opening plies. This common-opponent star is
connected and balanced: every candidate has the identical opponent and
opening-depth distribution. Development uses 25 color-swapped pairs per
depth/matchup; validation uses 50. Development is for operational checks,
runtime projection, diagnostics, and preregistered budget-scaling summaries.
Validation alone controls the latency gate, selection, calibration fitting,
and validation Pareto frontier.

The runtime projection is released only after the exact 25-pair development
unit has completed for every one of the nine candidate configurations at both
4 and 20 opening plies: 18 preregistered units, or half of the 36-unit
development design. For each configuration and sampled depth, the observed
rate is the unit's whole-process wall duration divided by its 50 completed,
non-truncated games. Depths 8 and 12 use that configuration's slower observed
endpoint for the conservative projection and its faster endpoint as a lower
planning proxy. Full validation preserves each configuration, depth, search
budget, and opening design and scales those rates to the exact 50-pair units.
Because the selected test entrants are not yet known, the lower test proxy uses
the slowest observed candidate rate applicable to each depth; the conservative
test estimate doubles that rate to allow two expensive entrants in a matchup.
The serialized projection records its complete coverage, per-configuration and
per-depth rates, rate ranges, assumptions, and separate ranges for remaining
development, full validation, and all 4,800 test games. The 18 already observed
development units are treated as sunk elapsed work and excluded from the
remaining total. These ranges are planning estimates, not statistical runtime
bounds, and no cheaper or reduced-budget unit may substitute for the required
coverage.

Decision time is the native single-thread wall duration of `choose_move` using
`steady_clock`. Bot construction, post-return legality validation, and
`apply_move` are outside the timer. Internal state preparation or copying done
by `choose_move` remains inside. The manifest records the discarded warm-up,
release flags, machine, compiler, operating system, and power conditions.
Every validation unit is accepted only when both immediately-before and
immediately-after snapshots show AC power, Low Power Mode disabled, and a
nominal macOS thermal/performance-warning state. An unavailable or
non-nominal thermal snapshot invalidates the unit. The validation report and
selection lock retain both raw power-setting and thermal snapshots.

For MCTS and both alpha-beta families, validation p95 uses every returned edge.
For **Rank5DerivedBot — fixed 50k demo profile**, the gate uses only decisions
whose diagnostics mark a fresh-root search. A second all-edge distribution
includes cached rebound
continuations and is reported but never substituted for the fresh-root gate.
Nearest-rank median, p90, p95, p99, and maximum are reported.

Within each tunable family, configurations with p95 greater than 50 ms are
ineligible. Among eligible candidates, validation strength is the mean
depth-stratified pair score against the common fixed comparator. Any candidate
within 1.0 percentage point of the strongest eligible value is practically
tied; the choice then minimizes p95, minimizes computation (iterations or node
budget), and finally uses stable configuration-ID order. If any family has no
eligible configuration, execution stops before test. **Rank5DerivedBot — fixed
50k demo profile** remains in the frozen test tournament even when its
fresh-root p95 exceeds 50 ms, but is then excluded from constrained claims.

## Frozen test tournament

The selection lock contains the manifest and opening hashes, every validation
strength and latency value, the three selected IDs, the fixed
**Rank5DerivedBot — fixed 50k demo profile** identity, validation-only
calibration mappings, validation Pareto data, and all preregistered
development/validation ablations. It must be committed before test.

The test tournament is the complete round robin of the four selected/fixed
entrants: six matchups times four opening depths times 100 color-swapped pairs
times two games, exactly 4,800 games. Test execution has one deterministic run
identity. Interrupted shards may resume that identity; a completed test marker
rejects another independent evaluation unless a user explicitly invokes the
documented destructive override. Test data never changes selection,
calibration coefficients, the Pareto frontier, hypotheses, or chart semantics.

## Pairwise intervals

Every test matchup reports game wins/losses, truncations, pairs won 2–0, split
1–1, and lost 0–2, and mean pair score. The deterministic percentile interval
uses 10,000 bootstrap resamples. Each resample draws whole color-swapped pairs
with replacement within each opening-depth stratum and preserves each stratum's
original size. Individual games are never independently resampled.

A report may say entrant A is stronger than entrant B only when A's paired
test 95% interval has a lower bound strictly above 0.50. Other comparisons are
described as statistically unresolved.

## Bradley–Terry model

The Bradley–Terry model uses binary, non-truncated test game outcomes. For bots
`i` and `j`, `P(i beats j) = logistic(s_i - s_j)`. Strengths obey a sum-to-zero
constraint; zero is only a relative identifiability convention and never
absolute skill. Optimization is deterministic Newton iteration with fixed
tolerance and iteration cap. The implementation rejects disconnected graphs,
singular curvature, non-convergence, and detected complete/quasi separation
instead of returning fabricated finite strengths.

Uncertainty uses 10,000 bootstraps of complete color-swapped pairs, stratified
by matchup and opening depth. Each replicate refits the same sum-to-zero model.
Failed bootstrap fits are counted and reported; intervals are emitted only when
the preregistered success threshold in the manifest is met.

## Calibration

Separate one-dimensional logistic mappings are fitted on validation decisions
for the three selected configurations and **Rank5DerivedBot — fixed 50k demo
profile**. The raw predictor is MCTS root value, hand alpha-beta root score,
neural alpha-beta root score, or the root score from **Rank5DerivedBot — fixed
50k demo profile**. Every score is originally Player-One-oriented and is
sign-flipped for a Player-Two decision so that positive always favors the
player to move. Alpha-beta predictions with no completed depth are invalid.
Cached continuations from **Rank5DerivedBot — fixed 50k demo profile** are not
independent predictions and are excluded; only fresh roots enter fitting and
evaluation.

The binary target is whether the player to move at the prediction eventually
wins that game. Each mapping stores the validation score mean and population
standard deviation plus fitted intercept and slope. Coefficients are frozen in
the committed selection lock. Deterministic Newton iteration accepts finite
coefficients of any magnitude when the gradient tolerance, finite likelihood,
and nonsingular information checks pass; it has no scale-dependent coefficient
cap. Complete and quasi-complete separation, non-finite steps, singular
information, failed line search, and iteration-limit exhaustion fail closed.
Test reports apply the frozen mappings without refitting and
give Brier score, clipped log loss, and ten equal-width probability bins with
prediction count, contributing pair-cluster count, mean prediction, and
observed frequency. Uncertainty uses 10,000 deterministic percentile
bootstraps of complete color-swapped-pair clusters within matchup ×
opening-depth strata. Every retained prediction from both games of a sampled
pair travels with that cluster, preserving within-game and within-pair
dependence. Brier score and log loss always report pair-clustered 95%
intervals. A reliability bin reports an observed-frequency interval only when
at least 1,000 replicates populate that bin; otherwise its interval is null and
its successful-replicate count remains explicit. Per-entrant seeds are domain
derived from the frozen test analysis seed. Decision counts describe
prediction opportunities, not independent samples; pair counts are the
uncertainty units.
For every entrant, the report gives retained predictions over total decision
opportunities and separately discloses exclusions caused by cached
continuations, truncations, and searches with no completed depth.

The analysis implementation uses only the Python standard library. The exact
Python interpreter version serialized in the manifest is enforced for every
flagship command, including aggregation, selection locking, and test analysis;
an interpreter-version mismatch stops execution rather than accepting
potentially different deterministic bootstrap, float, or JSON behavior.

## Pareto frontier and negative findings

The validation plot maximizes common-opponent validation pair-score point
estimates and minimizes validation p95 latency. A point is dominated when
another point is no slower and no weaker, with at least one strict inequality.
Candidate points retain their development and validation pair counts and
pair-bootstrap 95% intervals, plus the number of decisions entering p95. The
0.50 point for **Rank5DerivedBot — fixed 50k demo profile** is a defined common
reference level, not an empirical self-comparison and therefore has no
strength interval or strength sample size; its actual fresh-root timing sample
size remains explicit. The plot distinguishes constrained and unconstrained
Pareto-optimal, dominated, gate-rejected, selected, and fixed-reference points
and draws the preregistered 50 ms line. Test results never revise the frontier.

The preregistered development/validation ablations align configurations on the
same opening IDs and depths. Each contrast resamples whole aligned
color-swapped-pair score differences 10,000 times within opening-depth strata.
MCTS contrasts are 1k→2k, 2k→4k, and 1k→4k iterations. Each alpha-beta family
uses 20k→50k, 50k→100k, and 20k→100k nodes. Equal-budget evaluator contrasts
are neural-minus-hand at 20k, 50k, and 100k nodes. The practical threshold is
an absolute 0.01 pair-score difference. A scaling gain is supported only when
the interval lower bound exceeds +0.01; regression is supported when its upper
bound is below zero; no practical gain is supported when its upper bound is
below +0.01; all other cases are unresolved at 1 percentage point. For the
evaluator contrast, neural or hand material superiority requires the entire
interval above +0.01 or below −0.01 respectively; practical equivalence is
supported only when the entire interval lies inside [−0.01,+0.01]; otherwise
the contrast is unresolved. Point deltas and intervals are always reported,
including successful and unsuccessful results.

The report also preserves latency failures, the fixed strength/latency
tradeoff of **Rank5DerivedBot — fixed 50k demo profile**, and statistically
unresolved test comparisons. There are no post-hoc test ablations.
