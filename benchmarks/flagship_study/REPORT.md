# Competitive demo-rule Paper Soccer bot study

## Research question and hypotheses

**Question.** Under standard 8×10 demo rules, which of the four competitive entrants has the strongest frozen test performance, the best calibrated predictions, and the most favorable validation strength/latency tradeoff?

The preregistered hypotheses were:

- Additional fixed computation may improve paired strength, with a measurable latency cost.
- Hand-crafted and neural alpha-beta evaluation may differ in both strength and calibration at equal node budgets.
- The fixed Rank5DerivedBot — fixed 50k demo profile may fall on or off the constrained validation frontier without any profile tuning.

## Entrants

- **Tactical MctsBot.** Tactical rollouts, rollout-only leaves, deterministic tree reuse, and the frozen iteration sweep.
- **Hand-evaluated AlphaBetaBot.** Depth-six possession search with the hand evaluator and fixed node budgets.
- **Neural alpha-beta (JacekInspiredBot).** The separate depth-six neural entrant using the frozen model hash listed below.
- **Rank5DerivedBot — fixed 50k demo profile.** The exact hard-locked 32-turn-depth, 50,000-node demo comparator with replay corrections and learned-value blending disabled.

Rank5DerivedBot adapts search code from the rank 5/206 CodinGame submission to different demo rules and a fixed-work profile. These measurements do not evaluate the authentic ranked submission.

## Controls and frozen openings

Games used 8×10 demo rules, 316 playable edges, and a 512-ply safety limit. The engine has no draw status.

Openings at 4, 8, 12, 20 physical plies were produced by **Separate deterministic uniform legal-edge data-generation mechanism**. It was solely a data-generation mechanism, never an entrant or strength baseline. Transcripts were replay-validated, phase-disjoint, duplicate-screened, color-swapped, and frozen by the hashes below.

Opening ply: One physical legal edge application, including same-player rebound edges; opening plies count toward the 512-ply safety limit.

## Prospective recovery lineage

Version 4 prospectively superseded version 3 after the predecessor stopped before test because of a validation calibration implementation defect. No version-3 test outcomes were accessed, and no version-3 validation results were used for version-4 selection or calibration. Version 4 used fresh validation banks excluded from every predecessor opening bank while reusing the development and test banks byte-for-byte. The predecessor manifest and failure record are bound by SHA-256 in the artifact table.

## Candidate grids

| Family | Configuration | Frozen work/profile |
| --- | --- | --- |
| Tactical MCTS | mcts-1000 | 1,000 iterations; tactical rollout; rollout-only leaf; tree reuse |
| Tactical MCTS | mcts-2000 | 2,000 iterations; tactical rollout; rollout-only leaf; tree reuse |
| Tactical MCTS | mcts-4000 | 4,000 iterations; tactical rollout; rollout-only leaf; tree reuse |
| Hand alpha-beta | alpha-beta-20k | depth 6; 20,000 nodes; wall clock disabled; hand evaluator |
| Hand alpha-beta | alpha-beta-50k | depth 6; 50,000 nodes; wall clock disabled; hand evaluator |
| Hand alpha-beta | alpha-beta-100k | depth 6; 100,000 nodes; wall clock disabled; hand evaluator |
| Neural alpha-beta | jacek-20k | depth 6; 20,000 nodes; wall clock disabled; frozen neural model |
| Neural alpha-beta | jacek-50k | depth 6; 50,000 nodes; wall clock disabled; frozen neural model |
| Neural alpha-beta | jacek-100k | depth 6; 100,000 nodes; wall clock disabled; frozen neural model |
| Rank5DerivedBot — fixed 50k demo profile | rank5-fixed-50k | depth 32; 50,000 nodes; 65,536 TT; 32,768 eval cache; wall clock disabled; replay corrections disabled; 0% learned blend; seed ignored; standard-8x10-demo rules |

## Latency protocol

Validation used a native Release, one foreground thread, and a 50 ms p95 gate. Timer boundary: steady_clock immediately around Bot::choose_move(state). Warm-up: Eight untimed deterministic decisions per entrant in each arena process.

State/setup policy: Caller passes const state by reference; bot-internal setup/copying is timed. Power conditions: Gate machine on AC power, Low Power Mode disabled, nominal thermal state, single foreground arena process; observed state recorded with results.

Rank5DerivedBot — fixed 50k demo profile eligibility uses fresh-root p95; all returned edges, including cached continuations, are reported separately.

Gate machine: apple-m4-pro-local-gate; macOS 26.5.2; Apple M4 Pro; 21.0.0.21000101. Build flags: -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic.

## Selection rule

Selection used validation mean depth-stratified color-swapped pair score versus Rank5DerivedBot — fixed 50k demo profile. Candidates within 1 percentage point of the strongest eligible result were tied; ties were resolved by lower p95, smaller work budget, then stable configuration ID. A family with no eligible candidate would have stopped the study before test.

## Development and validation findings

| Configuration | Development score | Validation score | Validation p95 (ms) | Gate | Lock |
| --- | --- | --- | --- | --- | --- |
| alpha-beta-100k | 46.0% | 44.0% | 25.012 | eligible | — |
| alpha-beta-20k | 43.0% | 39.0% | 13.281 | eligible | — |
| alpha-beta-50k | 46.5% | 43.0% | 24.273 | eligible | selected |
| jacek-100k | 57.0% | 51.0% | 58.720 | rejected | — |
| jacek-20k | 54.0% | 55.5% | 35.718 | eligible | selected |
| jacek-50k | 54.0% | 52.8% | 58.231 | rejected | — |
| mcts-1000 | 14.0% | 17.5% | 35.684 | eligible | selected |
| mcts-2000 | 22.5% | 27.5% | 69.957 | rejected | — |
| mcts-4000 | 39.5% | 38.5% | 136.385 | rejected | — |

Rank5DerivedBot — fixed 50k demo profile fresh-root p95 was 31.383 ms; all-edge p95 was 27.237 ms. Its constrained status was eligible; the profile remained fixed either way.

### Preregistered development/validation ablations

All contrasts align the same opening pairs and use 10,000 whole-pair difference bootstraps stratified by opening depth. The frozen practical threshold is 1.0%. Test outcomes do not enter these classifications.

| Family | Contrast | Development scores | Development delta [95% CI] | Validation scores | Validation delta [95% CI] | Validation class |
| --- | --- | --- | --- | --- | --- | --- |
| Tactical MCTS | mcts-1000 → mcts-2000 | 14.0% → 22.5% (n=100) | +8.5 pp [+2.5, +14.5] pp | 17.5% → 27.5% (n=200) | +10.0 pp [+4.8, +15.2] pp | supported practical gain |
| Tactical MCTS | mcts-2000 → mcts-4000 | 22.5% → 39.5% (n=100) | +17.0 pp [+9.0, +25.0] pp | 27.5% → 38.5% (n=200) | +11.0 pp [+5.0, +16.8] pp | supported practical gain |
| Tactical MCTS | mcts-1000 → mcts-4000 | 14.0% → 39.5% (n=100) | +25.5 pp [+18.5, +32.5] pp | 17.5% → 38.5% (n=200) | +21.0 pp [+15.8, +26.0] pp | supported practical gain |
| Hand alpha-beta | alpha-beta-20k → alpha-beta-50k | 43.0% → 46.5% (n=100) | +3.5 pp [-4.5, +11.5] pp | 39.0% → 43.0% (n=200) | +4.0 pp [-1.8, +9.8] pp | unresolved at 1pp |
| Hand alpha-beta | alpha-beta-50k → alpha-beta-100k | 46.5% → 46.0% (n=100) | -0.5 pp [-4.5, +3.5] pp | 43.0% → 44.0% (n=200) | +1.0 pp [-1.8, +3.5] pp | unresolved at 1pp |
| Hand alpha-beta | alpha-beta-20k → alpha-beta-100k | 43.0% → 46.0% (n=100) | +3.0 pp [-4.5, +10.5] pp | 39.0% → 44.0% (n=200) | +5.0 pp [-0.5, +10.5] pp | unresolved at 1pp |
| Neural alpha-beta | jacek-20k → jacek-50k | 54.0% → 54.0% (n=100) | +0.0 pp [-8.5, +9.0] pp | 55.5% → 52.8% (n=200) | -2.8 pp [-9.2, +3.8] pp | unresolved at 1pp |
| Neural alpha-beta | jacek-50k → jacek-100k | 54.0% → 57.0% (n=100) | +3.0 pp [-2.0, +8.0] pp | 52.8% → 51.0% (n=200) | -1.7 pp [-4.8, +1.2] pp | unresolved at 1pp |
| Neural alpha-beta | jacek-20k → jacek-100k | 54.0% → 57.0% (n=100) | +3.0 pp [-5.5, +11.5] pp | 55.5% → 51.0% (n=200) | -4.5 pp [-11.0, +2.0] pp | unresolved at 1pp |
| Neural minus hand | alpha-beta-20k → jacek-20k (neural minus hand) | 43.0% → 54.0% (n=100) | +11.0 pp [+2.0, +20.0] pp | 39.0% → 55.5% (n=200) | +16.5 pp [+10.0, +23.0] pp | neural materially stronger |
| Neural minus hand | alpha-beta-50k → jacek-50k (neural minus hand) | 46.5% → 54.0% (n=100) | +7.5 pp [-1.5, +16.5] pp | 43.0% → 52.8% (n=200) | +9.7 pp [+2.8, +16.5] pp | neural materially stronger |
| Neural minus hand | alpha-beta-100k → jacek-100k (neural minus hand) | 46.0% → 57.0% (n=100) | +11.0 pp [+2.5, +19.5] pp | 44.0% → 51.0% (n=200) | +7.0 pp [+0.2, +14.0] pp | unresolved at 1pp |

## Locked configurations

| Entrant | Locked ID | Exact profile | Basis |
| --- | --- | --- | --- |
| Tactical MCTS | mcts-1000 | 1,000 iterations; tactical rollout; rollout-only leaf; tree reuse | validation selection |
| Hand alpha-beta | alpha-beta-50k | depth 6; 50,000 nodes; wall clock disabled; hand evaluator | validation selection |
| Neural alpha-beta | jacek-20k | depth 6; 20,000 nodes; wall clock disabled; frozen neural model | validation selection |
| Rank5DerivedBot — fixed 50k demo profile | rank5-fixed-50k | depth 32; 50,000 nodes; 65,536 TT; 32,768 eval cache; wall clock disabled; replay corrections disabled; 0% learned blend; seed ignored; standard-8x10-demo rules | fixed comparator |

## Frozen test results

The frozen tournament completed 4,800 decisive games with zero truncations. A 1–1 pair split is two decisive games, not a draw.

### Pairwise results

Intervals use 10,000 deterministic whole-pair bootstrap resamples while preserving opening-depth strata.

| Left entrant | Right entrant | Wins | Losses | 2–0 | 1–1 | 0–2 | Mean paired score | Paired bootstrap 95% CI | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tactical MctsBot | Hand-evaluated AlphaBetaBot | 146 | 654 | 6 | 134 | 260 | 18.2% | [15.8%, 20.8%] | **Hand-evaluated AlphaBetaBot is stronger than Tactical MctsBot.** |
| Tactical MctsBot | Neural alpha-beta (JacekInspiredBot) | 119 | 681 | 8 | 103 | 289 | 14.9% | [12.5%, 17.4%] | **Neural alpha-beta (JacekInspiredBot) is stronger than Tactical MctsBot.** |
| Tactical MctsBot | Rank5DerivedBot — fixed 50k demo profile | 157 | 643 | 8 | 141 | 251 | 19.6% | [17.0%, 22.2%] | **Rank5DerivedBot — fixed 50k demo profile is stronger than Tactical MctsBot.** |
| Hand-evaluated AlphaBetaBot | Neural alpha-beta (JacekInspiredBot) | 317 | 483 | 46 | 225 | 129 | 39.6% | [36.5%, 42.6%] | **Neural alpha-beta (JacekInspiredBot) is stronger than Hand-evaluated AlphaBetaBot.** |
| Hand-evaluated AlphaBetaBot | Rank5DerivedBot — fixed 50k demo profile | 337 | 463 | 47 | 243 | 110 | 42.1% | [39.2%, 45.0%] | **Rank5DerivedBot — fixed 50k demo profile is stronger than Hand-evaluated AlphaBetaBot.** |
| Neural alpha-beta (JacekInspiredBot) | Rank5DerivedBot — fixed 50k demo profile | 411 | 389 | 86 | 239 | 75 | 51.4% | [48.2%, 54.5%] | Statistically unresolved. |

## Bradley–Terry relative strength

Abilities use a sum-to-zero identifiability constraint; zero is a relative reference, not absolute skill. Uncertainty resamples complete color-swapped pairs within matchup and opening-depth strata.

![Test Bradley–Terry relative strength](charts/test_bradley_terry.svg)

| Entrant | Relative ability | Bootstrap 95% CI |
| --- | --- | --- |
| Tactical MctsBot | -1.165 | [-1.242, -1.092] |
| Hand-evaluated AlphaBetaBot | 0.176 | [0.120, 0.233] |
| Neural alpha-beta (JacekInspiredBot) | 0.551 | [0.490, 0.614] |
| Rank5DerivedBot — fixed 50k demo profile | 0.437 | [0.376, 0.499] |

## Calibration

Logistic mappings were fitted and frozen on validation scores after Player-One-to-player-to-move orientation and population-standardization. Test metrics apply those mappings without refitting. Rank5DerivedBot — fixed 50k demo profile uses fresh-root predictions only. Uncertainty resamples whole color-swapped pairs within matchup × opening-depth strata, preserving dependent decisions within both games.

![Test reliability and calibration](charts/test_calibration.svg)

| Entrant | Retained/decision opportunities | Excluded cached | Excluded invalid depth | Excluded truncation | Pair clusters | Brier [pair 95% CI] | Log loss [pair 95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tactical MctsBot | 60157/60157 | 0 | 0 | 0 | 1200 | 0.1479 [0.1402, 0.1559] | 0.4610 [0.4423, 0.4803] |
| Hand-evaluated AlphaBetaBot | 153385/153427 | 0 | 42 | 0 | 1200 | 0.2036 [0.2003, 0.2069] | 0.5818 [0.5725, 0.5915] |
| Neural alpha-beta (JacekInspiredBot) | 158831/158999 | 0 | 168 | 0 | 1200 | 0.1919 [0.1889, 0.1949] | 0.5512 [0.5440, 0.5583] |
| Rank5DerivedBot — fixed 50k demo profile | 39098/136088 | 96837 | 153 | 0 | 1200 | 0.2157 [0.2141, 0.2173] | 0.6146 [0.6085, 0.6214] |

### Ten-bin reliability summaries

| Entrant | Bin | Probability range | Prediction n | Pair n | Mean prediction | Observed frequency | Pair-bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tactical MctsBot | 0 | [0.0, 0.1) | 5932 | 688 | 0.044 | 0.039 | [0.028, 0.051] (10,000 populated replicates) |
| Tactical MctsBot | 1 | [0.1, 0.2) | 20659 | 1049 | 0.153 | 0.135 | [0.118, 0.154] (10,000 populated replicates) |
| Tactical MctsBot | 2 | [0.2, 0.3) | 12037 | 1052 | 0.243 | 0.188 | [0.165, 0.211] (10,000 populated replicates) |
| Tactical MctsBot | 3 | [0.3, 0.4) | 6860 | 1000 | 0.347 | 0.239 | [0.211, 0.268] (10,000 populated replicates) |
| Tactical MctsBot | 4 | [0.4, 0.5) | 4895 | 789 | 0.447 | 0.356 | [0.317, 0.395] (10,000 populated replicates) |
| Tactical MctsBot | 5 | [0.5, 0.6) | 4128 | 550 | 0.548 | 0.493 | [0.444, 0.541] (10,000 populated replicates) |
| Tactical MctsBot | 6 | [0.6, 0.7) | 2917 | 427 | 0.645 | 0.638 | [0.582, 0.692] (10,000 populated replicates) |
| Tactical MctsBot | 7 | [0.7, 0.8) | 1363 | 263 | 0.743 | 0.874 | [0.827, 0.913] (10,000 populated replicates) |
| Tactical MctsBot | 8 | [0.8, 0.9) | 611 | 189 | 0.843 | 0.964 | [0.923, 0.992] (10,000 populated replicates) |
| Tactical MctsBot | 9 | [0.9, 1.0] | 755 | 388 | 0.942 | 0.997 | [0.993, 1.000] (10,000 populated replicates) |
| Hand-evaluated AlphaBetaBot | 0 | [0.0, 0.1) | 13020 | 707 | 0.006 | 0.009 | [0.004, 0.015] (10,000 populated replicates) |
| Hand-evaluated AlphaBetaBot | 1 | [0.1, 0.2) | 0 | 0 | — | — | — (0 populated replicates) |
| Hand-evaluated AlphaBetaBot | 2 | [0.2, 0.3) | 3059 | 524 | 0.290 | 0.142 | [0.111, 0.175] (10,000 populated replicates) |
| Hand-evaluated AlphaBetaBot | 3 | [0.3, 0.4) | 43847 | 952 | 0.353 | 0.328 | [0.303, 0.353] (10,000 populated replicates) |
| Hand-evaluated AlphaBetaBot | 4 | [0.4, 0.5) | 56512 | 1025 | 0.459 | 0.524 | [0.499, 0.548] (10,000 populated replicates) |
| Hand-evaluated AlphaBetaBot | 5 | [0.5, 0.6) | 26787 | 1001 | 0.522 | 0.648 | [0.622, 0.675] (10,000 populated replicates) |
| Hand-evaluated AlphaBetaBot | 6 | [0.6, 0.7) | 3 | 3 | 0.608 | 0.667 | [0.000, 1.000] (9,515 populated replicates) |
| Hand-evaluated AlphaBetaBot | 7 | [0.7, 0.8) | 0 | 0 | — | — | — (0 populated replicates) |
| Hand-evaluated AlphaBetaBot | 8 | [0.8, 0.9) | 0 | 0 | — | — | — (0 populated replicates) |
| Hand-evaluated AlphaBetaBot | 9 | [0.9, 1.0] | 10157 | 962 | 0.991 | 0.962 | [0.942, 0.979] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 0 | [0.0, 0.1) | 8553 | 549 | 0.013 | 0.007 | [0.003, 0.013] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 1 | [0.1, 0.2) | 1073 | 229 | 0.147 | 0.087 | [0.041, 0.148] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 2 | [0.2, 0.3) | 1012 | 246 | 0.251 | 0.130 | [0.085, 0.184] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 3 | [0.3, 0.4) | 2090 | 395 | 0.359 | 0.255 | [0.200, 0.313] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 4 | [0.4, 0.5) | 24712 | 922 | 0.477 | 0.493 | [0.463, 0.522] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 5 | [0.5, 0.6) | 76908 | 1024 | 0.553 | 0.662 | [0.639, 0.684] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 6 | [0.6, 0.7) | 17603 | 955 | 0.639 | 0.734 | [0.706, 0.760] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 7 | [0.7, 0.8) | 6918 | 713 | 0.744 | 0.773 | [0.737, 0.808] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 8 | [0.8, 0.9) | 3952 | 537 | 0.848 | 0.843 | [0.803, 0.879] (10,000 populated replicates) |
| Neural alpha-beta (JacekInspiredBot) | 9 | [0.9, 1.0] | 16010 | 1083 | 0.984 | 0.977 | [0.968, 0.985] (10,000 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 0 | [0.0, 0.1) | 1177 | 594 | 0.000 | 0.008 | [0.003, 0.014] (10,000 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 1 | [0.1, 0.2) | 0 | 0 | — | — | — (0 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 2 | [0.2, 0.3) | 0 | 0 | — | — | — (0 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 3 | [0.3, 0.4) | 853 | 228 | 0.315 | 0.032 | [0.013, 0.058] (10,000 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 4 | [0.4, 0.5) | 39 | 34 | 0.498 | 0.154 | [0.024, 0.319] (10,000 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 5 | [0.5, 0.6) | 34254 | 1025 | 0.538 | 0.589 | [0.567, 0.611] (10,000 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 6 | [0.6, 0.7) | 0 | 0 | — | — | — (0 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 7 | [0.7, 0.8) | 561 | 251 | 0.743 | 0.986 | [0.974, 0.995] (10,000 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 8 | [0.8, 0.9) | 0 | 0 | — | — | — (0 populated replicates) |
| Rank5DerivedBot — fixed 50k demo profile | 9 | [0.9, 1.0] | 2214 | 1054 | 1.000 | 1.000 | [1.000, 1.000] (10,000 populated replicates) |

## Validation Pareto frontier

The constrained frontier includes only configurations at or below the 50 ms gate, maximizes common-opponent validation paired score, and minimizes validation p95 latency. Unconstrained status is shown separately; test results never revise either classification.

![Validation strength versus p95 latency](charts/validation_pareto.svg)

| Configuration | Development score [95% CI] | Validation score [95% CI] | Validation p95 ms (decision n) | Status |
| --- | --- | --- | --- | --- |
| alpha-beta-100k | 46.0% [40.5%, 51.5%] (n=100) | 44.0% [39.8%, 48.2%] (n=200) | 25.012 (n=31290) | constrained-Pareto, unconstrained-Pareto |
| alpha-beta-20k | 43.0% [37.0%, 49.5%] (n=100) | 39.0% [34.8%, 43.2%] (n=200) | 13.281 (n=31733) | constrained-Pareto, unconstrained-Pareto |
| alpha-beta-50k | 46.5% [40.5%, 52.5%] (n=100) | 43.0% [38.5%, 47.5%] (n=200) | 24.273 (n=31120) | constrained-Pareto, unconstrained-Pareto, selected |
| jacek-100k | 57.0% [51.0%, 63.0%] (n=100) | 51.0% [46.2%, 55.8%] (n=200) | 58.720 (n=33224) | outside constrained frontier, gate-rejected, unconstrained-dominated |
| jacek-20k | 54.0% [48.0%, 60.0%] (n=100) | 55.5% [51.0%, 59.8%] (n=200) | 35.718 (n=33161) | constrained-Pareto, unconstrained-Pareto, selected |
| jacek-50k | 54.0% [48.0%, 60.5%] (n=100) | 52.8% [47.8%, 57.8%] (n=200) | 58.231 (n=33369) | outside constrained frontier, gate-rejected, unconstrained-dominated |
| mcts-1000 | 14.0% [10.0%, 18.5%] (n=100) | 17.5% [14.2%, 21.2%] (n=200) | 35.684 (n=11089) | constrained-dominated, unconstrained-dominated, selected |
| mcts-2000 | 22.5% [17.5%, 28.0%] (n=100) | 27.5% [23.5%, 31.8%] (n=200) | 69.957 (n=14554) | outside constrained frontier, gate-rejected, unconstrained-dominated |
| mcts-4000 | 39.5% [33.5%, 46.0%] (n=100) | 38.5% [34.0%, 42.8%] (n=200) | 136.385 (n=16964) | outside constrained frontier, gate-rejected, unconstrained-dominated |
| Rank5DerivedBot — fixed 50k demo profile | 50.0% (defined; n=N/A) | 50.0% (defined; n=N/A) | 31.383 (n=65605) | constrained-Pareto, unconstrained-Pareto, selected, fixed |

## Negative and statistically unresolved findings

- jacek-100k missed the 50 ms validation p95 gate (58.720 ms).
- jacek-50k missed the 50 ms validation p95 gate (58.231 ms).
- mcts-2000 missed the 50 ms validation p95 gate (69.957 ms).
- mcts-4000 missed the 50 ms validation p95 gate (136.385 ms).
- jacek-100k missed the 50 ms gate and was excluded from the constrained frontier.
- jacek-50k missed the 50 ms gate and was excluded from the constrained frontier.
- mcts-1000 was dominated on the frozen validation frontier.
- mcts-2000 missed the 50 ms gate and was excluded from the constrained frontier.
- mcts-4000 missed the 50 ms gate and was excluded from the constrained frontier.
- alpha-beta-20k → alpha-beta-50k: validation classification was unresolved at 1pp; the paired interval and point delta are reported in the preregistered ablation table.
- alpha-beta-50k → alpha-beta-100k: validation classification was unresolved at 1pp; the paired interval and point delta are reported in the preregistered ablation table.
- alpha-beta-20k → alpha-beta-100k: validation classification was unresolved at 1pp; the paired interval and point delta are reported in the preregistered ablation table.
- jacek-20k → jacek-50k: validation classification was unresolved at 1pp; the paired interval and point delta are reported in the preregistered ablation table.
- jacek-50k → jacek-100k: validation classification was unresolved at 1pp; the paired interval and point delta are reported in the preregistered ablation table.
- jacek-20k → jacek-100k: validation classification was unresolved at 1pp; the paired interval and point delta are reported in the preregistered ablation table.
- alpha-beta-100k → jacek-100k (neural minus hand): validation classification was unresolved at 1pp; the paired interval and point delta are reported in the preregistered ablation table.
- Neural alpha-beta (JacekInspiredBot) versus Rank5DerivedBot — fixed 50k demo profile was statistically unresolved.

## Limitations and threats to validity

- Latency is machine-, compiler-, power-, and thermal-state-specific; fixed work improves reproducibility but not cross-machine timing equivalence.
- Frozen opening banks control color and opening variation but do not enumerate every reachable position.
- Bradley–Terry abilities are relative to this four-entrant comparison graph and ruleset.
- Calibration decisions within a game are dependent; prediction counts are not independent-game sample sizes.
- The hand-versus-neural comparison changes the evaluator within a shared search family and does not isolate every implementation interaction.
- Rank5DerivedBot — fixed 50k demo profile is measured only under demo rules and its fixed-work profile, as stated in the provenance disclaimer.
- Presentation-only correction (2026-08-04): the validation Pareto chart moved annotations into collision-free keyed callouts and a structured detail panel. Plotted data values, uncertainty intervals, selection, and constrained/unconstrained classifications are unchanged. The original frozen SVG remains recoverable from tag `flagship-study-v4-record` (SHA-256 `31e33102f42cbedfb9059b2a9b1c7dd44d97e0906098c7bf778422d3db5c7813`); no arena, test, calibration, or statistical analysis was rerun.

## Exact reproduction commands

From a clean clone, the following launches a new from-source rerun with the same preregistered inputs. It does not reuse or overwrite the completed frozen test identity reported above:

```bash
git switch -c reproduce-flagship-study 2c023167e4ed8b3d7b887c363b6ec4b431db2dd9
test "$(python3 -c 'import platform; print(platform.python_version())')" = "3.14.6"
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
  -DPAPERSOCCER_ENABLE_SANITIZERS=OFF
cmake --build build/release --parallel
test "$(shasum -a 256 build/release/papersoccer_arena | awk '{print $1}')" = "8ebac0e49d126ffe31ae9cc0743146312275b6e433d32b2da5e3b7821dce742f"
test "$(shasum -a 256 build/release/papersoccer_opening_bank | awk '{print $1}')" = "54e63bd7e6ee03f54a3b036fbb68dba4d33982e386d6fd2f288984133bb47465"
python3 benchmarks/flagship_study/prepare_manifest.py \
  --opening-tool build/release/papersoccer_opening_bank \
  --source-commit 2c023167e4ed8b3d7b887c363b6ec4b431db2dd9 \
  --preregistered-at-utc 2026-08-03T19:36:00+00:00 \
  --fresh-validation-keep-frozen-test
python3 benchmarks/flagship_study/run_study.py validate
git add benchmarks/flagship_study/manifest.json benchmarks/flagship_study/openings
git commit -m 'Freeze flagship manifest and opening banks'
for index in 0 3 4 7 8 11 12 15 16 19 20 23 24 27 28 31 32 35; do
  python3 benchmarks/flagship_study/run_study.py run --phase development \
    --arena build/release/papersoccer_arena --shard-count 36 --shard-index "$index"
done
python3 benchmarks/flagship_study/run_study.py project-runtime --write
for index in $(seq 0 35); do
  python3 benchmarks/flagship_study/run_study.py run --phase development \
    --arena build/release/papersoccer_arena --shard-count 36 --shard-index "$index"
done
python3 benchmarks/flagship_study/run_study.py aggregate --phase development
for index in $(seq 0 35); do
  python3 benchmarks/flagship_study/run_study.py run --phase validation \
    --arena build/release/papersoccer_arena --shard-count 36 --shard-index "$index"
done
python3 benchmarks/flagship_study/run_study.py aggregate --phase validation
python3 benchmarks/flagship_study/run_study.py lock-selection
git add benchmarks/flagship_study/data/development.json \
  benchmarks/flagship_study/data/validation.json \
  benchmarks/flagship_study/runtime_projection.json \
  benchmarks/flagship_study/selection_lock.json
git commit -m 'Lock flagship validation selection'
for index in $(seq 0 23); do
  python3 benchmarks/flagship_study/run_study.py run --phase test \
    --arena build/release/papersoccer_arena --shard-count 24 --shard-index "$index"
done
python3 benchmarks/flagship_study/run_study.py aggregate --phase test
python3 benchmarks/flagship_study/run_study.py analyze-test
git add benchmarks/flagship_study/data/test.json \
  benchmarks/flagship_study/charts benchmarks/flagship_study/REPORT.md
git commit -m 'Publish frozen flagship test analysis'
```

Each indexed test command resumes the same frozen run identity and refuses a second completed evaluation. No destructive override is used.

## Artifact hashes

Source commit: `2c023167e4ed8b3d7b887c363b6ec4b431db2dd9`

| Build executable | SHA-256 |
| --- | --- |
| Native arena | `8ebac0e49d126ffe31ae9cc0743146312275b6e433d32b2da5e3b7821dce742f` |
| Opening-bank generator | `54e63bd7e6ee03f54a3b036fbb68dba4d33982e386d6fd2f288984133bb47465` |

### Observed validation execution environment

| Run environment | Start UTC | End UTC | CPU | OS/kernel | Python | Compiler | Build flags | Power/settings start → end | Thermal start → end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-08-03T20:21:29+00:00 | 2026-08-03T20:23:17+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 2 | 2026-08-03T20:30:17+00:00 | 2026-08-03T20:33:17+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 3 | 2026-08-03T21:28:59+00:00 | 2026-08-03T21:32:41+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 4 | 2026-08-03T20:55:30+00:00 | 2026-08-03T20:59:08+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 5 | 2026-08-03T21:20:48+00:00 | 2026-08-03T21:23:49+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 6 | 2026-08-03T20:33:17+00:00 | 2026-08-03T20:35:58+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 7 | 2026-08-03T20:35:59+00:00 | 2026-08-03T20:37:46+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 8 | 2026-08-03T20:44:33+00:00 | 2026-08-03T20:50:16+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 9 | 2026-08-03T20:24:47+00:00 | 2026-08-03T20:25:56+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 10 | 2026-08-03T20:23:18+00:00 | 2026-08-03T20:24:46+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 11 | 2026-08-03T21:08:07+00:00 | 2026-08-03T21:09:42+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, runningboardd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 12 | 2026-08-03T21:32:42+00:00 | 2026-08-03T21:36:11+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 13 | 2026-08-03T20:50:16+00:00 | 2026-08-03T20:55:30+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 14 | 2026-08-03T20:25:57+00:00 | 2026-08-03T20:26:49+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 15 | 2026-08-03T21:04:36+00:00 | 2026-08-03T21:06:23+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, runningboardd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 16 | 2026-08-03T20:26:50+00:00 | 2026-08-03T20:30:16+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 17 | 2026-08-03T21:26:50+00:00 | 2026-08-03T21:28:59+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 18 | 2026-08-03T21:46:03+00:00 | 2026-08-03T21:49:36+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 19 | 2026-08-03T20:37:47+00:00 | 2026-08-03T20:44:32+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 20 | 2026-08-03T21:39:42+00:00 | 2026-08-03T21:42:16+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 21 | 2026-08-03T21:02:10+00:00 | 2026-08-03T21:03:34+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 22 | 2026-08-03T21:42:17+00:00 | 2026-08-03T21:46:02+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 23 | 2026-08-03T20:59:08+00:00 | 2026-08-03T21:00:42+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 24 | 2026-08-03T21:10:56+00:00 | 2026-08-03T21:12:45+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 25 | 2026-08-03T21:03:34+00:00 | 2026-08-03T21:04:35+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 26 | 2026-08-03T21:06:24+00:00 | 2026-08-03T21:08:06+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, runningboardd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 27 | 2026-08-03T21:09:43+00:00 | 2026-08-03T21:10:56+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, runningboardd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 28 | 2026-08-03T21:14:31+00:00 | 2026-08-03T21:16:11+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by dasd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 29 | 2026-08-03T21:00:43+00:00 | 2026-08-03T21:02:09+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 30 | 2026-08-03T21:16:12+00:00 | 2026-08-03T21:17:32+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 31 | 2026-08-03T21:23:49+00:00 | 2026-08-03T21:26:49+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 32 | 2026-08-03T21:36:12+00:00 | 2026-08-03T21:39:41+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 33 | 2026-08-03T21:17:32+00:00 | 2026-08-03T21:20:47+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 34 | 2026-08-03T21:53:13+00:00 | 2026-08-03T21:55:58+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 35 | 2026-08-03T21:49:37+00:00 | 2026-08-03T21:53:12+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |
| 36 | 2026-08-03T21:12:46+00:00 | 2026-08-03T21:14:30+00:00 | Apple M4 Pro | macOS-26.5.2-arm64-arm-64bit-Mach-O | 3.14.6 | AppleClang 21.0.0.21000101 | -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic | start: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by powerd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1; end: ac; System-wide power settings: Currently in use: standby 1 Sleep On Power Button 1 hibernatefile /var/vm/sleepimage powernap 1 networkoversleep 0 disksleep 10 sleep 1 (sleep prevented by dasd, ChatGPT) hibernatemode 3 ttyskeepawake 1 displaysleep 10 tcpkeepalive 1 powermode 0 womp 1 | start: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded; end: Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |

| Artifact | SHA-256 |
| --- | --- |
| [benchmarks/flagship_study/V3_VALIDATION_FAILURE.md](V3_VALIDATION_FAILURE.md) | `e85407700a5280ab40974a53ba436bebedb399488a937f3f38f266c8e68f1cc1` |
| [benchmarks/flagship_study/analysis_contract.md](analysis_contract.md) | `e37f0706cc10d337401f8d459a2f47182681bffa046c80013d4fa1f3ab341115` |
| [benchmarks/flagship_study/charts/test_bradley_terry.svg](charts/test_bradley_terry.svg) | `4f0a701c2f980599618682fd436a441f21c2de86e3eb3faaf1f89cf780257e18` |
| [benchmarks/flagship_study/charts/test_calibration.svg](charts/test_calibration.svg) | `8db847675af20365ba7fdca6fd087064761d9f474df34ab5efba2413dc6f6097` |
| [benchmarks/flagship_study/charts/validation_pareto.svg](charts/validation_pareto.svg) | `a5c525f838b24f3f41582efbb94e75862b647c3007c0d0a14fdf068006eaa482` |
| [benchmarks/flagship_study/data/development.json](data/development.json) | `8fbe8db7d139790dec05f6475e5f668868050dd5d3dd752de57d9e01505aa44b` |
| [benchmarks/flagship_study/data/test.json](data/test.json) | `3c966d24ad594dc53172efcea48ba532451b9777c784a126ab95ae54d05430ba` |
| [benchmarks/flagship_study/data/validation.json](data/validation.json) | `12d16cf3f6a942de5779465aa3b8043cc102961a50b89e79bfeaca36e59b3de8` |
| [benchmarks/flagship_study/manifest.json](manifest.json) | `7ce58f3e22d7540a7e72eacb11369e337cb359141b6542a3b6da080e9f244b31` |
| [benchmarks/flagship_study/openings/development_d04.tsv](openings/development_d04.tsv) | `99f5ffe035aed851c7b43b9f52a65c554418d4da52f1d8e3ab62476d5c206f19` |
| [benchmarks/flagship_study/openings/development_d08.tsv](openings/development_d08.tsv) | `81031d20b3cb4ab87ab2c48550010b19c2ae9e4d2a1be39f75d4aaee9bff1bfb` |
| [benchmarks/flagship_study/openings/development_d12.tsv](openings/development_d12.tsv) | `d14078679891e1cc5d96fb13cad008de2bd65e654ed97f2d138eb99715259d5f` |
| [benchmarks/flagship_study/openings/development_d20.tsv](openings/development_d20.tsv) | `d1c05512c50f45819b6ffac72e8d2814a6b87839fd790a6944b4a2313556176d` |
| [benchmarks/flagship_study/openings/test_d04.tsv](openings/test_d04.tsv) | `b9282d3f61a54638ec6b991a2944b6ad6f2c815188845a8e6e5863d3b6fcefe6` |
| [benchmarks/flagship_study/openings/test_d08.tsv](openings/test_d08.tsv) | `70103a63fb15f30f57a140467a07af666b5f71928b8291deb5a1eb3c400549ad` |
| [benchmarks/flagship_study/openings/test_d12.tsv](openings/test_d12.tsv) | `b3315e6aa8ec64b52ff14bfacfc3fa81d1d94de896bceb8ec5471cd6542ec0bf` |
| [benchmarks/flagship_study/openings/test_d20.tsv](openings/test_d20.tsv) | `edb686a25491163cf7cd85325901a069877f36b9f4b4d03fdebd34ee9acf109c` |
| [benchmarks/flagship_study/openings/validation_v4_d04.tsv](openings/validation_v4_d04.tsv) | `9cb8786782fb0302b46b508bdb7532c0c9c09628dc1ca9aa99adf2665195614a` |
| [benchmarks/flagship_study/openings/validation_v4_d08.tsv](openings/validation_v4_d08.tsv) | `cf92970053478ab4156df4035dd94e40af90dbe1d7411bf529c9441c85a81249` |
| [benchmarks/flagship_study/openings/validation_v4_d12.tsv](openings/validation_v4_d12.tsv) | `d7307d9cf17af14e5619eba8a843cff101d928e6be83eec5c68ca49e8200b314` |
| [benchmarks/flagship_study/openings/validation_v4_d20.tsv](openings/validation_v4_d20.tsv) | `84ee7fdbc7c83b274623d293bc59e9135e457b9b2abd23cd8812753f3f90fc20` |
| [benchmarks/flagship_study/runtime_projection.json](runtime_projection.json) | `5cdc6d2850177dd6c8e836766a218a55cd240a3a8a69956ba262df360b1a80fc` |
| [benchmarks/flagship_study/selection_lock.json](selection_lock.json) | `19e71e876b808284ebd94f249873625c76fa1fd328fd6da2bd6de5b83db13cdb` |
| [benchmarks/flagship_study/superseded/manifest-b7553a24.json](superseded/manifest-b7553a24.json) | `b7553a24f618a77162cb256d65fd7cf21a854a10f46726c4a32cd57982ace9fe` |
| [models/jacek_article_value_model.json](../../models/jacek_article_value_model.json) | `57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084` |
| [submissions/codingame/bots/rank_5/submission.cpp](../../submissions/codingame/bots/rank_5/submission.cpp) | `f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29` |

Curated inputs:

- [Development data](data/development.json)
- [Validation data](data/validation.json)
- [Test data](data/test.json)
- [Selection lock](selection_lock.json)

## Integrity

Development, validation, and frozen test aggregation each report zero truncations and complete unique game sets. No truncated game entered paired strength, Bradley–Terry, or calibration calculations.
