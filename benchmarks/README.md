# Benchmarks and results

These collections measure different parts of the project: bot strength,
position evaluation, complete-turn review, and model experiments. Each report
states its rules, comparison method, and the limits of its conclusions.

| Collection | What was measured | Read first |
| --- | --- | --- |
| CodinGame-style leaderboard | 22 standalone bots across 990 protocol-faithful games | [Leaderboard](codingame_leaderboard/README.md) |
| Flagship study | Four selected bots across 4,800 games with color-swapped openings | [Study report](flagship_study/REPORT.md) |
| Game Review | Complete-turn analysis, exact endgame checks, and browser integration | [Acceptance report](game_review_gate/REPORT.md) |
| Large teacher | Local strength, response time, and retained model quality | [Teacher report](large_teacher_campaign/REPORT.md) |
| Compact value models | Teacher distillation and a completed training experiment | [Training outcome](compact_value_bfm/TRAINED_V2_OUTCOME.md) |

The [featured standalone bot](../submissions/codingame/bots/quantized_alpha_beta/README.md)
records the best observed public result, rank 2 of 212 on 9 October 2026.
That dated platform result is separate from these local measurements.

The [experiment catalog](../docs/campaigns.md) also covers studies that did not
produce a stronger bot. Their negative results and incomplete stages remain
documented. Frozen reports and machine-readable outputs are retained without
editing their results for presentation.

See [Reproducibility](../docs/reproducibility.md) for artifact checks. Re-running a
full tournament is a separate operation from building or testing the engine.
