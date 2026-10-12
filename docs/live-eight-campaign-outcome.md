# Eight-unit model study

The experiment closed on **12 October 2026** without replacing the baseline bot.
No candidate passed the final selection requirements. The original bot was
restored for the later ranking submissions: a bias-free 6,301 → 12 → 8 → 1
network, four-bit weights and 550/140 ms search budgets. Its source bytes were
unchanged throughout that block.

This closes the [research protocol](live-eight-campaign.md) before its original
19 October finalist cutoff and 21 October deadline. Neither unused live slots
nor example commands authorize further experiments or submissions.

## Live exploration

Seven experimental versions completed 3,600 declared comparison games in total,
including the original-incumbent arms. Six versions used 8/8 networks; one
retimed the exact incumbent tensors. All seven matched point estimates were
negative. No own-bot operational failures were recorded in these windows.

| Exact experimental version | Search budgets, first/later ms | Declared comparison games | Matched difference against original incumbent, percentage points | Usable 95% interval, percentage points |
| --- | --- | ---: | ---: | --- |
| Initial 8/8, four-bit | 550/140 | 540 | −7.78 | Unavailable |
| Initial live-data precision continuation, six-bit | 990/180 | 540 | −10.22 | Unavailable |
| Exact-tensor 12/8 incumbent timing control | 990/180 | 540 | −8.90 | [−25.82, +1.25] |
| Continuation02, six-bit | 990/180 | 360 | −11.88 | [−16.76, −9.58] |
| Continuation03, six-bit | 990/180 | 540 | −10.38 | Unavailable |
| Exact continuation03 tensors at original clocks | 550/140 | 540 | −13.08 | Unavailable |
| Continuation04, four-bit | 990/180 | 540 | −18.96 | Unavailable |

Each estimate matches actual opponent submission versions and player colors,
then gives equal weight to the six frozen primary opponents and both colors.
The bootstrap resamples paired complete calibration windows. These games are
not independent per-game statistical observations, and the seven experiments
must not be pooled into a single independent sample or interpreted as a causal
training or timing trend.

Five assessments had unidentified opponent/color cells in some of their 10,000
whole-window bootstrap draws. Their confidence intervals are unavailable;
unsupported draws were not dropped to manufacture an interval. The negative
point estimates therefore do not supply seven statistically significant
regression claims. Rating scores and ranks remain separate descriptive results.

## Training and validation closure

Four-bit and six-bit QAT shared a selected common float parent, preserving useful
floating parameters and optimizer moments. Continuation04 selected four-bit
weights by the frozen native validation rule and passed the declared clock
ladder's source-bound mechanical and isolated whole-response checks. Those
checks establish correctness and operational evidence, not playing strength.

Continuation05 retained the selected continuation04 float checkpoint and
generated its frozen 128-root validation bank. Its teacher evaluations completed,
but one parent had a TRAIN/reflected-feature overlap. Although all 128 roots
still had represented comparable early positions, the strict zero-exclusion
corpus rule blocked admission. No quarantine exception, replacement bank or new
continuation05 float/QAT update was authorized.

All checkpoints, optimizer states, update journals, diagnostics, failed or
unsupported work and source/clock receipts remain in the private archive.
Validation, protected, private, formal and confirmation evidence remains
excluded from training. The campaign ended without an independent finalist
qualification, private panel or new formal promotion block.

## Original-bot ranking retries after research stopped

The owner separately authorized at most ten unchanged-incumbent submissions,
stopping at the first completed rank of two or better or after ten submissions.
All ten calibrations completed and their 900 declared games were archived.

| Submission | Completed rank | Rating score |
| ---: | ---: | ---: |
| 1 | 4 | 37.18 |
| 2 | 5 | 36.50 |
| 3 | 5 | 36.65 |
| 4 | 6 | 35.68 |
| 5 | 7 | 35.30 |
| 6 | 6 | 35.82 |
| 7 | 5 | 36.09 |
| 8 | 8 | 35.45 |
| 9 | 5 | 36.59 |
| 10 | 10 | 34.75 |

Rank two was not reached. The best completed rank was four; the final completed
rank was ten of 212 players. All ten retries had zero own-bot operational
failures. Source bytes were unchanged throughout and independently checked
against editor copybacks. These ranking-only retries are exclusion evidence,
not training data, a new qualification block or evidence of a strength increase.

The [9 October rank-2 observation](../submissions/codingame/bots/quantized_alpha_beta/README.md)
remains an immutable, dated historical observation. Its source hash is unchanged:

```text
78e731a81ce7cc702a200d0832a6c75e967a11b12452051724ab1387c226e4b8
```

That earlier snapshot does not describe the final 12 October ranking. The
campaign's recurring recovery task was retired. Private game transcripts,
checkpoints, source-bound receipts, host control state and private communications
remain local; this report publishes only the aggregate outcome and protocol.
