# Compact Value-BFM Rank-4 campaign outcome

This document is the sanitized public record of the completed
`compact-value-bfm-rank4-teacher-challenger-v1` campaign. It intentionally
excludes protected opening banks, raw gate shards, live replay payloads, and
machine-local paths.

## Qualified attempt

- Campaign attempt: `0`
- Release commit: `ae03687fdd5083f5453db6cc01389ef6604f145a`
- Architecture: bias-free scalar `6301 -> 12 -> 8 -> 1`
- Quantized payload SHA-256:
  `76739bbf764741a8722d873cef3b1efbac22b6f759106b6db00b2248fd4e4a00`
- Runtime-body SHA-256:
  `b34339a5cf5f6b1b037c28cacf1505618df3cfb54ec5748b29bb8f9e0a5bee99`
- Runtime-file SHA-256:
  `130c6ef1d2311a76c7a94fd144a805aa22477a32bced59a8079021e4293ea336`
- Generated finalist source: 94,834 ASCII bytes, SHA-256
  `f5e67d699be19c3d495673c04ee2453570391c59e5f7be2a779198ce98b2d621`
- Exact deployed source: 94,817 ASCII bytes, SHA-256
  `add71c369052f232209d69c3b40b6bb459a2d7326ef15c5980377b1526fb8ea9`

The exact deployed source remains tracked as
`submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.cpp`.
The maintained `submission.cpp` may be regenerated as implementation and
instrumentation evolve; it is not a replacement for that immutable release
artifact.

## Controlled qualification

Both banks were frozen independently and played with actual clocks against the
maintained historical Rank-4 source.

| Gate | Games | Wins | Player 0 wins | Player 1 wins | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1,000 | 622 | 310 | 312 | 0 |
| B | 1,000 | 597 | 301 | 296 | 0 |

- Gate A result SHA-256:
  `bc425c65ba9c632729805e046bb05b2f9b1ea034fe83a415127fb274a387be12`
- Gate B result SHA-256:
  `1ec80ab55b3eff6bc1ae08b9e7122a6d3faef157f03cbbddfb58b055abfacd21`
- Dual-qualification body SHA-256:
  `58c5f7c118ec4bbf20df46258a9cf2ca746b0572cfcb8d60b48e272ebeaed8f2`

These results establish performance against that exact historical opponent and
bank distribution. They do not establish equivalence to a current public
ladder rank.

## CI and live diagnostic

- GitHub Actions run: `33869184263`, successful for release commit `ae03687...`
- CodinGame agent: `6702073`
- CodinGame submission: `41243299`
- Submit clicks recorded by the campaign: `1`
- Live window: exactly 90 games, accepted diagnostic, with zero
  bot-attributable operational failures
- Corrected live-reference body SHA-256:
  `5c4a668169e502aeb6918cc6d6bb23a944b1e42e0c71c4248903e8428aca9884`
- Campaign-complete event body SHA-256:
  `7dc5de36fb7e33898c3779850ca0f7e0c59ff335126460d93adb4fa57967c4f0`

The live window is diagnostic-only and remains ineligible for training.

## Interpretation and continuation

Attempt zero qualified without running the proposed new large-teacher
distillation round: it generated no new campaign training games or teacher
labels and did not update the three student layers. The result therefore
qualifies the recovered incumbent; it should not be described as the output of
the planned ranking-loss distillation experiment.

The next campaign should separately test empty-board and early-ply positions,
multi-opponent generalization, teacher action regret, quantization action flips,
and actual-clock search throughput before opening any new protected final bank.

The completed campaign's one-upload evidence remains immutable. For subsequent
development, the user has authorized diagnostic CodinGame uploads without a
separate per-upload confirmation when they can produce useful new evidence.
Each upload should still be source-bound, recorded, operationally checked, and
kept separate from protected-data qualification.
