# Large-teacher self-search campaign: compact terminal evidence

## Result

The provenance-bound 10,000-game self-search campaign accepted one **local
teacher candidate**. The selected runtime identity is
`f7bdb201a377c04531f1ba98fd73457f7f77961aa0f0f9b1ac32c59b6e85ee75`.
All four frozen 500-pair strength panels passed, every game was legal and
finished, the uncontended maximum decision time was below 1,000 ms, and both
canonical retention comparisons passed.

This is not a canonical promotion, deployment, Rank-4 replacement, external
leaderboard result, or claim about an unpublished contest bot. Student
training is eligible but had not started when the terminal receipt was written.

## Publication boundary

The source campaign itself intentionally ended with publication and external
upload disabled. This directory is a separately authorized, post-campaign
derivation for public review. It does not alter the source receipts and contains
no runtime weights, training data, labels, NPZs, TSVs, opening transcripts,
game transcripts, or raw shard reports.

The compact evidence supports independent recomputation of the acceptance
arithmetic. It does not make the ignored 3.6 GiB campaign reproducible without
its retained local inputs. Exact hashes of the omitted source artifacts remain
in [`manifest.json`](manifest.json).

## Frozen contract

- Campaign implementation commit:
  `e91e09f5a6d5f1278f7e4a919a8819e1ceeab2cb`
- Exact base commit:
  `c2b06168676bfa0ea7e600da042710b29f3089c5`
- Final panels: 500 color-swapped pairs each, 980 ms per decision.
- Matched and pilot-teacher floors: 527/1,000 overall and 260 per color.
- Rank-4 and Jacek-NN floors: 501/1,000 overall and 238 per color.
- Safety: zero illegal and zero unfinished games.
- Authoritative latency: one uncontended worker, maximum strictly below
  1,000 ms.
- Retention: sign accuracy no worse than 0.005 below each reference and
  weighted Huber no greater than 1.02 times each reference.

The original 20 ms exploratory pilot did **not** pass. Its explicit
teacher-only continuation override recorded only the matched and incumbent
primary-strength failures; it did not bypass legality, completion, external
strength, retention, p99, or uncontended-latency requirements.

## Final strength gates

| Panel | Wins | Color 0 | Color 1 | Frozen floor | Illegal | Unfinished | Result |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Matched search-vs-Rank-4-trained arm | 531/1,000 | 270/500 | 261/500 | 527; 260/color | 0 | 0 | pass |
| Pilot teacher | 536/1,000 | 262/500 | 274/500 | 527; 260/color | 0 | 0 | pass |
| Rank-4 | 651/1,000 | 336/500 | 315/500 | 501; 238/color | 0 | 0 | pass |
| Jacek NN | 668/1,000 | 344/500 | 324/500 | 501; 238/color | 0 | 0 | pass |

[`paired_outcomes.json`](paired_outcomes.json) retains exactly 500 opening-state
identities and two compact color-swapped outcomes per panel. It omits every
transcript and all contended per-decision telemetry. `publish.py verify`
recomputes the totals, color splits, legality, completion, and thresholds from
those 4,000 compact game records.

## Retention

The candidate metrics are identical across the two reference comparisons
because both evaluate the same 121,052 canonical test samples.

| Reference | Candidate sign | Reference sign | Candidate weighted Huber | Reference weighted Huber | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Pilot teacher | 0.8676271354 | 0.8670571325 | 0.0487084053 | 0.0488501489 | pass |
| Original v6 reference | 0.8676271354 | 0.8671066979 | 0.0487084053 | 0.0487790965 | pass |

The original v6 retention-reference runtime identity is
`bfcc1755ab9b71261bedc9b9c9b59e38e3d440d7c80e7056a9f0bc812ffc9c80`.

## Uncontended latency

Stage 20 evaluated 20 games on one worker and recorded 334 candidate decision
samples. Candidate p99 was `972.562459` ms and the maximum was
`981.945875` ms, clearing the strict 1,000 ms maximum. All candidate samples
are retained in [`compact_results.json`](compact_results.json); the verifier
recomputes their count, p99, and maximum.

## Environment transition and homogeneous final gate

Stages 00–18 were completed on
`macOS-26.5.2-arm64-arm-64bit-Mach-O`. After a host update, the frozen launch
correctly rejected `macOS-26.6.2-arm64-arm-64bit-Mach-O`. At that point 151 of
400 final-gate shard reports and receipts had completed, leaving stage 19
incomplete. Those pre-update shards were archived as diagnostic-only evidence
and were not reused. All 400 authoritative stage-19 shards were rerun from
offset zero on 26.6.2.

The first transition runner then failed before creating any replacement shard
because its receipt-15 validator dynamically imported `jacek_replay_train`
after the tools path had been removed. A chained, hash-bound runner repair kept
the exact tools directory importable for the full invocation. The repaired
run started with zero replacement outputs, completed all final gates, and
preserved both failure chains. Their exact amendment hashes are recorded in
the compact results and manifest.

## Bundle files

- [`manifest.json`](manifest.json): implementation/base identities, source
  closure, omitted-source hashes, generated-file hashes, and publication scope.
- [`compact_results.json`](compact_results.json): terminal decision, panel
  summaries, all authoritative latency samples, retention metrics, policy, and
  normalized transition chain.
- [`paired_outcomes.json`](paired_outcomes.json): compact paired outcomes used
  to recompute every strength and safety count.
- [`publish.py`](publish.py): deterministic freezer and self-contained verifier.

Verify the checked-in bundle without local campaign data:

```bash
python3 benchmarks/large_teacher_campaign/publish.py verify
```

When the ignored terminal campaign is available, also re-derive every compact
record and verify all source hashes:

```bash
python3 benchmarks/large_teacher_campaign/publish.py verify \
  --campaign-root results/large_teacher/large-teacher-campaign-20260828-v1
```
