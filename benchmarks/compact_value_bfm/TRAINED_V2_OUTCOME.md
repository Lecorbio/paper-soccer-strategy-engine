# Mandatory-training Compact Value-BFM: closed without a qualifying candidate

The bounded training-v2 experiment stopped on 2026-09-09. Its goal was a newly
trained bias-free `6301 → 12 → 8 → 1` signed-three-bit model passing the original
qualification gates, followed by 90 clean live games and a calibrated score
strictly above `44.29750553418035`. That goal was not achieved.

Four production pilots produced **0/36 retention passes**. Six of eight short
research jobs completed, including the final frozen-warmup consistency pair.
The two remaining slots were conditional replications; their trigger failed.
All 704 exploratory games were already complete. No full pilot 5 or further
game allocation followed the stopping decision.

## Final paired experiment

Both arms used seed `20260907`, original initialization, one genuine float
warmup, four QAT epochs, and the same corpus. During QAT they added coefficient
1.0 weighted Huber (delta 0.25) to their own frozen warmup predictions on the
existing TRAIN scalar rows. Ranking weights were 0 and 0.10. Both warmups and
initial quantizations matched their exact no-consistency controls before the
first new QAT update. Both selected QAT epoch 3.

| Selected quantized result | Fresh scalar control | Ranking model |
| --- | ---: | ---: |
| Canonical sign accuracy | 85.9923% | 86.0087% |
| Canonical weighted Huber | 0.0529992821 | 0.0520013062 |
| Common sign accuracy | 86.0875% | 86.1875% |
| Common weighted Huber | 0.0509983453 | 0.0510715552 |
| Mean teacher regret | 0.2114002331 | 0.2188445540 |
| Complete retention | Fail | Fail |

Ranking regret improved **0.98%** against its matched prior student-rivals
control, but was **3.52% worse** than the fresh scalar control. The plan
required at least a 10% improvement over that fresh control, as well as complete
retention and improvement over the old ranking control. The joint criterion failed.

The ranking model passed common retention but failed three canonical checks:
sign accuracy below 86.13%, float-to-quantized sign loss of 0.5954 percentage
points rather than strictly below 0.5, and a Huber ratio of 1.0411 rather than
at most 1.02. The scalar model also failed common relative Huber. None of the
initial or four epoch states passed complete retention. Early-stratum evidence
was unavailable; no formal pilot admission was claimed.

## Execution and interpretation

Each arm executed all 104 QAT updates and 104 nonzero consistency-gradient
batches. Native scalar checks and 4,096 rows in two mover-feature orientations
passed. Both exported sources contained 92,165 ASCII characters with 2,835
characters of reserve. All five CI jobs and 580 local tests had passed on
source commit `8b3640ed5486256f63767b0cc919721a538e00b6` before execution.

The two training workers and four validation helpers completed the pair in
851.653 seconds, approximately 14 minutes 12 seconds. Peak sampled aggregate
RSS was 4.302 GiB under the 8 GiB stop. All owned processes exited and the
training lock was released. The hourly monitor was retired.

The result closes this learning branch. It does not prove that a stronger
three-bit network is impossible. Position accuracy is not game strength, and
execution digests do not establish independent gradient replay or per-term
parameter attribution.

[Sanitized outcome data](trained-v2-outcome.json) contains exact saved metrics,
gate errors, and source artifact hashes. Raw receipts, weights, data, and games
remain in the retained local campaign namespace
`results/compact_value_bfm/compact-value-bfm-trained-v2/`. This compact record
supports checking the reported arithmetic; it cannot reproduce training
without those local inputs. The [implementation contract](../../docs/compact-value-bfm-trained-v2.md)
documents the original execution and qualification requirements.
