# Flagship study v3: validation calibration failure

Status: **stopped before test** on 2026-08-03. The frozen test tournament was not run, no selection lock was created, and no test result namespace exists.

This is the durable failure record for manifest `competitive-demo-bots-flagship-2026-v3` (`b7553a24f618a77162cb256d65fd7cf21a854a10f46726c4a32cd57982ace9fe`). Development and validation completed at their preregistered sample sizes. Selection then failed while fitting the validation-only logistic mapping for `alpha-beta-50k`:

```text
validation calibration failed for alpha-beta-50k; stop before test: calibration coefficients diverged
```

The stop is binding for v3. Its calibration implementation is source-pinned to commit `ee9a1a886b60cf6672244e5f1ff84501be033129`; neither its criterion nor its result will be relaxed retrospectively.

## Completed evidence

| Phase | Units | Games | Decisions | Truncations | Curated SHA-256 |
|---|---:|---:|---:|---:|---|
| Development | 36/36 | 1,800/1,800 | 215,912 | 0 | `99cbc5d97346504152961b4e74b54cd0603f04bdcbc7044f520bb9a530fbec31` |
| Validation | 36/36 | 3,600/3,600 | 434,381 | 0 | `c3bb291d184cc77742bb388c07d7d47f217747242129608088cd2bc78fa6f965` |

The 18-unit runtime projection observed 900 games over 1,255.352 seconds. It projected 3.008-9.118 hours for the full 4,800-game test tournament and 4.398-11.215 hours for all remaining work at projection time. Projection SHA-256: `77751b9d798d52442347501eb96ac228481990cb33c9679e10d614c1f6484e87`.

All validation units passed the before/after AC-power, Low Power Mode, and nominal thermal/performance checks. Raw shards reaggregate byte-for-byte to the curated artifacts. The arena SHA-256 is `124bc600e408ce1565e67ce4789c7d0d623f595a1e2a1640b46d00dd71680ff2`.

## Validation gate and rule output

Strength is the mean score over 200 color-swapped pairs against **Rank5DerivedBot — fixed 50k demo profile**; intervals are depth-stratified, pair-clustered 95% percentile intervals with 10,000 resamples. A p95 above 50 ms is ineligible.

| Family | Configuration | Pair score (95% interval) | Validation p95 | Eligible | Rule output before calibration |
|---|---|---:|---:|:---:|:---:|
| Tactical MCTS | `mcts-1000` | 0.1450 [0.1125, 0.1800] | 36.043792 ms | yes | nominated |
| Tactical MCTS | `mcts-2000` | 0.2575 [0.2175, 0.2975] | 69.882625 ms | no | — |
| Tactical MCTS | `mcts-4000` | 0.3300 [0.2900, 0.3725] | 137.860125 ms | no | — |
| Hand alpha-beta | `alpha-beta-20k` | 0.3850 [0.3425, 0.4275] | 13.546541 ms | yes | — |
| Hand alpha-beta | `alpha-beta-50k` | 0.4250 [0.3825, 0.4700] | 23.165125 ms | yes | nominated |
| Hand alpha-beta | `alpha-beta-100k` | 0.4325 [0.3875, 0.4750] | 25.082250 ms | yes | practical tie |
| Neural alpha-beta | `jacek-20k` | 0.5200 [0.4750, 0.5650] | 35.606042 ms | yes | nominated |
| Neural alpha-beta | `jacek-50k` | 0.5325 [0.4875, 0.5775] | 58.281292 ms | no | — |
| Neural alpha-beta | `jacek-100k` | 0.5200 [0.4750, 0.5650] | 58.682167 ms | no | — |

These are deterministic outputs of the validation rule, not locked test entrants. The 50k hand configuration is within 0.75 percentage points of the strongest eligible 100k result and wins the preregistered lower-p95 tie-break. Because calibration failed, the nominations were never serialized as a selection lock.

The fixed Rank5Derived comparator passed its native gate: fresh-root p95 was 31.387375 ms over 61,510 searches; all-edge p95 was 27.295083 ms over 214,077 returned edges. It performed 61,510 fresh-root searches and 152,567 cached continuations. The constrained validation frontier contained `alpha-beta-20k`, `alpha-beta-50k`, `alpha-beta-100k`, `jacek-20k`, and `rank5-fixed-50k`. The unconstrained frontier additionally contained `jacek-50k`.

Rank5DerivedBot adapts search code from the rank 5/206 CodinGame submission to different demo rules and a fixed-work profile. These measurements do not evaluate the authentic ranked submission.

## Exact calibration blocker

The selected hand-alpha-beta payload is internally complete:

- 29,106 decision opportunities;
- 29,104 retained predictions and two preregistered depth-zero exclusions;
- 17,208 loss outcomes and 11,896 win outcomes;
- 200 complete color-swapped-pair clusters, with 50 in each opening-depth stratum;
- zero cached-continuation exclusions and zero truncation exclusions.

Raw validation shards reproduce every retained `(oriented score, outcome, pair, stratum)` column exactly. Outcome classes overlap: loss scores extend to `+14237`, while win scores extend to `-14131`. This is not complete or quasi-complete separation.

The defect is a parameterization-dependent implementation guard. Of the retained predictions, 2,611 have mate-scale magnitude at least 900,000 and all 2,611 agree with the eventual outcome. Those valid values inflate the population score standard deviation to 298,235.072. The accepted Newton update at iteration 8 improves log likelihood from -16,960.111 to -16,799.416 but moves the standardized slope from 36.510 to 58.949. The frozen implementation labels any standardized coefficient with magnitude above 50 as divergent and stops.

As a read-only diagnosis, continuing the identical finite Newton path without that cap converges at iteration 12: standardized intercept -7.225312, standardized slope 72.251098, raw-score intercept -0.377984, raw-score slope 0.000242262, maximum gradient `2.8e-11`, and positive information determinant 1,164.612. This diagnostic fit is **not** a v3 calibration lock and cannot authorize test access.

All six hand and neural alpha-beta candidate fits encounter the same scale-sensitive guard. All three MCTS fits converge, and the fixed Rank5 fit converges. Filtering, clipping, or transforming mate scores after seeing validation would change the frozen v3 analysis and was not done.

## Negative findings available before test

- MCTS gained paired strength at 2,000 and 4,000 iterations, but both larger configurations missed the 50 ms p95 gate.
- Hand alpha-beta budget increases were statistically unresolved at the preregistered 1 percentage-point practical threshold.
- Neural alpha-beta budget increases were also unresolved; its 50k and 100k candidates missed the latency gate.
- On the common validation openings, the neural evaluator was materially stronger than the hand evaluator at each equal node budget under the preregistered paired-difference rule.
- No test pairwise interval, Bradley-Terry estimate, test calibration metric, or test chart exists for v3.

## Integrity and reproduction

- Manifest: [`superseded/manifest-b7553a24.json`](superseded/manifest-b7553a24.json)
- Development data: [`superseded/v3-data/development.json`](superseded/v3-data/development.json)
- Validation data: [`superseded/v3-data/validation.json`](superseded/v3-data/validation.json)
- Runtime projection: [`superseded/v3-data/runtime_projection.json`](superseded/v3-data/runtime_projection.json)
- Analysis contract SHA-256: `a5f712bf353cf1109757f4ff1ff218c585d832147295fa50afad13b43b741272`
- Jacek model SHA-256: `57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084`
- Authentic rank-5 source SHA-256: `f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`

Reproduce the fail-closed decision from the committed v3 checkout and raw result namespace:

```sh
git switch --detach 77c8fdc2e417583a83c3004e2e15f18fbe4abaca
python3 benchmarks/flagship_study/run_study.py aggregate --phase development
python3 benchmarks/flagship_study/run_study.py aggregate --phase validation
python3 benchmarks/flagship_study/run_study.py lock-selection
```

The final command must exit nonzero with the calibration error above. It must not create `selection_lock.json` or authorize the test phase.
