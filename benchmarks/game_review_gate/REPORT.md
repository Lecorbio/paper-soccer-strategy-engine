# DeepTurnSearch strength and calibration gate

This report is generated only from the frozen Game Review gate artifacts. 
It is separate from the unchanged flagship study and does not evaluate the 
authentic ranked `rank_5` submission.

## Locked profile

Selected review profile: `deep-turn-search-400k`.

The profile was selected on validation strength within one percentage point 
of the eligible leader, then by lower WebAssembly p95 and lower fixed work. 
The selected Deep mapping was fit only on fresh validation decisions from 
that Deep profile. The separate Fast mapping was fit only on fresh validation 
decisions from the fixed Rank5Derived reference, whose immutable 50k search 
settings match Fast analysis. Their profile IDs and hashes remain distinct.

## Frozen test

| Reference | Pairs | Games | Score | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| `rank5-derived-fixed-50k` | 400 | 800 | 67.25% | 64.12%–70.25% |
| `jacek-inspired-20k` | 400 | 800 | 61.38% | 58.25%–64.38% |

All intervals use 10,000 opening-depth-stratified whole-pair bootstrap 
resamples. The test contains exactly 1,600 decisive games.

## Expert decision

The gate passed against both references. The playable selector may show **Expert — DeepTurnSearch**.

The decision requires zero illegal moves, incomplete actions, unexplained 
truncations, and parity failures, plus a paired 95% lower bound strictly above 
50% against each reference. No overall accuracy number is inferred.
