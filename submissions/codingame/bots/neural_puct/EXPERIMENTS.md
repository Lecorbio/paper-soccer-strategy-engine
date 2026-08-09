# Neural PUCT experiment ledger

This ledger records promotion decisions for the NN-first Paper Soccer line.
All gameplay screens below used paired physical colors, completed legally, and
used only exposed development openings. Validation and final promotion banks
were not consumed. A candidate advances only when it clears the gate declared
before its outcome is known.

## Maintained baseline

The maintained model is the seed-20260809 public-expert fit with the full
1,286-input schema and a 32-by-32 policy/value network. The production search
is atomic-edge PUCT with exact terminal rules and immediate wins, but no replay
book, handcrafted evaluator, alpha-beta fallback, or rank-5 decision override.

| Screen | Neural PUCT | Rank 5 | Decision |
|---|---:|---:|---|
| Six paired defaults, 2,000 simulations vs 5,000 nodes | 7 | 5 | bootstrap pass |
| 48 exposed openings in both colors, same fixed budgets | 42 | 54 | baseline only |
| Same exposed set, 20 ms construction-inclusive per decision | 39 | 57 | not promotable |

Sparse first-layer evaluation is behaviorally identical to the dense path on
4,096 reachable states and 274 fixed-search cases. It improved measured
inference throughput by about 19%, but additional simulations alone did not
close the match gap.

## Article-scale self-play

The largest iteration generated 4,096 neural-only games at 5,000 simulations.
Strict replay and trajectory grouping retained 3,104 unique games and roughly
343,000 canonical teacher states. Random opening prefixes were excluded from
both value and policy training. Whole duplicate trajectories were grouped
before splitting, and held-out canonical overlap was purged in train,
validation, then test order.

The hard visit-max policy head was selected by frozen quantized validation
rules. It improved the small fixed-work gate, but failed to transfer when the
benchmark received more search:

| Screen | Candidate | Rank 5 | Decision |
|---|---:|---:|---|
| Six paired defaults, 2,000 vs 5,000 | 9 | 3 | pass |
| Full exposed development, 2,000 vs 5,000 | 46 | 50 | diagnostic only |
| Full exposed development, equal 20 ms | 22 | 74 | reject |

The result isolates the main limitation: a better shallow policy cannot
compensate for a weak value representation as the opponent's alpha-beta search
scales.

## Value targets

Final-game outcome fine-tuning changed only the value head. The best source
balance improved held-out neural value loss from 0.653 to 0.621 while retaining
elite and Jacek floors. It passed defaults 8-4, then lost exposed development
38-58 and was rejected before equal-clock evaluation.

A second corpus recorded visit-weighted mover-relative values from searched
PUCT nodes. The audited corpus contained 512 games, 421 unique trajectories,
and 48,580 retained tree-backed primitive targets. Exact solved wins,
ordinary searched nodes, and unavailable fallbacks were represented
separately. The maintained value head already predicted its own held-out
5,000-simulation targets better than either distilled head:

| Model | Held-out search-value BCE | Decision |
|---|---:|---|
| Maintained baseline | 0.590915 | reference |
| Distilled source mass 1 | 0.618040 | reject |
| Distilled source mass 4 | 0.598314 | reject |

No gameplay gate was run for an ineligible search-value model.

## Rejected architecture and search families

- Article-exact 1,156 inputs: 4-8 on defaults despite much faster inference.
- Full inputs with a 28-by-64-by-32 factorized network: 6-6 on defaults.
- Lean layer-aware and joint degree-distance schemas: 4-8 and 3-9.
- Winner-only expert policy and soft root-visit training: 6-6 and 2-10.
- Uniform priors, parent-mean FPU, policy temperature, and dynamic exploration:
  none cleared the maintained 7-5 default gate.
- Exact solved-state PUCT: 6-6; correct but not stronger.
- Pure neural alpha-beta using the NN only for leaf value and ordering: 2-10
  at equal 20 ms.
- Cross-turn tree reuse increased work only slightly and regressed the fixed
  development result.

## Current conclusion

The implementation and data pipeline are sound, but the 32-dimensional hidden
representation does not yet supply a value signal strong enough to beat the
current rank-5 benchmark at equal time. Offline classification improvements
are not accepted as strength evidence. The next iteration should add genuinely
stronger value supervision or a representation that can exploit it, and must
repeat the fixed-work and equal-clock gates before any CodinGame submission.
