# Successor-boundary ranker experiment

Status: completed and rejected; not part of the generated CodinGame source.

The first 80-root corpus (`7b7d29d...34e0`) is an integrity-valid but discarded
calibration-only artifact. It is preserved only for provenance: no V1 row was
retained for fitting, model selection, or promotion, and it is not training
evidence. Because its procedural depth is counted in complete turns and all
four depths are even, all roots have player one to move and all successor rows
have player two to move. Horizontal reflection does not change that. A
qualifying corpus must additionally generate 180-degree player-swapped variants
and run their 30k/100k teacher searches independently; labels must not be copied
by symmetry. V2 does exactly that and the strict loader verifies four variants
and exact mover balance in every split/depth cell before fitting.

The hypothesis is deliberately narrow: a compact model may improve the order
in which the root tactical scout examines complete-turn endpoints. It must not
prune a legal first edge, replace exact goal/block/own-goal classification, or
change any non-root alpha-beta leaf value.

## Fresh data and teacher contract

`tools/rank4_jacek_hybrid_ranker_samples.cpp` creates boundary roots directly
from the public 8x10 Codingame rules using a fixed SplitMix64 seed and complete
random turns at depths 4, 8, 12, and 20. It reads no opening bank, replay,
arena payload, model corpus, validation bank, or final bank. Horizontal mirror
variants remain in the same whole-root split.

For every nonterminal, deduplicated complete-turn successor, frozen Rank-4
(proof off) supplies mover-relative counterfactual values at independently
constructed 30,000- and 100,000-node searches. Terminal successors are
excluded because the tactical scout must classify them exactly. Pair labels
are retained only when both searches agree in sign, both complete at least
depth 3, and both margins are at least 10,000 on Rank-4's 100,000-point
non-mate scale (0.10 normalized).

The corpus's first JSONL row freezes seed, rules, teacher, work budgets, root
count, action cap, and symmetry mode. The model report additionally binds the
SHA-256 identities of the corpus, generator, trainer, frozen Rank-4 generated
source, and incumbent residual model.

The V2 receipt has two byte-identical names with distinct lookup purposes:

- `manifests/3e4db036abfec4ca4e03f61622790378c75b2d61808fd1366001fc0aba8febb9.json`
  is keyed by the V2 corpus SHA-256 named inside the receipt;
- `manifests/5c2dc3cdea9875ae496ad0aa32e60e445fe01cfe61c27baea59db958f4c13f5c.json`
  is keyed by the receipt's own SHA-256.

Both files have SHA-256
`5c2dc3cdea9875ae496ad0aa32e60e445fe01cfe61c27baea59db958f4c13f5c`.
The corpus-keyed path remains the original receipt; the self-digest path is a
content-addressed alias, not a second generation or a second corpus.

## Compact model

The model is random-initialized pairwise logistic regression over the existing
24 mover-relative evaluation snapshot signals: the replay network's eight
second-layer activations plus its probability, hand/anchor values, geometry,
goal distances, mobility, tactical flags, and phase. It predicts successor
mover strength; lower is better for the player that just completed the turn.

Training uses only the train roots. Ridge and random seed are selected by
root-balanced validation pair accuracy. Test roots are reported once after
selection. The trainer also reports incumbent anchor/residual baselines,
whole-group top-one regret, horizontal-mirror selection agreement, and the
loss from signed-int8 quantization.

The generated model header contains only 24 signed bytes. The intended hook
reuses the scout endpoint's already-computed `EvaluationSnapshot`, keeps the
baseline scalar for `record_completed_action`, and uses the learned dot
product solely for `scout_first_scores_`. A second snapshot/network inference
per endpoint is disallowed because it would consume the scout's 12 ms budget.

The fitted header is 219 bytes. Against the selected 94,004-byte proof-only
source, the projected flattened source is 94,673-94,873 bytes after a lean
ordering hook. Size is therefore not the rejection reason.

## V2 result and decision

The color-balanced corpus contains 3,572 candidate rows from 80 root families
and 320 variant groups. Each variant contributes exactly 893 rows and each
root/successor mover sign contributes exactly 1,786. The corpus SHA-256 is
`3e4db036abfec4ca4e03f61622790378c75b2d61808fd1366001fc0aba8febb9`.

After requiring depth at least 3, ordering agreement at 30k and 100k work,
and a margin of at least 0.10 at both work levels, the train/validation/test
splits retain 458/58/49 pairs. Root-balanced int8 pair accuracy is
0.8133/0.9040/1.0000. The incumbent residual scores
0.7516/0.9008/0.8271 and the unmodified anchor scores
0.7415/0.9098/0.8033. The ranker's validation improvement over the incumbent
residual is only 0.0033 and it is 0.0058 worse than the simpler anchor.
Furthermore, the test stable pairs contain no depth-8 or depth-20 witness.

Whole-group selection is not persuasive: validation exact-best rate is
0.2708 with 1.7217 mean normalized regret, while test is 0.2292 with 2.8054
regret. Signed-int8 quantization loses no validation accuracy. Color-rotation
score deltas are near floating noise, but horizontal-truncation panels remain
asymmetric. Full metrics and all six seed/ridge trials are frozen in
`successor_ranker_v2.json`.

The tactical scout was independently rejected in whole-game development
evidence (proof-only 181-125 versus proof-plus-scout 158-148 at the fixed-work
gate). A marginal pairwise validation result cannot justify restoring that
scout. The trainer therefore records `recommended_for_timed_ablation=false`,
and no model or hook is integrated into production.

## Qualification before any integration

The ranker is only worth a production ablation if all of the following hold:

- int8 validation and untouched-test root-balanced pair accuracy beat both
  the anchor and incumbent residual ordering;
- the gain is not confined to one depth, mover, or symmetry variant;
- quantization loses at most 0.5 percentage point;
- mirrored root selection agreement does not regress materially;
- isolated scout timing/coverage stays inside the existing cap; and
- whole-game development evidence improves independently before any frozen
  validation or final bank is opened.

Until those gates pass, the correct operational result is to leave this model
out of `rank_4_jacek_hybrid`.
