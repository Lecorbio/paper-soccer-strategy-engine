# Rank-4/Jacek hybrid

`rank_4_jacek_hybrid` is an isolated derivative of the canonical `rank_4`
engine. Its frozen scaffold started with a byte-identical engine, replay,
replay-value, and teacher-residual inputs. It does not modify `../rank_4` and
must not be described as a clean-room or fresh-lineage bot.

The generated source removes leading indentation. The first isolated semantic
ablation changes only the final equal-score move-order tie: Player One prefers
smaller physical `x`, while Player Two prefers larger physical `x`. Those
choices are the same mover-relative direction after a 180-degree player/color
rotation. Replay content and all model parameters remain identical to Rank-4.

The frozen behavior-neutral scaffold is 86,930 ASCII characters with SHA-256
`b2ba9adc171bdbd2c52b9fcd76e3d1a8a024a695b396bfe4d5ace4ff795e95d0`.
It is exactly canonical Rank-4's 98,624-character source with leading
whitespace removed from every line, saving 11,694 characters and leaving
13,069 characters below CodinGame's 99,999-character maximum.

The first mover-relative-tie candidate was 86,988 ASCII characters with
SHA-256
`a092d879a53092b0c5a9c24bf43194226faf38be2cb4b4babc0b4c2c7666f394`.
It intentionally differed from Rank-4 on horizontal-tie states, so the parity
gate now freezes the compact scaffold identity and audits a narrow semantic
delta instead of requiring global action/stat equality. See `EXPERIMENTS.md`
for the paired-rotation corpus results and frozen witnesses.

The hybrid splits the exact rebound/exchange proof into independently
selectable root-goal, leaf-boundary, ply-one, and ply-two bits. Mask zero is
the same-binary control; the full DEVELOPMENT campaign selected mask `7`
(root, leaf, and ply one), not mask `15`. Unknown components preserve the
prior search. Mask 7 won 166-140 against mask 0 and 169-137 against canonical
Rank-4 over the complete 306-game DEVELOPMENT panel.

A later null-action proof fast path was rejected 36-40 on the authoritative
depth-20 DEVELOPMENT clock gate despite passing safety, timing, and search-
progress checks. At that pre-heldout development boundary, the working hybrid
path was rolled back narrowly to the archived pre-fast-path proof algorithm
while retaining operational mask 7. The regenerated rollback artifacts are:

- `bot.cpp`: SHA-256
  `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`.
- `submission.cpp`: 94,312 ASCII characters, SHA-256
  `2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
- `submission_test.cpp`: SHA-256
  `ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697`.

The authoritative held-out outcome for these restored rollback bytes is
recorded below. See `EXPERIMENTS.md` for exact, disabled-parity, all-mask
symmetry, frozen DEVELOPMENT results, and the null-fast-path rejection
lineage.

The later sole-legal-edge ordering bypass was also rejected by its one-shot
DEVELOPMENT gate. It tied 38-38 overall and passed safety, timing, and node-
throughput checks, but scored only 18-20 as physical color 1 against a frozen
minimum of 19 wins per color. The same identity may not be rerun, and the
hybrid files remained the exact rollback artifacts listed above at that
ablation boundary.

The later private PositionKey component cache was also rejected by its only
preregistered DEVELOPMENT clock match. It passed exact-key, fixed-work,
timing, proof-accounting, and search-progress checks, but lost 37-39 and won
only 16 games as physical color 1. The frozen floors were 38 total wins and
19 wins in each color. Its exact identity may not be rerun, and the hybrid was
again restored at that boundary to the 94,312-character mask-7 rollback source
listed above.

The last semantic candidate registered before held-out qualification was
safe-handoff frontier width with the single registered weight 10.
Only exact-proof `Unknown` leaf boundaries count unique safe fresh handoff
endpoints; exact mates bypass the term, while root and ply-one proof scans
retain the rollback algorithm exactly. The candidate was 95,272 ASCII
characters, SHA-256
`08d0c0859ef8a197f8bfdd89afb048bec41c3a888228433b85991cd937882550`,
from `bot.cpp` SHA-256
`408adc5288674550cc08274aec74380074117e32ad8f6915c7e39badc8dfba98`.
Its frozen packet is under
`results/rank_4_jacek_hybrid/frontier_semantic_prototype/`. Stage 0 passed,
but the sole locked Stage-1 benchmark failed: paired median overhead was
1.003703, within the 1.010 ceiling, while paired p99 was 1.106654, above the
1.020 ceiling. The feature is rejected without a retry or whole game, and the
working source was restored at that boundary to the exact mask-7 rollback
above. The canonical decision receipt is
`results/rank_4_jacek_hybrid/gates/frontier_semantic_timing/selection/f85a74985e56e3ad67d3602a44d712e5f511a4a491d313a160583bd764e9be89.json`.

## Authoritative held-out qualification

The restored mask-7 rollback passed the one-shot VALIDATION stage 61-45: 34
wins as physical color 0 and 27 as physical color 1, clearing the frozen
floors of 54 total and 26 in each color. FINAL then finished 104-108, with 48
wins as color 0 and 56 as color 1. It missed the 108-win total floor and the
53-win color-0 floor. Safety, timing, proof and sweep accounting, input and
compiler stability, and source, admin, binding, and portability provenance
all remained clean.

The canonical VALIDATION and FINAL reports are respectively
`results/rank_4_jacek_hybrid/gates/heldout_qualification/binding_recovery_v1/reports/validation/e0b5ed9bd6c77ce90317cc363ab19679e01216172de9ebb01b5eb05d2c6bc5cc.json`
and
`results/rank_4_jacek_hybrid/gates/heldout_qualification/binding_recovery_v1/reports/final/19e0d5e692d5afce2f9a83ef2247bcf53816bf7d66c6a60aa0bcef9205b4c271.json`.
The terminal decision is
`results/rank_4_jacek_hybrid/gates/heldout_qualification/binding_recovery_v1/decisions/9c12b44cc2ffa475e55e1e166c637f725e8107736677c432b03ea31ef376997f.json`:
final qualification and arena authorization are both false. This exact
qualification cannot be retried, and no live upload occurred.

## Post-heldout DEVELOPMENT mask-3 removal

A post-heldout, DEVELOPMENT-only campaign then tested the pre-heldout mask-`3`
fallback against the same-binary mask-`7` control. Stage 1 finished `152-154`:
mask 3 won 85 games as physical color 0 and 67 as physical color 1. It missed
the frozen floors of 160 total wins and 77 wins in each color; the exact misses
were eight total wins and ten color-1 wins. All safety, timing, proof,
engine-work, aggregation, and provenance checks passed.

The canonical report is
`results/rank_4_jacek_hybrid/gates/mask3_removal_clock/reports/64623834951fd4a00484ef0aa1a890127fe9b6a19539b65b3fb874fcf4794725.json`.
The terminal decision is
`results/rank_4_jacek_hybrid/gates/mask3_removal_clock/decisions/894442b0d0a81418d591469bb1b8c1d34cc6c0a8ed13371812a66a21d1e5bc48.json`.
It left Stages 2 and 3 unopened and authorizes no retry, source activation,
fresh held-out campaign, or arena action. Canonical Rank 4 remains the
incumbent; the mask-7 rollback remains historical/control evidence only.

## TT exact-collision generated DEVELOPMENT replication

The preregistered V19 successor reached Stage 0 after a clean nine-child
preexecution receipt, but its 1,324-child Stage-0 corpus did not finish within
the frozen 1,700-second aggregate limit. Children 0 through 966 succeeded;
child 967 was killed at the aggregate boundary after total elapsed time of
1,700.984 seconds, leaving 356 children unstarted. The report is deliberately
unparsed and provides no Stage-0 safety or performance conclusion. No
generated-bank game stage ran and no protected bank file was accessed.

The canonical report is
`results/rank_4_jacek_hybrid/gates/tt_exact_collision_generated_v19/reports/ef323674bbf22dcc18c938b3fcca4b40af3e1978aba0a0ee596addd72ba7de51.json`;
the terminal decision is
`results/rank_4_jacek_hybrid/gates/tt_exact_collision_generated_v19/decisions/3bff57ce07bd80ebc7107192841269a34c08e6e231f6c42fc1ba9c428b5347b0.json`.
It authorizes no retry, source activation, upload, held-out work, or arena
action. Canonical Rank 4 remains the incumbent.

Generate and run the scaffold gates from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs rank_4_jacek_hybrid
node submissions/codingame/tools/generate_submission.mjs rank_4_jacek_hybrid --check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target \
  papersoccer_codingame_rank_4_jacek_hybrid_submission \
  papersoccer_codingame_rank_4_jacek_hybrid_submission_test \
  papersoccer_codingame_rank_4_jacek_hybrid_parity_test
ctest --test-dir build -R rank_4_jacek_hybrid --output-on-failure
```

See `EXPERIMENTS.md` for frozen scaffold identities and all later ablations.
