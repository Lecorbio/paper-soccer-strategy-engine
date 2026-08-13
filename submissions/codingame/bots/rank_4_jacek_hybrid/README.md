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

The current mover-relative-tie candidate is 86,988 ASCII characters with
SHA-256
`a092d879a53092b0c5a9c24bf43194226faf38be2cb4b4babc0b4c2c7666f394`.
It intentionally differs from Rank-4 on horizontal-tie states, so the parity
gate now freezes the compact scaffold identity and audits a narrow semantic
delta instead of requiring global action/stat equality. See `EXPERIMENTS.md`
for the paired-rotation corpus results and frozen witnesses.

The current isolated candidate splits the exact rebound/exchange proof into
independently selectable root-goal, leaf-boundary, ply-one, and ply-two bits.
Mask zero is the same-binary control; the operational choice path explicitly
enables mask 15. Unknown components preserve the prior search. Its generated
identity is 94,004 ASCII characters with SHA-256
`6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f`.
See `EXPERIMENTS.md` for exact, disabled-parity, all-mask symmetry, and the
preregistered development matrix.

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
