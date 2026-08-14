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
progress checks. The production path was therefore rolled back narrowly to
the archived pre-fast-path proof algorithm while retaining operational mask
7. The regenerated rollback artifacts are:

- `bot.cpp`: SHA-256
  `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`.
- `submission.cpp`: 94,312 ASCII characters, SHA-256
  `2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
- `submission_test.cpp`: SHA-256
  `ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697`.

No protected heldout or final bank has been opened for this candidate, and it
has not been uploaded to the live arena. Final qualification remains false.
See `EXPERIMENTS.md` for exact, disabled-parity, all-mask symmetry, frozen
DEVELOPMENT results, and the null-fast-path rejection lineage.

The later sole-legal-edge ordering bypass was also rejected by its one-shot
DEVELOPMENT gate. It tied 38-38 overall and passed safety, timing, and node-
throughput checks, but scored only 18-20 as physical color 1 against a frozen
minimum of 19 wins per color. The same identity may not be rerun, and the
production files remain the exact rollback artifacts listed above.

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
