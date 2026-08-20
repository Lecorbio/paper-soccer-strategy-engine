# Goal-shell rebound-proof experiment

**Status:** archived experimental comparator, not a production or platform-rank
claim. It remains buildable and participates in the frozen local CodinGame-rules
leaderboard.

This candidate starts from the immutable `rank_5` engine. It keeps the exact
current-possession rebound-goal proof at the root and runs the same Win/Loss
component analysis at depth-zero boundaries only when the mover is near the
attacking goal shell. `Unknown` positions continue through the original
evaluation and complete-turn alpha-beta search unchanged.

The generated submission is 96,619 characters with SHA-256
`857f27ceeb5bb9cb514622eddb0b326b6fd2b1d55b6ed54e20869879f45fcce6`.
Its checked-in evidence is the source, focused tests, comparison harness, timing
probe, and frozen local leaderboard result; no CodinGame promotion result is
claimed.

Generate and verify it from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs shell_proof --check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target papersoccer_codingame_shell_proof_submission_test
ctest --test-dir build \
  -R '^papersoccer_codingame_shell_proof_submission_test$' \
  --output-on-failure
```
