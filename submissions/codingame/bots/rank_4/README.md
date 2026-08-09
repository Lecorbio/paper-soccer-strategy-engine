# Paper Soccer rank-4 bot

`rank_4` is the canonical production snapshot of the teacher-residual bot that
completed CodinGame history version 56 as agent `6604719`, submission
`41114327`. Its completed 90-game arena batch ranked 4 of 208 with score
`44.29750553418035` and a 66-24 record.

The paste-ready source is 98,624 characters with SHA-256
`5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`.
It is byte-identical to the retained 800/165 ms `selfplay_nn_v2` artifact. An
authenticated history inspection matched the active source's distinctive
clock constants, 100% teacher-residual setting, and model coefficients. The
CodinGame UI does not expose a remote source digest, so the version-56 binding
is a source fingerprint plus agent/submission identity rather than a claimed
remote SHA-256 measurement.

The engine retains the complete-turn alpha-beta search, exact-history replay
book, and 15% compact replay-value anchor from `rank_5`. It adds a capped
24-feature linear value residual trained on deeper soft search targets. The
residual is mover-relative, disabled in the opening phase, capped at 6,000
evaluation points, and tapered when the anchor is confident. There is no
policy head.

Historical training samples, trainer code, arena batches, and their frozen
path-based manifests remain in `../selfplay_nn_v2`. The rank-4 model generator
verifies those immutable provenance inputs instead of duplicating them. The
rank-4 directory owns the production source, generated headers, regression
headers, tests, timing probe, and comparison gate.

Generate and verify from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs rank_4
node submissions/codingame/tools/generate_submission.mjs rank_4 --check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Run the focused fixed-work comparison against `rank_5` with:

```sh
RANK_4_RESIDUAL_WEIGHT=100 \
RANK_4_CANDIDATE_REPLAY=1 RANK_4_REFERENCE_REPLAY=1 \
  ./build/papersoccer_codingame_rank_4_comparison_gate 12 30000 200
```

The promotion record, opponent breakdown, caveats, and bounded improvement
directions are in `EXPERIMENTS.md`. The original training and version 42-44
evidence remains in `../selfplay_nn_v2/EXPERIMENTS.md`.
