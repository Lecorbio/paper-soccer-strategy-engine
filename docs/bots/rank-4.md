# Teacher-residual alpha-beta

[Bot catalog](../../submissions/codingame/bots/README.md) · [Standalone C++20 source](../../submissions/codingame/bots/rank_4/submission.cpp)

## Design

Replay-guided search with a bounded linear correction learned from deeper teacher
searches.

The implementation ID is `rank_4`. This stable name is used by the build, generators,
and historical evidence.

## Usage

[`submission.cpp`](../../submissions/codingame/bots/rank_4/submission.cpp) is the self-contained contest program. The maintained
inputs and local checks live in the `rank_4` bot directory. Use the source for local comparisons; the
results below describe its measured status.

## Results

History version 56 ranked 4 of 208: 66 wins and 24 losses over 90 games, score
44.29750553418035. It is a historical snapshot, not the latest bot.

See the [frozen local
leaderboard](../../benchmarks/codingame_leaderboard/README.md) for comparisons
under the shared contest protocol. Local scores and historical platform ranks are
different measurements.

## Testing

From the repository root after [project
setup](../development.md#build-and-test):

```sh
node submissions/codingame/tools/generate_submission.mjs rank_4 --check
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2 --target \
  papersoccer_codingame_rank_4_submission \
  papersoccer_codingame_rank_4_submission_test
ctest --test-dir build/native \
  -R '^papersoccer_codingame_rank_4_(submission_current|submission_test|protocol_smoke_test)$' \
  --output-on-failure
```

## History

The [experiment log](../../submissions/codingame/bots/rank_4/EXPERIMENTS.md) retains the chronological comparisons, rejected
hypotheses, and source identities. Frozen source snapshots and reports keep their
recorded identities.

<details>
<summary>Detailed design and comparison notes</summary>

`rank_4` is the historical production snapshot of the teacher-residual bot that
completed CodinGame history version 56 as agent `6604719`, submission `41114327`. Its
completed 90-game arena batch ranked 4 of 208 with score `44.29750553418035` and a 66-24
record.

The paste-ready source is 98,624 characters with SHA-256
`5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`. It preserves the
retained 800/165 ms teacher-residual artifact exactly. An authenticated history
inspection matched the active source's distinctive clock constants, 100% residual
setting, and model coefficients. The CodinGame UI does not expose a remote source
digest, so the version-56 binding is a source fingerprint plus agent/submission identity
rather than a claimed remote SHA-256 measurement.

The engine retains the complete-turn alpha-beta search, exact-history replay book, and
15% compact replay-value anchor from `rank_5`. It adds a capped 24-feature linear value
residual trained on deeper soft search targets. The residual is mover-relative, disabled
in the opening phase, capped at 6,000 evaluation points, and tapered when the anchor is
confident. There is no policy head.

The complete development lineage is retained in the `rank_4` bot directory. It includes the
384-game teacher corpus, trainer, model, three chronological arena batches, 205
regression cases over 203 distinct states, and the generators that bind those inputs to
the production headers. Moving the evidence did not alter its bytes or the production
submission. Historical content-addressed campaign records retain their original path
strings as immutable audit evidence.


Recreate the teacher corpus and model in an isolated results directory with:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
mkdir -p results/codingame/rank_4
./build/native/papersoccer_codingame_rank_4_teacher_sample_generator \
  results/codingame/rank_4/teacher_samples.jsonl \
  384 16000 160 8549009122462926626
.venv/bin/python \
  submissions/codingame/bots/rank_4/train_teacher_residual.py \
  results/codingame/rank_4/teacher_samples.jsonl \
  results/codingame/rank_4/teacher_residual_model.json
cmp results/codingame/rank_4/teacher_samples.jsonl \
  submissions/codingame/bots/rank_4/teacher_samples.jsonl
cmp results/codingame/rank_4/teacher_residual_model.json \
  submissions/codingame/bots/rank_4/teacher_residual_model.json
node submissions/codingame/bots/rank_4/generate_teacher_residual_header.mjs --check
```

Run the focused fixed-work comparison against `rank_5` with:

```sh
RANK_4_RESIDUAL_WEIGHT=100 \
RANK_4_CANDIDATE_REPLAY=1 RANK_4_REFERENCE_REPLAY=1 \
  ./build/native/papersoccer_codingame_rank_4_comparison_gate 12 30000 200
```

The production promotion, opponent breakdown, caveats, and bounded improvement
directions are in `EXPERIMENTS.md`. The original teacher training, ablations, and
version 42-44 evidence are preserved in `DEVELOPMENT_HISTORY.md`.

</details>
