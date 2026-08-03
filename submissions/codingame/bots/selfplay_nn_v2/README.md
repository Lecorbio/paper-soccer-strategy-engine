# Self-play teacher-residual bot v2

`selfplay_nn_v2` is an independent challenger built on the verified `rank_5`
engine. The incumbent directories are immutable inputs: this bot copies the
rank-5 complete-turn search, 15% replay-value anchor, exact-history replay
book, terminal handling, and safe rebound-complete fallback. It adds only a
small mover-relative value correction; there is no policy head.

The correction is a 24-input linear model trained on soft teacher-search
scores. Eight inputs reuse the rank-5 replay evaluator's last hidden layer;
the rest are cheap mover-canonical hand features already available during
evaluation. The model is gated off for roots with fewer than 12 played edges,
so the verified opening search remains score- and node-identical. Direct-goal
leaves are also left to the rank-5 anchor. Every correction is capped at 6,000
evaluation points and tapered further when the anchor is confident.

The retained production setting uses the residual at 100% strength and
searches for 800 ms on the first response and 165 ms afterward. The generated
submission is 98,624 characters with SHA-256
`5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`.
Its final timing probe measured at most 800.114/165.089 ms. Locally it beat rank_5
178-128 at 5k nodes over 306 paired games, 58-48 at 30k nodes over 106 games,
and 13-5 over 18 paired timed games using the retained profiles.

The completed CodinGame v42 batch (agent `6567975`) scored
`42.82914600312645` at rank 5 of 206 with a 58-32 record. That exceeds the
verified rank_5 incumbent by `0.40141453016521`. A controlled 800/130 ms
follow-up scored only `40.46675247310762` at rank 6 and was rejected; the
paste-ready source was restored exactly to v42. Its exact-source confirmation
(agent `6567993`) again went 58-32 at rank 5 but scored
`42.001868057527055`. The two exact batches average `42.41550703032675`, just
below the incumbent, so v2 remains an unpromoted experiment rather than the
new champion.

The checked-in training corpus contains 384 deterministic teacher games. A
strict game-seed split supplies 296/43/45 train/validation/test games; exact
feature duplicates cannot cross splits. Shallow searches, mate scores,
opening-gated samples, and the frozen arena-loss suite are excluded. The final
artifact contains 9,625/1,016/1,413 samples and records SHA-256 hashes for both
the corpus and held-out suite. Header and submission generation fail if either
input becomes stale.

The three completed v42/v43/v44 batches remain chronologically unseen by
training. Their 101 losses contribute 192 distinct sampled entries to a
separate observed arena header; together with the original 13 cases, the
comparison gate replays 205 cases covering 203 distinct player/prefix states
for legality, rebound
completion, action, depth, nodes, timing, and residual-evaluation diagnostics.

Generate, build, and verify from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs selfplay_nn_v2
node submissions/codingame/tools/generate_submission.mjs selfplay_nn_v2 --check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Use the pinned research environment and keep regenerated artifacts under the
ignored `results/` tree:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
mkdir -p results/codingame/selfplay_nn_v2
./build/papersoccer_codingame_selfplay_nn_v2_teacher_sample_generator \
  results/codingame/selfplay_nn_v2/teacher_samples.jsonl \
  384 16000 160 8549009122462926626
.venv/bin/python \
  submissions/codingame/bots/selfplay_nn_v2/train_teacher_residual.py \
  results/codingame/selfplay_nn_v2/teacher_samples.jsonl \
  results/codingame/selfplay_nn_v2/teacher_residual_model.json
cmp results/codingame/selfplay_nn_v2/teacher_samples.jsonl \
  submissions/codingame/bots/selfplay_nn_v2/teacher_samples.jsonl
cmp results/codingame/selfplay_nn_v2/teacher_residual_model.json \
  submissions/codingame/bots/selfplay_nn_v2/teacher_residual_model.json
node submissions/codingame/bots/selfplay_nn_v2/generate_teacher_residual_header.mjs --check
```

Run a fixed-node paired gate against the full incumbent:

```sh
SELFPLAY_V2_RESIDUAL_WEIGHT=100 \
SELFPLAY_V2_CANDIDATE_REPLAY=1 SELFPLAY_V2_REFERENCE_REPLAY=1 \
  ./build/papersoccer_codingame_selfplay_nn_v2_comparison_gate 12 30000 200
```

`teacher_residual_model.json` is the auditable model;
`teacher_residual_model.hpp` and `submission.cpp` are generated. The submission
tests cover symmetry, deterministic fixed-node search, phase gating, every
replay-book decision, contest rules, deadline fallback, legality, and complete
rebound handling. Exact measurements, live arena evidence, and rejected
hypotheses are recorded in `EXPERIMENTS.md`.
