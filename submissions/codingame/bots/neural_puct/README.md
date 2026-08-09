# Neural PUCT

`neural_puct` is an NN-first, atomic-edge PUCT candidate for the CodinGame
paper-soccer rules. It is intentionally independent of `rank_5`: there is no
replay book, alpha-beta search, handcrafted evaluator, or inherited rank-5
decision. Exact rules resolve terminal states and one-edge wins; every other
leaf value and move prior comes from the policy/value network.

The `neural-puct-features-v1` input has 1,286 values in the current mover's
frame (Player One unchanged, Player Two rotated 180 degrees):

- 316 used-edge flags;
- one of eight true-turn-distance buckets for each of 105 vertices;
- remaining incident-edge degree divided by eight for each vertex;
- 25 global values: canonical ball coordinates, used/visited fractions, ball
  degree, attacking/own-goal turn distance, rebound-component and handoff
  summaries, and used-edge saturation for the 12 horizontal cuts.

The model is `1286 -> ReLU(32) -> ReLU(32) -> value + policy(8)`. Weights are
signed symmetric int4 values packed two per byte, with one scale per output
channel. The checked-in artifact is the deterministic seed-20260809 expert
imitation fit. It was selected from three seeds by the lowest combined
quantized validation value and policy loss; the test split was not consulted
for selection. Later self-play candidates remain experiments rather than
silently replacing this reproducible baseline. See `EXPERIMENTS.md` for the
match-backed promotion decisions.

Training is game-split and expert-first. It uses the exposed frozen elite
corpora plus the public Jacek games in `public_jacek_unlocked_v1.json`. The
Jacek fetcher filters every game id in `rank1_locked_games.json` before fetching
game details. Rank-5 self-play is not loaded by the maintained training command
and remains performance/diagnostic data only. Value weight is normalized per
expert and then per game. Policy weight is normalized the same way, with a
four-times Jacek expert multiplier. Exact duplicate self-play trajectories are
grouped before the whole-game split, so copies cannot cross train, validation,
and test. Random opening prefixes are fully replayed for legality but
contribute neither value nor policy samples. Validation canonical states found
in training are removed; test states found in training or the original
validation split are removed. Context-dependent targets for the same state
remain together within their assigned split.

Neural-only DAgger iterations may be trained as temporary A/B candidates with
repeatable `--selfplay-corpus` arguments and `--selfplay-multiplier`. The
multiplier applies to both value and policy mass for the self-play source; it
is recorded with every shard hash and corpus teacher in the model report.
Duplicate shard paths and duplicate `(seed, game)` identities across shards
are rejected. Candidate iterations should write to `/tmp` or `results` and
replace the maintained expert artifact only after an exposed gate win. Passing
a non-default multiplier without a self-play corpus is rejected.

The append-only CodinGame collection, labelling, retraining, and submission
protocol is documented in [LIVE_REPLAY_LOOP.md](LIVE_REPLAY_LOOP.md). It keeps
locked and promotion evidence outside training, uses elite actions as direct
labels, and relabels the neural bot's own positions with deeper neural search.

The first live round is archived in
[`live_replay/FIRST_ROUND_REPORT.md`](live_replay/FIRST_ROUND_REPORT.md). The
already active maintained source was reused byte-for-byte, and a top-20 sweep
plus ten follow-up polls produced 34 valid independent games, below the frozen
50-game training floor. No labels, model change, candidate match, or additional
submission was made. The append-only collector can resume the same run when
normal arena activity creates more strong-opponent games.

Build the neural-only self-play generator and create a deterministic shard with:

```sh
cmake --build build \
  --target papersoccer_codingame_neural_puct_selfplay_generator
./build/papersoccer_codingame_neural_puct_selfplay_generator \
  results/codingame/neural_puct/neural-selfplay.jsonl 128 5000 400 2026080900
```

Its arguments are `OUTPUT GAMES [SIMULATIONS] [MAX_TURNS] [SEED]`; the
simulation default is 2,000. Short deterministic-random opening prefixes give
the shard state diversity and are excluded from policy labels through
`teacher_start_turn`. Both players use the frozen `neural_puct` search after
that prefix. Each `papersoccer.selfplay.v1` record identifies
`teacher: "neural_puct"`, binds the exact model artifact SHA-256, and records
the requested budget, actual simulation count, neural evaluations, and node
cap. A deadline or node-cap truncation is a generation error.

Self-play records may additionally carry
`canonical-primitive-root-visits-v1` policy targets. The target list aligns
with turns and primitive rebound edges, keeps random-opening entries null, and
records each legal canonical direction's normalized root visits. Training uses
weighted soft cross-entropy for those distributions and mirrors all eight
probabilities with the spatial augmentation. Public games and older self-play
records remain exactly backward-compatible one-hot action targets. The loader
rejects missing teacher-model hashes, mixed teacher identities, illegal target
mass, non-normalized probabilities, and a played edge that is not visit-max.

PUCT stores values in Player-One coordinates after converting the network's
mover-relative result once. That keeps value backup correct through mandatory
rebounds, where the mover does not change. The binary-cross-entropy logit is
converted to a mover expectation as `tanh(logit / 2)`. A single tree is
searched per CodinGame turn. The response follows visit-max descendants until
possession changes, using a masked network policy fallback if the searched
tree lacks a required rebound child. The encoder supplies a sorted sparse
feature list to the first layer; exhaustive dense/sparse parity checks found
bit-identical outputs and search decisions, while fixed-work inference became
about 19% faster. The construction-inclusive clocks are 650 ms for the first
response and 130 ms thereafter, with a hard 100,000-node cap. The default PUCT
exploration constant is 0.45, selected by a small exposed bootstrap.

The first larger exposed-development diagnostic used all 48 T3 openings in
both physical colors at 2,000 candidate simulations versus 5,000 rank-5 nodes.
This artifact scored 42-54 (Player 0: 22/48; Player 1: 20/48), with no illegal
or unfinished games. With both bots limited by the same 20 ms clock, the
optimized baseline scored 39-57. It is therefore a maintained NN-first
research baseline, not a promotion candidate. No neural candidate has been
submitted to CodinGame from this directory.

Validate the frozen public corpus, reproduce the maintained model, and verify
the generated model header and submission with:

```sh
python3 submissions/codingame/bots/neural_puct/fetch_public_jacek_games.py --check
python3 submissions/codingame/bots/neural_puct/train_neural_puct.py \
  --seed 20260809 --epochs 40
node submissions/codingame/bots/neural_puct/generate_neural_puct_model_header.mjs
node submissions/codingame/bots/neural_puct/generate_neural_puct_model_header.mjs --check
node submissions/codingame/tools/generate_submission.mjs neural_puct
node submissions/codingame/tools/generate_submission.mjs neural_puct --check
```

Build and run the focused checks with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target papersoccer_codingame_neural_puct_submission_test \
  papersoccer_codingame_neural_puct_feature_probe
python3 submissions/codingame/bots/neural_puct/check_feature_parity.py
ctest --test-dir build --output-on-failure -R '^papersoccer_codingame_neural_puct_'
```
