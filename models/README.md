# Model artifacts

The repository tracks three independent model families. Two live directly in
this directory; the fresh-arena campaign keeps its selected models with its
curated evidence under `results/jacek_arena_bfm/`. They are not interchangeable
and do not share a deployment identity.

| Family | Role | Canonical documentation |
| --- | --- | --- |
| `jacek_replay_bfm_bootstrap.runtime` | Untrained external-model contract fixture for the offline `6301 -> 192 -> 32 -> 1` replay BFM | [`docs/jacek-replay-bfm.md`](../docs/jacek-replay-bfm.md) |
| `jacek_replay_bfm_development/` | Trained, game-tested development checkpoint for the same replay BFM; stronger than the bootstrap but not promotion-qualified | [`docs/jacek-replay-bfm.md`](../docs/jacek-replay-bfm.md) |
| `jacek_article_value_model.json` | Browser/native `JacekInspiredBot` under standard demo rules | The model card below |
| `jacek_native_*` JSON and runtime files | Independently trained checkpoints, selections, and deployment descriptors for the CodinGame `jacek_native_bfm` research track | [`jacek_native_bfm/README.md`](../submissions/codingame/bots/jacek_native_bfm/README.md) and [`ROUND2_SELECTION.md`](../submissions/codingame/bots/jacek_native_bfm/ROUND2_SELECTION.md) |
| `results/jacek_arena_bfm/models/` | Separate clean-room fresh-arena models; selected scratch-only 8k seed-101 identity `fresh-32x32-s101-778156217cbd`, offline and unqualified | [`jacek_arena_bfm/README.md`](../submissions/codingame/bots/jacek_arena_bfm/README.md) and [selection record](../results/jacek_arena_bfm/selection/c5e93f516fe6c754210a7a678af8c9f77c87879688afe8c4c836706664c8d6fe.json) |

The native-BFM deployment descriptor selects an exact runtime checkpoint and
generated header; merely placing another model in this directory does not
activate it. The browser model described below is generated independently into
`src/bots/jacek_inspired/jacek_neural_model.hpp`. The selected fresh-arena
model likewise remains an offline campaign artifact; it failed the mandatory
H62 and both-color strength requirements and was never uploaded.

`models/jacek_replay_bfm_promoted/` is reserved for the atomic output of the
replay campaign's `publish` command. It must not exist for an ineligible or
incomplete campaign and never replaces the maintained Rank-4 submission.

## Jacek-inspired demo model

`jacek_article_value_model.json` is an independently trained checkpoint for
`JacekInspiredBot`. It follows the representation and compact network described in
[Jacek Dermont's Paper Soccer article](https://www.codingame.com/playgrounds/157341/inputs-for-neural-networks-for-the-board-games/paper-soccer),
but it does not contain his unpublished weights and does not import code or artifacts from a
contest submission.

## Demo model contract

- Rules: the normal demo's standard 8x10 field, `OpponentGoalOnly`, and
  `PlayerToMoveLoses`.
- Perspective: the position is always encoded for the player to move. Player Two positions
  are rotated 180 degrees, so the mover always attacks upward.
- Inputs: 1,156 binary values.
  - 316 inputs mark drawn edges.
  - 105 groups of eight one-hot inputs encode cooperative true-turn distance to every
    vertex, capped at bucket seven.
- Network: `1156 -> ReLU(32) -> ReLU(32) -> 1`.
- Target: mover-relative soft alpha-beta root score with temperature 12,000.
- Output use: the scalar leaf value is converted back to a fixed Player-One score and used
  inside the engine's existing possession-aware iterative-deepening alpha-beta search.

The distance map charges one for entering a fresh interior vertex and zero for entering a
visited vertex, boundary, or goal. It is a deliberately cooperative representation of the
remaining graph, not a replacement for adversarial search or an exact statement about future
turn ownership under the demo's goal rule.

The first layer is evaluated sparsely: every position has exactly one active distance input per
vertex plus one active input per drawn edge. Later layers use ordinary float inference. The
checked-in header embeds the model and its SHA-256, while CTest runs the generator in `--check`
mode to reject stale generated weights.

## Training provenance

The checked-in model was produced on July 23, 2026 from scratch:

- 1,024 deterministic self-play games, 1,023 terminal.
- 22,597 exported samples before cross-split duplicate removal. Searches that did not
  complete teacher depth one were rejected instead of being mislabeled as neutral targets.
- Teacher: the normal demo's hand-evaluated `AlphaBetaBot`, maximum possession depth 5,
  4,000 nodes, 16,384 transposition entries, and a 12-ply soft physical horizon.
- Exploration: every move for the first six plies, then 18% random moves; at most 24 evenly
  spaced samples per game.
- Split: deterministic by complete game seed, so positions from one game cannot cross
  train/validation/test boundaries.
- Optimizer seed: `20260723`; early stopping selected epoch 9.

Artifact identities:

| Artifact | SHA-256 |
| --- | --- |
| Training corpus | `7ced833fcab014a9f960768eddde1f9374e6bec6e8360e08dbd38eb8a9554a63` |
| Trainer | `35c55c7575f2168d955dac585c31018ca64a5a05a81c87cd1975f25428677dbd` |
| Model JSON | `57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084` |
| Generated C++ header | `08bf42336214f4c5d3257fd0251b3116a09210cac44b4337b0710001427896f9` |

Held-out metrics:

| Split | Unique samples | Weighted soft BCE | Teacher-score correlation | Teacher-sign accuracy |
| --- | ---: | ---: | ---: | ---: |
| Train | 16,177 | 0.689270 | 0.8645 | 90.44% |
| Validation | 1,890 | 0.690715 | 0.8352 | 89.79% |
| Test | 2,140 | 0.688554 | 0.8491 | 88.13% |

An independent clean corpus generation and retraining reproduced the corpus hash and model JSON
byte-for-byte. The C++ test suite also binds a non-empty legal position to an exact active-feature
hash and an independently computed float32 logit, so edge-order drift cannot hide behind the
all-empty initial edge vector.

The large teacher-score MAE stored in the JSON is expected because proven terminal scores use
the engine's mate-scale values while the compact model is trained through a bounded soft
target. Correlation, sign agreement, paired games, and legal/timing gates are more useful here.

## Reproduction

NumPy is required only to train. The game and checked-in model have no Python or NumPy runtime
dependency. Create the pinned environment once and keep regenerated data under the ignored
`results/` tree. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
mkdir -p results/research/jacek_article
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target papersoccer_jacek_training_data papersoccer_arena

./build/papersoccer_jacek_training_data \
  results/research/jacek_article/training_v2.jsonl 1024 73194721 4000

.venv/bin/python tools/train_jacek_neural.py \
  results/research/jacek_article/training_v2.jsonl \
  --output results/research/jacek_article/value_model.json \
  --seed 20260723 \
  --epochs 50

cmp results/research/jacek_article/value_model.json \
  models/jacek_article_value_model.json
.venv/bin/python tools/generate_jacek_neural_model.py --check
```

The corpus is intentionally a generated build artifact rather than a checked-in multi-megabyte
dataset. Its hash is stored in the model, and all code, seeds, limits, and commands needed to
recreate it are versioned.

## Demo validation

The promoted browser profile uses depth 6 and a deterministic 20,000-node cap. Its wall-clock
limit is disabled, avoiding load-dependent move changes; a completed shallower iteration remains
usable if the node cap interrupts a deeper iteration.

Fixed-seed gates for the reviewed checkpoint on the development machine:

| Gate | Result | Jacek median / p95 | Safety |
| --- | --- | ---: | --- |
| 32 shared random midgames vs equal-budget `AlphaBetaBot` | 5/32 move changes | 27.68 / 36.76 ms | 0 illegal moves |
| 12 paired openings, 24 games vs equal-budget `AlphaBetaBot` | 12-12 | 11.85 / 31.31 ms | 0 illegal, 0 truncated |
| 12 paired openings, 24 games vs Tactical `MctsBot` at 2,000 iterations | 21-3 | 15.59 / 32.85 ms | 0 illegal, 0 truncated |
| 20 paired openings, 40 games vs `RandomBot` | 39-1 | 21.59 / 34.59 ms | 0 illegal, 0 truncated |

The shared-position gate completed depths `{0:2, 1:9, 2:3, 3:2, 4:2, 5:5, 6:9}` and hit the
node cap 15 times. The paired alpha-beta gate had no depth-zero fallback; 1,236 of 1,621
decisions completed depth 6. Arena JSON records the exact model hash, completed/attempted-depth
histograms, budget exhaustion, timings, and candidate/reference move changes.

The checked-in WebAssembly artifact produced 80 decisions over 12 scripted games and 46 distinct
positions: 21.53 ms median, 40.64 ms p95, 43.69 ms maximum, 13 node-cap interruptions, and no
invalid root scores. Completed depths were `{1:14, 5:13, 6:53}`. This is a deterministic path
probe, not a general browser latency guarantee.

A physical-horizon ablation kept the 12-ply profile. At the same 20,000-node budget, an 8-ply
candidate was much faster but scored 19-21 against 12 plies over 40 color-swapped games; a
10-ply candidate scored 16-24. Neither shorter horizon supplied evidence of greater playing
strength.

Reproduce the fixed gates with:

```bash
./build/papersoccer_arena positions \
  --positions 32 --generation-plies 24 --seed 828927513140 \
  --candidate-kind jacek-inspired \
  --candidate-alpha-beta-depth 6 \
  --candidate-alpha-beta-max-nodes 20000 \
  --reference-kind alpha-beta \
  --reference-alpha-beta-depth 6 \
  --reference-alpha-beta-max-nodes 20000

./build/papersoccer_arena matches \
  --pairs 12 --opening-plies 12 --max-plies 512 \
  --bootstrap-samples 2000 --seed 828927513140 \
  --candidate-kind jacek-inspired \
  --candidate-alpha-beta-depth 6 \
  --candidate-alpha-beta-max-nodes 20000 \
  --reference-kind alpha-beta \
  --reference-alpha-beta-depth 6 \
  --reference-alpha-beta-max-nodes 20000

./build/papersoccer_arena matches \
  --pairs 12 --opening-plies 12 --max-plies 512 \
  --bootstrap-samples 2000 --seed 828927513140 \
  --candidate-kind jacek-inspired \
  --candidate-alpha-beta-depth 6 \
  --candidate-alpha-beta-max-nodes 20000 \
  --reference-kind mcts --reference-iterations 2000

./build/papersoccer_arena matches \
  --pairs 20 --opening-plies 12 --max-plies 512 \
  --bootstrap-samples 4000 --seed 828927513140 \
  --candidate-kind jacek-inspired \
  --candidate-alpha-beta-depth 6 \
  --candidate-alpha-beta-max-nodes 20000 \
  --reference-kind random

node benchmarks/wasm_jacek.mjs 80 6 10
```

These are usability and regression measurements, not evidence that this model matches Jacek's
contest bot or improves the repository's CodinGame rank. The implementation is intentionally a
normal-demo experiment and leaves every contest submission unchanged.
