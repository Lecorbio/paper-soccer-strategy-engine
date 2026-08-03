# Self-play policy/value bot

`selfplay_nn` is an independent CodinGame candidate. It keeps the verified
complete-turn alpha-beta engine and its exact terminal search, transposition
table, and forced-rebound handling. A compact self-play network contributes a
conservative value residual. The retained release also restores the verified
rank-5 replay book and validates two exact opening safeguards before applying
them.

The checked-in submission is 93,621 characters with SHA-256
`1f980ece41a8ec86347deb77ec4625b1a402f15e1d82456ab04fe1bac40a6389`.
It searches for 650 ms on the first response and 130 ms afterward. The selected
learned configuration is:

- 423 mover-canonical inputs: 316 used edges, 105 clipped true-turn distance
  scalars, and normalized ball x/y;
- two 16-unit ReLU layers with one value logit and eight direction logits;
- symmetric int8 weights with float biases;
- 10% learned value / 90% hand evaluation at depth-zero leaves;
- policy ordering disabled after its ablation failed to add strength;
- exact legal actions `1 -> 1` for Player 1 and `0/6 -> 5` for Player 0,
  followed by the normal verified replay-book lookup and safe search fallback.

The model is trained from 1,024 deterministic rank-5 self-play games. One game
anchors the initial position; every other game starts after 4, 6, 8, 12, 16,
or 20 random legal complete turns. Teacher node budgets are independently
sampled from 2,000, 4,000, and 8,000 nodes. Horizontal reflections stay in the
same game split. Exact feature overlaps are removed from validation and test.
Policy loss is masked to legal directions, and value labels are consumed only
at complete-turn boundaries because those are the states evaluated by the
production alpha-beta search.

The generated model records 809/108/107 train/validation/test games. Its
post-quantization held-out test metrics are 53.86% value accuracy, 0.7146 value
log loss, 37.60% legal policy top-1, and 77.85% legal policy top-3. These
classifier metrics are screening signals, not promotion evidence.

Local promotion evidence for the original 10%/500 setting:

- 68-38 over 106 equal-node games against rank_5 with replay disabled;
- 66-40 over the same 106-game gate when rank_5 keeps replay corrections;
- 12-6 over 18 paired 130 ms games with replay disabled;
- 11-7 over 18 paired 130 ms games when rank_5 keeps replay corrections;
- at 130 ms on nine recurrent probe positions, completed depth improved on six,
  tied on two, and regressed on one.

The retained value-only/replay hybrid is 67-39 over the 106-game equal-node
gate against full rank_5 and 12-6 over the 18-game paired 130 ms gate. Three
complete CodinGame batches were evaluated:

| Version | Agent | Rank | Score | Record |
| ---: | ---: | ---: | ---: | ---: |
| 39 | `6567850` | 9 | `39.59113038088426` | 59-31 |
| 40 | `6567868` | 6 | `41.2027135828999` | 62-28 |
| 41 | `6567898` | 5 | `41.77236003585933` | 59-31 |

Version 41 is the strongest new bot by weighted score, but it does not exceed
the immutable rank-5 verification (`42.42773147296124`, rank 5). It therefore
remains `selfplay_nn`; `rank_5` is still the verified reference.

Generate, build, and test from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs selfplay_nn
node submissions/codingame/tools/generate_submission.mjs selfplay_nn --check
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Create the pinned research environment once, then recreate raw artifacts under
the ignored `results/` tree:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
mkdir -p results/codingame/selfplay_nn
./build/papersoccer_codingame_selfplay_nn_selfplay_generator \
  results/codingame/selfplay_nn/selfplay_bootstrap.jsonl \
  1024 4000 180 6789347283849021
.venv/bin/python \
  submissions/codingame/bots/selfplay_nn/train_selfplay_model.py \
  results/codingame/selfplay_nn/selfplay_bootstrap.jsonl \
  results/codingame/selfplay_nn/neural_model.json
cmp results/codingame/selfplay_nn/neural_model.json \
  submissions/codingame/bots/selfplay_nn/neural_model.json
node submissions/codingame/bots/selfplay_nn/generate_neural_model_header.mjs --check
```

Run the deterministic comparison gate against the full rank-5 control:

```sh
SELFPLAY_VALUE_BLEND=10 SELFPLAY_POLICY_WEIGHT=0 \
SELFPLAY_REFERENCE_REPLAY=1 SELFPLAY_CANDIDATE_REPLAY=1 \
  ./build/papersoccer_codingame_selfplay_nn_comparison_gate 12 5000 240
```

`neural_model.json` is the auditable model; `neural_model.hpp` and
`submission.cpp` are generated and must not be hand-edited. The submission
test includes Python-to-C++ golden logits for both players, rotation
invariance, exact replay reconstruction, contest-rule checks, and
mandatory-rebound completeness.
