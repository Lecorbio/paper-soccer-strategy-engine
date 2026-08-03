# Jacek-anchored search residual bot

`jacek_nn` is a separate challenger that preserves the verified `rank_5`
engine as its anchor. The incumbent `rank_5`, `selfplay_nn`, and
`selfplay_nn_v2` directories are unchanged.

The anchor includes rank_5's complete-turn alpha-beta search, terminal proofs,
transposition and evaluation caches, 24-path/511-decision replay book, hand
evaluation, and its proven 85/15 learned value blend. That learned component is
rank_5's compressed model over the mover-relative inputs described in
[Jacek's Paper Soccer article](https://www.codingame.com/playgrounds/157341/inputs-for-neural-networks-for-the-board-games/paper-soccer):

- 316 mover-canonical used-edge flags;
- one active true-turn-distance bucket out of eight for each of 105 vertices;
- 180-degree rotation for Player 2;
- rank_5's value-only `1156 -> 8 ReLU -> 8 ReLU -> 1` int8 network.

The article's CodinGame implementation uses 32-unit hidden layers. That exact
`1156 -> 32 -> 32 -> 1` replacement was implemented and trained first, but two
complete arena batches rejected it. The retained design keeps the incumbent's
faster 8-unit anchor and uses learning only as a bounded correction.

The new component is a bounded linear residual over 24 mover-relative signals:
the anchor network's second hidden layer and output, hand/anchor scores, ball
position, goal distances, mobility, rebound/goal flags, and game phase. It is
trained on `mover * (deeper teacher root score - full rank_5 anchor score)`.
The correction is disabled before 12 used edges, capped at 6,000 evaluation
points, and applied at the trainer-selected 50% strength. With strength zero,
the challenger matches rank_5 exactly rather than falling back to a weaker hand
evaluation.

The final paste-ready source is 98,623 characters, below CodinGame's 100,000
character limit. Its SHA-256 is
`fb570f7d60157ad1681569011b4249a5db415c1aeca6f665936b26ba5cc52102`.
The retained model JSON has SHA-256
`ed6274f195905536c12086c4317af2ec93320004cdcbc30c80807674e2d5bbb6`.

## Training artifact

The residual was fitted with deterministic Huber IRLS from 10,000 rank-5
teacher games. Labels come from last-completed searches at 8k, 16k, or 32k
nodes; final winners are not used as targets. The source corpus is 166,576,824
bytes with SHA-256
`fb1600d5a667f77a35f23121e2f036faac88877828080eb87b101fa89bde2da4`.

The game-seed split contains 7,960/978/1,062 train/validation/test games and
249,293/30,403/31,837 unique samples. It rejects shallow and mate labels,
states before the residual's 12-edge phase, duplicates across splits, and 110
exact states from the frozen arena holdout. The held-out live states comprise
the prior 203 distinct positions plus 141 deduplicated v45/v46 chronological
cases, for a 338-position distinct union.
The evolving chronological regression header is deliberately separate from
the frozen training-holdout header, so a later arena batch cannot make an
already-submitted model appear stale.

Validation selected 50% strength: anchor/50%-corrected teacher MAE was
4,842.69/4,781.89. The model's test residual sign accuracy at full diagnostic
strength is 62.74%; these metrics screen the fit, while paired games determine
promotion.

The 159 MiB corpus is deliberately untracked, but it is reproducible under
`results/codingame/jacek_nn/`. Build the teacher generator, run 14 deterministic
shards with base budget 16,000, maximum 160 turns, and root seed
`725138164020260722 + shard * 1000000000000`. Shards 0-3 contain 715 games;
shards 4-13 contain 714. Concatenate in shard order while adding the cumulative
prior line count to each JSON record's `game` field. Save the merged file as
`results/codingame/jacek_nn/teacher-10000.jsonl`; it must contain 10,000 lines,
occupy 166,576,824 bytes, and match the corpus hash above.

## Retained local validation

All games use fixed openings in both candidate colors against the unchanged,
replay-enabled rank_5 incumbent:

| Gate | `jacek_nn` | `rank_5` | Color split | Candidate/reference depth |
| --- | ---: | ---: | ---: | ---: |
| 5k nodes, 106 games | 64 | 42 | 36 / 28 | 2.552 / 2.596 |
| 5k nodes, 306 games | 174 | 132 | 91 / 83 | 2.511 / 2.547 |
| 30k nodes, 106 games | 58 | 48 | 34 / 24 | 3.337 / 3.368 |

There were no unfinished games, invalid actions, or incomplete rebounds. The
800/165 profile searched more deeply, but repeated direct tests against the
same challenger at 650/130 split 10-8 and 7-11: 17-19 combined. Equal-profile
650/130 tests against rank_5 likewise split 6-12 and 12-6. Independent complete
arena batches are not paired A/B tests, but the 800/165 v47 source scored
42.3232 at 59-31 while the otherwise identical 650/130 v48 source scored
39.7385 at 52-38. The safer clock was therefore not stronger in the live
sample, and the retained experiment uses 800/165. Fresh automated probes
exercise residual-enabled positions and stay below 801/166 ms.

On all 513 final regression entries, residual strength zero matched rank_5
exactly in action, root score, completed/attempted depth, nodes, and exhaustion.
At the retained 50% strength, 156/513 actions changed at 5k nodes and 184/513
changed at 30k; all remained legal and rebound-complete. The 513 raw entries
represent 497 distinct player/prefix states across the frozen and chronological
suites.
The original 32x replacement model (v45) and its opening-gated follow-up (v46)
failed complete live batches; their measurements and loss validation remain in
`EXPERIMENTS.md` as rejected hypotheses.

The first full-anchor live batch, v47 at the 800/165 profile, scored
`42.32319922389245`, rank 5/206, with a 59-31 record. It narrowly missed the
verified 42.4277 incumbent despite two additional wins. V48 is the controlled
clock ablation and scored only `39.73850251341997`, rank 8/206 at 52-38. The
byte-identical v49 confirmation went 61-29 but scored only
`40.39085147051933`, rank 6/206. The exact-source v47/v49 mean is 41.3570, so
the challenger does not beat the incumbent and remains an unpromoted
experiment. Its combined hardest-five record was 14-49; the extra aggregate
wins did not improve the elite-weighted profile that determines the top rank.

## Build and verification

From the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs jacek_nn
node submissions/codingame/tools/generate_submission.mjs jacek_nn --check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure -j8
```

After recreating the corpus described above, use the pinned research
environment and write the candidate model under the ignored `results/` tree:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
mkdir -p results/codingame/jacek_nn
.venv/bin/python \
  submissions/codingame/bots/jacek_nn/train_rank5_residual.py \
  results/codingame/jacek_nn/teacher-10000.jsonl \
  results/codingame/jacek_nn/teacher_residual_model.json
cmp results/codingame/jacek_nn/teacher_residual_model.json \
  submissions/codingame/bots/jacek_nn/teacher_residual_model.json
node submissions/codingame/tools/generate_submission.mjs jacek_nn --check
```

The model JSON is the auditable training result; the header and
`submission.cpp` are generated. Complete local and live evidence is recorded
in `EXPERIMENTS.md`.
