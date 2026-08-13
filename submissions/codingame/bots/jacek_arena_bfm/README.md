# `jacek_arena_bfm`

`jacek_arena_bfm` is the clean-room bot and data lineage for the fresh-arena
campaign.  Its engine, bootstrap weights, scratch games, and labels were
created in this namespace after the campaign boundary.  It does not load an
older submission, checkpoint, action, label, replay, or corpus.

## Engine contract

- The immutable board topology has 105 vertices and 316 playable undirected
  edges.  Goals, own goals, rebounds, boundary rebounds, blocking, and the
  contest direction protocol are handled directly by `engine.cpp`.
- Search actions are complete rebound turns.  Equal successor boundaries are
  deduplicated before search.
- `Fixed250NineOne`, `TacticalProgressive`, and `PriorityBeam` generators are
  measured against an 8,192-action high-cap recall generator.  The selected
  operational default is tactical progressive widening: at most 512 root and
  256 non-root actions.
- Features are mover-relative: 316 edge bits plus one of eight true-turn
  distance buckets for each of 105 vertices, for 1,156 inputs total.
- The scalar evaluator is bias-free `1156 -> H -> 32 -> 1`, with `tanh` after
  every layer.  `model.hpp` starts from deterministic random initialization;
  fresh training replaces it with RFC 4648 base64-packed symmetric-int8
  weights.
- Search is single-threaded UCT-style best-first minimax.  The contest entry
  uses 800 ms for its first decision and 155 ms thereafter, with 80,000 nodes
  as a second hard bound.

## Build and verify

From the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs jacek_arena_bfm
c++ -std=c++20 -O3 \
  submissions/codingame/bots/jacek_arena_bfm/submission_test.cpp \
  -o /tmp/jacek_arena_bfm_test
/tmp/jacek_arena_bfm_test
```

The optional standalone probes each include the exact generated artifact:

```sh
c++ -std=c++20 -O3 \
  submissions/codingame/bots/jacek_arena_bfm/generator_benchmark.cpp \
  -o /tmp/jacek_arena_bfm_generators
/tmp/jacek_arena_bfm_generators 128

c++ -std=c++20 -O3 \
  submissions/codingame/bots/jacek_arena_bfm/timing_probe.cpp \
  -o /tmp/jacek_arena_bfm_timing
/tmp/jacek_arena_bfm_timing
```

## Fresh scratch corpus

`selfplay_generator.cpp` accepts no checkpoint, replay, action, or label input.
It distributes games across procedural opening depths `0,4,8,12` and supports
non-overlapping deterministic shards through `--start-index`.  Game JSONL
retains complete turns for independent replay.  `--rows` instead writes the
validator-native dense 1,156-feature value rows, including a canonical SHA-256
for each generated game:

```sh
c++ -std=c++20 -O3 \
  submissions/codingame/bots/jacek_arena_bfm/selfplay_generator.cpp \
  -o /tmp/jacek_arena_bfm_selfplay
/tmp/jacek_arena_bfm_selfplay \
  --games 2000 --seed 0x4a4143454b465245 --rows \
  --campaign-id CAMPAIGN_ID --timestamp T0_OR_LATER_UTC \
  --producer-source-sha256 EXACT_SUBMISSION_SHA256 \
  --output CAMPAIGN_OWNED_OUTPUT.jsonl
```

Every row labels the pre-decision position from its mover's perspective.  The
raw trajectory outcome is a value target only; it is not presented as a proof
that every losing-game position was theoretically losing.
