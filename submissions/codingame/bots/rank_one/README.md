# Paper Soccer rank-one candidate

This folder is an isolated CodinGame candidate. The prior production reference
is maintained separately in `../alpha_beta/`.

`bot.cpp` is the maintained implementation and
`submission.cpp` is the single paste-ready C++20 submission. The
generated file is 93,005 ASCII characters, below CodinGame's 100,000-character
limit. Search uses 650 ms on the first execution and 130 ms afterward, leaving
margin below the contest's 1,000 ms and 200 ms limits.

The maintained source is byte-for-byte the strongest completed arena candidate:
history version 26, agent `6561779`, submission `41015554`, rank 5 of 206 with
score `42.42773147296124`. Its SHA-256 is
`f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`.

The candidate retains complete-turn alpha-beta search and its safe fallback.
Its static position evaluator is generated from `replay_value_model.json`, an
88-game public rank-leading replay corpus; the model generator is part of the
same freshness check as the submission. The candidate also adds a generated
exact-history replay book: a response is eligible only
when the player ID and the complete slash-delimited game history match a
publicly verified path. The proposed full turn is applied to a state copy and
used only if every edge is legal and the rebound sequence is complete.
Unmatched, ambiguous, or illegal entries fall back to normal search.

Generate and verify from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs rank_one
node submissions/codingame/tools/generate_submission.mjs rank_one --check
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Useful analysis gates:

```sh
python3 submissions/codingame/tools/screen_replay_book.py \
  AGENT_ID submissions/codingame/bots/rank_one/replay_book.json
python3 submissions/codingame/tools/analyze_arena.py AGENT_ID --pretty
./build/papersoccer_codingame_rank_one_timing_probe
```

`replay_book.json` is the auditable source of replay paths and provenance;
`generate_replay_book.mjs` rejects invalid actions, parity errors, and
conflicting responses at an identical full history. `replay_value_model.json`
is the corresponding auditable evaluator artifact. Do not hand-edit any
generated C++ files or headers. Arena evidence and rejected experiments are
recorded in `EXPERIMENTS.md`.
