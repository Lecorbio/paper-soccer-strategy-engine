# Tools

These programs build artifacts, inspect games, and reproduce the project's
experiments. The engine and its bot implementations live in `src/` and
`submissions/codingame/bots/`; this directory supplies their offline tooling.

## Useful starting points

| Task | Program or guide |
| --- | --- |
| Check documentation | [`check_documentation_links.py`](check_documentation_links.py) |
| Verify standalone submissions | [CodinGame tools](../submissions/codingame/tools/README.md) |
| Generate the demo model header | [`generate_jacek_neural_model.py`](generate_jacek_neural_model.py) |
| Prepare opening banks | [`opening_bank.cpp`](opening_bank.cpp) |
| Compare bots or inspect a replay | [Native commands](../docs/demo-and-replays.md) |
| Check curated benchmark outputs | [Reproducibility](../docs/reproducibility.md) |

## Research families

| Prefix | Purpose |
| --- | --- |
| `jacek_replay_` | Replay features, datasets, value models, and runtime checks |
| `jacek_selfsearch_` | Teacher searches and student-model comparisons |
| `compact_value_bfm_` | Compact value models and teacher distillation |
| `rank_two_focused_` | Small-network training and native deployment checks |
| `rank_two_live_eight_` | Quantization, source-size limits, timing, and live comparisons |
| `rank_two_live_` | Archived platform observations and matched comparisons |
| `top_three_` | Earlier search experiments and job queues |

See the [tooling guide](../docs/tooling.md) for responsibilities and the
[research archive](../docs/archive/README.md) for outcomes and historical
protocols. Versioned files retain the procedures used by their recorded runs.
They are kept for reproducibility rather than presented as a single current API.

Some historical recorders require machine-local files or a particular archived
workspace. Their original results remain useful, but those recorders are not
portable setup commands. Routine builds and tests do not start production training,
download game archives, or submit bots.
