# CodinGame submissions

CodinGame support is split into reusable tooling and self-contained bots:

```text
submissions/codingame/
├── bots/                 22 leaderboard artifacts plus research-only entrants
├── promotion/            historical rank-5-baseline promotion evidence
└── tools/                shared generation, protocol, and replay utilities
```

Every bot uses the same small contract. Its directory contains
`submission.json`, an ordered `sources.txt`, maintained implementation files,
`submission_test.cpp`, and the generated paste-ready `submission.cpp`. Optional
data generators and timing probes stay beside the bot that owns them. Historical
bot-specific ledgers normally stay with their source; the two newer campaign
tracks keep their content-addressed cross-tool evidence under
`results/rank_4_jacek_hybrid/` and selected `results/jacek_arena_bfm/` paths.

Generate or verify any registered submission from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs BOT_NAME
node submissions/codingame/tools/generate_submission.mjs BOT_NAME --check
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

The separate historical promotion manifest currently binds `all_depth_proof`.
Validate that frozen contract explicitly with:

```sh
python3 submissions/codingame/tools/promotion_gate.py validate \
  --bot all_depth_proof
```

Do not edit a `submission.cpp` directly. Change the maintained bot source or
data and regenerate it. See [bots/README.md](bots/README.md) for the directory
contract and the shortest path to starting another bot. Shared command details
are in [tools/README.md](tools/README.md).

## Current production and provenance

[`rank_4`](bots/rank_4/README.md) is the current production snapshot: CodinGame
history version 56, agent `6604719`, submission `41114327`, rank 4 of 208 with
score `44.29750553418035` and a 66-24 record. Its generated gameplay submission
remains byte-identical while its training corpus, teacher model, and version
42-44 arena evidence are consolidated in the same directory. The later hybrid
and fresh-arena campaigns did not supersede this incumbent.

[`rank_5`](bots/rank_5/README.md) is the immutable verified predecessor and the
source behind the separately named browser `Rank5DerivedBot`. Its historical
rank 5/206 result remains valid, but it is not the current production result.
The browser adapter uses different rules and fixed work, so it inherits neither
platform rank. Local tournament standings are likewise reported only as
**Local CodinGame-style score**.

## Registered artifact index

The CMake registry and local leaderboard contain the same 22 submission
directories, with every directory represented by one entrant. Leaderboard
inclusion records protocol-faithful local tournament performance and is
independent of live-upload or promotion status.

- **Current production:** [`rank_4`](bots/rank_4/README.md).
- **Verified lineage and prior baseline:**
  [`rank_5`](bots/rank_5/README.md) and
  [`alpha_beta`](bots/alpha_beta/README.md).
- **Rank-4 search experiments:**
  [`rank_4_jacek_hybrid`](bots/rank_4_jacek_hybrid/README.md),
  [`rank_4_exchange`](bots/rank_4_exchange/README.md), and
  [`rank_4_fullturn_bfm`](bots/rank_4_fullturn_bfm/README.md).
- **Other search and learning tracks:**
  [`challenger`](bots/challenger/README.md),
  [`selfplay_nn`](bots/selfplay_nn/README.md),
  [`jacek_nn`](bots/jacek_nn/README.md),
  [`jacek_native_bfm`](bots/jacek_native_bfm/README.md),
  [`jacek_arena_bfm`](bots/jacek_arena_bfm/README.md), and
  [`neural_puct`](bots/neural_puct/README.md).
- **Proof and ordering experiments:**
  [`topology`](bots/topology/README.md),
  [`safe_inward`](bots/safe_inward/README.md),
  [`rebound_proof`](bots/rebound_proof/README.md),
  [`defensive_proof`](bots/defensive_proof/README.md),
  [`shell_proof`](bots/shell_proof/README.md),
  [`all_depth_proof`](bots/all_depth_proof/README.md),
  [`reply_proof`](bots/reply_proof/README.md),
  [`exchange_proof`](bots/exchange_proof/README.md),
  [`frontier_proof`](bots/frontier_proof/README.md), and
  [`conservative_frontier_proof`](bots/conservative_frontier_proof/README.md).

All 22 entrants and their artifact hashes are in
[`benchmarks/codingame_leaderboard/roster.json`](../../benchmarks/codingame_leaderboard/roster.json);
each registered submission directory is represented exactly once.
