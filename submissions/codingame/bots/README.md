# Bot catalog

All 24 standalone bots are listed here by strategy. Start with
**[Quantized alpha-beta](quantized_alpha_beta/README.md)** for the strongest
recorded platform result, or choose a simpler baseline to study the search.
Every source link opens a self-contained C++20 program.

## Alpha-beta and learned evaluation

| Bot | Approach | Standalone source | Results and history |
| --- | --- | --- | --- |
| [Quantized alpha-beta](quantized_alpha_beta/README.md) | Complete-turn alpha-beta with a four-bit value network | [C++](quantized_alpha_beta/submission.cpp) | [Rank 2/212, 9 October 2026](quantized_alpha_beta/README.md#results-and-history); [source history](quantized_alpha_beta/history/2026-09-27/README.md) |
| [Alpha-beta baseline](alpha_beta/README.md) | Complete-turn alpha-beta, replay-trained value anchor, and exact-history corrections | [C++](alpha_beta/submission.cpp) | [Historical rank 8/206](alpha_beta/README.md#results); [history](alpha_beta/experiments/README.md) |
| [Replay-guided alpha-beta](rank_5/README.md) | Complete-turn alpha-beta with a compact replay-value model and verified replay book | [C++](rank_5/submission.cpp) | [Historical rank 5/206](rank_5/README.md#results); [history](rank_5/EXPERIMENTS.md) |
| [Teacher-residual alpha-beta](../../../docs/bots/rank-4.md) | Replay-guided search with a bounded linear correction learned from deeper teacher searches | [C++](rank_4/submission.cpp) | [Historical rank 4/208, 66–24](../../../docs/bots/rank-4.md#results); [history](rank_4/EXPERIMENTS.md) |
| [Rebound-proof challenger](challenger/README.md) | Alpha-beta with exact rebound-component proofs and narrowly matched replay corrections | [C++](challenger/submission.cpp) | [Recorded search ablation](challenger/README.md#results); [history](challenger/EXPERIMENTS.md) |
| [Self-play value search](selfplay_nn/README.md) | Alpha-beta with a small self-play value residual and the verified replay book | [C++](selfplay_nn/submission.cpp) | [Local wins; no live improvement](selfplay_nn/README.md#results); [history](selfplay_nn/EXPERIMENTS.md) |
| [Search-residual alpha-beta](../../../docs/bots/search-residual.md) | Replay-guided alpha-beta with a bounded residual fitted to deeper search targets | [C++](jacek_nn/submission.cpp) | [Local and live comparisons](../../../docs/bots/search-residual.md#results); [history](jacek_nn/EXPERIMENTS.md) |

## Neural and best-first search

| Bot | Approach | Standalone source | Results and history |
| --- | --- | --- | --- |
| [Neural PUCT](neural_puct/README.md) | Edge-level policy/value tree search with four-bit weights and no inherited replay book | [C++](neural_puct/submission.cpp) | [Fixed-work and clock comparisons](neural_puct/README.md#results); [history](../../../docs/archive/neural-puct/experiments.md) |
| [Full-turn value minimax](rank_4_fullturn_bfm/README.md) | Best-first minimax over complete turns using the teacher-residual evaluator | [C++](rank_4_fullturn_bfm/submission.cpp) | [24–82 equal-clock comparison](rank_4_fullturn_bfm/README.md#results); [history](rank_4_fullturn_bfm/EXPERIMENTS.md) |
| [Native neural minimax](../../../docs/bots/native-neural-minimax.md) | Independent complete-turn minimax with a compact neural evaluator | [C++](jacek_native_bfm/submission.cpp) | [Historical rank-5 checkpoint](../../../docs/bots/native-neural-minimax.md#results); [history](jacek_native_bfm/EXPERIMENTS.md) |
| [Fresh-corpus neural minimax](../../../docs/bots/fresh-neural-minimax.md) | Independent minimax trained from a new scratch corpus | [C++](jacek_arena_bfm/submission.cpp) | [Offline comparisons and live safety](../../../docs/bots/fresh-neural-minimax.md#results); [history](jacek_arena_bfm/EXPERIMENTS.md) |
| [Compact value minimax](compact_value_bfm/README.md) | Sparse value-only best-first minimax with 6,301 inputs and quantized weights | [C++](compact_value_bfm/submission.cpp) | [Build fixture; no qualified model](compact_value_bfm/README.md#results); [history](compact_value_bfm/EXPERIMENTS.md) |

## Proofs and move ordering

These bots isolate tactical ideas while retaining a reference engine. Their
local comparisons help explain the search; a buildable source alone does not
establish a stronger live bot.

| Bot | Approach | Standalone source | Results and history |
| --- | --- | --- | --- |
| [Teacher-residual proof hybrid](../../../docs/bots/rank-4-hybrid.md) | Selective rebound and exchange proofs with mover-relative ordering | [C++](rank_4_jacek_hybrid/submission.cpp) | [Validation passed; final failed](../../../docs/bots/rank-4-hybrid.md#results); [history](rank_4_jacek_hybrid/EXPERIMENTS.md) |
| [Teacher-residual exchange proof](rank_4_exchange/README.md) | Exact rebound proofs through the first reply and counterturn | [C++](rank_4_exchange/submission.cpp) | [8–10 clock comparison](rank_4_exchange/README.md#results); [history](rank_4_exchange/EXPERIMENTS.md) |
| [Topology ordering](topology/README.md) | Prioritize edges that interrupt short rebound-goal corridors | [C++](topology/submission.cpp) | [Historical comparison protocol](topology/README.md#results); [history](topology/EXPERIMENTS.md) |
| [Safe inward ordering](safe_inward/README.md) | Prefer nonterminal inward rebound edges near the mover’s own goal | [C++](safe_inward/submission.cpp) | [Development rejection](safe_inward/README.md#results); [history](safe_inward/EXPERIMENTS.md) |
| [Rebound-component proof](rebound_proof/README.md) | Prove current-turn scoring paths and blocked rebound-component losses | [C++](rebound_proof/submission.cpp) | [Validation color-guard rejection](rebound_proof/README.md#results); [history](rebound_proof/EXPERIMENTS.md) |
| [Defensive rebound proof](defensive_proof/README.md) | Apply exact leaf proofs when the opponent is to move | [C++](defensive_proof/submission.cpp) | [Frozen local comparator](defensive_proof/README.md#results) |
| [Goal-shell rebound proof](shell_proof/README.md) | Apply exact leaf proofs near the attacking goal shell | [C++](shell_proof/submission.cpp) | [Frozen local comparator](shell_proof/README.md#results) |
| [All-depth rebound proof](all_depth_proof/README.md) | Apply exact rebound proofs at every complete-turn search boundary | [C++](all_depth_proof/submission.cpp) | [Experimental comparison evidence](all_depth_proof/README.md#results); [history](all_depth_proof/EXPERIMENTS.md) |
| [First-reply proof](reply_proof/README.md) | Extend exact rebound proofs to the opponent’s immediate reply | [C++](reply_proof/submission.cpp) | [Development evidence](reply_proof/README.md#results); [history](reply_proof/EXPERIMENTS.md) |
| [First-exchange proof](exchange_proof/README.md) | Extend exact rebound proofs through the opponent’s reply and the root player’s counterturn | [C++](exchange_proof/submission.cpp) | [Development evidence](exchange_proof/README.md#results); [history](exchange_proof/EXPERIMENTS.md) |
| [Handoff-frontier proof](frontier_proof/README.md) | Add a small safe-handoff evaluation feature after an inconclusive exact proof | [C++](frontier_proof/submission.cpp) | [Sensitivity comparison](frontier_proof/README.md#results); [history](frontier_proof/EXPERIMENTS.md) |
| [Conservative frontier proof](conservative_frontier_proof/README.md) | Use half the handoff-frontier evaluation weight while preserving exact proofs | [C++](conservative_frontier_proof/submission.cpp) | [Sensitivity comparison](conservative_frontier_proof/README.md#results); [history](conservative_frontier_proof/EXPERIMENTS.md) |

## Working with a bot

The 22-bot [local leaderboard](../../../benchmarks/codingame_leaderboard/README.md)
has a frozen roster. Compact value minimax and Quantized alpha-beta are
additional bots, with build and test coverage outside that tournament.
Historical directory IDs such as `rank_4` remain stable so generators,
benchmarks, and recorded evidence continue to agree.

Most bot directories share these files:

| File | Purpose |
| --- | --- |
| `bot.cpp` and local headers | Maintained strategy and protocol implementation |
| `submission.json`, `sources.txt` | Generation settings and ordered source inputs |
| `submission.cpp` | Generated standalone C++20 program |
| `submission_test.cpp` | Focused artifact and rules checks |
| `README.md`, optional `EXPERIMENTS.md` | Design, results, and chronological history |

Quantized alpha-beta retains an exact standalone snapshot and identity test
instead of generated inputs. Compact value minimax uses its Python exporter.
For the remaining bots, check the generated source from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs BOT_ID --check
```

Edit maintained inputs before regenerating a bot. Keep frozen source snapshots,
models, reports, and game records unchanged; a new experiment needs its own
identity and output directory.

## Adding a bot

1. Start with the nearest strategy, keeping copied historical evidence out of
   the new directory.
2. Update its maintained source, generation settings, tests, and design notes.
3. Register the build in the root `CMakeLists.txt`; select the research registry
   for experimental entrants. Do not rewrite the frozen leaderboard roster.
4. Check source generation, standalone compilation, tests, and the contest
   protocol before making any performance claim.
5. Add a catalog entry with its readable name, source, approach, and measured
   status.

See [shared tools](../tools/README.md) and
[development](../../../docs/development.md) for the complete workflow.
