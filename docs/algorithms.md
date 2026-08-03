# Rules and algorithms

This document describes the game contract and the decision systems behind the
native and browser opponents. Performance measurements and promotion decisions
are kept in [Experiments](experiments.md); build and artifact checks are in
[Reproducibility](reproducibility.md).

## Rules contract

The normal demo uses Kurnik-style geometry with an explicit configurable rule
contract:

- Coordinates are zero-based. The field spans `x in [0,8]`, `y in [1,11]`;
  goal nodes are on rows `y=0` and `y=12`.
- Play starts at `(4,6)`. Player One attacks north and Player Two attacks south.
- A move follows one of eight neighboring directions. Segments are undirected
  and cannot be reused.
- Movement along the outside boundary lines is forbidden.
- Each goal contains `(3,y)`, `(4,y)`, and `(5,y)`. Goal entries are legal only
  from the three-point mouth, and goal-post segments still behave as walls.
- From `(3,1)` or `(5,1)`, the only legal north-goal destination is `(4,0)`;
  `(4,1)` can enter any north-goal node. The south end is symmetric.
- Under the demo's `OpponentGoalOnly` rule, entering the opponent's goal wins.
- Landing on a previously visited point or a wall boundary point grants a
  rebound, so the same player moves again. The open center of a goal mouth is
  not treated as boundary.
- Under `PlayerToMoveLoses`, a player with no legal move at turn start loses.

`RulesConfig.width` and `height` are field spans, not maximum coordinate values.
The default height is `10`, while the field boundaries are rows 1 and 11 and
the goal rows are 0 and 12.

The authentic CodinGame submission used a different explicit contract:
`OwnGoalsAllowed` and `MoverLoses`. That distinction is one reason its browser
adaptation cannot inherit the contest result. See
[Rank5DerivedBot](#rank5derivedbot).

## RandomBot

`RandomBot` chooses uniformly from the authoritative legal-move order using a
seeded generator. It is useful as a reproducible smoke opponent and as an
exploration source, not as a strength baseline. A complete run is reproducible
from the same starting state, seed, implementation, and played moves.

## Possession-aware alpha-beta

`AlphaBetaBot` is deterministic adversarial search. It enumerates a bounded game
tree, assuming both players choose their strongest reply, and prunes branches
that cannot change the root decision. A transposition table reuses positions
reached through different move orders.

### Turn depth and exact outcomes

Paper soccer cannot treat one edge as one adversarial turn. A rebound leaves
the same player on move, so search preserves the maximizing or minimizing
perspective and does not consume possession depth until control passes. Within
its physical-ply horizon, even depth one follows same-player rebound sequences
through a goal, trap, or handoff.

Terminal values are Player-One-oriented and close to `+1,000,000` for a Player
One win or `-1,000,000` for a Player Two win. The physical move count prefers
faster wins and delayed losses. Non-terminal values are clamped to
`[-100,000, 100,000]`, leaving a clear band between heuristic estimates and
proven outcomes.

The readable hand-written evaluator starts with these weights:

| Feature | Weight | Purpose |
| --- | ---: | --- |
| Direct goal available | 50,000 | Prefer a goal in the current possession |
| Unused-edge goal-distance difference | 120 | Compare remaining routes to goal |
| Vertical ball progress | 80 | Reward progress toward the relevant goal |
| Possession-preserving choices | 35 | Value rebound and boundary continuations |
| Legal mobility | 20 | Value current-player options |
| Forward choices | 10 | Reward legal moves aimed toward goal |
| Center alignment | 6 | Prefer useful central access |
| Tempo | 15 | Give a small value to retaining possession |

Goal distance is breadth-first search over unused edges. These are transparent
starting weights, not learned constants; arena measurements, not the numbers
alone, determine whether a change is retained.

### Budgets and state reuse

Iterative deepening keeps the last completed result if a deeper iteration hits
the node or optional time limit. A zero `max_time_ms`, including the default,
disables the wall-clock cutoff and preserves fixed-work determinism. The normal
profile uses:

- up to six possession handoffs;
- at most 100,000 visited nodes for the decision;
- 65,536 compact transposition entries;
- a soft horizon after 12 physical edges; and
- an absolute 512-edge recursion guard.

At the soft horizon, positions with multiple choices are evaluated, but forced
single-edge lines continue. This bounds unusually large same-possession trees
without discarding long forced combinations. If depth one cannot finish, a
deterministic legal fallback orders immediate outcomes, rebounds, progress,
center access, and child mobility.

The compact search position maintains a two-word incremental key covering used
edges, visited vertices, ball position, side to move, and status. Entries store
depth, a score bound, and the best move. Root alternatives are labelled
`Exact`, `Lower`, or `Upper`; a pruned alternative may be a bound rather than an
exact value.

```cpp
papersoccer::AlphaBetaConfig config{
    .max_turn_depth = 6,
    .max_nodes = 100'000,
    .transposition_table_entries = 65'536,
    .max_search_plies = 12,
    .max_time_ms = 0,
};
papersoccer::AlphaBetaBot bot(config);
papersoccer::Move move = bot.choose_move(state);
const auto &stats = bot.last_search_stats();
```

Diagnostics include attempted/completed depth, visited and evaluated nodes,
terminal nodes, alpha-beta cutoffs, transposition activity, physical-horizon
cutoffs, maximum physical ply, root score, principal variation, root
alternatives, and budget exhaustion.

## Monte Carlo tree search

MCTS estimates move strength by sampling possible futures instead of fully
enumerating the tree. Each iteration performs:

1. **Selection:** follow promising retained children.
2. **Expansion:** allocate one unexplored legal move.
3. **Frontier evaluation:** optionally run a bounded tactical proof probe, then
   use the configured rollout policy if no winner is proven.
4. **Backpropagation:** add the result to every visited node.

Selection uses UCT:

```text
mean value + exploration * sqrt(log(parent visits) / child visits)
```

The default exploration value is approximately `sqrt(2)`. Values always use a
Player-One perspective: a Player One win is `+1`, and a Player Two win is `-1`.
Player One nodes prefer larger values and Player Two nodes prefer smaller ones.
After a rebound, the decision owner comes from `GameState::to_move`; the search
never assumes players alternate after every edge.

An immediate legal win is taken before sampling. Otherwise, the final root move
is selected primarily by visit count, then by mean result from the root
player's perspective, then by the engine's legal-move order.

### Rollout and solver behavior

The default `Tactical` rollout policy takes immediate wins, follows forced
moves without consuming an unnecessary random number, avoids terminal
self-traps, and avoids handing over an immediate win when a safe alternative
exists. MCTS-Solver propagation carries proven winners through fully explored
branches, avoids proven losing children, and stops early when the root winner
is known.

`Uniform` retains the original uniform rollout and selection behavior as a
reproducible historical reference. `RolloutOnly` is the shipped leaf policy.

### Experimental tactical quiescence

`TacticalQuiescence` is an opt-in leaf policy, not the default. At a
non-terminal MCTS frontier it runs deterministic bounded adversarial proof
search before the unchanged Tactical rollout. It consumes no rollout random
numbers and does not replace any configured MCTS iterations.

A frontier is noisy when it contains an immediate goal or trap, a single legal
move, a same-player rebound, a move that leaves one forced reply, or a defensive
or escape context identified by one-ply reply lookahead. Once a position is
noisy, all legal moves are searched in authoritative order so a quiet escape is
not omitted.

Proof results are absolute winners or Unknown. One child proven for the player
to move proves a win; a loss is proven only if every legal child is proven for
the opponent. Quiet positions, cutoffs, and unresolved alternatives remain
Unknown, preventing an incomplete search from inventing a proof. The compact
position is made and unmade around every probe. The evaluation in
[Experiments](experiments.md#experimental-tactical-quiescence) did not justify
promotion, so `RolloutOnly` remains the engine and site default.

### Configuration and reuse

`MctsConfig` defaults to 2,000 new iterations per move, Tactical rollouts, tree
reuse, a 65,536-node bound, and `RolloutOnly`. Experimental quiescence defaults
to depth 8 and 256 tactical nodes per probe. Iterations are fixed work rather
than a time limit.

`MctsBot` re-roots retained work through explored moves, compacts unreachable
branches, and rebuilds deterministically when the supplied state is unrelated
or the played branch was not expanded. At the node bound, iterations continue
as rollouts without allocating more nodes. A standalone `MctsSearch` is
incremental: with the same state and seed, twenty 100-iteration calls preserve
the same random stream as one 2,000-iteration call.

```cpp
papersoccer::MctsConfig config{
    .seed = 12345,
    .iterations = 2000,
    .rollout_policy = papersoccer::MctsRolloutPolicy::Tactical,
    .reuse_tree = true,
    .max_nodes = 65536,
    .leaf_policy = papersoccer::MctsLeafPolicy::TacticalQuiescence,
    .quiescence_max_depth = 8,
    .quiescence_max_nodes = 256,
};
papersoccer::MctsBot bot(config);
papersoccer::Move move = bot.choose_move(state);

papersoccer::MctsSearch search(state, 12345);
search.run_iterations(100);
search.run_iterations(100);
papersoccer::Move incremental_move = search.best_move();
```

Search statistics report iterations, allocated nodes, rollout plies, total and
reused root visits, tree depth, proven nodes/winner, rebuilds, saturation, root
value, and separate tactical counters. Timings are measured by the arena rather
than stored in deterministic search statistics.

## JacekInspiredBot

`JacekInspiredBot` is an independently trained normal-demo opponent inspired by
the input representation in
[Jacek Dermont's article](https://www.codingame.com/playgrounds/157341/inputs-for-neural-networks-for-the-board-games/paper-soccer).
It is not a copy of his bot: the article does not publish its trained weights,
and this project does not compile, include, or load a contest submission.

Player Two positions are rotated so the mover always attacks upward. The model
encodes 316 drawn edges and eight one-hot true-turn-distance buckets for each of
105 vertices. A `1156 -> 32 -> 32 -> 1` network provides leaf values to the
existing possession-aware alpha-beta search. The first layer iterates only over
active binary inputs rather than multiplying a dense 1,156-value vector.

The checkpoint was trained from scratch under normal demo rules on mover-
relative soft alpha-beta teacher scores. It is intentionally restricted to the
standard 8x10 demo contract and rejects other boards or rule combinations. The
browser profile uses depth 6, 20,000 nodes, and no time limit. Exact features,
corpus/model hashes, held-out metrics, gates, and training commands are
canonical in the [model record](../models/README.md).

The bot's measurements do not reproduce Jacek's unpublished contest strength
and do not improve or modify any CodinGame submission.

## Rank5DerivedBot

`Rank5DerivedBot — 50k demo profile` adapts complete-turn search from the
maintained rank-5 source to the normal browser rules. It is not the authentic
ranked entrant.

| Property | Verified CodinGame `rank_5` | `Rank5DerivedBot` demo |
| --- | --- | --- |
| Rules | `OwnGoalsAllowed`, `MoverLoses` | `OpponentGoalOnly`, `PlayerToMoveLoses` |
| Work limit | 650 ms first response, 130 ms later; 3M-node guard | 50,000 nodes; no wall-clock cutoff |
| Turn depth | Up to 32 | Up to 32 |
| Tables | 262,144 transposition / 131,072 evaluation | 65,536 / 32,768 |
| Learned value | 15% replay-trained blend | 0% blend after the demo gate |
| Replay corrections | Exact full-history replay book enabled | Disabled |
| Evidence | Historical CodinGame rank 5/206 | Demo-rule gates and Wasm timing only |

The adapter includes
[`bot.cpp`](../submissions/codingame/bots/rank_5/bot.cpp) with its main function
disabled and constructs `CompleteTurnSearch` directly. It validates the returned
complete possession. Because the public `Bot` API returns one edge, later
rebound edges are cached and reused only if the next state is byte-for-byte
equivalent in rules, ball, player, status, path, used edges, and visit counts.

Its diagnostics separate fresh searches from cached continuations and include
requested/visited nodes, completed/attempted depth, root score, budget
exhaustion, planned action length, current edge index, and remaining cached
edges. The UI also records the original submission ID and SHA-256 as provenance;
that metadata is not a claim that the adapted decision has a contest rank.

The authentic artifact, arena result, and generation contract live in the
[rank-5 submission record](../submissions/codingame/bots/rank_5/README.md).
Demo-only evaluator and latency evidence is in
[Experiments](experiments.md#rank5derived-demo-gates).
