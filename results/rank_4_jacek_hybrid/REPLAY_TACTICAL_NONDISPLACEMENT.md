# Tracked replay tactical non-displacement audit

This is a local qualification audit of the inherited, tracked
`rank_4_jacek_hybrid/replay_book.hpp` only. It does not inspect an arena replay,
a protected bank, `matches.json`, or either sealed validation/final opening
bank. It does not qualify the source by itself and does not alter production
behavior.

The independent harness reconstructs every eligible activation prefix from the
24 tracked replay paths, checks lookup and legal atomic application, and
deduplicates one identical overlap. The frozen count is 512 eligible entries
and 511 unique `(player, prefix)` activations.

For each correction, the harness applies the exact five-level mover-relative
ordering:

- `+2`: immediate terminal mover win;
- `+1`: nonterminal handoff whose opponent root rebound component is `Loss`;
- `0`: nonterminal handoff whose opponent component is `Unknown`;
- `-1`: nonterminal handoff whose opponent component is `Win`;
- `-2`: immediate terminal mover loss.

It then finds the maximum category over every legal complete rebound turn. A
root-component `Win` or `Loss` fixes the appropriate terminal maximum exactly.
For root `Unknown`, exhaustive DFS enumerates complete turns. Because root
`Unknown` excludes both immediate terminal categories, finding `+1` is an exact
upper-bound stop. Any per-prefix or global state cap returns failure, as does
any correction category unequal to the maximum.

## Completed evidence

Command:

```text
build/papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit \
  --full --state-cap 2000000 --global-state-cap 100000000
```

Result:

```text
replay_tactical_audit mode=full replay_paths=24 eligible_entries=512 unique_activations=511 audited=511 exact_maxima=511 unresolved=0 capped=0 displaced=0 correction_categories=-2:0,-1:0,0:489,1:4,2:18 maximum_categories=-2:0,-1:0,0:489,1:4,2:18 expanded_states=2145467 maximum_prefix_states=948118 completed_actions=1293929 state_cap=2000000 global_state_cap=100000000
```

Conclusion: no tracked correction displaces a higher exact tactical category.
All 511 unique activation maxima completed without a cap. The exact correction
and maximum distributions are identical: 489 category-zero, four category
`+1`, and 18 immediate wins; neither side contains a negative-category action.
The worst activation expanded 948,118 states, below the strict 2,000,000-state
per-prefix cap.

The focused full CTest passed in 52.86 seconds on the campaign machine. A cheap
contract test also reconstructs all 511 activations and validates their legal
corrections without performing the expensive exhaustive maxima.
