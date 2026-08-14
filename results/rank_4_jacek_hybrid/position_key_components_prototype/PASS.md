# Hybrid-private PositionKey component cache: preintegration PASS

This is the immutable DEVELOPMENT-only preintegration receipt for the
safety-corrected PositionKey component-cache prototype. It is a throughput
qualification for integration, not a validation/final strength gate and not
arena evidence. No production source was changed by this experiment; the
candidate is preserved as a checked patch beside this receipt.

## Decision

**PASS the frozen preintegration gate.** The exact safety-corrected candidate
passed source size, key/component safety, make/unmake restoration, table
placement, fixed-work search parity, maintained tests, the full replay audit,
and both conjunctive production-table timing panels. It is eligible for a
separate audited integration step.

An earlier candidate using the shared `papersoccer::detail` namespace and raw
enum-to-array indexing also passed an exploratory timing run. Static review
superseded that identity before qualification. None of its metrics qualify the
candidate recorded here; the table below comes from the safety-corrected
private-namespace identity's single locked timing suite.

## Frozen base and candidate identities

Base commit: `db8aa306a10fc95548babc57234c510e55d74e69`.

| Artifact | Base bytes / SHA-256 | Candidate bytes / SHA-256 |
|---|---|---|
| shared `src/bots/mcts_internal.hpp` | 14,668 / `0a13e89e183666ce89e38d1eded1b26c02eaaba5460ba7e3ede9fda5d5e1dd04` | unchanged and not patched |
| private `submissions/codingame/bots/rank_4_jacek_hybrid/mcts_internal.hpp` | absent | 16,174 / `254ea592b3bca934dbfbbb5ebc838411b49a09ef3ec8b3d8ddc332bb7079b011` |
| maintained `bot.cpp` | 63,107 / `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b` | 63,158 / `439e1b17124ea7c81dd2c3cce66342953d5ed981e19d3947a42ea626ef19f2d2` |
| `sources.txt` | 423 / `c8e29e0fe2917609a126b158ea79c1891ea14cef66b24e688fdd64caf09b77dd` | 461 / `5f2fdce4b375a8fd91c73141d87f2b12a8aa3e61d9b18c46c442b325e51cbdda` |
| generated `submission.cpp` | 94,312 / `2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7` | 95,750 / `47f44e8e62d3aaa2a48f6eea6fca4d17cfbbfd3ff9a5ac01ca84b1e0bf4cca03` |
| `submission_test.cpp` | 39,137 / `ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697` | unchanged |

The generated source delta is +1,438 ASCII characters, within the frozen
+1,800 limit, and 95,750 is below the contest limit of 99,999.

The complete private integration patch is
`results/rank_4_jacek_hybrid/position_key_components_prototype/private_integration.patch`:
24,814 bytes, SHA-256
`59640f953414132e90b31b2a6e2a2dfa81a3e7dbaf83e1d236befa6f2a1f2997`.
It includes the private header, `bot.cpp`, `sources.txt`, and the generated
source; `git apply --check` passed against the frozen base. The patch does not
modify the shared MCTS header.

## Exact call-site map

The private header owns an immutable edge-component vector and an immutable
three-component vector per topology vertex. They are populated only after both
player adjacency tables have assigned the final contiguous edge indexes.

| Category | Meaning | Precomputation | SearchPosition/bot consumers |
|---:|---|---|---|
| 1 | used edge | `edge_key_components_` at private-header lines 145-147 | constructor line 276, `make_move` line 377, `unmake_move` line 427 |
| 2 | visited vertex | vertex slot 0 at lines 149-153 | constructor line 286, first-visit make line 380, matching unmake line 430 |
| 3 | current ball | vertex slot 1 at lines 149-153 | dynamic-key add/remove at line 459 |
| 4 | player to move | constexpr exhaustive `switch` at lines 49-54 | dynamic-key add/remove at line 460 |
| 5 | status | constexpr exhaustive `switch` at lines 56-63 | dynamic-key add/remove at line 461 |
| 6 | turn-boundary ball tag | vertex slot 2 at lines 149-153 | `CompleteTurnSearch::boundary_key` in `bot.cpp` lines 450-453 |

`bot.cpp` aliases its local `detail` name to
`::papersoccer::hybrid_detail` at line 26. Rank 4 continues to use the shared
`papersoccer::detail`; therefore the maintained parity test can include both
engines in one translation unit without type or pragma-once coupling.

## Implementation invariants

- `PositionKey`, `mix_position_key`, `position_key_component`, and XOR
  composition are byte-for-byte unchanged.
- Topology vertex and edge indexes are contiguous and stable before the cache
  vectors are populated. Accessors use the same index that the original hash
  call consumed.
- The vertex tuple mapping is fixed: slot 0/category 2, slot 1/category 3, and
  slot 2/category 6.
- Category 1 is XORed in construction/make/unmake at exactly the original
  transition points.
- Category 2 is added only on a first visit and removed only when undoing that
  first visit, preserving the original `destination_was_visited` condition.
- Dynamic categories 3/4/5 are removed and restored through the same XOR
  involution as before.
- Player and status components use exhaustive switches with a safe zero-key
  fallback. There is no unchecked enum-derived array index.
- Boundary keys still equal `position_key XOR category_6(ball_vertex)`; only
  the source of the exact 128-bit category-6 value changed.
- SearchPosition and Undo layouts are unchanged. The extra immutable storage
  is owned by the per-search topology, and timing includes topology creation.
- The generated source is derived from the private header through
  `sources.txt`; the shared header remains available to every other bot.

## Exact correctness evidence

Generator creation and `--check` both passed on the candidate identity above.

The archived component/key harness is
`position_key_harness.cpp`, 13,536 bytes, SHA-256
`879125499604c0dc0f644ef709b4cf4bfc74d2cf99aab96b75a6573198c7322c`.
It performed all of the following:

- exhaustively compared every cached edge and all three cached vertex
  components with direct `position_key_component` values;
- checked both Player and all three Status switch results;
- covered 20 topology/rules combinations: dimensions 1x1, 2x3, 5x4, 8x10,
  and 12x7 crossed with both GoalRule values and both BlockedRule values;
- ran 2,560 deterministic random make/unmake operations over that matrix and
  restored every root key;
- checked 399,450 standard construction, random-walk, and depth-six exhaustive
  observations against direct full-key reconstruction;
- checked boundary keys and actual small/production TT and evaluation-table
  bucket placement for every standard observation.

Control and candidate emitted the identical 96-byte result
`states=1004 events=399450 digest=5760913385319248740:1599381397890498060 vertices=105 edges=316`;
both streams have SHA-256
`ae78a18c78ccd50207fd1f4b6beb55b4405c07cc4498c40ddc549038a5d1fa88`.

The namespace-aware fixed-work harness is `fixed_work_harness.cpp`, SHA-256
`447c7fc7999e8357cffcaa79b437142584b90b799936aecadc722c1f50dfd2ce`.
Across 1,004 deterministic live states, proof masks 0 and 7, and node budgets
1/16/64/256/1,024, all 10,040 decisions had byte-identical encoded actions,
root scores, and every SearchStats field. Control and candidate stdout were
1,067,064 bytes with SHA-256
`99f09e979bc5e4b25e5878880d9b7a9acd83e3a6870adba2b8e3be9fc306bd42`;
stderr was 52 bytes with SHA-256
`f20d99fe7924f98b29c366e2494da2ac2681641d74b32e0c4fcdeb85371e327a`.

Maintained test evidence on the exact private candidate:

- all 18 submission tests passed; test source SHA-256
  `ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697`;
- Rank-4 compatibility passed with 19 exact decisions, 13 traversal-only
  deltas, and zero action deltas; parity source SHA-256
  `464a1b8b7dd5e574f9e9103e2848d2467839f086c72945845430699237ba6ca3`;
- cheap replay audit passed;
- full replay audit covered all 511 unique activations: 511 exact maxima,
  zero unresolved, zero capped, zero displaced, 2,145,467 expanded states,
  948,118 maximum prefix states, and 1,293,929 completed actions; replay-audit
  source SHA-256
  `52ff8dc33dfffd0872cb45feb94164b06041df0ab36070131f9fe241db5f6b81`.

## Frozen production-table timing gate

The exact microbenchmark harness is the existing
`gates/sole_legal_edge_clock/prototype/microbench.cpp`, SHA-256
`2007e0daee4282ee6924aa43bbb8ffec3844a6ab845d2db3e5e79c6634837510`.
The archived locked runner is `run_microbench.py`, SHA-256
`09fa52e37b2565d819d706778c9fe3013682071c43c65a0aaf0e10c1ca4a1a3e`.
The canonical JSON is `timing.json`, SHA-256
`b91990bad9d51f2f4b1ee01c3a88e5e22ad32e8ea5fb8bb20c0755b46959ad1d`.

The safety-corrected identity was measured exactly once. The runner held an
exclusive lock on `/tmp/rank4-hybrid-prototype-benchmark.lock` for both panels;
lock wait was 0.000003 seconds and total elapsed time was 74.347 seconds. Each
panel used 30 warm-up pairs and 300 measured pairs with alternating process
order. Builds used Apple clang 21.0.0, C++20, `-O3 -DNDEBUG`. Control and
candidate binary SHA-256 values were respectively
`0504a6c2099371452cfc7974a14f81bb33b9943d548d4e382bb78d15d0f17166`
and
`7be3796b9e4cbf58ff6750977ced86d49a848b846b3c0fce5121c629829c36a2`.

Frozen limits were candidate/control median ratio at most 0.99 and p99 ratio
at most 1.005 in both panels.

| Panel | Exact signature | Control median / p99 (ns) | Candidate median / p99 (ns) | Median ratio | p99 ratio | Result |
|---|---|---:|---:|---:|---:|---|
| forced-heavy | `(50000,15016,48,12774,0,8634)` | 18,237,729 / 19,687,708 | 17,816,458.5 / 19,331,042 | 0.976901154 | 0.981883823 | pass |
| mixed | `(50000,4109,2,163,4,4753)` | 89,142,458 / 94,483,084 | 88,170,395.5 / 93,274,166 | 0.989095404 | 0.987204927 | pass |

## Data boundary

This prototype read no validation or final opening bank, played no whole game,
ran no arena match, and performed no browser or live-arena operation. It used
only deterministic locally generated states, fixed-work decisions, maintained
tests, replay-prefix audits, and two fixed 50,000-node microbenchmark states.
No production candidate file was edited; only this evidence directory was
added to the shared worktree.
