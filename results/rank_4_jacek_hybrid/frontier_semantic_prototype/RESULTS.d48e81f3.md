# Safe-handoff frontier-width weight-10 prototype

Archived on 2026-08-14 as a correctness/source packet against exact rollback
commit `4c9afa70a36183fc1451e4335b7251651a7b2791`. The archive operation changes
no production, CMake, bot, generated submission, or existing campaign document.
No frontier timing benchmark or whole game has run. No heldout, VALIDATION, or
FINAL bank has been read.

## Exact semantic change

The single semantic parameter is `kSafeHandoffFrontierWeight = 10`; no other
weight was built or sampled. On an evaluation-cache miss at an exact-proof leaf
boundary only:

1. The already-paid rebound-component scan marks each fresh endpoint in its
   existing generation vector, deduplicating endpoints reached by multiple
   component arcs.
2. Its existing opponent-adjacency check counts the unique fresh endpoints
   with a legal reply after the incoming edge is consumed. There is no second
   topology or component scan.
3. Exact Win and Loss retain their unchanged mate returns before evaluation.
4. Exact Unknown evaluates normally, including replay and teacher residual,
   then adds
   `player_sign(to_move) * count * 10 * (100 - replay_blend) / 100`, clamps to
   the existing evaluation range, and stores the adjusted score in the existing
   evaluation cache.

Root, ply-one, and ply-two proof callers do not request the count. For those
null-output calls, the refrozen candidate preserves the baseline boolean scan:
it does not mark fresh endpoints or increment a unique count, and Loss remains
`!safe_handoff`. Proof-disabled evaluation is unchanged. A fresh endpoint with `N` unused
incident edges is safe for every possible incoming component arc exactly when
`N >= 2`, which makes the endpoint generation mark independent of arc order.

## Frozen identities and source projection

| Artifact | Rollback control | Weight-10 candidate |
|---|---|---|
| `bot.cpp` | 63,107 / `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b` | 64,521 / `408adc5288674550cc08274aec74380074117e32ad8f6915c7e39badc8dfba98` |
| generated `submission.cpp` | 94,312 / `2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7` | 95,272 / `08d0c0859ef8a197f8bfdd89afb048bec41c3a888228433b85991cd937882550` |

The generated candidate is ASCII, passes generator `--check`, is 960
characters larger than control, and leaves 4,727 characters below the 99,999
limit. Its small `PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING` hook is part of the
frozen bot/generated-source identity.

Unchanged bindings:

- `sources.txt`: 423 bytes,
  `c8e29e0fe2917609a126b158ea79c1891ea14cef66b24e688fdd64caf09b77dd`;
- shared MCTS header: 14,668 bytes,
  `0a13e89e183666ce89e38d1eded1b26c02eaaba5460ba7e3ede9fda5d5e1dd04`;
- maintained submission test: 39,137 bytes,
  `ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697`;
- `CMakeLists.txt`: 52,948 bytes,
  `10ff6cad72c75d2c06cac56831defe16e2d7b34a5cbc5f948dfcdf74d9559bdd`;
- retained comparison wrapper: 2,687 bytes,
  `8f2842afff80fd6054b1ea123ae3852c985aaff453040638e72c2ca5852a5223`.

The packet also preserves the exact rollback bot bytes as
`control_bot.34b1dd62.cpp`, 63,107 bytes, SHA-256
`34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`,
so future gates can compile a dedicated control wrapper without relying on an
older archived control. The generated control remains commit/hash-bound above.

## Reproducible transforms

The repo-relative, tool-native patch is stored as deterministic base64 in
`frontier_weight10.408adc52.apply_patch.base64`, 10,085 bytes, SHA-256
`9860eea274fdb45061404622443a973bde533782cda9594488ddf53c6a36471e`.
Decoding yields the exact 7,561-byte raw patch with SHA-256
`c6c2a8c1d998a0b0e9f3eac9f0326a56a30a8bf0377bb54be6a3a61b8d59055e`.
Decode with `base64 -D` on macOS or `base64 --decode` on GNU systems, verify
the decoded byte count and hash above, then pass those exact decoded bytes to
the actual `apply_patch` tool at repository root. This was verified in a fresh
detached `4c9afa7` clone. It changed only bot and generated submission. The
resulting Git diff SHA-256 was
`1bafcad34233394a3c88923a2399a0dcfee7db8021110d09bdeb1a38c1377cb6`,
exactly matching the frozen prototype reference diff; the resulting source
hashes are the candidate identities above.

The Git-format provenance patch is stored as deterministic base64 in
`frontier_weight10.408adc52.git.patch.base64`, 12,293 bytes, SHA-256
`9973733c27c78942666caf6b1e4c77322449b2cbbbd339d7fa15eccb8572171f`.
Decoding yields the exact 9,218-byte raw Git patch with SHA-256
`b9dbb178ef681bc8136bc194414183ade2cc382f9c2b442a95c2d9450726b145`.
After the same decode-and-verify step, `git apply --check` and `git apply`
both pass at `4c9afa7` in a fresh clone.

## Correctness evidence

Maintained evidence on the exact candidate:

- generated-source suite: 18/18;
- bounded-delta Rank-4 parity: 19 exact decisions, 13 traversal-only deltas,
  zero action deltas;
- cheap replay tactical audit: all 511 unique activations accounted for;
- generator creation and `--check`: passed.

The original focused source remains byte-frozen as
`frontier_semantic_test.da72af45.cpp`, 20,466 bytes, SHA-256
`da72af458185f74acfb34497fee2622cb17d8fee6ebf3cfc5eea2aadde7f85d7`.
It is raw prototype provenance: its historical `../submissions/...` include is
not correct from this archive directory and must not be silently changed.

The separately bound archive-runnable Stage-0 test is
`frontier_semantic_test_archive.e50e90db.cpp`, 24,290 bytes, SHA-256
`e50e90db2ea219ef1f21fa663154f7bc2d1d246c15f59ee5cc4aa3da7c68f533`.
It statically registers literal weight 10, uses literal 10 in expected-delta
calculations, and passes 8/8 normally and 8/8 under AddressSanitizer plus
UndefinedBehaviorSanitizer. It retains all frozen 7/7 coverage and adds a
public-transcript witness with at least 12 used edges that:

- remains exact-proof Unknown with a nonempty safe frontier;
- records exactly one teacher-residual evaluation and proves the residual
  correction is nonzero versus residual weight zero;
- proves the exact mover-relative frontier delta on top of that active
  residual evaluation;
- proves the adjusted value is returned by the next evaluation-cache hit
  without another proof or residual evaluation.

The combined focused coverage includes 648 exhaustive local trit/rotation
cases, 32 goal masks, an independent public-rules graph oracle, converging-arc
endpoint deduplication, replay blends 0/15/100 for both movers, four mate-bypass
witnesses, one-scan accounting, cache reuse, TT reuse, fixed-depth
determinism, fixed-node determinism, and rotation symmetry.

The direct null-output isolation harness
`null_output_isolation_harness.ce98802e.cpp` is 6,044 bytes, SHA-256
`ce98802e06046ccd52c8a48abfbeed0d1302cae6e9db4c1f06f7ca927518888e`.
It compared exact control and refrozen candidate across 1,004 deterministic
live states, masks 1 (root) and 5 (root+ply1), and budgets
1/16/64/256/1,024. Leaf frontier collection is excluded in both masks. All
10,040 encoded actions, root scores, and every emitted SearchStats field were
byte-identical. The common 1,059,169-byte stdout SHA-256 is
`5742b850fc7ce285f3936c43a38565b1e50ecd56599971956f328e0cee4926cf`;
the common 52-byte stderr SHA-256 is
`f20d99fe7924f98b29c366e2494da2ac2681641d74b32e0c4fcdeb85371e327a`.
The content-addressed JSON receipt records zero deltas and no timing/game/bank
use.

The rollback base itself was compared with the rejected PositionKey candidate
over 1,004 deterministic states, masks 0 and 7, and budgets
1/16/64/256/1,024. All 10,040 actions, root scores, and SearchStats fields were
byte-identical. Stdout SHA-256 was
`99f09e979bc5e4b25e5878880d9b7a9acd83e3a6870adba2b8e3be9fc306bd42`;
stderr SHA-256 was
`f20d99fe7924f98b29c366e2494da2ac2681641d74b32e0c4fcdeb85371e327a`.

## Deterministic weight-10 delta panel

The reporter source is 3,402 bytes, SHA-256
`93169282ae1d8eaa23243a0d96549c01ffbf907a8e0bd60290ba15b57bd1dd3a`.
It binds eight public test transcripts and budgets 251, 2,000, and 10,000,
with mask 7, replay blend 15, residual weight 100, and no deadline.

- control CSV: 1,298 bytes,
  `b055266f25b29657cfd7f1e6ccb3d477b12e127d60c19b9baac1d6fa0e47c41d`;
- candidate CSV: 1,300 bytes,
  `6dcb2829a2dca42ff6aea8773443fe7f24f5133859c20d41a83a50cc4dfb5f6a`;
- immediate candidate repeat: the same 1,300-byte hash.

All 24 decisions returned legal complete actions. Nodes were identical in all
24 capped runs. Scores changed in 24/24, actions changed in 2/24, leaf-proof
work changed in 16/24 through ordinary alpha-beta consequences, and completed
depth changed in 1/24. Score deltas ranged from -144 to +110, mean -3.208333,
and mean absolute 75.125.

| Case | Nodes | Control action | Candidate action | Control score | Candidate score | Delta | Depth C/W10 |
|---:|---:|---|---|---:|---:|---:|---|
| 0 | 251 | 3 | 3 | -11429 | -11370 | +59 | 2/2 |
| 0 | 2000 | 2 | 2 | -12714 | -12629 | +85 | 4/4 |
| 0 | 10000 | 0 | 0 | 11073 | 10971 | -102 | 5/5 |
| 1 | 251 | 4 | 4 | 12452 | 12511 | +59 | 2/2 |
| 1 | 2000 | 5 | 5 | 14146 | 14070 | -76 | 3/3 |
| 1 | 10000 | 33 | 33 | 4284 | 4369 | +85 | 4/4 |
| 2 | 251 | 43 | 43 | 9173 | 9232 | +59 | 2/2 |
| 2 | 2000 | 41 | 41 | 14554 | 14495 | -59 | 3/3 |
| 2 | 10000 | 43 | 43 | 4032 | 4091 | +59 | 4/4 |
| 3 | 251 | 5001 | 5001 | 13938 | 13879 | -59 | 1/1 |
| 3 | 2000 | 1 | 47411 | 13937 | 13873 | -64 | 3/3 |
| 3 | 10000 | 4701 | 4701 | 14249 | 14317 | +68 | 4/4 |
| 4 | 251 | 27 | 27 | -14215 | -14283 | -68 | 1/1 |
| 4 | 2000 | 6 | 6 | -14519 | -14578 | -59 | 3/3 |
| 4 | 10000 | 6 | 6 | -14549 | -14578 | -29 | 4/3 |
| 5 | 251 | 27 | 27 | -7978 | -8080 | -102 | 1/1 |
| 5 | 2000 | 6 | 6 | -9156 | -9232 | -76 | 3/3 |
| 5 | 10000 | 7 | 7 | -14288 | -14178 | +110 | 4/4 |
| 6 | 251 | 0225 | 1642 | 13892 | 13849 | -43 | 1/1 |
| 6 | 2000 | 0235 | 0235 | 14198 | 14257 | +59 | 2/2 |
| 6 | 10000 | 0253 | 0253 | 13954 | 13810 | -144 | 3/3 |
| 7 | 251 | 54 | 54 | 1546 | 1487 | -59 | 1/1 |
| 7 | 2000 | 7470 | 7470 | -14058 | -13948 | +110 | 2/2 |
| 7 | 10000 | 7470 | 7470 | -14058 | -13948 | +110 | 2/2 |

## Frozen timing instrument — not run

`leaf_proof_microbenchmark.549b6c29.cpp` is 6,595 bytes, SHA-256
`549b6c293f05d05b8e1724073decee034fc2959c2981902215925a53c91de059`,
and compiles against both exact sources;
the two binaries were not executed. It preconstructs 512 identical fixtures
(eight public tactical transcripts plus a deterministic public-rules random
walk), excludes corpus/topology/search construction from the timer, and times
one first-use leaf-proof/evaluation cache miss per search.

`run_leaf_proof_microbenchmark.29497387.py` is 10,013 bytes, SHA-256
`29497387ad9a48e54efbedc57d63e9df1b11fc52d71dec714eb78741a990bbe9`,
passes `py_compile`, and freezes:

- exclusive nonblocking `/tmp/rank4-hybrid-prototype-benchmark.lock` held for
  both builds, warmups, measurements, and exclusive report creation;
- 7 warmup pairs and 31 measured alternating AB/BA pairs;
- every raw sample and deterministic fixture/work/result signatures;
- paired candidate/control median at most 1.010 and p99 at most 1.020.

The runner was deliberately not executed. This archive contains no frontier
timing result.

## Recommendation and boundary

The correctness, safety, source-size, and material deterministic action deltas
justify exactly one execution of the frozen DEVELOPMENT plan after explicit
shared-lock release. They do not establish strength.

Stage 3 is the only operational d20 execution. Stage 4 runs only d04, d08, and
d12, then reuses the immutable content-addressed Stage-3 d20 receipt to form the
306-game aggregate; a second operational d20 attempt is forbidden.

At archive time: whole games run = false; heldout qualification run = false;
VALIDATION bank read = false; FINAL bank read = false; timing benchmark run =
false; arena upload = false; production source mutated = false.
