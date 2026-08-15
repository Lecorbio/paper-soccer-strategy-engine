# Frozen DEVELOPMENT selection: safe-handoff frontier weight 10

Frozen on 2026-08-14 before any frontier timing benchmark or whole-game result
was observed. This plan is DEVELOPMENT-only. It does not authorize reading a
VALIDATION or FINAL bank, running heldout qualification, changing the live bot,
or uploading a source.

## Bound identities

The control is tracked commit
`4c9afa70a36183fc1451e4335b7251651a7b2791`, with rollback bot
`34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`
and generated source
`2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
The only candidate semantic change is safe-handoff frontier weight 10 at an
exact-proof Unknown leaf boundary. Its bot is
`408adc5288674550cc08274aec74380074117e32ad8f6915c7e39badc8dfba98`
and its generated source is
`08d0c0859ef8a197f8bfdd89afb048bec41c3a888228433b85991cd937882550`.

When a root, ply-one, or ply-two caller does not request a frontier count, the
candidate retains the original boolean handoff scan: it neither marks fresh
endpoints nor increments a unique count, and Loss remains `!safe_handoff`.
Deduplication/counting is isolated to the leaf caller that requests the
semantic value.

Candidate and control both use proof mask 7, replay blend 15, teacher-residual
weight 100, and the same replay/model files and shared PositionKey
implementation. There is one weight and one exact source identity. No weight
sweep, source edit, threshold relaxation, or retry is allowed after a stage
result is read.

Before any stage, bind the exact control and candidate commits, every source
and dependency hash, generated source size/hash, test hashes, gate wrapper and
recorder hashes, compiler version and flags, and binary hashes. A mismatch is a
stage failure, not permission to repair and retry the same identity.

## Frozen DEVELOPMENT banks

| Pool | Games | Per physical color | SHA-256 |
|---|---:|---:|---|
| `development_d04.tsv` | 78 | 39 | `984fbb78d85d7f9806c77e675b9b22a9b047bd15311f510ab0cedcd9a63244dc` |
| `development_d08.tsv` | 76 | 38 | `6dbec157e7094f07796a9aa1ac97b43919930377ec315c761eaace216630259e` |
| `development_d12.tsv` | 76 | 38 | `d30d087020e4946ce77b6d6e578484d583f0cf25f2ffa90a1918f8d9a9a8a11a` |
| `development_d20.tsv` | 76 | 38 | `2aa4b635dcaf23b2587b22fdb7558f4c8d6b4dd5a33e3fec2c164931b3fcd8d4` |

The shallow fixed-work pool is d04+d08: 154 games and 77 per color. The deep
fixed-work pool is d12+d20: 152 games and 76 per color. The unique operational
aggregate is 306 games and 153 per color. Replay corrections are disabled,
maximum game length is 320 turns, and games retain no transcripts.

## Sequential stop-on-failure rule

### Stage 0: source and semantic safety

Apply the archived repo-relative `apply_patch`-native transform to an isolated
exact `4c9afa7` tree. Require:

- the patch changes only `bot.cpp` and generated `submission.cpp`;
- generator `--check`, ASCII source no larger than 99,999 characters, and the
  exact candidate hashes above;
- maintained submission, parity, current-source, protocol, replay tactical,
  and fixed-work tests;
- the null-output isolation harness reproduces byte-identical actions, root
  scores, and every emitted SearchStats field for masks 1 and 5 across all
  10,040 registered fixed-work decisions;
- the archive-runnable focused suite passes normally and under AddressSanitizer
  plus UndefinedBehaviorSanitizer;
- the oracle suite covers outcome, path behavior, 648 local rotation cases,
  32 goal masks, unique-endpoint deduplication, mate bypass, replay-blend
  scaling, one-scan behavior, evaluation-cache reuse, TT reuse, fixed-depth
  determinism, and fixed-node determinism;
- a permanent witness has at least 12 used edges, produces exact-proof Unknown
  with a nonempty safe frontier, proves a nonzero teacher-residual correction,
  then proves the exact frontier delta and one-proof cached result;
- full replay nondisplacement covers 511/511 exact prefixes with zero capped,
  unresolved, or displaced decisions.

The original 7/7 focused source remains byte-frozen as provenance even though
its old relative include is not runnable from the archive directory. The
separate archive-runnable 8/8 source is the Stage-0 integration test and binds
its own hash; this changes no bot or generated-source byte.

Any Stage-0 failure rejects the feature without running timing or a game.

### Stage 1: locked leaf-proof microbenchmark

Run the archived harness/runner only after the campaign owner explicitly
releases `/tmp/rank4-hybrid-prototype-benchmark.lock`. The runner must acquire
that lock exclusively and nonblocking for both builds, warmups, all samples,
and report creation. It uses identical public tactical/procedural fixtures,
preconstructs each search outside the timed region, and times one first-use
exact leaf-boundary proof/evaluation call per fixture.

The frozen run has 7 warmup pairs and 31 measured pairs in alternating AB/BA
order, retains every raw sample, and uses the exact source hashes above.
Candidate/control paired ratios must pass both:

- median at most 1.010;
- p99 at most 1.020.

Failure rejects weight 10. There is no benchmark-driven optimization retry
under the same hypothesis.

### Stage 2: deterministic fixed-node screen

Run candidate versus the exact control with 30,000 nodes per decision, no
wall-clock deadline, proof mask 7 on both sides, and paired physical colors.
Run shallow first and advance only on a pass:

- d04+d08: at least 78/154 wins overall and at least 39/77 in each color;
- d12+d20: at least 77/152 wins overall and at least 39/76 in each color.

Every pool must have zero unfinished games, failed games, illegal actions,
operational failures, exceptions, or accounting inconsistencies. Failure stops
the sequence immediately.

### Stage 3: one-shot d20 operational-clock gate

Run the exact d20 bank once for this exact identity. Candidate and control each
receive a 3,000,000-node ceiling, 800 ms on the first decision, 165 ms later,
and operational hard clocks of 1000/200 ms.

Advance only with at least 40/76 candidate wins overall and at least 19/38 in
each physical color; zero unfinished, failed, illegal, operational, exception,
or hard-timeout outcomes; first-decision p99 below 900 ms and max below 990 ms;
later-decision p99 below 180 ms and max below 198 ms for both engines; and exact
outcome/work/proof/timing accounting.

The canonical Stage-3 report must be immutable and content-addressed before
Stage 4. This is the sole operational-clock execution of d20 for this source
identity.

### Stage 4: three-bank operational confirmation plus immutable d20 reuse

Without changing source or configuration, execute only d04, d08, and d12, in
that order, under the same node and clock limits. Do not execute d20 again.
Bind and reuse the exact immutable Stage-3 d20 report, including its source,
binary, dependency, configuration, bank, outcome, work, proof, and timing
identities, to form the unique 306-game d04+d08+d12+d20 aggregate.

Select only with at least 160/306 candidate wins overall, at least 77/153 in
each physical color, the Stage-3 zero-failure requirements, the same absolute
p99 and max clock limits for both engines, and exact per-bank and aggregate
accounting. A second operational d20 attempt is explicitly forbidden; it would
invalidate Stage 4 rather than replace or supplement the Stage-3 receipt.

Passing Stage 4 freezes the candidate for a separately preregistered untouched
heldout sequence. It does not authorize reading VALIDATION or FINAL, running
heldout qualification, changing production, or uploading a source.

## Failure action

At the first failed stage, record a content-addressed rejection receipt,
restore or retain exact control `4c9afa7`, and end this semantic hypothesis.
Weight 10 receives no second attempt, weight tuning, adjacent feature, or
result-based exception.
