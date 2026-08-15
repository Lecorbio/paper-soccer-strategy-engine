# Root tactical scout rejection

Date: 2026-08-13

Status: **rejected; do not ship or retain in the production source**.

## Frozen stage under test

- Source: `source_payloads/1192d91be6526924a7ae5930e12d6b44ddd6848931522c8073ed07d2e43a932c.source`
- SHA-256: `1192d91be6526924a7ae5930e12d6b44ddd6848931522c8073ed07d2e43a932c`
- ASCII bytes: 98,690
- Regression test: `root_scout_rejection_test.cpp`
- Test SHA-256: `663b23916283c24e112a25af4cf30210f8e9ec42e9ddfd464ae6dae5a6cbed53`

The test includes the immutable content-addressed source above. It does not use a
validation/final opening bank or a protected historical replay payload. The
second witness is a newly generated procedural scratch transcript.

## Blocking defects

### P1: scout heuristic entries poison the exact-proof evaluation cache

`retain_scout_endpoint` evaluates every nonterminal scout endpoint through
`cached_evaluate` (source lines 2186-2204). Exact leaf search probes that same
cache and returns a hit before calling `analyze_rebound_component` (lines
2436-2446). The cache invariant required by exact proof is therefore false:
scout-created heuristic entries have not first received an `Unknown` proof.

Minimal witness:

```text
cache_poison control=1/999998 proof_loss_hits=1
cache_poison scout=1/15960 proof_loss_hits=0 cache_hits=2
```

The scout suppresses a proven opponent-component loss and replaces the exact
mate value `+999998` with heuristic value `+15960`.

### P1: scout heuristic capture can survive exact minimax correction

`retain_scout_endpoint` also sends heuristic scores to
`record_completed_action` (line 2204). That method replaces the captured action
only on a strict score improvement (lines 1949-1962). Later exhaustive search
can correct the minimax score without replacing an earlier, spuriously stronger
heuristic capture. `search_root` then pairs the corrected result score with the
stale captured action (lines 2521-2538).

Minimal newly generated scratch witness, with the evaluation cache disabled so
this defect is isolated from cache poisoning:

```text
transcript=6/5/4/1/4/2/4/2/1/6/7/70/1/1/1
control=74/4446
scout=135/4446
exact leaf score of returned scout action=999998
scout endpoints=20
```

Thus the returned action is not the action represented by the root minimax
score. Player Two is the minimizing root mover in this witness, but the scout
returns an action that gives Player One a proven `+999998` win while reporting
the corrected root value `+4446`. The test disables the evaluation cache,
replays the returned complete action inside a search constructed at the
original root (preserving its residual phase gate), invokes the immutable
stage's exact component classifier at the successor, and evaluates only after
an `Unknown` classification.

## Development evidence and interpretation

- Fixed-500-node DEVELOPMENT result: proof only `181-125`; proof plus scout
  `158-148`; scout only `154-152`; control `154-152`.
- Proof scans and scout work are not charged to the node counter. These numbers
  are algorithmic development signals, not equal-work promotion evidence.
- Actual-clock DEVELOPMENT result with exact proof enabled and scout disabled:
  `164-142`, Player 0 `83-70`, Player 1 `81-72`, zero unfinished games, illegal
  actions, hard timeouts, or other operational failures. Candidate later-turn
  p99/max was `165.161/165.269 ms`; Rank-4 was `165.157/165.187 ms`.
- That clock run used the 98,690-byte stage with its scout flag disabled. It is
  selection evidence for exact proof, not final-source identity qualification.

## Decision

Keep the rejected root-scout production path removed. Its immutable source and
regression witness remain only as rejection evidence; they are not a rollback
or integration target.

Qualification continues from the current independent proof-scope-mask stage:

- generated source bytes: 94,004 ASCII bytes;
- generated source SHA-256:
  `6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f`;
- source path: `submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp`.

The earlier 92,830-byte exact-proof source
(`2ca2280f533bffc1750732ae55926b8353b16299a40c456e0b58a4cb8d426468`)
is a superseded development checkpoint, not the source to restore or qualify.
Run compiler, sanitizer, protocol, rotation, cold-process timing, and comparison
gates against the exact proof-mask identity ultimately selected from the
current stage. The 94,004-byte identity above is still under qualification; it
is not yet a finalist or an upload claim.

## Reproduction

```sh
c++ -std=c++20 -O2 \
  results/rank_4_jacek_hybrid/root_scout_rejection_test.cpp \
  -o build/rank4_hybrid_root_scout_rejection_test
build/rank4_hybrid_root_scout_rejection_test
```

Expected exit status is `1`: both tests intentionally assert the safety
properties that the rejected stage violates.
