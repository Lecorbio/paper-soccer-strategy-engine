# CodinGame promotion gate

This directory freezes the evidence used to decide whether a challenger may
replace `rank_5`. It deliberately uses the authentic CodinGame rules
(`OwnGoalsAllowed`, `MoverLoses`) and complete-turn transcripts. The flagship
demo benchmark uses different rules and is not valid for this decision.

The banks are temporally disjoint:

- `development.tsv`: goal-shell states from the five older checked-in batches;
- `validation.tsv`: states from the three later checked-in batches;
- `test.tsv`: newer wins by rank-one agent `6273433`, frozen before the
  strategy experiment and process-sealed from tuning;
- `initial.tsv`: the initial-position sentinel, excluded from bootstrap
  inference.

States are normalized by rotating the eventual winner to Player 0, deduplicated
with horizontal reflection, and capped per source game. Confidence intervals
cluster all positions from the same source game, so several snapshots from one
trajectory cannot masquerade as independent evidence. The match runner still
reconstructs and plays the original physical state. `manifest.json` binds every
source, harness component, and bank by SHA-256 and declares all thresholds
before evaluation. Every official command also reruns the deterministic builder
check, so editing a manifest threshold or merely updating a listed hash cannot
self-authorize a result.

Typical workflow:

```sh
python3 submissions/codingame/promotion/build_goal_shell_banks.py --check
python3 submissions/codingame/tools/promotion_gate.py validate
python3 submissions/codingame/tools/promotion_gate.py preflight --bot topology
python3 submissions/codingame/tools/promotion_gate.py run --bot topology --stage initial
python3 submissions/codingame/tools/promotion_gate.py run --bot topology --stage development
python3 submissions/codingame/tools/promotion_gate.py run --bot topology --stage validation
python3 submissions/codingame/tools/promotion_gate.py run --bot topology --stage test
python3 submissions/codingame/tools/promotion_gate.py timing --bot topology
python3 submissions/codingame/tools/promotion_gate.py evaluate --bot topology
```

`all` runs the same ladder and stops at the first rejection. Strength shards
run in parallel at fixed node budgets; the locked test must persist at both
30,000 and 100,000 nodes before timing runs serially in fresh processes. Each
stage requires every earlier stage for the exact candidate and manifest hashes.
The first locked-test use writes a fixed ledger under `.git/papersoccer-promotion`
that cannot be redirected with `--results`: a different candidate must use a
newly frozen, game-disjoint test bank rather than tune against an exposed
holdout. Evaluation requires the matching ledger entry.

Every run rechecks both generated submissions. Reports record the compiled
runner and timing identities, live under a candidate-plus-manifest result key,
and are reaggregated from raw shards before a decision can be reused. Timing
includes dense elite/rank-one goal-shell states, while every strength stage
after the initial sentinel enforces a candidate/incumbent throughput ratio.
Starting any rerun first invalidates a prior decision; every deterministic
failure writes `decision.json`, with later stages explicitly marked as not run.
The controller exits `0` only for `PROMOTE`, `10` for `REJECT`, `20` for an
incomplete decision, `64` for invalid configuration, and `70` for an operational
game failure. A completed but losing match batch is never success.

## Promotion policy

Freeze one mechanism and its thresholds before implementation. Use focused
correctness states first, then the initial sentinel and cheap development bank;
most variants should end there. Only a development survivor may see validation,
and only a validation survivor may consume the locked rank-one bank. Source
size and timing are hard requirements, not post-hoc diagnostics. The only live
submission allowed is the exact SHA-256 that receives `PROMOTE`.

Arena score remains opponent-weighted and noisier than local head-to-head play.
After a promoted source is submitted, keep its hash fixed and wait for the whole
batch before accepting it; use that waiting time only for analysis or a next
hypothesis on a fresh development corpus. Never tune against the in-flight or
locked-test results.

The checked-in holdout is procedurally sealed, not cryptographically blind: a
developer can read it or delete local state. Once anyone inspects its actions,
it must be retired. A genuinely blind final gate should live in an external
evaluator that reveals only a one-shot verdict. The strength runner compiles the
generator-checked maintained engines rather than spawning the paste-ready
protocol programs; before any future live promotion, add exact-submission
decision parity over the frozen roots or use a process-level protocol arena.
