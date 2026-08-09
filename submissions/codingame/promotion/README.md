# CodinGame promotion gate

This directory freezes the evidence used to decide whether a challenger may
replace `rank_5`. It uses the authentic CodinGame rules (`OwnGoalsAllowed`,
`MoverLoses`) and complete-turn transcripts. The flagship demo benchmark uses
different rules and is not valid for this decision.

## Frozen T9 carry-forward evidence

T9 is frozen but not candidate-bound. Its validation and sealed-final references
are byte-identical aliases of the T8 banks, and it inherits the exact T8 strength
protocol with no semantic or threshold changes. The T9 evidence manifest keeps
both candidate fields null.

The only T8-bound candidate, `frontier_proof`, was rejected by the exposed T3
development prerequisite. Its deterministic decision marks validation and test
as not run. The T9 freezer also verifies that the T8 result identity contains no
validation or test report, shard directory, immutable bank snapshot, or final
ledger marker. Consequently neither prospective bank was consumed, so the
versioned T9 ladder carries their bytes forward without inspecting outcomes.

Verify the unbound freeze with:

```sh
python3 submissions/codingame/promotion/freeze_t9_banks.py --check
```

No promotion command should run until a future active manifest binds exactly
one candidate and its harness identities to the T9 evidence-manifest hash.

## Historical T7 evidence

The earlier T7 ladder bound `exchange_proof` to candidate-independent evidence:

- `initial.tsv` is the byte-preserved initial-position sentinel;
- `development.tsv` is the byte-preserved T3 development bank;
- `validation.tsv` is copied byte-for-byte from the prospective T7 validation
  reference; and
- `test.tsv` is copied byte-for-byte from the disjoint, sealed T7 final
  reference.

The reference-bank hashes, selection provenance, prospective protocol, and
prior-decision provenance are pinned by `reference/t7_evidence_manifest.json`.
That manifest and both T7 reference banks were frozen before candidate binding.
The active manifest additionally pins the evidence-manifest and protocol hashes,
the exact candidate submission, all three `exchange_proof` harness sources, the
freezer, acquisition controller, and promotion controller.

This is a deliberate pre/post-binding provenance split. Source hashes inside the
immutable evidence manifest identify the candidate-independent freeze-time tools;
they are historical provenance, not expectations for the later binding tools.
The active manifest pins that pre-binding snapshot and the final post-binding
builder, controller, candidate, and harness identities.

The T6 validation rejection and T6 final-bank use remain immutable historical
provenance. The T7 builder neither reconstructs them from raw games nor
re-evaluates them. It only verifies frozen hashes and copies already-correct TSV
bytes to the active validation and test paths. The active initial and development
bytes are unchanged.

States were normalized by rotating the historical winner to Player 0 and
deduplicated with horizontal reflection. Confidence intervals cluster all
positions from the same source game, so several snapshots from one trajectory
cannot masquerade as independent evidence. The runner reconstructs and plays
the original physical state.

## Inherited strength protocol

Every profile in a stage uses that stage's single bank and common shard count.
The stages are fixed as follows:

- initial: legacy 5,000-node sentinel;
- development: legacy 5,000- and 30,000-node profiles;
- validation: explicit `30k-nodes` and `130ms` profiles; and
- test: explicit `100k-nodes` and `130ms` profiles.

Each 130 ms profile uses a 3,000,000-node safety cap. Node-profile throughput
must remain at least 0.9 of the incumbent; equal-time throughput is diagnostic.
Historical-winner and historical-opponent scores and uplifts are also diagnostic
only. The hard control-normalized role requirement is the frozen physical-color
uplift threshold. All other stage thresholds are copied exactly from the frozen
prospective protocol.

The historical T7 candidate `exchange_proof` retained the unconditional
current-turn root and depth-zero
component proofs. At positive depth, after the transposition-table probe, it
runs exact component proof only at `turn_ply == 1` and `turn_ply == 2`: the
opponent reply and root-player counterturn. `Unknown` continues through the
unchanged search. The comparison report keeps the existing component totals and
separate ply-one/ply-two probe, Win-hit, Loss-hit, and exact-cutoff counters.

## Historical T7 invocation

The commands below document the prior T7 binding. They are not valid T9 run
instructions while the T9 candidate remains null.

First verify the frozen binding:

```sh
python3 submissions/codingame/promotion/build_goal_shell_banks.py --check
python3 submissions/codingame/tools/promotion_gate.py validate --bot exchange_proof
```

The normal ladder is then:

```sh
python3 submissions/codingame/tools/promotion_gate.py preflight --bot exchange_proof
python3 submissions/codingame/tools/promotion_gate.py run --bot exchange_proof --stage initial
python3 submissions/codingame/tools/promotion_gate.py run --bot exchange_proof --stage development
python3 submissions/codingame/tools/promotion_gate.py run --bot exchange_proof --stage validation
python3 submissions/codingame/tools/promotion_gate.py run --bot exchange_proof --stage test
python3 submissions/codingame/tools/promotion_gate.py timing --bot exchange_proof
python3 submissions/codingame/tools/promotion_gate.py evaluate --bot exchange_proof
```

`all` runs the same ladder and stops at the first rejection. Each stage requires
every earlier stage for the exact candidate, manifest, bank, execution-profile,
and shard identities. The first sealed-final use writes a fixed v2 ledger under
`.git/papersoccer-promotion`; it binds the ordered profile identities and common
raw-evidence shard count. A different candidate requires a newly frozen,
game-disjoint final bank rather than tuning against an exposed holdout.

Every official command reruns the deterministic builder check. The controller
verifies each bank, snapshots its bytes into the result identity before workers
start, and reaggregates reports from raw shards before accepting a cached
decision. The two candidate-color games and same-budget `rank_5` control for an
opening execute sequentially in one runner process. Strength profiles can run
in parallel; the separate latency stage runs serially in fresh processes after
the locked test passes.

Starting a rerun invalidates a prior decision. Deterministic failures write
`decision.json`, with later stages marked as not run. The controller exits `0`
only for `PROMOTE`, `10` for `REJECT`, `20` for an incomplete decision, `64` for
invalid configuration, and `70` for an operational game failure.

The checked-in final bank is procedurally sealed, not cryptographically blind.
Once its actions or outcomes are inspected, it must be retired. A genuinely
blind final gate should live in an external evaluator that reveals only a
one-shot verdict.
