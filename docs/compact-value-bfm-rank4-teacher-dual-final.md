# Rank-4 teacher challenger: dual-final execution

`tools/compact_value_bfm_rank4_teacher_dual_final.py` is the one-shot execution
bridge between challenger governance and the maintained Compact Value-BFM
Rank-4 gate. It works for attempt zero and later trained candidates; identity is
derived from the challenger authorization rather than from a fixed source path.

The bridge deliberately requires one source-specific preflight authority and
exact five-job `pages.yml` CI evidence before opening either protected bank.
Attempt zero may use the discrete-v3 deployment-preflight reference. Later
SEARCH_VARIANT candidates use the generalized challenger release evidence,
which proves exporter base → compile-time search variant → seven-slot deployed
source. Either route must prove that its Rank-4 gate executable embeds the
authorized candidate source, record the compiler and actual deployment search
configuration, and supply the uncontended one-process timing samples. CI must
bind the same full candidate commit.

Preparation copies that verified gate into a non-writable, content-addressed
executable under the sole execution root. Gate bindings, shard commands, and
raw shard evidence use this frozen copy, while retaining the original preflight
binary as compile ancestry. This prevents a later build from silently changing
the executable during the multi-hour final run.

## Lifecycle

Run the commands only after challenger governance emits a valid
`dual-final-authorization.json`:

```text
prepare      validate/freeze authorization, candidate, runtime, preflight, CI,
             exact Rank-4, and every required exclusion
materialize  claim Gate A before entropy, create A; then claim Gate B before a
             second entropy draw and create B excluding every A fingerprint
run          consume/run Gate A as 100 five-pair shards on four workers; run
             Gate B identically only after Gate A passes; record governance
             results and dual qualification
abandon      seal a spent bank/shard interruption, preserve every materialized
             bank's fingerprints, reject the candidate, and open no retry
validate     revalidate an unmaterialized or materialized execution plan
```

There is exactly one execution directory for an authorization:
`<campaign-root>/dual-final/attempt-NNN/execution`. The v2 execution plan binds
that absolute path, the campaign-plan body, authorization body, and attempt.
Supplying another `--output-root` is rejected, so one authorization cannot mint
another pair of protected banks. A resumed `prepared.json` also records both
exact protected-bank and gate-bank identities; the bridge reopens challenger
governance and requires those same banks before returning or launching anything.

Production execution accepts only the module's built-in validators, loaders,
entropy source, generator, process runner, clock, result recorder, completer,
and executor. Callable injection is a test-only interface: the campaign must be
nonproduction and the Python caller must explicitly set
`allow_injected_test_evidence=True`. This opt-in is intentionally absent from
the CLI.

By default the CLI discovers every exclusion path from the validated campaign
ledger. Explicit `--exclusion` arguments may instead supply exactly one artifact
for every SHA-256 named by the authorization. The resolver follows immutable
JSON references and understands maintained opening banks, historical TSV banks,
challenger fingerprint lists, and protected canonical-position rows. ID-only
live records remain source-bound but contribute no state hash. The combined
state set must be nonempty.

Each protected bank publishes a transcript-free artifact with schema
`papersoccer.compact-value-bfm.rank4-teacher-challenger-fingerprint-exclusion.v1`.
It contains exactly the 500 sorted canonical fingerprints plus attempt, gate,
candidate/runtime, bank, and seed origins. `prepared.json` exposes both artifacts
before the first game, so a Gate-A rejection still permanently excludes both
banks from later development and finals.

## Fail-closed resume policy

A bank claim is written before its entropy draw. A shard claim is written before
the corresponding gate process. A claim without its matching receipt is terminal:
the bridge never redraws that bank and never reruns that shard. Resume only
validates/reuses completed receipts and schedules shards with no claim at all.
After an infrastructure interruption, run `abandon --plan EXECUTION_PLAN`
instead of `materialize` or `run`. It acquires both the campaign heavy-stage
lease and the bank-materialization lock, repeats the clean process/load audit
to rule out an orphaned gate worker, rederives the first spent claim/raw state,
and seals `dual-final-protected-stage-aborted.v1`. Any atomically
complete opening bank is converted to the same transcript-free canonical
fingerprint exclusion used by a normal prepared run, including a bank whose
final receipt was interrupted. The challenger then records
`protected-stage-aborted`, rejects the candidate without reading partial game
metrics, and authorizes a fresh leakage-isolated attempt. Repeating `abandon`
is idempotent; the original bank/shard remains permanently spent.
The runner captures a fresh launch timestamp internally at each gate boundary;
the CLI timestamp is only the overall run-start lower bound. Gate-B consumption
must be at or after the recursively validated Gate-A aggregate completion time,
including on resume. That chronology is part of the deep evidence contract.

The runner acquires `<campaign-root>/.rank4-teacher-heavy-stage.lock` with a
nonblocking exclusive `flock` and holds the descriptor continuously across Gate
A and Gate B. Generation, labeling, training, and development-gate runners use
the same campaign-global lease. Before each gate launch the bridge additionally
seals an append-only prelaunch audit. The audit proves the lease identity,
nice-zero parent process, one-thread BLAS/OpenMP environment, finite host load
below the logical-CPU oversubscription boundary, four-worker gate shape, and an
empty roster of competing Rank-4, development, recovery, or campaign gate
processes. Every shard's raw evidence references the audit active before its
claim. A resume receives a new audit before any still-unclaimed shard is
scheduled; completed shards retain their original audit ancestry.

The gate binary is invoked in `actual-clock` mode with the frozen deployment
configuration, exact Rank-4, two games per opening, and no per-shard early
threshold. The maintained aggregate applies the global 1,000-game thresholds:
at least 527 wins, at least 260 wins in each candidate color, and zero failures,
with the established timing limits.

## Evidence closure

Every governance evidence adapter uses the existing
`FINAL_GATE_EVIDENCE_SCHEMA` plus the discriminator
`dual-final-deep-gate-evidence.v2`. Its normalized aggregate is only a compatibility
view. Validation recursively reproduces the maintained aggregate from all 100
claim/receipt pairs and verifies every raw Rank-4 gate result, bank consumption,
gate binding, candidate source/runtime hashes, actual-clock configuration,
source-specific binary/compiler record, preflight timing, and exact CI record.
Gate-B validation additionally reopens the passing Gate-A aggregate and proves
its consumption timestamp follows Gate-A completion. Neither the bridge nor any
evidence receipt authorizes upload.

The execution plan also freezes the candidate's effective search-throughput
profile from release derivation. Every raw v2 gate shard must report that exact
compiled profile and internally reproduce its candidate intervention counters.
Deep Gate-A/Gate-B evidence aggregates all 100 shard documents: standard
requires every intervention counter to remain zero, while an intervention
profile requires all counters outside that profile to remain zero and records
its exact aggregate activity. Positive hit/restriction/reuse evidence is a
development A/B retention condition, not an extra protected-gate threshold;
this keeps the protected verdict exactly 527/260/zero failures while still
proving which profile was compiled and what it did.

The internal execution schemas changed with this closure:
`dual-final-execution-plan.v2`, `dual-final-execution-prepared.v2`,
`dual-final-gate-consumption.v2`, `dual-final-raw-shard-evidence.v2`, and
`dual-final-execution-receipt.v2`. Prelaunch records use
`dual-final-prelaunch-audit.v1`. The outer challenger
`FINAL_GATE_EVIDENCE_SCHEMA`, CodinGame protocol, and model runtime format are
unchanged.

When an execution receipt exists, `validate` also reopens it. Validation pins
every claim, raw result, and raw-evidence file to its deterministic ledger path;
reconstructs their exact sealed bodies; runs challenger validation on each
governance result; reconstructs the dual qualification from those exact two
results; and checks the corresponding append-only campaign-ledger events.
