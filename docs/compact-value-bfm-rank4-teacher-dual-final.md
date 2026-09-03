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
validate     revalidate an unmaterialized or materialized execution plan
```

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

The gate binary is invoked in `actual-clock` mode with the frozen deployment
configuration, exact Rank-4, two games per opening, and no per-shard early
threshold. The maintained aggregate applies the global 1,000-game thresholds:
at least 527 wins, at least 260 wins in each candidate color, and zero failures,
with the established timing limits.

## Evidence closure

Every governance evidence adapter uses the existing
`FINAL_GATE_EVIDENCE_SCHEMA` plus the discriminator
`dual-final-deep-gate-evidence.v1`. Its normalized aggregate is only a compatibility
view. Validation recursively reproduces the maintained aggregate from all 100
claim/receipt pairs and verifies every raw Rank-4 gate result, bank consumption,
gate binding, candidate source/runtime hashes, actual-clock configuration,
source-specific binary/compiler record, preflight timing, and exact CI record.
Neither the bridge nor any evidence receipt authorizes upload.
