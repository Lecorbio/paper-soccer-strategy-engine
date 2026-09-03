# Compact Value-BFM teacher training and selection

`tools/compact_value_bfm_teacher_training.py` connects a finalized
pilot/full teacher-label pipeline to the maintained Compact Value-BFM trainer.
It preserves the deployable `6301 -> 12 -> 8 -> 1` bias-free scalar network,
three-bit weights, activations, feature schema, and single-thread protocol.
There is no policy head or runtime feature change.

## Bound inputs

`prepare` accepts only a validated Rank-4 challenger campaign and phase, a
completed `07-finalize-labels` pipeline receipt. The phase plan, rather than a
CLI override, supplies a copied regular `SHA256.float.npz` 12x8 checkpoint. For
attempt one this is exactly
`0dbe279295bcfeb80392e10ff9b04f9bd5c0eb4e1ea45c974b6a92d266f37729`.
The adapter uses the copied original `FrozenBundle`
for canonical anchors, the common adjudicator, canonical validation, and the
teacher identity. The new phase train CSR replaces only the trainer's `new`
dataset. The phase validation CSR remains an unseen-root audit set and binds
the complete-turn validation ranking groups.

Before a plan is sealed, the adapter verifies:

- the current HEAD and complete training/screening tool closure match the
  phase's clean build manifest, extending the already-validated pipeline
  closure with the trainer, exporters, gate source/support, `submission.json`,
  `sources.txt` and every member it names, and the exact Rank-4 source;
- the CSR manifests/NPZ and successor aggregate are content/receipt bound;
- successor-label train and validation roots exactly match the frozen phase
  root assignments;
- phase CSR provenance binds the same root manifest;
- external parents and every legal complete-turn successor are isolated by
  root and exact/rotate/reflect/rotate-reflect identity from all validation;
- the external corpus does not intersect prior development, prior
  train/validation, protected, or live fingerprint-only exclusions; and
- no protected label, transcript, or metric is opened.

Example:

```sh
python3 tools/compact_value_bfm_teacher_training.py prepare \
  --campaign-plan CAMPAIGN_PLAN \
  --phase-reference PILOT_PHASE_REFERENCE \
  --pipeline-plan PILOT_PIPELINE_PLAN \
  --output-root RESULTS_ROOT \
  --created-at-utc 2026-09-04T00:00:00Z
```

For `full`, also pass the admitted pilot receipt with `--pilot-admission`.
The full phase inherits that pilot's ranking weight and search variant.

## Training and model-first selection

```sh
python3 tools/compact_value_bfm_teacher_training.py train \
  --plan TRAINING_PLAN [--resume]
```

Pilot training runs scalar λ=0, ranking λ=0.10, and ranking λ=0.25. Full
training runs scalar λ=0 plus the pilot-selected λ. Every arm uses the
trainer's fixed seeds `20260907`, `20260908`, and `20260909` with exactly two
seed workers. The maintained trainer enforces 64 new plus 192 anchor rows per
batch, one all-layer float warm-up epoch at `6e-5`, and four all-layer
fake-three-bit QAT epochs. Per-seed receipts remain independently resumable.

The selected seed for each λ is exported with the maintained model and source
exporters. The deterministic C++ compactor must be idempotent, the source must
be strictly below 95,000 ASCII bytes, and at least 2,000 bytes of reserve must
remain. Model selection is frozen before a screening bank is read. A ranking
model advances only when canonical retention passes, mean teacher regret is at
least 10% below the scalar control, and its float/quantized action-flip rate is
at most scalar control plus 0.005. Its validation ranking evidence must also
contain at least 100 comparable exhaustive groups covering at least 80% of all
validation groups; the same coverage floor is rechecked at pilot and full
admission.

## Search A/B requests and admission

After model selection, provide a fresh content-addressed unprotected opening
bank: 100 pairs for pilot or 500 pairs for full.

```sh
python3 tools/compact_value_bfm_teacher_training.py prepare-gate \
  --plan TRAINING_PLAN \
  --selection TRAINING_SELECTION \
  --bank FRESH_OPENING_BANK.opening-bank.json [--resume]
```

Pilot emits four 200-game actual-clock requests around the identical runtime:

| Variant | Embedded compile-time macros | Behavior |
| --- | --- | --- |
| `baseline` | `REFERENCE_FEATURE_SORT`, `REFERENCE_DESCENDANT_SORT` | both legacy paths |
| `no-feature-sort-only` | `REFERENCE_DESCENDANT_SORT` | optimized feature path only |
| `single-pass-selection-only` | `REFERENCE_FEATURE_SORT` | optimized descendant selection only |
| `combined` | none | both optimized paths |

The full macro names are
`COMPACT_VALUE_BFM_REFERENCE_FEATURE_SORT` and
`COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT`. Each request binds the exact
macro list, standalone source, compiler identity, compiled Rank-4 gate binary,
runtime/payload, bank TSV, Rank-4 source, clocks, work limits, and argv. Because
the macros are embedded at the top of each standalone source, the selected
source's default behavior is the tested variant.

Run the requests externally, replacing the terminal `RESULT_PATH` argv token,
then validate them together:

```sh
python3 tools/compact_value_bfm_teacher_training.py admit \
  --plan TRAINING_PLAN \
  --gate-plan GATE_PLAN \
  --result baseline=BASELINE.json \
  --result no-feature-sort-only=NO_FEATURE_SORT.json \
  --result single-pass-selection-only=SINGLE_PASS.json \
  --result combined=COMBINED.json
```

A search change is retained only when its paired result improves on baseline;
the combined arm additionally requires both individual arms to improve and may
not trail either. Pilot admission then requires at least 105/200 wins and zero
failures. Full accepts only the inherited pilot variant and requires at least
550/1000 wins, at least 265 wins per color, a deterministic pair-cluster
bootstrap lower 95% bound above 50%, and zero failures.

Admission emits a content-addressed receipt, a complete development-fingerprint
exclusion, and a `phase-outcome-evidence.v1` artifact directly consumable by
the challenger governance tool. Pass the admission itself to governance's
required `record-outcome --admission-receipt`; governance deep-loads the
pipeline, selection, gate plan, exact gate results, candidate, and audit hash
instead of trusting copied summary metrics. The evidence also binds the full
build-source closure hash; admission loading reopens the training plan and
revalidates that closure even when dataset revalidation is disabled.
`--resume` reuses only byte-identical stable
references; it never reruns completed seeds or recompiles a completed gate
plan.
