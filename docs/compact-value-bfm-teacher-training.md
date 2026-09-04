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

- before promotion, the current HEAD and complete training/screening tool
  closure match the phase's clean build manifest; after a prior dual-final
  authorization, the persisted post-promotion mode permits drift only in the
  four declared deployment outputs while every other source and all frozen
  producers still match that manifest. The closure extends the
  already-validated pipeline with the trainer, exporters, gate source/support,
  `submission.json`, `sources.txt` and every member it names, and the exact
  Rank-4 source;
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
The full phase inherits the pilot's ranking weight and immutable adaptation
profile. Its selected pilot search variant is tie context only: the full-trained
model reruns the complete search roster before qualification.

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
fake-three-bit QAT epochs. Before NumPy is imported, the CLI re-executes with
every BLAS/OpenMP limit fixed to one; training then fails closed if that
pre-import snapshot or the launch environment differs. When `threadpoolctl` is
available, one outer limiter is established before either seed worker starts
and its controller inventory is recorded; the environment contract remains the
authority for runtimes such as Accelerate that are not exposed by that library.
Per-seed receipts remain independently resumable and bind this execution
evidence.

Every active ranking epoch deterministically permutes the complete
density-weighted action-group pool and partitions it across the unchanged
64/192 scalar batches with microbatch sizes differing by at most one. The
normalized per-group losses and gradients are averaged within each microbatch,
then the configured ranking lambda is applied exactly once. Thus each ordinary
group is consumed once per epoch, every registered hard-state duplicate is
consumed once per epoch, and increasing group count does not silently multiply
lambda. Warm-up and QAT receipts record unique, weighted, and hard-entry
coverage; ranking-mode shuffle epochs are continuous `1..5`.

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

## Development screens, full search A/B, and admission

After model selection, a nonproduction test may provide a fresh
content-addressed unprotected opening bank: 100 pairs for pilot or 500 pairs
for full.

For an allowlisted production campaign, omit `--bank`. The tool first seals a
post-selection seed claim at the campaign-derived phase root, derives its seed
from the exact selection, purpose, campaign identity, and complete fingerprint
exclusion union, and deterministically generates the sole permitted bank.
Supplying a caller-selected production bank is rejected. Nonproduction tests
may still supply `--bank`. Training, compilation, and actual-clock execution
hold `CAMPAIGN/.rank4-teacher-heavy-stage.lock`; injected runners, compilers,
auditors, renderers, or training producers require explicit nonproduction test
authorization.

```sh
python3 tools/compact_value_bfm_teacher_training.py prepare-gate \
  --plan TRAINING_PLAN \
  --selection TRAINING_SELECTION [--resume]
```

The development roster is phase-specific. `active_search_variants()` remains
the complete full-round registry, while the gate plan seals the smaller pilot
roster when appropriate:

| Phase/profile | Variants | Pairs per variant | Exact games |
| --- | ---: | ---: | ---: |
| standard pilot | baseline only | 100 | 200 |
| intervention pilot | baseline plus its baseline treatment | 100 | 400 |
| standard full search A/B | all four standard variants | 500 | 4,000 |
| intervention full search A/B | four controls plus four treatments | 500 | 8,000 |
| post-selection full qualification | selected variant only | 500 | 1,000 |

The standard pilot therefore measures the trained model without selecting a
search optimization on its 105/200 screen. An intervention pilot is the only
two-arm exception: its treatment must beat the baseline control on the exact
paired roster and pass activation, source, parity, actual-clock timing, and
zero-failure checks. Otherwise the clean baseline is the control fallback.

The complete full-round standard roster is:

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

Every development variant runs as one whole-bank process, serially, with one
worker and one thread. The runner seals a prelaunch no-retry claim, verifies
nice level zero and no competing Rank-4 gate process, fixes all BLAS/OpenMP
thread environments to one, and records claim/launch/finish chronology and the
raw v2 result/profile-activation closure.

```sh
python3 tools/compact_value_bfm_teacher_training.py run-gate \
  --plan TRAINING_PLAN \
  --gate-plan GATE_PLAN [--resume]
```

If infrastructure interrupts a claimed whole-bank process before a valid
receipt, never invoke `run-gate` again for that bank. Seal its terminal,
metric-free state instead:

```sh
python3 tools/compact_value_bfm_teacher_training.py abandon-gate-execution \
  --plan TRAINING_PLAN \
  --gate-plan GATE_PLAN \
  --variant SEARCH_VARIANT \
  --abandoned-at-utc UTC

python3 tools/compact_value_bfm_rank4_teacher_challenger.py \
  record-development-gate-abort \
  --plan CAMPAIGN_PLAN \
  --abandonment GATE_ABANDONMENT \
  --created-at-utc UTC
```

The receipt binds the exact bank, candidate, claim/raw snapshots, build-source
closure, and the precomputed fingerprint-only development exclusion. It never
uses partial metrics, never counts an improvement, and authorizes only a new
leakage-isolated same-contract attempt. Abandonment takes the same campaign
heavy-stage lease as execution and repeats the clean process audit, so a stale
claim alone can never be used to abandon a worker that is still running.

For a pilot, admit that execution directly:

```sh
python3 tools/compact_value_bfm_teacher_training.py admit \
  --plan TRAINING_PLAN \
  --gate-plan GATE_PLAN \
  --execution GATE_EXECUTION [--resume]
```

For a full phase, the first execution is selection evidence only. Freeze its
winner before opening another bank:

```sh
python3 tools/compact_value_bfm_teacher_training.py select-full-search \
  --plan FULL_TRAINING_PLAN \
  --gate-plan FULL_AB_GATE_PLAN \
  --execution FULL_AB_EXECUTION [--resume]

python3 tools/compact_value_bfm_teacher_training.py prepare-full-qualification \
  --plan FULL_TRAINING_PLAN \
  --selection FULL_SEARCH_SELECTION [--resume]

python3 tools/compact_value_bfm_teacher_training.py run-gate \
  --plan FULL_TRAINING_PLAN \
  --gate-plan FULL_QUALIFICATION_PLAN [--resume]

python3 tools/compact_value_bfm_teacher_training.py admit \
  --plan FULL_TRAINING_PLAN \
  --gate-plan FULL_AB_GATE_PLAN \
  --execution FULL_AB_EXECUTION \
  --full-search-selection FULL_SEARCH_SELECTION \
  --qualification-plan FULL_QUALIFICATION_PLAN \
  --qualification-execution FULL_QUALIFICATION_EXECUTION [--resume]
```

Production omits `--bank` from both preparation commands. Each command creates
its own post-selection deterministic seed claim and generated bank. The full
qualification bank is generated only after search selection is sealed, must be
symmetry-disjoint from the A/B bank, and extends the development exclusion with
both banks. A nonproduction test may supply a bank explicitly.

In the full A/B, a legacy search change is retained only when its identical-bank
paired result improves on baseline and source/timing checks remain clean; the
combined arm additionally requires both individual arms. Each intervention
treatment is compared only with its corresponding standard base and must pass
the same activation/source/parity/timing/zero-failure checks. The pilot variant
is used only to resolve an exact full-A/B win tie.

Pilot admission requires at least 105/200 wins and zero failures for the
selected pilot screen candidate. Full A/B does not itself qualify a candidate.
The separately generated 1,000-game qualifier requires at least 550 wins, at
least 265 wins per color, a deterministic pair-cluster bootstrap lower 95%
bound above 50%, and zero failures.

Admission emits a content-addressed receipt, a complete development-fingerprint
exclusion, and a `phase-outcome-evidence.v1` artifact directly consumable by
the challenger governance tool. Pass the admission itself to governance's
required `record-outcome --admission-receipt`; governance deep-loads the
pipeline, selection, gate plan, exact gate results, candidate, and audit hash
instead of trusting copied summary metrics. The evidence also binds the full
build-source closure hash; admission loading reopens the training plan and
revalidates that closure even when dataset revalidation is disabled.
`--resume` reuses only byte-identical stable references. It never reruns
completed seeds, retries a claimed whole-bank gate, recompiles a completed gate
plan, reselects full search after the qualification bank is opened, or replaces
either deterministic production bank.
