# Rank-4/Jacek hybrid campaign evidence

This directory freezes the provenance and procedural evaluation boundary for
the 36-hour `rank_4_jacek_hybrid` campaign.  It is evidence, not bot source.

## Time and control

- T0: `2026-08-13T19:15:07Z` (`2026-08-13T21:15:07+02:00`, Europe/Warsaw),
  Unix `1786648507`.
- Deadline: `2026-08-15T07:15:07Z`
  (`2026-08-15T09:15:07+02:00`), Unix `1786778107`.
- Rank-4 control source: 98,624 bytes, SHA-256
  `5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`.
- Historical control identity: agent `6604719`, submission `41114327`, rank 4,
  66-24 over its completed 90-game window.

The hybrid is explicitly Rank-4-derived.  It inherits Rank-4 source, search,
models, replay/value assets, and historical training lineage.  Jacek's guide
supplies design ideas.  This campaign must not be described as clean-room or
fresh-only.

## Frozen evidence

The canonical campaign manifest is `campaign.json`.  Its byte-identical,
content-addressed copy is:

```
manifests/aed2a52f7a59c2b1988b5c365c23b57f8ec41fbfb50927211655a8565df63fa7.json
```

The manifest records aggregate tree hashes for existing protected promotion,
flagship, game-review, arena, Jacek, and Rank-4-training evidence.  Protected
banks and replay archives were treated as opaque bytes: the freezer hashed
them but did not parse sealed bank or replay payload content.  Root
`matches.json` was absent at T0 and must remain absent.

The arena exclusion registry is canonical, content-addressed, and contains
4,205 pre-T0 or already-known game IDs:

```
arena/exclusions/578524a7cf0908fa29df3541b4f0b1f7f9e85c1c0374d0dbc0d656686e9df159.json
```

Every hybrid arena collection must pass this exact path and SHA-256 to
`collect_arena_batch.py`.  It was assembled from the prior ID-only registry
and numeric prior-campaign game-record directory names; replay payloads were
not opened.

## Procedural evaluation assignments

All banks are color-swapped and span opening depths 4, 8, 12, and 20.

| Role | Openings | Games | Permitted use |
|---|---:|---:|---|
| Development | 153 | 306 | Iterative ablation/model selection |
| Validation | 53 | 106 | Candidate selection only |
| Final | 106 | 212 | One shot after finalist source/SHA lock |

The banks use deterministic SHA-256-domain-separated record seeds derived
from T0.  Generated records were rejected on exact-state or horizontal-mirror
overlap with the metadata-only 1,600-opening predecessor registry and every
earlier new bank.  No protected bank was passed to the generator.  The older
promotion banks do not expose a state-ID-only registry, so their bytes remain
unopened; this limitation is explicit in `campaign.json`.

Before the closed one-shot qualification recorded below, the frozen rule was
not to run, parse, summarize, or inspect a `final_*.tsv` result until exactly
one finalist source byte count and SHA-256 had been locked. That prerequisite
authorized only the completed qualification; it does not authorize a retry or
a fresh held-out campaign.

## Authoritative held-out qualification

The locked finalist completed its one-shot qualification. VALIDATION passed
61-45, with 34 wins as physical color 0 and 27 as physical color 1, clearing
the frozen floors of 54 total and 26 in each color. FINAL finished 104-108,
with 48 wins as color 0 and 56 as color 1; it failed the 108-win total floor
and the 53-win color-0 floor. Safety, timing, proof and sweep accounting,
input and compiler stability, and source, admin, binding, and portability
provenance all remained clean.

The canonical VALIDATION and FINAL reports are
`gates/heldout_qualification/binding_recovery_v1/reports/validation/e0b5ed9bd6c77ce90317cc363ab19679e01216172de9ebb01b5eb05d2c6bc5cc.json`
and
`gates/heldout_qualification/binding_recovery_v1/reports/final/19e0d5e692d5afce2f9a83ef2247bcf53816bf7d66c6a60aa0bcef9205b4c271.json`.
The terminal decision is
`gates/heldout_qualification/binding_recovery_v1/decisions/9c12b44cc2ffa475e55e1e166c637f725e8107736677c432b03ea31ef376997f.json`:
`final_qualification=false` and arena authorization is false. This exact
qualification cannot be retried, and no live upload occurred.

## Post-heldout DEVELOPMENT mask-3 removal

After that terminal held-out rejection, a separately preregistered
DEVELOPMENT-only campaign tested whether removing the ply-one exact-proof scope
could rescue the hybrid without using held-out scores for selection. Stage 1
compared mask `3` directly with the same-binary mask-`7` control over the four
existing DEVELOPMENT banks. Mask 3 finished `152-154`: `85-68` as physical
color 0 and `67-86` as physical color 1. It therefore missed the frozen floor
of 160 total wins by eight and the floor of 77 color-1 wins by ten. Those were
the only selection-threshold errors.

All 306 games completed with zero failed, unfinished, illegal, operational,
exception, or hard-timeout results. Candidate first/later maxima were
`800.185/165.237 ms`; reference maxima were `800.260/165.335 ms`, all within
the strict `990/198 ms` ceilings. Timing, proof-scope, engine-work, color and
bank aggregation, process safety, and source/binary/admin/environment
provenance all replayed cleanly with stable before/after evidence.

The canonical Stage-1 report is
`gates/mask3_removal_clock/reports/64623834951fd4a00484ef0aa1a890127fe9b6a19539b65b3fb874fcf4794725.json`.
The terminal decision is
`gates/mask3_removal_clock/decisions/894442b0d0a81418d591469bb1b8c1d34cc6c0a8ed13371812a66a21d1e5bc48.json`.
Stages 2 and 3 were never claimed or opened. The decision authorizes no retry,
source activation, fresh held-out campaign, or arena action. Canonical Rank 4,
source SHA-256
`5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`,
remains the incumbent.

## TT exact-collision generated DEVELOPMENT replication

The separately preregistered V19 campaign evaluated candidate identity
`a72aac1d148d7e85ed896f4b994593eb847335b8d79f31ac14d604d3888a618d`.
Preexecution completed successfully, including the focused 48-test gate,
sealed dependency and compiler checks, and an in-memory-only deterministic
generation self-test. No protected bank file was accessed.

Stage 0 exhausted its frozen 1,700-second aggregate budget before completing
the corpus. It persisted 968 of 1,324 scheduled children: children 0 through
966 returned successfully, child 967 was killed at the aggregate boundary,
and 356 children were never started. The execution lasted 1,700.984 seconds.
The incomplete output was not parsed as a Stage-0 result and supplies no
performance or safety conclusion. Neither DEVELOPMENT game stage ran.

The canonical report is
`gates/tt_exact_collision_generated_v19/reports/ef323674bbf22dcc18c938b3fcca4b40af3e1978aba0a0ee596addd72ba7de51.json`.
The terminal decision is
`gates/tt_exact_collision_generated_v19/decisions/3bff57ce07bd80ebc7107192841269a34c08e6e231f6c42fc1ba9c428b5347b0.json`.
It records `terminal-development-clock-or-deadline-rejection`, authorizes no
retry, source activation, upload, held-out work, or arena action, and leaves
canonical Rank 4 as the incumbent.

## Verification

Build the existing generic opening-bank tool, then run the read-only check:

```sh
cmake --build build --target papersoccer_opening_bank -j4
python3 results/rank_4_jacek_hybrid/tools/freeze_campaign.py check
```

The expected manifest SHA-256 is
`aed2a52f7a59c2b1988b5c365c23b57f8ec41fbfb50927211655a8565df63fa7`.
The verifier fails if Rank 4, any protected tree, any new bank, the arena
exclusion registry, or the absent `matches.json` boundary changes.

CodinGame upload bytes remain editor-attested/fingerprinted; the public API
does not expose remote source bytes or a remote digest.
