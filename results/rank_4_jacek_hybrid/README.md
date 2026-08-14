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

Do not run, parse, summarize, or inspect a `final_*.tsv` result until exactly
one finalist source byte count and SHA-256 have been locked.  Merely hashing
the frozen bank files is permitted.

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
