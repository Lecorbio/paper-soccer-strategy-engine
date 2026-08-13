# Fresh-arena campaign provenance

`campaign_provenance.py` is the clean-room boundary for the five-hour
`jacek_arena_bfm` campaign. It imports no historical bot, model, replay parser,
label, or training pipeline.

## Frozen campaign artifacts

- T0: `2026-08-13T10:12:52Z`.
- Exclusion registry:
  `results/jacek_arena_bfm/exclusions/0ce7cff1d29cbceab1d06a9066eebb6647a31515437cd468ff1a9b1b5f8407f9.json`.
  It contains 4,025 game IDs. Its only new inputs are a sorted inventory of
  numeric protected-directory names and a strictly validated 90-item battle
  metadata snapshot. The builder rejects frames, stdin/stdout/stderr,
  transcripts, turns, replay fields, and every unexpected battle key.
- Blind window plan:
  `results/jacek_arena_bfm/window_plans/0ca942c0a05af9e80197b5c95ea3ef32aed78192753f313a1015a22bd3b90d29.json`.
  Eight collection slots were assigned `training, training, training,
  arena-validation` twice before any fresh result, followed by the untouched
  finalist holdout and optional safe-rollback accounting window. Every slot is
  exactly 90 games.

## Command interfaces

Run `python3 campaign_provenance.py COMMAND --help` for complete arguments.

- `build-exclusions` accepts `--base-registry`, its approved SHA-256, one or
  more `--protected-inventory` files, one or more `--battle-snapshot` files,
  T0, and an output directory. It writes canonical JSON named by its SHA-256
  and reloads it through the collector-compatible ID-only schema before
  returning.
- `create-window-plan` freezes the 3-train/1-validation cadence and campaign
  cutoffs in schema `papersoccer.jacek-arena-bfm.window-plan.v1`.
- `attest-editor` requires every compilation/protocol/legal-action/purity/
  source-size/both-color-timing flag, a legal Play stdout flag, expected Play
  telemetry, and a copied-back source file. Generated and copy-back files must
  be identical ASCII and at most 99,999 characters. It binds commit, source
  SHA, agent ID, submission ID, upload time, and planned window. CodinGame
  upload bytes remain **editor-attested, not API-readable**.
  The rollback slot is additionally hard-bound to the frozen safe H62 source
  SHA `d9d96f83197f13b7212e7b652851097053ee7f1662845e06dd722d1c0bc24f71`
  and exactly 99,810 ASCII bytes; it may be used immediately as an emergency
  path or after the scheduled T+4:20 retain/rollback decision.
- `derive-arena-window` accepts only content-addressed plan, attestation,
  exclusion registry, and generic arena manifest files with approved hashes.
  It requires a fully accounted exact 90-game window, exact source/agent/
  submission identity, campaign cutoffs, post-T0 game IDs, and immutable
  per-game raw and normalized payloads. A timeout, illegal/malformed/partial
  transcript, operational signal from either player, or ambiguous terminal
  state rejects the entire game. Training roles expose only raw terminal-value
  candidates and opponent-action **reanalysis candidates**; validation and
  final-holdout roles can never produce training uses.
- `validate-derivation` is the corpus-side trust gate. It rechecks the
  content-addressed upstream plan, attestation, arena manifest, and exclusion
  registry; exact role-derived uses; sorted fresh IDs; all payload/record
  hashes; and summary counts before returning `status: valid`.
- `validate-sequence` accepts repeated derivations and rejects a skipped or
  reordered collection slot, repeated/non-distinct submission source, upload
  before the preceding exact window completed, experimental upload after the
  finalist, or rollback that does not follow a completed finalist window.
- `snapshot-protected` records explicit protected path topology, modes, sizes,
  and SHA-256 values without parsing or modifying file content. It refuses `/`,
  the home directory, overlapping roots, and an output directory overlapping a
  protected root. `verify-protected` fails if any protected byte, mode, or path
  changed.

All emitted manifests use canonical JSON with a trailing newline and a
filename equal to the SHA-256 of those exact bytes. Existing evidence paths are
never written by these commands.
