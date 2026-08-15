# Full-development proof-mask gate plan

Frozen at `2026-08-13T22:13:35Z`, before either full-development result was
observed. This is DEVELOPMENT-only selection evidence. It neither opens a
validation/final bank nor authorizes an arena upload.

## Candidate and controls

- Candidate gate configuration: exact proof mask `7` (root, leaf, and ply-one
  scopes), with the mover-relative tie fix. The mask-capable generated source
  used to compile the gate has SHA-256
  `6f3abb4bed53050937ee36789ec5cf1bfc22ad02f0ea13e7db6575a11ec06d6f`
  and 94,004 ASCII bytes. Its ordinary protocol path still selects mask `15`
  at this freeze; it must not be called the final mask-`7` source. Only after
  both development comparisons pass may that one operational constant change,
  followed by source regeneration and all final-source gates.
- Same-binary control: mask `0` with all other candidate code unchanged.
- External control: frozen Rank-4 source SHA-256
  `5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`
  and 98,624 ASCII bytes.
- Fixed settings: 3,000,000-node ceilings, first/later clocks 800/165 ms,
  operational clocks 1000/200 ms, maximum 320 turns, replay corrections
  disabled, and no retained match transcripts.

The four preregistered DEVELOPMENT banks are used together, in their already
frozen order: depth 4 (78 paired-color games), then depths 8, 12, and 20 (76
each), for exactly 306 games and 153 games in each physical color. The sealed
validation and final banks remain unread.

## Sequential decision rule

1. Run mask `7` against the same-binary mask-`0` control. Advance only with
   zero unfinished games, failures, illegal actions, operational failures,
   exceptions, or hard timeouts; at least `160-146` overall; and at least 77
   candidate wins in each physical color.
2. Only after step 1 is accepted, run mask `7` against exact Rank 4. Retain
   mask `7` only with the same zero-failure requirements, at least `160-146`
   overall, and at least 77 candidate wins in each physical color.
3. Both runs must have internally consistent bank/aggregate outcome, color,
   engine-work, proof-scope, and timing accounting. Candidate and reference
   first-decision maxima must remain below 990 ms and later-decision maxima
   below 198 ms; p99 must be finite, nonnegative, and no greater than max.

Failure of either comparison rejects mask `7` as a full-development winner.
The preregistered fallback is mask `3` (root plus leaf): it had the strongest
positive conditional margin in the completed nested matrix, while mask `1`
was neutral and masks `7`/`15` add successively more work. Mask `3` must then
pass the same two sequential 306-game comparisons and the same thresholds;
it receives no result-dependent relaxation. Masks `1` and `15` do not advance
under this plan. No losing scope is bundled with a later hypothesis. Passing
both comparisons merely freezes a candidate for fresh-source gates and the
untouched validation bank.
