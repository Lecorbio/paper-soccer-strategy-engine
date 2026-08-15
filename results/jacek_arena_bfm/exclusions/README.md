# Fresh-arena exclusion inputs

This directory contains metadata-only campaign boundary artifacts frozen after
`T0 = 2026-08-13T10:12:52Z`.  It never contains historical replay content.

- `protected-diagnostic-game-ids.txt` is the sorted inventory of numeric
  directory names shared by the protected `game_records` and
  `replay_payloads` archives in the original worktree.
- `t0-h62-agent-battles.raw.json` is the public battle-list response for the
  frozen H62 agent (`6613798`) fetched after T0.  It contains battle metadata
  only and was acquired without requesting any game replay detail.

The content-addressed exclusion registry generated from these inputs, the
tracked ID-only registry, and the T0 metadata snapshot is the sole registry
accepted by this campaign's arena collectors.
