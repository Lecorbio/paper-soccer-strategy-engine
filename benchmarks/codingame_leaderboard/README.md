# Local CodinGame-style leaderboard artifact

This directory owns the frozen 20-entrant roster, deterministic 900-game
schedule, TrueSkill contract, serial tournament runner, raw artifact validator,
and compact website publisher. The browser never runs bots; it consumes only a
reviewed `leaderboard-results.js` derived from a complete native tournament.

The local score is not claimed to equal CodinGame's unpublished complete
matchmaking and rating contract. Games use the public Paper Soccer rules and
protocol through the repository's native referee, with 1,000 ms for each bot's
first response and 200 ms for later responses.

## Validate contracts

From the repository root:

```sh
python3 benchmarks/codingame_leaderboard/leaderboard.py validate
```

This checks that all 23 CMake-registered generated submissions are accounted
for. The frozen tournament keeps its original 20 entrants: the unqualified
`rank_4_jacek_hybrid` and `jacek_arena_bfm` campaign artifacts are explicit
non-entrants, while `selfplay_nn_v2` remains a byte-identical alias of
`rank_4`. Source hashes must be current, and the seeded schedule must retain
all frozen balance properties.

## Run or resume

Build the Release referee and all submission executables, then run games
serially:

```sh
python3 benchmarks/codingame_leaderboard/leaderboard.py run \
  --referee build/leaderboard/papersoccer_codingame_referee \
  --build-dir build/leaderboard \
  --checkpoint build/leaderboard/leaderboard-checkpoint.json \
  --output benchmarks/codingame_leaderboard/tournament.json
```

If interrupted, repeat the command with `--resume`. Resume fails closed when
the roster, schedule, contract sources, executables, referee, source tree, or
runtime environment differs from the checkpoint. The runner uses direct argv,
a minimal environment, a temporary working directory, bounded output, and a
process-group watchdog; a nonzero referee exit aborts rather than scoring a bot.

`--stop-after N` is available for smoke testing and intentionally leaves only a
resumable checkpoint when fewer than all 900 games have run.

## Validate and publish

After the complete run:

```sh
python3 benchmarks/codingame_leaderboard/leaderboard.py validate \
  --artifact benchmarks/codingame_leaderboard/tournament.json \
  --referee build/leaderboard/papersoccer_codingame_referee
python3 benchmarks/codingame_leaderboard/leaderboard.py publish \
  --input benchmarks/codingame_leaderboard/tournament.json \
  --output web/leaderboard/leaderboard-results.js \
  --referee build/leaderboard/papersoccer_codingame_referee
python3 benchmarks/codingame_leaderboard/leaderboard.py publish \
  --input benchmarks/codingame_leaderboard/tournament.json \
  --output web/leaderboard/leaderboard-results.js \
  --referee build/leaderboard/papersoccer_codingame_referee --check
```

Do not publish a partial or synthetic development snapshot. The checked-in raw
artifact must contain all 900 matches and the compact output is a deterministic
function of it.
