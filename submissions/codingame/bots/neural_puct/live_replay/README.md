# Neural PUCT live replay archive

This directory is the append-only public CodinGame evidence store used by the
protocol in [`../LIVE_REPLAY_LOOP.md`](../LIVE_REPLAY_LOOP.md). The collector
freezes an exclusion registry before making any public API request, fetches
battle metadata before replay details, and requests a detail only when the game
is unseen, complete, and contains a current strong player whose actions are
eligible for direct policy supervision.

The store is content addressed. Existing files are verified and never
replaced:

- `exclusions/` contains deterministic ID-only boundary registries. Promotion,
  prospective/final, rank-one, development, and arena-diagnostic files
  contribute hashes and game IDs only; their actions are not loaded by the
  collector. The root `matches.json` is never read.
- `raw/` preserves the exact response bytes. `normalized/` preserves canonical
  JSON for conflict detection. `receipts/` binds each request schema, payload,
  timestamp, response hashes, status, and retry count.
- `replay_payloads/` and `games/` contain rule-validated accepted games.
  `discoveries/` binds genuinely new IDs to a collection run.
- `events/` records acceptances and deterministic rejection reasons. Conflicting
  normalized payloads are preserved under `conflicts/` and quarantined.
- `polls/` records local cursors, leaderboard rank/score snapshots, agent and
  session IDs, metadata-window hashes, and every per-game decision. `runs/`
  records the threshold decision across polls.
- `active_submissions/` binds authenticated UI source inspection to the public
  agent, submission, and session identity. `decisions/` records whether model
  work or another upload was authorized.

The local cursor is an audit high-water mark, not a filtering shortcut. An
unseen game below the cursor is still processed. Overlapping top-player
windows are deduplicated by game ID before any detail request.

Build the deterministic exclusion registry without networking:

```sh
.venv/bin/python \
  submissions/codingame/bots/neural_puct/collect_live_replays.py \
  --build-exclusions-only
```

Resume the first collection run using the already active neural agent:

```sh
.venv/bin/python \
  submissions/codingame/bots/neural_puct/collect_live_replays.py \
  --run-id first-live-replay-20260809 \
  --polls 1 --minimum-new-games 50 \
  --initial-top 5 --expanded-top 20 \
  --focus-agent-id 6604625 --own-agent-id 6604625 \
  --maximum-workers 2
```

Validate every content-addressed response and accepted payload offline:

```sh
.venv/bin/python \
  submissions/codingame/bots/neural_puct/collect_live_replays.py --check
```

The exact collector used for the first live run has SHA-256
`4272f3ba3ada8ce7b9377d4b3c0647633ea0af4b2a4d4be6d00d9d08ddf0176b`
and is preserved at Git commit `c5958cd`. The maintained follow-up fixes only
network-request accounting and makes permanent structural rejections part of
future exclusion registries; it does not change any archived response or game
decision. Its SHA-256 is
`5c73139e9607ea97477aef212b6c015207cf66fb462fda8759003f98a4e897ef`.
