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
- `corpora/` binds whole-game training snapshots, deterministic relabel input,
  and deeper neural-only relabel output. `candidates/` stores deterministic
  content-addressed seed artifacts, and `evaluations/` stores the allowed
  exposed gate reports.

The local cursor is an audit high-water mark, not a filtering shortcut. An
unseen game below the cursor is still processed. Overlapping top-player
windows are deduplicated by game ID before any detail request.

Build the deterministic exclusion registry without networking:

```sh
.venv/bin/python \
  submissions/codingame/bots/neural_puct/collect_live_replays.py \
  --build-exclusions-only
```

Resume the first collection run using both maintained collection identities:

```sh
.venv/bin/python \
  submissions/codingame/bots/neural_puct/collect_live_replays.py \
  --run-id first-live-replay-20260809 \
  --polls 1 --minimum-new-games 50 \
  --initial-top 5 --expanded-top 20 \
  --focus-agent-id 6604698 \
  --own-agent-id 6604625 --own-agent-id 6604698 \
  --maximum-workers 2
```

Freeze the threshold-crossing snapshot and reproduce its deeper relabels:

```sh
.venv/bin/python \
  submissions/codingame/bots/neural_puct/build_live_replay_corpus.py \
  --run-id first-live-replay-20260809 --minimum-games 50 \
  --own-agent-id 6604625 --own-agent-id 6604698

cmake --build build --target \
  papersoccer_codingame_neural_puct_live_replay_relabeler
build/papersoccer_codingame_neural_puct_live_replay_relabeler \
  submissions/codingame/bots/neural_puct/live_replay/corpora/a6ba0d2e76b44d22432070e85bf215e6ab395e361d978b43e1915e05965ac3da.relabel.tsv \
  /tmp/live-relabel.jsonl 10000 100000
```

The executed relabel output has SHA-256
`e406c9ad5d3796524e29ba2e052a84f778c43ab8f1eb7c5817f6ad1da6ff2e98`.
Train a candidate seed with `--policy-head shared-action-conditioned-v1`, the
snapshot supplied through `--live-replay-manifest`, the relabel JSONL through
`--live-relabel-corpus`, and `--live-mass 0.25`.

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

The completed first iteration is documented in
[`FIRST_ITERATION_REPORT.md`](FIRST_ITERATION_REPORT.md). It crossed the
threshold with 68 games, trained three action-conditioned seeds, passed the
default and fixed-work gates, then failed equal-clock development 29-67. No
performance candidate was submitted.
