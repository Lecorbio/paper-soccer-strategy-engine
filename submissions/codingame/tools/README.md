# Shared CodinGame tools

This directory contains strategy-independent utilities. Bot implementations,
models, replay books, timing probes, and experiment evidence belong under
`../bots/<name>/`.

## Submission generation

```sh
node submissions/codingame/tools/generate_submission.mjs BOT_NAME [--check]
```

The generator reads `bots/BOT_NAME/submission.json` and `sources.txt`, runs any
configured bot-local generators, removes repository-local includes, combines
standard headers, strips comments and redundant blank lines, rejects unexpected
dependencies, and enforces the configured source-size limit.

## Protocol smoke test

`protocol_smoke_test.mjs` starts a compiled submission as each player ID and
checks its first protocol response. CTest invokes it for every registered bot.

## Arena replay analysis

```sh
python3 submissions/codingame/tools/analyze_arena.py AGENT_ID --pretty
python3 submissions/codingame/tools/screen_replay_book.py \
  AGENT_ID submissions/codingame/bots/BOT_NAME/replay_book.json
```

The first command downloads completed public battles and summarizes opponents
and same-color divergences. The second measures which completed games a replay
book would change first.
