# Bot directory contract

Each directory under `bots/` is an independently maintainable CodinGame bot.
Use the same filenames so shared tooling and CMake can treat every bot alike:

- `submission.json` configures generation, allowed local includes, the source
  limit, and optional bot-specific data generators.
- `sources.txt` lists source files in concatenation order, relative to the
  repository root.
- `bot.cpp` contains the maintained strategy and CodinGame protocol entrypoint.
- `submission.cpp` is generated, self-contained, and ready to paste.
- `submission_test.cpp` tests the self-contained artifact.
- `README.md` records the bot's design, arena evidence, and commands.
- Optional model, replay-book, experiment, local-gate, and timing files remain
  in that bot directory.

## Start a new bot

1. Copy the existing bot closest to the design you want into `bots/<new_name>`.
2. Remove copied experiment evidence and optional data generators that the new
   bot does not use.
3. Update `submission.json`, `sources.txt`, `bot.cpp`, the tests, and README.
4. Add `<new_name>` to `PAPERSOCCER_CODINGAME_BOTS` in the root `CMakeLists.txt`.
5. Generate and run the standard gates:

```sh
node submissions/codingame/tools/generate_submission.mjs <new_name>
node submissions/codingame/tools/generate_submission.mjs <new_name> --check
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

The shared generator automatically runs scripts listed in the config's
`generators` array before assembling the submission. Each listed generator must
accept an optional `--check`, write only its bot-owned artifacts in normal mode,
and fail when those artifacts are stale in check mode.
