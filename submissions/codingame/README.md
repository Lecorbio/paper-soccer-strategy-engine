# CodinGame submissions

CodinGame support is split into reusable tooling and self-contained bots:

```text
submissions/codingame/
├── bots/                 maintained source, tests, data, and output per bot
│   ├── alpha_beta/       production baseline and its experiment archive
│   ├── rank_5/           strongest verified arena candidate
│   ├── challenger/       completed exact-topology challenger experiment
│   └── selfplay_nn/      learned value and self-play training experiment
└── tools/                shared generation, protocol, and replay utilities
```

Every bot uses the same small contract. Its directory contains
`submission.json`, an ordered `sources.txt`, maintained implementation files,
`submission_test.cpp`, and the generated paste-ready `submission.cpp`. Optional
data generators and timing probes stay beside the bot that owns them. Historical
experiments also stay with their bot instead of becoming shared dependencies.

Generate or verify either submission from the repository root:

```sh
node submissions/codingame/tools/generate_submission.mjs alpha_beta
node submissions/codingame/tools/generate_submission.mjs alpha_beta --check
node submissions/codingame/tools/generate_submission.mjs rank_5 --check
node submissions/codingame/tools/generate_submission.mjs selfplay_nn --check
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Do not edit a `submission.cpp` directly. Change the maintained bot source or
data and regenerate it. See [bots/README.md](bots/README.md) for the directory
contract and the shortest path to starting another bot. Shared command details
are in [tools/README.md](tools/README.md).

## Current bots

- [`alpha_beta`](bots/alpha_beta/README.md) is the prior production baseline
  and retains its compact historical experiment evidence. Its paste-ready file
  is [`submission.cpp`](bots/alpha_beta/submission.cpp).
- [`rank_5`](bots/rank_5/README.md) is the strongest completed arena
  candidate: version 26, agent `6561779`, rank 5 of 206 with score
  `42.42773147296124`. Its paste-ready file is
  [`submission.cpp`](bots/rank_5/submission.cpp).
- [`challenger`](bots/challenger/README.md) is the independently maintained
  candidate with exact rebound analysis, audited search experiments, and
  narrow replay repairs. Its strongest completed result is version 38 at rank
  5 with score `42.280679071799426`, below the accepted `rank_5` bot; the full
  evidence is recorded in [`EXPERIMENTS.md`](bots/challenger/EXPERIMENTS.md).
- [`selfplay_nn`](bots/selfplay_nn/README.md) is the compact learned-value
  experiment with a reproducible 1,024-game training corpus. Its strongest
  completed result is version 41 at rank 5 with score `41.77236003585933`,
  below the accepted `rank_5` bot; all three live batches are recorded in
  [`EXPERIMENTS.md`](bots/selfplay_nn/EXPERIMENTS.md).
