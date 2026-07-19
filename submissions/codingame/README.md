# CodinGame submissions

CodinGame support is split into reusable tooling and self-contained bots:

```text
submissions/codingame/
├── bots/                 maintained source, tests, data, and output per bot
│   ├── alpha_beta/       production baseline and its experiment archive
│   └── rank_5/           strongest verified arena candidate
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
