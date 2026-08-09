# CodinGame submissions

CodinGame support is split into reusable tooling and self-contained bots:

```text
submissions/codingame/
├── bots/                 maintained source, tests, data, and output per bot
│   ├── alpha_beta/       production baseline and its experiment archive
│   ├── rank_5/           immutable previous arena incumbent
│   ├── rank_4/           current live teacher-residual production bot
│   ├── rank_4_exchange/  unpromoted exact-exchange proof experiment
│   ├── challenger/       completed exact-topology challenger experiment
│   ├── selfplay_nn/      learned value and self-play training experiment
│   ├── selfplay_nn_v2/   historical rank-4 training and arena provenance
│   ├── jacek_nn/         Jacek-input, rank-5-anchored residual experiment
│   └── topology/         goal-connectivity ordering experiment
├── promotion/            frozen elite banks and promotion manifest
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
node submissions/codingame/tools/generate_submission.mjs rank_4 --check
node submissions/codingame/tools/generate_submission.mjs rank_4_exchange --check
node submissions/codingame/tools/generate_submission.mjs selfplay_nn --check
node submissions/codingame/tools/generate_submission.mjs selfplay_nn_v2 --check
node submissions/codingame/tools/generate_submission.mjs jacek_nn --check
node submissions/codingame/tools/generate_submission.mjs topology --check
python3 submissions/codingame/tools/promotion_gate.py validate
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Do not edit a `submission.cpp` directly. Change the maintained bot source or
data and regenerate it. See [bots/README.md](bots/README.md) for the directory
contract and the shortest path to starting another bot. Shared command details
are in [tools/README.md](tools/README.md).

## Current bots

- [`rank_4`](bots/rank_4/README.md) is the current live production bot:
  history version 56, agent `6604719`, submission `41114327`, rank 4 of 208
  with score `44.29750553418035` and a 66-24 record. It is the canonical,
  byte-identical local snapshot of the retained `selfplay_nn_v2` artifact;
  the historical training and arena inputs stay at their original paths.
- [`rank_4_exchange`](bots/rank_4_exchange/README.md) adds exact rebound
  win/loss proofs to rank 4. It beat rank 4 58-48 at both 5k and 30k nodes,
  but lost 51-55 at 10k and 8-10 on the final equal-clock screen, so it
  remains unsubmitted.
- [`alpha_beta`](bots/alpha_beta/README.md) is the prior production baseline
  and retains its compact historical experiment evidence. Its paste-ready file
  is [`submission.cpp`](bots/alpha_beta/submission.cpp).
- [`rank_5`](bots/rank_5/README.md) is the immutable previous incumbent:
  version 26, agent `6561779`, rank 5 of 206 with score
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
- [`selfplay_nn_v2`](bots/selfplay_nn_v2/README.md) preserves the original
  training corpus, model provenance, and version 42-44 arena evidence behind
  `rank_4`. Its paths remain unchanged because frozen promotion and leakage
  registries bind those historical artifacts by path and hash.
- [`jacek_nn`](bots/jacek_nn/README.md) is the separate Jacek-input
  challenger. It preserves rank_5's complete search, replay book, and compact
  `1156 -> 8 -> 8 -> 1` value anchor, then adds a bounded 24-feature residual
  trained from 10,000 games of soft search targets. At its selected 50%
  strength it beat rank_5 174-132 over 306 games at 5k nodes and 58-48 over
  106 games at 30k nodes. The identical 800/165 source scored 42.3232 at 59-31
  in v47 and 40.3909 at 61-29 in v49; their mean 41.3570 did not beat the
  incumbent's 42.4277 weighted score, so the bot remains unpromoted. The
  independent v48 650/130 clock ablation scored 39.7385. Earlier 32x
  replacement batches and post-v49 deeper-training/search experiments were
  also rejected. The retained paste-ready source is 98,623 characters,
  SHA-256
  `fb570f7d60157ad1681569011b4249a5db415c1aeca6f665936b26ba5cc52102`;
  complete evidence is in
  [`EXPERIMENTS.md`](bots/jacek_nn/EXPERIMENTS.md).
- [`topology`](bots/topology/README.md) is the capped rebound-goal connectivity
  ordering experiment. It was rejected on the frozen 96-pair development bank
  at 94-98 before validation or live submission. Its experiment ledger and the
  reusable disjoint-bank promotion harness preserve the result.
