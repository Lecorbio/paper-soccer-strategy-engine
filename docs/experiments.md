# Experiments and benchmark evidence

The project separates deterministic correctness checks from longer strength and
latency experiments. CTest uses small budgets to validate accounting, legal
moves, report schemas, and command behavior. Promotion decisions use explicit
paired gates and retain negative results as well as successes.

Raw local output belongs in ignored `results/` paths. Curated, compact reports
that support general benchmark claims remain tracked under `benchmarks/`, while
the rank-4/Jacek hybrid campaign and selected fresh-arena evidence use explicit
tracked allowlists under `results/rank_4_jacek_hybrid/` and
`results/jacek_arena_bfm/`. Verified CodinGame source artifacts and their
bot-specific historical ledgers stay under `submissions/codingame/`.

The accepted local large-teacher campaign follows that split: its multi-gigabyte
models, labels, banks, shards, and logs remain ignored locally, while a compact
self-verifying outcome is tracked under
[`benchmarks/large_teacher_campaign/`](../benchmarks/large_teacher_campaign/REPORT.md).
That record publishes scores, paired outcomes, latency samples, retention
metrics, and provenance hashes without publishing the teacher runtime or
changing its noncanonical status.

For a reader-focused presentation of the current frozen four-bot result, see
the live [benchmark overview](https://lecorbio.github.io/paper-soccer-strategy-engine/benchmarks/).
The [flagship study report](../benchmarks/flagship_study/REPORT.md) retains the
full technical analysis.

## Arena methodology

The native arena has two JSON-report modes:

- `matches` creates fresh bots for deterministic seed pairs and swaps colors in
  the second game of every pair.
- `positions` measures both entrants on the same generated non-terminal states.

Optional random openings are generated once per pair, so both color assignments
start from the same state. Opening generation uses an independent seed stream
and records every accepted seed and move sequence. Enabling openings therefore
does not change bot seeds.

Both modes emit `papersoccer.arena.v1`. Match reports include configurations,
lossless string seeds, outcomes, plies, per-decision timings and search
counters, color splits, throughput, and a deterministic paired-bootstrap 95%
interval. Position reports contain equivalent per-entrant work on shared states.

Create a release build and reserve the local output directory:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
mkdir -p results
```

The default comparison is experimental Tactical + TacticalQuiescence MCTS
against the frozen Tactical + RolloutOnly MCTS reference, both at 2,000 new
iterations and the same exploration and tree-node limits:

```bash
./build/release/papersoccer_arena positions --positions 16 \
  > results/quiescence-positions.json

./build/release/papersoccer_arena matches --pairs 20 \
  > results/quiescence-equal-iterations-20-pairs.json

./build/release/papersoccer_arena matches --pairs 20 --opening-plies 8 \
  > results/quiescence-randomized-openings.json
```

Use `--opening-plies N` for independent 4-, 8-, 12-, or 20-ply banks rather
than mixing game phases inside one pair definition. `--max-plies` counts opening
moves toward the total limit; timing and search statistics cover only moves
chosen by the compared bots.

Quiescence is configured independently with `--candidate-leaf-policy`,
`--reference-leaf-policy`, and their `--*-quiescence-max-depth` and
`--*-quiescence-max-nodes` limits. `--help` lists policy, reuse, iterations,
node limits, exploration, rules, and sampling options.

The offline Jacek replay BFM entrant requires the exact CodinGame rule profile
and an external versioned checkpoint. A loader/search smoke run with the
checked-in, explicitly untrained bootstrap checkpoint is:

```bash
./build/release/papersoccer_arena positions \
  --rules codingame \
  --positions 16 \
  --candidate-kind jacek-replay-bfm \
  --candidate-jacek-replay-model \
    models/jacek_replay_bfm_development/jacek_replay_bfm.runtime \
  --candidate-jacek-replay-max-time-ms 980 \
  --candidate-jacek-replay-max-tree-nodes 1000000 \
  --reference-kind random \
  > results/jacek-replay-bfm-positions.json
```

That bootstrap verifies the binary contract and runtime path only; it is not a
selected model and must not support a strength claim.

Candidate and reference variants of the model, time, tree-node, retained-action,
partial-path, exploration, and FPU options are available. Reports bind the
feature schema in configuration and record the loaded model SHA-256 plus BFM
action-generation, neural-evaluation, visit, tree, tactical, truncation,
deadline, and cached-continuation diagnostics on every decision.

## Alpha-beta evaluations

### Initial smoke evaluation — July 16, 2026

With the original 10-ply soft physical horizon, eight shared positions and ten
color-swapped pairs compared alpha-beta with Tactical MCTS at 2,000 iterations:

| Check | Alpha-beta | MCTS reference |
| --- | ---: | ---: |
| Shared-position median decision time | 3.457 ms | 56.075 ms |
| Shared-position p95 decision time | 41.954 ms | 63.333 ms |
| Match median decision time | 2.460 ms | 41.978 ms |
| Match p95 decision time | 7.043 ms | 77.758 ms |
| Ten-pair record | 13 wins / 7 losses | 7 wins / 13 losses |

There were no illegal moves, truncations, or alpha-beta node-budget
exhaustions. The paired interval was wide at 40%–90%, so the 65% score was smoke
evidence, not a strength conclusion. Timings are machine-specific.

### Soft-horizon promotion — July 18, 2026

The paired-opening arena compared a 12-ply candidate directly with the earlier
10-ply alpha-beta bot. Both used possession depth 6, a 100,000-node hard limit,
and identical 12-ply random opening states:

| Opening bank | 12-ply record | Score | Paired 95% interval | 12-ply median | 10-ply median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default seed, 50 pairs / 100 games | 64-36 | 64.0% | 55.0%–73.0% | 5.133 ms | 2.035 ms |
| Independent seed, 30 pairs / 60 games | 35-25 | 58.3% | 48.3%–68.3% | 5.042 ms | 2.157 ms |

The combined record was 99-61 (61.9%), with no illegal moves or truncations.
Both banks favored 12 plies from both colors. The change uses more of the
existing node allowance; it does not increase that allowance. This evidence
promoted 12 plies to the default. Timings are machine-specific.

Reproduce an alpha-beta versus MCTS report without writing raw JSON at the
repository root:

```bash
./build/release/papersoccer_arena positions \
  --positions 32 \
  --candidate-kind alpha-beta \
  --candidate-alpha-beta-depth 6 \
  --candidate-alpha-beta-max-nodes 100000 \
  --reference-kind mcts \
  --reference-iterations 2000 \
  > results/alpha-beta-positions.json

./build/release/papersoccer_arena matches \
  --pairs 200 \
  --candidate-kind alpha-beta \
  --candidate-alpha-beta-depth 6 \
  --candidate-alpha-beta-max-nodes 100000 \
  --reference-kind mcts \
  --reference-iterations 2000 \
  > results/alpha-beta-vs-mcts.json
```

Alpha-beta reports keep completed/attempted depth, node and leaf counts,
terminal nodes, pruning and transposition activity, physical-horizon cutoffs,
root scores/bounds, principal variations, and budget exhaustion separate from
MCTS statistics. Timing summaries include median nodes per second.

## Tactical MCTS promotion

The current Tactical + RolloutOnly MCTS baseline was promoted on July 14, 2026
after a 389-11 result over 200 color-swapped seed pairs (400 games) against the
older Uniform, non-reusing MCTS reference. The recorded paired-bootstrap 95%
interval was 95.5%–98.75%, with no illegal moves or truncations.

This is a comparison with a specific older in-repository baseline, not a general
97.25% win-rate claim and not evidence about a CodinGame submission. The full
promotion tournament is intentionally not part of CTest. Its historical summary
is preserved here; a raw compact report was not committed with the original
July 14 result.

## Experimental tactical quiescence

The later staged evaluation kept TacticalQuiescence experimental. The exact
equal-iteration preliminary command was:

```bash
./build/release/papersoccer_arena matches \
  --pairs 20 \
  --seed 828927513140 \
  --candidate-policy tactical \
  --candidate-leaf-policy tactical-quiescence \
  --candidate-quiescence-max-depth 8 \
  --candidate-quiescence-max-nodes 256 \
  --candidate-reuse true \
  --candidate-iterations 2000 \
  --reference-policy tactical \
  --reference-leaf-policy rollout-only \
  --reference-reuse true \
  --reference-iterations 2000 \
  > results/quiescence-equal-iterations-20-pairs.json
```

Equal iteration counts do not mean equal latency because every quiescence
iteration can perform extra tactical search. Shared-position measurements were
used to choose an approximately latency-matched 1,250-iteration candidate:

```bash
./build/release/papersoccer_arena positions \
  --positions 32 \
  --candidate-iterations 2000 \
  --reference-iterations 2000 \
  > results/quiescence-latency-tuning.json

./build/release/papersoccer_arena matches \
  --pairs 20 \
  --candidate-iterations 1250 \
  --reference-iterations 2000 \
  > results/quiescence-latency-matched-20-pairs.json
```

Recorded results:

| Evaluation | Result |
| --- | --- |
| Correctness gate | Release, ASan/UBSan, browser/Wasm, compact-rule parity, restoration, cutoff, proof-soundness, reproducibility, and native/Wasm arena smoke checks passed; zero illegal moves or unexpected truncations |
| Equal-iteration record | 19-21 at 2,000 versus 2,000; 47.5%; paired interval 37.5%–57.5% |
| Equal-iteration work | Candidate: 4,093,635 iterations, 3,947,878 probes, 65,477 solved (1.659%); reference: 4,342,896 iterations, zero probes |
| Native match median / p95 | Candidate 36.757 / 66.853 ms; reference 22.683 / 45.968 ms |
| Native position median / p95 | Candidate 49.862 / 61.204 ms; reference 33.049 / 36.004 ms |
| Wasm position median / p95 | Candidate 56.088 / 71.932 ms; reference 37.384 / 41.799 ms; deterministic counters matched native |
| Latency-matched record | 11-29 at 1,250 versus 2,000; 27.5%; paired interval 12.5%–42.5% |
| Latency-matched work | Candidate 2,433,972 iterations, 2,341,396 probes, 42,004 solved (1.794%), median/p95 23.459/43.570 ms; reference 4,004,291 iterations, median/p95 23.608/47.535 ms |
| Full 200-pair gate | Not run because the preliminary strength gate failed |

The probe solved real frontier positions and remained operationally healthy,
but its low solution rate did not offset its cost. Reducing iterations to match
latency weakened it further. These are negative promotion results;
`RolloutOnly` remains the engine and website default.

The frozen regression fixture derived from arena pair 103 preserves an escape:
Player Two at `(6,2)` has five defenses, and `(5,3)` is the only one the bounded
proof search does not refute. The frozen Tactical bot chose `(7,3)` and allowed
a same-player rebound attack to goal. The proof probe must keep the root Unknown
because the escape survives, even while proving representative alternatives
losing. No human-match export was supplied for that fixture.

Promotion would require correctness fixtures, supplied exploit regressions,
operational health, acceptable native/Wasm latency, and a full paired gate whose
95% lower bound exceeds 50%.

## Jacek-inspired demo gates

The browser checkpoint is independently trained and restricted to normal demo
rules. Fixed-seed gates recorded:

| Gate | Result | Jacek median / p95 | Safety |
| --- | --- | ---: | --- |
| 32 shared midgames versus equal-budget AlphaBetaBot | 5/32 move changes | 27.68 / 36.76 ms | 0 illegal |
| 12 pairs / 24 games versus equal-budget AlphaBetaBot | 12-12 | 11.85 / 31.31 ms | 0 illegal, 0 truncated |
| 12 pairs / 24 games versus Tactical MctsBot at 2,000 iterations | 21-3 | 15.59 / 32.85 ms | 0 illegal, 0 truncated |
| 20 pairs / 40 games versus RandomBot | 39-1 | 21.59 / 34.59 ms | 0 illegal, 0 truncated |

The checked-in Wasm path probe produced 80 decisions over 12 scripted games and
46 positions: 21.53 ms median, 40.64 ms p95, 43.69 ms maximum, 13 node-cap
interruptions, and no invalid root scores. It is a deterministic path probe,
not a general browser-latency guarantee.

The complete feature definition, held-out metrics, hashes, clean retraining
record, commands, depth histograms, and horizon ablations remain canonical in
[the model record](../models/README.md). None of these gates claims to match
Jacek Dermont's unpublished contest bot or to improve the repository's
CodinGame rank.

## Game Review strength, calibration, and Expert gate

Game Review has its own frozen study. It neither edits the flagship study nor
uses any of its openings for development, validation, calibration, or test.
The manifest freezes horizontal-reflection-aware opening identities at 4,
8, 12, and 20 physical plies and explicitly excludes every flagship bank.

The three `DeepTurnSearchBot` candidates are depth 32 with 100,000, 200,000, or
400,000 nodes. Each keeps 65,536 transposition entries, 32,768 evaluation-cache
entries, no clock, no replay corrections, and a 0% learned-value blend. They
are evaluated against both references:

- `Rank5DerivedBot`, the immutable 50,000-node demo profile; and
- the selected `JacekInspiredBot`, depth 6 and 20,000 nodes.

Development uses 25 color-swapped pairs per opening depth, reference, and
candidate. Validation uses 50 pairs for the same cells. Eligible candidates
within one percentage point of the validation leader form the selection set;
the gate chooses lower Wasm p95 latency, then lower node budget. A candidate
must also measure at most 400 ms p95 and 750 ms maximum in the dedicated Wasm
artifact. Every Wasm recommendation and deterministic counter in the latency
sample is checked against the corresponding native validation decision.

Validation decisions fit two separate logistic maps: one for the fixed Fast
50k profile and one for the selected Deep profile. Scores are oriented to the
player making the possession and outcomes are eventual binary wins. The fit
never reads development or held-out test outcomes and never applies one
profile's coefficients to another profile. Selection, Wasm latency, and both
calibrations are locked before test.

The selected profile is tested exactly once on 100 color-swapped pairs at each
opening depth against each reference: 1,600 games total. Each paired interval
uses 10,000 bootstrap resamples of whole color-swapped pairs. The operational
gate requires zero illegal moves, incomplete rebound actions, unexplained
truncations, and native/Wasm parity failures.

The browser may display **Expert — DeepTurnSearch** only if the selected
profile's paired 95% lower confidence bound is strictly above 50% against each
reference separately. Failure against either reference is an honest negative
result: Deep Game Review still ships, but the opponent selector and Expert
strength claim do not.

**Expert selector status (frozen-test result): Enabled.** The locked 400k
profile scored 67.25% against fixed Rank5Derived (paired 95% interval
64.12%–70.25%) and 61.38% against JacekInspired-20k (58.25%–64.38%) across
the exactly-once 1,600-game test, with zero operational failures. See the
[generated gate report](../benchmarks/game_review_gate/REPORT.md).

The frozen manifest, opening identities, selection/calibration lock, phase
summaries, Wasm latency, compact result, and report live under
`benchmarks/game_review_gate/`. Commands are in
[Reproducibility](reproducibility.md#game-review-gate). The arena exposes
`deep-turn-search` and records its diagnostics under that distinct identity;
custom complete-turn work cannot report as Rank5Derived.

## Rank5Derived demo gates

The demo profile adapts search from the authentic `rank_5` source to different
rules and fixed 50,000-node work. Its measurements cannot be assigned to the
ranked submission.

The historical evaluator-selection gate compared the original 15%
replay-trained value blend with 0% search-only evaluation over 60 frozen
color-swapped pairs (120 games): 20 pairs after each of 4, 12, and 20 opening
plies. The 15% candidate went 70-50 with zero illegal moves, errors,
unexplained truncations, or incomplete actions, but the 10,000-resample paired
95% interval was 49.17%–67.50%. Because its lower bound did not strictly exceed
50%, the demo profile selected 0%.

The exact frozen configuration and compact summary are tracked as
[`evaluator_gate_config.json`](../benchmarks/rank5_derived/evaluator_gate_config.json)
and
[`evaluator_gate_result.json`](../benchmarks/rank5_derived/evaluator_gate_result.json).
These files are archival evidence from the configurable experiment. The
current `Rank5DerivedBot` identity is strictly fixed and cannot construct a
replay-blend candidate. Accordingly, `benchmarks/run_rank5_derived_gate.sh`
now exits with an explanation instead of rerunning configurable work under the
Rank5Derived name. Exact reproduction requires the source revision recorded
with the archived artifacts; current configurable regression work uses the
separately named complete-turn analysis executable.

The intentionally unavailable historical command is retained only as a clear
pointer to that policy:

```bash
benchmarks/run_rank5_derived_gate.sh \
  --output results/rank5-derived-gate-raw.json \
  --summary results/rank5-derived-gate-summary.json
```

It exits nonzero by design and does not modify the curated artifacts.

The checked-in browser module was also probed on an Apple M4 Pro with Node
26.5.0 and Emscripten 6.0.2. Across 80 actual complete-turn searches, the 50k
profile measured 42.564 ms median, 50.311 ms p95, and 54.355 ms maximum. It
excluded 64 cached continuation edges, validated 144 action edges, and recorded
zero rejected commands or incomplete actions. The limits were 120 ms p95 and
250 ms maximum. These timings are machine-specific; fixed-node decisions and
action diagnostics are deterministic. The curated report is
[`wasm_result.json`](../benchmarks/rank5_derived/wasm_result.json).

```bash
node benchmarks/wasm_rank5_derived.mjs 80 10 120 250 \
  > results/rank5-derived-wasm.json
```

For historical context, the Tactical RolloutOnly Wasm initial-position median
on its gate machine was 57.700 ms, below an earlier 174.120 ms baseline. That is
a separate MCTS measurement and does not measure quiescence or Rank5Derived.

## CodinGame evidence

The canonical current local snapshot associated with the platform result is the
`rank_4` generated source:
history version 56, agent `6604719`, submission `41114327`, rank 4 of 208 with
score `44.29750553418035` after a completed 66-24 batch. Its maintained local
source is 98,624 characters with SHA-256
`5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`.
CodinGame does not expose a remote source digest, so the binding is the
documented authenticated history/source fingerprint rather than a claimed
remote SHA-256 measurement. See the
[rank-4 record](../submissions/codingame/bots/rank_4/README.md) and
[experiment history](../submissions/codingame/bots/rank_4/EXPERIMENTS.md).

The immutable predecessor remains important for demo provenance: `rank_5`
history version 26, agent `6561779`, submission `41015554`, ranked 5 of 206 with
score `42.42773147296124` after a completed 57-33 batch. Its generated source
SHA-256 is `f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`.
`Rank5DerivedBot` adapts that source—not `rank_4`—to different browser rules
and fixed work, so its measurements cannot be assigned to either platform
artifact. The [submission archive](../submissions/codingame/README.md) records
the full lineage and every separate experiment.
