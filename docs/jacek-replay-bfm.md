# Offline Jacek replay BFM

`JacekReplayBfmBot` is a native, single-thread research bot for the 8x10
CodinGame rule profile. It combines complete-turn best-first minimax search
with the larger value representation described in Jacek Dermont's
[Paper Soccer article](https://github.com/jdermont/playground-kvhfh5iv/blob/5f369d6e64870690d100f0c3fe2f8e75c447dac3/papersoccer.md).
It does not modify or replace `rank_4`, `neural_puct`, or
`jacek_native_bfm`.

The implementation is an independent adaptation. It contains no upstream
checkpoint, unpublished source, player identity input, policy head, or exact
transcript lookup. Public and owned replays determine the training-state
distribution; fixed-node Rank-4 searches provide the value labels.

## Model and features

The value network is bias-free `6301 -> 192 -> 32 -> 1`:

- Inputs `0..315` identify used playable edges.
- Each of 105 mover-canonical vertices selects one of 57 inputs.
- Distance seven or unreachable selects category 56. Other vertices select
  `8 * true_turn_distance + max(free_degree - 1, 0)`.
- Player Two positions rotate 180 degrees before feature indexing.
- Hidden activations are square/leaky-0.01 and leaky-ReLU-0.01; the scalar
  output uses `tanh`.

Only active edge and vertex rows are accumulated at inference time. The dense
model has 1,215,968 float32 weights and occupies 4,864,000 bytes with its
runtime header.

The runtime format has a fixed 128-byte little-endian header followed by
input-major `w1`, input-major `w2`, and `w3`. The loader verifies dimensions,
activation identifiers, feature-schema SHA-256, payload SHA-256, finite
weights, zero reserved bytes, and exact file length before exposing the model
to search.

`models/jacek_replay_bfm_bootstrap.runtime` is deliberately untrained. It
exists so normal builds can exercise the complete loader, search, Arena, and
CLI path. Its manifest says `bootstrap-not-trained-not-selected`; it must not
be reported as a playing-strength checkpoint.

`models/jacek_replay_bfm_development/jacek_replay_bfm.runtime` is the current
trained development checkpoint. It merges 32k Rank-4 labels already collected
for the pilot with 200 continuation-game labels, then repacks them through the
current leakage-safe streaming packer. The selected seed saw 1,909 training
and 168 validation positions; validation sign accuracy is 86.3% and weighted
Huber loss is 0.05747. On a fixed 30-opening, color-swapped development panel
at 20 ms it won 52/120 games versus 28/120 for the bootstrap and 43/120 for
the earlier pilot. In a direct 30-pair Arena match on the final search code it
beat the bootstrap 40-20, scoring 21-9 as Player One and 19-11 as Player Two;
the deterministic pair-bootstrap 95% interval was 53.3%-78.3%. It remains
below Rank 4, has not passed the canonical three-round campaign or protected
gates, and is explicitly not a promoted checkpoint.

## Search

One tree edge is one legal complete turn, including every mandatory rebound.
The generator retains at most 250 unique resulting boundary states. It uses
nine LIFO deque extractions followed by one FIFO extraction, mover-canonical
seeded neighbour shuffling, a bounded tactical witness pass, and explicit
goal-path probes. Immediate wins, forced cutoffs, safe handoffs, opponent
immediate goals, and own goals are ordered separately. The retained-action
limit is also the enumeration limit; proven wins from the tactical pre-pass
return immediately instead of filling an irrelevant ordinary action sample.

New children receive the neural value immediately. Frontier selection uses
UCT-style best-first minimax, values are overwritten by minimax backup, and
the final root choice adds a visit term. Siblings reuse a parent first-layer
accumulator and apply only changed sparse feature rows. Only root children
retain returnable action transcripts. The default native profile is one
thread, a 980 ms total decision budget, 1,000,000 tree nodes, 50,000 partial paths
per expansion, `C=0.95`, and `FPU=0.5`. The returned turn is validated through
the authoritative rules engine and cached one edge at a time for the public
`Bot` interface. The configured time is a total decision budget; the search
reserves up to 25 ms of it for tree destruction, validation, and cache setup.

## Reproducible replay pipeline

Install the pinned research dependency and build the Rank-4 teacher:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --target \
  papersoccer_jacek_replay_teacher \
  papersoccer_jacek_replay_continuations \
  papersoccer_jacek_replay_comparison
```

The workflow reads the exclusion registry before either replay source,
validates every complete turn locally, freezes whole-game 80/10/10 splits,
keeps reflected rows with their parent game, purges exact/rotated/reflected
cross-split overlaps, writes content-addressed CSR NPZ shards, trains three
fixed seeds, and selects on validation only. Intermediate rounds never inspect
the test split; only the explicitly marked final round reveals it after
validation selection:

The workflow uses the SQLite-backed streaming packer, so teacher JSONL and
CSR construction remain bounded-memory at article-scale corpus sizes.
The canonical mode freezes all three rounds at 10,000 continuations, 32k bulk
labels, 400k root/deep labels, 10% deep uncertainty selection, 100 samples per
game, and the documented optimizer settings. Smaller plumbing runs must pass
`--development-mode`; their model manifests are marked non-promotable and the
matched-baseline gate rejects them.

Campaign execution is resumable and fail-closed. Canonical commands bind the
campaign identifier `canonical-20260823-v1`, split teacher input into stable
contiguous 25-game chunks, run ten teacher processes, and train at most two of
the three seeds concurrently. These execution values are frozen alongside the
scientific parameters. Add the following options to every canonical round:

```text
  --campaign-id canonical-20260823-v1 \
  --teacher-workers 10 --teacher-chunk-games 25 --seed-workers 2
```

Each roots, TSV, label, continuation, concatenation, packing, training,
selected-runtime, and final-workflow stage has an atomic receipt under the
round's `receipts/` directory. Teacher chunks have their own receipts and are
always merged by ordinal, so `--teacher-workers 1`, `2`, and `10` produce the
same bytes for a fixed chunk size. The continuation sidecar additionally binds
the TSV hash, exact actor quotas, row lineage, and candidate model.
Each training seed similarly publishes a runtime and receipt under
`training-seeds/`; an interrupted training stage resumes only those exact
checkpoints, and the chosen runtime must hash to the selected seed checkpoint.
The final `receipts/09-workflow.json` sidecar is derived from the fixed round
directory and validates `workflow.json`; it is intentionally not embedded in
that file, which avoids a self-hash cycle while still making deletion or
corruption fail closed.

Every round also records the repository commit, tree, branch, cleanliness,
candidate/shared-core source closure, CMake cache, configured C++ compiler,
and exact teacher/continuation binary hashes. Canonical execution refuses a
dirty or detached checkout and refuses producers that do not come from the
same CMake `Release` directory. Development and smoke runs record the same
identity but remain allowed to say that the checkout is dirty. Current-state
and predecessor validation recomputes these identities, so do not edit,
reconfigure, or rebuild anything between canonical rounds.

After an interruption, rerun the byte-for-byte same command with `--resume`.
A stage is skipped only when its configuration, producers, inputs, recursive
outputs, counts, environment, and child chunk receipts all revalidate. A
missing receipt beside an existing output, any corrupted artifact, or any
changed producer aborts without adopting or overwriting data; use a new output
directory for that new attempt. Do not edit or rebuild campaign producers
between canonical rounds.

Before freezing a real campaign, run the three-round development smoke. The
`--smoke-profile` flag overrides training/search sizes with a frozen 40-game,
100-node, one-epoch profile, forces campaign eligibility off, reveals test
metrics only in Round 2, and requires the same predecessor inputs as the real
chain. The command shape is otherwise identical: use a fresh output directory
for each round, pass the preceding workflow/roots/runtime and every earlier
pack report to R1/R2, then repeat R2 with `--resume`. The checked end-to-end
driver performs that entire sequence:

```bash
.venv/bin/python tests/codingame/jacek_replay_workflow_smoke.py \
  --repository . \
  --teacher build/release/papersoccer_jacek_replay_teacher \
  --continuations build/release/papersoccer_jacek_replay_continuations
```

Smoke receipts and checkpoints are deliberately marked
`development-candidate-not-promotable`; they can never satisfy canonical or
promotion validation.

Before Round 0, create and commit the campaign branch, verify `git status` is
empty, then build the two producers once in `build/release`. The workflow will
reject Round 0 until those conditions are true.

```bash
snapshot=submissions/codingame/bots/neural_puct/live_replay/corpora/\
687468e84c475107eee840f4d731fbc51182e8cfc20d2bf2cd7039d344f48f97.json
exclusions=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["exclusion_registry_path"])' \
  "$snapshot")

.venv/bin/python tools/jacek_replay_workflow.py \
  --repository . \
  --exclusions "$exclusions" \
  --public-jacek submissions/codingame/bots/neural_puct/public_jacek_unlocked_v1.json \
  --live-snapshot "$snapshot" \
  --teacher build/release/papersoccer_jacek_replay_teacher \
  --continuation-generator build/release/papersoccer_jacek_replay_continuations \
  --round 0 --continuation-games 10000 \
  --actor-nodes 16000 --candidate-tree-nodes 2000 \
  --campaign-id canonical-20260823-v1 \
  --teacher-workers 10 --teacher-chunk-games 25 --seed-workers 2 \
  --output-directory results/jacek_replay_bfm/round-0 \
  --nodes 32000 --root-nodes 400000 --deep-percent 10 \
  --max-samples 100 \
  --epochs 50 --patience 8 --batch-size 256
```

Later rounds pass the previously selected runtime and use the frozen 50/50
candidate-self-play versus color-balanced candidate/Rank-4 mix:

```text
# Rounds 1 and 2 additionally require the preceding selected checkpoint,
# frozen root assignments, and every earlier round's pack report.
  --continuation-generator build/release/papersoccer_jacek_replay_continuations \
  --continuation-model PREVIOUS_ROUND.runtime \
  --previous-workflow ROUND_0/workflow.json \
  --round 1 --continuation-games 10000 \
  --previous-roots PREVIOUS_ROUND/replay-roots.json \
  --prior-pack-report ROUND_0/shards/pack-report.json \
  --actor-nodes 16000 --candidate-tree-nodes 2000

# Round 2 passes both earlier pack reports. Add this only to the final run:
  --previous-workflow ROUND_1/workflow.json \
  --previous-roots ROUND_1/replay-roots.json \
  --prior-pack-report ROUND_0/shards/pack-report.json \
  --prior-pack-report ROUND_1/shards/pack-report.json \
  --final-test-reveal
```

Canonical rounds validate the entire file-backed receipt chain before doing
work. Round 0 rejects prior-round arguments. Rounds 1 and 2 require the exact
immediately preceding workflow receipt, selected runtime, frozen roots, and
ordered prior pack reports; hashes, canonical configuration, and R0/R1
ancestry must all agree.

Continuation records retain the originating replay `group_id`, so every
generated descendant inherits its source game's frozen split. The generator
uses seeded 15% early-turn exploration, writes only complete terminal games,
and never uses observed replay actions as model policy targets.

Ordinary root scores are converted to mover-relative
`tanh(player_sign * score / 12000)`. Proven values are +/-1, and the retained
target is 75% teacher value plus 25% mover-relative final outcome. Positions
where the teacher does not complete depth one are rejected rather than
silently labelled neutral. Original replay boundaries use the 400k root
budget; continuation labels use 32k in bulk and rerun the 10% closest-to-zero
teacher values at 400k before packing.

Large corpora remain ignored under `results/`. A selected model may be copied
to `models/` only after its manifest, runtime, fixed-seed metrics, and game
gates have been reviewed.

Before spending a protected opening bank, train the matched
`1156 -> 32 -> 32 -> 1` distance-only control on the identical cumulative
shards. The large candidate advances only when validation Huber loss improves
without reducing sign accuracy:

```bash
.venv/bin/python tools/jacek_replay_baseline.py \
  --pack-report ROUND_0/shards/pack-report.json \
  --pack-report ROUND_1/shards/pack-report.json \
  --pack-report ROUND_2/shards/pack-report.json \
  --candidate-manifest ROUND_2/model/jacek_replay_bfm.runtime.json \
  --workflow-receipt ROUND_2/workflow.json \
  --output results/jacek_replay_bfm/baseline-gate.json
```

This baseline command is intentionally not tunable: it always uses the three
fixed seeds, 50 epochs, patience 8, batch size 256, AdamW 0.001/1e-5,
weighted Huber 0.25, and no test reveal. It also writes
`baseline-gate.baseline.runtime`, a diagnostic-only small-model checkpoint.
The tuning and promotion gates reload that checkpoint and the exact cumulative
shards and recompute both validation panels rather than trusting the receipt's
metric fields.

## Native use and strength gates

Run the bot in the native Arena with an explicit checkpoint:

```bash
./build/release/papersoccer_arena positions \
  --rules codingame \
  --candidate-kind jacek-replay-bfm \
  --candidate-jacek-replay-model \
    models/jacek_replay_bfm_development/jacek_replay_bfm.runtime \
  --candidate-jacek-replay-max-time-ms 980 \
  --reference-kind random
```

The dedicated comparison executable binds the maintained Rank-4 and neural
PUCT controls and gives each engine the same deadline:

```bash
# Freeze a development bank, then run the complete C grid on identical inputs.
./build/release/papersoccer_jacek_replay_comparison \
  --generate-bank results/jacek_replay_bfm/development-openings.tsv \
  --bank-classification development \
  --pairs 200 --opening-plies 12 --seed 123456789

for exploration in 0.25 0.5 0.95; do
  ./build/release/papersoccer_jacek_replay_comparison \
    --model PATH_TO_RUNTIME --opponent both \
    --bank results/jacek_replay_bfm/development-openings.tsv \
    --bank-classification development \
    --baseline-receipt results/jacek_replay_bfm/baseline-gate.json \
    --pairs 200 --time-ms 20 --exploration "$exploration" \
    --output "results/jacek_replay_bfm/development-c${exploration}.json"
done

.venv/bin/python tools/jacek_replay_tuning_gate.py \
  results/jacek_replay_bfm/development-c0.25.json \
  results/jacek_replay_bfm/development-c0.5.json \
  results/jacek_replay_bfm/development-c0.95.json \
  --baseline-receipt results/jacek_replay_bfm/baseline-gate.json \
  --output results/jacek_replay_bfm/tuning-receipt.json

# Prepare one protected attempt before the final bank or report exists. The
# ledger binds the tuned model, baseline, executable, and both future paths.
.venv/bin/python tools/jacek_replay_promotion_gate.py prepare-final-attempt \
  --ledger results/jacek_replay_bfm/final-attempt.json \
  --model PATH_TO_RUNTIME \
  --comparison-executable build/release/papersoccer_jacek_replay_comparison \
  --baseline-receipt results/jacek_replay_bfm/baseline-gate.json \
  --tuning-receipt results/jacek_replay_bfm/tuning-receipt.json \
  --bank results/jacek_replay_bfm/final-openings.tsv \
  --report results/jacek_replay_bfm/final-comparison.json

# Only now freeze and register the distinct final bank.
./build/release/papersoccer_jacek_replay_comparison \
  --generate-bank results/jacek_replay_bfm/final-openings.tsv \
  --bank-classification final \
  --pairs 500 --opening-plies 12 --seed 987654321

.venv/bin/python tools/jacek_replay_promotion_gate.py register-final-bank \
  --ledger results/jacek_replay_bfm/final-attempt.json

# This wrapper derives all frozen arguments from the tuning receipt, runs the
# 2,000 games sequentially, and consumes the attempt even on process failure.
.venv/bin/python tools/jacek_replay_promotion_gate.py run-final-attempt \
  --ledger results/jacek_replay_bfm/final-attempt.json

.venv/bin/python tools/jacek_replay_promotion_gate.py decide \
  results/jacek_replay_bfm/final-comparison.json \
  --baseline-receipt results/jacek_replay_bfm/baseline-gate.json \
  --tuning-receipt results/jacek_replay_bfm/tuning-receipt.json \
  --attempt-ledger results/jacek_replay_bfm/final-attempt.json \
  --output results/jacek_replay_bfm/final-decision.json

# Run this only when final-decision.json says eligible=true. The command
# recomputes the complete gate and atomically refuses every failed decision.
.venv/bin/python tools/jacek_replay_promotion_gate.py publish \
  --decision results/jacek_replay_bfm/final-decision.json \
  --candidate-manifest ROUND_2/model/jacek_replay_bfm.runtime.json \
  --baseline-receipt results/jacek_replay_bfm/baseline-gate.json \
  --tuning-receipt results/jacek_replay_bfm/tuning-receipt.json \
  --attempt-ledger results/jacek_replay_bfm/final-attempt.json \
  --report results/jacek_replay_bfm/final-comparison.json \
  --output-directory models/jacek_replay_bfm_promoted
```

Promotion requires zero illegal or unfinished games and every response below
1,000 ms. Against neural PUCT, the one-sided 95% Wilson lower bound must exceed
50% (at least 527/1,000 wins) and each color must win at least 52% (260/500).
Against Rank-4, the lower bound must exceed 47.5% (at least 501/1,000) and each
color must win at least 47.5% (238/500). A failed candidate remains
an explicitly unpromoted research artifact. The decision binds the absolute
raw comparison-report path and SHA-256 plus the consumed ledger identity; the
protected panel cannot be silently replaced or reused for another candidate.
The promoted directory is created atomically and contains the exact Round-2
runtime and manifest, diagnostic baseline artifact, tuning/final receipts,
raw comparison report, consumed attempt ledger, and a publication manifest.

## Current boundary

The checked-in deliverable implements and tests the runtime, search, data,
training, comparison, and promotion contracts. The article-scale campaign of
three approximately one-million-position rounds is an expensive explicit
research run; it is not performed by normal builds or CI, and the bootstrap
checkpoint is not a substitute for it.
