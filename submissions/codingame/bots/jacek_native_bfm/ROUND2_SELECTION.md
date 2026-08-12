# Round-two actual-clock selection

Round-two training artifacts are immutable evidence. Their `chosen_seed` is
always `null`; the validation-MSE winner is only `provisional_seed`. Promotion
is represented by a separate content-addressed selection sidecar after every
retained seed has complete actual-clock evidence.

## Frozen gates

Run gates serially. The recorder holds a repository-wide nonblocking lock, and
rejects concurrent timing runs. It always uses the deployment-constant shuffle
seed and these profiles; none of their clocks, openings, or thresholds are CLI
options:

| Profile | Pairs / games | Clock | Total wins | Per-color wins |
| --- | ---: | ---: | ---: | ---: |
| `screen` | 500 / 1,000 | 50 / 10 ms | at least 530 | none |
| `decisive` | 106 / 212 | 800 / 155 ms | at least 112 | at least 50 |

Both profiles require zero unfinished games, candidate/baseline headroom
failures, and candidate/baseline operational timeouts. Openings cycle through
depths `0,4,8,12` from seed `6517766227279252335`. A varied shuffle seed is
diagnostic only and cannot produce promotion evidence.

Before any new training, preserve the deployed native champion under fresh
paths. The selector recognizes this historical checkpoint only by its exact
content-addressed deployment identity; it does not waive an unknown trainer:

```sh
test ! -e models/jacek_native_history62_champion.json
test ! -e models/jacek_native_history62_champion.runtime
test ! -e models/jacek_native_history62_selection.json
test ! -e models/jacek_native_history62_deployment.json
cp models/jacek_native_round2_candidate.json \
  models/jacek_native_history62_champion.json
cp models/jacek_native_round2_selected.runtime \
  models/jacek_native_history62_champion.runtime
cp models/jacek_native_round2_selection.json \
  models/jacek_native_history62_selection.json
cp models/jacek_native_round2_deployment.json \
  models/jacek_native_history62_deployment.json
test "$(sha256sum models/jacek_native_history62_champion.json | cut -d ' ' -f 1)" = \
  b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14
test "$(sha256sum models/jacek_native_history62_champion.runtime | cut -d ' ' -f 1)" = \
  17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3
test "$(sha256sum models/jacek_native_history62_selection.json | cut -d ' ' -f 1)" = \
  5597e4228850cd44aac4adc5f11e3d6533e5528e3e04c51700d2f04b2cbe2cef
test "$(sha256sum models/jacek_native_history62_deployment.json | cut -d ' ' -f 1)" = \
  88092ac6601faac0f3da31bdaa1e2a5eca15bdb762b18810d450b33ee0d6ef2f
```

Training requires an explicit fresh `--output` and rejects an existing path.
Never point it at the deployed `models/jacek_native_round2_candidate.json`.
For archived restart shards, preflight historical identities explicitly with
`jacek_native_restart_corpus_round2.py --archived`; without that flag the CLI
requires the current local compiler/source build identity.

After all four new league manifests and the new restart manifest exist,
preflight and train with the frozen virtual-environment interpreter. Positional
shards are strict-current; every archive category is explicit and terminated
by the next option:

```sh
test "$(find results/jacek_native_bfm/rank1-native-league-4k-9b531a7-seed-2026083111 \
  -mindepth 2 -maxdepth 2 -name manifest.json | wc -l | tr -d ' ')" = 4
test -f results/jacek_native_bfm/rank1-h62-restart-9b531a7-seed-2026083104/manifest.json
test ! -e models/jacek_native_round3_rank1_2026083111.json

.venv/bin/python tools/jacek_native_restart_corpus_round2.py --archived \
  results/jacek_native_bfm/restart-round2-v1-full-41123817-632c731-model19f954-seed-2026082205/shard-*.jsonl
.venv/bin/python tools/jacek_native_restart_corpus_round2.py \
  results/jacek_native_bfm/rank1-h62-restart-9b531a7-seed-2026083104/shard-*.jsonl

.venv/bin/python tools/train_jacek_native_round2.py \
  results/jacek_native_bfm/rank1-native-league-4k-9b531a7-seed-2026083111/*/shard-*.jsonl \
  --archived-round1 \
    results/jacek_native_bfm/bootstrap-v1-10k-hardened-seed-73194721/shard-*.jsonl \
  --archived-round2 \
    results/jacek_native_bfm/round2-v1-12k-632c731-model19f954-seed-20260822/*/shard-*.jsonl \
  --restart-round2 \
    results/jacek_native_bfm/rank1-h62-restart-9b531a7-seed-2026083104/shard-*.jsonl \
  --archived-restart-round2 \
    results/jacek_native_bfm/restart-round2-v1-full-41123817-632c731-model19f954-seed-2026082205/shard-*.jsonl \
  --seeds 20260831,20260832,20260833 \
  --output models/jacek_native_round3_rank1_2026083111.json
```

Export every retained seed explicitly after training. The round-two exporter
intentionally rejects an omitted seed while the training artifact is pending:

```sh
mkdir -p build/round3-rank1-2026083111
for seed in 20260831 20260832 20260833; do
  python3 submissions/codingame/tools/generate_jacek_native_model_round2.py \
    --model models/jacek_native_round3_rank1_2026083111.json \
    --seed "$seed" \
    --output "build/round3-rank1-2026083111/$seed.hpp" \
    --runtime-output "build/round3-rank1-2026083111/$seed.runtime"
done
```

Record both profiles for every retained seed. Use retained native seed
`20260822` as the previous champion:

```sh
test ! -e results/jacek_native_bfm/round3-rank1-2026083111-gates
python3 tools/jacek_native_round2_selection.py record \
  --profile screen \
  --model models/jacek_native_round3_rank1_2026083111.json \
  --seed 20260831 \
  --candidate-runtime build/round3-rank1-2026083111/20260831.runtime \
  --baseline-model models/jacek_native_history62_champion.json \
  --baseline-seed 20260822 \
  --baseline-runtime models/jacek_native_history62_champion.runtime \
  --baseline-selection models/jacek_native_history62_selection.json \
  --baseline-deployment models/jacek_native_history62_deployment.json \
  --gate-binary build/papersoccer_jacek_native_model_gate \
  --output-dir results/jacek_native_bfm/round3-rank1-2026083111-gates
```

Repeat with `--profile decisive`, then repeat both commands for every other
seed. Each invocation archives canonical JSON and the complete stdout under
SHA-256 filenames. It verifies every pair result against the summary, binds
the exact model/checkpoint/runtime, gate binary, material source files and
exporters, and refuses to overwrite an existing artifact. A clean threshold
failure is valid evidence and is archived with `passed: false`.

Finalize only with complete evidence:

```sh
test ! -e models/jacek_native_round3_rank1_2026083111_selection.json
python3 tools/jacek_native_round2_selection.py finalize \
  --model models/jacek_native_round3_rank1_2026083111.json \
  --baseline-model models/jacek_native_history62_champion.json \
  --baseline-seed 20260822 \
  --baseline-runtime models/jacek_native_history62_champion.runtime \
  --baseline-selection models/jacek_native_history62_selection.json \
  --baseline-deployment models/jacek_native_history62_deployment.json \
  --reports results/jacek_native_bfm/round3-rank1-2026083111-gates \
  --output models/jacek_native_round3_rank1_2026083111_selection.json
```

Finalization rejects missing or duplicate seed/profile reports, changed model
or runtime bytes, stale source/exporter identities, altered stdout, operational
failures, and a field that disagrees with the full transcript. If no seed
passes both profiles, it refuses canonical promotion; a seed can still be
exported explicitly for a non-promoting CodinGame experiment.

Passing seeds are ordered by decisive wins, decisive worst-color wins, screen
wins, screen worst-color wins, validation outcome MSE, then numeric seed. The
selected runtime SHA-256 is recorded identically as both the tested and
deployment runtime identity. Activation must use the immutable model, selected
seed, and sidecar together; it must never rewrite the training model.

## Exploratory fallback

If and only if no retained seed passes both strength gates, the same complete
report set may produce a non-promoting exploratory selection:

```sh
test ! -e models/jacek_native_round3_rank1_2026083111_selection.json
python3 tools/jacek_native_round2_selection.py finalize-exploratory \
  --model models/jacek_native_round3_rank1_2026083111.json \
  --baseline-model models/jacek_native_history62_champion.json \
  --baseline-seed 20260822 \
  --baseline-runtime models/jacek_native_history62_champion.runtime \
  --baseline-selection models/jacek_native_history62_selection.json \
  --baseline-deployment models/jacek_native_history62_deployment.json \
  --reports results/jacek_native_bfm/round3-rank1-2026083111-gates \
  --output models/jacek_native_round3_rank1_2026083111_selection.json
```

This command refuses to run if any seed passed canonically. It also refuses a
failed seed unless both its screen and decisive reports have zero unfinished
games, candidate/baseline headroom failures, and candidate/baseline operational
timeouts. Safe failed seeds use the canonical ranking order. The sidecar says
`promotion_eligible: false` and records every positive strength-threshold
shortfall; it cannot be mistaken for promotion evidence.

## Immutable production activation

Selection does not alter the training JSON. Keep every new model, selection,
runtime, evidence set, and staged descriptor on a fresh generation-specific
path. The canonical deployment JSON is the sole mutable activation pointer.
Before activation, install the complete content-addressed reports in a fresh
evidence subdirectory and install all six file-backed parent runtimes named by
the cumulative corpus (three history-62 checkpoints and three bootstrap
checkpoints):

```sh
test ! -e models/jacek_native_round2_gate_evidence/round3-rank1-2026083111
test ! -e models/jacek_native_round3_rank1_2026083111_selected.runtime
test ! -e models/jacek_native_round3_rank1_2026083111_deployment.json
mkdir -p models/jacek_native_round2_gate_evidence/round3-rank1-2026083111
cp results/jacek_native_bfm/round3-rank1-2026083111-gates/*.json \
  results/jacek_native_bfm/round3-rank1-2026083111-gates/*.stdout.txt \
  models/jacek_native_round2_gate_evidence/round3-rank1-2026083111/

test ! -e models/jacek_native_history62_seed_20260821.runtime
test ! -e models/jacek_native_history62_seed_20260823.runtime
cp build/rank1-campaign/seed21.runtime \
  models/jacek_native_history62_seed_20260821.runtime
cp build/rank1-campaign/seed23.runtime \
  models/jacek_native_history62_seed_20260823.runtime
test "$(sha256sum models/jacek_native_history62_seed_20260821.runtime | cut -d ' ' -f 1)" = \
  1f8d5696742c94aec74df1cf107cf478cdf885d1a9866b46a93e4ec32359b9c8
test "$(sha256sum models/jacek_native_history62_champion.runtime | cut -d ' ' -f 1)" = \
  17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3
test "$(sha256sum models/jacek_native_history62_seed_20260823.runtime | cut -d ' ' -f 1)" = \
  d0d5478385f52dd80b8073ce24ae587afe7f8b55d3123f0939d55716b5451ed0

test "$(sha256sum models/jacek_native_bootstrap_seed_20260811.runtime | cut -d ' ' -f 1)" = \
  4bebe5df37379881187e01297813472825307ed1041f4dce22c04788eb42e5fe
test "$(sha256sum models/jacek_native_bootstrap_seed_20260812.runtime | cut -d ' ' -f 1)" = \
  d6236e108fcb563d630db73c8e0c41a009efb8dbd3fc2f7e05f529daae685c4f
test "$(sha256sum models/jacek_native_bootstrap_seed_20260813.runtime | cut -d ' ' -f 1)" = \
  877ee8d0afdb20cf3466bee4c09f654d33c6ac4ecc230b8022f570a31e60f93d

python3 submissions/codingame/tools/generate_jacek_native_model_round2.py \
  --model models/jacek_native_round3_rank1_2026083111.json \
  --seed SELECTED_SEED \
  --output build/round3-rank1-2026083111-selected.hpp \
  --runtime-output models/jacek_native_round3_rank1_2026083111_selected.runtime

python3 submissions/codingame/tools/jacek_native_round2_activation.py create \
  --model models/jacek_native_round3_rank1_2026083111.json \
  --selection models/jacek_native_round3_rank1_2026083111_selection.json \
  --runtime models/jacek_native_round3_rank1_2026083111_selected.runtime \
  --baseline-model models/jacek_native_history62_champion.json \
  --baseline-seed 20260822 \
  --baseline-runtime models/jacek_native_history62_champion.runtime \
  --baseline-selection models/jacek_native_history62_selection.json \
  --baseline-deployment models/jacek_native_history62_deployment.json \
  --reports models/jacek_native_round2_gate_evidence/round3-rank1-2026083111 \
  --checkpoint models/jacek_native_history62_champion.json \
    models/jacek_native_history62_seed_20260821.runtime \
  --checkpoint models/jacek_native_history62_champion.json \
    models/jacek_native_history62_champion.runtime \
  --checkpoint models/jacek_native_history62_champion.json \
    models/jacek_native_history62_seed_20260823.runtime \
  --checkpoint models/jacek_native_bootstrap_model.json \
    models/jacek_native_bootstrap_seed_20260811.runtime \
  --checkpoint models/jacek_native_bootstrap_model.json \
    models/jacek_native_bootstrap_seed_20260812.runtime \
  --checkpoint models/jacek_native_bootstrap_model.json \
    models/jacek_native_bootstrap_seed_20260813.runtime \
  --output models/jacek_native_round3_rank1_2026083111_deployment.json

python3 submissions/codingame/tools/jacek_native_round2_activation.py validate \
  --deployment models/jacek_native_round3_rank1_2026083111_deployment.json
python3 submissions/codingame/tools/jacek_native_round2_activation.py generate \
  --deployment models/jacek_native_round3_rank1_2026083111_deployment.json \
  --output build/round3-rank1-2026083111-staged.hpp \
  --runtime-output build/round3-rank1-2026083111-staged.runtime \
  --metadata
```

The `--checkpoint` declarations must exactly cover the model's non-seed
checkpoint artifacts; the tracked untrained seed root is verified implicitly.
No extra, missing, self-referential, or non-generated runtime is accepted.

Only after the staged descriptor, exact generated source, construction-inclusive
timing probe, and upload safety gates pass may the canonical pointer switch.
The install is an atomic, locked compare-and-swap; a stale expected hash fails
without changing production:

```sh
python3 submissions/codingame/tools/jacek_native_round2_activation.py install \
  --deployment models/jacek_native_round3_rank1_2026083111_deployment.json \
  --expected-current-sha256 \
    88092ac6601faac0f3da31bdaa1e2a5eca15bdb762b18810d450b33ee0d6ef2f
```

Do not run that install command during training or gate evaluation. Preserve
the exact history-62 submission source before switching; it is the operational
rollback artifact if post-switch generation or CodinGame validation fails.

The descriptor binds the exact model, sidecar, selected runtime, selected seed,
decision kind, baseline, every report/stdout byte, tool identities, and the
complete recursive checkpoint ancestry. Loading it reruns report/stdout
validation and deterministic selection, rather than trusting a rehashable
sidecar. Its presence is the deliberate production switch. Without it, the
Node and CMake generators remain on the checked-in round-one bootstrap. With
it, both route through the selection-aware activation tool. Generation and
purity fail closed on a changed byte, stale tool, path escape, ambiguous
selection, or mutated `chosen_seed`/selection status.
