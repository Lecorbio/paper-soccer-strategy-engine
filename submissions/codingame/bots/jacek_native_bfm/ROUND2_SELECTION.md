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

Export every retained seed explicitly. The round-two exporter intentionally
rejects an omitted seed while the training artifact is pending:

```sh
python3 submissions/codingame/tools/generate_jacek_native_model_round2.py \
  --model models/jacek_native_round2_candidate.json \
  --seed 20260821 \
  --output build/round2-gates/20260821.hpp \
  --runtime-output build/round2-gates/20260821.runtime
```

Record both profiles for every retained seed. Use the freshly exported
bootstrap seed `20260813` runtime as the previous native champion:

```sh
python3 tools/jacek_native_round2_selection.py record \
  --profile screen \
  --model models/jacek_native_round2_candidate.json \
  --seed 20260821 \
  --candidate-runtime build/round2-gates/20260821.runtime \
  --baseline-model models/jacek_native_bootstrap_model.json \
  --baseline-seed 20260813 \
  --baseline-runtime \
    results/jacek_native_bfm/round2-runtime-inputs-632c731/seed-20260813.runtime \
  --gate-binary build/release/papersoccer_jacek_native_model_gate \
  --output-dir results/jacek_native_bfm/round2-gates
```

Repeat with `--profile decisive`, then repeat both commands for every other
seed. Each invocation archives canonical JSON and the complete stdout under
SHA-256 filenames. It verifies every pair result against the summary, binds
the exact model/checkpoint/runtime, gate binary, material source files and
exporters, and refuses to overwrite an existing artifact. A clean threshold
failure is valid evidence and is archived with `passed: false`.

Finalize only with complete evidence:

```sh
python3 tools/jacek_native_round2_selection.py finalize \
  --model models/jacek_native_round2_candidate.json \
  --baseline-model models/jacek_native_bootstrap_model.json \
  --baseline-seed 20260813 \
  --baseline-runtime \
    results/jacek_native_bfm/round2-runtime-inputs-632c731/seed-20260813.runtime \
  --reports results/jacek_native_bfm/round2-gates \
  --output models/jacek_native_round2_selection.json
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
python3 tools/jacek_native_round2_selection.py finalize-exploratory \
  --model models/jacek_native_round2_candidate.json \
  --baseline-model models/jacek_native_bootstrap_model.json \
  --baseline-seed 20260813 \
  --baseline-runtime \
    results/jacek_native_bfm/round2-runtime-inputs-632c731/seed-20260813.runtime \
  --reports results/jacek_native_bfm/round2-gates \
  --output models/jacek_native_round2_selection.json
```

This command refuses to run if any seed passed canonically. It also refuses a
failed seed unless both its screen and decisive reports have zero unfinished
games, candidate/baseline headroom failures, and candidate/baseline operational
timeouts. Safe failed seeds use the canonical ranking order. The sidecar says
`promotion_eligible: false` and records every positive strength-threshold
shortfall; it cannot be mistaken for promotion evidence.

## Immutable production activation

Selection does not alter the training JSON. Export the selected runtime to a
tracked path. Before activation, copy the complete content-addressed reports
and stdout transcripts into the tracked evidence directory and reproduce the
three bootstrap checkpoint runtimes named by the current cumulative training
model:

```sh
mkdir -p models/jacek_native_round2_gate_evidence
cp results/jacek_native_bfm/round2-gates/*.json \
  results/jacek_native_bfm/round2-gates/*.stdout.txt \
  models/jacek_native_round2_gate_evidence/

for seed in 20260811 20260812 20260813; do
  python3 submissions/codingame/tools/generate_jacek_native_model.py \
    --model models/jacek_native_bootstrap_model.json \
    --seed "$seed" \
    --output "build/jacek-native-bootstrap-$seed.hpp" \
    --runtime-output "models/jacek_native_bootstrap_seed_$seed.runtime"
done

python3 submissions/codingame/tools/generate_jacek_native_model_round2.py \
  --model models/jacek_native_round2_candidate.json \
  --seed SELECTED_SEED \
  --output build/round2-selected.hpp \
  --runtime-output models/jacek_native_round2_selected.runtime

python3 submissions/codingame/tools/jacek_native_round2_activation.py create \
  --model models/jacek_native_round2_candidate.json \
  --selection models/jacek_native_round2_selection.json \
  --runtime models/jacek_native_round2_selected.runtime \
  --baseline-model models/jacek_native_bootstrap_model.json \
  --baseline-seed 20260813 \
  --baseline-runtime models/jacek_native_bootstrap_seed_20260813.runtime \
  --reports models/jacek_native_round2_gate_evidence \
  --checkpoint models/jacek_native_bootstrap_model.json \
    models/jacek_native_bootstrap_seed_20260811.runtime \
  --checkpoint models/jacek_native_bootstrap_model.json \
    models/jacek_native_bootstrap_seed_20260812.runtime \
  --checkpoint models/jacek_native_bootstrap_model.json \
    models/jacek_native_bootstrap_seed_20260813.runtime \
  --output models/jacek_native_round2_deployment.json
```

The `--checkpoint` declarations must exactly cover the model's non-seed
checkpoint artifacts; the tracked untrained seed root is verified implicitly.
No extra, missing, self-referential, or non-generated runtime is accepted.

The descriptor binds the exact model, sidecar, selected runtime, selected seed,
decision kind, baseline, every report/stdout byte, tool identities, and the
complete recursive checkpoint ancestry. Loading it reruns report/stdout
validation and deterministic selection, rather than trusting a rehashable
sidecar. Its presence is the deliberate production switch. Without it, the
Node and CMake generators remain on the checked-in round-one bootstrap. With
it, both route through the selection-aware activation tool. Generation and
purity fail closed on a changed byte, stale tool, path escape, ambiguous
selection, or mutated `chosen_seed`/selection status.
