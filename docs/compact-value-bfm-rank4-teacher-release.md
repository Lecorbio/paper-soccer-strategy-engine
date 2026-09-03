# Rank-4 teacher challenger: release and upload bridge

`tools/compact_value_bfm_rank4_teacher_release.py` is the only release path
from a selected challenger to the two protected gates. It supports both
attempt zero and later teacher-training attempts:

- attempt zero selects an exporter-default source and its recovered
  tuple/profile. Its exact `f5e67d…` finalist is reproduced with the frozen
  `c380ae7…` pre-minifier exporter closure instead of being compared with the
  current minifying exporter;
- teacher training selects an exporter source with an exact, allowlisted
  search A/B macro prefix;
- both routes retain that selected source as provenance and derive a distinct
  CodinGame source by replacing the seven frozen deployment defaults;
- the runtime remains the exact selected `6301 -> 12 -> 8 -> 1`, bias-free,
  signed-three-bit scalar-value runtime.

The bridge never commits, pushes, starts GitHub Actions, opens a protected
bank, clicks Submit, or accesses CodinGame. Those state changes remain explicit
operator actions. The bridge does not create or depend on a recurring
automation.

## Release sequence

Run the release steps in a dedicated checkout of the exact
`compact-value-bfm` branch. `promote` refuses another branch and refuses
unrelated changes. It writes only these four repository artifacts:

```text
submissions/codingame/bots/compact_value_bfm/model.hpp
submissions/codingame/bots/compact_value_bfm/submission.cpp
submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.cpp
submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.json
```

The first two are the exact runtime export. The latter two are the configured
deployment source and its byte-level derivation manifest. A later trained
candidate's macro-prefixed selected source stays in the campaign ledger; it is
not substituted for the canonical exporter output.

```bash
PY=.venv/bin/python
GNU_CXX=/opt/homebrew/bin/g++-15
CAMPAIGN=/absolute/path/to/rank4-teacher-campaign
PLAN="$CAMPAIGN/campaign-plan.json"
RELEASE="$CAMPAIGN/release/attempt-001"
RUNTIME=/absolute/path/to/selected.runtime.json
SELECTED=/absolute/path/to/selected.generated.submission.cpp

"$PY" tools/compact_value_bfm_rank4_teacher_release.py promote \
  --campaign-plan "$PLAN" --attempt 1 \
  --candidate-runtime "$RUNTIME" --candidate-source "$SELECTED" \
  --repository "$PWD" --output "$RELEASE/promotion.json"
```

Inspect the four-file diff, then explicitly commit it on
`compact-value-bfm`. Push that exact commit only after the local preflights
below pass. The release validator requires a completely clean worktree and
proves every release artifact is tracked with bytes equal to `HEAD`.

## Maintained and release preflights

First run the maintained Compact Value-BFM preflight. Its output receipt must
be content-addressed. It covers fresh GCC, Clang, and sanitizer builds, complete
Compact Python discovery, exporter freshness, protocol, 4,096-state
feature/inference parity, and exact 1/2/10-process timing for both colors.

```bash
"$PY" tools/compact_value_bfm_preflight.py run \
  --repository "$PWD" \
  --output-root "$RELEASE/base-preflight" \
  --build-root "$RELEASE/base-preflight-build" \
  --runtime "$RUNTIME" --claimed-at-utc 2026-09-04T00:00:00Z
```

Then prepare and execute the challenger release preflight, substituting the
content-addressed maintained receipt for `BASE_RECEIPT`:

```bash
"$PY" tools/compact_value_bfm_rank4_teacher_release.py prepare-preflight \
  --campaign-plan "$PLAN" --attempt 1 \
  --candidate-runtime "$RUNTIME" --candidate-source "$SELECTED" \
  --repository "$PWD" --promotion "$RELEASE/promotion.json" \
  --base-preflight "$BASE_RECEIPT" --output-root "$RELEASE" \
  --gcc "$GNU_CXX"

"$PY" tools/compact_value_bfm_rank4_teacher_release.py run-preflight \
  --campaign-plan "$PLAN" --attempt 1 \
  --candidate-runtime "$RUNTIME" --candidate-source "$SELECTED" \
  --repository "$PWD" --promotion "$RELEASE/promotion.json" \
  --base-preflight "$BASE_RECEIPT" \
  --plan "$RELEASE/release-preflight/plan.json"
```

This second preflight compiles the exact deployed source with GCC, Clang, and
ASan/UBSan, runs native and both-color protocol tests, compiles a Rank-4 gate
with the deployed source path embedded, checks the compiled search
configuration, repeats 4,096-state parity, and records the maintained full
timing suite. Validation additionally requires exactly one uncontended timing
sample for each color, both strictly below 900/180 ms.

Release and preflight commands require an explicit `--repository`; this avoids
silently operating in the r4v2 development checkout instead of the dedicated
release branch. When `--gcc` is omitted the bridge probes versioned GNU drivers
(`g++-15`, `g++-14`, …) before the unversioned name, and rejects Apple's
`/usr/bin/g++` Clang alias.

## Exact CI and release evidence

Push the clean release commit and manually dispatch `.github/workflows/pages.yml`
on `compact-value-bfm`. It must be attempt one of workflow database ID
`316333312`, bind the exact commit, and contain exactly the five maintained
successful jobs. Seal either a fetched run or previously captured `gh` JSON:

```bash
"$PY" tools/compact_value_bfm_rank4_teacher_release.py seal-ci \
  --output "$RELEASE/github-ci.json" --head "$(git rev-parse HEAD)" \
  --run-id RUN_ID

"$PY" tools/compact_value_bfm_rank4_teacher_release.py seal-release-evidence \
  --campaign-plan "$PLAN" --attempt 1 \
  --candidate-runtime "$RUNTIME" --candidate-source "$SELECTED" \
  --repository "$PWD" --promotion "$RELEASE/promotion.json" \
  --preflight "$RELEASE/release-preflight/reference.json" \
  --ci "$RELEASE/github-ci.json" \
  --output "$RELEASE/release-evidence.json"
```

`release-evidence.json` is sealed before dual authorization. It binds the
selected generated source, exact deployed source, runtime, architecture,
search macros, seven-slot configuration, four committed artifacts, maintained
and source-specific preflights, compiler/gate, full timing receipt, exact CI,
and a maintained source binding. Challenger `authorize-dual-final` consumes
this evidence and uses its deployed source; it never treats the selected
generated source as directly uploadable.

```bash
DEPLOYED="$PWD/submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.cpp"
"$PY" tools/compact_value_bfm_rank4_teacher_challenger.py authorize-dual-final \
  --plan "$PLAN" --attempt 1 --candidate-runtime "$RUNTIME" \
  --candidate-source "$SELECTED" --deployed-source "$DEPLOYED" \
  --release-evidence "$RELEASE/release-evidence.json" \
  --created-at-utc 2026-09-04T00:03:00Z
```

## Dual qualification and one upload

After governance authorizes the candidate, use the dual-final runner described
in `docs/compact-value-bfm-rank4-teacher-dual-final.md`. Its generalized route
calls `dual_final_preflight_state`, which recursively validates the release
evidence before either bank is materialized. Gate A and Gate B must each pass
their independent 1,000-game actual-clock gate unchanged.

Only after `dual-qualified.json` exists may the upload authorization be made:

```bash
"$PY" tools/compact_value_bfm_rank4_teacher_release.py authorize-upload \
  --campaign-plan "$PLAN" --attempt 1 \
  --candidate-runtime "$RUNTIME" --candidate-source "$SELECTED" \
  --release-evidence "$RELEASE/release-evidence.json" \
  --dual-qualified "$DUAL_ROOT/dual-qualified.json" \
  --output-root "$RELEASE/upload"

"$PY" tools/compact_value_bfm_rank4_teacher_release.py validate-upload \
  --campaign-plan "$PLAN" --attempt 1 \
  --candidate-runtime "$RUNTIME" --candidate-source "$SELECTED" \
  --release-evidence "$RELEASE/release-evidence.json" \
  --dual-qualified "$DUAL_ROOT/dual-qualified.json" \
  --output-root "$RELEASE/upload"
```

The emitted `one-upload-authorization.json` uses the existing
`papersoccer.compact-value-bfm.one-upload-authorization.v1` schema. It
authorizes exactly one upload and one Submit click, keeps Rank-4 replacement
unauthorized, and additionally binds the release evidence and both-gate dual
qualification. The maintained fresh-editor, copyback, Play, one-shot Submit,
attestation, and exact-90 live-window tools consume that authorization through
a strict lazy revalidation of the release/dual chain; none is run automatically
by this bridge. Set `UPLOAD_ROOT="$RELEASE/upload"` and
`DEPLOYED="$PWD/submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.cpp"`;
the maintained lifecycle is:

```bash
"$PY" tools/compact_value_bfm_upload.py fresh-editor \
  --ledger-root "$UPLOAD_ROOT" --session-id FRESH_EDITOR_SESSION
"$PY" tools/compact_value_bfm_upload.py copyback \
  --ledger-root "$UPLOAD_ROOT" --generated-source "$DEPLOYED" \
  --copied-back-source /absolute/path/to/editor-copyback.cpp
"$PY" tools/compact_value_bfm_upload.py play \
  --ledger-root "$UPLOAD_ROOT" --legal-stdout --expected-telemetry
"$PY" tools/compact_value_bfm_upload.py start-submit \
  --ledger-root "$UPLOAD_ROOT"
"$PY" tools/compact_value_bfm_upload.py attest \
  --ledger-root "$UPLOAD_ROOT" --agent-id AGENT_ID \
  --submission-id SUBMISSION_ID
```
