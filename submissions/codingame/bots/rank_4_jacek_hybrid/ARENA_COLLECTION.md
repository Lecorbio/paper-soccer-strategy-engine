# Hybrid arena validation window

`arena_window.py` is the hybrid-local entry point for one validation-only live
CodinGame window. Fresh arena data never becomes training data. The immutable
plan has two 90-game roles:

- `hybrid-validation-001`: required hybrid validation.
- `safe-h62-rollback-accounting`: optional accounting after a separately
  authorized manual restore of the exact safe H62 source.

The campaign runs from `2026-08-13T19:15:07Z`
(`2026-08-13T21:15:07+02:00`, Europe/Warsaw) through
`2026-08-15T07:15:07Z` (`2026-08-15T09:15:07+02:00`, Europe/Warsaw).
The wrapper requires plan creation inside that interval and requires
`T0 <= Play UTC <= upload UTC <= attestation UTC <= deadline`.

## 1. Freeze the plan before results

Run this before inspecting any result and retain the emitted `path` and
`sha256` as `PLAN.json` and `PLAN_SHA` below:

```sh
python3 submissions/codingame/bots/rank_4_jacek_hybrid/arena_window.py plan \
  --campaign results/rank_4_jacek_hybrid/campaign.json \
  --output-root results/rank_4_jacek_hybrid/arena \
  --repository .
```

## 2. Perform the hybrid browser ceremony

Do this only after all local gates pass and the exact generated source is
tracked, committed, and the tracked worktree is clean.

1. In the browser, paste the exact bytes of
   `submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp` into a fresh
   CodinGame editor session.
2. Before Play or Submit, select and copy the complete editor contents back to
   a new retained file such as
   `results/rank_4_jacek_hybrid/arena/editor-copybacks/hybrid-validation-001-<UTC-NONCE>.cpp`.
   Never overwrite or reuse a copy-back file.
3. Run **Play My Code**. Confirm legal stdout and expected telemetry, then
   record its UTC completion time as `PLAY_UTC`.
4. Click **Submit** once for this source. Record the completion time as
   `UPLOAD_UTC`, and capture the new positive agent ID and submission ID
   created by that exact Submit.
5. Do not reuse an agent or submission ID from Rank4, H62, another control,
   Play My Code, a dry run, an earlier campaign, or the other planned window.
   Freshness and cross-window uniqueness are manual audit requirements; the
   wrapper binds the supplied IDs but cannot infer their browser history.

CodinGame's public API does not expose editor source bytes. The copy-back is
therefore retained evidence. `attest` checks it is a distinct path from the
generated source and is byte-for-byte identical, ASCII, within 99,999 bytes,
and equal in byte count and SHA-256. It also checks the committed source and
clean tracked worktree.

Immediately after Submit, create the attestation. `ATTEST_UTC` must be no
earlier than `UPLOAD_UTC` and no later than the campaign deadline.

```sh
python3 submissions/codingame/bots/rank_4_jacek_hybrid/arena_window.py attest \
  --plan PLAN.json --plan-sha256 PLAN_SHA \
  --window-id hybrid-validation-001 \
  --generated-source submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp \
  --copied-back-source results/rank_4_jacek_hybrid/arena/editor-copybacks/hybrid-validation-001-UTC-NONCE.cpp \
  --repository . --repository-commit COMMIT \
  --agent-id FRESH_HYBRID_AGENT_ID \
  --submission-id FRESH_HYBRID_SUBMISSION_ID \
  --play-checked-at-utc PLAY_UTC --uploaded-at-utc UPLOAD_UTC \
  --created-at-utc ATTEST_UTC \
  --output-root results/rank_4_jacek_hybrid/arena \
  --compilation-ok --legal-action-ok --protocol-ok --purity-ok \
  --source-size-ok --timing-both-colors-ok \
  --play-stdout-legal --play-telemetry-ok
```

Retain the emitted attestation `path` and `sha256` as `ATTESTATION.json` and
`ATTESTATION_SHA`.

## 3. Monitor only the post-upload identity

Monitoring starts after the upload and attestation, never from a control ID or
before Submit. Start it immediately with the fresh identity bound in the
attestation:

```sh
python3 submissions/codingame/bots/rank_4_jacek_hybrid/arena_window.py watch-collect \
  --plan PLAN.json --plan-sha256 PLAN_SHA \
  --attestation ATTESTATION.json --attestation-sha256 ATTESTATION_SHA \
  --exclusion-registry results/rank_4_jacek_hybrid/arena/exclusions/578524a7cf0908fa29df3541b4f0b1f7f9e85c1c0374d0dbc0d656686e9df159.json \
  --exclusion-registry-sha256 578524a7cf0908fa29df3541b4f0b1f7f9e85c1c0374d0dbc0d656686e9df159 \
  --data-root results/rank_4_jacek_hybrid/arena \
  --repository .
```

Before completion, the command reports only progress counts or an attributable
focus-failure category. It refuses 91 or more matching games. Handle its exit
status exactly:

- `0`: exactly 90 complete games passed focus-safety inspection and a
  collection receipt was emitted. Retain its `path` and `sha256` as
  `COLLECTION.json` and `COLLECTION_SHA`.
- `2` with JSON status `timed-out-waiting`: no receipt was emitted. Continue
  the same bound window with the same attestation; do not Submit again or
  substitute IDs. A command-line usage error can also use status 2, so require
  the JSON status before treating it as a polling timeout.
- `42`: an attributable timeout, illegal action, crash, or malformed transcript
  produced a content-addressed failure receipt whose payload says
  `rollback_required: true`. This is a stop-and-escalate signal, not an
  automatic rollback: the wrapper does not edit the source, operate the
  browser, or submit H62. Preserve and check the failure receipt before any
  separately authorized manual rollback.
- `1`: provenance, artifact, filesystem, API, or collector failure. Stop and
  correct the stated cause; do not reinterpret it as a completed window.

## 4. Derive validation and recheck artifacts

Only a successful `hybrid-validation-001` receipt can be derived. The command
emits validation bindings and eligibility flags while recording zero training
games, value rows, action-ranking games, and policy rows:

```sh
python3 submissions/codingame/bots/rank_4_jacek_hybrid/arena_window.py derive \
  --plan PLAN.json --plan-sha256 PLAN_SHA \
  --attestation ATTESTATION.json --attestation-sha256 ATTESTATION_SHA \
  --collection-receipt COLLECTION.json \
  --collection-receipt-sha256 COLLECTION_SHA \
  --output-root results/rank_4_jacek_hybrid/arena \
  --repository .
```

Retain the emitted derivation `path` as `DERIVATION.json`. Revalidate all four
content-addressed wrapper artifacts with:

```sh
python3 submissions/codingame/bots/rank_4_jacek_hybrid/arena_window.py check \
  --artifact PLAN.json \
  --artifact ATTESTATION.json \
  --artifact COLLECTION.json \
  --artifact DERIVATION.json \
  --repository .
```

For an exit-42 path, run the same `check` command with the failure-receipt path
as `--artifact` instead of nonexistent collection/derivation artifacts.

## 5. Optional exact-H62 rollback accounting

This section records a manual rollback only after that action is separately
authorized; `arena_window.py` never performs it. The only accepted source for
`safe-h62-rollback-accounting` is the committed
`submissions/codingame/bots/jacek_native_bfm/submission.cpp`, exactly 99,810
ASCII bytes with SHA-256
`d9d96f83197f13b7212e7b652851097053ee7f1662845e06dd722d1c0bc24f71`.
Verify those facts before opening the browser:

```sh
test "$(LC_ALL=C wc -c < submissions/codingame/bots/jacek_native_bfm/submission.cpp | tr -d ' ')" = 99810
test "$(shasum -a 256 submissions/codingame/bots/jacek_native_bfm/submission.cpp | awk '{print $1}')" = d9d96f83197f13b7212e7b652851097053ee7f1662845e06dd722d1c0bc24f71
```

Repeat the full paste -> unique copy-back -> Play -> Submit -> fresh-ID and
timestamp ceremony. Use a distinct retained path such as
`results/rank_4_jacek_hybrid/arena/editor-copybacks/safe-h62-rollback-accounting-<UTC-NONCE>.cpp`
and a newly created agent/submission pair different from every hybrid or
control pair. Attest that exact copy with:

```sh
python3 submissions/codingame/bots/rank_4_jacek_hybrid/arena_window.py attest \
  --plan PLAN.json --plan-sha256 PLAN_SHA \
  --window-id safe-h62-rollback-accounting \
  --generated-source submissions/codingame/bots/jacek_native_bfm/submission.cpp \
  --copied-back-source results/rank_4_jacek_hybrid/arena/editor-copybacks/safe-h62-rollback-accounting-UTC-NONCE.cpp \
  --repository . --repository-commit COMMIT \
  --agent-id FRESH_H62_AGENT_ID \
  --submission-id FRESH_H62_SUBMISSION_ID \
  --play-checked-at-utc H62_PLAY_UTC --uploaded-at-utc H62_UPLOAD_UTC \
  --created-at-utc H62_ATTEST_UTC \
  --output-root results/rank_4_jacek_hybrid/arena \
  --compilation-ok --legal-action-ok --protocol-ok --purity-ok \
  --source-size-ok --timing-both-colors-ok \
  --play-stdout-legal --play-telemetry-ok
```

Start its `watch-collect` only after that H62 upload and attestation, using the
H62 attestation path/hash and otherwise the same command and exclusion binding.

The optional H62 receipt is accounting-only and training-forbidden. `derive`
deliberately rejects the rollback-accounting role; use `check` on its plan,
attestation, and collection or failure receipt instead.
