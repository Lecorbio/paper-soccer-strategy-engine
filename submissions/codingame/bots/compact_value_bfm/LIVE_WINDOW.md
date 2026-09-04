# Compact value-BFM live window

The live arena window is diagnostic-only. It cannot train a model, authorize
another upload, trigger a rollback, replace Rank 4, or support a Rank-1 claim.

Before the one permitted upload, freeze the existing ID-only exclusion
registry. This command reads no replay payloads:

```sh
python submissions/codingame/bots/compact_value_bfm/live_window.py \
  bind-exclusions \
  --registry /path/to/id-only-exclusions.json \
  --frozen-at-utc 2026-08-31T12:00:00Z \
  --output results/compact_value_bfm/compact-value-bfm-20260831-v1/live/exclusion-binding.json
```

After the upload ledger reaches the immutable `submission-attested` event,
collect the exact matching window:

```sh
python submissions/codingame/bots/compact_value_bfm/live_window.py watch \
  --submission-attestation results/compact_value_bfm/compact-value-bfm-20260831-v1/upload/05-submission-attested.json \
  --exclusion-binding results/compact_value_bfm/compact-value-bfm-20260831-v1/live/exclusion-binding.json \
  --data-root results/compact_value_bfm/compact-value-bfm-20260831-v1/live/archive
```

The wrapper filters by the exact agent and submission IDs. At 89 complete
games it waits; 91 matching games are rejected. It keeps collecting through
game 90 after a focus timeout, crash, illegal action, or malformed output, then
marks the completed window rejected. Opponent failures are recorded separately
and are never counted as strength wins.

Raw API payloads, acquisition receipts, sanitized monitor records, generic
collector records, manifests, and the final window receipt are append-only and
content-addressed. Re-running `watch` after completion validates and returns
the existing immutable reference.

Verify the archive with:

```sh
python submissions/codingame/bots/compact_value_bfm/live_window.py verify \
  --reference results/compact_value_bfm/compact-value-bfm-20260831-v1/live/archive/live-window.reference.json \
  --data-root results/compact_value_bfm/compact-value-bfm-20260831-v1/live/archive
```
