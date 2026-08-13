# Hybrid arena validation window

`arena_window.py` is the only hybrid-local entry point for a live CodinGame
window. It deliberately supports validation, not training. The blind plan
contains exactly two 90-game roles:

- `hybrid-validation-001`: required arena validation; every game and every
  derived row is forbidden for training.
- `safe-h62-rollback-accounting`: optional accounting after restoring the
  exact 99,810-byte H62 source with SHA-256
  `d9d96f83197f13b7212e7b652851097053ee7f1662845e06dd722d1c0bc24f71`.

Create the plan before inspecting any result:

```sh
python3 submissions/codingame/bots/rank_4_jacek_hybrid/arena_window.py plan \
  --campaign results/rank_4_jacek_hybrid/campaign.json \
  --output-root results/rank_4_jacek_hybrid/arena
```

After all local gates pass, commit the exact generated source. Paste it into
the IDE, copy it back to a retained file, run Play My Code, and attest it. The
attestation command rejects a dirty tracked worktree, an untracked/uncommitted
generated source, a copy-back path aliased to the generated file, non-ASCII,
more than 99,999 bytes, a byte/count/SHA mismatch, or any false preflight/Play
flag.

```sh
python3 submissions/codingame/bots/rank_4_jacek_hybrid/arena_window.py attest \
  --plan PLAN.json --plan-sha256 PLAN_SHA \
  --window-id hybrid-validation-001 \
  --generated-source submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp \
  --copied-back-source results/rank_4_jacek_hybrid/arena/editor-copyback.cpp \
  --repository . --repository-commit COMMIT \
  --agent-id AGENT --submission-id SUBMISSION \
  --play-checked-at-utc PLAY_UTC --uploaded-at-utc UPLOAD_UTC \
  --compilation-ok --legal-action-ok --protocol-ok --purity-ok \
  --source-size-ok --timing-both-colors-ok \
  --play-stdout-legal --play-telemetry-ok
```

The CodinGame public API cannot return editor source bytes. Upload identity is
therefore editor-attested by exact paste/copy-back byte, count, and SHA-256
equality; every attestation records that limitation explicitly.

`watch-collect` polls only the bound agent/submission. It incrementally archives
new metadata/details, but before completion prints only progress counts. An
attributable focus timeout, illegal action, crash, or malformed transcript
returns exit code `42` immediately and requires rollback. It refuses 91 or more
matching games. The generic collector is invoked only after exactly 90 complete
games have all passed the focus-safety inspection, with internally constructed
immutable source/commit/submission/exclusion bindings. Its content-addressed
source, registry, manifest, all 90 game records, and their raw/normalized/replay
payloads are verified before a collection receipt is emitted.

```sh
python3 submissions/codingame/bots/rank_4_jacek_hybrid/arena_window.py watch-collect \
  --plan PLAN.json --plan-sha256 PLAN_SHA \
  --attestation ATTESTATION.json --attestation-sha256 ATTESTATION_SHA \
  --exclusion-registry results/rank_4_jacek_hybrid/arena/exclusions/578524a7cf0908fa29df3541b4f0b1f7f9e85c1c0374d0dbc0d656686e9df159.json \
  --exclusion-registry-sha256 578524a7cf0908fa29df3541b4f0b1f7f9e85c1c0374d0dbc0d656686e9df159 \
  --data-root results/rank_4_jacek_hybrid/arena
```

Finally, `derive` emits only validation bindings and eligibility flags. It
always records zero training games, value rows, action-ranking games, and policy
rows. `check --artifact PATH` revalidates any content-addressed plan,
attestation, collection receipt, or derivation.
