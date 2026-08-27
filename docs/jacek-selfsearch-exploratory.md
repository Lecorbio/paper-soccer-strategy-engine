# Exploratory replay self-search pilot

`tools/jacek_selfsearch_exploratory.py` runs one local 2,000-game experiment.
It keeps the v6 pilot runtime (`bfcc1755…`) as the frozen experimental
reference and uses the development-gate-failed `canonical-r2-s20260825`
runtime (`5ecbd618…`) only in the existing runner-up diversity slots: 100
games in each color. It does not classify either runtime as a canonical
incumbent.

The pilot otherwise reuses the unchanged self-search pilot profile and
`run_phase` implementation. Search and Rank-4 labels still require their
strict proof or completed fixed-work-cap evidence. Training still uses the
paired search/Rank-4 arms, canonical anchors, fixed retention margins, four
unchanged game panels, legality/completion checks, and latency limits. The only
profile changes are a new campaign ID and fresh fixed game/opening seeds.

The exploratory supervisor is pilot-only. A rejection stops. A pass writes an
immutable `full-continuation-eligible.json` receipt for a separately authorized
10,000-game run, but this tool has no full-phase or publication path and never
creates a `full` directory. It never uploads, replaces Rank-4, promotes a
model, or changes the canonical incumbent.

## Evidence boundary

The interactive freeze binds the completed no-candidate rebuild, including its
status, canonical phase, selected near-pass candidate, development reports,
and exact actor hashes. All preserved report snapshots are checked before
their decisions are locally recomputed. The rebuild development and already
opened evaluation/v5/v6 banks are opening exclusions.

The sealed-final opening TSV is not read, hashed, parsed, or used as an
exclusion. Its opaque identity from the already-frozen bank manifest is
recorded only to attest that it remained untouched. Blind-holdout labels are
never opened. Canonical protected test shards are not used to choose the
failed diversity source; they are later used only by the unchanged
post-training retention gate for the newly trained pilot model.

## Freeze and launch

Use the campaign Python environment for both commands. First create the normal
self-search Release build manifest at the clean launch commit. Then run the
interactive freeze, which is the only step allowed to inspect Git:

```text
build/selfsearch-python/bin/python tools/jacek_selfsearch_exploratory.py \
  freeze-launch \
  --repository REPOSITORY --expected-commit COMMIT \
  --rebuild-inputs REBUILD/evidence/rebuild-inputs.json \
  --rebuild-summary REBUILD/run/final-summary.json \
  --canonical-campaign CANONICAL \
  --output-directory RESULTS/selfsearch-exploratory-20260827-v1 \
  --continuation-generator BUILD/papersoccer_jacek_replay_continuations \
  --search-teacher BUILD/papersoccer_jacek_replay_search_teacher \
  --rank4-teacher BUILD/papersoccer_jacek_replay_rank4_position_teacher \
  --comparison BUILD/papersoccer_jacek_replay_comparison \
  --pack-tool REPOSITORY/tools/jacek_replay_pack.py \
  --trainer REPOSITORY/tools/jacek_replay_train.py \
  --build-manifest RESULTS/selfsearch-exploratory-20260827-v1/release-build.json
```

The persistent process consumes only the frozen launch receipt:

```text
build/selfsearch-python/bin/python tools/jacek_selfsearch_exploratory.py run \
  --launch-receipt \
    RESULTS/selfsearch-exploratory-20260827-v1/exploratory-launch.json \
  --output-directory RESULTS/selfsearch-exploratory-20260827-v1 \
  --resume
```

`run` validates file hashes, the exact Python/NumPy environment, binaries,
tools, canonical shards, actors, and opening exclusions. It never invokes Git
or reads `.git`, making it suitable for the persistent Interactive
LaunchAgent.
