# Jacek-native replay decision auditor

`papersoccer_jacek_native_replay_decision_auditor` turns an already archived,
operationally clean CodinGame arena transcript into one diagnostic row per
candidate decision. It runs only `jacek_native_bfm`: the native complete-turn
generator, native value network, and native BFM search. It does not load an
incumbent evaluator, replay override, residual model, action label, match bank,
or protected evidence.

The auditor is diagnostic, not a training-label generator. An observed move is
reported as `actual_action`; it is never asserted to be a correct action or
converted into a policy target. A live loss can be restarted by the separate
native self-play pipeline only after its state/provenance treatment is recorded
as development data.

## Provenance-safe input

The input contract is exactly the four-column TSV emitted by
`collect_arena_batch.py --export-auditor-tsv`:

```text
# agent_id=6600001
# arena_manifest_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
# asserted_source_sha256=abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789
# asserted_submission_id=41100001
# source_binding_status=asserted-not-api-verified
game_id	candidate_player	winner	turns
123456789	0	1	2/6/71/0/33
```

Every leading `# safe_key=printable_value` pair is bounded, must be unique, and
is copied unchanged into `input_provenance` on every JSONL row and
`input_provenance_json` on every TSV row. The collector's source-binding status
remains an assertion; the auditor never upgrades it to API verification.

Before emitting any row, the auditor replays every complete game under the
production `OwnGoalsAllowed + MoverLoses` rules. It rejects illegal directions,
an omitted mandatory rebound, action bytes after handoff or termination,
nonterminal transcripts, repeated game IDs, and winner mismatches. The
collector already excludes operationally terminated games and protected or
previously known IDs before producing this TSV.

The auditor reads only the explicit `--input` path (or standard input). It does
not scan replay directories, make network requests, or open any match bank.

## Deterministic work and real-clock modes

Use fixed tree-node work for reproducible comparisons:

```sh
build/papersoccer_jacek_native_replay_decision_auditor \
  --input arena-games.tsv --fixed-work 30000 --format jsonl \
  > native-decisions.fixed.jsonl
```

`--fixed-work` maps directly to the native search's tree-node cap. This is a
debugging unit, not a strength metric. Neighbor shuffling, generation, UCT tie
breaks, and output actions are deterministic when no clock is active.

Use the deployed search profile when the hypothesis is specifically about
CodinGame timing:

```sh
build/papersoccer_jacek_native_replay_decision_auditor \
  --input arena-games.tsv --codingame-clocks --format tsv \
  > native-decisions.clock.tsv
```

`--codingame-clocks` freezes 800 ms for the first own decision and 155 ms for
later own decisions, with the production 250-action, 4,000-root-path,
80,000-tree-node, 2,000,000-expansion profile. The first budget is keyed to
`own_decision_index == 0`, including when the candidate plays second. Clocked
actions and elapsed times can vary with host load; playing-strength decisions
must use the established paired actual-clock gates, not this offline replay.

`--first-ms` and `--later-ms` are available for explicit clock experiments and
must be supplied together. `--max-actions`, `--max-partial-paths`,
`--max-expansions`, `--exploration`, and `--fpu` are bounded diagnostic knobs.
Every row records the complete configuration, model artifact SHA-256, and
packed-weight SHA-256.

## Per-decision evidence

For both the observed (`actual_*`) and counterfactual native (`chosen_*`)
action, schema `jacek-native-decision-audit-v1` records:

- zero-based exact-encoding and full-boundary retained ordinals (`-1` means
  absent);
- the generator's canonical retained encoding and tactical class;
- nullable initial neural value, mover-relative initial action value, and
  one-based initial rank;
- nullable final backed value, visits, BFM selection visits, and the tactical
  class attached to that searched root child.

Exact tactical proofs intentionally have a null neural value: terminal goals,
own goals, forced cutoffs, and immediate opponent goals are scored by native
mate-distance logic. Safe handoffs receive the value network's mover-relative
score. Initial ranks sort those action values descending and break ties by the
retained encoding. Full boundary equivalence uses the complete compact state,
not only the final ball vertex.

Search rows additionally report tree nodes, expansions, generated children,
network child evaluations, tactical child values, partial paths, completed
actions, duplicate boundaries, LIFO/FIFO extractions, tactical actions,
truncations, and deadline/tree/expansion-cap flags. A separate deterministic
root pass reports maximum deque size and exhaustiveness; this makes a missing
move distinguishable from a searched move whose value changed.

## Classification contract

The `classification` field explains an observed-versus-counterfactual
difference; it does not assert that the observed action was objectively best.
Precedence is fixed:

1. `match`: identical complete-turn encoding.
2. `boundary-equivalent`: different encoding, identical complete boundary.
3. `generator-omission`: the observed boundary is absent from the deterministic
   diagnostic retained set.
4. `operational-failure`: deadline/root-set pressure prevents a comparable
   retained final value.
5. `tactical-miss`: the observed boundary has a strictly stronger exact
   tactical class than the chosen boundary.
6. `bfm-override`: BFM selects a retained boundary other than the initial
   value leader.
7. `initial-evaluator-ordering`: the search preserves the initial native value
   leader over the observed retained boundary.

`classification_reason` stores the corresponding human-readable condition.
The operational category here is limited to decision-search pressure. Arena
forfeits, timeouts, invalid output, and runtime errors remain the collector's
game-level operational classifications and are deliberately absent from its
clean auditor export.

## Build and focused verification

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 --target \
  papersoccer_jacek_native_replay_decision_auditor \
  papersoccer_jacek_native_replay_decision_auditor_test
ctest --test-dir build --output-on-failure \
  -R jacek_native_replay_decision_auditor
```

The focused tests freeze the collector TSV parser and provenance behavior,
production replay rules, all classification buckets, fixed-work determinism,
work/action/deque counters, initial/final action diagnostics, JSONL/TSV field
parity, exact 800/155 clock selection, bounded options, and fail-before-output
validation.
