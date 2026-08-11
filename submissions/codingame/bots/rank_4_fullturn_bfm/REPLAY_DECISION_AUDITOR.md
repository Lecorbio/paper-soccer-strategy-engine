# Full-turn replay decision auditor

`papersoccer_fullturn_replay_decision_auditor` replays already-collected,
complete CodinGame games under `OwnGoalsAllowed + MoverLoses` and audits every
decision made by one selected player. From the same pre-action state it runs:

- the full-turn BFM as a raw counterfactual search, without letting the replay
  book replace its action;
- the canonical rank-4 search;
- the deployed candidate's replay-book lookup and legality check.

The reconstructed deployed action is the valid replay correction when one
exists, and the raw BFM action otherwise. This keeps search-quality diagnosis
separate from exact-history corrections while still explaining the action the
deployed control flow would choose. Search elapsed time always measures the raw
counterfactual search, even on a row where deployment would have returned a
replay correction without searching.

## Input and provenance

The input is strict UTF-8/ASCII-compatible TSV. The header and four fields are
required:

```text
# agent_id=6606663
# asserted_submission_id=41119120
# source_binding_status=asserted-not-api-verified
# asserted_source_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
game_id	candidate_player	winner	turns
synthetic-own-goal	0	0	0/0/0/0/0/0
```

- `candidate_player` and `winner` use CodinGame IDs `0` and `1`.
- `turns` contains non-empty, slash-separated complete-turn actions using only
  directions `0` through `7`.
- Each game must end exactly after its final action, and its declared winner
  must match the replay. The full input is validated before output begins.
- Leading `# safe_key=printable_value` comments are bounded, must have unique
  keys, and are copied into `input_provenance` on every JSONL row and
  `input_provenance_json` on every TSV row. Ordinary descriptive comments are
  ignored. Metadata after the header is not treated as provenance.

The collector deliberately labels the source binding as
`asserted-not-api-verified`: public CodinGame battle data identifies the agent
and submission but does not prove editor-source bytes. The auditor preserves
that distinction and does not upgrade the assertion into source verification.

## Deterministic and clock-limited modes

Fixed work is the default deterministic diagnostic:

```sh
build/papersoccer_fullturn_replay_decision_auditor \
  --input validated-games.tsv \
  --candidate-work 30000 \
  --reference-nodes 30000 \
  --format jsonl > decision-audit.jsonl
```

In this mode all reported time limits are zero. Work and node counts are useful
for reproducibility and profiling, not as evidence of playing strength.

Use the convenience flag for the deployed CodinGame clock model:

```sh
build/papersoccer_fullturn_replay_decision_auditor \
  --input validated-games.tsv \
  --codingame-clocks \
  --format jsonl > decision-audit-clock.jsonl
```

`--codingame-clocks` sets both engines to 800 ms for their first own decision
and 165 ms for every later own decision. It also restores the production hard
caps of 3,000,000 candidate work units and 3,000,000 reference nodes, regardless
of lower work/node values elsewhere on the command line. The first clock is
keyed by `own_decision_index == 0`, so a candidate playing second still receives
the first clock on its first response.

For controlled experiments, set the four clocks independently with
`--candidate-first-ms`, `--candidate-later-ms`, `--reference-first-ms`, and
`--reference-later-ms`. The compatibility options `--candidate-time-ms` and
`--reference-time-ms` set both clocks for their engine. Every row reports the
configured first/later pair and the actual per-decision limit selected from it.
The deadline includes search-object construction. A multi-game auditor process
does reuse immutable static model decoding after its first search, whereas each
CodinGame battle starts a fresh process; use action/stability evidence rather
than treating local elapsed time as exact platform timing.

## Output

Each JSONL or TSV row identifies the game, exact transcript prefix, canonical
state digest, color, result, and input provenance. Action fields distinguish:

- `actual_action`: action observed in the archived game;
- `candidate_action`: raw BFM counterfactual action;
- `reference_action`: canonical rank-4 counterfactual action;
- replay lookup, validity, and correction action;
- `reconstructed_deployed_action` and whether it matches the observed action.

Rows also contain BFM work, tree size, expansions, child evaluations, generator
partial paths/completions/duplicates/tactical actions/truncations, deadline and
node-cap flags, and root score. A separate deterministic root-generation pass
reports the retained action count, exhaustiveness, duplicate/truncation counts,
and the exact-action and boundary-state ordinal for the observed, BFM, and
rank-4 choices. An ordinal of `-1` means the action or endpoint was absent;
these fields separate generator omission from evaluator/search selection.
Rank-4 node/depth counters and both elapsed times are included for context. Run
`--help` for all bounded generator, UCT/FPU/final-visit-weight, evaluator,
reference-depth, and clock configuration hooks.
The run configuration also records `candidate_nonroot_actions` (`0` means the
root cap is reused) and `candidate_root_only`, so width and one-ply diagnostic
audits cannot be confused with the deployed full-width BFM policy.

Schema `fullturn-decision-audit-v3` also scores every boundary retained by that
separate root-generation pass with the candidate's exact depth-one child-value
semantics. Terminal actions use the depth-one mate score; forced cutoffs and
opponent-immediate-win classifications use the same depth-two proof score as
root expansion; all other actions use the candidate heuristic, replay blend,
and teacher residual. Residual enablement is inherited from the pre-action
root's used-edge count, exactly as it is by the search object, rather than being
recomputed from each child boundary.

The initial-evaluation fields are:

- `initial_eval_best_action`, `initial_eval_best_score`, and its zero-based
  `initial_eval_best_retained_ordinal`;
- nullable candidate-evaluator scores and one-based ranks for the retained
  observed, BFM, and rank-4-reference boundaries; the corresponding existing
  `*_boundary_retained_ordinal` identifies the scored boundary;
- `candidate_bfm_change_assessable` and
  `candidate_bfm_changed_from_initial_best`.

Ranks reproduce the root-only candidate choice order. Candidate desirability
is oriented to the root mover, solved mate values use the search's normalized
proof value, and ties use the retained action's encoded string. Thus rank 1 is
the action that the candidate would select after root expansion without deeper
BFM. Because root generation deduplicates endpoint states, an observed or
counterfactual encoding is scored through its retained boundary representative;
its exact encoding can differ from `initial_eval_best_action` while reaching
the same boundary. A boundary absent from the diagnostic set has a `null` score
and rank `-1`; in that case BFM change assessment is false and explicitly
marked unassessable.

`candidate_bfm_changed_from_initial_best` reports a boundary difference, not
causal proof that deeper search alone made the change. A timed candidate search
can see a smaller root set or fall back under deadline, work, tree, or generator
pressure. Interpret it together with retention and pressure fields. Likewise,
the observed and rank-4 scores are values assigned by the candidate evaluator
to those retained boundaries; they are not recovered scores from the deployed
historical process or from rank 4's evaluator.

This field set intentionally bumps v2 to v3. The analyzer is strict about exact
field sets, so reusing the v2 identifier for append-only fields would make two
incompatible documents claim the same schema.

## Offline aggregation and hypothesis comparison

`analyze_decision_audit.py` reads one explicitly named auditor JSONL or TSV
file; it performs no directory scan, network request, replay collection, or
match-bank access. It fails closed on schema drift, duplicate JSON keys,
malformed derived fields, noncontiguous decision histories, mixed run
configurations, or provenance that changes within a file. Its report includes
win/loss decision counts, candidate/observed/rank-4 agreement, exact-action and
boundary-state retention ordinals, deadline and truncation pressure, slices by
result/color/phase, an exact first-own-decision versus later-own-decisions clock
slice, and the first action divergence in each affected game. It
also reports initial-evaluator ranks, BFM changes away from rank 1, and mechanism
buckets that distinguish an initial evaluator preference from a later BFM
override. In particular, `bfm_corrected_initial_misranking` means the observed
boundary was not initially rank 1 but was selected by BFM, while
`bfm_overrode_actual_initial_best` means the observed boundary began at rank 1
and BFM selected another retained boundary. These are diagnostic labels, not
claims that the observed action was objectively optimal.
Phases are fixed by candidate decision index: opening 0--3, midgame 4--11, and
late game 12 or later.

The analyzer mirrors the native auditor's configuration bounds: candidate work
1--3,000,000, tree nodes 2--120,000, retained actions 1--250, non-root actions
0--250, partial paths 1--50,000, reference nodes 1--3,000,000, and reference
depth 1--32. Exploration and final-visit weight must be finite and nonnegative;
FPU must be finite. Reported work, tree, reference, depth, and semantically
related generator counters cannot exceed their configured limits. A missing
boundary has exactly score `null` and rank `-1`; a retained boundary has an
integer score and a one-based rank within the retained root set. An exact
retained action always has the same ordinal as its retained boundary, and
same-action/ordinal/evaluation tuples must agree across observed, candidate,
and rank-4 fields.
Initial scores are either bounded heuristic values in `[-100000, 100000]` or
the exact depth-one/depth-two proof magnitudes 999999/999998. Observed and
rank-4 tactical classes additionally fix the proof sign from the candidate
player, while `safe-handoff` must use a heuristic-range score. Root search
scores remain bounded by the separate ±1,000,000 mate scale. Deadline or node
cap pressure must imply candidate budget exhaustion, and diagnostic-root
exhaustiveness must agree with whether truncation occurred.
When `codingame_clock_mode` is true, the analyzer additionally requires the
producer's canonical 800/165 ms first/later clocks for both engines and the
3,000,000 candidate-work/reference-node caps; the flag cannot be paired with a
custom-clock configuration.

```sh
python3 submissions/codingame/bots/rank_4_fullturn_bfm/analyze_decision_audit.py \
  --input decision-audit.jsonl --label baseline --format json
```

An audit exported from `collect_arena_batch.py` can be joined back to its one
explicitly named collector manifest:

```sh
python3 submissions/codingame/bots/rank_4_fullturn_bfm/analyze_decision_audit.py \
  --input decision-audit.jsonl \
  --arena-manifest results/codingame_arena_diagnostics/manifests/6606663/41119120/SOURCE_SHA/MANIFEST_SHA.json \
  --label arena --format json
```

This option still performs no directory scan, network request, or implicit
cross-reference read. It reads only the requested audit and manifest. The
manifest must be strict canonical JSON with the collector batch, binding, game,
and diagnostic-purpose schemas. A 64-lowercase-hex manifest filename is treated
as a content-addressed name and must match the bytes. Every embedded game record
must match its canonical record hash and its `record_path` must name that hash;
the analyzer validates the embedded record rather than opening that path.

The join fails closed unless every audit row repeats the manifest's agent,
asserted submission, asserted source hash, collector hash, exclusion-registry
hash, repository commit, run ID, source-binding status, and actual manifest
hash. Accepted records must agree with the binding. Only operationally clean,
rule-terminal accepted records are eligible. Game IDs are unique, and each
eligible game's candidate player, color, winner, result, transcript prefixes,
observed actions, and complete candidate-decision count must agree with the
audit.

Manifest mode also requires `full_window_accounted=true` and independently
recomputes that claim from the expected game count and every embedded record's
status. An incomplete battle window is rejected rather than presented as an
arena-wide rank or opponent result.

Coverage is exact: the audit must represent every eligible manifest game in
which the candidate had at least one decision, and no other game. A clean game
that terminated before the candidate's first move is validated and reported in
`clean_games_without_candidate_decision`, but correctly has no audit row. This
is the only subset rule; arbitrary partial-manifest audits are rejected.

The joined report adds frozen-rank cohorts for ranks 1--5, 1--10, 1--20, 21+,
and unranked opponents. The top-5, top-10, and top-20 cohorts are deliberately
cumulative. It also adds per-opponent display-name summaries only after the
same opponent agent/name has at least three audited games, preventing a single
long replay from satisfying the threshold. Comparisons use the same validated
manifest join and receive the same rank and named-opponent slices. Every cohort
and named-opponent bucket retains decision-level diagnostics while also
reporting deduplicated game totals, wins, and losses, so long games do not
silently stand in for multiple arena results.

Two hypotheses can be compared only when every decision identity, replay
context, observed action, and provenance field aligns. Run configuration may
differ; the report records those differences and good/bad event transitions:

```sh
python3 submissions/codingame/bots/rank_4_fullturn_bfm/analyze_decision_audit.py \
  --input baseline.jsonl --label baseline \
  --compare hypothesis.jsonl --compare-label hypothesis \
  --arena-manifest MANIFEST_SHA.json --format json
```

Without `--arena-manifest`, the game count is explicitly described as
"represented": a valid terminal input game in which the candidate never moved
produces no decision row and therefore cannot be inferred from the standalone
audit. With a manifest join, such games are instead validated from the embedded
clean records and listed under `clean_games_without_candidate_decision` as
described above. In standalone mode terminal cleanliness is established by the
replay auditor before it emits rows; the aggregator does not reopen the original
replay input to re-prove it.
