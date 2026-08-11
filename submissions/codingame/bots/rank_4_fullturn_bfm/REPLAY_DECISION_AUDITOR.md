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

## Offline aggregation and hypothesis comparison

`analyze_decision_audit.py` reads one explicitly named auditor JSONL or TSV
file; it performs no directory scan, network request, replay collection, or
match-bank access. It fails closed on schema drift, duplicate JSON keys,
malformed derived fields, noncontiguous decision histories, mixed run
configurations, or provenance that changes within a file. Its report includes
win/loss decision counts, candidate/observed/rank-4 agreement, exact-action and
boundary-state retention ordinals, deadline and truncation pressure, slices by
result/color/phase, and the first action divergence in each affected game.
Phases are fixed by candidate decision index: opening 0--3, midgame 4--11, and
late game 12 or later.

```sh
python3 submissions/codingame/bots/rank_4_fullturn_bfm/analyze_decision_audit.py \
  --input decision-audit.jsonl --label baseline --format json
```

Two hypotheses can be compared only when every decision identity, replay
context, observed action, and provenance field aligns. Run configuration may
differ; the report records those differences and good/bad event transitions:

```sh
python3 submissions/codingame/bots/rank_4_fullturn_bfm/analyze_decision_audit.py \
  --input baseline.jsonl --label baseline \
  --compare hypothesis.jsonl --compare-label hypothesis --format json
```

The game count is explicitly described as "represented": a valid terminal
input game in which the candidate never moved produces no decision row and
therefore cannot be inferred by this downstream tool. Terminal cleanliness is
established by the replay auditor before it emits rows; the aggregator does not
reopen the original replay input to re-prove it.
