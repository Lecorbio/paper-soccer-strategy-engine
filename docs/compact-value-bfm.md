# Compact value-BFM challenger

The sanitized outcome of the completed Rank-4 qualification and live diagnostic
is recorded in [the Rank-4 campaign outcome](compact-value-bfm-rank4-campaign-outcome.md).

`compact_value_bfm` is a separate, value-only CodinGame research entrant.  It
does not replace `rank_4`, inherit a public rank, or make a Rank-1 claim.  Its
deployment state is determined only by the content-addressed receipts in its
own campaign namespace.

## Design boundary

The bot uses the compact 105-vertex, 316-edge state representation developed
for `jacek_arena_bfm`, but it does not reuse that entrant's checkpoint, corpus,
arena results, or average-return search.  One tree edge is one legal complete
rebound turn.  Generated turns are deduplicated by their full boundary state,
tactical goal and cutoff witnesses are retained ahead of ordinary handoffs,
and ancestor values are overwritten by negamax/minimax.  No policy head,
player identity feature, replay lookup, or transcript book is present.

The 6,301 sparse inputs follow the public value representation described by
Jacek Dermont in [Paper Soccer](https://github.com/jdermont/playground-kvhfh5iv/blob/5f369d6e64870690d100f0c3fe2f8e75c447dac3/papersoccer.md)
and his [CodinGame article](https://www.codingame.com/playgrounds/157341/inputs-for-neural-networks-for-the-board-games/paper-soccer).
The complete-turn search is informed by the public
[`QtPaperSoccer` source at revision `366d5304`](https://github.com/jdermont/QtPaperSoccer/tree/366d5304c09c2c820bd3ef4ea94624c034b8d955):
316 used-edge inputs plus one of 57 distance/free-degree categories for each
of 105 vertices.  This repository's implementation is independent.  Player
Two positions rotate 180 degrees, active indices are sorted, and native parity
is checked against the independent Python research encoder.

The deployment network is bias-free and value-only.  Its first hidden layer
uses square/leaky-0.01, the second uses leaky-ReLU-0.01, and the final scalar
uses the deployment polynomial approximation to `tanh`.  Weights use
per-layer symmetric signed three-bit codes in `[-3,3]`; the unused two's-
complement code for `-4` is rejected.

## Frozen inputs and protected tests

The accepted large-teacher campaign is an input source, not a strength claim.
Its compact-student handoff explicitly records that the earlier 20 ms pilot
failed its primary gates.  The later teacher-only override authorized local
teacher-data generation; it did not authorize deployment, canonical promotion,
or replacement of Rank 4.

Only `freeze-inputs` may inspect that retained campaign.  It follows a fixed
allowlist without directory enumeration, rejects `sealed-final`,
`sealed_final`, `blind-label`, and `blind_label` markers before resolving or
statting a path, and atomically creates a relative-path bundle.  The bundle
contains the paired Search and Rank-4 shards, ordered canonical R0/R1/R2
anchors, the common adjudicator, the accepted teacher runtime, canonical
roots, and the seven already-open opening banks.  Runtime and resume commands
use only the copied bundle and do not inspect Git or an old worktree.

Search, matched Rank-4, and canonical test NPZs remain semantically locked
through model, seed, float epoch, scale, QAT, and quantized-artifact selection.
Only the separate post-selection command can open them, and their metrics
cannot change selection.

```sh
accepted_teacher_campaign=/absolute/path/to/large-teacher-campaign-20260828-v1
.venv/bin/python tools/compact_value_bfm_workflow.py freeze-inputs \
  --source-campaign "$accepted_teacher_campaign" \
  --output-directory \
  results/compact_value_bfm/compact-value-bfm-20260831-v1

.venv/bin/python tools/compact_value_bfm_workflow.py verify \
  --bundle-manifest \
  results/compact_value_bfm/compact-value-bfm-20260831-v1/input-bundle/bundle-manifest.json

.venv/bin/python tools/compact_value_bfm_workflow.py run \
  --bundle-manifest \
  results/compact_value_bfm/compact-value-bfm-20260831-v1/input-bundle/bundle-manifest.json \
  --output-directory \
  results/compact_value_bfm/compact-value-bfm-20260831-v1/family-run \
  --workers 2

# After interruption, the same command must add --resume. Completed seed
# references are revalidated; an unfinished seed restarts at epoch zero.
.venv/bin/python tools/compact_value_bfm_workflow.py verify \
  --bundle-manifest \
  results/compact_value_bfm/compact-value-bfm-20260831-v1/input-bundle/bundle-manifest.json \
  --run-output-directory \
  results/compact_value_bfm/compact-value-bfm-20260831-v1/family-run \
  --run-reference \
  results/compact_value_bfm/compact-value-bfm-20260831-v1/family-run/run-state/run-reference.json
```

Raw shards, sidecars, checkpoints, opening banks, games, and transcripts stay
under ignored `results/compact_value_bfm/`.  Only a selected deployable source,
its compact public evidence, and non-sensitive receipts may be tracked.

The submission exporter uses a deterministic token-boundary-preserving C++
compaction pass.  Preprocessor directives and literals remain byte-exact while
comments and non-semantic whitespace are removed.  The hard CodinGame limit is
still 95,000 ASCII characters; new search candidates target at least 2,000
characters of reserve before qualification.

## Advancement and local/live limits

Offline evaluator metrics are necessary but not sufficient.  A candidate that
meets the common-adjudicator and canonical-validation thresholds is classified
`offline-evaluator-qualified-not-game-gated`.  It must then survive disjoint
development screens and a fresh 500-pair, both-color gate against the exact
maintained Rank-4 source.  Any illegal action, unfinished game, timeout, crash,
malformed response, overlong game, source mismatch, stale shard, or spent
one-shot claim is a failure rather than a statistical loss.

Local paired games establish only the recorded rules, binaries, openings,
machine, and clocks.  They do not predict CodinGame matchmaking, opponent mix,
host contention, or rating.  Likewise, a legal 90-game live window is
diagnostic: it cannot revise model selection and is never training data.  An
operational failure attributable to this bot rejects the whole live window;
an opponent failure is reported separately and is not credited as a strength
win.

Prior negative results remain relevant context.  The scratch-trained
`jacek_arena_bfm` family lost its mandatory H62 screens despite clean local
protocol and timing checks, and the large-teacher pilot's 20 ms primary gate
was false.  Those outcomes motivate the stronger value teacher, exact
complete-turn BFM, both-color gates, and source-bound clocks here; they are not
evidence that this compact family will pass.

Exactly one authenticated upload is permitted, and only after local
qualification, a clean implementation commit, a pushed neutral topic branch,
and green CI for that commit.  Editor copy-back must match the generated source
byte for byte before `Play My Code`; a Submit attempt is durably claimed before
the click.  An ambiguous response is resolved from submission history or API
identity and is never followed by a second click.  There is no rollback or
second upload.
