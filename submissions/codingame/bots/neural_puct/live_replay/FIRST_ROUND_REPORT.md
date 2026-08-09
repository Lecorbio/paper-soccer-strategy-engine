# First live replay round: waiting for data

The first append-only round completed on 2026-08-09 with **34 independent,
rule-valid new games**, below the predeclared floor of 50. The result is the
protocol's completion path B: retain the maintained neural model, perform no
training or candidate matches, and make no performance submission.

## Active collection identity

Authenticated Chrome inspection found CodinGame history version 54 active.
The editor contained 99,697 characters with SHA-256
`4d77f9e20e756b2e11b0f4dc04ff9ed2fed37e85158f358dda104d8281b5b243`,
exactly matching the maintained local `submission.cpp`. It was therefore
reused rather than uploaded again.

| Identity | Value |
|---|---|
| Agent | `6604625` |
| Submission | `41113702` |
| Session | `77331573cfcf07bc257c0a715fe9675351097800` |
| UI/public snapshot | rank 23 of 208, score 35.66 |
| Model SHA-256 | `d7e5255fd2c3f7a203796829c21c8a7580b384de13942e466f4069081e2126b2` |
| Collection upload | none; identical active build reused |

The full binding is in
`active_submissions/20260809T191950Z-agent-6604625.json`.

## Boundary and acquisition audit

The initial exclusion registry
`531342acdafe02b5c21650ff4e82c4e48e2262dbdc6270c7cfce22aab208148d`
contained 2,867 known IDs, of which 2,653 were protected. It existed before
the first replay-detail request.

- The current top-five windows contained exactly 208 unique games; all 208
  were already in the initial registry. This independently reproduced the
  starting audit fact.
- The top-20 windows contained 665 unique games: 625 already known and 40
  unseen. The active agent window contained 90 games; its intersection with
  current top-20 opponents was those 40 unseen candidates.
- All 40 candidate details were requested once. Thirty-four passed exact
  identity, rank/result, transcript, legality, rebound, terminal, and winner
  checks. Six were rejected because their final played frame had an empty
  action; they are incomplete/invalid evidence.
- Eleven snapshots ran from `2026-08-09T19:20:48Z` through
  `2026-08-09T19:31:53Z`. Follow-up snapshots made no additional replay-detail
  requests and found no new eligible games.
- The archive has zero response conflicts and zero request failures. All 282
  public requests completed on their first attempt: 11 leaderboard snapshots,
  231 conservative battle-metadata reads, and 40 replay-detail reads.
- The final boundary registry
  `f20baf50fb4a0d3574027e3e8c2731542082149fe3c3dfec89b83734cf9f71d8`
  contains 2,907 known IDs and now binds the 34 acceptances plus six permanent
  structural rejections.

The historical polls numbered 1--10 report six `detail_request_count` entries
because the executed collector counted cached validation attempts in that
field. The content-addressed receipts are authoritative and show only the 40
initial network detail requests. The maintained collector corrects this
accounting field and has a regression test proving cached rejections cause zero
network reads.

## Reproducibility checks

The maintained collector SHA-256 is
`5c73139e9607ea97477aef212b6c015207cf66fb462fda8759003f98a4e897ef`.
Its offline store audit verified 34 accepted records, 34 discovery bindings,
62 raw response objects, 62 normalized objects, and 282 receipts. The pinned
research environment passed all 29 CodinGame Python tests; the minimal system
Python passed the collector and promotion suite with the four NumPy-dependent
trainer tests skipped. A complete release build also passed, followed by the
maintained neural submission, timing, generated-source, and protocol checks.
Python/C++ feature parity was exact for all 5,144 float32 values.

Sanitizers and candidate arena gates were deliberately not run: there is no
candidate source or model after the independent-game sufficiency gate failed.

## Available supervision, not yet a training set

The 34 whole games contain 1,514 CodinGame turns and 4,738 primitive edges.
They expose 2,659 primitive actions from frozen current-rank 10--20 opponents.
The neural agent's 2,079 primitive actions are explicitly marked
`self-relabel-only`; none are expert labels. The direct expert distribution is
one game from rank 10 and 33 games from ranks 11--20. The active neural agent
played 20 games as player zero and 14 as player one.

No train/validation/test split, reflection, policy label, value target, or
sample weighting was constructed. When the independent-game floor is reached,
whole games and their reflections must remain together and the existing
trainer must purge canonical-state overlap before training. Final outcomes
(15 neural wins, 19 neural losses) are recorded only as weak diagnostics.

## Decision

- New independent valid games: **34 / 50 required**.
- Strong-player labels: deferred; insufficient independent games.
- Deeper neural-only PUCT relabeling: not run.
- Shared action-conditioned policy head: not implemented or trained in this
  round, because the data gate failed first.
- Deterministic three-seed training: not run.
- Default, fixed-work, equal-clock, prospective, or sealed candidate gates:
  not run.
- Performance submission: **none**.

The maintained seed-20260809 model and its mover-relative, neural-only decision
path remain unchanged.
