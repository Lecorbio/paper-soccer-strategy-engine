# Rank-4 teacher challenger governance

The Rank-4 teacher challenger is a separate, value-only campaign. It preserves
the `6301-12-8-1` Compact Value-BFM architecture and never adds a policy head.
`tools/compact_value_bfm_rank4_teacher_challenger.py` is governance only: it
freezes inputs, creates deterministic schedules, records durable progress and
outcomes, and validates final evidence. It never trains, runs games, accesses a
network, uploads, replaces Rank 4, or creates recurring automations.

Attempt zero is the already-running discrete-v3 recovery. Its recovery plan is
copied into the frozen input bundle and every accepted result must bind that
exact sealed plan. No pilot may be materialized for attempt zero. Recording a
pass copies the finalist reference, referenced finalist, and referenced
recovery result into the new campaign before it routes to the final gates. A
terminal no-retry recovery-journal event routes to attempt one.

Each later attempt has a 2,000-game pilot:

- 500 student self-play games;
- 500 games with the student as each color against Rank 4;
- 125 games with the student as each color against the prior incumbent; and
- 125 games with the accepted teacher as each color against Rank 4.

The full schedule is exactly five times every pilot quota, for 10,000 games.
The generated TSV uses the maintained continuation parser's exact columns:
`game_ordinal`, `actor_mode`, and `base_seed`.
Every progress receipt carries the complete per-mode counters; their sum must
equal the total and a terminal receipt must equal every frozen quota exactly.
An outcome is accepted only through a sealed receipt binding the phase plan,
schedule, candidate, metrics, exact quotas, and the attempt's new development
fingerprint exclusion. Those exclusions accumulate into all later final-bank
requests.

Pilot admission requires an unseen-root, unprotected evaluation of a quantized
candidate, canonical retention, at least 10% mean teacher-regret reduction,
quantized action-flip rate no more than 0.005 above its scalar control, and at
least 105 wins with the exact zero-failure roster in a fresh 100-pair/200-game
Rank-4 screen. Full admission requires 500 pairs/1,000 fresh unprotected
actual-clock games, at least 550 wins, at least 265 wins as each color, a paired
lower 95% bound strictly above 50%, and the exact zero-failure roster.

Unchanged progress polls never count as failed attempts. Attribution/adaptation
is selected after exactly two consecutive *completed* trained attempts achieve
neither a 1.5 percentage-point Rank-4 gain nor a 10% teacher-regret reduction;
an admitted pilot is not a completed attempt. The next attempt must bind a
sealed attribution audit selecting one of the quantization, teacher-ranking,
or search-throughput interventions. The counter then restarts. Every attempt
records its hypothesis and intervention, attempt numbers have no hard cap, and
the goal persists until verified success.

Final qualification has a two-stage secrecy boundary. First,
`authorize-dual-final` freezes the admitted runtime, generated source/search
parameters, accumulated exclusions, thresholds, and zero-bank state. Only
after that receipt exists may two protected banks be materialized. Each bank
must have a distinct post-authorization entropy receipt and the same candidate
source binding; Gate B must explicitly exclude Gate A and their symmetry sets
must be disjoint. `prepare-dual-final` then freezes both banks before either
gate can run.

The gates run sequentially, use exactly four single-thread workers apiece, and
consume separately bound aggregate evidence. Each 500-pair/1,000-game gate
must achieve at least 527 total wins, 260 wins as each color, and zero failures
under the maintained strict timing verdict. Both gates bind the identical
candidate runtime and source. A protected failure is sealed and opens a fresh
leakage-isolated attempt; its results never become training inputs.

Dual qualification is not campaign completion. The ledger next binds one
already-existing maintained `submission-attested` receipt for the exact source,
then validates (without starting or polling it) the maintained live-window
reference and all of its evidence. Completion requires exactly 90 distinct
games, `complete-accepted-diagnostic`, and zero focus-bot operational failures.
Only then is `campaign-complete.json` sealed and the ledger becomes terminal.
A rejected live window cannot open another attempt or accept another upload
until a separate sealed, explicit, single-use user authorization names the
next attempt and upload ordinal.

## Resource and resume policy

- At most eight single-thread generation/label workers may run concurrently.
- At most two training seeds may run concurrently.
- Uncontended timing uses one worker.
- Each strict final gate uses exactly four workers, and the two gates run
  sequentially.
- OpenMP, BLAS, MKL, NumExpr, and Accelerate thread counts are fixed to one per
  worker.
- There is no recurring automation assumption. A human or foreground agent
  explicitly resumes from the append-only ledger after a durable checkpoint.

The freeze command copies the allowlisted teacher, attempt-zero candidate,
recovery/exclusion evidence, governance and execution-tool source closure,
compiler binary, every named training manifest's adjacent NPZ, and every
artifact/dependency declared by the compact training-bundle manifest into a
self-contained content-addressed input tree. Source-worktree, Git, and compiler
paths remain provenance only; subsequent validation reads the frozen copies,
so later campaign attempts may evolve on new commits without rewriting the
original input freeze. Such an attempt must supply its own clean build manifest;
each pipeline phase requires the active HEAD and executable tool bytes to match
that per-attempt manifest exactly.

Production freeze additionally requires the exact successor-trained
`0dbe2792…` float checkpoint, `e4d814…` prior diversity runtime, canonical
root TSV/manifest, and a sealed clean-checkout build manifest. Run
`build-manifest` only after every campaign source is tracked and the worktree is
clean. It freezes the complete C++/Python source closure, compiler identity,
and exact continuation/action-teacher/Rank-4-teacher/gate binaries. A dirty or
untracked task source makes production freeze fail closed.

Every `attempt-opened` event binds its student, prior runtime, initial float
checkpoint, roots, and build manifest. Pilot phase plans inherit those exact
artifacts; a full phase replaces only the student runtime with the admitted
pilot candidate. The pipeline derives files and binaries from the phase plan.
CLI path overrides are checks only and are rejected when they differ.

Progress in a production campaign is derived from the pipeline game/position
receipts (`record-progress --pipeline-plan ...`); caller-supplied counters are
not accepted. Phase outcomes carry and revalidate the finalized label receipt,
training plan/selection, gate plan and raw gate results, selected source/runtime,
input audit, and the teacher-training admission. Ranking admission also requires
at least 100 comparable exhaustive validation groups and at least 80% of all
validation groups. The phase evidence additionally binds the normalized
training build-source closure hash, so governance cannot accept results from a
training plan whose mutable tools no longer match its clean build.

Each protected bank is reduced before any game to a sealed fingerprint-only
artifact containing 500 canonical hashes and no transcript, label, or metric.
If either final gate fails, both already-materialized bank fingerprint sets are
bound into the next attempt and every later final authorization. A rejected
live window likewise requires a fingerprint-only live artifact before another
attempt can be authorized. Phase pipelines consume only these sanitized hashes,
never the protected bank/live payloads.
`sanitize-live-fingerprints` accepts only the exact sealed live-window reference,
its data root, and the frozen attempt/upload/source identity. It first reruns the
full live-window validator, then the trusted extractor reopens the verified
collector manifest and all 90 accepted records, checks their game IDs and source
identity, and replays each complete-turn boundary. There is no CLI/API input for
caller-supplied fingerprint lists. The resulting dynamic exclusion contains only
canonical hashes and a sealed reference to transcript-free extraction evidence;
that evidence binds the collector, receipt, exact game-ID roster, and source.
`record-live-window` and explicit additional-upload authorization re-extract from
the same archive and reject any mismatch before accepting the exclusion.

`authorize-dual-final` requires the sealed release evidence in production. Its
`--candidate-source` is the selected generated source used to prove lineage;
`--deployed-source` is the exact tracked seven-slot deployment derivative. Only
the latter becomes the protected-gate/upload candidate, while both identities
remain frozen in the authorization.
