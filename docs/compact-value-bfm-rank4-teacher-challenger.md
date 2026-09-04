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
terminal no-retry recovery-journal event routes to attempt one. Both routes
first invoke the recovery runner's read-only verifier against the exact
external plan namespace: a pass must rederive the complete journal, all four
stages, actual-clock result, finalist, and final reference; a rejection must be
the genuine last chained journal event and must rederive as terminal without
launching missing work. The verifier and journal head are copied into the
attempt ledger before the transition is committed.

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
requests and every later pilot/full training boundary. The pilot exclusion is
also active before the same attempt's full round.

Pilot admission requires an unseen-root, unprotected evaluation of a quantized
candidate, canonical retention, at least 10% mean teacher-regret reduction,
quantized action-flip rate no more than 0.005 above its scalar control, and at
least 105 wins with the exact zero-failure roster in a fresh 100-pair/200-game
Rank-4 screen. Full admission requires 500 pairs/1,000 fresh unprotected
actual-clock games, at least 550 wins, at least 265 wins as each color, a paired
lower 95% bound strictly above 50%, and the exact zero-failure roster.

The standard pilot screen is baseline-only: one 100-pair process and exactly
200 games. A search-throughput intervention pilot adds only the corresponding
baseline treatment, for two serial processes and exactly 400 games; the
treatment must pass the sealed paired activation/source/parity/timing evidence
or the baseline remains the control fallback. Search optimization is decided
in the full round: standard full A/B runs four 500-pair variants (4,000 games),
and an intervention full A/B runs four controls plus four treatments (8,000
games). The selected full variant is then frozen before a separately generated,
symmetry-disjoint 500-pair/1,000-game qualification bank is opened. Full A/B
results cannot serve as that qualifying gate.

Unchanged progress polls never count as failed attempts. Attribution/adaptation
is selected after exactly two consecutive *completed* trained attempts achieve
neither a 1.5 percentage-point Rank-4 gain nor a 10% teacher-regret reduction;
an admitted pilot is not a completed attempt. The Rank-4 gain is the terminal
candidate's win-rate change against the best comparable earlier completed
attempt; with no earlier comparable attempt, no incremental gain is claimed.
The next attempt must bind a
sealed attribution audit selecting one of the quantization, teacher-ranking,
or search-throughput interventions. The audit is derived from exactly the two
sealed unprotected outcome/metric closures; it cannot accept a caller's
classification. Quantization selects refined adaptive-scale QAT,
teacher-ranking selects exact hardest-5%-at-2,000,000-node labeling, and a
residual search-throughput gap selects state/evaluation caching before later
progressive-widening and subtree-reuse trials. The concrete intervention
contract is carried through the attempt, phase, pipeline, training, selection,
and admission evidence. The counter then restarts. Every attempt records its
hypothesis and intervention, attempt numbers have no hard cap, and the goal
persists until verified success.

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
An infrastructure interruption after a protected bank or shard claim has the
same candidate disposition: `record-protected-stage-abort` binds the
source-bound abort receipt, rejects the candidate without interpreting partial
metrics, retains every materialized bank's fingerprint-only exclusion, and
authorizes a clean protected-rejection attempt. The spent bank or shard is
never retried.

Dual qualification is not campaign completion. The release bridge first
appends a unique fixed-root `upload-authorization-claimed` event, publishes
capability/input files that bind that claim hash, and appends the exact
`upload-authorized` file binding. Only then can the ledger bind one
already-existing maintained
`submission-attested` receipt for the exact source,
then validates (without starting or polling it) the maintained live-window
reference and all of its evidence. Completion requires exactly 90 distinct
games, `complete-accepted-diagnostic`, and zero focus-bot operational failures.
Only then is `campaign-complete.json` sealed and the ledger becomes terminal.
A completion call, including an idempotent repeat after the terminal event,
reopens and recursively validates the original reference, receipt, collector,
90 monitor records, submission identity, and operational summary from the
exact recorded live-data root; copied summary fields alone are insufficient.
A rejected live window cannot open another attempt or accept another upload
until a separate sealed, explicit, single-use user authorization names the
next attempt and upload ordinal.

## Resource and resume policy

- At most eight single-thread generation/label workers may run concurrently.
- At most two training seeds may run concurrently.
- Uncontended timing uses one worker.
- Each strict final gate uses exactly four workers, and the two gates run
  sequentially.
- Generation, labeling, training, development gates, release timing, and both
  protected gates share one nonblocking campaign heavy-stage lock. Every gate
  start or abandonment additionally binds a clean process/load/nice audit.
- OpenMP, BLAS, MKL, NumExpr, and Accelerate thread counts are fixed to one per
  worker.
- There is no recurring automation assumption. A human or foreground agent
  explicitly resumes from the append-only ledger after a durable checkpoint.
- Expected post-tooling compute is approximately 13–21 hours for a pilot and
  30–45 hours for a full round, plus one to two days for development and final
  gates. These are iteration-cadence estimates, not completion guarantees.

The freeze command copies the allowlisted teacher, attempt-zero candidate,
recovery/exclusion evidence, governance and execution-tool source closure,
compiler binary, every named training manifest's adjacent NPZ, and every
artifact/dependency declared by the compact training-bundle manifest into a
self-contained content-addressed input tree. Source-worktree, Git, and compiler
paths remain provenance only; subsequent validation reads the frozen copies,
so later campaign attempts may evolve on new commits without rewriting the
original input freeze. Before release promotion, each pipeline phase requires
active HEAD and executable tool bytes to match the attempt's clean build
manifest exactly. Attempts opened after a prior dual-final authorization reuse
the frozen producer binaries and persist a narrower
`post-promotion-allowed-drift` closure: only the four declared deployment
outputs may differ, while every other source, bundle member, compiler, and
producer binary must still match the original manifest. This mode persists
through later same-contract attempts and permits no arbitrary source drift.

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

If infrastructure stops an unprotected development gate after its one-shot
claim or raw output exists but before a valid receipt, first seal
`abandon-gate-execution` in the teacher runner, then record it with
`record-development-gate-abort`. The transition copies the fixed candidate
identity and the gate plan's fingerprint-only development exclusion, does not
inspect partial metrics or count an improvement, and authorizes the next
leakage-isolated same-contract attempt. The spent bank is never rerun and an
aborted attempt cannot contribute loss roots.

After an unprotected trained attempt fails, `compact_value_bfm_loss_reuse.py`
may create the next roots override. It recursively validates the rejected
outcome, selected development gate, and phase game manifest, then extracts only
complete student-versus-Rank-4 losses from that unprotected manifest. Whole
root groups are canonically re-split and symmetry-deduplicated against frozen
train/validation, every earlier development exclusion, and fingerprint-only
protected/live exclusions. Its content-addressed TSV and replay-roots manifest
bind that derivation and never expose protected/live positions, transcripts,
metrics, or labels.
Before admission, the governor binds the reuse verifier to the attempt's
frozen source closure and snapshots both roots files. The archived manifest is
thereafter validated against the campaign-local TSV, so ledger resume neither
trusts a post-validation file swap nor depends on the disposable staging path.

Progress in a production campaign is derived from the pipeline game/position
receipts (`record-progress --pipeline-plan ...`); caller-supplied counters are
not accepted. Phase outcomes carry and revalidate the finalized label receipt,
training plan/selection, gate plan and raw gate results, selected source/runtime,
input audit, and the teacher-training admission. Ranking admission also requires
at least 100 comparable exhaustive validation groups and at least 80% of all
validation groups. Every retained successor-label group itself must be
exhaustive: nonexhaustive shallow groups are mandatory deep-label candidates,
and final packing rejects the phase if any legal complete-turn successor is
still missing. The phase evidence additionally binds the normalized
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
Attempt-zero recovery validation and live-window validation also bind the
verifier plus its complete clean source closure to the relevant attempt build
manifest. Current code must match that build when new evidence is recorded;
later ledger replay checks the archived verifier and manifest metadata without
depending on whatever commit happens to be checked out then. Live reference
and receipt bytes are snapshotted, copied from those snapshots, and recursively
revalidated after copying to close source-change races.

`authorize-dual-final` requires the sealed release evidence in production. Its
`--candidate-source` is the selected generated source used to prove lineage;
`--deployed-source` is the exact tracked seven-slot deployment derivative. Only
the latter becomes the protected-gate/upload candidate, while both identities
remain frozen in the authorization.
