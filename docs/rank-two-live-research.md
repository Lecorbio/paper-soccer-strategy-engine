# Live-driven Paper Soccer research

The rank-two research campaign was stopped by its owner on 12 October 2026.
The [eight-unit extension outcome](live-eight-campaign-outcome.md) records its
terminal result and the separate unchanged-incumbent ranking retries. The
following workflow is historical and does not allocate another run.

The campaign collected ranked CodinGame games. Local
engines remained useful controls, while live opponents drove diagnostic questions.
Research submissions and promotion evidence have separate admission rules.

## Operating policy

The successor manifest is under `results/rank_two_20260927/live-research-v2`.
It links the original campaign without changing its policy, release or receipts.
The rank-three source `78e731a8…` remains the comparison and rollback source.

Incumbent research packets contain three completed calibrations. Candidate
packets use the fixed sequence incumbent, candidate, candidate, incumbent,
incumbent, candidate. Each attempt has a unique ID even when source bytes repeat.
Ranked opponents are assigned by the platform. Every completed window is archived
before the next submission; rank snapshots cannot be selected retrospectively.

New candidates need source-bound correctness, isolated timing safety and a
64-root pilot with no operational failures and an upper confidence bound at
least zero. Research admission does not require a positive local development
result. Promotion still requires 256-root development uplift of at least three
percentage points, opponent floor of minus five points, zero operational
failures and the private twelve-game diagnostic panel.

Formal qualification freezes one three-window block per exact source. At least
two windows must complete at rank one or two; all must be operationally clean.
Research windows never substitute for qualification. A failed block cannot be
restarted for unchanged source. Fresh confirmation then uses 800 roots and
3,200 games, with positive lower 95% root-cluster bound, the same opponent floor,
and no operational failures. Passing the live block alone is not campaign success.

## Implementation

`tools/rank_two_campaign_v2.py` defines successor policy and gates. Successor
bank, runtime and strength tools load that policy; historical producers remain
unchanged. `tools/rank_two_live_v2.py` implements the serial lifecycle. Its CLI
never clicks Submit: a fresh independent editor copyback and durable claim
precede one explicit browser submission flow. An ambiguous response permits
further observation, never a blind retry click. A new submission ID identifies
a new attempt even when CodinGame reuses the editor session handle.

A bounded UI resource lease holds the same lock as campaign compute queues.
Archive and analysis jobs run through the existing resource-limited queue using
`tools/rank_two_live_queue.py`. Interrupted acquisitions retain cached responses;
unknown game outcomes are not silently replayed. Source changes, duplicate game
IDs, incomplete accounting and unexpected window sizes fail visibly. An own
operational failure blocks further candidate admissions and requires restoration.

`tools/rank_two_live_research.py` reports opponent/version/color coverage,
trajectories, opening families and canonical positions. It selects at most 96
previously unanalysed positions, with a maximum of three per game, balancing
wins/losses, colors and phases while prioritizing strong opponents. Observed
moves are not assumed optimal. Research traces feed the existing resumable
exclusion census. Qualification trajectories use a separate outcome-free export.
New banks must account for completed live exposure at their frozen cutoff.

## Recovery and audit

Recovery checks occur every twenty minutes during live collection. After three
calibrations, a median duration below two hours warrants ten-minute checks.
Hourly checks apply only when no live work needs attention and a long local job
is healthy. A healthy live calibration gets one observation per wake while useful
independent work continues. There is no always-continuing model loop.

Tests cover source and attempt identity, duplicate claims, same-session repeat
submissions, incomplete/mixed windows, locks, stale leases, failure stops,
diagnostic selection, policy compatibility, qualification blocks and exclusion
cutoffs. A retained 90-game archive fixture exercises reporting and normalized
exposure export without running new games. Detailed run receipts remain local.
