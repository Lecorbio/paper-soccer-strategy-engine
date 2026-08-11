# Full-turn BFM improvement plan

This is the development plan for the Jacek-style complete-turn generator and
best-first minimax bot. It is a research track beside canonical rank 4, not a
replacement for it until the clock-led evidence says so.

## Measurement rule

CodinGame strength is measured at the contest clocks: 800 ms for the bot's
first decision and 165 ms for each later decision. Fixed node or work budgets
are useful for determinism, profiling, and explaining why two searches differ;
they are not a strength comparison. A fixed-work win may advance a debugging
hypothesis, but only an actual-clock screen or source-bound live batch may
advance a playing-strength hypothesis.

Every locally decisive screen must:

- pass explicit 800/165 ms clocks to both candidate and rank-4 control;
- retain the production 3,000,000 work and tree ceilings as safety stops;
- report first/later response maxima and all operational counters;
- use both player colors and mixed even/odd opening depths;
- use an independent deterministic seed batch for confirmation.

## Current diagnosis

Rank 4 already treats a rebound chain as one search ply: its primitive-edge
recursion decrements depth only when possession changes. The experimental bot
therefore did not gain a new game horizon merely by materializing complete
turns. It exchanged iterative-deepening alpha-beta, transposition-table
cutoffs, and strong move ordering for a wide explicit tree whose value depends
more heavily on the inherited evaluator.

The current evidence routes the main weakness away from legal generation:

- sampled live decisions retained the recorded action at every inspected root;
- rank 4's chosen boundary was almost always retained as well;
- the candidate record worsened as both candidate and rank-4 budgets rose in
  the frozen fixed-work screens;
- root-only evaluation was worse than deeper BFM, so the tree is helping even
  though the complete policy remains weaker than rank 4;
- an isolated 165-to-100 ms later-clock change altered 653 of 1,586 later
  decisions (41.2 percent), despite identical first-decision settings;
- lowering only UCT exploration from `1.5` to `0.5` improved two 24-game clock
  screen totals from 5-19 to 11-13 and from 3-21 to 6-18, but their genuinely
  independent seeded subsets improved once and were flat once;
- narrowing every deeper node to 64 actions lost badly in its first clock
  screen;
- retained-boundary deduplication improved throughput without changing search
  decisions, but only by about 1.28 percent.

The leading hypotheses are therefore evaluator ordering, BFM allocation and
backup policy, and excessive construction cost. Generator omission remains a
measured possibility, not the default explanation.

## Experiment ladder

### 1. Source-bound live baseline and decision forensics

Archive one complete 90-game CodinGame batch for the exact uploaded hash. For
each clean game, compare the observed action with:

- the deployed replay-book correction path;
- raw BFM at 800/165 ms;
- canonical rank 4 at 800/165 ms;
- the deterministic retained root set;
- the best one-ply evaluator action and the evaluator rank of the observed,
  BFM, and rank-4 actions.

Classify the first meaningful divergence as one of:

1. action/boundary omitted by the capped generator;
2. retained but ranked below the one-ply evaluator's initial choice;
3. initially preferred, then displaced by deeper BFM;
4. rank 4 and BFM agree while the observed action or result differs;
5. replay-book correction changed the deployed decision;
6. operational or provenance failure.

Aggregate clean results by color, first/later decision, win/loss, opponent,
and frozen opponent-rank tier. A game batch used for diagnosis is development
data and cannot later serve as untouched promotion evidence.

### 2. Search-policy isolation with the evaluator frozen

Change one policy variable at a time. The first candidates are:

- final action chosen by backed minimax value only (the maintained exploratory
  default) versus a calibrated, small visit term;
- selection exploration and first-play urgency scaled to the actual evaluator
  range rather than a nominal normalized range;
- robust root choice that requires a minimum visit count before a shallow
  heuristic can displace the current principal action;
- resumable widening at non-root nodes, only after the generator can resume
  without making an incomplete action set look exhaustive.

The first screen is 24 paired actual-clock games on a development seed batch.
An apparent improvement must repeat on an independent seed batch. Reject any
variant with a candidate timeout, invalid action, empty output, or clear color
collapse. Do not select parameters by fixed-node wins.

The isolated `C=0.5` experiment is provisionally the next exploratory default.
The full-screen aggregate was 17-31, but the independent seeded aggregate was
only 10-18 versus 7-21 for the control. It remains below rank-4 parity and is a
live-diagnostic candidate rather than a promotion candidate. Do not bundle
another search or evaluator change into its first source-bound arena test.

### 3. Behavior-preserving construction optimization

Before changing search semantics, target complete children evaluated per
millisecond while preserving retained actions, ordering, chosen moves, and
fixed-work statistics. The highest-value refactors are:

- store partial actions as parent links in an arena instead of deep-copying
  vector-backed positions and move vectors;
- reuse topology, symmetry maps, and tactical scratch buffers across node
  expansions;
- maintain canonical hashes incrementally instead of rescanning the full
  board for every partial path;
- avoid classifying the same retained boundary more than once;
- make generation resumable so a deadline does not pay for and then discard a
  whole atomic action batch.

Require a byte-for-byte fixed-work decision comparison on curated states,
focused sanitizer coverage, no peak-memory regression, and a material
throughput improvement before combining a refactor with policy tuning. The
first meaningful target is at least 2x complete child evaluations per second;
single-digit improvements are useful but do not explain the current strength
gap.

### 4. Evaluator work only when replay evidence calls for it

If retained actions are consistently present but the one-ply rank ordering is
wrong, train a boundary-state value or action-ranking model specifically for
the BFM policy. Use deeper rank-4 labels and later public development games;
never use the bot's own move as a strong label without relabeling it. Split by
whole game, keep symmetric states in the same split, and purge canonical-state
overlap. Select models by top-k action coverage and calibration as well as
loss. Do not revive a previously rejected model solely because it has more
parameters.

### 5. Hybrid or pivot criterion

Keep the complete-turn BFM as a research implementation, but do not make it the
only promotion path. If optimized and calibrated BFM cannot approach parity
with rank 4 under actual clocks, reuse the strongest Jacek components as a
tactical/root ordering layer for rank 4's alpha-beta search. The safe control
chooser remains canonical rank 4 until a later candidate passes both fresh
local clock screens and an independent source-bound live repeat.

## Exploratory live cadence

An exploratory upload does not need a local superiority claim. It does need:

- current generated source below 100,000 characters;
- an exact source hash copied back from the authenticated editor before the
  arena action;
- clean compilation, legality, protocol, sanitizer, and timing checks;
- construction-inclusive local maxima below 900/180 ms;
- one named hypothesis and no bundled unrelated policy changes.

Wait for the full arena batch before the next upload. Separate opponent
forfeits from clean games, preserve both colors, and treat rank and score as
descriptive. After two consecutive non-improving variants, restore canonical
rank 4 and change hypothesis family. If testing pauses, leave canonical rank 4
active rather than an unconfirmed experiment.

## Data boundary

Public replay collection starts from a caller-approved, frozen ID-only
exclusion registry. The collector must skip excluded IDs before requesting
details, preserve raw and normalized responses append-only, and record the
source/submission assertion. Never inspect `matches.json`, sealed banks,
prospective banks, or final-bank actions for this development loop.
