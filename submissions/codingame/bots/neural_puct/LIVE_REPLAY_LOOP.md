# Live replay improvement loop

This protocol turns normal CodinGame arena submissions into append-only neural
training data without weakening the existing evaluation boundaries. It is an
online imitation/DAgger loop, not permission to train on promotion banks or to
copy every move made by the current bot.

## Audited starting point

As of 2026-08-09, the public battle windows for the current top five contain
208 unique game IDs. Every one of those IDs is already represented somewhere
in the local evidence archive. The public API exposes a recent window and does
not provide useful backward pagination, so re-downloading that window adds no
new games. New capacity must come from future arena games, newly active agent
versions, or a deliberately broader strong-player set such as the top 20.

The maintained neural baseline is a research/data-collection bot, not a
promotion candidate. Bind every collection run to the exact generated
submission, model, agent, and CodinGame submission IDs.

## Hard data boundaries

- Treat all promotion validation and test banks, sealed-final aliases,
  rank-one locked games, and arena loss diagnostics as evaluation-only. Never
  read their actions into a training corpus.
- Keep the exposed development bank out of training. It remains an adaptive
  gate and diagnostic set.
- Build a game-ID exclusion set before fetching replay details. Query battle
  metadata first and request details only for IDs not already known or locked.
- Store raw responses append-only, hash them, and never silently replace a
  snapshot. Preserve player name, agent ID, submission/session ID, game ID,
  fetch time, leaderboard rank/score snapshot, and structural rejection
  reason.
- Split by whole trajectory before producing primitive states. Reflections and
  every state from the same game stay in one split. Purge canonical feature
  overlap from validation and test using the existing trainer rules.
- Do not touch the user-owned root `matches.json`.

## Collection order

1. Implement and test the collector before making a CodinGame submission.
2. Snapshot the current leaderboard and active agent IDs. Start with the top
   five and expand to the top 20 when the top-five window yields too little
   genuinely new data.
3. Submit at most one explicitly labelled collection build. Record its exact
   source and model SHA-256 plus its CodinGame agent/submission identity.
4. Poll the new agent's battles conservatively, cache every new game ID, back
   off on rate limits, and stop after the bounded collection window. Normal
   submissions may create only a handful of games against each elite player;
   do not spam submissions to manufacture data.
5. Replay every accepted transcript through the local rules engine and reject
   incomplete, illegal, duplicated, locked, or structurally inconsistent
   games deterministically.

## Labels and weighting

- A strong opponent's played primitive edges are direct policy supervision,
  including moves from games the opponent eventually lost. Weight them by a
  frozen strength tier and normalize per player and per game so a large public
  window cannot dominate the corpus.
- The neural bot's own played action is not an expert label. Re-evaluate its
  positions offline with a deeper neural-only search and train from the root
  visit distribution. Prioritize positions where the played action had low
  teacher probability, the policy was uncertain, or the elite response
  exposed a tactical error.
- Keep final-game winner targets as a weak diagnostic. Previous experiments
  showed that improving outcome BCE did not improve play. Prefer validated
  search or temporal-difference value targets, and introduce them only behind
  a predeclared held-out criterion.
- Begin with roughly 70--80% anchored public-expert mass and 20--30% new
  on-policy/relabelled mass. Train at least three deterministic seeds, report
  source-separated metrics, quantize each candidate, and choose without
  consulting the test split.

## First representation experiment

The highest-priority model change is a shared action-conditioned policy head.
For each legal direction, derive a small canonical feature vector describing
the consequence of that edge: rebound versus handoff, immediate win/loss,
change in both goal distances, remaining degree, safe/dead handoff frontier,
continuation-component size, layer closure, escape routes, and opponent
mobility. Score all directions with the same small network so parameter cost
does not grow eightfold. Preserve the mover-relative value head and the
neural-only decision path.

The generated submission currently has little textual headroom. Apply only
verified behavior-neutral source compaction, or replace weaker inputs, before
adding the new head. Require exact Python/C++ feature parity, rotation and
reflection tests, int4 golden tests, and a source size below 100,000
characters.

## Gates and submission policy

Freeze thresholds before observing candidate match outcomes. At minimum:

1. All focused unit, sanitizer, generation-current, feature-parity, legality,
   node-cap, and construction-inclusive timing checks pass.
2. The six paired default openings score at least 7-5 at 2,000 neural
   simulations versus 5,000 rank-5 nodes.
3. Only then run the 48-opening exposed development bank in both colors. A
   candidate intended for deployment should reach at least 48-48 at fixed
   work, with no material color collapse.
4. Only then run equal-clock development. Require a clear improvement over the
   maintained 39-57 baseline and no operational failures before calling the
   model worthy of a performance submission.
5. Never inspect or run a sealed prospective/final bank for an exploratory
   candidate. Freeze a new candidate-independent ladder if a genuine
   promotion decision needs one.

The first arena upload may be an explicitly documented collection submission.
A second upload is allowed only for a candidate that clears the predeclared
deployment gates. If no candidate clears them, retain the maintained model,
publish the negative result and corpus provenance, and do not submit it as an
improvement.
