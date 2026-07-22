# Challenger experiment record

## Immutable reference

The reference is `../rank_5/`: 93,005 characters, SHA-256
`f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`.
Its completed version-26 batch was agent `6561779`, submission `41015554`,
rank 5 of 206, score `42.42773147296124`, with 57 wins and 33 losses. The
folder and generated artifact remain byte-identical.

The exact source was restored as history version 31, agent `6566182`, to obtain
a fresh public control corpus after candidate v1. Its current-distribution
batch finished rank 6 of 206 with score `41.741245253608554`, 55 wins and 35
losses: player 0 went 24-15 and player 1 went 31-20. Against the five leaders
it went 0-4 versus jacek, 1-6 versus Marchete, 1-7 versus Deltaspace, 2-9
versus Laars, and 0-3 versus Snekkers. This variance does not replace the
stronger immutable version-26 verification, but it is the current live control
and proves that raw batch win totals are not comparable across schedules.

## Candidate v1: exact rebound-component frontier proof

- Maintained change: exact current-turn rebound reachability and closed-
  component loss proofs at the search root and depth-zero boundaries.
- Replay book/model/search clocks: unchanged from `rank_5`.
- Paste-ready source: 96,306 characters.
- SHA-256: `e58fce3bf1314f480f76160471f7f7140fedfff39f71ffbe45c04c4ba4f0595f`.
- Arena result: **rejected**. History version 30, agent `6566176`, completed
  90 games at rank 5 of 206 with score `42.24210800267009`, 61 wins, and 29
  losses. That is four more raw wins but a lower opponent-weighted score than
  both the immutable version-26 reference (`42.42773147296124`) and the
  currently downloadable version-29 control (`42.32812937013547`).

The proof walks only unused edges and permits an intermediate vertex only when
it was already visited or is a rebound boundary point. It never traverses a
goal. Reaching the mover's attacking goal proves a winning complete action; an
own goal is not success. With CodinGame's `MoverLoses` blocked rule, a finite
component with no attacking goal and no unvisited landing that leaves the next
player a legal edge proves the mover loses during the current turn. Unknown
components fall back to the unchanged evaluator.

### Arena-loss evidence

All 33 archived version-26 losses were recovered from the preserved session:
31 ended in opponent goals and two in candidate self-blocks. There was no
candidate root where an immediate rebound goal was missed. There were 32
opponent exact-goal roots after candidate handoffs, so the intended benefit is
the depth-zero defensive proof, not an opening override.

Representative recurrent novel states are:

- `0/6/5/4/5/53/61/0633`: Deltaspace leaves the ball at `(2,10)` with a latent
  player-1 south path `33`; the old bot chose `5024701`.
- `0/6/5/4/5/53/61/431`: the latent player-1 south path is `43`; the old bot
  chose `7`.
- `1/1/7/6/0/75/74/3/003`: the latent player-0 north path is `01`; the old bot
  chose `54`.

These pairs preserve similar ball progress and shortest physical goal distance
while changing occupied-edge topology, which is precisely what the exact
component scan observes. Exhaustive enumeration at four representative final
candidate turns found every action already losing; useful intervention must
occur at an earlier frontier, before the forced corridor is entered.

The jacek `0/0` family remains intentionally free of a broad override. Earlier
direct imitation of action `3` regressed. The prior player-1 correction `67`
survives longer than `1`, but version 26 still loses after jacek's stronger
later `5330` reply; normal tactical search remains the fallback at that novel
state.

### Local correctness and strength

Candidate-specific artifact tests cover both protocol colors, exact replay
matching, safe fallback, complete mandatory rebounds, north/south goal parity,
visited versus fresh center-mouth points, boundary-post continuation, used
edges, own goals, and closed rebound components. All 11 tests pass.

The true two-engine comparison gate uses fixed node counts and color-swapped
starting states. It is not the historical `local_gate.cpp`, which compared one
engine against itself. Completed results so far:

- Smoke, 9 openings / 18 games at 2,000 nodes: challenger 11-7, zero unfinished,
  with 6/5 wins by candidate color.
- Development bank, 37 openings / 74 games at 30,000 nodes: challenger 40-34,
  zero unfinished, with 25/15 wins by candidate color.

At 100,000 fixed nodes, shared recurrent positions usually retain the rank-5
move, proving the feature is selective. Exact goal leaves often avoid neural
evaluation: candidate throughput improved materially in the Deltaspace
player-1 and late jacek positions. At the actual 130 ms budget, the challenger
searched 162,016 versus 129,856 nodes and completed depth 6 versus 5 at
`1/1/7/6/0/75/74/3/003`; it searched 114,624 versus 88,784 nodes at the later
jacek `5330` state. Some quiet openings were roughly even or slightly slower.

Release timing remains within the contest limits at about 650.1 ms first and
130.1 ms later for both roles. Generator freshness and the complete repository
suite must be rerun immediately before any arena submission.

The completed arena batch confirmed that the proof is selective but not yet a
rank improvement. The strongest-opponent split was 2-3 against jacek, 2-4
against Marchete, 0-5 against Deltaspace, 3-4 against Laars, and 4-3 against
Snekkers. The candidate beat more of the field overall, but the Deltaspace
sweep dominated CodinGame's opponent-weighted score. Candidate v1 is therefore
evidence for keeping the exact primitive, not evidence for accepting this
search placement unchanged.

## Candidate v2: selective goal-zone tactical extension

- Maintained change over v1: when a depth-zero boundary is in the mover's
  attacking goal zone and an attacking goal is at most one fresh landing away,
  search exactly one complete tactical turn before static evaluation.
- The extension is non-recursive; its children retain the v1 exact rebound
  win/loss proof and then use the unchanged evaluator.
- Replay book/model/search clocks: unchanged from `rank_5`.
- Paste-ready source: 97,379 characters.
- SHA-256: `b8e1d45aa8d222ec9dc23e52f31d971722a05eaaaf9941f8c7ad2c135397bde7`.
- Arena result: **rejected before submission**. An independent correctness
  audit found that tactical-mode continuation entries shared transposition
  keys with normal search even though their leaf-extension semantics differ.
  That could let a later iterative-deepening pass reuse an exact/bound entry
  computed with the second extension suppressed.

The physical goal-zone guard (`y <= 3` north, `y >= height - 1` south) is
important. An earlier unrestricted one-fresh-landing extension consumed about
25-35% of node throughput and changed the later jacek action at the real clock;
it was rejected before submission. The guarded version retained the rank-5
action at all recurrent elite probes except where the deeper completed result
changed evaluation, and it completed depth 6 at the Deltaspace player-1 threat
with 173,280 nodes versus rank-5's depth 5 and 128,192 nodes in one 130 ms run.

Before that audit, candidate v2 passed 12 artifact tests. Its fixed-node gates
were 11-7 over 18 smoke games and 46-28 over 74 development games, but the
larger result is not acceptance evidence because it was measured with the
transposition-mode alias.

## Candidate v3: sound tactical mode plus proven jacek branches

- Maintained change over v2: tactical continuations XOR a dedicated mode
  component into every transposition/evaluation boundary key.
- Narrow replay additions: candidate-v1 wins `896549711` and `896549482`
  supply exact actions `7` after `0/0/1/6` and `2722` after the recurrent
  player-1 `.../5330` branch. Both beat rank-leading jacek; neither changes a
  broad opening or activates if the opponent deviates.
- Replay book: 26 paths / 564 decisions. Evaluator and 650/130 ms clocks remain
  unchanged from `rank_5`.
- Paste-ready source: 98,111 characters.
- SHA-256: `e559ca976a9c4dd990a97dc7b824b1aa08525a840aaf9d7f0070f90b437184ba`.
- Arena result: **rejected**. History version 32, agent `6566218`, submission
  `41025711`, completed rank 9 of 206 with score `38.99557898194705`, 50
  wins and 40 losses. Player 0 went 27-21 and player 1 went 23-19.

The TT fix has a deterministic two-ended regression: TT-off and TT-on both
produce action `0`, score `1731`, completed depth 2, and exactly two tactical
extensions, while the cached search records no cross-mode continuation hit.
The artifact suite is 14/14 and covers north/south activation, an outside-zone
negative, complete-action replay, the two winning jacek branches, and every
eligible decision in the complete replay book.

On the exact replay-augmented candidate, the fixed-node gates are:

- Smoke, 9 openings / 18 games at 2,000 nodes: challenger 11-7, zero unfinished,
  with 6/5 wins by candidate color.
- Development bank, 37 openings / 74 games at 30,000 nodes: challenger 43-31,
  zero unfinished, with 21/22 wins by candidate color. This remains above v1's
  40-34 result on the identical bank.

Release timing remains about 650.1 ms first and 130.1 ms later for both roles.
The full repository suite, timing, and generated-source hash check remain
mandatory immediately before submission.

The completed batch was dominated by two repeatable horizon regressions. It
went 0-8 against EricSMSO and 2-6 against derjack, as well as 0-5 against
Laars, 0-4 against Marchete, 1-3 against Deltaspace, 2-3 against Snekkers, and
0-2 against jacek. Six derjack losses chose `71` after exact prefix `6/7/5`;
the maintained source replay chose `61` only from turn 19 even though `61`
won all eight matching v1/control games. Four EricSMSO losses chose `344`
after exact prefix `0/0/0/6/6/5/2/5/0527271`; the stored `614347` response
was similarly ineligible until two turns later. These failures justify
earlier eligibility of the existing exact winning paths, not removal of the
sound rebound proof.

## Candidate v4: restore stable exact horizon responses

- Maintained change over v3: the existing derjack and EricSMSO winning paths
  become eligible at their first evidenced divergence, turns 3 and 9.
- The derjack change targets six v3 losses and one v3 win while aligning with
  eight of eight matching wins from v1/control. The EricSMSO change targets
  four v3 losses; its source action went 3-2 in matching control games, while
  v3's replaced action went 0-4. One sibling control win is collateral. A new
  player-0 EricSMSO path changes two further v3 losses at turn 14 and aligns
  with six matching control/v1 wins without known winning-game collateral.
- Replay book: 27 paths / 589 decisions. Paste-ready source: 98,298
  characters, SHA-256
  `cbaeca6748c3b4668d17f89dfbad8ff96826a78894abbd31788cb2b0c27ddb76`.
- The complete 15-test artifact suite passes. Exact screening changes 12 v3
  losses and one v3 win. The 74-game development gate remains 43-31 with no
  unfinished games and a larger 306-game color-swapped gate finished 169-137.
- Arena result: **rejected**. History version 33, agent `6566237`, submission
  `41025850`, completed rank 7 of 206 with score `41.38247809221063`, 58 wins,
  and 32 losses. Player 0 went 32-20 and player 1 went 26-12. This improves
  v3 but remains below both the immutable reference score and the rank-4 goal.
  It went 0-8 against jacek, 3-6 against Marchete, 3-3 against Deltaspace,
  0-2 against Laars, 0-2 against Snekkers, 2-3 against EricSMSO, and 4-2
  against derjack.

## Candidate v5: cross-response transposition reuse

- Experiment: retain transposition entries between responses, permitting
  cross-generation cutoffs only for exact non-mate values. Old bounds and
  root-ply-relative mate values remained move-order hints. Generation-first
  replacement prevented stale entries from freezing a bucket.
- A focused regression proved that old exact non-mates were reused while old
  bounds and mates could not cut off a new response.
- Result: **rejected before submission**. The feature was heavily exercised
  (hundreds of old-generation cutoffs per paired game) but the 74-game,
  30,000-node gate fell from 43-31 to 35-39. The exact v4 source was restored
  byte-for-byte after the experiment.

## Candidate v6: current-leader replay repairs

- The candidate-v1 player-0 jacek win becomes eligible at turn 2, selecting
  `1` after `0/0` instead of v4's repeated `3`/`0`. A public player-1 win
  selects `71` after `0/0/3/67/27/45/5` instead of `2`.
- Two current-Snekkers wins select `6672` after
  `7/6/7/53/10/34/71` and `2` after the exact player-0 alternate branch
  ending `.../31/4167`.
- Exact screening changes 10 v4 elite losses and three lower-field v4 wins.
  Across 110 retrievable v1/control/v3/v4 games against the five leaders, the
  four paths change 23 losses and one win. Across a separate pool of 79 public
  wins they change six. The clean player-0 Snekkers path changes no known win;
  the other three are explicit opponent-weighted bets.
- Replay book: 28 paths / 594 decisions. Paste-ready source: 98,356
  characters, SHA-256
  `aac15134769111d60f58a13a581c5c3c3ad1f7d1163fe3851ec40174489de1dc`.
- The complete 15-test artifact suite passes. The 74-game development gate is
  unchanged at 43-31 with no unfinished games. The smoke gate is 10-8 versus
  v4's 11-7; isolating the paths attributes that one-game regression to the
  broad player-0 jacek correction.
- Arena result: **rejected**. History version 34, agent `6566259`, submission
  `41026035`, completed rank 9 of 206 with score `38.677486684679195`, 50 wins,
  and 40 losses. Player 0 went 26-26 and player 1 went 24-14. It went 0-4
  against jacek, 3-3 against Snekkers, 0-2 against Marchete, 1-3 against
  Deltaspace, 2-3 against Laars, 1-5 against EricSMSO, and 0-6 against
  derjack. The broad player-0 jacek path activated four times but split into
  two lower-field wins and two jacek losses; its stored suffix lost all four
  matching jacek games. The new player-1 jacek response activated once and
  lost. Neither Snekkers addition activated. These live outcomes disprove the
  three risky replacement bets and leave only the collision-free player-0
  Snekkers suffix worth retaining.

## Candidate v7: principal variation search plus audited exact repairs

- General search change: boundary-level principal variation search gives each
  non-first, non-root complete action a null-window probe, followed by a full
  re-search only when its fail-soft result improves strictly inside the
  original window. Root action capture remains full-window.
- A 64-case regression crosses eight legal states, depth 2/3, TT disabled or
  enabled, and PVS disabled or enabled. Every run must complete the requested
  depth with identical score and complete action; the suite also requires at
  least one actual PVS re-search.
- Replay cleanup restores the prior public-winning player-1 jacek action `2`
  and Snekkers action `45`. The player-0 jacek replay is ineligible until the
  exact `0/0/1/6` suffix and then selects `43` from public win `896344362`, so
  normal search still owns the broad `0/0` decision.
- The exact derjack player-1 action after
  `6/7/5/61/44/53/0` changes from `0617227` to `0617271`. Across recovered
  games the old action was 5-5 and lost all three matching v6 games; the new
  action was 10-0, including three candidate-v1 and three control wins. The
  replay is truncated immediately after the action because the full source
  transcript collides with wins at two later turns.
- A proposed player-0 derjack replacement at turn 26 was rejected: it targets
  seven losses but overwrites four known wins and has only one public source
  win.
- Replay book: 28 paths / 586 decisions. Paste-ready source: 99,263
  characters, SHA-256
  `d1fad603b06fff8578aaf0b04836309e457c46a218e9b308fc466f35bb7ecdad`.
  The artifact suite is 17/17. Fixed-node gates are 12-6 at 10,000 nodes and
  42-32 at 30,000 nodes, both with zero unfinished games. The larger 306-game
  color-swapped gate is 166-140 (105/61 wins by candidate color), also with no
  unfinished games.
- Arena result: **rejected**. History version 35, agent `6566278`, submission
  `41026144`, settled rank 8 of 206 with score `40.30299402098817`, 53 wins,
  and 37 losses. Player 0 went 24-27 while player 1 went 29-10. It went 0-5
  against jacek, 1-2 against Marchete, 1-7 against Deltaspace, 0-3 against
  Laars, 0-1 against Snekkers, 1-5 against EricSMSO, and 2-6 against derjack.
  Three repeated player-0 EricSMSO losses chose `5` after `0/5`, where v4's
  full-window search chose `21`; the change was search-only. The derjack
  `0617271` replay activated in player-1 losses but normal search then chose
  `1` after the next exact opponent reply, while a recovered public win chose
  `42`. PVS therefore improved deterministic node gates but materially harmed
  the real-clock player-0 distribution and is not enabled in the next build.

## Candidate v8: restore full windows and finish exact winning suffixes

- The rejected boundary PVS and TT-retention experiment is removed, restoring
  v4's full-window real-clock search and transposition replacement exactly.
- Before replay additions, the 30,000-node development gate returns to 43-31
  with balanced 21/22 color wins; the 10,000-node smoke gate is 9-9. Further
  exact branches require collision screening before arena use.
- The derjack player-1 replay extends only through action `42` after
  `6/7/5/61/44/53/0/0617271/3`. That action is 10-0 in recovered exact-state
  games, while v7's search action `1` is 0-2. It is truncated there: the full
  source suffix collides with three wins at turn 19 and another three at turn
  27.
- Widening the player-0 jacek source to turn 2 was rejected after a 540-game
  screen. It would change 13 losses but collide with ten wins immediately and
  a further v1 jacek win at its `43` suffix. The replay remains selective from
  turn 4.
- Replay book: 28 paths / 587 decisions. Paste-ready source: 98,577
  characters, SHA-256
  `788593dd435d8f864cb07fd5d81b63d2f8c76a9cc9ee975f6b037ed69ff34875`.
  The artifact suite is 15/15; fixed-node gates are 9-9 at 10,000 nodes and
  43-31 at 30,000 nodes. The 306-game gate is 169-137 (95/74 wins by
  candidate color), with no unfinished games.
- Arena result: **rejected**. History version 36, agent `6566346`, submission
  `41026268`, settled rank 10 of 206 with score `38.49433715163945`, 47 wins,
  and 43 losses. Player 0 went 23-22 and player 1 went 24-21. It went 1-5
  against jacek, 0-6 against Marchete, 1-4 against Deltaspace, 1-1 against
  Laars, 1-4 against Snekkers, 1-5 against EricSMSO, and 1-2 against derjack.
  Restoring full windows did not reproduce v4's schedule result and remained
  materially below the immutable reference.

## Candidate v9: return to the proof-only live foundation

- Removes the selective goal-zone tactical leaf extension and its dedicated
  transposition mode. The resulting search/evaluation path returns to
  candidate v1's live-tested foundation, which scored `42.24210800267009`
  with 61-29, while retaining the later audited exact replay repairs.
- Replay book: 28 paths / 587 decisions. Paste-ready source: 97,611
  characters, SHA-256
  `b3d5659abc686dc31fe04e337d940d12e55540d7fe48a7a83a4b280e530b1ff7`.
  The artifact suite is 13/13; fixed-node gates are 11-7 at 10,000 nodes and
  40-34 at 30,000 nodes. This deliberately accepts v1's weaker deterministic
  gate because its completed live result was stronger than v3-v8. The
  306-game gate is 178-128 (107/71 wins by candidate color), the strongest
  large local result in this series, with no unfinished games.
- Arena result: **rejected**. History version 37, agent `6566354`, submission
  `41026314`, settled rank 7 of 206 with score `41.50701665063712`, 61 wins,
  and 29 losses. Player 0 went 36-13 and player 1 went 25-16. Against the
  strongest recurrent opponents it went 0-6 against jacek, 0-3 against
  Marchete, 0-2 against Deltaspace, 0-4 against Laars, 3-4 against Snekkers,
  3-0 against EricSMSO, and 5-1 against derjack. Removing the tactical
  extension recovered the strong total and the derjack repairs worked, but
  the opponent-weighted score remained below the immutable reference.

## Candidate v10: restore the evidenced jacek suffix

- The only behavioral change from v9 restores action `7` after exact prefix
  `0/0/1/6`, from candidate-v1 win `896549711`, in place of v9 action `43`.
  Screening all 90 completed v9 games changes exactly two games: both are
  player-0 losses to jacek, and no completed win changes. Both losses reached
  the exact prefix and played `43`; v9 went 0-6 against jacek overall.
- Replay book: 28 paths / 587 decisions. Paste-ready source: 97,585
  characters, SHA-256
  `7ac49501930daa14a938543cd251c6772913b098270653d8ff3faf990fd8794f`.
  The artifact suite is 13/13; fixed-node gates remain 11-7 at 10,000 nodes
  and 40-34 at 30,000 nodes, both with no unfinished games. The full
  repository suite is 18/18, the generated sources are current, and measured
  release timing remains about 650.1 ms first and 130.1 ms later for both
  roles. The 306-game gate remains 178-128 (107/71 wins by candidate color),
  with no unfinished games. History version 38, agent `6566383`, submission
  `41026363`, settled rank 5 of 206 with score `42.280679071799426`, 57 wins,
  and 33 losses. Player 0 went 35-14 and player 1 went 22-19. It went 0-6
  against jacek, 1-8 against Marchete, 1-5 against Deltaspace, 2-5 against
  Laars, 5-1 against Snekkers, 6-0 against EricSMSO, and 6-1 against derjack.
  The restored suffix matched ten candidate decisions across both targeted
  player-0 jacek games, but jacek deviated later and won both. The batch
  improved v9 by `0.773662421162306` score points and two leaderboard places,
  yet remained `0.147052401161814` below the immutable reference. It is the
  strongest completed challenger in this series but is **rejected as a
  rank-5 replacement**.
- A final public-game audit found no safe v11 replay. All six jacek losses
  first diverged from known winning transcripts on jacek's move, leaving no
  same-state winning response to copy. The closest Marchete alternative,
  action `3` after `0/3`, would overwrite 14 v10 wins; the closest Laars
  alternative, action `5` after `0/5`, would overwrite four. No exact-prefix
  Deltaspace winner was retrievable. A further submission would therefore be
  an unscreened gamble rather than a reliable challenger.

## Rejected or deferred work

- Do not repeat the 900/180 ms clocks, retrained 135-game evaluator, direct
  rank-leader opening imitation, broad overrides, or conflicting partial
  histories; `../rank_5/EXPERIMENTS.md` records their completed regressions.
- Cross-response transposition reuse is deferred. Mate scores are root-ply
  relative; the sound guarded experiment still regressed the paired gate.
- Wider tactical solvers remain deferred until this small exact proof has a
  completed arena result and enough source-size/timing margin.
- A symmetric defensive extension for zero-fresh opponent goal corridors was
  tested after v3 and rejected before submission. It preserved the target
  actions but expanded too many leaves: at 30,000 nodes the `.../431` probe
  fell from completed depth 5 to 3 and the player-1 `.../003` probe from depth
  4 to 3; at 130 ms `.../0633` fell from depth 6 to 5. The narrower attacking
  extension was retained through v8, then removed with tactical extensions in
  the stronger proof-only v9 foundation.
- A depth-zero penalty for latent opponent rebound corridors was tried at
  10,000 and 5,000 centipoints. The former fell to 35-39 and the latter reached
  only 39-35 with severe color skew on the 74-game gate, so both were removed.

## Next action

No further arena submission is justified without a collision-screened exact
leader response or a general search improvement that preserves the completed
local gates. Candidate v10 is the strongest coherent challenger result, but
`../rank_5/` remains the immutable accepted bot because v10 did not exceed its
verified score or reach rank 4.
