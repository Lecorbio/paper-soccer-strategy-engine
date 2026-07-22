# Self-play neural v2 experiment record

## Immutable controls

Work began from commit `4ebb6ffb00acaa04937ecb0235fad885044e9762` on
branch `codex/selfplay-nn-v2`. The two controls remain separate and unchanged:

| Bot | Agent | Complete arena result |
| --- | ---: | --- |
| `rank_5` | `6561779` | rank 5/206, score `42.42773147296124`, 57-33 |
| `selfplay_nn` | `6567898` | rank 5/206, score `41.77236003585933`, 59-31 |

The replacement 16x16 policy/value network in `selfplay_nn` was locally
promising but did not exceed the live incumbent. Its policy-head ablation was
also negative. V2 therefore retains rank_5's proven evaluator and search as
the anchor, removes policy inference entirely, and learns only a bounded value
residual from soft teacher-search targets.

The original 23 CTest targets passed before v2 work. Rank_5 timing reproduced
at about 650/130 ms, and its fixed-node control reproduced exactly when the v2
residual was disabled: 53-53 over 106 paired games with identical actions,
scores, nodes, and completed depths.

## Teacher model

The deterministic generator plays 384 games from varied legal openings. Each
record stores the mover, transcript, chosen complete turn, rank-5 anchor score,
deeper root teacher score, completed and attempted depth, nodes, exhaustion,
and 24 mover-canonical features. Teacher budgets are 8k, 16k, or 32k nodes.
Targets are clipped `(mover_sign * (teacher - anchor)) / 20000`, not final-game
winners.

The retained model is Huber-IRLS ridge regression. This linear head reuses the
eight hidden activations already computed by the rank-5 replay evaluator and
adds only 24 multiply-adds at an eligible leaf. The split and fit are:

| Split | Games | Samples | Anchor MAE | Corrected MAE at 100% |
| --- | ---: | ---: | ---: | ---: |
| train | 296 | 9,625 | 4,801.48 | 4,764.03 |
| validation | 43 | 1,016 | 4,710.66 | 4,703.01 |
| test | 45 | 1,413 | 5,363.67 | 5,309.39 |

Validation preferred a 50% correction (MAE 4,671.99), while test preferred
50% narrowly over 100% (5,295.27 versus 5,309.39). Playing strength, however,
is the promotion gate rather than teacher-score fit.

The trainer rejected 782 shallow samples, 590 mate scores, 902 opening-gated
samples, two exact arena-held-out states, and 53 exact duplicate features (48
within train and five train/validation overlaps).
The original 13-state arena regression suite freezes documented `rank_5`,
v39, v40, and v41 loss families. The three later complete live batches are
strictly chronological unseen evidence: they were not used to refit the
submitted model. Their loss transcripts supply 192 distinct sampled entries.
Two state keys overlap the original suite, so the comparison gate validates
205 cases covering 203 distinct player/prefix states.

## Residual and phase-gate ablations

An unconditional residual changed the verified early `0/6` choice under a
deadline even at only 10-25% strength, so it was rejected. The retained design
enables residual leaves only when the root already has at least 12 played
edges, and independently suppresses leaves below that phase. Tests prove the
early gate performs zero residual evaluations and that rotation plus player
swap preserves every mover-relative feature while negating the anchor.

Fixed-node results use the same deterministic paired openings, both colors,
full replay behavior, complete games, and no unfinished or illegal results:

| Residual | 5k nodes, 106 games | 5k nodes, 306 games | 30k nodes, 106 games |
| ---: | ---: | ---: | ---: |
| 25% | 60-46 | 168-138 | 54-52 |
| 50% | 59-47 | 165-141 | 52-54 |
| 75% | 55-51 | not run | not run |
| 100% | 61-45 | 178-128 | 58-48 |

At 30k nodes, 100% was the only setting with a meaningful positive margin.
It averaged completed depth 3.2908 versus rank_5's 3.3873 because the changed
evaluation alters pruning and search shape, but still won 58-48. Its color
split was 35/23. It changed three of 13 held-out actions at that horizon
(`deltaspace-0633`, `deltaspace-431`, and `jacek-5330`) while preserving all
ten opening/family gates; every changed action remained legal and
rebound-complete.

## Deadline and search-management experiments

At 130 ms, the 100% candidate preserved all eight early verified actions and
all but the two eligible Deltaspace probe actions. At 800/165 versus the same
650/130 v2 engine, all 13 probe actions were unchanged; the larger budget
completed one additional depth on `deltaspace-0633` and `jacek-5330`. The
900/180 profile changed three verified early actions, including `0/6`, and was
rejected. Independent historical evidence agrees: rank_5's complete 900/180
batch scored `42.32812937013547`, below its 650/130 incumbent.

The 800/165 profile then beat the identical 650/130 v2 engine 10-8 over 18
paired timed games. It searched 20.7% more nodes, improved average completed
depth from 3.9098 to 4.0151, and had no unfinished or invalid games. Against
rank_5 at its verified 650/130 profile, v2 at 800/165 won 13-5, searched 30.1%
more nodes, and completed average depth 4.0887 versus 4.0339. Maximum observed
deadline overshoot was 0.13 ms for v2. These paired results support the larger
profile without approaching the published hard limits.

A compact iterative-deepening experiment reused every completed shallower
root-edge score to order the next iteration. Across all 13 held-out states at
exact depths 1-4 it preserved every action and root score and reduced the
depth-4 node total by 750/63,177 (1.19%). That small synthetic gain did not
transfer: against the identical v2 control at 5k nodes it lost 52-54, used
9,808,896 versus 9,800,765 nodes, and completed slightly less depth (2.4680
versus 2.4699). It was removed. An unfinished deeper iteration is still never
promoted; only the last fully completed depth, or the complete-action fallback
before depth one, may be returned.

## Live arena

### Version 42: 800/165 ms

- Paste-ready source: 98,624 characters; SHA-256
  `5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`
- Agent: `6567975`; history version 42
- Preflight game: `896616390`, legal goal with the editor copy matching the
  local source byte-for-byte
- Complete arena result: rank 5 of 206, score `42.82914600312645`, 58-32
  after all 90 games

Version 42 exceeds rank_5's verified score by `0.40141453016521`. It went 4-4
against Laars, 4-3 against Snekkers, 4-2 against derjack, 7-1 against
EricSMSO, 4-7 against Deltaspace, 0-10 against Marchete, and 0-4 against
jacek. There were no invalid moves, incomplete rebounds, crashes, or timing
failures. The complete batch, every loss transcript, and 70 sampled loss
states are frozen in `arena_batch_6567975.json`.

Seven player-0 losses shared the clock-specific `0/6/33` branch: four against
Deltaspace, two against Laars, and one against Snekkers. There were no losses
after `0/6/5`, while three proven replay-book wins begin with that branch. The
residual is disabled at this two-edge root; the server instead completed an
extra anchor-search depth and selected `33` rather than rank_5's proven `5`.
Locally, 130 ms returned `5` at completed/attempted depth 5/6 while 180 ms
returned `33` at depth 6/7. A hybrid 800/130 profile preserved all eight
documented early positions and still beat rank_5 11-7 in the 18-game timed
gate. That evidence justified one controlled follow-up rather than an exact
replay patch.

### Version 43: 800/130 ms

- Paste-ready source: 98,624 characters; SHA-256
  `5421cc7903d8509a3aa9fef360c2304b7e661116b49752780ace8b3e234b603d`
- Agent: `6567983`; history version 43
- Preflight game: `896617647`, legal goal with an exact editor copy
- Complete arena result: rank 6 of 206, score `40.46675247310762`, 53-37
  after all 90 games

Version 43 fell `2.362393530018835` points below v42 and
`1.960978999853623` below rank_5. It went 2-4 against Deltaspace, 1-3 against
Marchete, 1-4 against jacek, 2-8 against Laars, 3-2 against Snekkers, and 0-3
against EricSMSO. The reduced clock only redistributed the targeted family:
`0/6/33` fell from seven losses to two, but `0/6/5` then produced three
losses. It broadly lost playing strength, especially against Laars and
EricSMSO. An independent replay of all 37 losses (1,825 turns and 5,616 edges)
found every action legal, correctly alternating, rebound-complete, and
terminal with the opponent winning; there was no timeout, crash, empty, or
truncated-action evidence. The complete batch and 78 sampled loss states are
frozen in `arena_batch_6567983.json`.

The controlled live result rejects the 130 ms follow-up. Version 42 is the
retained challenger and `bot.cpp` plus the generated submission have been
restored byte-for-byte to its 800/165 ms source and SHA-256. The opening
regression remains documented evidence for a future phase-aware clock, not a
state-specific replay patch.

### Version 44: exact 800/165 ms confirmation

- Paste-ready source and SHA-256 are byte-for-byte identical to v42
- Agent: `6567993`; history version 44
- Preflight game: `896619755`, legal goal with an exact editor copy
- Complete arena result: rank 5 of 206, score `42.001868057527055`, 58-32
  after all 90 games

The exact rerun reproduced v42's rank and 58-32 record but finished
`0.42586341543418627` below rank_5 and `0.8272779455993984` below v42. It
went 0-5 against jacek, 1-5 against Marchete, 3-4 against Deltaspace, 5-2
against Laars, and 2-8 against Snekkers: 11-24 (31.4%) against the fixed top
five. Together, the two exact 800/165 batches went 23-52 (30.7%) against that
group, above rank_5's observed 8-27 (22.9%), but their mean score is
`42.41550703032675`, or `0.012224442634490629` below the incumbent.

An independent replay of all 32 losses (1,553 turns and 5,018 edges) again
found complete legality, mandatory rebounds, natural opponent wins, and no
timeout, crash, empty, malformed, or truncated-action evidence. Losses split
18/14 across player 0/player 1. The `0/6/33` player-0 branch recurred eight
times while `0/6/5` lost once, confirming an unstable general horizon family
rather than a safe exact-history repair. Only one full loss transcript was
shared by the two exact-source batches.

This confirmation does not meet the reproducible incumbent-beating score
gate. Rank_5 therefore remains the verified champion. V2 is preserved as the
strongest useful experiment, not promoted as a replacement: one batch cleared
the incumbent and the elite-opponent profile replicated, but the exact-source
score improvement did not. The next evidence-backed step is a general
phase-aware clock/search correction for the unstable early horizon, followed
by a fresh teacher cycle with all 203 regression-state keys held out.
