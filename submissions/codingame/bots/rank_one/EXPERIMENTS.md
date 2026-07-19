# Arena experiment record

## Baseline

The preserved live reference was agent `6561733`, submission `41015213`.
After all 90 games it was rank 6 of 206 with score `42.19914837204899`, 60
wins, and 30 losses. Its most important opponent records were 0-5 against
jacek, 0-5 against Marchete, 0-4 against Deltaspace, 3-6 against Laars, and
5-2 against YurkovAS.

All 30 losses were downloaded through CodinGame's public game-result service.
The first differing complete turn was compared with same-color public wins,
including rank-leading jacek's games against Laars and derjack. Representative
replays were watched through the goal frame. Recurring failures were early
branch choices after openings `0/0`, `0/1`, `0/3`, and `0/6/5/4`, plus long
forced rebound chains along a side or goal line. In the strongest games, the
decisive choice often preceded the visible goal by many forced edges.

## Candidate v1

- Maintained source: `bot.cpp`
- Paste-ready source: `submission.cpp`
- Source: 91,461 characters; SHA-256
  `30ac27e3d0b5e0f02aa3701c281fcf1d96250655b9a06bb026865d55b9f0e805`
- Replay book: 17 full paths and 350 exact-history decisions
- Arena history version: 24
- Agent: `6561767` (distinct from baseline `6561733`)
- Arena result: rank 9 of 206, score `40.68622842413971`, 56 wins and
  34 losses after all 90 games

Preflight passed a clean full build and all 13 CTest targets. Candidate-specific
coverage reconstructs every stored path, proves every eligible response legal
and rebound-complete, checks exact full-history matching and safe fallback,
smoke-tests both protocol colors, and verifies generator freshness. The timing
probe measured about 650 ms for the first response and 130 ms afterward. The
CodinGame editor was copied back byte-for-byte before submission, and “Play my
code” compiled and completed game `896344007` successfully. History advanced
from 23 to 24, confirming a new version.

The offline screen against the baseline batch found 16 games whose first
decision would change: 12 losses and 4 wins. Targeted loss branches cover
derjack, Deltaspace, Laars, Marchete, jacek, YurkovAS, field3, and saraneth.
The four collateral wins were against EricSMSO, field3, trictrac, and mdaw, so
the arena batch is required before accepting the candidate. The completed
batch regressed from the preserved rank-6 baseline and was rejected. Its key
records were 0-7 against Marchete, 0-5 against Deltaspace, 1-5 against jacek,
2-6 against Snekkers, 2-3 against Laars, 3-2 against derjack, 2-2 against
trictrac, and 2-1 against YurkovAS. It remained perfect against field3 and
saraneth (6-0 each).

## Candidate v2

- Paste-ready source: 93,005 characters; SHA-256
  `f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29`
- Replay book: 24 full paths and 511 exact-history decisions
- Arena history versions: 25 and 26
- Agents: `6561777` / submission `41015547`, and `6561779` / submission
  `41015554`
- Arena result: version 26 stabilized at rank 5 of 206, score
  `42.42773147296124`, with 57 wins and 33 losses after all 90 games.
  Version 25 stopped with no pending games after 88 results: score
  `37.11056795346562`, 48 wins, and 40 losses. Its different, much harsher
  opponent sample makes it useful as a stability warning but not a direct
  head-to-head replacement for the completed version-26 batch.

The two v2 submissions are byte-identical. CodinGame accepted both while the
History counter lagged, so both batches are retained as a stability replicate.
Each editor copy was checked byte-for-byte and compiled before submission.
V2 replaces candidate-v1 losses using same-color public winning paths: rank-1
jacek continuations against Deltaspace, Marchete, Snekkers, and derjack; a
rank-3 continuation shared by Snekkers, Laars, and Marchete losses; and later
exact prior wins against YurkovAS, EricSMSO, trictrac, and Shurishe. Screening
against the completed v1 batch changed 21 losses and 11 wins, so complete
arena batches—not the promising early win rate—determine acceptance. The
completed version-26 batch improved both rank and score over the preserved
reference and is the strongest verified live candidate before v3. Its key
records were 0-8 against jacek, 0-6 against Marchete, 2-3 against Deltaspace,
2-5 against Laars, 4-5 against Snekkers, 5-2 against YurkovAS, 6-1 against
red1ynx, 5-1 against EricSMSO, and 4-0 against saraneth.

After the later experiments, the exact version-26 source was reconstructed from
the preserved compiled binary. Its 24 replay records, timing constants, source
length, and SHA-256 all reproduce exactly; this byte-identical source is the
maintained final candidate rather than an approximation. Final verification
passed all 13 CTest targets; the timing probe measured at most 651.2 ms for a
first response and 131.4 ms afterward. The CodinGame editor copied back at the
exact recorded hash and compiled to a legal goal in game `896354994`. It was
not resubmitted, preserving the already-completed version-26 arena evidence.

## Candidate v3

- Paste-ready source: 93,474 characters; SHA-256
  `ea1897e222448f908785b1c96b3f31a7226d7148df978db2940b1f1341cb907c`
- Replay book: 26 full paths and 561 exact-history decisions
- Arena history version: 27
- Agent: `6561801`; submission `41015649`
- Arena result: rank 7 of 206, score `41.60256379708299`, with 60 wins and
  30 losses after all 90 games; rejected in favor of v2.

V3 retrains the general static evaluator on 135 recent public games from
rank-leading jacek. On the same held-out recent replay split, classification
accuracy rose from 80.527% for the preserved model to 97.769%, while log loss
fell from 0.94997 to 0.03775. The evaluator JSON and generated header are kept
inside this isolated candidate and the existing reference bot remains intact.
V3 also uses jacek's own same-color response at the recurrent `0/0` state and
adds later exact branches for Deltaspace and Laars failures. Screening against
the completed version-26 batch found 16 changed games: 10 losses and 6 wins.

Before submission, all 13 CTest targets passed. The timing probe measured
651 ms on first responses and 131 ms afterward for both colors. The generated
source was copied back from the CodinGame editor exactly at the SHA-256 above,
and “Play my code” compiled it and completed game `896349849` with a legal
goal. History advanced exactly once from 26 to 27, confirming a distinct new
version and agent.

The complete batch confirmed that classification accuracy was not a useful
proxy for arena strength. V3 went 0-4 against both jacek and derjack, 1-3
against Marchete, 2-2 against Laars, and 3-3 against each of Deltaspace,
Snekkers, and YurkovAS. In every sampled player-0 loss to jacek, the new model
changed the established northward opening `0` to south-west `5`; the exact
`0/0` continuation therefore never became eligible, and jacek converted a
sparse path to the bottom goal in 22 turns. Representative Deltaspace and
Snekkers losses ended in a forced bottom goal and a left-wall block after long
rebound chains. The 135-game model was removed from the maintained candidate.

## Candidate v4

- Paste-ready source: 93,474 characters; SHA-256
  `de8df51da3a8d517f95702c98cae9cb9f633254d4c9943379bf5108eafade4e1`
- Replay book: 26 full paths and 561 exact-history decisions
- Arena history version: 28
- Agent: `6561808`; submission `41015724`
- Arena result: rank 7 of 206, score `41.5464903097154`, with 56 wins and
  34 losses after all 90 games; rejected in favor of v2.

V4 is the controlled ablation: it restores v2's arena-proven 88-game
evaluator while retaining only the v3 replay-book additions. The first two
rank-leader branches were exercised locally. Against histories `0/0` and
`0/0/3`, v2 chose `1` and `67`; v4 chooses the source rank leader's `3` and
`54`. All proposed turns remain legality checked on a copied state with normal
search as fallback. All 13 tests passed, explicit timing was 651 ms / 132 ms
or less, the CodinGame editor copied back exactly at the hash above, and the
compiled test completed game `896351871` with a legal goal. History advanced
exactly once from 27 to 28.

The completed ablation showed that the early exact substitutions themselves
were harmful, independently of the rejected v3 evaluator. V4 finished 0-7
against jacek, 0-6 against Marchete, 0-5 against Laars, 0-3 against
Deltaspace, and 1-3 against Snekkers. Its jacek games confirmed why imitation
failed: after our copied rank-leader choice, jacek supplied a stronger reply
than the source opponent, immediately leaving the stored full-history path;
normal search then returned to the losing continuation. The four v3 changes
were removed rather than expanded with unsafe partial-history matching.

## Candidate v5

- Paste-ready source: 92,715 characters; SHA-256
  `cd9df059bcb6637e66e091de0f258904f0750ee7bb4712ba21e087008f678211`
- Replay book: 23 full paths and 482 exact-history decisions
- Arena history version: 29
- Agent: `6561813`; submission `41015794`
- Arena result: rank 5 of 206, score `42.32812937013547`, with 57 wins and
  33 losses after all 90 games; rejected in favor of v2.

V5 returns to v2's proven evaluator and pre-v3 replay behavior, drops the
unsuccessful jacek-imitation continuation, and tests a general search increase:
900 ms on the first response and 180 ms afterward instead of 650/130 ms. The
timing probe measured 901.1 ms and 181.1 ms at worst, leaving 98.9 ms and
18.9 ms under the published limits. All 13 tests passed, generator freshness
passed, the editor copy matched the SHA-256 above, and CodinGame compiled and
completed game `896353741` with a legal goal. History advanced exactly once
from 28 to 29.

The completed batch did not justify the extra search time. It went 0-8 against
jacek, 2-5 against Marchete, 1-5 against Deltaspace, 1-2 against Laars, 5-3
against Snekkers, 5-2 against derjack, and 7-1 against YurkovAS. The live rank
was visibly 5 of 206 after stabilization, but score remained below v2 by
`0.09960210182577` and its strongest-opponent results were weaker. Replays again
showed the first important split in the opening or early rebound chain: jacek
losses began through `0/0/1` or `0/0/3`, while Deltaspace repeatedly converted
the `0/6/5` family. A representative new jacek loss was game `896354047`; the
larger budget kept the same early decision, so horizon alone did not repair the
position. The exact v2 source was restored.

## Rejected approaches

- A Snekkers response conflicted with a Deltaspace response at the identical
  full history `0/6`; the generator rejected it and the path was removed.
- Broad first-turn replay overrides changed 37 baseline games (19 wins and 18
  losses). They were replaced with later exact divergences wherever possible.
- A broad Marchete-versus-jacek response after opening `0` affected 12 wins for
  only four losses; it was removed in favor of a late jacek-specific branch.
- A trictrac player-0 path changed two wins and one loss and was removed.
- Early Laars player-1 and other opening-level corrections were rejected when
  local evidence could not separate the target opponent from successful
  baseline histories.
- Increasing search from 650/130 ms to 900/180 ms consumed most of the timing
  margin but scored slightly worse over a complete batch, so the final source
  restores the smaller arena-proven budgets.

## Next improvements if rank 1 is not reached

Prioritize the recurrent `0/0` jacek and `0/6/5` Deltaspace families. The next
general improvement should value forced rebound corridors and goal-line entry
parity earlier, or improve move ordering so the existing budget reaches those
tactics; merely lengthening the clock did not help. Validate changes first with
deterministic equal-node paired games and the full replay corpus. Compare the
first candidate divergence with same-color wins from jacek, Marchete, or
another rank-leading bot, then encode the behavior as a general search feature
when possible. Only add an exact response when the complete player/state
history is available, the local legality suite passes, and normal search
remains a fallback. Re-screen every proposed book against both wins and losses
before another complete arena batch.
