# Jacek-style neural challenger experiment record

## Controls and objective

Work continued on `codex/selfplay-nn-v2`, whose requested starting commit was
`4ebb6ffb00acaa04937ecb0235fad885044e9762`. The branch already contained the
completed v2 experiment at `522e13d83311caa2e0660f4cb23503d43ddf522b`
before this challenger was added. The protected controls remain unchanged:

| Bot | Verified live result |
| --- | --- |
| `rank_5` | agent `6561779`, rank 5/206, score `42.42773147296124` |
| `selfplay_nn` | agent `6567898`, rank 5/206, score `41.77236003585933` |

The live leaderboard showed the incumbent account at rank 5 with score 42.00
immediately before this submission; rank 4 was 44.35. The experiment's target
is a reproducible improvement over the verified 42.4277 incumbent, preferably
rank 4 or better.

## v47 anchor-preserving pivot

The two completed replacement-model batches below exposed a structural error:
turning their learned value off restored only the hand evaluator, not rank_5's
complete proven evaluation. Rank_5 already contains Jacek's 1,156-input
used-edge/true-turn-distance representation in an `8 -> 8 -> 1` value model,
trained from strong public Jacek games and blended 15% into the hand score. The
replacement also removed rank_5's 24-path/511-decision verified replay book.
That explained why strong local games did not transfer to elite openings.

V47 instead copies the complete rank_5 search, replay book, and 85/15
hand/article-network evaluation byte-for-byte, then adds a bounded linear
correction after 12 used edges. Its 24 inputs reuse the incumbent model's
second hidden layer plus mover-relative hand, anchor, geometry, mobility, and
phase signals. The soft target is
`mover * (teacher root score - full rank_5 anchor score)`, clipped at 20,000.
The correction is capped at 6,000 points. This is a general state evaluator;
there are no new replay paths or arena-prefix branches.

The deterministic 10,000-game round-0 corpus already contained the required
full-anchor snapshots. After excluding the frozen v45/v46 holdouts, its split
contains 7,960/978/1,062 games and 249,293/30,403/31,837 unique samples. Huber
IRLS validation selected 50% strength: anchor MAE 4,842.69 fell to 4,781.89.
The model parses 338 distinct held-out live positions and rejected 110 exact
training matches. Its corpus SHA-256 is
`fb1600d5a667f77a35f23121e2f036faac88877828080eb87b101fa89bde2da4`.
The retained model JSON SHA-256 is
`ed6274f195905536c12086c4317af2ec93320004cdcbc30c80807674e2d5bbb6`.

After v47, all 404 then-current regression entries matched rank_5 exactly at
zero residual strength in action, root score, completed/attempted depth, nodes,
and exhaustion. The
strength sweep over the same 106-game 5k bank was 53-53 at 25%, 64-42 at 50%,
60-46 at 75%, and 58-48 at 100%. The trainer-selected 50% setting was retained:

| Gate | Candidate | Reference | Candidate colors | Candidate/reference depth | Candidate/reference nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5k, 106 games | 64 | 42 | 36 / 28 | 2.5519 / 2.5960 | 9,471,869 / 9,422,706 |
| 5k, 306 games | 174 | 132 | 91 / 83 | 2.5113 / 2.5469 | 27,405,707 / 27,299,768 |
| 30k, 106 games | 58 | 48 | 34 / 24 | 3.3373 / 3.3677 | 56,520,800 / 56,401,001 |
| 800/165 vs 650/130 ms, 18 games | 10 | 8 | 4 / 6 | 4.0150 / 3.9575 | 57,124,263 / 47,291,552 |

Every gate finished without an invalid action or incomplete rebound. The timed
800/165 run searched 20.8% more nodes and completed more depth. Replication
rejected the apparent timing gain: direct 800/165 versus the same challenger at
650/130 split 10-8 and 7-11 across two runs, 17-19 combined. Equal-profile
650/130 challenger-versus-rank_5 runs split 6-12 and 12-6, 18-18 combined. The
small timed bank is horizon-sensitive, so the larger clock is not promoted in
v48. The exact v47 800/165 source is 98,623 characters with SHA-256
`fb570f7d60157ad1681569011b4249a5db415c1aeca6f665936b26ba5cc52102`.

## Rejected v45/v46 replacement design

The v45/v46 engine started from rank-5 complete-turn alpha-beta, terminal
proofs, transposition/evaluation caches, legal fallback, and hand evaluation.
It deliberately removes the exact-history replay book from the challenger so
arena replay patches cannot masquerade as general learning. The learned
component is the article's original 1,156-input, 32x32 value network:

```text
316 used-edge flags + 105 x 8 true-turn-distance one-hot
    -> 32 ReLU -> 32 ReLU -> mover-relative teacher value
```

There is no policy head. Player 2 positions rotate 180 degrees. Three-bit
per-channel quantization plus Base64 packing keeps the model within the source
limit. Integer first-layer accumulation and scale hoisting reduced the small-
model fixed-node CPU ratio from about 1.43x to 1.31x; the final 108-game gate
measured roughly 1.25x, with search-shape differences included.

## Training rounds

### Small 384-game feasibility model

The first model reused the existing 384-game teacher corpus. A 15% residual
beat the full replay-enabled incumbent 58-48 at 5k nodes over 106 games. Larger
unconditional blends were negative: 25% scored 50-56, 50% scored 41-65, 75%
scored 43-63, and 100% scored 45-61. This established a narrow residual route
but was not enough data for promotion.

Excluding mate scores improved ordinary validation correlation, while keeping
mates produced stronger paired play. Both hypotheses were carried into the
larger corpus rather than selected only by validation loss.

### Round 0: 10,000 teacher games

Fourteen deterministic shards generated 10,000 rank-5 trajectories in about
15 minutes. The merged corpus is 166,576,824 bytes with SHA-256
`fb1600d5a667f77a35f23121e2f036faac88877828080eb87b101fa89bde2da4`.
The mate-inclusive model reached a preliminary 181-125 over 306 games at 5k
nodes and 62-44 over 106 games at 30k nodes. Those numbers are retained only
as development evidence: an audit found the observed-loss header parser was
excluding only the original 13 cases, not all 203 distinct states, so the
model was retrained and the preliminary gates are not promotion evidence.

### Round 1: 2,000 mixed-actor games

The second generator labels every position with rank-5 search but varies the
actor across four balanced modes: challenger as Player 1, challenger as
Player 2, challenger self-play, and teacher self-play. Eight shards produced
2,000 games and 22,020,800 bytes with SHA-256
`1147e3c37ddc4d8f69c784e3da05c63d3e542d4d2159a0c63a5ea23e0c6d2aa6`.

The final merged corpus contains 12,000 games and 188,600,734 bytes, SHA-256
`fbfc48f1a3cf071ffdae7600792f6541511b0a60faf7d0ee2871207734a45beb`.
Its retained quantized model has SHA-256
`7818d9041334294b1c78fa4c27240a867585ed2b2055e4350109b6d103a87ea0`.
The strict split and final metrics are:

| Split | Games | Samples | Soft BCE | Sign accuracy |
| --- | ---: | ---: | ---: | ---: |
| train | 9,649 | 375,045 | 0.588109 | 85.18% |
| validation | 1,141 | 44,000 | 0.598729 | 83.53% |
| test | 1,210 | 46,400 | 0.596075 | 83.83% |

The trainer rejects shallow depth-1 labels, exact arena prefixes, and any
original or reflected feature vector shared with an arena holdout. All 203
distinct states are parsed with a declared-count assertion. The final fit
rejected 135 exact prefixes and another 90 augmented representation matches.

## Calibration and residual sweep

An audit found that the inherited winner-probability mapping
`tanh(logit / 2) * 100000` amplified ordinary teacher-score logits by roughly
two to four times. The final engine instead inverts the training target with
`logit * 12000`. Target kind, feature schema, and temperature are emitted and
validated by the generated header.

On the final representation-held-out model, the calibrated 108-game sweep was:

| Anchor/model blend | Record |
| ---: | ---: |
| 92/8 | 59-49 |
| 90/10 | 60-48 |
| 88/12 | 66-42 |
| 85/15 | 59-49 |
| 82/18 | 64-44 |
| 80/20 | **67-41** |
| 75/25 | 57-51 |

The non-monotonic result is expected because the evaluation changes pruning
and completed actions, not just a final numeric ranking. The balanced 20%
setting was retained and revalidated at larger work budgets.

## Ungated v45 local gates

Every gate uses fixed openings, both candidate colors, the replay-enabled
rank-5 incumbent, complete-turn legality checks, and the 205-case chronological
loss suite. Candidate replay corrections remain disabled.

| Gate | Candidate | Reference | Candidate colors | Candidate/reference depth | Candidate/reference nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5k, 108 games | 67 | 41 | 35 / 32 | 2.5840 / 2.5560 | 9,664,247 / 9,581,435 |
| 5k, 308 games | 181 | 127 | 101 / 80 | 2.5786 / 2.5398 | 26,867,707 / 26,701,807 |
| 30k, 108 games | 60 | 48 | 34 / 26 | 3.3633 / 3.3037 | 61,807,363 / 61,543,370 |

All three gates completed with zero unfinished, invalid, malformed, or
rebound-truncated games. Independent inference testing over 120 legal states
found maximum Python/C++ logit error `9.54e-7` and exact player-swap rotation
symmetry. At the 5k horizon, 127 of the 205 chronological regression cases
changed complete actions relative to rank_5 and 78 remained identical. Every
changed action was legal and rebound-complete; the changes are accepted by the
two-color game gates, not by state-specific replay exceptions.

The first live batch nevertheless exposed an opening-distribution mismatch:
the candidate's model changed the true initial search, reduced completed depth,
and repeatedly chose opening `5` in losses to the strongest opponents. The
resulting v46 change is a state-general phase gate, not an arena-prefix lookup:
the neural residual remains exactly disabled until four board edges have been
played. A deterministic unit test verifies that the gated initial search has
the same action, score, nodes, and completed depth as a zero-blend rank-5
search, with zero neural evaluations.

## Phase-gated v46 local gates

The phase threshold sweep at 5k nodes over 108 games scored 67-41 with no gate,
65-43 after four edges, 64-44 after eight, 64-44 after twelve, and 61-47 after
sixteen. Four edges retained most of the learned gain while restoring the
verified anchor for the most distribution-sensitive search, so it was selected
before the second live batch.

| Gate | Candidate | Reference | Candidate colors | Candidate/reference depth | Candidate/reference nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5k, 108 games | 65 | 43 | 32 / 33 | 2.6233 / 2.5636 | 9,307,637 / 9,246,575 |
| 5k, 308 games | 176 | 132 | 95 / 81 | 2.6004 / 2.5263 | 26,749,130 / 26,619,485 |
| 30k, 108 games | 62 | 46 | 37 / 25 | 3.4307 / 3.3383 | 60,611,326 / 60,405,793 |

The gated source completed every game without an invalid action, unfinished
turn, or rebound failure. It also replayed all 205 prior chronological cases
and the 72 new v45 loss cases. Its equal-time 650/130 ms result was 9-11 on the
small 20-game bank, with candidate/reference completed depths 3.9125/3.9143;
that is a timing safety check rather than evidence of superiority.

## Deadline profiles and rejected search changes

The 20-game timed bank compared the candidate directly with rank_5 at its
650/130 ms profile:

| Candidate profile | Record | Nodes | Completed depth |
| --- | ---: | ---: | ---: |
| 650/130 | **10-10** | 39,584,694 | 3.7330 |
| 800/165 | 9-11 | 53,909,902 | 3.7969 |
| 900/180 | 8-12 | 52,347,033 | 3.9375 |

The larger profiles completed more depth but did not play more strongly on
the fixed pairs. They were rejected rather than promoted on the assumption
that unused clock must help. V46 retained 650/130; repeated final
Release probes remained below 651 ms on the first turn and below 131 ms on
later turns for both player positions.

The inherited iterative deepening still starts the next depth and returns only
the last fully completed result. A prior v2 root-order reuse experiment was
locally negative, and no new partial-depth policy had controlled evidence, so
this challenger does not promote unfinished work. Improving that safely is a
future search experiment, not an unmeasured patch.

## v45/v46 verification

The final Release build passed all 33 CTest targets. These cover contest
rules, own goals, blocked loss, atomic action parsing, mandatory rebounds,
deadline fallback, packed Python/C++ goldens, mover symmetry, deterministic
fixed-node search, disabled replay corrections, source freshness and size,
both-player protocol smoke, and the comparison gate. The generated v46 source
is 96,859 characters with SHA-256
`15de72119c204260f59e7c8dadc5190dbe8ceb4186df7d8cb7ad41f0aff51b1f`.

## Live arena

### v45: ungated residual, rejected

History version 45, agent `6568126`, completed 90 games at 54-36 and scored
`41.69791926441203`, rank 6/206. That is below both the verified 42.4277
incumbent and the promotion threshold despite the strong fixed-node gates.
The decisive elite matchups were 0-9 against `jacek`, 1-9 against `Marchete`,
1-5 against `YurkovAS`, 2-4 against `Deltaspace`, and 2-3 against `Snekkers`;
the bot went 9-1 against `EricSMSO`, 6-0 against `red1ynx`, 4-1 against
`derjack`, 4-3 against `Laars`, and 3-1 against `trictrac`. Losses split 22 as
Player 1 and 14 as Player 2, so there was no single-color legality failure.

All 36 loss transcripts were fetched from the completed batch. An independent
C++ replay validator accepted all 1,824 complete turns and 5,764 primitive
edges, including every mandatory rebound, and every transcript ended in a
natural opponent win. The retained suite samples first, one-third, and
two-thirds own-turn states into 72 deduplicated chronological cases; no timing,
empty-action, malformed-action, or invalid-move evidence was observed.

### v46: four-edge phase gate

The exact 96,859-character gated source was submitted as history version 46,
agent `6568130`. It completed 90 games at 51-39, score
`40.23086717434655`, rank 6/206, and was rejected as worse than both v45 and
the incumbent. Its elite records were 0-5 against `jacek`, 0-6 against
`YurkovAS`, 1-8 against `Deltaspace`, 2-5 against `Marchete`, 4-3 against
`Snekkers`, and 2-4 against `Laars`; losses split 18/21 by player position.

All 39 losses independently replayed as 2,038 legal complete turns and 6,480
primitive edges with every mandatory rebound complete and a natural opponent
win. They added 75 sampled cases; together with v45, the chronological header
contains 141 player/prefix-deduplicated entries.

### v47: full rank_5 anchor plus retrained residual

The exact 98,623-character source was submitted as history version 47, agent
`6568141`. It completed 90 games at 59-31, rank 5/206, score
`42.32319922389245`. That is 0.1045 below the verified incumbent score despite
two additional wins, so it is stronger by raw record but does not clear the
weighted-score promotion threshold. Key records were 1-6 against `jacek`, 0-7
against `Marchete`, 3-4 against `Deltaspace`, 2-3 against `Laars`, 3-6 against
`Snekkers`, 6-2 against `YurkovAS`, 6-2 against `trictrac`, 4-0 against both
`EricSMSO` and `About`, and 3-1 against `derjack`. Losses split 19/12 by player
position.

All 31 losses independently replayed as 1,511 legal complete turns and 4,682
primitive edges, with every mandatory rebound complete and every transcript
ending in a natural opponent win. They add 62 sampled states; the evolving
chronological suite now has 199 player/prefix-deduplicated entries.

### v48: 650/130 confirmation

V48 changes only the rejected timing profile back to rank_5's safe 650/130 ms.
Its paste-ready source remains 98,623 characters and has SHA-256
`3debbcd3043d3c0fb18ddac982d05a6ca82835f01b610ea256ea2c940cbd080a`.
It was submitted as history version 48, agent `6568150`, and completed at
52-38, score `39.73850251341997`, rank 8/206. This independent clock-ablation
result is much worse than v47's 59-31 and 42.3232. Because arena opponent samples are not
paired, this is an independent full-batch clock ablation rather than an exact
A/B; it is nevertheless enough to reject the lower clock alongside the noisy
local timing pairs. Key records were 0-3 against `jacek`, 1-4 against
`Marchete`, 0-3 against `Deltaspace`, 3-5 against `Laars`, 5-4 against
`Snekkers`, 0-7 against `EricSMSO`, 2-2 against `YurkovAS`, 3-5 against
`trictrac`, and 7-1 against `saraneth`; losses split 22/16 by position.

All 38 losses independently replayed as 1,897 legal complete turns and 5,948
primitive edges, including every mandatory rebound and a natural opponent
win. They contributed 69 sampled cases and grew the deduplicated chronological
header to 261 entries.

### v49: byte-identical v47 confirmation

The exact v47 source was restored byte-for-byte: 98,623 characters, SHA-256
`fb570f7d60157ad1681569011b4249a5db415c1aeca6f665936b26ba5cc52102`.
It was submitted as history version 49, agent `6568158`, and completed all 90
games at 61-29, score `40.39085147051933`, rank 6/206. The raw record improved
on v47, but the opponent-weighted score did not. Key records were 0-7 against
`jacek`, 1-4 against `Marchete`, 0-3 against `Deltaspace`, 2-6 against
`Snekkers`, 2-3 against `Laars`, 2-1 against `YurkovAS`, 2-2 against
`EricSMSO`, 4-2 against `derjack`, 4-0 against `trictrac`, and 6-0 against
`red1ynx`. Losses split 15/14 by player position.

All 29 losses independently replayed as 1,476 legal complete turns and 4,770
primitive edges, including every mandatory rebound, and ended in natural
opponent wins. They contributed 59 sampled cases and grew the deduplicated
chronological header to 308 entries. Across all suites there are 513 raw
regression entries representing 497 distinct player/prefix states. At zero
residual strength, all 513 match rank_5 exactly in action, root score,
completed/attempted depth, nodes, and exhaustion. At the retained 50% strength,
156 actions change at 5k nodes and 184 at 30k; all are legal and complete.

The byte-identical v47/v49 batches average `41.35702534720589`, 1.0707 below
the incumbent. Their combined raw record is 120-60, but the hardest five
opponents were only 14-49: `jacek` 1-13, `Marchete` 1-11, `Deltaspace` 3-7,
`Snekkers` 5-12, and `Laars` 4-6. Rank_5's verified batch was 8-27 against the
same names. The residual created wins elsewhere without raising the
elite-weighted ceiling, so aggregate wins are not promotion evidence.

## Post-v49 controlled rejections

Three general improvements were tested after the confirmation; none was
submitted live.

The first added a +/-2,048 aspiration window from completed depth four, with
correct bound handling and a full-window retry on failure. It matched the
window-disabled engine exactly on all 513 regression entries at 50k nodes.
An equal-time 800/165 paired bank nevertheless lost 6-12 despite raising mean
completed depth from 3.997 to 4.079; 309 of 754 windowed searches required a
retry. More completed depth was not stronger play, so the change was removed.

The second halved residual strength only after 64 used edges while the ball was
in the mover-canonical defensive midfield. This rotation-symmetric gate was
suggested by held-out calibration and live-loss analysis, but the 106-game 5k
gate fell from the retained model's 64-42 to 58-48. Its 30k run was stopped
after that promotion gate failed, and the change was removed.

The final training round added 1,024 deterministic games labeled at
32k/64k/128k nodes to the original corpus. The 11,024-game combined corpus is
184,226,089 bytes with SHA-256
`95d78b9c8f800bf1bdcdb5f916b61039135ac19db23254b519acbaa3a5760954`;
the experimental model SHA-256 was
`03c85c295153da390dc87c21bc424f9d40445b9deaf24f891bccc0557fa7291c`.
All 308 chronological cases were frozen out of this fit, which rejected 151
exact arena states. The new model tied 64-42 on the 106-game 5k bank and
improved the 106-game 30k bank from 58-48 to 60-46, but regressed the expanded
306-game 5k bank from 174-132 to 168-138. On 120 elite-loss states it reduced
deep-anchor agreement from 59 to 58 and completed less depth. Its 98,624-byte
source SHA-256 was
`83674cc415e2be1fec8b24eba75dca71fbdbd1af7c74fbd2b31f79c0d67ef26e`.
The broader regression outweighed the isolated deep-bank gain; the original
model and exact v47/v49 source were restored.

## Final disposition and next step

No `jacek_nn` batch beat the verified `42.42773147296124` incumbent. The bot is
therefore preserved as the strongest useful experiment, not promoted as the
new champion; `rank_5` remains unchanged.

The central problem is now measured rather than architectural: the linear
residual improves validation MAE by only about 1.26%, yet alpha-beta amplifies
small value errors into many move and depth changes. On 120 v47/v48 loss states
at 30k, it changed 43 actions; a 100k-node rank_5 search favored the original
rank_5 action in 28 cases and the residual action in only four. Eight of the 24
residual inputs were constant on the held-out teacher sample, and the model
lacks explicit corridor and goal-line parity features.

The most promising next step is a compact uncertainty-gated residual trained
on stronger, elite-weighted soft targets with richer topology/parity signals.
It should down-weight corrections where teacher disagreement is high, retain
the complete rank_5 anchor, and clear both the 306-game broad gate and a frozen
elite-loss/deeper-teacher agreement gate before another arena submission.
