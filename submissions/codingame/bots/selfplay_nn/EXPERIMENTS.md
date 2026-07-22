# Experiment record

## Objective

Test the strongest general explanation for the leading public bots: learned
self-play evaluation and policy guidance, while preserving the verified
rank-5 bot's exact complete-turn search. Candidate v1 deliberately removed
exact-history replay corrections so the learned signal could be isolated.
Later versions restored only verified exact paths after the completed live
batch exposed specific opening holes.

## Data-pipeline corrections

The first 256-game bootstrap was rejected before promotion. Opening depth and
teacher budget were coupled by their loop indices, and every depth-zero game
was identical. Exact labeled features leaked into 19.2% of validation and
40.9% of test. The corrected generator keeps one depth-zero anchor, uses only
deeper randomized prefixes for the other games, and draws the node budget from
an independent random value. The trainer removes any exact train/held-out
feature overlap; the frozen corpus removed 269 validation and 326 test rows.

The first corrected trainer also supervised eventual value on every physical
edge. That was a domain mismatch: production alpha-beta evaluates only after a
complete turn. The retained trainer gives value loss weight only to the first
edge of each teacher turn while still training policy on every rebound edge.
Policy softmax is masked to legal normalized directions.

## Width and search ablations

The initial corrected 32x32 model was source-feasible at 96,303 characters. At
the selected 5% value / 500 policy setting it won 67-39 over 106 equal-node
games, but its first paired 130 ms sample was only 8-10. The larger evaluator
roughly halved search throughput on difficult positions and was rejected for
the production candidate.

The 16x16 model reduces the generated submission to 85,549 characters. The
following equal-node screens used the same 42-game opening set:

| Value blend | Policy weight | Candidate | Rank 5 |
| ---: | ---: | ---: | ---: |
| 0% | 500 | 20 | 22 |
| 5% | 250 | 20 | 22 |
| 5% | 750 | 17 | 25 |
| 10% | 500 | 24 | 18 |

The 10%/500 setting was frozen. On the larger 106-game, 5,000-node gate it won
68-38 without reference replay and 66-40 with rank_5 replay corrections
enabled. The latter color split was 34 wins as Player 0 and 32 as Player 1.

At 130 ms, the frozen candidate reached a deeper completed search on six of
nine recurrent probes, tied two, and trailed one. The small full-game gates
were 12-6 without reference replay and 11-7 with it. These timed samples are
positive but still too small to substitute for a complete live arena batch.

After v1's live regression, policy and value were isolated. The policy-free
10% value model won 27-15 on the 42-game screen, 69-37 on the 106-game
replay-free gate, and 67-39 when rank_5 kept replay corrections. The 0%/0
baseline was 21-21 and 49-57 respectively, proving that the learned value—not
policy ordering—created the local gain. On the 42-game blend sweep, 5% went
20-22, 10% went 27-15, 12% and 15% both went 24-18, and 20% went 25-17. The
retained release therefore uses 10% value and zero policy weight.

## Correctness and packaging

- Python and C++ quantized logits match on three golden states, including a
  Player-2 canonicalized state.
- Player rotation and board rotation preserve all policy/value outputs.
- One-millisecond and one-node searches always return legal, complete rebound
  actions.
- Every exact opening/replay action is reconstructed against the local game
  state before use; an illegal or incomplete action falls back to search.
- Model-header and paste-ready submission freshness are checked by CTest.
- The retained paste-ready source is 93,621 characters, 6,379 below the
  contest limit, with SHA-256
  `1f980ece41a8ec86347deb77ec4625b1a402f15e1d82456ab04fe1bac40a6389`.

## Arena v1: isolated neural candidate

History version 39, agent `6567850`, submission `41028505`, stabilized at rank
9 of 206 with score `39.59113038088426`, 59 wins, and 31 losses. It went 36-12
as Player 0 and 23-19 as Player 1. Against the strongest five it went 0-3
versus jacek, 0-1 versus Marchete, 1-3 versus Deltaspace, 1-3 versus Laars, and
2-4 versus Snekkers.

The dominant repeatable failure was Player 1 after opponent opening `1`: the
model chose `2` and the `1/2` family went 0-9. Rank_5's older evaluator chooses
`1`, after which four independently proven `1/1` paths are available. Removing
policy ordering did not change that root; this was lost baseline knowledge,
not a policy-head error.

## Arena v2: value-only plus verified replay recovery

V2 disabled policy ordering, restored the verified 24-path replay book, and
added the exact state-validated `1 -> 1` Player-1 response. Its source was
93,534 characters with SHA-256
`3480cc3a62fb917283fc1e1ed5202dd591f1374bf5b6d7f96654acf706933478`.

History version 40, agent `6567868`, submission `41028608`, completed rank 6
of 206 with score `41.2027135828999`, 62 wins, and 28 losses. It went 30-15 as
Player 0 and 32-13 as Player 1, reversing v1's color deficit. Strong-opponent
results were 0-4 jacek, 0-2 Marchete, 0-4 Deltaspace, 1-6 Laars, and 2-1
Snekkers. The repair was material but still below the immutable rank-5 score.

V2's Player-0 `0/6/1` family then went 2-7. At exact history `0/6`, the neural
value search chose `1` while verified rank_5 chose `5`, enabling its proven
Deltaspace and Snekkers branches. This was the only next opening correction
with both repeated live-loss evidence and a verified general baseline action.
Broad `0/0` overrides remained rejected.

## Arena v3: retained strongest candidate

V3 adds only the exact state-validated `0/6 -> 5` Player-0 response. It kept
the 67-39 equal-node result and improved the paired 130 ms gate from 10-8 to
12-6. Its paste-ready source is the retained 93,621-character artifact.

History version 41, agent `6567898`, submission `41028701`, stabilized at rank
5 of 206 with score `41.77236003585933`, 59 wins, and 31 losses. It went 31-17
as Player 0 and 28-14 as Player 1. Its strongest-opponent split was 0-5 versus
jacek, 1-8 versus Marchete, 1-2 versus Deltaspace, 2-8 versus Laars, and 5-2
versus Snekkers. Other recurrent results included 6-1 versus EricSMSO, 3-0
versus YurkovAS, 1-3 versus red1ynx, and 2-0 versus trictrac. Both cached and
uncached progress reached 100% before recording the result.

This is the strongest new neural bot by opponent-weighted score, but it did
not visibly surpass rank 5 or beat the immutable reference score
`42.42773147296124`. Preserve `rank_5` unchanged and do not rename this folder.

The next general neural cycle should optimize first-layer inference, replace
the expensive distance map with cheap hand-evaluator scalars, and train a
smaller value-only model on mover-relative teacher root scores rather than
only final outcomes. The 31 v1, 28 v2, and 31 v3 live losses should remain a
never-trained-on regression set, with explicit completed-depth stability
checks on recurrent jacek, Marchete, and Laars positions.
