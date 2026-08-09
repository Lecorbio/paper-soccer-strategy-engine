# First-full-exchange exact proof ledger

## T7: reply plus counterturn

T7 retains the unconditional current-turn root proof and depth-zero proof from
T3. At positive depth it probes the transposition table first. If no adequate
cached cutoff applies, exact rebound-component proof runs only at
`turn_ply == 1` or `turn_ply == 2`.

`search()` is entered at complete-turn boundaries and increments `turn_ply`
once per completed player turn. Ply one is therefore the opponent reply to a
candidate root action; ply two is the root player's counterturn after that
reply. This covers one full exchange without a board, cohort, score, or
learned threshold. Exact Win and Loss return mate-band scores, while `Unknown`
never changes evaluation or pruning.

Search statistics expose `exchange_ply{1,2}_probes`, `_win_hits`,
`_loss_hits`, and `_cutoffs`. The comparison JSON carries all eight counters
per candidate game so the two placements can be audited separately. In time
mode both engines receive a construction-inclusive absolute deadline and the
declared `max_nodes`; the runner rejects any decision exceeding that safety
cap.

## Frozen artifact identity

Relative to `reply_proof/bot.cpp`, T7 changes 34 lines: 29 insertions and five
deletions, including its separate placement diagnostics. The replay book and
replay-value model remain byte-identical.

The generated submission is 98,125 characters with SHA-256
`7d7b1a16d173bce56af021f6cb723587ae59006696c55436bd08320f4b2fe800`.
Its maintained `bot.cpp` has SHA-256
`fe98010e57451b79133c8a54fe2c430a782e1a759549400751be0b1e1996283b`.
The contest limit leaves 1,875 characters of headroom.

## Correctness coverage

Sixteen candidate tests cover both colors, root and depth-zero proofs, ply-one
Win and Loss, and ply-two Win and Loss. The ply-two fixtures compare exact
fixed-depth actions and mate scores with the leaf-only T3 reference, repeat
with the transposition table disabled, replay the proven terminal outcome, and
verify input immutability. A combined time-and-node configuration proves that
the node safety cap remains active in time mode. Existing replay, contest-rule,
atomic-input, interrupted-search, and full replay-book coverage is retained.

## Exposed-bank transfer screen

All fixed-node games use paired colors against immutable `rank_5` and reuse a
same-bank, same-budget `rank_5` control. Uplifts are physical Player 0,
physical Player 1, historical winner, and historical opponent.

| Exposed bank | Nodes | Games | Control uplifts | Elite/rank1 | d0 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| development | 5,000 | 53-43 | +0.042 / +0.062 / +0.062 / +0.042 | elite 0.545 | 0.667 | 1.068x |
| old rank-one | 5,000 | 75-61 | +0.029 / +0.074 / +0.015 / +0.088 | rank1 0.551 | 0.571 | 1.090x |
| T4 reference | 5,000 | 55-41 | +0.021 / +0.125 / +0.104 / +0.042 | elite 0.587 | 0.611 | 1.054x |
| retired ae5c | 5,000 | 77-67 | +0.014 / +0.056 / +0.111 / -0.042 | elite 0.500 | 0.500 | 1.071x |
| exposed ef48 | 30,000 | 76-68 | -0.028 / +0.083 / +0.028 / +0.028 | elite 0.594; rank1 0.389 | 0.577 | 1.093x |
| exposed ef48 | 100,000 | 86-58 | +0.083 / +0.111 / +0.111 / +0.083 | elite 0.625; rank1 0.444 | 0.538 | 1.097x |

At 30k ef48 the candidate retained `0.861` of control winners and completed
mean depth `3.261` versus `3.239`; at 100k it retained `0.889` and completed
depth `3.849` versus `3.816`. The sole negative 5k ae5 role uplift is `-0.042`,
inside the later `-0.05` tolerance and paired with `+0.111` historical-winner
uplift.

## Deployment-profile screen

At a true 130 ms per engine decision on exposed ef48, T7 won **77-67**. Its
control uplifts were `-0.028 / +0.097 / +0.000 / +0.069`; elite, field, and
rank-one tier scores were `0.510 / 0.500 / 0.722`; and d0/d1/d2 scores were
`0.615 / 0.581 / 0.446`. It completed mean depth `4.073` versus `4.023`,
processed about `154k` versus `142k` nodes per searched decision, and reached
`1.098x` incumbent throughput.

Under the identical screen, frontier-only and adaptive-depth proof scored
72-72, while all-depth and first-reply proof scored 74-70. T7 therefore won the
deployment-oriented model selection while keeping its semantics narrower than
all-depth proof.

## Freeze rule

These are selection results from already exposed banks, not prospective
promotion evidence. Do not change this placement in response to the next
frozen validation result, and do not bind or run a new frozen test bank until
the normal protocol permits it. No promotion manifest or opening bank was
modified while creating T7.
