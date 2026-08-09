# First-reply exact rebound proof ledger

## T6: first opponent-boundary proof

T6 starts from `rebound_proof` (T3). Root and depth-zero component proofs are
unchanged. At positive depth, the transposition table is probed first; only
when no adequate cached cutoff applies and `turn_ply == 1` does T6 run the
exact component analyzer before enumerating the opponent reply.

`search()` is entered at complete-turn boundaries and increments `turn_ply`
once per completed player turn. The added placement therefore checks exactly
the opponent state produced by a candidate root action. A proven Win rejects a
root action which concedes an immediate rebound combination; a proven Loss
identifies a reply trapped in a closed component. `Unknown` never prunes.
Proof returns are not stored because their mate-distance score depends on the
current turn ply.

## Frozen artifact identity

Relative to `rebound_proof/bot.cpp`, the candidate has exactly 23 inserted
lines: 13 implement and document the production reply proof, while 10 expose a
fixed-depth entry point only when `PAPER_SOCCER_REPLY_PROOF_TESTING` is defined.
The replay book and replay-value model are byte-identical to T3.

The generated paste-ready source is 96,957 characters with SHA-256
`de2cdd18ae37b93bb0a443cdcfe13e52f6992aa0ac1c7fb43a544a874cb59789`.
The generator's `--check` mode accepts it and the 100,000-character contest
limit leaves 3,043 characters of headroom.

## Fixed-depth correctness

Fourteen candidate tests pass. The positive-depth fixtures cover component Win
and Loss for both colors, compare the Win fixture's completed fixed-depth
action and score with the leaf-only T3 reference, repeat with the transposition
table disabled, and check that searches do not mutate their input. Root proof,
depth-zero proof, replay legality, atomic input handling, both contest colors,
own goals, used edges, boundary posts, closed components, and interrupted
searches retain their T3 coverage.

## Exposed-bank selection screen

All games below use paired colors against immutable `rank_5`, with a same-node
`rank_5`-versus-`rank_5` control for every opening. The four uplift columns are
physical Player 0, physical Player 1, historical winner, and historical
opponent.

| Bank | Nodes | Candidate games | Improvements / regressions | Control uplifts | Throughput |
|---|---:|---:|---:|---:|---:|
| development (48) | 5,000 | 51-45 | 10 / 7 | +0.000 / +0.063 / +0.042 / +0.021 | 1.088x |
| rank-one reference (68) | 5,000 | 75-61 | 17 / 10 | +0.015 / +0.088 / +0.000 / +0.103 | 1.114x |
| T4 reference (48) | 5,000 | 52-44 | 8 / 4 | +0.021 / +0.063 / +0.042 / +0.042 | 1.077x |
| retired ae5c (72) | 5,000 | 78-66 | 16 / 10 | +0.000 / +0.083 / +0.097 / -0.014 | 1.085x |
| development (48) | 30,000 | 55-41 | 12 / 5 | +0.063 / +0.083 / +0.083 / +0.063 | 1.150x |
| T4 reference (48) | 30,000 | 55-41 | 10 / 3 | +0.146 / +0.000 / +0.083 / +0.063 | 1.096x |
| retired ae5c (72) | 30,000 | 82-62 | 17 / 7 | +0.097 / +0.042 / +0.083 / +0.056 | 1.081x |

At 30,000 nodes the candidate/control winner-retention scores are `0.896`,
`0.938`, and `0.903` on development, T4, and ae5c. Candidate versus incumbent
mean completed depths are `3.271/3.224`, `3.211/3.212`, and `3.220/3.207`.
Every 30k role uplift is nonnegative.

The 30k stratum scores are development `d0=0.500`, `d1=0.583`,
`d2=0.571`; T4 `d0=0.444`, `d1=0.553`, `d2=0.650`; and ae5c
`d0=0.556`, `d1=0.621`, `d2=0.517`. Winner-tier scores are development
`elite=0.500`, `incumbent=0.595`; T4 `elite=0.565`, `field=0.708`,
`incumbent=0.462`; and ae5c `elite=0.611`, `field=0.563`.

These are model-selection results on already exposed banks, not prospective
promotion evidence.

## Freeze rule

Do not change this placement in response to prospective validation. Do not run
the sealed final bank unless T6 passes the frozen development and prospective
validation protocol. The sealed bank and promotion manifest were not used or
modified while selecting and freezing T6.
