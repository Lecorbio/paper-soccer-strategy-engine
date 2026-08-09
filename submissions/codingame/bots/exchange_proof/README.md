# First-full-exchange exact rebound proof

This T7 candidate keeps the exact current-turn root and depth-zero proofs from
`rebound_proof`, then checks the first two complete-turn boundaries below each
root action. After the transposition-table probe, exact component proof runs
only when `turn_ply == 1 || turn_ply == 2`.

Those boundaries cover the opponent's immediate reply and the root player's
counterturn: one full exchange. The placement has no board coordinate, tier,
phase, score, or learned threshold. `Unknown` positions continue through the
original search unchanged.

The comparison runner reports separate ply-one and ply-two probes, Win hits,
Loss hits, and exact cutoffs. In wall-clock mode it also enforces the declared
node budget as a hard safety cap.

T7 is frozen for the next prospective gate. Exposed model-selection evidence
is recorded in `EXPERIMENTS.md`; no result on a new frozen bank is claimed.

Regenerate and verify the paste-ready artifact with:

```sh
node submissions/codingame/tools/generate_submission.mjs exchange_proof
node submissions/codingame/tools/generate_submission.mjs exchange_proof --check
```
