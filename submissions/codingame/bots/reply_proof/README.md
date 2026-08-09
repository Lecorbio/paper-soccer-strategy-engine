# First-reply exact rebound proof

This T6 candidate starts from the exact `rebound_proof` implementation. It
keeps the unconditional current-turn root proof and depth-zero proof, then
adds one TT-first proof at the first complete-turn boundary below each root
action (`turn_ply == 1`).

The placement asks a semantic tactical question: does this candidate action
hand the opponent a forced rebound-component Win or Loss? It has no board
coordinate, phase, degree, or learned threshold. `Unknown` positions continue
through the original search unchanged.

The candidate is frozen for prospective validation. The exposed-bank evidence
and decision constraints are recorded in `EXPERIMENTS.md`; no sealed final-test
result is claimed here.

Regenerate and verify the paste-ready artifact with:

```sh
node submissions/codingame/tools/generate_submission.mjs reply_proof
node submissions/codingame/tools/generate_submission.mjs reply_proof --check
```
