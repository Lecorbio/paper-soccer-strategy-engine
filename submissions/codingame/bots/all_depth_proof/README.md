# All-depth exact rebound proof experiment

This candidate is the immutable `rank_5` bot plus the exact rebound-component
proofs from `rebound_proof`, applied at every non-root complete-turn search
boundary. A proof can terminate a subtree before complete actions are
enumerated; `Unknown` positions continue through the original search without
an evaluation change.

At depth zero the proof runs before evaluation. At greater depth the
transposition table is probed first, and the proof runs only if no valid cached
cutoff applies. Exact proof scores are deliberately not stored because their
mate-distance value depends on the current turn ply.

This directory is an experimental candidate, not a promoted bot. See
`EXPERIMENTS.md` for the exposed-bank evidence and the required gates.
