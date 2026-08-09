# Safe-handoff frontier proof experiment

This candidate starts from `all_depth_proof`. Its exact rebound-component
Win/Loss checks still run at every complete-turn search boundary. When the
depth-zero check returns `Unknown`, the same scan also supplies the number of
unique fresh handoff endpoints at the component frontier.

The evaluator adds `mover_sign * frontier_count * 10`. The weight is one half
of the existing mobility weight. Endpoint uniqueness makes the feature
invariant under horizontal reflection, while mover sign makes it antisymmetric
under the 180-degree player rotation. Proven Win/Loss scores and pruning are
unchanged; the term applies only to previously heuristic `Unknown` leaves.

This directory is an experimental candidate, not a promoted bot. Its current
evidence is an exposed-bank sensitivity screen documented in `EXPERIMENTS.md`.
Fresh T8 evidence remains unconsumed and the candidate is not bound to it.
