# Conservative safe-handoff frontier proof experiment

This candidate starts from `all_depth_proof`. Its exact rebound-component
Win/Loss checks still run at every complete-turn search boundary. When the
depth-zero check returns `Unknown`, the same scan also supplies the number of
unique fresh handoff endpoints at the component frontier.

The evaluator adds `mover_sign * frontier_count * 5`. Endpoint uniqueness makes
the feature invariant under horizontal reflection, while mover sign makes it
antisymmetric under the 180-degree player rotation. Proven Win/Loss scores and
pruning are unchanged; the term applies only to previously heuristic `Unknown`
leaves.

This is the conservative member of the frontier experiment family. It remains
an experimental candidate rather than a promoted bot. Its exposed-bank
sensitivity results are recorded in `EXPERIMENTS.md`.

Regenerate and verify the paste-ready artifact with:

```sh
node submissions/codingame/tools/generate_submission.mjs conservative_frontier_proof
node submissions/codingame/tools/generate_submission.mjs conservative_frontier_proof --check
```

The focused CMake targets are
`papersoccer_codingame_conservative_frontier_proof_submission`,
`papersoccer_codingame_conservative_frontier_proof_submission_test`,
`papersoccer_codingame_conservative_frontier_proof_timing_probe`, and
`papersoccer_codingame_conservative_frontier_proof_comparison_gate`.
