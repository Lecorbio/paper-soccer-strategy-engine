# Safe-handoff frontier proof ledger

## Hypothesis

`all_depth_proof` proves goals and closed-component losses exactly but treats
every safe handoff as the same `Unknown` outcome. The frontier term distinguishes
those unresolved states by counting unique reachable fresh endpoints where the
opponent will retain at least one legal edge after the incoming edge is used.

The only strategy weight is 10, predeclared as one half of the existing
mobility weight. A frontier vertex is marked after its first scan so converging
component arcs cannot count it twice. The count is applied with mover sign only
after `Unknown`; exact Win/Loss behavior, move ordering, replay data,
transposition behavior, and time budgets are unchanged.

## Exposed T7-final sensitivity screen

The screen was frozen before execution and used only the already exposed T7
final bank: 72 openings, two candidate colors plus a same-opening
`rank_5`-versus-`rank_5` control, four workers/shards, a construction-inclusive
130 ms steady-clock budget, a 3,000,000-node safety cap, and 320 maximum turns.

The candidate won 82-62 games. Opening pairs were 17 sweeps, 48 splits, and 7
losses. The source-game-cluster estimate was 0.56915 with 95% interval
[0.50000, 0.63830]. Control-winner retention was 57/72 (0.79167). Physical
uplifts were +0.06944 for both colors; historical winner/opponent uplifts were
+0.09722/+0.04167. Scores by goal distance were d0 0.50000, d1 0.50000, and d2
0.69231; elite/field tier scores were 0.65789/0.53774. Every operational count
was zero. The screen passed every predeclared exposed-final floor.

The frozen screen protocol SHA-256 was
`c7ab0bdcdb393585bb1e5ad9ac3d645beb6a4e81bf86945420d36c51562b1426`;
the full aggregated report SHA-256 was
`cb7048b723c5d70944946901362784fced394cae470eca56c5f5eb7fab7223f8`.

This result is a bounded sensitivity screen on exposed evidence, not an
unbiased promotion result. Do not infer a T8 outcome from it and do not consume
T8 validation or final evidence until candidate binding is explicitly frozen.
