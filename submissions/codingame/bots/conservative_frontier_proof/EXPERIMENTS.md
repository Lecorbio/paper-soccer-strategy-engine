# Conservative safe-handoff frontier proof ledger

## Hypothesis

`all_depth_proof` proves goals and closed-component losses exactly but treats
every safe handoff as the same `Unknown` outcome. The frontier term distinguishes
those unresolved states by counting unique reachable fresh endpoints where the
opponent will retain at least one legal edge after the incoming edge is used.

The only strategy weight is 5. A frontier vertex is marked after its first scan,
so converging component arcs cannot count it twice. The count is applied with
mover sign only after `Unknown`; exact Win/Loss behavior and proof scores are
unchanged.

## Exposed development sensitivity screen

The fixed-node development screen used 48 already exposed openings, two
candidate colors plus a same-opening `rank_5` control, and four workers/shards.
At 5,000 nodes the candidate won 54-42 games, with source-game-cluster estimate
0.55929 and candidate-to-incumbent throughput ratio 1.08549. At 30,000 nodes it
won 50-46, with cluster estimate 0.55769 and throughput ratio 1.11947. Both
development profiles passed their predeclared floors.

## Exposed final sensitivity screen

The time screen used the already exposed 72-opening final bank, four
workers/shards, a construction-inclusive 130 ms steady-clock budget, a
3,000,000-node safety cap, and 320 maximum turns. The candidate won 78-66 games.
It cleared the strength, uncertainty, color, stratum, and tier floors but
retained 53/72 control winners (0.73611), below the 0.75 retention floor. The
screen was therefore rejected on that single requirement.

These are bounded sensitivity screens on exposed evidence, not unbiased
promotion results. No fresh evidence is claimed for this candidate.
