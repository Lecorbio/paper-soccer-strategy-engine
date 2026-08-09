# Exact rebound-component proof ledger

## T3: pure exact proof

T3 starts from immutable `rank_5` and contains exactly 85 insertions: one
outcome enum, three counters, a root current-turn goal proof, one component
scan, and exact depth-zero win/loss scores. Replay data, evaluator, move
ordering, transposition semantics, and clocks are byte-identical to the
incumbent.

The generated source is the independently reconstructed historical artifact:
96,306 characters, SHA-256
`e58fce3bf1314f480f76160471f7f7140fedfff39f71ffbe45c04c4ba4f0595f`.
Twelve artifact tests pass, including both colors, visited versus fresh goal
mouths, used edges, boundary-post continuation, own-goal rejection, closed
component loss, exact action replay, and depth-zero integration.

## Development result

T3 passed the predeclared two-horizon development gate on 48 paired openings:

- 5,000 nodes: **51-45**, clustered score `0.559295`, 95% interval
  `[0.483974, 0.639423]`, throughput `1.0836x`.
- 30,000 nodes: **55-41**, clustered score `0.581731`, 95% interval
  `[0.519231, 0.650641]`, throughput `1.1504x`.

The 30k elite-winner cohort scored `0.500000`; the incumbent-winner cohort
scored `0.594595`.

## Validation result: rejected

On 68 paired rank-one validation openings at 5,000 nodes, T3 won **72-64**.
Its clustered score was `0.535985`, elite score `0.529412`, and throughput
`1.1089x`. However, the result was strongly color-skewed: `0.661765` when the
candidate controlled the historically winning side and only `0.397059` when
it controlled the defender. The clustered interval lower bound was
`0.458333`.

Those values fail the frozen `0.48` minimum-color and `>0.47` interval-lower
requirements. The overall win is therefore not promotion evidence. The fresh
final test, timing gate, and live Arena were not run. T3 is `REJECT` under
manifest SHA-256
`1062389ed0053d12274e203da19ad4006f5fc9b3274bae82d62ace4fe5523ce4`.
