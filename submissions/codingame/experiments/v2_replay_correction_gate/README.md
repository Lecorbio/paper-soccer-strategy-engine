# Accepted V2 replay correction: compact archive

This folder is a historical evidence archive for the one replay correction
accepted into the live complete-turn V2 bot. It intentionally does not contain
the old runners, adapter tests, compiled files, frozen reference engines, or
rejected experiment implementations.

## Historical artifact

The submitted artifact was named `paper_soccer_v2_replay_book.cpp`, contained
64,818 characters, and had SHA-256
`e79390a1833d9b9b28a22d7ff8a662bdb32d77c57b6d24152e0a888aef0cb66c`.
The byte-exact file is not retained because it contained an attribution banner
disallowed by the repository's public-file policy. The maintained
`../../paper_soccer_alpha_beta.cpp` is behavior-equivalent, but its consolidated
comments make it byte-distinct from the submitted artifact.

The live result was rank 8 of 206 with score 41.65, compared with rank 13 and
37.67 for the preceding pure V2 submission. The five inspected public replays
did not reach the exact correction transcript, so this is observational evidence
for the combined candidate, not causal evidence that the correction activated.

## Frozen gate result

The gate evaluated four exact Player-0 replay states against V2, C9,
selective-H16, and rank10 references at 1,000,000 and 3,000,000 nodes. Each
proposed action had to win at least 6/8 cells, improve by at least two wins, and
avoid losing any cell won by the recorded action.

Only state `60efd6e1ae7043a4` passed. Its recorded action `42474177` scored 0/8;
the accepted action `42474176` scored 8/8. The exact prior transcript was:

```text
7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474
```

The other three entries were rejected independently and are not present in the
production lookup.

## Retained files

- `benchmark_results.json`: complete summarized gate outcomes and historical
  artifact metadata.
- `evaluation_freeze.json`: criteria and pre-result experiment definition.
- `live_arena_result.json`: the recorded rank-8 Arena observation.
- `preflight_results.json`: deterministic validation summary.
- `reports/exact_replay_screen.json`: the complete frozen 64-game raw report.
- `reports/gate_evaluation.json`: concise per-entry verdicts.

These files preserve the decision and its evidence. They are not intended to
recreate the removed experimental toolchain.
