# Rank-three campaign result

The released source reached third place out of 211 entrants after one completed
CodinGame calibration. A separate, frozen-source confirmation then passed all
predeclared gates. The final live observation still showed third place.

## Design and selection

The bot searches complete turns, including rebound chains, with alpha–beta
search and a small quantized neural evaluator. Terminal proofs are separate
from heuristic values. An exact flat goal-distance traversal reduces evaluator
work without changing its values. The source uses 550 ms initially and 140 ms
thereafter, below the platform's 1,000/200 ms hard limits.

The controlled matrix compared the corrected parent, reflection averaging,
exact traversal, and their combination, with historical Rank-4 as the external
control. All four used the same corrected model and 550/140 ms clocks.

| Development comparison | Win-rate difference | 95% root-cluster interval |
| --- | ---: | ---: |
| Selected exact-traversal source versus historical control | +16.52 pp | +11.61 to +21.43 pp |
| Exact traversal versus corrected parent | 0.00 pp | −4.02 to +4.02 pp |
| Reflection versus corrected parent | −3.35 pp | −7.81 to +1.12 pp |
| Combined versus corrected parent | −2.23 pp | −6.47 to +2.23 pp |

Development used 224 paired roots and 2,240 unique games, including the pilot,
with zero operational failures. The combination did not establish a benefit;
the condition for a mirror-trained student was not met. The selected source's
overall improvement cannot be attributed to traversal alone.

The earlier 650 ms prototype failed a cold-start safety case at 1,018.617 ms.
A bounded diagnostic identified startup overhead, and a uniform 550 ms
successor was tested across all four arms. The failed receipt was retained.
The successor passed 20 safety cases and eight full native games across the
matrix. The selected kernel passed 327,904 distance comparisons, 163,952
feature/evaluator bit comparisons, 5,620 make/unmake checks, 114 fixed-work
comparisons, 22 legal-fallback checks, and 120 terminal-proof checks, plus
address/undefined-behavior sanitizer checks.

A fixed private platform panel finished 7–5 with no operational failures.
It tested six opponents in both colors. These repeated opening positions are
descriptive diagnostics, not twelve independent strength samples. The eight
remaining opening-probe slots were also completed, preserving the four
previously spent slots and keeping different source cohorts separate.

## Live calibration

Agent `6764344`, submission `41384976`, completed 90/90 archived games at
rank 3/211, score 43.36, and 100% calibration. There were 79 clean rule-terminal
games (52 wins, 27 losses), 11 opponent operational failures, and zero own
crashes, illegal moves, or timeouts. Opponent failures are excluded from the
clean playing record.

Source identity is supported by an independent editor copyback, its SHA-256,
one arena activation, and the resulting agent/submission IDs. CodinGame does
not provide a remote source digest. No identical-source resubmission was used
to seek a more favorable ranking. The final check at 08:31 UTC on 2026-09-27
found the same source in the editor, the same active submission, the same 90
game IDs, and rank 3/211. Rankings can subsequently change.

## Fresh confirmation

The source was frozen before reserving 100 fresh paired roots for each of
seven frozen opponents. Each root ran candidate and control in both colors,
with alternate arm order: 700 independent roots and 2,800 unique games.
Prefix-derived deployment clocks were preserved. No outcomes were inspected
for tuning while the run was active.

The candidate won 1,024/1,400 games (73.14%); the control won 757/1,400 (54.07%).
The paired uplift was **19.07 percentage points**, with a 10,000-replicate
root-cluster bootstrap 95% interval of **16.21 to 21.93 points**.

| Frozen opponent | Paired uplift |
| --- | ---: |
| Rank-4 | +16.5 pp |
| Compact deployed | +21.5 pp |
| h62 | +16.0 pp |
| Rank-4 hybrid | +30.5 pp |
| Rank-4 full-turn BFM | +11.5 pp |
| Challenger | +23.0 pp |
| Neural PUCT | +14.5 pp |

All predeclared confirmation gates passed: positive lower 95% bound, no
opponent regression worse than five points, and zero operational failures.
The three-point minimum applied to development, not as an added confirmation
gate. The final audit independently replayed and checked timing for all 2,800
saved games, verified their claims/exposure receipts, and recomputed the
assessment. No games were rerun for the audit.

Freshness was enforced by resuming the exposure census to 23,363/23,363 records,
then adding explicit exposure deltas for subsequent local and platform games.
The old failed source's 700-root bank was not reused. The new bank was screened
against exposure, learning, and prior-development inventories and reserved to
this exact source. Reflections did not create extra independent samples.

## Reproducibility and retained evidence

[README.md](README.md) contains the exact source hash and C++20 build command.
The release was rebuilt successfully with Apple clang 21.0.0. The editor,
qualification source, and release source were byte-identical. The public JSON
files preserve source identities, live game transcripts and classifications,
and every confirmation root's paired outcomes and receipt hashes.

Detailed local evidence remains under `results/top_three_20260927/`: immutable
producer snapshots, the resumable queue and census receipts, safety checks,
development and platform reports, original calibration archive, fresh bank,
per-game claims/results/exposures, and the final audit. These bulky research
artifacts are retained locally; this release does not claim that the compact
public evidence alone reproduces the entire training or research campaign.

The queue detects stale owners, PID reuse, changed inputs, resource limits,
and interrupted claims; completed units are not silently replayed. Confirmation
ran with a peak of four processes and approximately 2.02 GB recorded RSS.
The original dirty research work and failed attempts were preserved.
