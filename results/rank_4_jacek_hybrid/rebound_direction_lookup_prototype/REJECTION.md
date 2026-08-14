# Rebound direction-rank lookup: rejected

This was an isolated `/tmp` prototype against exact rollback HEAD
`f287efe788c70e3d26debca2c2677212132e784a`. It replaced the rebound scan's
eight-way mover-relative direction search with the constant rank table
`{7,6,5,0,0,4,1,2,3}` indexed by `3*(dx+1)+dy+1`, after the existing
Player-Two coordinate inversion. The frozen identity was evaluated once and
stopped on its first gate miss; no variant or retry was run.

## Bound identities

- Control `bot.cpp`: 63,107 bytes,
  `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`.
- Candidate `bot.cpp`: 62,945 bytes,
  `b45a79a5a61dae252a17191d1b823791ed74f1ed0d939ebdfa2e5917076a499d`
  (162-byte shrink).
- Deterministic labeled unified patch:
  `863eb2505306896f99c9ec5d9462cd74e600d16b15a960d1fffab5cae6ef0f8b`.
- Control generated source: 94,312 ASCII characters,
  `2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
- Candidate generated source: 94,225 ASCII characters,
  `bd045e4812bdff31bcc7071fbc8db0f640eac2f7fda4af772cad0fae93103fa2`
  (87-character shrink).
- Unchanged production test:
  `ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697`.
- Component harness:
  `6a4986e09c705b823d641edb71bffa405471245800e9a652259fab9523ace9f6`.
- Fixed-work parity harness:
  `9e2ebd21c90fa4e93020efc6d5c7cea4c9d6cac84c4759711c3d81dea572af17`.
- Timing harness:
  `19bc916e44352eed7f6a05f47149be3cb5577a707fb81a76124c5274564dc6b4`.
- Locked alternating runner:
  `2e72b84a69a022722e4f64d58dc7fb4ec4e75f14311a3069bac2115d39f658d3`.

## Exact correctness gates

- Component gate: byte-identical outcomes, reconstructed paths, and proof
  counters over eight tactical plus 10,000 deterministic procedural
  components. Colors were balanced at 5,004/5,004 including tactical pairs;
  results were 7,756 Unknown, 2,248 Win, and four Loss, with 2,248 nonempty
  paths. Both streams had SHA-256
  `87f2900b76651b8a4bcd4427e30ab54d14ef7808ec94e78a8ba7fc1d2ffacb4c`.
- Fixed-work gate: byte-identical encoded action, root score, and every
  `SearchStats` field for 1,004 deterministic live states, proof masks 0 and
  7, and node budgets 1, 16, 64, 256, and 1,024: 10,040 paired decisions.
  Both streams had SHA-256
  `99f09e979bc5e4b25e5878880d9b7a9acd83e3a6870adba2b8e3be9fc306bd42`.

## Frozen timing gate and decision

The gate required each production-table panel's candidate/control median
ratio to be at most 0.995 and p99 ratio to be at most 1.005. Each panel used
30 warm-up plus 300 measured alternating process-order pairs. The runner held
an exclusive `fcntl.flock` on
`/tmp/rank4-hybrid-prototype-benchmark.lock` for the entire panel (lock waits
0.000014 s forced and 0.000011 s mixed).

- Forced-heavy signature `(50000,15016,48,12774,0,8634)`: control
  median/p99 18,370,354.5/21,177,250 ns; candidate
  18,171,562/20,529,625 ns. Distribution ratios were median 0.989178625 and
  p99 0.969418834: pass.
- Mixed signature `(50000,4109,2,163,4,4753)`: control median/p99
  88,903,687.5/92,743,875 ns; candidate 88,202,146/93,821,958 ns.
  Median ratio 0.992108972 passed, but p99 ratio **1.011624304** exceeded
  1.005: fail.

Decision: reject permanently on the frozen mixed-panel p99 miss. No whole
games were played, no DEVELOPMENT/VALIDATION/FINAL bank was read, and no arena
operation occurred. Apart from this evidence report, shared production files
were not changed.
