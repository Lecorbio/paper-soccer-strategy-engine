# TT entry layout compression: immutable rejection receipt

Classification: DEVELOPMENT-only isolated prototype. This experiment read no
validation or final bank, played no whole game, produced no arena evidence,
and made no production-source change.

## Frozen gate

The candidate had to satisfy every condition before any game comparison:

- exact store/find and fixed-work search parity;
- generated-source delta at most 512 characters;
- `sizeof(TranspositionEntry) == 32` instead of 40;
- in both production-table 50,000-node panels, candidate/control median ratio
  at most 0.995 and p99 ratio at most 1.005.

The alternating timing experiment was one shot: 30 warm-up pairs and 300
measured pairs per panel, with an exclusive flock held on
`/tmp/rank4-hybrid-prototype-benchmark.lock` for its full 74.874 seconds.

## Frozen identities

- Base commit: `f287efe788c70e3d26debca2c2677212132e784a`.
- Control `bot.cpp`: 63,107 bytes, SHA-256
  `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`.
- Control generated source: 94,312 ASCII characters, SHA-256
  `2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
- Control/candidate submission test: 39,137 bytes, SHA-256
  `ba5c8e25ac3d446558e4be4ed4a41993dd2bfaac9cd05dd13677617f445bf697`.
- Candidate `bot.cpp`: 63,383 bytes, SHA-256
  `fd5413d86fd27f0ba403204ddfaeabdb5401fd439edf5c35afe4fd8fffbfd90c`.
- Candidate generated source: 94,533 ASCII characters, SHA-256
  `bb59326f9265d83fa493eaf4bbcc33739db5c6b9f8459ee4efe5649ed8f89304`.
- Maintained-bot diff SHA-256:
  `361f1152c6ee18eba126a0d1f8f9b23ddbfa2b3481f439428a1b8d4d291baf76`.
- Generated-source diff SHA-256:
  `aaf19c2b04c95d651cf63e0d684b1fe9848399a56e92f12c06e80a774705ed94`.

The generated-source delta was +221 characters, within the +512 limit.

## Candidate layout and depth safety

The prototype changed `ScoreBound` to an explicitly `uint8_t`-backed enum and
laid out each entry as the 16-byte key, 8-byte move, 4-byte score, 1-byte depth,
1-byte bound, and occupied flag. Static assertions proved the configured
maximum turn depth is exactly 32, fits in `uint8_t`, and the packed entry is
32 bytes. The store boundary accepted depth 255 and rejected 256 and
`UINT32_MAX` without mutating the table.

## Exact-parity evidence

- Table harness SHA-256:
  `2937b2254b98b8d4a0d1abd72f9a4f9fbd3a00549928f8e97f7d00b0f7c4fb93`.
- Table result: exact logical entry contents, lookup results, store results,
  bucket mapping, replacement choice, best move, bounds, scores, and depth
  across 186,624 exhaustive four-operation traces and 1,150,000 randomized
  operations. Sizes covered 0 through small/non-power-of-two tables, 4,096,
  and the production 262,144 entries. Entry size was 40 control versus 32
  candidate; bound size was one byte.
- Fixed-work harness SHA-256:
  `12d4db655376df1c3e3556615a1f7562137539448ecd8eb58dab6767bae9842f`.
- Fixed-work result: 1,004 deterministic live states, proof masks 0 and 7,
  and node budgets 1, 16, 64, 256, and 1,024: 10,040 paired decisions. Encoded
  action, root score, and every emitted search-stat field were byte-identical.
  Each stream was 1,067,064 bytes with SHA-256
  `99f09e979bc5e4b25e5878880d9b7a9acd83e3a6870adba2b8e3be9fc306bd42`.
- Both control and candidate passed all 18 submission tests.

## Locked timing result

Microbenchmark harness SHA-256:
`2007e0daee4282ee6924aa43bbb8ffec3844a6ab845d2db3e5e79c6634837510`.
Runner SHA-256:
`5cd2cd2cece5a06f46893249ef06b9ffde204c95bd09de6697bc40af5c0fbd92`.
Every control/candidate result signature matched exactly.

- Forced-heavy signature `(50000,15016,48,12774,0,8634)`: control
  median/p99 18,429,604/20,633,083 ns; candidate
  18,034,124.5/20,841,375 ns. Ratios were 0.978541074 median and
  **1.010095050 p99**.
- Mixed signature `(50000,4109,2,163,4,4753)`: control median/p99
  89,136,541.5/101,693,333 ns; candidate 88,330,000/99,995,250 ns. Ratios
  were 0.990951618 median and 0.983301924 p99.

## Decision

**Rejected before games.** The forced-heavy p99 ratio 1.010095050 exceeded
the frozen maximum 1.005. The other three timing thresholds, all structural
checks, and all exact-parity checks passed, but the gate was conjunctive. No
retry, gameplay gate, held-out inspection, arena upload, or production edit
was performed for this candidate.
