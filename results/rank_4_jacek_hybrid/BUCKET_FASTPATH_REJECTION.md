# Power-of-two cache bucket fast path: rejected

This is the immutable rejection receipt for the isolated power-of-two bucket
prototype. It was evaluated against exact rollback commit
`e6fc14b6adc7ca8d1d0ff515b9519da60cc0c217` with operational proof mask 7.
No production source was edited or integrated.

## Bound identities

- Control `bot.cpp`: 63,107 bytes,
  `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`.
- Control generated source: 94,312 ASCII characters,
  `2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
- Candidate `bot.cpp`: 63,732 bytes (+625),
  `a45581da15d53a53a6fcb9bec5cd5de623db6582014fcb385d0156dd0b0b9fb4`.
- Candidate generated source: 94,861 ASCII characters (+549),
  `ec4469601a93f59e96135021d03b8bccebfc043b9301569a86f7e3f1ebc3a5c3`.
- Exact patch: 2,482 bytes,
  `908d56457dee292abd70c1c1361913d78b0c26a075ff6d43228ab336421be0ba`.

The patch used one shared `TwoWayBucketReducer` with a cached mask for
power-of-two bucket counts. Bucket count one and every non-power-of-two count
retained the original `combined % buckets` reduction.

## Semantic evidence

- Arithmetic harness `bucket_parity.cpp`:
  `319ea3fe74ac279bd7a3171f1a33015af2fc8e8914df53ee55f28b64a850fdb6`.
  It matched 1,000,360 reducer results to the legacy modulo expression.
- The same harness matched 800,000 transposition-table and 800,000
  evaluation-table store/find operations, including irregular capacities and
  production capacities 131,072 and 262,144.
- Fixed-work harness:
  `12d4db655376df1c3e3556615a1f7562137539448ecd8eb58dab6767bae9842f`.
  Across 1,004 live states, masks 0 and 7, and budgets 1/16/64/256/1,024,
  all 10,040 decisions had byte-identical actions, root scores, and every
  emitted search statistic. Control and candidate stdout were each 1,067,064
  bytes with SHA-256
  `99f09e979bc5e4b25e5878880d9b7a9acd83e3a6870adba2b8e3be9fc306bd42`;
  stderr was also identical with SHA-256
  `f20d99fe7924f98b29c366e2494da2ac2681641d74b32e0c4fcdeb85371e327a`.
- Microbenchmark harness:
  `2007e0daee4282ee6924aa43bbb8ffec3844a6ab845d2db3e5e79c6634837510`;
  runner:
  `6cb1f305051f040dec77b52381f22f63276b8d96213a32a052fe6a9f7c21d116`.

## Frozen production-table timing gate

Each panel used 30 warm-up pairs and 300 measured pairs in alternating order.
Exact result signatures were required. Per-panel limits were median ratio
at most 0.995 and p99-distribution ratio at most 1.005.

| Panel | Control median / p99 (ns) | Candidate median / p99 (ns) | Median ratio | p99 ratio | Result |
|---|---:|---:|---:|---:|---|
| Forced-heavy | 18,353,438.0 / 20,100,667 | 18,091,166.5 / 19,943,542 | 0.985709953 | 0.992183095 | pass |
| Mixed | 88,720,979.5 / 93,320,250 | 88,369,187.5 / 93,148,500 | 0.996034850 | 0.998159563 | **fail** |

The forced-heavy signature was `(50000,15016,48,12774,0,8634)` and the mixed
signature was `(50000,4109,2,163,4,4753)` for both binaries. Raw panel output
hashes were
`8ebf0f7910db95c74c4861e195856a6562e5f1f83d2858667692dc0fc8bae836`
and
`70ab7c0de386bf0d773e6c4a13bfa9a2b6f3872078bba87e233f516f5841876f`.
Measurements used Apple clang 21.0.0 on arm64 macOS.

## Decision

**Reject and do not integrate.** The mixed-panel median ratio 0.996034850
exceeded 0.995, and the generated-source delta +549 exceeded the +512
prototype cap. The experiment stopped at the frozen gate. It read no held-out
or protected opening bank and ran no whole-game or arena matches.
