# Sole-legal-edge bypass rebased on operational rollback

Scope: isolated `/tmp` experiment only. No workspace edit, protected bank,
held-out/full game, or arena evidence was read or produced.

## Frozen inputs

- HEAD: `c6377824feae6c21df6a0753713e690afc9bb31c`.
- Control bot: 63,107 bytes,
  `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`.
- Control generated source: 94,312 ASCII characters,
  `2293bc87d022e97301cdd0e86db35ea168100b9d1e800be4dc7583bbedfb52e7`.
- Shared search header:
  `0a13e89e183666ce89e38d1eded1b26c02eaaba5460ba7e3ede9fda5d5e1dd04`.
- Candidate bot:
  `2103363a7c121d2252110617791df2ca868ccb0ca611914fb75001e3af1fdff5`.

## Fixed-work exact parity

- Corpus: 1,000 deterministic procedural live states plus four tactical
  transcript states, 1,004 total; 18 natural singleton roots.
- Matrix: exact-proof masks 0 and 7 at node budgets 1, 16, 64, 256, and
  1,024, for 10,040 paired decisions.
- Compared encoded action, root score, and every emitted `SearchStats` field.
- Both stdout streams were byte-identical: 1,067,064 bytes,
  `99f09e979bc5e4b25e5878880d9b7a9acd83e3a6870adba2b8e3be9fc306bd42`.
- Both exited zero. Singleton heuristic nonzero scores were 18 in control and
  zero in candidate, proving the bypass executed.
- Harness:
  `12d4db655376df1c3e3556615a1f7562137539448ecd8eb58dab6767bae9842f`.

## Alternating construction-plus-50k-node microbenchmarks

Each panel used 30 warm-up pairs and 300 measured pairs, alternating process
order and requiring exact result signatures.

- Production tables, forced-heavy signature
  `(50000,15016,48,12774,0,8634)`: control median/p99
  17,600,854/18,832,500 ns; candidate 17,350,292/18,720,334 ns. Candidate
  delta: median -1.423579%, p99 -0.595598%.
- Production tables, mixed signature `(50000,4109,2,163,4,4753)`: control
  median/p99 84,917,417/90,464,625 ns; candidate 84,987,208/89,575,333 ns.
  Candidate delta: median +0.082187%, p99 -0.983027%.
- Small tables, forced-heavy signature `(50000,15016,48,12757,0,8062)`:
  candidate delta median -1.359808%, p99 +0.294924%.
- Small tables, mixed signature `(50000,4109,2,171,4,3114)`: candidate delta
  median +0.098041%, p99 +0.320376%.
- Microbenchmark harness:
  `2007e0daee4282ee6924aa43bbb8ffec3844a6ab845d2db3e5e79c6634837510`.
- Runner:
  `48c3c5a468bb086c6dc55d2d0bc529741ee3eaedbba8f753afa7af9503dbaa21`.

The prior prototype contract required at least 0.5% forced-heavy median
speedup and no more than 0.5% mixed median or p99 regression. Both production
and small-table panels pass.

## Projection and conclusion

The one hunk adds 145 generated characters. Projected submission: 94,457 ASCII
characters, SHA-256
`cf031b73e430b27285c79c0d3e2f90ec056b10aa21e52b722f71e554686839ff`,
leaving 5,542 characters below 99,999. Exact fixed-work semantics pass and the
operational production-table throughput result supports a DEVELOPMENT-only
actual-clock comparison against the exact rollback control.
