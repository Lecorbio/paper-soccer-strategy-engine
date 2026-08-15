# Rotation-equivariant TT placement rejection

Status: **rejected before game gates; never integrated into the hybrid**.

This isolated prototype canonicalized only transposition/evaluation-table
placement while retaining the exact physical 128-bit `PositionKey` as entry
identity. It was developed and tested under
`/tmp/jacek-tt-prototype.hRK0fw`; no prototype source or patch was copied into
the bot namespace. No development, validation, final, arena, or other opening
bank was read or used, and no whole-game comparison was run.

Immutable isolated-artifact identities:

- unified patch SHA-256:
  `9f13547898a898a63e29b5e184d12684f1c75a159731c3dd190cfb3cd0b5ec5a`
- 1,200-state identity/table test SHA-256:
  `717396d736e4cf973b78cf7d6ce6638581960a2a350710f9ae92ca53c870c61a`
- alternating microbenchmark driver SHA-256:
  `f3ac2f8881bdddaa152b449d63d9898e6b1a670322f8344740d9ebe285df920b`

The semantic test passed on 1,200 fresh procedural states and table sizes
`2`, `4`, `8`, and `262144`: the incumbent physical-key formula was unchanged;
physical and rotated placement words swapped exactly; make/unmake restored
both accumulators; rotated exact identities did not cross-hit; both exact
entries coexisted in a two-way bucket; and paired replacement statistics
matched.

The fixed-work rotation panel also passed functionally with provisional exact
proof mask `7`. The physical-placement control retained traversal-stat
mismatches in `2/51` completed depth-one pairs and `1/23` completed depth-two
pairs. Canonical placement reduced both to zero. Both engines had zero action
or score mismatches at those depths and zero action, score, or tracked-stat
mismatches across 200 pairs at each node budget `1`, `16`, `64`, `256`, and
`1024`.

Generated-source projection was `95,995` ASCII characters versus the frozen
`94,004`-character control: a `+1,991` delta and `4,004` characters of total
headroom. This narrowly satisfied the `+2,000` source-delta cap.

Timing rejected the hypothesis. After 20 warm-up pairs, 300 alternating
constructor-inclusive fixed-20,000-node pairs all returned the identical
signature `(nodes=20000, score=15097, action_edges=22)`. Control median/p99
were `29.5960005/30.886042 ms`; prototype median/p99 were
`29.971438/31.114125 ms`. Distribution slowdown was `+1.268541%` at the
median and `+0.738466%` at p99; paired-ratio slowdown was `+1.245471%` at the
median and `+4.032930%` at p99. The frozen bounds were at most `+0.5%` median
and `+1.0%` p99, so the prototype failed before any game gate.

Conclusion: symmetry behavior improved, but the resource contract did not
pass. The exact control bytes remain selected; canonical table placement must
not be integrated or submitted from this experiment.
