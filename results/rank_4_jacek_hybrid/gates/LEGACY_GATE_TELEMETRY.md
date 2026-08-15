# Legacy fixed-work gate telemetry

Status: **nonqualification development telemetry**.

The seven directories below are preserved as immutable historical receipts.
They used only the frozen DEVELOPMENT opening banks, but they do not qualify the
current proof-scope-mask source. They were produced by older gate/source
identities, used a fixed 500-node profile rather than the required clock gates,
and either lack an archived source payload or include the now-rejected root
scout stage. Their clean game counts remain useful development observations;
the label here does not invalidate or revise their raw results.

| Directory | Candidate configuration | Raw result | Why it is nonqualification legacy telemetry |
| --- | --- | ---: | --- |
| `5dd14ee411446c16feb1aa33c610c565b6ccfee9d3ff6d5c4a64095a98bb266c` | control: proof off, scout off vs Rank-4 | 154-152 | Archived 98,690-byte scout-stage source with the scout disabled; its gate and source predate the current proof-mask implementation. Fixed-node control only. |
| `60f090b4c13956f58d3c07e1a978175ecf55237491a2e5068843fea2f54d25de` | proof on, scout on vs Rank-4 | 158-148 | Archived 98,690-byte intermediate with the rejected scout active. The result remains rejection/ablation evidence, not qualification. |
| `8598cbe32eccf8e36efe7951e3b64ef82baca2d6958734fef2a8744825150cfe` | proof on, scout off vs Rank-4 | 181-125 | Preliminary source `be9b789e...27bd` was not archived and the gate/source identity is stale. Selection signal only. |
| `885b116e7292b8a3fb04d2a824bc2e6725966cc1a47e58d6049b9ea23197ceb1` | proof on, scout off vs Rank-4 | 181-125 | Archived 98,690-byte scout-stage source with scout disabled; useful fixed-node selection evidence, but not the current proof-mask source or gate. |
| `bda3156688304a6801ade4511570913066bd9c08985a2894b009dbdfabce9522` | control: proof off, scout off vs Rank-4 | 154-152 | Preliminary source `be9b789e...27bd` was not archived and the gate/source identity is stale. Fixed-node control only. |
| `ca44fb9ef872906819eaf1f687187c94f2d5c4986d457ed68fc0bc340423c94d` | proof on vs same-source hybrid control | 180-126 | Preliminary source `be9b789e...27bd` was not archived and the gate/source identity is stale. Algorithmic isolation signal only. |
| `e0aade22aeec8223dc781bec5cf4365d491c55d0b4ce08d5b68e48eb31b21031` | proof off, scout on vs Rank-4 | 154-152 | Archived 98,690-byte intermediate with the rejected scout active. The result remains scout rejection/ablation evidence, not qualification. |

The `proof_scope_clock/` subtree is intentionally excluded from this legacy
index. It belongs to the current proof-mask qualification workflow and must be
judged by its own content-addressed receipts and exact source/gate identities.

No validation or final opening bank was read to produce this index.
