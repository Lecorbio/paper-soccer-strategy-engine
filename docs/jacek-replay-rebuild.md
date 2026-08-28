# Evidence-gated replay rebuild

`tools/jacek_replay_rebuild.py` is a local-only research ladder for finding a
replay-BFM model that combines the v5 pilot's playing strength with the v6
pilot's canonical retention.  It does not change either completed campaign,
regenerate their teacher labels, replace Rank-4, or upload a model.

## Frozen inputs

The rebuild corpus binds all canonical R0/R1/R2 train, validation, and test
shards plus the v5/v6 search, Rank-4, and adjudicator shards.  Search, Rank-4,
and adjudicator rows are deduplicated by the rotate/reflection-canonical feature
fingerprint, with v6 taking precedence over v5.  Materialized CSR shards are
used by training; the canonical test shards are exposed only through a
protected-test interface.

Before training, the workflow also freezes:

- a development opening bank and a disjoint sealed final bank;
- 600 whole procedural Rank-4-vs-Rank-4 root groups with exactly 20 positions
  per group;
- the clean producer commit, comparison/teacher binaries, incumbent runtime,
  corpus, candidate matrix, and the 24-hour same-architecture deadline.

The blind holdout receives strict 400,000-node Rank-4 labels only after a
selected-runtime receipt exists.  It is training-incompatible and requires
both point and root-cluster noninferiority.

## Candidate ladder

The first phase recovers the paired v5 search/Rank-4 models across three layer
freeze masks and learning rates `3e-6`, `1e-5`, and `3e-5`.  The second phase
uses all nine saved canonical seed runtimes as paired bases.  If time remains,
six seeds are pretrained from random initialization on canonical train and
validation only; the protected test is never loaded.  The three strongest
operational bases receive joint search/Rank-4 training.

Every joint phase uses 64 new and 192 anchor rows with separately normalized
0.25/0.75 Huber loss and requires complete new/anchor coverage.  The v5 recovery
phase uses fixed checkpoints every 782 updates and at most two anchor passes.
Canonical-basin and scratch-joint phases instead use the frozen v6
retention-safe epoch schedule: up to 50 epochs, patience 8 after complete anchor
coverage, a fresh new-data permutation each epoch, and a continuous anchor
stream.  Canonical retention is measured against the incumbent.  v5 recovery
is also noninferior to v5 on the new adjudicator; other bases must improve epoch
zero.

At most six offline survivors receive 100-pair incumbent/matched screens.  The
top two receive all unchanged 300-pair pilot panels.  The ladder stops at the
first development-qualified phase.  If no same-architecture model qualifies,
the sole fallback freezes the paired v5 bases and trains the version-2
rank-16 residual adapter at `1e-4`, `3e-4`, and `1e-3`.

## Runtime version 2

Version 1 remains byte- and inference-compatible.  Version 2 appends a base
gain, residual bias, a `192 x 16` adapter matrix, and a 16-element output
vector.  With gain one, bias zero, and output vector zero, its predictions are
exactly the version-1 base predictions.  Only the gain, bias, and adapter
parameters train; all base tensors remain bit-exact.

## Qualification

A candidate is selected before any protected evidence is opened.  Final
qualification conjunctively requires canonical-test retention, the procedural
blind holdout, the sealed four-panel 600-game bank, zero illegal/unfinished
games, p99 at most 25 ms, and uncontended maximum below 1,000 ms.  Primary
panels require 325 wins and 156 per color; external panels require 306 and 143.

A passing runtime is published only as a local research incumbent with a
provenance-bound v7 starting-actor receipt.  A fresh 2,000-game pilot and
conditional 10,000-game campaign remain mandatory before any later promotion.
