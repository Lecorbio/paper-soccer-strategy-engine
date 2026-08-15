# Frontier semantic prototype archive

This directory is the immutable correctness/source packet for the refrozen
safe-handoff frontier-width weight-10 hypothesis on rollback commit
`4c9afa70a36183fc1451e4335b7251651a7b2791`.

The control bot is `34b1dd621e894e996df3249b209540fb85f2715f174298bbb1c69b2ec8a69b7b`;
the candidate bot is
`408adc5288674550cc08274aec74380074117e32ad8f6915c7e39badc8dfba98`.
The refreeze confines unique-endpoint marking/counting to leaf calls that
request the semantic value. Root, ply-one, and ply-two null-output proof scans
retain the baseline boolean handoff path.

Primary artifacts:

- `manifest.408adc52.json`: machine-readable identities, evidence, thresholds,
  and no-run/no-heldout attestation;
- `RESULTS.d48e81f3.md`: design, source projection, correctness evidence, and
  the complete deterministic delta panel;
- `DEVELOPMENT_SELECTION_PLAN.9c550604.md`: frozen sequential gate, including
  the rule that Stage 4 executes only d04/d08/d12 and reuses the immutable
  Stage-3 d20 receipt;
- `frontier_weight10.408adc52.apply_patch.base64`: deterministic base64 of the
  repo-relative tool-native patch;
- `frontier_weight10.408adc52.git.patch.base64`: deterministic base64 of the
  Git-format provenance patch;
- `control_bot.34b1dd62.cpp`: exact rollback control bot bytes;
- `frontier_semantic_test.da72af45.cpp`: byte-frozen raw prototype test source;
- `frontier_semantic_test_archive.e50e90db.cpp`: archive-runnable 8/8 Stage-0
  source with literal-weight registration and the active teacher-residual
  witness;
- `null_output_isolation_harness.ce98802e.cpp` and its keyed receipt: 10,040
  mask-1/mask-5 decisions proving exact null-output action/score/stat parity;
- `leaf_proof_microbenchmark.549b6c29.cpp` and
  `run_leaf_proof_microbenchmark.29497387.py`: frozen no-run timing instrument.

Each encoded patch binds both its encoded-file identity and the exact decoded
raw-patch identity in the manifest and results. Decode with `base64 -D` on
macOS or `base64 --decode` on GNU systems. The decoded native patch was
exercised through the actual `apply_patch` tool in a fresh detached `4c9afa7`
clone and produced the exact candidate hashes. The decoded Git-format patch
passes `git apply --check` and was applied in a separate fresh clone on the
same base.

The historical raw focused test intentionally retains its old relative include
and is provenance only; use the separately hashed archive-runnable test for
Stage 0. No bot or generated-source byte is changed by that test wiring.

At archive time no frontier timing benchmark, whole game, heldout
qualification, VALIDATION/FINAL read, arena upload, or production mutation had
occurred. Execution requires the campaign owner's separately granted authority
and the frozen plan's stop-on-failure rules.
