# Mandatory-training Compact Value-BFM campaign

This campaign starts from main commit
`231b5ce0af9171670d68fe5fe022a67f86122cfd`. Its success condition is a newly
trained 6301→12→8→1 bias-free, signed-three-bit scalar candidate completing
90 clean live games, finishing calibration, and scoring strictly above
44.29750553418035. Report rank separately. Offline qualification alone does
not complete the campaign.

The original campaign and recovery evidence remain immutable. V2 has its own
contract, input copies, append-only hash-linked ledger, and per-command
receipts. Protected and live transcripts are not training inputs. The previous
corrected live reference is verified before its legal boundaries are reduced
to fingerprint-only exclusions. A verified operational ending before any legal
turn contributes the empty-board fingerprint only.

## Native data path

`tools/compact_value_bfm_campaign_v2.py` implements the fresh campaign data
adapter. The native continuation producer's opt-in
`--root-policy fresh-complete-v2` plays from the full supplied legal root.
A `-` transcript denotes the empty board in this mode only. The mode retains
short completed games. Default legacy generation is unchanged.

The frozen schedule matches both actor and root-depth marginals. A quarter of
games start empty, for training only. The remaining games start from freshly
rules-generated complete-turn roots with exactly 8, 12, or 20 drawn edges.
Fixed board boundaries do not count. Nonempty canonical roots are assigned
80/20 before continuation generation. Each native game binds the scheduled
root, seed, actor, complete prefix, terminal result, and exact producer/runtime.

Position selection uses absolute drawn-edge progress and tactical evidence,
with at most 20 samples per game. Before teacher search, a rules-only DFS
checks every legal complete-turn successor. A limit or forbidden intersection
rejects the whole group. Validation is processed first. The narrow overlap
exception applies independently to each train parent and successor at at most
six drawn edges, for prior-train/live overlap only. Protected, validation, and
development exclusions never receive that exception. Repeated train examples
are deduplicated; shared prefixes are not held-out evidence.

Teacher labels use the accepted large network at 64,000 fixed-work nodes.
The hardest quarter (rounded upward when necessary) is replaced at 500,000
nodes using teacher/student regret and action/Rank-4/terminal disagreement.
The existing rich-label validator and scalar CSR writer preserve mover-relative
values, exhaustive successor identities, proof evidence, and 75/25
search-backed/terminal targets.

## Execution and resume

Use the research Python environment and a Release native build containing
`papersoccer_jacek_replay_continuations`,
`papersoccer_jacek_replay_search_teacher`, and
`papersoccer_jacek_replay_rank4_position_teacher`.
All campaign outputs belong under the ignored `results/compact_value_bfm/`
directory. Commands require an explicit `--root`.

```sh
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN freeze \
  --previous PREVIOUS_CAMPAIGN --build NATIVE_BUILD
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN games \
  --phase smoke-064 --games 64 --workers 8
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN positions \
  --phase smoke-064
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN labels \
  --phase smoke-064 --workers 8
python tools/compact_value_bfm_campaign_v2.py --root CAMPAIGN train-smoke \
  --phase smoke-064
```

Completed native commands are reused only when their command, input bindings,
and output hashes match. Unprotected producer work without a completion receipt
may restart deterministically; this policy does not apply to protected gates.
A nonblocking file lock excludes concurrent heavy campaign stages. Maximum
workers are eight for generation/labels, two for training seeds, and one for
uncontended timing. Numerical runtimes use one thread each.

Smoke runs all three loss recipes with one seed and the actual warm-up plus
four-epoch QAT producer. Production pilot training requires all three approved
seeds. Smoke data and trained smoke candidates are ineligible for qualification.
Extend the smoke to 128 games if needed for 100 eligible validation groups;
insufficient early coverage must be reported, never scored as a passing gate.

## Qualification still required

The executable smoke path is an implementation checkpoint, not production
admission. Before pilot/full release, complete and verify:

- source-bound upload/maintained-engine comparison with identical weights and
  settings; their differing closure semantics mean this is not a pure speed A/B;
- the 2,000-game pilot, all nine training runs, canonical retention, overall and
  early regret/flip/coverage checks, and the 105/200 Rank-4 screen;
- admitted 10,000-game training and the six-opponent early/late state-capable
  evaluation suite, with frozen deployed controls and paired root bootstrap;
- uncontended profiling, individual search interventions and required strength,
  throughput, latency, and invariant checks;
- the fresh 1,000-game development gate, exact-source CI and preflights, two
  independent protected 1,000-game gates, and source-bound live evaluation.

Diagnostic uploads are already authorized when tied to a useful concrete
experiment and appropriate offline checks. Identical-source reuploads to chase
score variation are forbidden. No recurring automation is assumed.

## Lossless ranking storage

The native smoke exposed substantial rich-JSON memory overhead. The separate
`compact_value_bfm_ranking_store.py` adapter validates every original rich row
with the maintained validator, then stores all successor features, exact
teacher values, identities, canonical transcripts, visit counts, proofs, and
termination evidence in read-only memory-mapped arrays. Group metadata retains
source bindings and work budgets. Nothing is removed from exhaustive groups.

The trainer's source binding distinguishes the v2 storage-index schema from
the existing v1 aggregate schema. Existing v1 loading and validation retain
their original behavior. The runtime schema and network are unchanged. Tests
compare eager and mapped groups, scalar targets, losses, and gradients exactly;
altered cache bytes are rejected. The smoke's 341,685 successors occupy about
154 MB in this representation.

`compact_value_bfm_state_adapter.cpp` provides direct state initialization for
all six local opponents. Empty-board games are explicitly routed to the process
referee. Compile the four `turn_action_v2` opponents with
`CAMPAIGN_OPPONENT_REPLAY_CORRECTION` to retain their source's replay-correction
check; other opponents use their normal complete-turn chooser. Qualification
uses 800/155 ms internal clocks. Reduced-clock adapter tests establish legal
state handling only.

## First native smoke outcome

The first 64-game smoke completed generation, independent isolation preflight,
large-teacher labeling, real training, export, and native checks. It retained
994 groups (829 train, 165 validation) and replaced 249 groups at the deeper
budget. All 64 generated trajectories were distinct; 18 short completed games
were retained. There were 148 comparable validation groups, including 12 early groups. That early sample is small, and the smoke
supports no pilot-admission conclusion.

Each loss recipe ran one float warm-up and four QAT epochs from the frozen
initialization. All three master tensors changed with finite nonzero norms.

| Ranking weight | Changed W1 codes | Changed W2 codes | Changed W3 codes | Canonical sign accuracy |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,070 | 0 | 1 | 85.68% |
| 0.10 | 1,274 | 1 | 1 | 85.39% |
| 0.25 | 937 | 1 | 1 | 85.48% |

Every export is 92,165 ASCII bytes, with 2,835 bytes spare. Exact exports passed
native unit tests and 4,096-state inference parity each. Six true empty-board
process-referee checks, covering both colors for all three exports, completed
without forfeits under 1,000/200 ms external deadlines.

All three smoke models failed canonical retention. None is qualified, no live
success is claimed, and these diagnostic models do not count as completed
production-attempt failures. The 2,000-game pilot must use fresh data, the
original float initialization, all three approved seeds, and its own admission
evidence. `compact_value_bfm_pilot_v2.py --root CAMPAIGN games` freezes that
pilot's context and runs the native generation stage while preserving smoke
exclusions and the campaign-global heavy-stage lock.

## Streaming pilot execution

`compact_value_bfm_stream_v2.py` checks positions in per-game process workers,
with a complete validation barrier before training-root checks. Validation
fingerprints use read-only binary indexes, and each completed game has an
immutable receipt and compressed successor census. The phase index retains
lightweight metadata rather than every successor dictionary.

`compact_value_bfm_labels_v2.py` runs source-bound native label chunks, uses the
maintained rich validators, checks the full preflight successor census, and
preserves canonical rich rows in deterministic gzip files. Source identities
are independently reconstructed in CMake's declared order. Hard selection is
global; incomplete shallow groups are mandatory deep replacements, and no
incomplete group can reach training. Final merging is streamed, and the mapped
ranking store accepts both plain and compressed rich input.

Pilot model selection is frozen before training. Seed selection reuses the
maintained validation key, and both overall and early strata must have at
least 100 comparable groups with at least 80% coverage. Each ranking candidate
must reduce regret by 10% in both strata, stay within the action-flip margin,
preserve canonical accuracy, and retain the source reserve. Selection remains
conditional on a separate 105/200, zero-failure Rank-4 screen.

`compact_value_bfm_pilot_selection_v2.py --root CAMPAIGN --context CONTEXT
--phase PHASE --wait-for-labels train` is a one-shot sequential continuation:
it waits on the shared campaign lock, verifies completed labels, runs all nine
trainings, and freezes offline model selection. It is not a recurring
monitor and does not claim pilot admission or live success.
