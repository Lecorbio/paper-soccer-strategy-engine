# Full-turn best-first minimax experiment

## Frozen design and gates

The numerical promotion gates were frozen on 2026-08-10 before any
candidate-versus-rank-4 match was run. The corrected implementation formulas
below were re-frozen after focused correctness review; only comparisons rerun
from the final generated-source hash count as evidence. The starting revision
was `c7516350e1017dd5d4ec986eb70ab6cd8fe440e1`.
The maintained incumbent is the byte-identical
`../rank_4/submission.cpp`, 98,624 characters, SHA-256
`5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9`.

### Directly reported by Jacek

- One tree action covers the mover's entire turn, including every rebound.
- A position can have thousands of such actions, so at most 250 are retained.
- Action generation is iterative and uses a deque mostly as a stack, with
  roughly every tenth extraction from the other end and randomized neighbour
  order to obtain a diverse capped sample.
- Every newly expanded complete-turn child is evaluated by the neural value
  function; values are propagated in minimax style.
- Frontier exploration uses Best-First Minimax Search with UCT. The referenced
  article specifies selection by `eval + C * sqrt(log(parent visits) / visits)`,
  unvisited selection by `eval + FPU`, expansion of every child, negamax-style
  overwrite backup, and final choice by `eval + log(visits)`.
- Complete-turn actions expose current-turn goals, goal cutoffs, and handoffs
  from which the opponent can score in one turn.

### Reproducible approximation

The experiment uses absolute Player-One integer values so it can reuse the
rank-4 evaluator without changing or retraining it. Nonterminal values are
normalized by 100,000 only when used in a UCT formula. Proven mate values map
to `+10000` or `-10000` in those formulas, preserving the article's terminal
dominance.

- Selection constant: `C = 1.5`.
- First-play urgency: `FPU = 0.5`.
- Visited-child selection at a node whose mover has sign `s` (`+1` for Player
  One, `-1` for Player Two):
  `s * normalized(child.value) + C * sqrt(log(max(1,parent.visits)) / child.visits)`.
- Unvisited-child selection: `s * normalized(child.value) + FPU`.
- A selected unexpanded leaf is expanded once by its complete retained action
  set. Every created child receives a value immediately: terminal and exact
  tactical proofs receive mate-distance scores, while other children use the
  unchanged evaluator. No rollout or value average is used. Expansion is
  atomic: if the full retained set does not fit the remaining work or
  tree-node allowance, none of that leaf's children are installed. The sole
  exception is a generated proven win for the node's mover—either a terminal
  win or an exact `ForcedCutoff`. If one child slot and one work unit remain,
  that winning child alone is installed, the node is marked non-exhaustive,
  and its exact win can still be propagated.
- The root is expanded first and then starts with one visit. After each
  successful frontier expansion, every node on the selected root-to-leaf path
  gains one visit; newly created children remain unvisited. Closed children are
  excluded from later selection. Equal selection and final-choice scores use
  the lexicographically smallest encoded complete action.
- Backup overwrites a Player-One node with the maximum child value and a
  Player-Two node with the minimum child value. Terminal values include
  complete-turn ply distance. Values never change sign merely because a
  handoff occurred; mover signs are applied only at selection/final-choice
  boundaries.
- Final root choice maximizes
  `root_mover_sign * normalized(child.value) + log(max(1,child.visits))`.
- A node is solved by a proven winning child for its mover, or when its action
  set was exhaustively generated and every child is solved. A truncated set
  can prove a found win but cannot prove a loss from absence. A node is closed
  for further exploration when it is solved or all of its retained children
  are closed; closing a truncated node is not promoted to an exact proof.
- A retained `ForcedCutoff` is an exact win for the player who handed off, one
  complete-turn ply after the child. A retained `OpponentImmediateWin` is an
  exact win for the child position's mover on that same next ply. These two
  classifications create solved children with mate-distance values rather
  than being discarded or reduced to heuristic evaluator scores.
- The production protocol emits the selected complete action as one encoded
  line. It does not cache primitive continuations across protocol rounds.
  Received opponent turns and replay corrections are first replayed onto a
  copy and update the authoritative game state only after the full encoded
  turn validates.

Generator rules are also fixed:

- Extraction numbers are one-based. Every tenth extraction pops the deque
  front (FIFO); all others pop the back (LIFO).
- Legal neighbours are Fisher-Yates shuffled by SplitMix64. Before seeding or
  ordering directions, every partial position is represented in four frames:
  identity, horizontal reflection, 180-degree rotation with players/status
  swapped, and their composition. The smallest 64-bit symmetry hash and its
  canonical direction frame seed the sampler. The rebound-goal BFS also visits
  arcs in canonical direction order, and a legal direct center-goal edge is
  preferred before that BFS. These choices make the tested transformed-state
  fixtures equivariant. No wall clock or cross-game PRNG state is consulted.
- This is not a universal stabilizer-equivariance guarantee. If a position is
  itself unchanged by a reflection or rotation, a cap-one retained set would
  need a completed turn fixed by that symmetry. Such a turn need not be the
  first retained result and may not exist. Four-frame canonicalization,
  canonical goal-BFS ordering, and direct-center preference therefore do not
  make cap-one completed-turn retention equivariant on every self-symmetric
  state. This known limitation weighs against submission even though the
  focused reflection and player-rotation fixtures pass.
- Completed actions are deduplicated by their full resulting compact position
  key. The total retained set is at most 250.
- The hard per-expansion partial-path and pending-deque ceilings are 50,000.
  If unexplored work remains when another extraction, pending entry, retained
  action, or shared work unit would exceed its ceiling, generation is marked
  truncated. Search otherwise rejects a full generated action set that does
  not fit the remaining work or tree-node allowance atomically, subject only
  to the proven-win exception above.
- Before ordinary sampling, a goal probe breadth-first-searches the component
  reachable through vertices already visited in the input position or through
  boundary vertices. It visits each topology vertex at most once, stores one
  BFS parent, and can retain one shortest attacking-goal path and one shortest
  own-goal path. Subject to the action cap and deadline, goal existence in this
  fixed rebound component is exact. This goal BFS is separate from the bounded
  handoff sampler and is deadline-limited but does not consume fixed-work
  units.
- Between the exact attacking-goal and own-goal probes, the handoff pre-pass
  performs at most 32 partial-path extractions. Each deque entry carries its
  fully mutated compact position and move sequence, so classification observes
  the edges actually consumed by that rebound path rather than reusing a
  one-parent vertex-only path. These extractions use the same mixed FIFO/LIFO
  schedule and symmetry-canonical neighbour sampler as ordinary generation,
  and they count against the per-expansion partial-path and shared work caps.
  The probe retains the first encountered forced cutoff, safe handoff,
  opponent immediate win, and other terminal loss, in that priority order.
  Exact component scans used to classify completed results are deadline-bound
  but do not add work units beyond the extraction that produced the result.
  It is deliberately bounded: it can guarantee the classification of a found
  result but never proves that an unobserved class is absent.
- The final retained set is stably ordered by tactical class (immediate win,
  forced cutoff, safe handoff, opponent immediate win, terminal loss) and then
  encoded-action lexicographic order; an own goal is a terminal loss, not a
  separate final sort class.
- Every handoff that is actually retained is classified on its complete result
  position. The opponent-component scan exactly detects a reachable immediate
  goal and whether at least one handoff avoids an immediate mover-loses block;
  terminal winners and losers come directly from the rules engine. Each scan
  visits a topology vertex at most once and checks the deadline. Deadline aborts
  yield no proof and mark generation truncated.
- Beyond the retained 250, the experiment does not prove that every safe
  handoff, cutoff, trap, or alternate tactical path has been sampled.

Operational caps are frozen at 250 actions per node, 50,000 generator partial
paths per node, 120,000 stored tree nodes, 3,000,000 fixed-work units, and the
incumbent's construction-inclusive 800 ms first / 165 ms later deadlines.
One work unit is one extracted generator partial path—including the bounded
tactical pre-pass—or one newly valued complete-turn child. Exact goal and
handoff-classification component scans are deadline-bound but not
work-counted. Equal-work data is diagnostic because the candidate's units are
more expensive; equal clock is decisive.
The operational timing gate is stricter than merely observing the internal
deadline: every measured first response must be below 1,000 ms and every later
response below 200 ms.
Construction and replay-correction lookup are included in those response
measurements. The comparison harness reports each game's and each summary's
maximum first/later response, deadline, node-cap and operational-timeout
counts, generator partial/completion/deduplication/tactical/truncation counts,
maximum pending-deque/action length, and maximum reported post-search
tree-node count.
Reference deadline versus node-cap counts are inferred from rank 4's exposed
`budget_exhausted` and node count; the independent operational timeout is
measured directly. The tree figure is the maximum reported post-search node
count, not a byte or peak-allocation measurement. Candidate-only peak resident
memory is measured externally around the final timing probe because that count
does not include reserved capacity, topology, evaluator-cache, or process
overhead. Any RSS captured around the decisive comparison is labeled
process-wide because that binary contains both engines.
The candidate-only measurement command is:

```sh
/usr/bin/time -l \
  build/fullturn-bfm-release/papersoccer_codingame_rank_4_fullturn_bfm_timing_probe
```

The synchronized generated source is 99,925 ASCII characters, SHA-256
`dd119224a296672daed6c897d1f848b3ee37a66046b6f165724b6393b6e4f995`.
All subsequent evidence and any submission decision must recheck this exact
size/hash; results from an earlier generated artifact do not qualify.

### Promotion gates

1. Focused legality, turn-boundary, rebound, goal, own-goal, dead-end, trap,
   determinism, 250-cap, deduplication, deque-schedule, tactical-retention,
   reflection, 180-degree rotation, minimax-sign, UCT-tie, whole-turn protocol
   emission, atomic received-turn validation, generated-source,
   character-limit, generated-compilation, sanitizer, protocol, full
   protected-data-safe repository suite, timing, hard-work, and hard-node
   checks must pass.
2. Paired deterministic smoke games against canonical rank 4 must be clean.
3. Fixed-work comparisons must include 5,000, 20,000, and 30,000 work/node
   reference budgets and report wins plus both candidate colour splits.
4. The decisive gate is paired equal clock at 800/165 ms with both the
   candidate work ceiling and rank-4 node ceiling set to their frozen maximum
   of 3,000,000, so the 30,000-work diagnostic defaults cannot accidentally be
   presented as the equal-clock result.
5. After clean smoke, the decisive run uses `pairs_per_depth=6`, seed batches
   0 and 1, and both colours. The five curated opening pairs contribute 10
   games and the two seeded batches contribute `2 * 4 * 6 * 2 = 96`, for
   exactly 106 games. Promotion requires all of the following predeclared
   conditions: at least 58-48 overall; at least 25 candidate wins in each
   48-game seeded batch; at least 25 candidate wins in each 53-game colour
   split; zero unfinished games; zero illegal-action failures; and zero
   operational timeouts for either engine under the 1,000/200 ms limits.
   These numerical batch, colour, and timeout interpretations were made
   explicit before any decisive equal-clock result was observed.

   The exact decisive invocation is frozen as:

   ```sh
   RANK_4_FULLTURN_BFM_CANDIDATE_WORK=3000000 \
   RANK_4_FULLTURN_BFM_REFERENCE_NODES=3000000 \
   RANK_4_FULLTURN_BFM_CANDIDATE_FIRST_MS=800 \
   RANK_4_FULLTURN_BFM_CANDIDATE_LATER_MS=165 \
   RANK_4_FULLTURN_BFM_REFERENCE_FIRST_MS=800 \
   RANK_4_FULLTURN_BFM_REFERENCE_LATER_MS=165 \
   RANK_4_FULLTURN_BFM_CANDIDATE_REPLAY=1 \
   RANK_4_FULLTURN_BFM_REFERENCE_REPLAY=1 \
   build/fullturn-bfm-release/papersoccer_codingame_rank_4_fullturn_bfm_comparison_gate \
     --pairs-per-depth 6 --batch-start 0 --batch-count 2 --max-turns 200
   ```
6. A failed gate ends promotion. No sealed prospective/final bank is opened.
7. Submission is allowed at most once, only after every gate clearly passes,
   after rechecking the active CodinGame agent and exact source size/hash.

## Public CodinGame recheck

The public leaderboard and battle metadata were queried before implementation
on 2026-08-10. The account was no longer using the objective's older agent
`6604719`: the active entry was agent `6604822`, submission `41114860`, rank 7,
score `40.95`, with processing complete. This does not change the frozen local
rank-4 benchmark hash and prevents the older identity from being reported as
current.

## Commands and results

Implementation and verification results are appended here without changing
the frozen formulas or thresholds above.

### Final artifact and non-match verification

- The shared generator reports the final artifact current at 99,925 ASCII
  characters with SHA-256
  `dd119224a296672daed6c897d1f848b3ee37a66046b6f165724b6393b6e4f995`.
- The final Release focused binary passes 19/19 tests. The nineteenth test is
  test-only coverage that an exact forced cutoff survives the proven-win
  exception when a complete child batch cannot fit.
- A complete Release build passed, followed by 115/115 repository tests in
  132.58 seconds. Five audited tests that read sealed/prospective/final banks
  were excluded; no protected bank was inspected or executed.

  ```sh
  ctest --test-dir build/fullturn-bfm-release --output-on-failure \
    -E '^(papersoccer_flagship_study_unit_tests|papersoccer_game_review_gate_unit_tests|papersoccer_codingame_promotion_banks_current|papersoccer_codingame_promotion_manifest_valid|papersoccer_codingame_topology_comparison_gate_smoke)$'
  ```
- An exact-`dd119224...` ASan+UBSan focused build passed all 18 tests present
  in that snapshot with no sanitizer findings. The later nineteenth case is a
  test-only addition; it does not alter the generated artifact.
- The sanitizer timing probe reported Player 0 first/later responses of
  810.560/169.429 ms and Player 1 responses of 813.453/170.863 ms.
- The final Release timing probe reported Player 0 first/later responses of
  811.576/169.276 ms and Player 1 responses of 809.155/166.414 ms. Every value
  is below the frozen 1,000/200 ms operational thresholds. Candidate-only
  maximum resident set size was 140,771,328 bytes (134.25 MiB).

### Final-hash fixed-work comparisons

The harness reported zero unfinished games and zero operational-timeout
counters in every completed run below. These fixed-work invocations did not
enable equal-clock deadlines, so the timeout counters are not decisive timing
evidence.

| Run | Games | Candidate-reference | Candidate as color 0 | Candidate as color 1 |
| --- | ---: | ---: | ---: | ---: |
| 2,000-work smoke | 18 | 11-7 | 7/9 | 4/9 |
| 5,000 work/node | 106 | 38-68 | 24/53 | 14/53 |
| 20,000 work/node | 106 | 33-73 | 22/53 | 11/53 |
| 30,000 work/node | 106 | 30-76 | 23/53 | 7/53 |

The corresponding candidate/reference maximum first/later response times were
11.804/12.419 versus 5.936/5.761 ms at 2,000 work, 27.246/32.848 versus
14.728/14.588 ms at 5,000, 109.327/111.053 versus 50.441/51.876 ms at
20,000, and 156.768/167.377 versus 74.319/76.957 ms at 30,000. Candidate
deadline and node-cap counters were zero in all four fixed-work runs. Rank 4
reached its fixed node cap normally; neither engine recorded an operational
timeout.

Reproduction commands for the completed fixed-work configurations are:

```sh
build/fullturn-bfm-release/papersoccer_codingame_rank_4_fullturn_bfm_comparison_gate \
  1 2000 200
build/fullturn-bfm-release/papersoccer_codingame_rank_4_fullturn_bfm_comparison_gate \
  12 5000 5000 200 0 1
build/fullturn-bfm-release/papersoccer_codingame_rank_4_fullturn_bfm_comparison_gate \
  12 20000 20000 200 0 1
build/fullturn-bfm-release/papersoccer_codingame_rank_4_fullturn_bfm_comparison_gate \
  12 30000 30000 200 0 1
```

All three required fixed-work diagnostics are materially below rank 4 overall
and show a large color split. They are evidence against promotion, not an
improvement claim.

### Decisive equal-clock result

The exact frozen 800/165 ms invocation above used opening-seed salt
`10322096095657500011` (`0x8f3f73b5cf1c9d6b`) and completed all 106 games:

| Opening block | Games | Candidate-reference | Candidate as color 0 | Candidate as color 1 |
| --- | ---: | ---: | ---: | ---: |
| Curated | 10 | 3-7 | 2/5 | 1/5 |
| Seed batch 0 | 48 | 12-36 | 8/24 | 4/24 |
| Seed batch 1 | 48 | 9-39 | 6/24 | 3/24 |
| **All** | **106** | **24-82** | **16/53** | **8/53** |

There were zero unfinished games, illegal-action failures, and operational
timeouts. Candidate/reference aggregate measured response time was
370,514/362,516 ms. Maximum candidate first/later response times were
819.918/168.575 ms; rank 4 maxima were 803.935/168.949 ms, all below the
predeclared 1,000/200 ms operational thresholds. Internal deadline counters
were 1,881/1,847 and node-cap counters were zero for candidate/reference;
deadline exhaustion is expected in a clock-led run.

Candidate work accounting over the decisive run was 92,105,141 work units,
651,454 expanded parent nodes, 36,623,934 child evaluations, 68,179,443
completed actions, and 55,481,207 generator partial paths. It reported
31,412,043 duplicates, 14,357,291 tactical actions, 49,068 truncations, a
maximum stored tree of 109,615 nodes, a maximum pending deque of 6,462, and a
maximum action length of 37 edges. Rank 4 reported 368,216,318 atomic nodes.

The candidate misses every frozen win threshold: 24 wins is below 58
overall, seed batches 0 and 1 are below 25 wins at 12 and 9, and both color
splits are below 25 wins at 16 and 8. Together with the residual cap-one
stabilizer limitation, this is an unambiguous no-submission decision. The
maintained rank-4 bot and active CodinGame submission remain unchanged.

## Post-frozen live diagnostics and development (2026-08-11)

This section records later exploratory work and does not revise the frozen
design, gates, or 24-82 result above. Fixed-work replay is useful for
reproducibility and diagnosis, but Codingame's 800 ms first / 165 ms later
clocks remain the decisive measure of playing strength.

### Live batch and provenance

The public battle window for experimental agent `6606663`, submission
`41119120`, contained 90 games. The source binding is explicitly
**asserted, not API-verified**: the asserted uploaded artifact has SHA-256
`dd119224a296672daed6c897d1f848b3ee37a66046b6f165724b6393b6e4f995`,
but the public APIs do not expose editor bytes with which to prove that
binding. Strict rules/frame validation excluded 14 opponent operational
failures. The remaining clean terminal set is 76 games with a 40-36 record;
no own operational failure was found in that window.

The append-only diagnostic archive and auditor input are content-addressed:

- Batch manifest SHA-256:
  `f74bab7af4e7a9988b50104428bf2211f0b5bac7fb70098f11b4a9aa231f4572`.
- Clean 76-game auditor TSV SHA-256:
  `b78eb173d58c155a698d359e413caaf47166117264ebcb332b0768c63731e392`.
- Approved protected-ID exclusion-registry SHA-256:
  `ac5d335a8e084e782be93f9c53635896f16344f08e9164481dc7b54eaf923a60`.
- Collector source SHA-256:
  `ea2760b5153219a2e844076b5cbf198b34a631286c8db8b694b7436b6380492a`.
- Replay auditor source SHA-256:
  `d61beda7e8ea1d330adc0080f1a496c3870f62eafa737c80fd85686763f83a50`.

The collector validates cached receipts, normalized records, replay payloads,
cross-references, exact terminal ranks, and the caller-approved exclusion
registry. The TSV carries the manifest, source/submission assertion,
collector, exclusion-registry, repository, and run provenance into every
auditor output. Registry completeness remains an external caller contract;
the collector does not inspect protected banks to infer it.

### Replay diagnosis and generator coverage

At deterministic 30,000-work replay across all 1,663 own decisions, the
candidate reproduced the recorded action 772 times, canonical rank 4 did so
376 times, and the two local policies agreed 478 times. Candidate generation
reported truncation on 1,598 decisions, so action agreement alone is not a
clock-strength measure.

Root-coverage instrumentation gives a more useful explanation. In the first
10 clean games (165 decisions), every recorded action was present both as an
exact retained action and as a retained resulting boundary state. Rank 4's
chosen action was present exactly 133 times and by boundary state 162 times.
Across 10 loss games (152 decisions), the recorded action again had exact and
boundary coverage on all 152 decisions; rank 4 had exact coverage on 119 and
boundary coverage on 151. Only 25 of the first sample's roots and seven of
the loss sample's roots filled the 250-action cap. These samples argue against
generator omission as the main cause of the live losses; evaluator and search
selection are the stronger current hypotheses.

Clock-mode replay uses the actual 800/165 ms deadlines, but local hardware and
timing do not recreate the uploaded process exactly. Its action matches are
therefore stability and coverage diagnostics, not proof of uploaded decisions.

### Behavior-preserving generator optimization

Complete-turn boundary deduplication now uses a fixed 512-slot,
allocation-free open-addressed table instead of a linear vector scan. Five
curated fixed-work states retained byte-identical chosen actions and generator
statistics. An alternating benchmark improved from 465.855 ms to 459.911 ms,
about 1.28 percent; this is a performance optimization, not a strength claim.

The current generated local artifact is 99,955 ASCII characters, SHA-256
`c268286020ae841e4b5442e3578107ab32ded151e460a5bbe0495cb9b0b19d87`.
It is distinct from the earlier asserted live `dd119224...` artifact. On
2026-08-11 it was copied into the authenticated CodinGame editor, copied back,
and verified byte-for-byte immediately before the arena action. CodinGame
history recorded version 60 and public battle metadata assigned agent
`6608659`, submission `41121957`. This establishes the local editor procedure,
but the public API still cannot return editor bytes; exported replay metadata
therefore continues to use `source_binding_status=asserted-not-api-verified`.
The upload is an exploratory search diagnostic, not promotion evidence.

### Exploratory actual-clock parameter screens

Each batch-0 screen below used 24 games over mixed opening depths
`0,1,2,3,6,7,12`, the 800/165 ms clocks, replay enabled, and 3,000,000 work
and node ceilings. They completed without operational failures. The independent
default batch-1 screen used the same clock-led setup.

| Configuration | Seed batch | Candidate-reference |
| --- | ---: | ---: |
| Former default: `C=1.5`, `FPU=0.5`, final-visit weight `1`, replay blend `15`, residual weight `100` | 0 | 8-16 |
| Final-visit weight `0`; other defaults unchanged | 0 | 9-15 |
| Low exploration: `C=0.25`, `FPU=0`, final-visit weight `0.25` | 0 | 5-19 |
| Replay-only evaluator: replay blend `100`, residual weight `0` | 0 | 7-17 |
| Residual disabled: replay blend `15`, residual weight `0` | 0 | 1-23 |
| Final-visit weight `0.25`; default exploration/evaluator | 0 | 6-18 |
| Former default configuration, independent openings | 1 | 5-19 |
| Final-visit weight `0`, independent openings | 1 | 8-16 |
| Former default, independent batch 2 | 2 | 0-24 |
| Final-visit weight `0`, independent batch 2 | 2 | 3-21 |
| Final-visit `0`, root `250` / deeper `64` | 0 | 3-21 |
| Root evaluator only; no deeper BFM | 0 | 1-23 |

The completed screens reject low exploration, replay-only evaluation,
residual removal, final-visit weight `0.25`, deeper cap `64`, and root-only
selection as tested. Final-visit weight `0` improved from 8-16 to 9-15 on
batch 0, from 5-19 to 8-16 on batch 1, and from 0-24 to 3-21 on batch 2: it is
20-52 combined versus the former default's 13-59. The maintained exploratory
default is therefore final-visit weight `0`, with full `250` width at every
depth and deeper BFM enabled. This repeat is sufficient to justify one
source-bound exploratory live upload, not promotion: neither configuration
approached parity and no 24-game screen supersedes the frozen multi-batch
promotion gate.
