# Bounded network and search research

This implementation supports a fixed research wave around the verified rank-three release. It exports bias-free four-bit networks with the existing activations and fixed scales, rejects sources at or above 100,000 characters, and keeps source generation separate from correctness, timing, live admission, and promotion.

The approved designs are a historical distance-only 32/32 network with modern 990/180 ms clocks, matched degree-aware 8/8 teacher and observed-parent outcome arms, a degree-aware 12/16 Huber-plus-ranking arm, and an incumbent-network best-first minimax pair with exploration 0 and 0.95. The search pair shares 250 actions per expansion, 50,000 partial paths, 32,768 tree nodes, FPU 0.5, minimax backup, deterministic ordering, legal fallback, and explicit unknown truncation.

## Numerical and lifecycle contracts

The training wrapper fixes one float epoch, four QAT epochs, one seed, the existing optimizer and root order, and the original teacher data. The matched 8/8 arms use identical initialization and data. Their group supervision is half original successors and half exact originally played parents; only the mixed arm blends 20% mover-signed outcomes with 80% teacher value for those parents. Proof and terminal targets remain unchanged. Unplayed successors and development or confirmation outcomes cannot receive outcome labels.

`rank_two_design_network_v1.py` and `rank_two_design_bfm_v1.py` produce research sources and reports. They do not establish deployment safety or live qualification. `rank_two_design_training_v2.py` requires retained, hash-bound local numerical recipes and a bound input plan. Those local recipes, game transcripts, checkpoints, and campaign control state are intentionally not included in this source-only delivery. The artifact-integrity test runs when those local recipe artifacts are available; the target, gradient, export, and lifecycle tests use ordinary fixtures.

`rank_two_design_wave_v2.py` requires an explicit execution owner. It starts the single seven-day clock only after the predecessor comparison, required restoration, all window archives, and source/editor/deployed identity audit pass. Six sources are a ceiling, not a quota. Existing failure bans and historical receipts are retained.

## Recorded outcome

Five sources were tested. The historical D32 and teacher-only 8/8 sources each encountered an own timeout and their exact hashes were banned. The mixed 8/8 source reached third place without own failures but failed the development gate. The 12/16 source passed development with a mean paired uplift of 3.32 percentage points, won eight of twelve private games, and finished third in all three formal windows with zero own failures. It did not meet the required top-two formal result and was not promoted.

Exploration 0 failed the pilot strength gate and then finished its first full live window at rank 62 with 34 recorded own timeouts. Its exact hash was banned, every game record was retained, only unclaimed slots were cancelled, and the incumbent was restored once. The timeout cause remains unknown. Exploration 0.95 remained unallocated and unplayed because the shared pipeline did not establish whole-response timing: search can consume the deadline before result construction, tree cleanup, action encoding, and output. No replacement variant, rescue training, larger clock, or quota-filling admission was used.

The final incumbent source, editor copyback, and deployed identity matched after a complete 90-game restoration window with zero own failures. The campaign's best confirmed completed live finish remained third place; no new source was promoted. These results are complete-window records, not favorable subsets or a claim that transient ranks demonstrate strength.
