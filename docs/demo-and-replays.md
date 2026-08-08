# Demo, CLI, and replays

The [live GitHub Pages demo](https://lecorbio.github.io/paper-soccer-strategy-engine/)
is a static browser application backed by the same C++ rules and bot code as
the native tools. It supports human-versus-bot games, takebacks, diagnostics,
bot-versus-bot replay generation, and imported replay inspection.
The separate **[Benchmark results](https://lecorbio.github.io/paper-soccer-strategy-engine/benchmarks/)**
link opens a static overview of the frozen flagship study.

## Run the browser locally

Both generated single-file WebAssembly modules are checked in, so the app does
not need a local server or compiler:

```bash
open web/index.html
```

On other platforms, open `web/index.html` as a local file in a modern browser.
See [Reproducibility](reproducibility.md#webassembly-artifacts) only when
changing a C++ browser boundary or verifying the checked-in modules. Under
`file://`, Game Review automatically uses its direct-file main-thread fallback
if the browser refuses to create or load its classic worker.

## Play against a bot

The app starts in **Play vs bot** mode:

1. Choose `RandomBot`, `MctsBot`, `AlphaBetaBot`, `JacekInspiredBot`, or
   `Rank5DerivedBot — 50k demo profile`. `Expert — DeepTurnSearch` is present
   only when the frozen held-out gate supports that exact label.
2. Choose **Move first** to play as Player 1 and attack the top goal, or **Move
   second** to play as Player 2, let the bot open, and attack the bottom goal.
3. Configure the opponent where applicable. MCTS exposes a seed and a fixed
   number of new simulations per move. Alpha-beta and Jacek-inspired profiles
   expose possession-handoff depth. They are deterministic, so their recorded
   seed is metadata rather than a decision input.
4. Select **Start**, then click a highlighted destination. A destination marked
   `↻` grants another edge in the same possession.

The Rank5Derived profile hides seed and search controls because its browser
contract is fixed: 50,000 nodes, no wall-clock cutoff, 0% learned-value blend,
and no replay-book corrections. It reuses search code from the authentic
rank-5 submission under different game rules. It is not the ranked entrant, and
its demo measurements must not be attributed to that entrant. The precise
differences are in [Rules and algorithms](algorithms.md#rank5derivedbot).

The setup remains unlocked on first load so the turn order can be chosen before
the game starts. Once play begins, the opponent, turn order, seed, and search
setting remain visible and locked. Search opponents expose their latest
diagnostics. Depending on the bot, these include MCTS visits and reuse,
alpha-beta depth and pruning, neural model identity, or Rank5Derived
complete-action and cached-edge details.

The selected Expert profile, if admitted, is a locked 100k, 200k, or 400k
`DeepTurnSearchBot`; it is not a configurable Rank5Derived profile and does not
inherit the authentic contest rank. RandomBot remains the default opponent.

## Takebacks and exports

**Undo move** removes the latest physical edge, whether it belongs to the human
or bot. If a bot edge is undone, the restored game pauses instead of
immediately asking the bot again. Select **Continue bot** when ready. Stateful
or random bots may choose a different continuation after a takeback; starting a
fresh game with the same settings and human choices is the reproducible path.

**Change settings** stops the active game without discarding it. **Export game**
writes a `papersoccer.human-match.v1` document containing:

- the standard replay;
- human and bot configuration metadata;
- every recorded bot search; and
- typed Rank5Derived provenance and complete-action diagnostics when used.

The exported human match can be opened through the normal replay loader.

## Review a finished game

**Review game** appears only after a completed live game, a generated bot
replay, or an imported replay that passes validation. It does not appear as
a live hint or coaching control during active play. Review accepts
`papersoccer.replay.v1`, `papersoccer.replay.v2`, and the replay nested in
`papersoccer.human-match.v1`; a prior `papersoccer.game-review.v1` export can be
reopened with its analyzed snapshot.

Import preparation checks the schema and standard 8x10 start, then the analysis
bridge replays every edge through the authoritative C++ `Match`. A wrong ply,
player, source point, rebound declaration, intermediate status, winner, or
truncation declaration rejects the import. Existing replay and human-match
schemas are not changed.

The review controls provide:

- **Fast preview**, the fixed depth-32, 50,000-node `fast-50k` profile;
- **Deep refinement**, which produces Fast first and then analyzes with the
  locked depth-32 100k, 200k, or 400k Deep profile;
- possession-level progress and cancellation between synchronous searches;
- one badge per complete possession while the ordinary edge controls remain
  available;
- distinct played and recommended rebound paths plus the first-divergence
  marker;
- grade, estimated loss, borderline or confidence state, proof status, depth,
  node, cache, and oracle diagnostics; and
- **Try this line**, which asks the authoritative C++ rules engine to fork at
  the possession boundary and validate the complete recommendation on a
  separate reversible state. The recommendation appears as a ghost path; you
  choose legal edges from the boundary, can follow or deviate from it, and then
  continue while controlling either side. Closing the sandbox returns to the
  immutable original replay.

Cancellation may become visible only after the current synchronous search
finishes. A new or closed session invalidates old session IDs, so late worker
results cannot overwrite the current review. Hosted pages use a classic worker
with its own analysis Wasm instance. If that path fails under `file://`, a
second main-thread module performs one possession search per event-loop task.

The summary deliberately contains grade counts, best-action rate, and the
largest estimated loss. It has no win-probability graph and no invented overall
accuracy number. `ProvenWin` and `ProvenLoss` come from the exact endgame oracle;
all other labels are deterministic estimates tied to the named search profile
and its own validation calibration.

**Export review** writes `papersoccer.game-review.v1`. It contains an unchanged
copy of the source replay, its canonical JSON SHA-256, analyzer, search,
calibration, oracle, and ranked-source identities, and deterministic
per-possession played and recommended actions, divergence, grade, loss,
borderline state, proof, and search diagnostics. `confidenceState` is one of
`exact`, `deterministic-estimate`, `borderline-estimate`, or `unclear`;
borderline remains a threshold warning, not a confidence interval. Wall-clock
timing is omitted from this deterministic export. Reimport verifies the replay
hash before using the analysis data.

## Watch or generate a replay

Select **Watch replay**, then choose **Generate replay** to create a
bot-versus-bot match with the settings below, or **Open existing** to load a
replay JSON file from disk. Each player has an independent bot and seed, plus
an MCTS iteration or alpha-beta depth setting when relevant. Either player can
use Rank5Derived; its fixed profile and provenance are retained in the replay
metadata. An admitted Expert profile is likewise identified as
DeepTurnSearch—not Rank5Derived—in generated replay metadata.

Generation advances one edge at a time so controls and diagnostics can update
between searches. It stops after 512 plies if no winner has been reached,
reports progress during longer matches, and opens the result directly in the
viewer.

The browser remains a presentation layer. C++ owns state, legal moves,
rebounds, winners, bot RNG, and history. JavaScript owns canvas drawing,
controls, accessibility, timers, and animation. See
[Architecture](architecture.md#browser-and-analysis-boundaries) for the command/snapshot
boundary.

## Native replay exporter

After a native build, generate a deterministic RandomBot-versus-RandomBot
replay:

```bash
mkdir -p results
./build/native/papersoccer_replay_export 12345 512 > results/replay.json
```

Arguments are optional:

- `base-seed` defaults to `RandomBot::default_seed()`;
- `max-plies` defaults to `512`.

The exporter intentionally remains a RandomBot-versus-RandomBot tool. Adding
search bots to the CLI and browser did not change its arguments, player
metadata, or replay schema.

It writes `papersoccer.replay.v2`. Points use the engine's zero-based board
coordinates. The viewer also accepts legacy `papersoccer.replay.v1` and shifts
every v1 point one row down (`y + 1`) during import, preserving the old visual
geometry.

Raw local replays and human-match exports belong under the ignored `results/`
directory unless deliberately selected as a small regression fixture.

## Terminal CLI

Build the native project, then run:

```bash
./build/native/papersoccer_cli
```

Commands during a game are:

- `<index>`: play the displayed legal move;
- `b`: print the ASCII board;
- `a`: toggle automatic board printing;
- `h`: show help;
- `q`: quit.

The CLI prints the board automatically by default. It supports human, seeded
RandomBot, MCTS, alpha-beta, and Jacek-inspired controllers in the following
pairings:

1. Human versus Human
2. Human versus any supported bot
3. RandomBot versus RandomBot, MCTS, alpha-beta, or Jacek-inspired
4. MCTS versus MCTS, alpha-beta, or Jacek-inspired
5. Alpha-beta versus Alpha-beta or Jacek-inspired
6. Jacek-inspired versus Jacek-inspired

Mixed-controller modes ask which side uses which controller. RandomBot and
MCTS modes ask for base seeds; MCTS also asks for a per-move iteration budget,
defaulting to 2,000. Alpha-beta modes default to possession depth 6 and 100,000
nodes. Jacek-inspired modes default to depth 6 and 20,000 nodes because every
leaf evaluates the compact network.

Player One uses the selected base seed and Player Two uses `base_seed + 1`.
After each search move the CLI prints the relevant deterministic counters and
timings: MCTS tree work and reuse, alpha-beta depth/pruning/transpositions, or
the neural model hash and neural-search counters. Root scores are explicitly
Player-One-oriented and reported as unavailable if the first iteration did not
complete.

The browser offers Rank5Derived as a fixed demo profile; the current terminal
menu does not expose it.

## Coordinates in text and UI

Engine examples use `Point{x, y}`. The CLI and renderer display
`(row, column)`, equivalent to `(y, x)`. The initial engine point `(4,6)` is
therefore displayed as `(6,4)`. All coordinates are zero-based.

For the complete board and rebound contract, see
[Rules and algorithms](algorithms.md#rules-contract).
