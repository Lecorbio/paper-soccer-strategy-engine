import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const compiledModule = await import("../../web/papersoccer-wasm.js");
globalThis.createPaperSoccerModule = compiledModule.default;
await import("../../web/game-engine.js");

const { BotKind, Player, Status, ready } = globalThis.PaperSoccer;
const gameEngine = await ready;
const JACEK_MODEL_SHA256 =
  "57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084";
const RANK5_ARTIFACT_SHA256 =
  "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29";
const RANK5_TRANSCRIPT_FIXTURE = readFileSync(
  new URL("../fixtures/rank5_derived_vs_mcts_path.txt", import.meta.url),
  "utf8",
).split(/\r?\n/u).filter((line) => line && !line.startsWith("#"))
  .map((line) => {
    const [x, y] = line.split(" ").map(Number);
    return { x, y };
  });

function botConfig(kind, seed, iterations = 8) {
  return { kind, seed, iterations };
}

function finishBotReplay(oneConfig, twoConfig, maxPlies) {
  let snapshot = gameEngine.startBotReplay(oneConfig, twoConfig, maxPlies);
  let commands = 0;
  while (!snapshot.done) {
    snapshot = gameEngine.playBotReplay(snapshot.sessionId, snapshot.revision);
    commands += 1;
    assert.ok(commands <= maxPlies, "the bounded bot replay should stop at max plies");
  }
  return snapshot;
}

test("browser client exposes the compiled C++ session instead of JavaScript rules", () => {
  assert.equal(Player.One, "one");
  assert.equal(Player.Two, "two");
  assert.equal(Status.InProgress, "inProgress");
  assert.equal(BotKind.Random, "random");
  assert.equal(BotKind.Mcts, "mcts");
  assert.equal(BotKind.AlphaBeta, "alphaBeta");
  assert.equal(BotKind.JacekInspired, "jacekInspired");
  assert.equal(BotKind.Rank5Derived, "rank5Derived");
  assert.equal(typeof gameEngine.startGame, "function");
  assert.equal(typeof gameEngine.playHuman, "function");
  assert.equal(typeof gameEngine.playBot, "function");
  assert.equal(typeof gameEngine.undo, "function");
  assert.equal(typeof gameEngine.humanMatch, "function");
  assert.equal(typeof gameEngine.startBotReplay, "function");
  assert.equal(typeof gameEngine.playBotReplay, "function");
  assert.equal(typeof gameEngine.botReplaySnapshot, "function");
  assert.equal(globalThis.PaperSoccer.applyMove, undefined);
  assert.equal(globalThis.PaperSoccer.RandomBot, undefined);
});

test("C++ snapshot contains the initial position, legal moves, and exact uint64 seed", () => {
  const snapshot = gameEngine.startGame("18446744073709551615", Player.One);

  assert.equal(snapshot.schema, "papersoccer.web-session.v1");
  assert.ok(Number.isInteger(snapshot.sessionId) && snapshot.sessionId > 0);
  assert.equal(snapshot.revision, 0);
  assert.equal(snapshot.humanPlayer, Player.One);
  assert.deepEqual(snapshot.state, {
    ball: { x: 4, y: 6 },
    toMove: Player.One,
    status: Status.InProgress,
    winner: null,
  });
  assert.deepEqual(
    snapshot.legalMoves.map((move) => move.to),
    [
      { x: 3, y: 5 },
      { x: 3, y: 6 },
      { x: 3, y: 7 },
      { x: 4, y: 5 },
      { x: 4, y: 7 },
      { x: 5, y: 5 },
      { x: 5, y: 6 },
      { x: 5, y: 7 },
    ],
  );
  assert.equal(
    snapshot.replay.players.two.seed,
    "18446744073709551615",
  );
  assert.equal(snapshot.replay.players.two.kind, "RandomBot");
  assert.equal(snapshot.replay.truncated, true);
  assert.deepEqual(snapshot.diagnostics, {
    schema: "papersoccer.bot-search-diagnostics.v1",
    botConfiguration: {
      kind: "RandomBot",
      seed: "18446744073709551615",
    },
    lastBotSearch: null,
    botSearches: [],
  });
});

test("human commands advance the authoritative C++ state and replay", () => {
  const initial = gameEngine.startGame("17", Player.One);
  const next = gameEngine.playHuman(initial.sessionId, initial.revision, 3);

  assert.equal(next.revision, 1);
  assert.deepEqual(next.state.ball, { x: 4, y: 5 });
  assert.equal(next.state.toMove, Player.Two);
  assert.equal(next.replay.moves.length, 1);
  assert.deepEqual(next.replay.moves[0], {
    ply: 1,
    player: Player.One,
    from: { x: 4, y: 6 },
    to: { x: 4, y: 5 },
    extraTurn: false,
    statusAfter: Status.InProgress,
  });
});

test("stale and wrong-turn commands are rejected without changing the session", () => {
  const initial = gameEngine.startGame("17", Player.One);
  const afterHuman = gameEngine.playHuman(
    initial.sessionId,
    initial.revision,
    0,
  );

  assert.throws(
    () => gameEngine.playHuman(initial.sessionId, initial.revision, 0),
    /stale_revision/,
  );
  assert.throws(
    () => gameEngine.playHuman(
      afterHuman.sessionId,
      afterHuman.revision,
      0,
    ),
    /wrong_turn/,
  );

  const unchanged = gameEngine.snapshot();
  assert.equal(unchanged.revision, afterHuman.revision);
  assert.deepEqual(unchanged.state.ball, afterHuman.state.ball);
  assert.deepEqual(unchanged.replay.moves, afterHuman.replay.moves);
});

test("the browser bot move is selected by the compiled C++ RandomBot", () => {
  const initial = gameEngine.startGame("17", Player.Two);

  assert.equal(initial.humanPlayer, Player.Two);
  assert.equal(initial.state.toMove, Player.One);
  assert.equal(initial.replay.players.one.kind, "RandomBot");
  assert.equal(initial.replay.players.two.kind, "Human");

  const afterBot = gameEngine.playBot(initial.sessionId, initial.revision);

  assert.equal(afterBot.revision, 1);
  assert.deepEqual(afterBot.state.ball, { x: 4, y: 5 });
  assert.equal(afterBot.state.toMove, Player.Two);
  assert.deepEqual(afterBot.replay.moves[0], {
    ply: 1,
    player: Player.One,
    from: { x: 4, y: 6 },
    to: { x: 4, y: 5 },
    extraTurn: false,
    statusAfter: Status.InProgress,
  });
  assert.equal(afterBot.diagnostics.lastBotSearch, null);
  assert.deepEqual(afterBot.diagnostics.botSearches, []);
});

test("browser undo removes human and bot plies one at a time", () => {
  const initial = gameEngine.startGame("17", Player.Two);
  const afterBot = gameEngine.playBot(initial.sessionId, initial.revision);
  const afterHuman = gameEngine.playHuman(
    afterBot.sessionId,
    afterBot.revision,
    0,
  );

  const withoutHuman = gameEngine.undo(
    afterHuman.sessionId,
    afterHuman.revision,
  );
  assert.equal(withoutHuman.revision, 3);
  assert.deepEqual(withoutHuman.replay.moves, afterBot.replay.moves);
  assert.deepEqual(withoutHuman.state, afterBot.state);

  assert.throws(
    () => gameEngine.undo(withoutHuman.sessionId, afterHuman.revision),
    /stale_revision/,
  );

  const withoutBot = gameEngine.undo(
    withoutHuman.sessionId,
    withoutHuman.revision,
  );
  assert.equal(withoutBot.revision, 4);
  assert.deepEqual(withoutBot.replay.moves, []);
  assert.deepEqual(withoutBot.state, initial.state);
  assert.throws(
    () => gameEngine.undo(withoutBot.sessionId, withoutBot.revision),
    /no_moves_to_undo/,
  );
  assert.deepEqual(gameEngine.snapshot(), withoutBot);
});

test("a 10,000-search budget reaches the authoritative live configuration", () => {
  const snapshot = gameEngine.startGame(
    "18446744073709551615",
    Player.One,
    BotKind.Mcts,
    10000,
  );

  assert.deepEqual(snapshot.diagnostics.botConfiguration, {
    kind: "MctsBot",
    seed: "18446744073709551615",
    iterations: 10000,
  });
  assert.equal(snapshot.diagnostics.lastBotSearch, null);
  assert.deepEqual(snapshot.diagnostics.botSearches, []);
});

test("live games expose the selected deterministic MctsBot configuration", () => {
  const first = gameEngine.startGame(
    "18446744073709551615",
    Player.Two,
    BotKind.Mcts,
    8,
  );
  assert.deepEqual(first.replay.players.one, {
    kind: "MctsBot",
    seed: "18446744073709551615",
    iterations: 8,
  });

  const firstMove = gameEngine.playBot(first.sessionId, first.revision);
  const search = firstMove.diagnostics.lastBotSearch;
  assert.equal(firstMove.diagnostics.schema,
    "papersoccer.bot-search-diagnostics.v1");
  assert.equal(search.ply, 1);
  assert.equal(search.player, Player.One);
  assert.deepEqual(search.chosenMove, {
    from: { x: 4, y: 6 },
    to: firstMove.replay.moves[0].to,
  });
  assert.equal(search.requestedIterations, 8);
  assert.ok(search.completedIterations <= search.requestedIterations);
  assert.equal(typeof search.decisionTimeNs, "number");
  assert.equal(typeof search.nodes, "number");
  assert.equal(typeof search.simulatedPlies, "number");
  assert.equal(typeof search.totalRootVisits, "number");
  assert.equal(typeof search.reusedVisits, "number");
  assert.equal(typeof search.maxDepth, "number");
  assert.equal(typeof search.provenNodes, "number");
  assert.ok(search.provenWinner === null ||
    search.provenWinner === Player.One || search.provenWinner === Player.Two);
  assert.equal(typeof search.tacticalProbes, "number");
  assert.equal(typeof search.tacticalNodes, "number");
  assert.equal(typeof search.tacticalSolvedPositions, "number");
  assert.equal(typeof search.tacticalDepthCutoffs, "number");
  assert.equal(typeof search.tacticalNodeCutoffs, "number");
  assert.equal(typeof search.maxTacticalDepth, "number");
  assert.equal(search.tacticalProbes, 0);
  assert.equal(search.tacticalNodes, 0);
  assert.equal(search.tacticalSolvedPositions, 0);
  assert.equal(search.tacticalDepthCutoffs, 0);
  assert.equal(search.tacticalNodeCutoffs, 0);
  assert.equal(search.maxTacticalDepth, 0);
  assert.equal(typeof search.rebuildCount, "number");
  assert.equal(typeof search.expansionSaturated, "boolean");
  assert.equal(typeof search.rootValue, "number");
  assert.deepEqual(firstMove.diagnostics.botSearches, [search]);
  assert.deepEqual(gameEngine.snapshot(), firstMove);

  const exported = gameEngine.humanMatch();
  assert.equal(exported.schema, "papersoccer.human-match.v1");
  assert.deepEqual(exported.replay, firstMove.replay);
  assert.deepEqual(exported.botConfiguration, {
    kind: "MctsBot",
    seed: "18446744073709551615",
    iterations: 8,
  });
  assert.deepEqual(exported.botSearches, [search]);

  const second = gameEngine.startGame(
    "18446744073709551615",
    Player.Two,
    BotKind.Mcts,
    8,
  );
  const secondMove = gameEngine.playBot(second.sessionId, second.revision);

  assert.equal(firstMove.replay.moves.length, 1);
  assert.deepEqual(firstMove.replay, secondMove.replay);
  const replayJson = JSON.stringify(firstMove.replay);
  assert.doesNotMatch(replayJson, /decisionTimeNs|botSearches|diagnostics/);
});

test("live MctsBot games default to 2000 iterations", () => {
  const snapshot = gameEngine.startGame("19", Player.One, BotKind.Mcts);

  assert.deepEqual(snapshot.replay.players.two, {
    kind: "MctsBot",
    seed: "19",
    iterations: 2000,
  });
});

test("live games expose and run the selected AlphaBetaBot depth", () => {
  const initial = gameEngine.startGame(
    "23",
    Player.Two,
    BotKind.AlphaBeta,
    1,
  );

  assert.deepEqual(initial.replay.players.one, {
    kind: "AlphaBetaBot",
    seed: "23",
    depth: 1,
  });
  assert.deepEqual(initial.diagnostics.botConfiguration, {
    kind: "AlphaBetaBot",
    seed: "23",
    depth: 1,
  });

  const moved = gameEngine.playBot(initial.sessionId, initial.revision);
  assert.equal(moved.revision, 1);
  assert.equal(moved.replay.moves.length, 1);
  const search = moved.diagnostics.lastBotSearch;
  assert.equal(search.searchType, "alphaBeta");
  assert.equal(search.requestedDepth, 1);
  assert.equal(search.completedDepth, 1);
  assert.equal(search.attemptedDepth, 1);
  assert.ok(search.nodes > 0);
  assert.equal(typeof search.leafEvaluations, "number");
  assert.equal(typeof search.terminalNodes, "number");
  assert.equal(typeof search.cutoffs, "number");
  assert.equal(typeof search.transpositionProbes, "number");
  assert.equal(typeof search.transpositionHits, "number");
  assert.equal(typeof search.maxPhysicalPly, "number");
  assert.equal(typeof search.rootScore, "number");
  assert.equal(search.rootScoreValid, true);
  assert.equal(typeof search.budgetExhausted, "boolean");
  assert.deepEqual(moved.diagnostics.botSearches, [search]);

  const exported = gameEngine.humanMatch();
  assert.equal(exported.replay.truncated, true);
  assert.deepEqual(exported.botSearches, [search]);
});

test("live AlphaBetaBot games default to depth 6", () => {
  const snapshot = gameEngine.startGame(
    "29",
    Player.One,
    BotKind.AlphaBeta,
  );

  assert.equal(snapshot.replay.players.two.depth, 6);
});

test("live games expose and run JacekInspiredBot with neural-search diagnostics", () => {
  const initial = gameEngine.startGame(
    "31",
    Player.Two,
    BotKind.JacekInspired,
    2,
  );

  const configuration = initial.diagnostics.botConfiguration;
  assert.equal(configuration.kind, "JacekInspiredBot");
  assert.equal(configuration.seed, "31");
  assert.equal(configuration.depth, 2);
  assert.equal(configuration.maxNodes, 20000);
  assert.equal(configuration.maxTimeMs, 0);
  assert.equal(configuration.modelSha256, JACEK_MODEL_SHA256);
  assert.deepEqual(initial.replay.players.one, configuration);

  const moved = gameEngine.playBot(initial.sessionId, initial.revision);
  assert.equal(moved.revision, 1);
  assert.equal(moved.replay.moves.length, 1);
  const search = moved.diagnostics.lastBotSearch;
  assert.equal(search.searchType, "alphaBeta");
  assert.equal(search.ply, 1);
  assert.equal(search.player, Player.One);
  assert.equal(search.requestedDepth, 2);
  assert.ok(search.completedDepth >= 1 && search.completedDepth <= 2);
  assert.ok(search.attemptedDepth >= search.completedDepth);
  assert.ok(search.nodes > 0 && search.nodes <= configuration.maxNodes);
  assert.ok(search.leafEvaluations > 0);
  assert.equal(typeof search.decisionTimeNs, "number");
  assert.equal(typeof search.terminalNodes, "number");
  assert.equal(typeof search.cutoffs, "number");
  assert.equal(typeof search.transpositionProbes, "number");
  assert.equal(typeof search.transpositionHits, "number");
  assert.equal(typeof search.maxPhysicalPly, "number");
  assert.equal(typeof search.rootScore, "number");
  assert.equal(search.rootScoreValid, true);
  assert.equal(typeof search.budgetExhausted, "boolean");
  assert.deepEqual(moved.diagnostics.botSearches, [search]);

  const repeated = gameEngine.startGame(
    "31",
    Player.Two,
    BotKind.JacekInspired,
    2,
  );
  const repeatedMove = gameEngine.playBot(
    repeated.sessionId,
    repeated.revision,
  );
  assert.deepEqual(moved.replay, repeatedMove.replay);
});

test("live games expose the fixed Rank5DerivedBot demo profile and diagnostics", () => {
  const initial = gameEngine.startGame(
    "18446744073709551615",
    Player.Two,
    BotKind.Rank5Derived,
    1,
  );

  const configuration = initial.diagnostics.botConfiguration;
  assert.equal(configuration.kind, "Rank5DerivedBot");
  assert.equal(configuration.profile, "50k-demo");
  assert.equal(configuration.maxNodes, 50000);
  assert.equal(configuration.modelBlendPercent, 0);
  assert.equal(configuration.replayBookEnabled, false);
  assert.deepEqual(configuration.originalArtifact, {
    name: "rank_5",
    rank: 5,
    fieldSize: 206,
    submissionId: "41015554",
    sha256: RANK5_ARTIFACT_SHA256,
  });
  assert.deepEqual(initial.replay.players.one, configuration);

  const moved = gameEngine.playBot(initial.sessionId, initial.revision);
  const search = moved.diagnostics.lastBotSearch;
  assert.equal(search.searchType, "rank5Derived");
  assert.equal(search.ply, 1);
  assert.equal(search.player, Player.One);
  assert.equal(search.requestedNodes, 50000);
  assert.ok(search.visitedNodes > 0 && search.visitedNodes <= 50000);
  assert.ok(search.completedDepth > 0);
  assert.ok(search.attemptedDepth >= search.completedDepth);
  assert.equal(typeof search.rootScore, "number");
  assert.equal(typeof search.budgetExhausted, "boolean");
  assert.ok(search.plannedActionLength > 0);
  assert.equal(search.currentEdgeIndex, 0);
  assert.equal(search.cachedContinuation, false);
  assert.deepEqual(moved.diagnostics.botSearches, [search]);
});

test("invalid seeds are rejected by C++ without replacing the live game", () => {
  const initial = gameEngine.startGame("23", Player.One);

  assert.throws(
    () => gameEngine.startGame("18446744073709551616", Player.One),
    /unsigned 64-bit decimal integer/,
  );
  assert.throws(
    () => gameEngine.startGame("-1", Player.One),
    /unsigned 64-bit decimal integer/,
  );

  assert.deepEqual(gameEngine.snapshot(), initial);
});

test("unknown bot kinds are rejected by the client without replacing the live game", () => {
  const initial = gameEngine.startGame("29", Player.One, BotKind.Random, 8);

  assert.throws(
    () => gameEngine.startGame("31", Player.One, "unknown", 8),
    /Unsupported bot kind: unknown/,
  );

  assert.deepEqual(gameEngine.snapshot(), initial);
});

test("commands captured from a replaced game cannot mutate the new session", () => {
  const replaced = gameEngine.startGame("17", Player.Two);
  const current = gameEngine.startGame("23", Player.Two);

  assert.notEqual(replaced.sessionId, current.sessionId);
  assert.throws(
    () => gameEngine.playBot(replaced.sessionId, replaced.revision),
    /stale_session/,
  );
  assert.deepEqual(gameEngine.snapshot(), current);
});

test("a complete browser session remains in C++ through the terminal position", () => {
  let snapshot = gameEngine.startGame("41", Player.One);
  let plies = 0;

  while (snapshot.state.status === Status.InProgress) {
    snapshot = snapshot.state.toMove === snapshot.humanPlayer
      ? gameEngine.playHuman(snapshot.sessionId, snapshot.revision, 0)
      : gameEngine.playBot(snapshot.sessionId, snapshot.revision);
    plies += 1;
    assert.ok(plies <= 512, "the deterministic browser game should terminate");
  }

  assert.equal(snapshot.revision, plies);
  assert.equal(snapshot.replay.moves.length, plies);
  assert.equal(snapshot.replay.status, snapshot.state.status);
  assert.equal(snapshot.replay.winner, snapshot.state.winner);
  assert.notEqual(snapshot.state.winner, null);
  assert.deepEqual(snapshot.legalMoves, []);
});

test("bot replay sessions preserve independent bot kinds and exact uint64 seeds", () => {
  const snapshot = gameEngine.startBotReplay(
    botConfig(BotKind.Random, "18446744073709551615"),
    botConfig(BotKind.Mcts, "18446744073709551614"),
    0,
  );

  assert.equal(snapshot.schema, "papersoccer.bot-replay-session.v1");
  assert.ok(Number.isInteger(snapshot.sessionId) && snapshot.sessionId > 0);
  assert.equal(snapshot.revision, 0);
  assert.equal(snapshot.done, true);
  assert.deepEqual(snapshot.replay.players, {
    one: {
      kind: "RandomBot",
      seed: "18446744073709551615",
    },
    two: {
      kind: "MctsBot",
      seed: "18446744073709551614",
      iterations: 8,
    },
  });
  assert.equal(snapshot.replay.truncated, true);
  assert.deepEqual(snapshot.replay.moves, []);
});

test("bot replay sessions preserve AlphaBetaBot depth", () => {
  const snapshot = gameEngine.startBotReplay(
    { kind: BotKind.AlphaBeta, seed: "101", depth: 2 },
    botConfig(BotKind.Random, "103"),
    0,
  );

  assert.deepEqual(snapshot.replay.players.one, {
    kind: "AlphaBetaBot",
    seed: "101",
    depth: 2,
  });
  assert.equal(snapshot.done, true);
});

test("bot replay sessions preserve JacekInspiredBot model and limits", () => {
  const snapshot = gameEngine.startBotReplay(
    { kind: BotKind.JacekInspired, seed: "107", depth: 3 },
    botConfig(BotKind.Random, "109"),
    0,
  );

  assert.equal(snapshot.replay.players.one.kind, "JacekInspiredBot");
  assert.equal(snapshot.replay.players.one.seed, "107");
  assert.equal(snapshot.replay.players.one.depth, 3);
  assert.equal(snapshot.replay.players.one.maxNodes, 20000);
  assert.equal(snapshot.replay.players.one.maxTimeMs, 0);
  assert.equal(snapshot.replay.players.one.modelSha256, JACEK_MODEL_SHA256);
  assert.equal(snapshot.done, true);
});

test("bot replay sessions preserve the fixed Rank5DerivedBot demo profile", () => {
  const snapshot = gameEngine.startBotReplay(
    { kind: BotKind.Rank5Derived, seed: "107", iterations: 1 },
    botConfig(BotKind.Random, "109"),
    0,
  );

  const configuration = snapshot.replay.players.one;
  assert.equal(configuration.kind, "Rank5DerivedBot");
  assert.equal(configuration.profile, "50k-demo");
  assert.equal(configuration.maxNodes, 50000);
  assert.equal(configuration.modelBlendPercent, 0);
  assert.equal(configuration.replayBookEnabled, false);
  assert.equal(configuration.originalArtifact.sha256, RANK5_ARTIFACT_SHA256);
  assert.equal(snapshot.done, true);
});

test("Rank5Derived-vs-MCTS replay completes with contiguous action diagnostics", () => {
  const snapshot = finishBotReplay(
    { kind: BotKind.Rank5Derived, seed: "0" },
    botConfig(BotKind.Mcts, "23"),
    512,
  );

  assert.equal(snapshot.done, true);
  assert.equal(snapshot.replay.truncated, false);
  assert.equal(snapshot.revision, snapshot.replay.moves.length);
  assert.deepEqual(
    snapshot.replay.moves.map((move) => move.to),
    RANK5_TRANSCRIPT_FIXTURE,
    "the Wasm path should match the shared native transcript fixture",
  );

  const searches = snapshot.botSearches.filter(
    (search) => search.searchType === "rank5Derived",
  );
  let roots = 0;
  let cachedEdges = 0;
  for (let searchIndex = 0; searchIndex < searches.length;) {
    const root = searches[searchIndex];
    roots += 1;
    assert.equal(root.cachedContinuation, false);
    assert.equal(root.requestedNodes, 50000);
    assert.ok(root.visitedNodes > 0 && root.visitedNodes <= root.requestedNodes);
    assert.equal(root.currentEdgeIndex, 0);
    assert.ok(root.plannedActionLength > 0);

    for (let edgeIndex = 1; edgeIndex < root.plannedActionLength; edgeIndex += 1) {
      const continuation = searches[searchIndex + edgeIndex];
      assert.ok(continuation, "every planned Rank5 action edge should be present");
      cachedEdges += 1;
      assert.equal(continuation.cachedContinuation, true);
      assert.equal(continuation.requestedNodes, 50000);
      assert.equal(continuation.visitedNodes, 0);
      assert.equal(continuation.plannedActionLength, root.plannedActionLength);
      assert.equal(continuation.currentEdgeIndex, edgeIndex);
      assert.equal(continuation.ply, root.ply + edgeIndex);
      assert.equal(continuation.player, root.player);
      assert.equal(continuation.completedDepth, root.completedDepth);
      assert.equal(continuation.attemptedDepth, root.attemptedDepth);
      assert.equal(continuation.rootScore, root.rootScore);
      assert.equal(continuation.budgetExhausted, root.budgetExhausted);
    }
    searchIndex += root.plannedActionLength;
  }

  assert.ok(roots > 0, "the completed replay should contain Rank5 root searches");
  assert.ok(cachedEdges > 0, "the completed replay should exercise cached edges");
});

test("the open center of a goal mouth does not grant an extra turn", () => {
  const snapshot = finishBotReplay(
    botConfig(BotKind.Random, "17"),
    botConfig(BotKind.Random, "18"),
    8,
  );

  assert.equal(snapshot.replay.truncated, false);
  assert.equal(snapshot.replay.status, Status.WonByOne);
  assert.deepEqual(snapshot.replay.moves.slice(-2), [
    {
      ply: 6,
      player: Player.Two,
      from: { x: 3, y: 2 },
      to: { x: 4, y: 1 },
      extraTurn: false,
      statusAfter: Status.InProgress,
    },
    {
      ply: 7,
      player: Player.One,
      from: { x: 4, y: 1 },
      to: { x: 4, y: 0 },
      extraTurn: false,
      statusAfter: Status.WonByOne,
    },
  ]);
});

test("a bounded mixed-bot replay is deterministic for identical configurations", () => {
  const one = botConfig(BotKind.Mcts, "17");
  const two = botConfig(BotKind.Random, "23");
  const first = finishBotReplay(one, two, 4);
  const second = finishBotReplay(one, two, 4);

  assert.equal(first.revision, 4);
  assert.equal(first.replay.moves.length, 4);
  assert.equal(first.replay.truncated, true);
  assert.equal(first.replay.status, Status.InProgress);
  assert.equal(first.replay.winner, null);
  assert.deepEqual(first.replay, second.replay);
});

test("bot replay generation is independent from the active live game", () => {
  const live = gameEngine.startGame("37", Player.One, BotKind.Mcts, 8);
  let botReplay = gameEngine.startBotReplay(
    botConfig(BotKind.Random, "41"),
    botConfig(BotKind.Mcts, "43"),
    2,
  );

  assert.deepEqual(gameEngine.snapshot(), live);
  botReplay = gameEngine.playBotReplay(botReplay.sessionId, botReplay.revision);
  assert.deepEqual(gameEngine.snapshot(), live);
  assert.deepEqual(gameEngine.botReplaySnapshot(), botReplay);
});

test("invalid replay configurations do not replace either active session", () => {
  const live = gameEngine.startGame("47", Player.One);
  const botReplay = gameEngine.startBotReplay(
    botConfig(BotKind.Random, "53"),
    botConfig(BotKind.Mcts, "59"),
    2,
  );

  assert.throws(
    () => gameEngine.startBotReplay(
      botConfig(BotKind.Random, "61"),
      botConfig("unknown", "67"),
      2,
    ),
    /Unsupported bot kind: unknown/,
  );
  assert.throws(
    () => gameEngine.startBotReplay(
      botConfig(BotKind.Random, "18446744073709551616"),
      botConfig(BotKind.Mcts, "71"),
      2,
    ),
    /unsigned 64-bit decimal integer/,
  );
  assert.throws(
    () => gameEngine.startBotReplay(
      botConfig(BotKind.Random, "73", 0),
      botConfig(BotKind.Mcts, "79"),
      2,
    ),
    /Player 1 new simulations per bot move must be an integer between 1 and 4294967295/,
  );
  assert.throws(
    () => gameEngine.startBotReplay(
      botConfig(BotKind.Random, "83"),
      botConfig(BotKind.Mcts, "89"),
      -1,
    ),
    /Maximum replay moves must be an integer between 0 and 4294967295/,
  );
  assert.throws(
    () => gameEngine.startGame("97", Player.One, BotKind.Mcts, 1.5),
    /New simulations per bot move must be an integer between 1 and 4294967295/,
  );
  assert.throws(
    () => gameEngine.startGame("101", Player.One, BotKind.AlphaBeta, 65),
    /AlphaBetaBot depth must be an integer between 1 and 64/,
  );
  assert.throws(
    () => gameEngine.startGame(
      "103",
      Player.One,
      BotKind.JacekInspired,
      65,
    ),
    /JacekInspiredBot depth must be an integer between 1 and 64/,
  );

  assert.deepEqual(gameEngine.snapshot(), live);
  assert.deepEqual(gameEngine.botReplaySnapshot(), botReplay);
});
