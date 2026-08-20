import assert from "node:assert/strict";
import test from "node:test";

const [analysisModule, gameplayModule] = await Promise.all([
  import("../../web/papersoccer-analysis-wasm.js"),
  import("../../web/papersoccer-wasm.js"),
]);
globalThis.createPaperSoccerModule = gameplayModule.default;
await import("../../web/game-engine.js");
await import("../../web/game-review-client.js");

const gameplay = await globalThis.PaperSoccer.ready;
const review = globalThis.PaperSoccerGameReview;
const analysisWasm = await analysisModule.default();
const analysis = review.createCabiClient(analysisWasm);
const { BotKind, Player } = globalThis.PaperSoccer;

function terminalReplay() {
  let generated = gameplay.startBotReplay(
    { kind: BotKind.Random, seed: "17" },
    { kind: BotKind.Random, seed: "18" },
    512,
  );
  while (!generated.done) {
    generated = gameplay.playBotReplay(
      generated.sessionId,
      generated.revision,
    );
  }
  assert.equal(generated.replay.truncated, false);
  return generated.replay;
}

function terminalReboundReplay() {
  const endpoints = [
    [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [3, 2],
    [2, 3], [3, 3], [4, 2], [3, 1], [4, 0],
  ];
  const extraTurns = [
    false, false, false, false, false, false,
    false, false, true, true, false,
  ];
  let from = { x: 4, y: 6 };
  let player = "one";
  const moves = endpoints.map(([x, y], index) => {
    const extraTurn = extraTurns[index];
    const move = {
      ply: index + 1,
      player,
      from,
      to: { x, y },
      extraTurn,
      statusAfter: index === endpoints.length - 1
        ? "wonByOne"
        : "inProgress",
    };
    from = move.to;
    if (!extraTurn) {
      player = player === "one" ? "two" : "one";
    }
    return move;
  });
  return {
    schema: "papersoccer.replay.v2",
    rules: { width: 8, height: 10 },
    players: { one: "Rebound fixture", two: "Rebound fixture" },
    start: { x: 4, y: 6 },
    status: "wonByOne",
    winner: "one",
    truncated: false,
    moves,
  };
}

function finish(prepared, mode) {
  let snapshot = review.populateSession(analysis, prepared, mode);
  let steps = 0;
  while (!review.reviewDone(snapshot)) {
    analysis.step();
    snapshot = analysis.snapshot();
    steps += 1;
    assert.ok(steps <= 2 * prepared.moves.length + 2);
  }
  return snapshot;
}

test("the dedicated artifact exposes the complete review C ABI", () => {
  for (const method of [
    "start", "appendMove", "finalize", "step", "snapshot", "cancel",
    "sandboxStart", "sandboxPlay", "sandboxSnapshot", "sandboxClose",
    "lastError", "heapBytes",
  ]) {
    assert.equal(typeof analysis[method], "function");
  }
});

test("an unfinished gameplay export round-trips through authoritative review validation", () => {
  const initial = gameplay.startGame("17", Player.One, BotKind.Random);
  const moved = gameplay.playHuman(initial.sessionId, initial.revision, 0);
  const exported = gameplay.humanMatch();

  assert.equal(moved.state.status, "inProgress");
  assert.equal(exported.schema, "papersoccer.human-match.v1");
  assert.equal(exported.replay.truncated, true);
  assert.equal(exported.replay.status, "inProgress");

  const prepared = review.prepareReplay(exported);
  const snapshot = review.populateSession(analysis, prepared, review.ReviewMode.Fast);
  assert.equal(snapshot.finalized, true);
  assert.equal(snapshot.cancelled, false);
  assert.equal(snapshot.replayStatus, "inProgress");
  assert.equal(snapshot.truncated, true);
  assert.equal(snapshot.possessions.at(-1).truncated, false,
    "ending at a possession boundary truncates the replay, not the possession");
});

test("the pre-lock candidate probe performs exactly one fresh complete-turn search", () => {
  const call = (name, returnType, argumentTypes = [], argumentsList = []) =>
    analysisWasm.ccall(name, returnType, argumentTypes, argumentsList);
  assert.equal(call("ps_analysis_probe_start", "number", ["number"], [100000]), 1);
  for (const point of [[3, 5], [4, 4], [5, 5], [4, 5]]) {
    assert.equal(call(
      "ps_analysis_probe_append_move",
      "number",
      ["number", "number"],
      point,
    ), 1);
  }
  assert.equal(call("ps_analysis_probe_run", "number"), 1);
  const result = JSON.parse(call("ps_analysis_probe_result_json", "string"));
  assert.equal(result.schema, "papersoccer.analysis-probe.v1");
  assert.equal(result.candidateIdentity, "deep-turn-search-100k");
  assert.equal(result.profile.maxNodes, 100000);
  assert.equal(result.openingPlies, 4);
  assert.equal(result.position.possessionBoundary, true);
  assert.equal(result.completeAction, true);
  assert.ok(result.action.length > 0);
  assert.equal(result.diagnostics.nodes, 100000);
  assert.equal(call("ps_analysis_probe_run", "number"), 0);
  assert.match(call("ps_analysis_probe_last_error", "string"), /fresh start/u);
  assert.equal(call("ps_analysis_probe_start", "number", ["number"], [50000]), 0);
  assert.match(call("ps_analysis_probe_last_error", "string"), /100k, 200k, or 400k/u);
});

test("Fast review validates and grades complete possessions deterministically", () => {
  const prepared = review.prepareReplay(terminalReboundReplay());
  const first = finish(prepared, review.ReviewMode.Fast);
  const second = finish(prepared, review.ReviewMode.Fast);

  assert.equal(first.schema, "papersoccer.review-session.v1");
  assert.equal(first.mode, "fast");
  assert.equal(first.complete, true);
  assert.ok(first.possessions.length > 0);
  assert.deepEqual(first, second);
  assert.ok(first.possessions.every((possession) =>
    possession.endPly >= possession.startPly &&
    possession.playedAction.length === possession.edgeCount));
  assert.ok(first.possessions.every((possession) =>
    possession.analyzed === true &&
    ["exact", "deterministic-estimate", "borderline-estimate", "unclear"]
      .includes(possession.confidenceState)));
  assert.ok(first.possessions.some((possession) => possession.edgeCount > 1),
    "the fixture should exercise a rebound possession");
});

test("try-line sandbox starts at the reviewed boundary and remains playable", () => {
  const prepared = review.prepareReplay(terminalReplay());
  const snapshot = finish(prepared, review.ReviewMode.Fast);
  const index = snapshot.possessions.findIndex((possession) =>
    possession.recommendedAction.length > 0 && possession.terminal !== true);
  assert.ok(index >= 0);
  const possession = snapshot.possessions[index];
  const sourceFingerprint = review.canonicalJson(prepared.sourceReplay);

  const opened = analysis.sandboxStart(index);
  assert.equal(opened.schema, "papersoccer.review-sandbox.v1");
  assert.equal(opened.boundaryPly, possession.startPly - 1);
  assert.equal(opened.replay.moves.length, possession.startPly - 1);
  assert.deepEqual(opened.recommendedAction, possession.recommendedAction);
  assert.ok(opened.legalMoves.length > 0);
  assert.throws(
    () => analysis.sandboxPlay({ x: -99, y: -99 }),
    /illegal move/u,
  );
  assert.deepEqual(analysis.sandboxSnapshot(), opened,
    "a rejected sandbox edge must not mutate the fork");

  const continued = analysis.sandboxPlay(opened.legalMoves[0].to);
  assert.equal(continued.replay.moves.length, opened.replay.moves.length + 1);
  assert.equal(
    review.canonicalJson(prepared.sourceReplay),
    sourceFingerprint,
    "the authoritative sandbox must not mutate its source replay",
  );
  assert.equal(analysis.sandboxClose(), 1);
});

test("Deep review includes its Fast comparison and deterministic diagnostics", () => {
  const prepared = review.prepareReplay(terminalReplay());
  const snapshot = finish(prepared, review.ReviewMode.Deep);

  assert.equal(snapshot.mode, "deep");
  assert.equal(snapshot.complete, true);
  assert.match(snapshot.profile.identity, /^deep-(100|200|400)k$/u);
  assert.ok(snapshot.possessions.every((possession) =>
    possession.fastGrade !== undefined));
  assert.ok(snapshot.possessions.every((possession) =>
    Number.isInteger(possession.before.search.completedDepth) &&
    Number.isInteger(possession.before.search.nodes)));
});

test("the locked DeepTurnSearch profile is playable live and in bot replays", () => {
  const live = gameplay.startGame(
    "0",
    globalThis.PaperSoccer.Player.Two,
    BotKind.DeepTurnSearch,
  );
  const configuration = live.diagnostics.botConfiguration;
  assert.equal(configuration.kind, "DeepTurnSearchBot");
  assert.match(configuration.profile, /^deep-(100|200|400)k$/u);
  assert.ok([100000, 200000, 400000].includes(configuration.maxNodes));
  assert.equal(configuration.maxTurnDepth, 32);
  assert.equal(configuration.wallClock, false);
  assert.equal(configuration.replayCorrections, false);
  assert.equal(configuration.learnedValueBlendPercent, 0);

  const moved = gameplay.playBot(live.sessionId, live.revision);
  assert.equal(moved.diagnostics.lastBotSearch.searchType, "deepTurnSearch");
  assert.equal(
    moved.diagnostics.lastBotSearch.requestedNodes,
    configuration.maxNodes,
  );

  let replay = gameplay.startBotReplay(
    { kind: BotKind.DeepTurnSearch },
    { kind: BotKind.Random, seed: "211" },
    512,
  );
  assert.equal(replay.replay.players.one.kind, "DeepTurnSearchBot");
  assert.equal(replay.replay.players.one.maxNodes, configuration.maxNodes);
  replay = gameplay.playBotReplay(replay.sessionId, replay.revision);
  assert.equal(replay.botSearches[0].searchType, "deepTurnSearch");
});

test("illegal moves and inconsistent declarations never reach analysis", () => {
  const replay = terminalReplay();
  const wrongPlayer = structuredClone(replay);
  wrongPlayer.moves[0].player = "two";
  assert.throws(
    () => review.populateSession(
      analysis,
      review.prepareReplay(wrongPlayer),
      review.ReviewMode.Fast,
    ),
    /wrong player/u,
  );

  const wrongExtraTurn = structuredClone(replay);
  wrongExtraTurn.moves[0].extraTurn = !wrongExtraTurn.moves[0].extraTurn;
  assert.throws(
    () => review.populateSession(
      analysis,
      review.prepareReplay(wrongExtraTurn),
      review.ReviewMode.Fast,
    ),
    /extraTurn/u,
  );

  const wrongOutcome = structuredClone(replay);
  wrongOutcome.status = "inProgress";
  wrongOutcome.winner = null;
  wrongOutcome.truncated = true;
  assert.throws(
    () => review.populateSession(
      analysis,
      review.prepareReplay(wrongOutcome),
      review.ReviewMode.Fast,
    ),
    /outcome/u,
  );
});

test("game-review export round trips without nondeterministic timings", () => {
  const prepared = review.prepareReplay(terminalReplay());
  const snapshot = finish(prepared, review.ReviewMode.Fast);
  const exported = review.buildReviewExport(prepared, snapshot);
  const encoded = JSON.stringify(exported);

  assert.equal(exported.schema, "papersoccer.game-review.v1");
  assert.deepEqual(exported.sourceReplay, prepared.sourceReplay);
  assert.equal(exported.replaySha256, prepared.replaySha256);
  assert.doesNotMatch(encoded, /elapsed|latency|decisionTime|wallClock/iu);
  assert.deepEqual(
    review.prepareReplay(exported).sourceReplay,
    prepared.sourceReplay,
  );
});

test("a long frozen truncated replay stays inside the fixed 64 MiB heap", () => {
  const endpoints = (
    "5,5;6,4;7,3;7,4;6,4;6,5;7,4;7,5;8,5;7,6;6,6;5,6;6,7;7,7;6,6;" +
    "6,5;7,6;7,5;8,4;7,4;6,3;6,2;5,3;5,2;4,2;3,3;2,4;2,5;3,4;4,3;4,2;" +
    "3,1;2,2;1,2;1,3;2,3;1,4;1,5;2,6;2,7;1,7;2,8;3,8;3,7;2,7;1,8;1,9;" +
    "1,10;2,9;1,9;2,10;3,10;2,11;2,10;3,9;4,10;3,10;4,9;5,10;6,10;7,9;7,8;" +
    "6,9;5,8;4,7;4,8;5,9;6,8;5,7;6,7;6,8;7,8;7,7;7,6;6,7;5,8;4,8;3,9;2,8;" +
    "1,9;0,9;1,10;1,11;2,10;2,9;1,8;1,7;1,6;2,6;3,5;4,5;5,6;5,7;4,7;4,6;4,5;" +
    "4,4;5,4;5,5;4,4;5,3;4,3;3,2;2,2;2,1;1,2;1,1;2,2;3,3;3,4;3,5;3,6;4,6;3,5;" +
    "2,5;1,5;2,4;1,3;1,4;2,5;3,6;3,7;2,8;1,8;0,7;1,6;1,5;0,4;1,4;2,4;2,3;3,2;" +
    "4,1;5,2;6,3;7,2;7,3;6,2;7,1;7,2;8,2;7,3;6,3;6,4;5,4;4,3;4,4;3,4;2,3;2,2;" +
    "1,3;0,2;1,2;0,3;1,4;0,5;1,6;0,6;1,7;0,8;1,9;0,10;1,10;2,10;3,11;4,11;" +
    "5,10;6,9;6,10;7,10;8,10;7,9;8,9;7,10;6,9;5,9;4,9;4,10;3,11;3,10;2,9;3,9;" +
    "3,8;4,7;3,7;4,8;4,9;3,9;3,10;4,11;5,11;5,10;5,9;5,8;6,8;6,9;7,9;6,8;7,7;" +
    "8,8;7,8;8,7;7,7;8,6;7,5;6,5;5,6;5,5;4,5;5,4;5,3;6,4;7,5;6,6;5,7;5,8;4,9;" +
    "3,8;2,7;3,6;2,6;2,5;1,6;2,7;2,8;2,9;3,8;4,8;5,7;4,6;5,6;4,7;3,6;4,5;3,4;" +
    "2,4;3,5;4,4;3,3;4,3;5,2;5,1;4,2;3,2;3,1;4,1;4,2;5,3;6,3;5,4;6,5;5,5;6,6;" +
    "6,7;7,8;8,9"
  ).split(";").map(function (encoded) {
    const [x, y] = encoded.split(",").map(Number);
    return { x, y };
  });
  assert.equal(endpoints.length, 256);

  const visits = new Set(["4,6"]);
  const moves = [];
  let from = { x: 4, y: 6 };
  let player = "one";
  for (const [index, to] of endpoints.slice(0, -1).entries()) {
    const key = `${to.x},${to.y}`;
    const boundary = to.x === 0 || to.x === 8 ||
      ((to.y === 1 || to.y === 11) && to.x !== 4);
    const extraTurn = boundary || visits.has(key);
    moves.push({
      ply: index + 1,
      player,
      from,
      to,
      extraTurn,
      statusAfter: "inProgress",
    });
    visits.add(key);
    from = to;
    if (!extraTurn) {
      player = player === "one" ? "two" : "one";
    }
  }
  const replay = {
    schema: "papersoccer.replay.v2",
    rules: { width: 8, height: 10 },
    players: { one: "Frozen long path", two: "Frozen long path" },
    start: { x: 4, y: 6 },
    status: "inProgress",
    winner: null,
    truncated: true,
    moves,
  };
  assert.equal(replay.moves.length, 255);
  assert.equal(analysis.heapBytes(), 64 * 1024 * 1024);
  const snapshot = finish(
    review.prepareReplay(replay),
    review.ReviewMode.Fast,
  );
  assert.equal(snapshot.complete, true);
  assert.equal(snapshot.possessions.at(-1).truncated, true);
  assert.equal(analysis.heapBytes(), 64 * 1024 * 1024);
});
