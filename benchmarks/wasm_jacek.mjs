import assert from "node:assert/strict";

const requestedDecisions = Number(process.argv[2] ?? 80);
const requestedDepth = Number(process.argv[3] ?? 6);
const decisionsPerGame = Number(process.argv[4] ?? 10);
assert.ok(Number.isInteger(requestedDecisions) && requestedDecisions > 0,
  "decision count must be a positive integer");
assert.ok(Number.isInteger(requestedDepth) && requestedDepth >= 1 &&
  requestedDepth <= 64, "depth must be an integer between 1 and 64");
assert.ok(Number.isInteger(decisionsPerGame) && decisionsPerGame > 0,
  "decisions per game must be a positive integer");

const compiledModule = await import("../web/papersoccer-wasm.js");
globalThis.createPaperSoccerModule = compiledModule.default;
await import("../web/game-engine.js");

const { BotKind, Player, Status, ready } = globalThis.PaperSoccer;
const engine = await ready;
const timings = [];
const nodes = [];
const depths = [];
let budgetExhaustions = 0;
let invalidRootScores = 0;
let games = 0;
let decisionsInGame = 0;
let humanRandomState = 0;
let snapshot = null;
const uniquePositions = new Set();

function startGame() {
  const gameIndex = games;
  const humanPlayer = gameIndex % 2 === 0 ? Player.Two : Player.One;
  const started = engine.startGame(
    String(1000 + gameIndex),
    humanPlayer,
    BotKind.JacekInspired,
    requestedDepth,
  );
  games += 1;
  decisionsInGame = 0;
  humanRandomState = (0x9e3779b9 ^ gameIndex) >>> 0;
  return started;
}

snapshot = startGame();
while (timings.length < requestedDecisions) {
  if (snapshot.state.status !== Status.InProgress ||
      decisionsInGame >= decisionsPerGame) {
    snapshot = startGame();
  } else if (snapshot.state.toMove === snapshot.humanPlayer) {
    humanRandomState =
      (Math.imul(humanRandomState, 1664525) + 1013904223) >>> 0;
    const move = snapshot.legalMoves[
      humanRandomState % snapshot.legalMoves.length
    ];
    snapshot = engine.playHuman(
      snapshot.sessionId,
      snapshot.revision,
      move.id,
    );
  } else {
    uniquePositions.add(JSON.stringify(snapshot.state));
    snapshot = engine.playBot(snapshot.sessionId, snapshot.revision);
    const search = snapshot.diagnostics.lastBotSearch;
    assert.equal(search.searchType, "alphaBeta");
    timings.push(search.decisionTimeNs / 1_000_000);
    nodes.push(search.nodes);
    depths.push(search.completedDepth);
    budgetExhaustions += search.budgetExhausted ? 1 : 0;
    invalidRootScores += search.rootScoreValid ? 0 : 1;
    decisionsInGame += 1;
  }
}

function percentile(values, fraction) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.ceil(sorted.length * fraction) - 1];
}

const depthHistogram = Object.fromEntries(
  [...new Set(depths)]
    .sort((left, right) => left - right)
    .map((depth) => [
      String(depth),
      depths.filter((value) => value === depth).length,
    ]),
);
const timingByDepth = Object.fromEntries(
  Object.keys(depthHistogram).map((depth) => {
    const values = timings.filter((_, index) =>
      depths[index] === Number(depth));
    return [depth, {
      decisions: values.length,
      median: percentile(values, 0.5),
      p95: percentile(values, 0.95),
      max: Math.max(...values),
    }];
  }),
);

console.log(JSON.stringify({
  schema: "papersoccer.wasm-jacek-benchmark.v1",
  requested_decisions: requestedDecisions,
  requested_depth: requestedDepth,
  decisions_per_game: decisionsPerGame,
  games,
  unique_positions: uniquePositions.size,
  model_sha256: snapshot.diagnostics.botConfiguration.modelSha256,
  search_limits: {
    max_nodes: snapshot.diagnostics.botConfiguration.maxNodes,
    max_time_ms: snapshot.diagnostics.botConfiguration.maxTimeMs,
  },
  timing_ms: {
    median: percentile(timings, 0.5),
    p95: percentile(timings, 0.95),
    max: Math.max(...timings),
  },
  timing_ms_by_completed_depth: timingByDepth,
  nodes: {
    median: percentile(nodes, 0.5),
    p95: percentile(nodes, 0.95),
    max: Math.max(...nodes),
  },
  completed_depth_histogram: depthHistogram,
  budget_exhaustions: budgetExhaustions,
  invalid_root_scores: invalidRootScores,
}, null, 2));
