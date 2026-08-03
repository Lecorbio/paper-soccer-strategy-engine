import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const requestedSearches = positiveInteger(
  process.argv[2],
  80,
  "complete-turn search count",
);
const searchesPerGame = positiveInteger(
  process.argv[3],
  10,
  "complete-turn searches per game",
);
const p95LimitMs = positiveNumber(process.argv[4], 120, "p95 limit");
const maxLimitMs = positiveNumber(process.argv[5], 250, "max limit");

function positiveInteger(raw, fallback, label) {
  const value = Number(raw ?? fallback);
  assert.ok(Number.isSafeInteger(value) && value > 0,
    `${label} must be a positive integer`);
  return value;
}

function positiveNumber(raw, fallback, label) {
  const value = Number(raw ?? fallback);
  assert.ok(Number.isFinite(value) && value > 0,
    `${label} must be a positive number`);
  return value;
}

const moduleUrl = process.env.PAPERSOCCER_WASM_MODULE
  ? pathToFileURL(resolve(process.env.PAPERSOCCER_WASM_MODULE)).href
  : new URL("../web/papersoccer-wasm.js", import.meta.url).href;
const compiledModule = await import(moduleUrl);
globalThis.createPaperSoccerModule = compiledModule.default;
await import("../web/game-engine.js");

const { BotKind, Player, Status, ready } = globalThis.PaperSoccer;
const engine = await ready;
assert.ok(requestedSearches >= 80,
  "the acceptance benchmark requires at least 80 actual searches");
const timings = [];
const nodes = [];
const depths = [];
let games = 0;
let searchesInGame = 0;
let cachedContinuationEdges = 0;
let validatedActionEdges = 0;
let illegalOrRejectedCommands = 0;
let incompleteActions = 0;
let randomState = 1;
let snapshot;

function startGame() {
  const humanPlayer = games % 2 === 0 ? Player.Two : Player.One;
  let started;
  try {
    started = engine.startGame("0", humanPlayer, BotKind.Rank5Derived, 50000);
  } catch (error) {
    illegalOrRejectedCommands += 1;
    throw error;
  }
  games += 1;
  searchesInGame = 0;
  randomState = (0x9e3779b9 ^ games) >>> 0;
  return started;
}

function playHumanEdge(current) {
  assert.equal(current.state.toMove, current.humanPlayer);
  assert.ok(current.legalMoves.length > 0, "human turn must have a legal edge");
  randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
  const move = current.legalMoves[randomState % current.legalMoves.length];
  try {
    return engine.playHuman(current.sessionId, current.revision, move.id);
  } catch (error) {
    illegalOrRejectedCommands += 1;
    throw error;
  }
}

function playCompleteBotAction(current) {
  assert.notEqual(current.state.toMove, current.humanPlayer);
  const mover = current.state.toMove;
  const beforeSearchCount = current.diagnostics.botSearches.length;
  let next;
  try {
    next = engine.playBot(current.sessionId, current.revision);
  } catch (error) {
    illegalOrRejectedCommands += 1;
    throw error;
  }
  const fresh = next.diagnostics.lastBotSearch;
  assert.equal(fresh.searchType, "rank5Derived");
  assert.equal(fresh.cachedContinuation, false,
    "a measured complete action must begin with a fresh search");
  assert.equal(fresh.currentEdgeIndex, 0);
  assert.ok(fresh.plannedActionLength > 0);
  assert.ok(fresh.visitedNodes > 0 && fresh.visitedNodes <= 50000);
  assert.equal(next.diagnostics.botSearches.length, beforeSearchCount + 1);
  assert.ok(Number.isFinite(fresh.decisionTimeNs) && fresh.decisionTimeNs > 0);
  timings.push(fresh.decisionTimeNs / 1_000_000);
  nodes.push(fresh.visitedNodes);
  depths.push(fresh.completedDepth);
  validatedActionEdges += 1;

  for (let edge = 1; edge < fresh.plannedActionLength; edge += 1) {
    if (next.state.status !== Status.InProgress || next.state.toMove !== mover) {
      incompleteActions += 1;
      assert.fail("planned Rank5 action crossed a terminal or turn boundary early");
    }
    try {
      next = engine.playBot(next.sessionId, next.revision);
    } catch (error) {
      illegalOrRejectedCommands += 1;
      throw error;
    }
    const continuation = next.diagnostics.lastBotSearch;
    assert.equal(continuation.searchType, "rank5Derived");
    assert.equal(continuation.cachedContinuation, true,
      "every remaining complete-action edge must use the cache");
    assert.equal(continuation.currentEdgeIndex, edge);
    assert.equal(continuation.plannedActionLength, fresh.plannedActionLength);
    assert.equal(continuation.visitedNodes, 0);
    cachedContinuationEdges += 1;
    validatedActionEdges += 1;
  }

  if (next.state.status === Status.InProgress && next.state.toMove === mover) {
    incompleteActions += 1;
    assert.fail("cached complete action ended without handing off the turn");
  }
  return next;
}

snapshot = startGame();
while (timings.length < requestedSearches) {
  if (snapshot.state.status !== Status.InProgress ||
      searchesInGame >= searchesPerGame) {
    snapshot = startGame();
  } else if (snapshot.state.toMove === snapshot.humanPlayer) {
    snapshot = playHumanEdge(snapshot);
  } else {
    snapshot = playCompleteBotAction(snapshot);
    searchesInGame += 1;
  }
}

assert.equal(illegalOrRejectedCommands, 0);
assert.equal(incompleteActions, 0);

function percentile(values, fraction) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.ceil(sorted.length * fraction) - 1];
}

const p95Milliseconds = percentile(timings, 0.95);
const maxMilliseconds = Math.max(...timings);
const report = {
  schema: "papersoccer.wasm-rank5-derived-benchmark.v1",
  profile: {
    rules: "demo-default",
    max_nodes: 50000,
    max_turn_depth: 32,
    transposition_entries: 65536,
    evaluation_entries: 32768,
    max_time_ms: 0,
    replay_value_blend_percent: 0,
    replay_book_enabled: false,
  },
  requested_complete_turn_searches: requestedSearches,
  actual_complete_turn_searches: timings.length,
  excluded_cached_continuation_edges: cachedContinuationEdges,
  validated_complete_action_edges: validatedActionEdges,
  games,
  searches_per_game_limit: searchesPerGame,
  operational_counts: {
    illegal_or_rejected_commands: illegalOrRejectedCommands,
    incomplete_actions: incompleteActions,
  },
  timing_ms: {
    median: percentile(timings, 0.5),
    p95: p95Milliseconds,
    max: maxMilliseconds,
    samples: timings,
  },
  nodes: {
    median: percentile(nodes, 0.5),
    p95: percentile(nodes, 0.95),
    max: Math.max(...nodes),
  },
  completed_depths: Object.fromEntries(
    [...new Set(depths)].sort((a, b) => a - b).map((depth) => [
      String(depth),
      depths.filter((value) => value === depth).length,
    ]),
  ),
  limits_ms: {
    p95: p95LimitMs,
    max: maxLimitMs,
  },
  passed: p95Milliseconds <= p95LimitMs && maxMilliseconds <= maxLimitMs,
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
assert.ok(p95Milliseconds <= p95LimitMs,
  `Rank5Derived Wasm p95 ${p95Milliseconds.toFixed(3)} ms exceeds ${p95LimitMs} ms`);
assert.ok(maxMilliseconds <= maxLimitMs,
  `Rank5Derived Wasm max ${maxMilliseconds.toFixed(3)} ms exceeds ${maxLimitMs} ms`);
