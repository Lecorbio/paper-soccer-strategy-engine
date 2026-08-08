import assert from "node:assert/strict";
import test from "node:test";

await import("../../web/game-review-client.js");

const review = globalThis.PaperSoccerGameReview;

function replayFixture() {
  return {
    schema: "papersoccer.replay.v2",
    rules: { width: 8, height: 10 },
    players: {
      one: { kind: "Human" },
      two: { kind: "RandomBot", seed: "17" },
    },
    start: { x: 4, y: 6 },
    status: "inProgress",
    winner: null,
    truncated: true,
    moves: [{
      ply: 1,
      player: "one",
      from: { x: 4, y: 6 },
      to: { x: 4, y: 5 },
      extraTurn: false,
      statusAfter: "inProgress",
    }],
  };
}

test("replay preparation preserves the source and creates a stable SHA-256", () => {
  const source = replayFixture();
  const prepared = review.prepareReplay(source);

  assert.deepEqual(prepared.sourceReplay, source);
  assert.notEqual(prepared.sourceReplay, source);
  assert.equal(prepared.replaySha256.length, 64);
  assert.match(prepared.replaySha256, /^[0-9a-f]{64}$/u);

  const reordered = {
    moves: source.moves,
    truncated: source.truncated,
    winner: source.winner,
    status: source.status,
    start: source.start,
    players: source.players,
    rules: source.rules,
    schema: source.schema,
  };
  assert.equal(review.prepareReplay(reordered).replaySha256,
    prepared.replaySha256);
  assert.equal(
    review.sha256("abc"),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
});

test("human-match and game-review wrappers retain their exact source replay", () => {
  const source = replayFixture();
  const human = {
    schema: "papersoccer.human-match.v1",
    replay: source,
    botSearches: [{ decisionTimeNs: 42 }],
  };
  const exported = {
    schema: "papersoccer.game-review.v1",
    sourceReplay: source,
    replaySha256: review.prepareReplay(source).replaySha256,
    possessions: [],
  };

  assert.deepEqual(review.prepareReplay(human).sourceReplay, source);
  assert.deepEqual(review.prepareReplay(exported).sourceReplay, source);
  assert.throws(
    () => review.prepareReplay({ schema: "papersoccer.game-review.v1" }),
    /does not contain its source replay/u,
  );
  assert.throws(
    () => review.prepareReplay({ ...exported, replaySha256: "0".repeat(64) }),
    /SHA-256 does not match/u,
  );
});

test("preparation rejects malformed declarations instead of normalizing them", () => {
  const cases = [
    [() => ({ ...replayFixture(), rules: { width: 9, height: 10 } }), /8×10/u],
    [() => ({ ...replayFixture(), start: { x: 4, y: 5 } }), /\(4, 6\)/u],
    [() => ({ ...replayFixture(), moves: null }), /moves must be an array/u],
    [() => ({
      ...replayFixture(),
      moves: [{ ...replayFixture().moves[0], player: "unknown" }],
    }), /must be Player 1 or Player 2/u],
    [() => ({
      ...replayFixture(),
      moves: [{ ...replayFixture().moves[0], statusAfter: "draw" }],
    }), /unsupported status/u],
    [() => ({ ...replayFixture(), truncated: "false" }), /true or false/u],
  ];
  for (const [make, expected] of cases) {
    assert.throws(() => review.prepareReplay(make()), expected);
  }
});

test("standard zero-move and legacy starts are validated after conversion", () => {
  const empty = {
    ...replayFixture(),
    moves: [],
  };
  assert.deepEqual(review.prepareReplay(empty).start, { x: 4, y: 6 });

  const legacy = {
    ...empty,
    schema: "papersoccer.replay.v1",
    start: { x: 4, y: 5 },
  };
  assert.deepEqual(review.prepareReplay(legacy).start, { x: 4, y: 6 });
});

test("the C ABI adapter forwards every declared move field", () => {
  const calls = [];
  const snapshot = {
    schema: "papersoccer.review-session.v1",
    finalized: true,
    complete: false,
    completedPossessions: 0,
    totalPossessions: 1,
    possessions: [],
  };
  const module = {
    ccall(name, returnType, argumentTypes = [], argumentsList = []) {
      calls.push({ name, returnType, argumentTypes, argumentsList });
      if (name === "ps_review_snapshot_json") {
        return JSON.stringify(snapshot);
      }
      if (name === "ps_review_last_error") {
        return "test error";
      }
      return 1;
    },
  };
  const cabi = review.createCabiClient(module);
  const prepared = review.prepareReplay(replayFixture());

  review.populateSession(cabi, prepared, review.ReviewMode.Deep);

  assert.deepEqual(
    calls.find((call) => call.name === "ps_review_start").argumentsList,
    [1],
  );
  assert.deepEqual(
    calls.find((call) => call.name === "ps_review_append_move").argumentsList,
    [1, 1, 4, 6, 4, 5, 0, 0],
  );
  assert.deepEqual(
    calls.find((call) => call.name === "ps_review_finalize").argumentsList,
    [0, 0, 1],
  );
});

test("the C ABI adapter owns an authoritative playable try-line fork", () => {
  const calls = [];
  let sandboxPly = 4;
  const module = {
    ccall(name, returnType, argumentTypes = [], argumentsList = []) {
      calls.push({ name, returnType, argumentTypes, argumentsList });
      if (name === "ps_review_sandbox_snapshot_json") {
        return JSON.stringify({
          schema: "papersoccer.review-sandbox.v1",
          state: {
            ball: sandboxPly === 4 ? { x: 4, y: 4 } : { x: 5, y: 4 },
            toMove: "two",
            status: "inProgress",
          },
          legalMoves: [{ id: 0, to: { x: 5, y: 4 }, extraTurn: false }],
          replay: { moves: Array.from({ length: sandboxPly }) },
        });
      }
      if (name === "ps_review_sandbox_play") {
        sandboxPly += 1;
      }
      if (name === "ps_review_last_error") {
        return "sandbox error";
      }
      return 1;
    },
  };
  const cabi = review.createCabiClient(module);

  const opened = cabi.sandboxStart(2);
  const continued = cabi.sandboxPlay({ x: 5, y: 4 });
  assert.equal(opened.replay.moves.length, 4);
  assert.equal(continued.replay.moves.length, 5);
  assert.deepEqual(
    calls.find((call) => call.name === "ps_review_sandbox_start").argumentsList,
    [2],
  );
  assert.deepEqual(
    calls.find((call) => call.name === "ps_review_sandbox_play").argumentsList,
    [5, 4],
  );
  assert.equal(cabi.sandboxClose(), 1);
});

test("game-review exports omit wall-clock fields and round-trip the source", () => {
  const prepared = review.prepareReplay(replayFixture());
  const snapshot = {
    schema: "papersoccer.review-session.v1",
    mode: "deep",
    analyzer: { identity: "complete-turn-v1", decisionTimeNs: 123 },
    profile: { identity: "deep-200k", wallClockMs: 0 },
    calibration: { identity: "validation-deep-200k" },
    oracle: { identity: "exact-18-edges", elapsedMs: 1 },
    rankedSource: { sha256: "f29959" },
    summary: { bestActions: 1 },
    possessions: [{ grade: "Best", durationMs: 7 }],
  };

  const exported = review.buildReviewExport(prepared, snapshot);

  assert.equal(exported.schema, "papersoccer.game-review.v1");
  assert.deepEqual(exported.sourceReplay, prepared.sourceReplay);
  assert.equal(exported.replaySha256, prepared.replaySha256);
  assert.equal(exported.analyzer.decisionTimeNs, undefined);
  assert.equal(exported.profile.wallClockMs, undefined);
  assert.equal(exported.oracle.elapsedMs, undefined);
  assert.equal(exported.possessions[0].durationMs, undefined);
  assert.deepEqual(review.prepareReplay(exported).sourceReplay,
    prepared.sourceReplay);
});

test("direct-file fallback schedules one synchronous step per task", async () => {
  const timers = [];
  const events = [];
  let steps = 0;
  let completed = 0;
  const timerHost = {
    setTimeout(callback) {
      timers.push(callback);
      return timers.length;
    },
    clearTimeout() {},
  };
  const module = {
    ccall(name) {
      if (name === "ps_review_snapshot_json") {
        return JSON.stringify({
          schema: "papersoccer.review-session.v1",
          finalized: true,
          complete: completed >= 2,
          completedPossessions: completed,
          totalPossessions: 2,
          possessions: [],
        });
      }
      if (name === "ps_review_last_error") {
        return "fallback error";
      }
      if (name === "ps_review_step") {
        steps += 1;
        completed += 1;
      }
      return 1;
    },
  };
  const fakeGlobal = {};
  const runner = review.createAnalysisRunner({
    globalObject: fakeGlobal,
    documentObject: {},
    timerHost,
    protocol: "file:",
    WorkerConstructor: function () {
      throw new Error("file worker blocked");
    },
    loadScript: async function () {
      fakeGlobal.createPaperSoccerAnalysisModule = async () => module;
    },
    onEvent(event) { events.push(event); },
  });

  runner.start(replayFixture(), review.ReviewMode.Fast);
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(runner.transport, "main-thread-fallback");
  assert.equal(steps, 0);
  assert.equal(timers.length, 1);
  timers.shift()();
  assert.equal(steps, 1);
  assert.equal(timers.length, 1);
  timers.shift()();
  assert.equal(steps, 2);
  assert.ok(events.some((event) => event.type === "progress"));
  assert.ok(events.some((event) => event.type === "complete"));
});

test("cancelled fallback work rejects stale queued results", async () => {
  const timers = [];
  const events = [];
  let cancelCalls = 0;
  const timerHost = {
    setTimeout(callback) {
      timers.push(callback);
      return timers.length;
    },
    clearTimeout() {},
  };
  const module = {
    ccall(name) {
      if (name === "ps_review_snapshot_json") {
        return JSON.stringify({
          complete: false,
          completedPossessions: 0,
          totalPossessions: 1,
          possessions: [],
        });
      }
      if (name === "ps_review_last_error") {
        return "cancel error";
      }
      if (name === "ps_review_cancel") {
        cancelCalls += 1;
      }
      return 1;
    },
  };
  const fakeGlobal = {};
  const runner = review.createAnalysisRunner({
    globalObject: fakeGlobal,
    documentObject: {},
    timerHost,
    protocol: "file:",
    WorkerConstructor: function () { throw new Error("blocked"); },
    loadScript: async function () {
      fakeGlobal.createPaperSoccerAnalysisModule = async () => module;
    },
    onEvent(event) { events.push(event); },
  });

  const first = runner.start(replayFixture());
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(runner.cancel(), true);
  for (const callback of timers.splice(0)) {
    callback();
  }

  assert.equal(cancelCalls, 1);
  assert.ok(events.some((event) =>
    event.type === "cancelled" && event.sessionId === first.sessionId));
  assert.equal(events.some((event) =>
    event.type === "complete" && event.sessionId === first.sessionId), false);
});

test("hosted worker protocol rejects stale sessions and resolves validation", async () => {
  const events = [];
  const workers = [];
  class MockClassicWorker {
    constructor(...argumentsList) {
      this.argumentsList = argumentsList;
      this.messages = [];
      this.terminated = false;
      workers.push(this);
    }

    postMessage(message) {
      this.messages.push(message);
    }

    terminate() {
      this.terminated = true;
    }

    emit(message) {
      this.onmessage({ data: message });
    }
  }

  const runner = review.createAnalysisRunner({
    globalObject: {},
    documentObject: {},
    timerHost: globalThis,
    protocol: "https:",
    WorkerConstructor: MockClassicWorker,
    onEvent(event) { events.push(event); },
  });
  const first = runner.start(replayFixture(), review.ReviewMode.Fast);
  assert.equal(workers.length, 1);
  assert.deepEqual(workers[0].argumentsList, ["game-review-worker.js"]);
  workers[0].emit({ type: "ready" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(workers[0].messages[0].type, "start");
  assert.equal(workers[0].messages[0].sessionId, first.sessionId);

  const second = runner.start(replayFixture(), review.ReviewMode.Deep);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(
    workers[0].messages.slice(1).map((message) => message.type),
    ["cancel", "start"],
  );
  workers[0].emit({
    type: "complete",
    sessionId: first.sessionId,
    snapshot: { complete: true, possessions: [] },
  });
  workers[0].emit({
    type: "progress",
    sessionId: second.sessionId,
    snapshot: { complete: false, possessions: [] },
  });
  workers[0].emit({
    type: "complete",
    sessionId: second.sessionId,
    snapshot: { complete: true, possessions: [] },
  });
  assert.equal(events.some((event) =>
    event.type === "complete" && event.sessionId === first.sessionId), false);
  assert.ok(events.some((event) =>
    event.type === "progress" && event.sessionId === second.sessionId));
  assert.ok(events.some((event) =>
    event.type === "complete" && event.sessionId === second.sessionId));

  const sandbox = runner.startSandbox(1);
  await new Promise((resolve) => setImmediate(resolve));
  const sandboxStart = workers[0].messages.at(-1);
  assert.equal(sandboxStart.type, "sandbox-start");
  assert.equal(sandboxStart.possession, 1);
  const openedSnapshot = {
    schema: "papersoccer.review-sandbox.v1",
    state: { ball: { x: 3, y: 4 }, status: "inProgress" },
    legalMoves: [{ id: 0, to: { x: 4, y: 4 }, extraTurn: false }],
    replay: { moves: [{ ply: 1 }] },
  };
  workers[0].emit({
    type: "sandbox",
    sessionId: sandboxStart.sessionId,
    requestId: sandboxStart.requestId,
    snapshot: openedSnapshot,
  });
  assert.deepEqual(await sandbox, openedSnapshot);

  const continuation = runner.playSandbox({ x: 4, y: 4 });
  await new Promise((resolve) => setImmediate(resolve));
  const sandboxPlay = workers[0].messages.at(-1);
  assert.equal(sandboxPlay.type, "sandbox-play");
  const continuedSnapshot = {
    ...openedSnapshot,
    state: { ball: { x: 4, y: 4 }, status: "inProgress" },
    replay: { moves: [{ ply: 1 }, { ply: 2 }] },
  };
  workers[0].emit({
    type: "sandbox",
    sessionId: sandboxPlay.sessionId,
    requestId: sandboxPlay.requestId,
    snapshot: continuedSnapshot,
  });
  assert.deepEqual(await continuation, continuedSnapshot);
  assert.equal(runner.closeSandbox(), true);
  assert.equal(workers[0].messages.at(-1).type, "sandbox-close");

  const validation = runner.validate(replayFixture());
  await new Promise((resolve) => setImmediate(resolve));
  const validateMessage = workers[0].messages.at(-1);
  assert.equal(validateMessage.type, "validate");
  workers[0].emit({
    type: "validated",
    sessionId: validateMessage.sessionId,
    snapshot: { finalized: true, complete: false },
  });
  assert.deepEqual(await validation, { finalized: true, complete: false });
  assert.equal(runner.transport, "worker");

  runner.dispose();
  assert.equal(workers[0].terminated, true);
});
