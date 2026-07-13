import assert from "node:assert/strict";
import test from "node:test";

const compiledModule = await import("../web/papersoccer-wasm.js");
globalThis.createPaperSoccerModule = compiledModule.default;
await import("../web/game-engine.js");

const { Player, Status, ready } = globalThis.PaperSoccer;
const gameEngine = await ready;

test("browser client exposes the compiled C++ session instead of JavaScript rules", () => {
  assert.equal(Player.One, "one");
  assert.equal(Player.Two, "two");
  assert.equal(Status.InProgress, "inProgress");
  assert.equal(typeof gameEngine.startGame, "function");
  assert.equal(typeof gameEngine.playHuman, "function");
  assert.equal(typeof gameEngine.playBot, "function");
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
