import assert from "node:assert/strict";
import test from "node:test";

const compiledModule = await import("../web/papersoccer-wasm.js");
globalThis.createPaperSoccerModule = compiledModule.default;
await import("../web/game-engine.js");

const { BotKind, Player, Status, ready } = globalThis.PaperSoccer;
const gameEngine = await ready;

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
  assert.equal(typeof gameEngine.startGame, "function");
  assert.equal(typeof gameEngine.playHuman, "function");
  assert.equal(typeof gameEngine.playBot, "function");
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
  const second = gameEngine.startGame(
    "18446744073709551615",
    Player.Two,
    BotKind.Mcts,
    8,
  );
  const secondMove = gameEngine.playBot(second.sessionId, second.revision);

  assert.equal(firstMove.replay.moves.length, 1);
  assert.deepEqual(firstMove.replay.moves, secondMove.replay.moves);
});

test("live MctsBot games default to 2000 iterations", () => {
  const snapshot = gameEngine.startGame("19", Player.One, BotKind.Mcts);

  assert.deepEqual(snapshot.replay.players.two, {
    kind: "MctsBot",
    seed: "19",
    iterations: 2000,
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
    /Player 1 bot iterations must be an integer between 1 and 4294967295/,
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
    /Bot iterations must be an integer between 1 and 4294967295/,
  );

  assert.deepEqual(gameEngine.snapshot(), live);
  assert.deepEqual(gameEngine.botReplaySnapshot(), botReplay);
});
