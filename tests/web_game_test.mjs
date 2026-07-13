import assert from "node:assert/strict";
import test from "node:test";

await import("../web/game-engine.js");

const {
  Player,
  Status,
  RandomBot,
  applyMove,
  cloneState,
  grantsExtraTurn,
  isTerminal,
  legalMoves,
  makeInitialState,
  pointKey,
  segmentKey,
  winner,
} = globalThis.PaperSoccer;

function destinations(state) {
  return legalMoves(state).map((move) => pointKey(move.to));
}

function makeCleanStateAt(point, toMove = Player.One) {
  const state = makeInitialState();
  state.ball = { ...point };
  state.toMove = toMove;
  state.status = Status.InProgress;
  state.path = [{ ...point }];
  state.usedSegments.clear();
  state.visitCount.clear();
  state.visitCount.set(pointKey(point), 1);
  return state;
}

test("classic script exposes the expected browser API", () => {
  assert.equal(Player.One, "one");
  assert.equal(Player.Two, "two");
  assert.equal(Status.InProgress, "inProgress");
  assert.equal(typeof globalThis.PaperSoccer.makeInitialState, "function");
  assert.equal(typeof globalThis.PaperSoccer.RandomBot, "function");
});

test("initial state and legal move ordering match the C++ engine", () => {
  const state = makeInitialState();

  assert.deepEqual(state.ball, { x: 4, y: 6 });
  assert.equal(state.toMove, Player.One);
  assert.equal(state.status, Status.InProgress);
  assert.deepEqual(destinations(state), [
    "3,5",
    "3,6",
    "3,7",
    "4,5",
    "4,7",
    "5,5",
    "5,6",
    "5,7",
  ]);
});

test("used edges and outer walls cannot be traversed", () => {
  let state = makeInitialState();
  state = applyMove(state, { to: { x: 4, y: 5 } });
  assert.equal(destinations(state).includes("4,6"), false);
  assert.throws(
    () => applyMove(state, { to: { x: 4, y: 6 } }),
    /illegal move/,
  );

  const wallState = makeCleanStateAt({ x: 0, y: 6 });
  assert.equal(destinations(wallState).includes("0,5"), false);
  assert.equal(destinations(wallState).includes("0,7"), false);
  assert.equal(destinations(wallState).includes("1,6"), true);
});

test("visited points and field boundaries grant rebounds", () => {
  let state = makeInitialState();
  state = applyMove(state, { to: { x: 4, y: 5 } });
  state = applyMove(state, { to: { x: 5, y: 5 } });
  state = applyMove(state, { to: { x: 5, y: 6 } });

  assert.equal(state.toMove, Player.Two);
  assert.equal(grantsExtraTurn(state, { x: 4, y: 6 }), true);
  state = applyMove(state, { to: { x: 4, y: 6 } });
  assert.equal(state.toMove, Player.Two);

  let boundaryState = makeCleanStateAt({ x: 4, y: 2 });
  assert.equal(grantsExtraTurn(boundaryState, { x: 4, y: 1 }), true);
  boundaryState = applyMove(boundaryState, { to: { x: 4, y: 1 } });
  assert.equal(boundaryState.toMove, Player.One);
});

test("goal mouths, posts, direction, and winning moves match C++", () => {
  const leftMouth = makeCleanStateAt({ x: 3, y: 1 }, Player.One);
  assert.equal(destinations(leftMouth).includes("3,0"), false);
  assert.equal(destinations(leftMouth).includes("4,0"), true);
  assert.equal(destinations(leftMouth).includes("5,0"), false);

  const centerMouth = makeCleanStateAt({ x: 4, y: 1 }, Player.One);
  assert.deepEqual(
    destinations(centerMouth).filter((key) => key.endsWith(",0")),
    ["3,0", "4,0", "5,0"],
  );

  const wrongDirection = makeCleanStateAt({ x: 4, y: 1 }, Player.Two);
  assert.equal(destinations(wrongDirection).some((key) => key.endsWith(",0")), false);

  const northWin = applyMove(centerMouth, { to: { x: 4, y: 0 } });
  assert.equal(isTerminal(northWin), true);
  assert.equal(northWin.status, Status.WonByOne);
  assert.equal(winner(northWin), Player.One);
  assert.deepEqual(northWin.ball, { x: 4, y: 0 });
  assert.deepEqual(legalMoves(northWin), []);

  const southMouth = makeCleanStateAt({ x: 4, y: 11 }, Player.Two);
  const southWin = applyMove(southMouth, { to: { x: 4, y: 12 } });
  assert.equal(southWin.status, Status.WonByTwo);
  assert.equal(winner(southWin), Player.Two);
});

test("a player with no continuation loses, including after a rebound", () => {
  const blockers = [
    { x: 3, y: 4 },
    { x: 4, y: 4 },
    { x: 5, y: 4 },
    { x: 3, y: 5 },
    { x: 5, y: 5 },
    { x: 3, y: 6 },
    { x: 5, y: 6 },
  ];

  const trapForNextPlayer = makeCleanStateAt({ x: 4, y: 6 }, Player.One);
  for (const blocker of blockers) {
    trapForNextPlayer.usedSegments.add(segmentKey({ x: 4, y: 5 }, blocker));
  }
  const winningTrap = applyMove(trapForNextPlayer, { to: { x: 4, y: 5 } });
  assert.equal(winningTrap.status, Status.WonByOne);
  assert.equal(winner(winningTrap), Player.One);

  const reboundTrap = cloneState(trapForNextPlayer);
  reboundTrap.visitCount.set("4,5", 1);
  const losingTrap = applyMove(reboundTrap, { to: { x: 4, y: 5 } });
  assert.equal(losingTrap.toMove, Player.One);
  assert.equal(losingTrap.status, Status.WonByTwo);
  assert.equal(winner(losingTrap), Player.Two);
});

test("state transitions are pure and do not share mutable state", () => {
  const original = makeInitialState();
  const snapshot = cloneState(original);
  const firstBranch = applyMove(original, { to: { x: 3, y: 5 } });
  const secondBranch = applyMove(original, { to: { x: 5, y: 7 } });

  assert.deepEqual(original, snapshot);
  assert.notEqual(firstBranch.config, original.config);
  assert.notEqual(firstBranch.ball, original.ball);
  assert.notEqual(firstBranch.path, original.path);
  assert.notEqual(firstBranch.path[0], original.path[0]);
  assert.notEqual(firstBranch.usedSegments, original.usedSegments);
  assert.notEqual(firstBranch.visitCount, original.visitCount);

  firstBranch.config.width = 99;
  firstBranch.ball.x = 99;
  firstBranch.path[0].x = 99;
  firstBranch.usedSegments.add("unrelated");
  firstBranch.visitCount.set("unrelated", 99);

  assert.deepEqual(original, snapshot);
  assert.equal(secondBranch.config.width, 8);
  assert.deepEqual(secondBranch.ball, { x: 5, y: 7 });
  assert.equal(secondBranch.path[0].x, 4);
  assert.equal(secondBranch.usedSegments.has("unrelated"), false);
  assert.equal(secondBranch.visitCount.has("unrelated"), false);

  assert.throws(
    () => applyMove(original, { to: { x: 8, y: 8 } }),
    /illegal move/,
  );
  assert.deepEqual(original, snapshot);
});

test("SplitMix64 outputs match the C++ RandomBot implementation", () => {
  const bot = new RandomBot(0n);
  assert.equal(bot.nextRandom(), 0xe220a8397b1dcdafn);
  assert.equal(bot.nextRandom(), 0x6e789e6aa1b965f4n);
  assert.equal(bot.nextRandom(), 0x06c45d188009454fn);

  const wrappedSeed = new RandomBot(1n << 64n);
  assert.equal(wrappedSeed.nextRandom(), 0xe220a8397b1dcdafn);
  assert.equal(new RandomBot("17").name(), "RandomBot");
});

test("seeded RandomBot self-play matches a C++ replay exactly", () => {
  const bots = {
    [Player.One]: new RandomBot(17),
    [Player.Two]: new RandomBot(18),
  };
  let state = makeInitialState();
  let plies = 0;

  while (!isTerminal(state)) {
    const move = bots[state.toMove].chooseMove(state);
    state = applyMove(state, move);
    plies += 1;
    assert.ok(plies <= 256, "seeded self-play should terminate");
  }

  assert.deepEqual(state.path.map(pointKey), [
    "4,6",
    "4,5",
    "3,5",
    "4,4",
    "3,3",
    "3,2",
    "4,1",
    "4,2",
    "5,2",
    "6,3",
    "6,2",
    "5,2",
    "5,3",
    "5,4",
    "4,5",
    "5,5",
    "5,6",
    "4,6",
    "3,5",
    "3,4",
    "2,3",
    "2,4",
    "1,3",
    "0,3",
    "1,4",
    "0,5",
    "1,5",
    "0,4",
    "1,3",
    "0,2",
    "1,1",
    "1,2",
    "0,3",
  ]);
  assert.equal(plies, 32);
  assert.equal(state.status, Status.WonByTwo);
  assert.equal(winner(state), Player.Two);
});

test("RandomBot rejects states without a legal move", () => {
  const terminal = makeInitialState();
  terminal.status = Status.WonByOne;
  assert.throws(
    () => new RandomBot(5).chooseMove(terminal),
    /without legal moves/,
  );
});
