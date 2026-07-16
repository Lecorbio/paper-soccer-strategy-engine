import assert from "node:assert/strict";
import test from "node:test";

await import("../web/app-support.js");

const support = globalThis.PaperSoccerWebSupport;

function controls() {
  return {
    botSelect: {},
    sideSelect: {},
    seedInput: {},
    budgetInput: {},
    depthInput: {},
    startButton: {},
    changeSettingsButton: {},
    exportButton: {},
  };
}

test("human-match exports unwrap to their standard replay", () => {
  const replay = { schema: "papersoccer.replay.v2", moves: [] };
  const wrapper = {
    schema: "papersoccer.human-match.v1",
    replay,
    botConfiguration: {
      kind: "MctsBot",
      seed: "18446744073709551615",
      iterations: 10000,
    },
    botSearches: [],
  };

  assert.equal(support.unwrapReplayDocument(wrapper), replay);
  assert.equal(support.unwrapReplayDocument(replay), replay);
  assert.throws(
    () => support.unwrapReplayDocument({ schema: wrapper.schema }),
    /does not contain a replay/,
  );
});

test("human-match filenames describe the bot, lossless seed, and budget", () => {
  assert.equal(
    support.humanMatchFilename({
      kind: "MctsBot",
      seed: "18446744073709551615",
      iterations: 10000,
    }),
    "papersoccer-human-match-MctsBot-seed-18446744073709551615-" +
      "10000-new-simulations-per-move.json",
  );
  assert.equal(
    support.humanMatchFilename({ kind: "RandomBot", seed: "17" }),
    "papersoccer-human-match-RandomBot-seed-17-no-search.json",
  );
  assert.equal(
    support.humanMatchFilename({
      kind: "AlphaBetaBot",
      seed: "23",
      depth: 6,
    }),
    "papersoccer-human-match-AlphaBetaBot-seed-23-" +
    "handoff-depth-6.json",
  );
});

test("successful games lock every configuration control", () => {
  const elements = controls();
  const game = support.createLiveGameLifecycle({
    setTimeout() { return 1; },
    clearTimeout() {},
  });
  game.start();
  const state = support.liveGameControlState({
    hasGame: true,
    movesEnabled: game.movesEnabled,
    isTerminal: false,
    usesMcts: true,
    usesAlphaBeta: false,
  });

  support.syncConfigurationControls(elements, state);

  assert.equal(state.locked, true);
  assert.equal(elements.botSelect.disabled, true);
  assert.equal(elements.sideSelect.disabled, true);
  assert.equal(elements.seedInput.disabled, true);
  assert.equal(elements.budgetInput.disabled, true);
  assert.equal(elements.depthInput.disabled, true);
  assert.equal(elements.startButton.hidden, true);
  assert.equal(elements.changeSettingsButton.hidden, false);
  assert.equal(elements.exportButton.hidden, false);
});

test("changing settings unlocks controls while retaining export access", () => {
  const elements = controls();
  const game = support.createLiveGameLifecycle({
    setTimeout() { return 1; },
    clearTimeout() {},
  });
  game.start();
  game.changeSettings();
  const state = support.liveGameControlState({
    hasGame: true,
    movesEnabled: game.movesEnabled,
    isTerminal: false,
    usesMcts: true,
    usesAlphaBeta: false,
  });

  support.syncConfigurationControls(elements, state);

  assert.equal(state.locked, false);
  assert.equal(elements.botSelect.disabled, false);
  assert.equal(elements.sideSelect.disabled, false);
  assert.equal(elements.seedInput.disabled, false);
  assert.equal(elements.budgetInput.disabled, false);
  assert.equal(elements.depthInput.disabled, true);
  assert.equal(elements.startButton.hidden, false);
  assert.equal(elements.startButton.textContent, "Start new game");
  assert.equal(elements.changeSettingsButton.hidden, true);
  assert.equal(elements.exportButton.hidden, false);
});

test("AlphaBeta settings enable only the possession-handoff depth control", () => {
  const elements = controls();
  const state = support.liveGameControlState({
    hasGame: false,
    movesEnabled: false,
    isTerminal: false,
    usesMcts: false,
    usesAlphaBeta: true,
  });

  support.syncConfigurationControls(elements, state);

  assert.equal(elements.budgetInput.disabled, true);
  assert.equal(elements.depthInput.disabled, false);
});

test("terminal games unlock controls and offer a new game", () => {
  const elements = controls();
  const game = support.createLiveGameLifecycle({
    setTimeout() { return 1; },
    clearTimeout() {},
  });
  game.start();
  game.finish();
  const state = support.liveGameControlState({
    hasGame: true,
    movesEnabled: game.movesEnabled,
    isTerminal: true,
    usesMcts: true,
    usesAlphaBeta: false,
  });

  support.syncConfigurationControls(elements, state);

  assert.equal(state.locked, false);
  assert.equal(elements.botSelect.disabled, false);
  assert.equal(elements.sideSelect.disabled, false);
  assert.equal(elements.seedInput.disabled, false);
  assert.equal(elements.budgetInput.disabled, false);
  assert.equal(elements.depthInput.disabled, true);
  assert.equal(elements.startButton.textContent, "Start new game");
  assert.equal(elements.changeSettingsButton.hidden, true);
  assert.equal(elements.exportButton.hidden, false);
});

test("cancelled bot work cannot run through a stale timer callback", () => {
  let nextHandle = 1;
  const callbacks = new Map();
  const timerHost = {
    setTimeout(callback) {
      const handle = nextHandle;
      nextHandle += 1;
      callbacks.set(handle, callback);
      return handle;
    },
    clearTimeout() {},
  };
  const game = support.createLiveGameLifecycle(timerHost);
  let botMoves = 0;

  game.start();
  assert.equal(game.scheduleBot(function () {
    botMoves += 1;
  }, 520), true);
  const staleCallback = callbacks.get(1);
  assert.equal(game.botThinking, true);

  game.changeSettings();
  staleCallback();

  assert.equal(game.movesEnabled, false);
  assert.equal(game.botThinking, false);
  assert.equal(botMoves, 0);
  assert.equal(game.scheduleBot(function () {
    botMoves += 1;
  }, 520), false);

  game.start();
  game.scheduleBot(function () {
    botMoves += 1;
  }, 520);
  callbacks.get(2)();
  assert.equal(botMoves, 1);
});
