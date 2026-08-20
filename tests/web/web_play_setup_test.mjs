import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const html = await readFile(
  new URL("../../web/index.html", import.meta.url),
  "utf8",
);
const appSource = await readFile(
  new URL("../../web/app.js", import.meta.url),
  "utf8",
);
await import("../../web/app-support.js");

const support = globalThis.PaperSoccerWebSupport;
const players = { One: "one", Two: "two" };

test("play setup offers explicit first- and second-move choices", () => {
  assert.match(
    html,
    /<span>Turn order<\/span>\s*<select id="sideSelect"[^>]*>/,
  );
  assert.match(
    html,
    /<option value="one" selected>Move first · Player 1 · attacks top<\/option>/,
  );
  assert.match(
    html,
    /<option value="two">Move second · Player 2 · attacks bottom<\/option>/,
  );
  assert.match(html, /Moving second lets the bot take the opening move\./);
  assert.match(
    html,
    /<button id="newGameButton"[^>]*type="submit"\s*disabled>Start<\/button>/,
  );
  assert.match(html, /<h2 id="statusLabel"[^>]*>Ready to start<\/h2>/);
});

test("a fresh page waits for the player to choose settings", () => {
  assert.match(
    appSource,
    /let replay = newGamePreviewReplay\(defaultReplay\);/,
  );
  assert.match(
    appSource,
    /renderInterface\(\);\s*window\.requestAnimationFrame\(tick\);/,
  );
  assert.doesNotMatch(
    appSource,
    /startNewGame\(\);\s*window\.requestAnimationFrame\(tick\);/,
  );
  assert.match(
    appSource,
    /newGameButton\.disabled = false;\s*renderInterface\(\);/,
  );
  assert.match(
    appSource,
    /gameSetup\.addEventListener\("submit", preventSetupSubmit\);\s*const gameEngine = await engine\.ready;/,
  );
});

test("choosing Player 2 starts the bot's opening turn", () => {
  const selectedHuman = support.humanPlayerForTurnOrder("two", players);
  const snapshot = {
    humanPlayer: selectedHuman,
    state: { toMove: players.One },
  };
  let botTurns = 0;

  const scheduled = support.scheduleOpeningBotTurn({
    snapshot,
    movesEnabled: true,
    gameError: "",
    schedule() { botTurns += 1; },
  });

  assert.equal(selectedHuman, players.Two);
  assert.equal(scheduled, true);
  assert.equal(botTurns, 1);
  assert.match(appSource, /support\.humanPlayerForTurnOrder\(/);
  assert.match(appSource, /support\.scheduleOpeningBotTurn\(/);
});

test("depth-bot diagnostics distinguish hand and neural evaluation", () => {
  assert.match(
    appSource,
    /const usesNeuralEvaluation = configuration\.kind === "JacekInspiredBot";/,
  );
  assert.match(
    appSource,
    /" · model SHA-256 " \+ configuration\.modelSha256\.slice\(0, 12\) \+ "…"/,
  );
  assert.match(
    appSource,
    /searchPending\.hidden = \(!usesMcts && !usesDepth && !usesRank5Profile &&/,
  );
  assert.match(
    appSource,
    /usesNeuralEvaluation \? " neural evaluations · " :\s*" hand evaluations · "/,
  );
});
