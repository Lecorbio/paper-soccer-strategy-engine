import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const calls = [];
globalThis.createPaperSoccerModule = async function () {
  return {
    ccall(name, returnType, argumentTypes, argumentsList) {
      calls.push({ name, returnType, argumentTypes, argumentsList });
      if (name === "ps_start_game" || name === "ps_start_bot_replay") {
        return 1;
      }
      if (name === "ps_snapshot_json") {
        return JSON.stringify({ schema: "papersoccer.web-session.v1" });
      }
      if (name === "ps_bot_replay_snapshot_json") {
        return JSON.stringify({ schema: "papersoccer.bot-replay-session.v1" });
      }
      throw new Error("Unexpected mock C ABI call: " + name);
    },
  };
};

await import("../../web/game-engine.js");

const { BotKind, Player, ready } = globalThis.PaperSoccer;
const gameEngine = await ready;

function lastCall(name) {
  return calls.findLast((call) => call.name === name);
}

test("Rank5Derived maps to C ABI kind 4 with fixed deterministic work", () => {
  assert.equal(BotKind.Rank5Derived, "rank5Derived");

  gameEngine.startGame(
    "18446744073709551615",
    Player.Two,
    BotKind.Rank5Derived,
    1,
  );
  assert.deepEqual(
    lastCall("ps_start_game").argumentsList,
    ["0", 2, 4, 50000],
  );

  gameEngine.startBotReplay(
    { kind: BotKind.Rank5Derived },
    { kind: BotKind.Rank5Derived, seed: "999", iterations: 7 },
    0,
  );
  assert.deepEqual(
    lastCall("ps_start_bot_replay").argumentsList,
    ["0", 4, 50000, "0", 4, 50000, 0],
  );
});

test("all bot selectors offer the adapted profile without changing defaults", async () => {
  const html = await readFile(new URL("../../web/index.html", import.meta.url), "utf8");
  const rank5Options = html.match(
    /<option value="rank5Derived">Rank5DerivedBot — 50k demo profile<\/option>/g,
  );
  assert.equal(rank5Options?.length, 3);

  const defaultBotOptions = html.match(/<option value="random" selected>/g);
  assert.equal(defaultBotOptions?.length, 3);
  assert.doesNotMatch(html, /<option value="rank5Derived" selected>/);

  const profileCopy =
    "Complete-turn search derived from the rank 5/206 CodinGame submission, " +
    "adapted to demo rules and deterministic 50,000-node work.";
  assert.equal(html.split(profileCopy).length - 1, 3);
  assert.match(html, /id="botSeedField"/);
  assert.match(html, /id="replayOneSeedField"/);
  assert.match(html, /id="replayTwoSeedField"/);
});

test("the UI hides configurable fields and renders Rank5 diagnostics", async () => {
  const source = await readFile(new URL("../../web/app.js", import.meta.url), "utf8");

  assert.match(source, /botSeedField\.hidden = playUsesRank5Profile/);
  assert.match(source, /replayOneSeedField\.hidden = playerOneUsesRank5Profile/);
  assert.match(source, /replayTwoSeedField\.hidden = playerTwoUsesRank5Profile/);
  assert.match(source, /botRank5Profile\.hidden = !playUsesRank5Profile/);
  assert.match(source, /search\.searchType === "rank5Derived"/);
  assert.match(source, /search\.currentEdgeIndex \+ 1/);
});
