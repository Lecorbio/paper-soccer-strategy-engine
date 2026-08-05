import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const [html, appSource] = await Promise.all([
  readFile(new URL("../../web/index.html", import.meta.url), "utf8"),
  readFile(new URL("../../web/app.js", import.meta.url), "utf8"),
]);

test("Watch replay offers generate and open-existing choices together", () => {
  const replaySetup = html.match(
    /<form\b(?=[^>]*\bid="replaySetup")[^>]*>[\s\S]*?<\/form>/,
  )?.[0];
  assert.ok(replaySetup, "the page should contain its replay setup form");

  assert.match(replaySetup, />\s*Choose a replay\s*<\/h2>/);
  assert.match(
    replaySetup,
    /<button\b(?=[^>]*\bid="generateReplayButton")[^>]*>[\s\S]*?Generate replay\s*<\/button>/,
  );
  assert.match(replaySetup, /Open existing/);
  assert.match(
    replaySetup,
    /<input\b(?=[^>]*\bid="fileInput")(?=[^>]*\btype="file")(?=[^>]*\baccept="application\/json,\.json")[^>]*>/,
  );
  assert.equal(html.match(/\bid="fileInput"/g)?.length, 1);
});

test("opening an existing replay retains the established loading behavior", () => {
  assert.match(appSource, /replaySetup\.hidden = playingGame;/);
  assert.match(
    appSource,
    /control !== replayTwoDepthInput &&\s*control !== fileInput/,
  );
  assert.match(
    appSource,
    /fileInput\.addEventListener\("change", async function \(\) \{[\s\S]*?cancelReplayGeneration\(\);[\s\S]*?JSON\.parse\(await file\.text\(\)\)[\s\S]*?loadReplay\(raw\)[\s\S]*?fileInput\.value = "";/,
  );
});
