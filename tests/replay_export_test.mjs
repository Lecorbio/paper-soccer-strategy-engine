import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";

const exporter = process.env.PAPERSOCCER_REPLAY_EXPORTER;

test("replay exporter preserves the full uint64 seed range", () => {
  assert.ok(exporter, "the replay exporter path is required");
  const output = execFileSync(
    exporter,
    ["18446744073709551615", "0"],
    { encoding: "utf8" },
  );
  const replay = JSON.parse(output);

  assert.equal(replay.schema, "papersoccer.replay.v2");
  assert.equal(replay.players.one.seed, "18446744073709551615");
  assert.equal(replay.players.two.seed, "0");
  assert.equal(replay.truncated, true);
  assert.deepEqual(replay.moves, []);
});
