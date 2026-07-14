import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import test from "node:test";

const arena = process.env.PAPERSOCCER_ARENA;
assert.ok(arena, "PAPERSOCCER_ARENA must name the arena executable");

function runJson(arguments_) {
  return JSON.parse(execFileSync(arena, arguments_, { encoding: "utf8" }));
}

test("matches mode emits parseable paired JSON", () => {
  const report = runJson([
    "matches",
    "--pairs", "1",
    "--max-plies", "8",
    "--bootstrap-samples", "32",
    "--candidate-iterations", "2",
    "--reference-iterations", "2",
  ]);

  assert.equal(report.schema, "papersoccer.arena.v1");
  assert.equal(report.mode, "matches");
  assert.equal(report.games.length, 2);
  assert.equal(report.games[0].player_one.bot, "candidate");
  assert.equal(report.games[1].player_one.bot, "reference");
  assert.equal(report.summary.illegal_moves, 0);
  assert.equal(report.summary.pair_scores.length, 1);
});

test("positions mode evaluates both bots on parseable shared positions", () => {
  const report = runJson([
    "positions",
    "--positions", "2",
    "--generation-plies", "0",
    "--candidate-iterations", "2",
    "--reference-iterations", "2",
  ]);

  assert.equal(report.schema, "papersoccer.arena.v1");
  assert.equal(report.mode, "positions");
  assert.equal(report.positions.length, 2);
  assert.deepEqual(
    report.positions[0].evaluations.map(({ bot }) => bot),
    ["candidate", "reference"],
  );
  assert.equal(report.summary.illegal_moves, 0);
});

test("mode-specific options are rejected outside their mode", () => {
  const positions = spawnSync(arena, ["positions", "--pairs", "1"], {
    encoding: "utf8",
  });
  assert.notEqual(positions.status, 0);
  assert.match(positions.stderr, /--pairs is only valid in matches mode/);

  const matches = spawnSync(arena, ["matches", "--positions", "1"], {
    encoding: "utf8",
  });
  assert.notEqual(matches.status, 0);
  assert.match(matches.stderr, /--positions is only valid in positions mode/);
});
