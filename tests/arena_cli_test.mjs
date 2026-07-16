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
    "--candidate-leaf-policy", "tactical-quiescence",
    "--candidate-quiescence-max-depth", "3",
    "--candidate-quiescence-max-nodes", "17",
    "--reference-leaf-policy", "rollout-only",
    "--reference-quiescence-max-depth", "4",
    "--reference-quiescence-max-nodes", "18",
  ]);

  assert.equal(report.schema, "papersoccer.arena.v1");
  assert.equal(report.mode, "matches");
  assert.equal(report.games.length, 2);
  assert.equal(report.games[0].player_one.bot, "candidate");
  assert.equal(report.games[1].player_one.bot, "reference");
  assert.equal(report.summary.illegal_moves, 0);
  assert.equal(report.summary.pair_scores.length, 1);
  assert.equal(report.configuration.candidate.rollout_policy, "tactical");
  assert.equal(report.configuration.candidate.leaf_policy,
    "tactical_quiescence");
  assert.equal(report.configuration.candidate.quiescence_max_depth, 3);
  assert.equal(report.configuration.candidate.quiescence_max_nodes, 17);
  assert.equal(report.configuration.reference.rollout_policy, "tactical");
  assert.equal(report.configuration.reference.leaf_policy, "rollout_only");
  assert.equal(report.configuration.reference.quiescence_max_depth, 4);
  assert.equal(report.configuration.reference.quiescence_max_nodes, 18);
  assert.equal(typeof report.games[0].decisions[0].mcts.tacticalProbes,
    "undefined");
  assert.equal(typeof report.games[0].decisions[0].mcts.tactical_probes,
    "number");
  assert.equal(typeof report.summary.candidate.mcts.tactical_solved_positions,
    "number");
  assert.equal(typeof report.summary.candidate.mcts.tactical_solution_rate,
    "number");
  assert.equal(typeof report.summary.candidate.mcts.max_tactical_depth,
    "number");

  const candidateSearches = report.games.flatMap((game) => game.decisions)
    .filter((decision) => decision.bot === "candidate")
    .map((decision) => decision.mcts);
  for (const counter of [
    "tactical_probes",
    "tactical_nodes",
    "tactical_solved_positions",
    "tactical_depth_cutoffs",
    "tactical_node_cutoffs",
  ]) {
    const expected = candidateSearches.reduce(
      (sum, search) => sum + search[counter],
      0,
    );
    assert.equal(report.summary.candidate.mcts[counter], expected);
  }
  assert.equal(report.summary.candidate.mcts.max_tactical_depth,
    Math.max(...candidateSearches.map((search) => search.max_tactical_depth)));
  assert.equal(report.summary.reference.mcts.tactical_probes, 0);
  assert.equal(report.summary.reference.mcts.tactical_solution_rate, 0);
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

test("invalid tactical arena settings are rejected", () => {
  const zeroDepth = spawnSync(arena, [
    "matches",
    "--candidate-quiescence-max-depth", "0",
  ], { encoding: "utf8" });
  assert.notEqual(zeroDepth.status, 0);
  assert.match(zeroDepth.stderr, /quiescence max depth must be between/);

  const excessiveNodes = spawnSync(arena, [
    "positions",
    "--candidate-quiescence-max-nodes", "1000001",
  ], { encoding: "utf8" });
  assert.notEqual(excessiveNodes.status, 0);
  assert.match(excessiveNodes.stderr, /quiescence max nodes must be between/);

  const incompatiblePolicy = spawnSync(arena, [
    "positions",
    "--candidate-policy", "uniform",
    "--candidate-leaf-policy", "tactical-quiescence",
  ], { encoding: "utf8" });
  assert.notEqual(incompatiblePolicy.status, 0);
  assert.match(incompatiblePolicy.stderr,
    /tactical quiescence requires the tactical rollout policy/);

  const unknownLeafPolicy = spawnSync(arena, [
    "positions",
    "--candidate-leaf-policy", "unknown",
  ], { encoding: "utf8" });
  assert.notEqual(unknownLeafPolicy.status, 0);
  assert.match(unknownLeafPolicy.stderr,
    /requires rollout-only or tactical-quiescence/);
});
