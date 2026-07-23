import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import test from "node:test";

const arena = process.env.PAPERSOCCER_ARENA;
assert.ok(arena, "PAPERSOCCER_ARENA must name the arena executable");
const JACEK_MODEL_SHA256 =
  "57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084";

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

test("matches mode shares deterministic randomized openings across colors", () => {
  const common = [
    "matches",
    "--seed", "123456789",
    "--pairs", "1",
    "--max-plies", "8",
    "--bootstrap-samples", "32",
    "--candidate-kind", "random",
    "--reference-kind", "random",
  ];
  const report = runJson([...common, "--opening-plies", "4"]);

  assert.equal(report.configuration.opening_plies, 4);
  assert.equal(report.configuration.opening_generator, "uniform_random");
  assert.equal(report.configuration.opening_seed_derivation,
    "domain_separated_splitmix64");
  assert.equal(report.openings.length, 1);

  const opening = report.openings[0];
  assert.equal(opening.pair_index, 0);
  assert.equal(opening.generation_seed, "3010655281933515797");
  assert.equal(opening.requested_plies, 4);
  assert.equal(opening.actual_plies, 4);
  assert.equal(opening.moves.length, 4);
  assert.deepEqual(opening.moves, [
    { x: 5, y: 6 },
    { x: 6, y: 5 },
    { x: 7, y: 4 },
    { x: 7, y: 5 },
  ]);
  assert.deepEqual(opening.state.ball, opening.moves.at(-1));
  assert.match(opening.state.to_move, /^(one|two)$/);

  assert.equal(report.games[0].pair_index, opening.pair_index);
  assert.equal(report.games[1].pair_index, opening.pair_index);
  assert.equal(report.games[0].player_one.bot, "candidate");
  assert.equal(report.games[1].player_one.bot, "reference");
  for (const game of report.games) {
    assert.equal(game.decisions[0].ply, opening.actual_plies + 1);
    assert.deepEqual(game.decisions[0].from, opening.state.ball);
    assert.equal(game.decisions[0].player, opening.state.to_move);
    assert.equal(game.outcome.plies,
      opening.actual_plies + game.decisions.length);
    assert.ok(game.outcome.plies <= report.configuration.max_plies);
  }

  const repeated = runJson([...common, "--opening-plies", "4"]);
  assert.deepEqual(repeated.openings, report.openings);

  const noOpening = runJson([...common, "--opening-plies", "0"]);
  const participantSeeds = ({ games }) => games.map((game) => [
    game.player_one.seed,
    game.player_two.seed,
  ]);
  assert.deepEqual(participantSeeds(report), participantSeeds(noOpening));
  assert.deepEqual(participantSeeds(report), [
    ["2466975172287755897", "8832083440362974766"],
    ["8832083440362974766", "2466975172287755897"],
  ]);
  assert.deepEqual(noOpening.openings, []);
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
  const agreements = report.positions.filter(({ evaluations }) =>
    evaluations[0].move.x === evaluations[1].move.x &&
    evaluations[0].move.y === evaluations[1].move.y).length;
  assert.deepEqual(report.summary.move_comparison, {
    positions: 2,
    agreements,
    changes: 2 - agreements,
  });
});

test("alpha-beta arena settings reach the bot and its diagnostics", () => {
  const report = runJson([
    "positions",
    "--positions", "1",
    "--generation-plies", "0",
    "--candidate-kind", "alpha-beta",
    "--candidate-alpha-beta-depth", "1",
    "--candidate-alpha-beta-max-nodes", "128",
    "--candidate-alpha-beta-table-entries", "0",
    "--candidate-alpha-beta-max-search-plies", "7",
    "--reference-kind", "random",
  ]);

  assert.deepEqual(report.configuration.candidate, {
    kind: "alpha-beta",
    max_turn_depth: 1,
    max_nodes: 128,
    transposition_table_entries: 0,
    max_search_plies: 7,
  });
  assert.deepEqual(report.configuration.reference, { kind: "random" });

  const candidate = report.positions[0].evaluations[0];
  assert.equal(candidate.bot, "candidate");
  assert.equal(candidate.mcts, null);
  assert.equal(typeof candidate.alpha_beta.nodes, "number");
  assert.equal(typeof candidate.alpha_beta.physical_ply_cutoffs, "number");
  assert.ok(candidate.alpha_beta.nodes <= 128);
  assert.ok(Array.isArray(candidate.alpha_beta.principal_variation));
  assert.ok(Array.isArray(candidate.alpha_beta.root_moves));

  assert.equal(report.summary.candidate.alpha_beta.searches, 1);
  assert.equal(report.summary.candidate.alpha_beta.nodes_sum,
    candidate.alpha_beta.nodes);
  assert.equal(report.summary.candidate.alpha_beta.physical_ply_cutoffs_sum,
    candidate.alpha_beta.physical_ply_cutoffs);
  const completedHistogram =
    report.summary.candidate.alpha_beta.completed_turn_depth_histogram;
  const attemptedHistogram =
    report.summary.candidate.alpha_beta.attempted_turn_depth_histogram;
  assert.equal(Object.values(completedHistogram)
    .reduce((sum, count) => sum + count, 0), 1);
  assert.equal(Object.values(attemptedHistogram)
    .reduce((sum, count) => sum + count, 0), 1);
  assert.equal(
    report.summary.candidate.alpha_beta.completed_turn_depth_zero_searches,
    completedHistogram["0"] ?? 0,
  );
  assert.equal(completedHistogram[String(candidate.alpha_beta.completed_turn_depth)], 1);
  assert.equal(attemptedHistogram[String(candidate.alpha_beta.attempted_turn_depth)], 1);
  assert.equal(report.summary.candidate.mcts.searches, 0);
  assert.equal(report.summary.reference.alpha_beta.searches, 0);
  assert.equal(typeof report.summary.candidate.timing.median_nodes_per_second,
    "number");

  const interrupted = runJson([
    "positions",
    "--positions", "1",
    "--generation-plies", "0",
    "--candidate-kind", "alpha-beta",
    "--candidate-alpha-beta-depth", "6",
    "--candidate-alpha-beta-max-nodes", "1",
    "--candidate-alpha-beta-table-entries", "0",
    "--reference-kind", "random",
  ]);
  assert.equal(
    interrupted.summary.candidate.alpha_beta.completed_turn_depth_zero_searches,
    1,
  );
  assert.deepEqual(
    interrupted.summary.candidate.alpha_beta.completed_turn_depth_histogram,
    { 0: 1 },
  );
  assert.deepEqual(
    interrupted.summary.candidate.alpha_beta.attempted_turn_depth_histogram,
    { 1: 1 },
  );
});

test("Jacek-inspired arena settings reach the neural evaluator and diagnostics", () => {
  const report = runJson([
    "positions",
    "--positions", "1",
    "--generation-plies", "0",
    "--candidate-kind", "jacek-inspired",
    "--candidate-alpha-beta-depth", "1",
    "--candidate-alpha-beta-max-nodes", "256",
    "--candidate-alpha-beta-table-entries", "0",
    "--candidate-alpha-beta-max-search-plies", "7",
    "--reference-kind", "random",
  ]);

  assert.equal(report.configuration.candidate.model_sha256, JACEK_MODEL_SHA256);
  assert.deepEqual(report.configuration.candidate, {
    kind: "jacek-inspired",
    max_turn_depth: 1,
    max_nodes: 256,
    transposition_table_entries: 0,
    max_search_plies: 7,
    model_sha256: JACEK_MODEL_SHA256,
  });
  const candidate = report.positions[0].evaluations[0];
  assert.equal(candidate.mcts, null);
  assert.ok(candidate.alpha_beta.nodes > 0);
  assert.ok(candidate.alpha_beta.nodes <= 256);
  assert.equal(report.summary.candidate.alpha_beta.searches, 1);
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

  const opening = spawnSync(arena, [
    "positions", "--opening-plies", "4",
  ], { encoding: "utf8" });
  assert.notEqual(opening.status, 0);
  assert.match(opening.stderr,
    /--opening-plies is only valid in matches mode/);
});

test("opening length must leave room below the total ply limit", () => {
  const result = spawnSync(arena, [
    "matches",
    "--pairs", "1",
    "--opening-plies", "8",
    "--max-plies", "8",
  ], { encoding: "utf8" });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /opening plies must be less than max plies/);
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

test("invalid alpha-beta arena settings are rejected", () => {
  const zeroDepth = spawnSync(arena, [
    "positions",
    "--candidate-kind", "alpha-beta",
    "--candidate-alpha-beta-depth", "0",
  ], { encoding: "utf8" });
  assert.notEqual(zeroDepth.status, 0);
  assert.match(zeroDepth.stderr, /alpha-beta depth must be between/);

  const excessiveDepth = spawnSync(arena, [
    "positions",
    "--candidate-kind", "alpha-beta",
    "--candidate-alpha-beta-depth", "65",
  ], { encoding: "utf8" });
  assert.notEqual(excessiveDepth.status, 0);
  assert.match(excessiveDepth.stderr, /alpha-beta depth must be between/);

  const zeroNodes = spawnSync(arena, [
    "positions",
    "--candidate-kind", "alpha-beta",
    "--candidate-alpha-beta-max-nodes", "0",
  ], { encoding: "utf8" });
  assert.notEqual(zeroNodes.status, 0);
  assert.match(zeroNodes.stderr,
    /alpha-beta max nodes must be greater than zero/);

  const zeroSearchPlies = spawnSync(arena, [
    "positions",
    "--candidate-kind", "alpha-beta",
    "--candidate-alpha-beta-max-search-plies", "0",
  ], { encoding: "utf8" });
  assert.notEqual(zeroSearchPlies.status, 0);
  assert.match(zeroSearchPlies.stderr,
    /alpha-beta max search plies must be between/);

  const unknownKind = spawnSync(arena, [
    "positions",
    "--candidate-kind", "minimax",
  ], { encoding: "utf8" });
  assert.notEqual(unknownKind.status, 0);
  assert.match(unknownKind.stderr,
    /requires random, mcts, alpha-beta, or jacek-inspired/);
});
