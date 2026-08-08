import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const arena = process.env.PAPERSOCCER_ARENA;
assert.ok(arena, "PAPERSOCCER_ARENA must name the arena executable");
const openingBank = process.env.PAPERSOCCER_OPENING_BANK;
assert.ok(openingBank,
  "PAPERSOCCER_OPENING_BANK must name the opening-bank executable");
const sanitizersEnabled = process.env.PAPERSOCCER_SANITIZERS_ENABLED;
assert.ok(["0", "1"].includes(sanitizersEnabled),
  "PAPERSOCCER_SANITIZERS_ENABLED must identify the build configuration");
const JACEK_MODEL_SHA256 =
  "57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084";
const RANK5_FRESH_COUNTERS = [
  "leaf_evaluations",
  "terminal_nodes",
  "completed_actions",
  "cutoffs",
  "transposition_probes",
  "transposition_hits",
  "transposition_cutoffs",
  "transposition_stores",
  "continuation_transposition_hits",
  "evaluation_cache_probes",
  "evaluation_cache_hits",
  "terminal_bound_cutoffs",
  "forced_edges",
  "root_seed_actions",
  "root_transposition_reuses",
  "max_action_edges",
];

function runJson(arguments_) {
  return JSON.parse(execFileSync(arena, arguments_, { encoding: "utf8" }));
}

test("arena exposes compile-time build provenance without running bots", () => {
  const report = runJson(["provenance"]);
  assert.deepEqual(Object.keys(report).sort(), [
    "schema",
    "runtime",
    "build_type",
    "ndebug",
    "sanitizers_enabled",
    "compiler_id",
    "compiler_version",
    "configured_flags",
    "cxx_standard",
    "source_commit",
    "source_dirty",
  ].sort());
  assert.equal(report.schema, "papersoccer.arena-build.v1");
  assert.equal(report.runtime, "native");
  assert.equal(typeof report.build_type, "string");
  assert.equal(typeof report.ndebug, "boolean");
  assert.equal(report.sanitizers_enabled, sanitizersEnabled === "1");
  assert.equal(typeof report.compiler_id, "string");
  assert.equal(typeof report.compiler_version, "string");
  assert.equal(typeof report.configured_flags, "string");
  assert.ok(report.cxx_standard >= 202002);
  assert.match(report.source_commit, /^(?:[0-9a-f]{40}|unknown)$/);
  assert.equal(typeof report.source_dirty, "boolean");
});

function parseOpeningBank(contents) {
  const lines = contents.trimEnd().split("\n");
  const header = lines.findIndex((line) => line.startsWith("opening_id\t"));
  assert.ok(header >= 0, "generated bank must include its record header");
  return lines.slice(header + 1).map((line) => {
    const [opening_id, phase, depth, generation_seed, state_hash,
      canonical_key, to_move, transcript] = line.split("\t");
    return {
      opening_id,
      phase,
      depth: Number(depth),
      generation_seed,
      state_hash,
      canonical_key,
      to_move,
      moves: transcript.split(";").map((endpoint) => {
        const [x, y] = endpoint.split(",").map(Number);
        return { x, y };
      }),
    };
  });
}

function generateOpeningBank(t, { phase = "development", depth = 4,
  pairs = 1, seed = "1234" } = {}) {
  const directory = mkdtempSync(join(tmpdir(), "papersoccer-arena-bank-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const contents = execFileSync(openingBank, [
    "generate",
    "--phase", phase,
    "--depth", String(depth),
    "--pairs", String(pairs),
    "--seed", String(seed),
  ], { encoding: "utf8" });
  const path = join(directory, `${phase}-${depth}.tsv`);
  writeFileSync(path, contents);
  return { contents, path, records: parseOpeningBank(contents), directory };
}

test("matches mode emits parseable paired JSON", () => {
  const report = runJson([
    "matches",
    "--pairs", "1",
    "--max-plies", "1",
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
  assert.equal(report.summary.truncations, 2);
  assert.deepEqual(report.summary.pair_scores, [null]);
  assert.equal(report.summary.candidate.overall.scored_games, 0);
  assert.equal(report.summary.candidate.overall.score_points, 0);
  assert.equal(report.summary.reference.overall.scored_games, 0);
  assert.equal(report.summary.reference.overall.score_points, 0);
  assert.equal(report.summary.pair_bootstrap_95.valid_pairs, 0);
  assert.equal(report.summary.pair_bootstrap_95.invalid_pairs, 1);
  assert.equal(report.summary.pair_bootstrap_95.lower_percent, null);
  assert.equal(report.summary.pair_bootstrap_95.upper_percent, null);
  assert.deepEqual(report.configuration.warmup, {
    decisions_per_entrant: 0,
    timed: false,
    generation_plies: 24,
    position_generator: "uniform_legal_move_generator",
    seed_derivation: "domain_separated_splitmix64",
    bot_instances: "separate_from_measured_games",
  });
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
  assert.equal(typeof report.summary.candidate.timing.p90_ns, "number");
  assert.equal(typeof report.summary.candidate.timing.p99_ns, "number");

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

test("completed pairs remain scored and have zero truncations", () => {
  const report = runJson([
    "matches",
    "--pairs", "1",
    "--max-plies", "512",
    "--bootstrap-samples", "32",
    "--candidate-kind", "random",
    "--reference-kind", "random",
  ]);

  assert.equal(report.summary.truncations, 0);
  assert.ok(report.games.every(({ outcome }) => outcome.truncated === false));
  assert.equal(report.summary.candidate.overall.scored_games, 2);
  assert.equal(report.summary.reference.overall.scored_games, 2);
  assert.equal(report.summary.pair_bootstrap_95.valid_pairs, 1);
  assert.equal(report.summary.pair_bootstrap_95.invalid_pairs, 0);
  assert.equal(typeof report.summary.pair_scores[0], "number");
  assert.ok([0, 0.5, 1].includes(report.summary.pair_scores[0]));
  assert.equal(typeof report.summary.pair_bootstrap_95.lower_percent, "number");
  assert.equal(typeof report.summary.pair_bootstrap_95.upper_percent, "number");
});

test("match warm-up is untimed and leaves measured games unchanged", () => {
  const common = [
    "matches",
    "--seed", "271828",
    "--pairs", "1",
    "--max-plies", "1",
    "--bootstrap-samples", "32",
    "--candidate-kind", "random",
    "--reference-kind", "random",
  ];
  const cold = runJson(common);
  const warm = runJson([...common, "--warmup-decisions", "2"]);

  assert.deepEqual(warm.configuration.warmup, {
    decisions_per_entrant: 2,
    timed: false,
    generation_plies: 24,
    position_generator: "uniform_legal_move_generator",
    seed_derivation: "domain_separated_splitmix64",
    bot_instances: "separate_from_measured_games",
  });
  const measuredGames = ({ games }) => games.map((game) => ({
    pair_index: game.pair_index,
    game_in_pair: game.game_in_pair,
    player_one: game.player_one,
    player_two: game.player_two,
    outcome: game.outcome,
    decisions: game.decisions.map((decision) => ({
      ply: decision.ply,
      bot: decision.bot,
      player: decision.player,
      from: decision.from,
      to: decision.to,
      legal: decision.legal,
    })),
  }));
  assert.deepEqual(measuredGames(warm), measuredGames(cold));
  assert.deepEqual(warm.summary.pair_scores, cold.summary.pair_scores);
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

test("matches replay frozen opening-bank transcripts exactly in record order", (t) => {
  const bank = generateOpeningBank(t, {
    phase: "validation",
    depth: 4,
    pairs: 2,
    seed: "987654321",
  });
  const arguments_ = [
    "matches",
    "--seed", "314159",
    "--pairs", "2",
    "--opening-bank", bank.path,
    "--max-plies", "5",
    "--bootstrap-samples", "32",
    "--candidate-kind", "random",
    "--reference-kind", "random",
  ];
  const report = runJson(arguments_);

  assert.equal(report.configuration.opening_plies, 4);
  assert.equal(report.configuration.opening_generator,
    "frozen_uniform_legal_move_data_generation_bank");
  assert.equal(report.configuration.opening_seed_derivation,
    "committed_bank_accepted_generation_seeds");
  assert.equal(report.openings.length, bank.records.length);
  for (const [index, expected] of bank.records.entries()) {
    const actual = report.openings[index];
    assert.equal(actual.pair_index, index);
    assert.equal(actual.opening_id, expected.opening_id);
    assert.equal(actual.phase, expected.phase);
    assert.equal(actual.generation_seed, expected.generation_seed);
    assert.equal(actual.state_hash, expected.state_hash);
    assert.equal(actual.canonical_key, expected.canonical_key);
    assert.equal(actual.requested_plies, expected.depth);
    assert.equal(actual.actual_plies, expected.depth);
    assert.deepEqual(actual.moves, expected.moves);
    assert.equal(actual.state.to_move, expected.to_move);
    assert.deepEqual(actual.state.ball, expected.moves.at(-1));

    const pairedGames = report.games.filter(({ pair_index }) =>
      pair_index === index);
    assert.equal(pairedGames.length, 2);
    assert.equal(pairedGames[0].player_one.bot, "candidate");
    assert.equal(pairedGames[1].player_one.bot, "reference");
    for (const game of pairedGames) {
      assert.equal(game.decisions[0].ply, expected.depth + 1);
      assert.deepEqual(game.decisions[0].from, actual.state.ball);
      assert.equal(game.decisions[0].player, actual.state.to_move);
    }
  }

  const repeated = runJson(arguments_);
  assert.deepEqual(repeated.openings, report.openings);
  const measuredTrace = ({ games }) => games.map((game) => ({
    pair_index: game.pair_index,
    game_in_pair: game.game_in_pair,
    player_one: game.player_one,
    player_two: game.player_two,
    outcome: game.outcome,
    moves: game.decisions.map(({ ply, bot, player, from, to, legal }) =>
      ({ ply, bot, player, from, to, legal })),
  }));
  assert.deepEqual(measuredTrace(repeated), measuredTrace(report));
});

test("frozen opening banks reject mismatches, tampering, and generated options", (t) => {
  const bank = generateOpeningBank(t, { pairs: 1 });
  const common = [
    "matches",
    "--opening-bank", bank.path,
    "--candidate-kind", "random",
    "--reference-kind", "random",
  ];

  const pairMismatch = spawnSync(arena, [...common, "--pairs", "2"], {
    encoding: "utf8",
  });
  assert.notEqual(pairMismatch.status, 0);
  assert.match(pairMismatch.stderr,
    /frozen opening count must equal the seed pair count/);

  const generatedOpening = spawnSync(arena, [
    ...common,
    "--pairs", "1",
    "--opening-plies", "4",
  ], { encoding: "utf8" });
  assert.notEqual(generatedOpening.status, 0);
  assert.match(generatedOpening.stderr,
    /--opening-bank is incompatible with --opening-plies/);

  const ruleMismatch = spawnSync(arena, [
    ...common,
    "--pairs", "1",
    "--width", "6",
  ], { encoding: "utf8" });
  assert.notEqual(ruleMismatch.status, 0);
  assert.match(ruleMismatch.stderr, /uses rules that differ from the arena/);

  const lines = bank.contents.trimEnd().split("\n");
  const header = lines.findIndex((line) => line.startsWith("opening_id\t"));
  const fields = lines[header + 1].split("\t");
  const moves = fields[7].split(";");
  moves[0] = "0,0";
  fields[7] = moves.join(";");
  lines[header + 1] = fields.join("\t");
  const tamperedPath = join(bank.directory, "tampered.tsv");
  writeFileSync(tamperedPath, `${lines.join("\n")}\n`);
  const tampered = spawnSync(arena, [
    "matches",
    "--pairs", "1",
    "--opening-bank", tamperedPath,
  ], { encoding: "utf8" });
  assert.notEqual(tampered.status, 0);
  assert.match(tampered.stderr, /opening transcript has illegal ply/);
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

test("Rank5Derived CLI reports fixed-profile diagnostics and fresh-root timing", () => {
  const report = runJson([
    "positions",
    "--positions", "1",
    "--generation-plies", "0",
    "--candidate-kind", "rank5-derived",
    "--reference-kind", "random",
  ]);

  assert.deepEqual(report.configuration.candidate, {
    kind: "rank5-derived",
    profile: "50k-demo",
    max_turn_depth: 32,
    max_nodes: 50000,
    transposition_table_entries: 65536,
    evaluation_cache_entries: 32768,
    max_time_ms: 0,
    model_blend_percent: 0,
    replay_corrections: false,
    replay_book_enabled: false,
    original_sha256:
      "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29",
  });

  const candidate = report.positions[0].evaluations[0];
  assert.equal(candidate.mcts, null);
  assert.equal(candidate.alpha_beta, null);
  assert.equal(candidate.rank5_derived.profile_node_budget, 50000);
  assert.equal(candidate.rank5_derived.requested_nodes, 50000);
  assert.ok(candidate.rank5_derived.visited_nodes > 0);
  assert.ok(candidate.rank5_derived.visited_nodes <= 50000);
  assert.equal(typeof candidate.rank5_derived.completed_turn_depth, "number");
  assert.equal(typeof candidate.rank5_derived.attempted_turn_depth, "number");
  assert.equal(typeof candidate.rank5_derived.root_score, "number");
  assert.equal(typeof candidate.rank5_derived.budget_exhausted, "boolean");
  assert.ok(candidate.rank5_derived.planned_action_length > 0);
  assert.equal(candidate.rank5_derived.current_edge_index, 0);
  assert.equal(candidate.rank5_derived.cached_continuation, false);
  assert.equal(typeof candidate.rank5_derived.cached_moves_remaining, "number");
  assert.equal(candidate.rank5_derived.search_ordinal_in_game, 1);

  const summary = report.summary.candidate.rank5_derived;
  assert.equal("searches_sum" in summary, false);
  assert.equal("searches_max" in summary, false);
  assert.equal(summary.decisions, 1);
  assert.equal(summary.fresh_root_searches, 1);
  assert.equal(summary.cached_continuation_edges, 0);
  assert.equal(summary.requested_nodes_per_fresh_search, 50000);
  assert.equal(summary.visited_nodes_sum,
    candidate.rank5_derived.visited_nodes);
  for (const counter of RANK5_FRESH_COUNTERS) {
    assert.equal(summary[`${counter}_sum`], candidate.rank5_derived[counter]);
    assert.equal(summary[`${counter}_max`], candidate.rank5_derived[counter]);
  }
  assert.equal(summary.fresh_root_timing.decisions, 1);
  assert.equal(summary.all_edge_timing.decisions, 1);
  assert.equal(typeof summary.fresh_root_timing.p90_ns, "number");
  assert.equal(typeof summary.fresh_root_timing.p95_ns, "number");
  assert.equal(typeof summary.fresh_root_timing.p99_ns, "number");
  assert.ok(summary.fresh_root_timing.p90_ns <=
    summary.fresh_root_timing.p95_ns);
  assert.ok(summary.fresh_root_timing.p95_ns <=
    summary.fresh_root_timing.p99_ns);
  assert.deepEqual(report.configuration.reference, { kind: "random" });
  assert.equal(report.summary.reference.rank5_derived.decisions, 0);
});

test("Rank5Derived summaries separate cached edges from fresh-root searches", () => {
  const report = runJson([
    "matches",
    "--pairs", "1",
    "--max-plies", "24",
    "--bootstrap-samples", "32",
    "--candidate-kind", "rank5-derived",
    "--reference-kind", "random",
  ]);
  const decisions = report.games.flatMap(({ decisions: gameDecisions }) =>
    gameDecisions.filter(({ bot }) => bot === "candidate"));
  const freshRoots = decisions.filter(({ rank5_derived: diagnostics }) =>
    diagnostics.cached_continuation === false);
  const cachedEdges = decisions.filter(({ rank5_derived: diagnostics }) =>
    diagnostics.cached_continuation === true);

  const perGameFreshRoots = report.games.map(({ decisions: gameDecisions }) =>
    gameDecisions.filter(({ bot, rank5_derived: diagnostics }) =>
      bot === "candidate" && diagnostics.cached_continuation === false));
  assert.ok(perGameFreshRoots.some((roots) => roots.length >= 2));
  for (const roots of perGameFreshRoots) {
    assert.deepEqual(
      roots.map(({ rank5_derived: diagnostics }) =>
        diagnostics.search_ordinal_in_game),
      roots.map((_, index) => index + 1),
    );
  }

  assert.ok(freshRoots.length > 0);
  assert.ok(cachedEdges.length > 0);
  for (const { rank5_derived: diagnostics } of freshRoots) {
    assert.equal(diagnostics.requested_nodes, 50000);
    assert.ok(diagnostics.visited_nodes > 0);
    assert.equal(diagnostics.current_edge_index, 0);
  }
  for (const { rank5_derived: diagnostics } of cachedEdges) {
    assert.equal(diagnostics.requested_nodes, 0);
    assert.equal(diagnostics.visited_nodes, 0);
    assert.ok(diagnostics.current_edge_index > 0);
  }

  const summary = report.summary.candidate.rank5_derived;
  assert.equal("searches_sum" in summary, false);
  assert.equal("searches_max" in summary, false);
  assert.equal(summary.decisions, decisions.length);
  assert.equal(summary.fresh_root_searches, freshRoots.length);
  assert.equal(summary.cached_continuation_edges, cachedEdges.length);
  assert.equal(summary.decisions,
    summary.fresh_root_searches + summary.cached_continuation_edges);
  assert.equal(summary.requested_nodes_sum, freshRoots.length * 50000);
  assert.equal(summary.visited_nodes_sum, freshRoots.reduce(
    (sum, { rank5_derived: diagnostics }) =>
      sum + diagnostics.visited_nodes,
    0,
  ));
  for (const counter of RANK5_FRESH_COUNTERS) {
    const values = freshRoots.map(({ rank5_derived: diagnostics }) =>
      diagnostics[counter]);
    assert.equal(summary[`${counter}_sum`],
      values.reduce((sum, value) => sum + value, 0));
    assert.equal(summary[`${counter}_max`], Math.max(...values));
  }
  for (const histogram of [
    summary.completed_turn_depth_histogram,
    summary.attempted_turn_depth_histogram,
    summary.planned_action_length_histogram,
  ]) {
    assert.equal(Object.values(histogram).reduce(
      (sum, count) => sum + count, 0), freshRoots.length);
  }
  assert.equal(summary.fresh_root_timing.decisions, freshRoots.length);
  assert.equal(summary.all_edge_timing.decisions, decisions.length);
});

test("DeepTurnSearch has a distinct fixed-profile identity and diagnostics", () => {
  const report = runJson([
    "positions",
    "--positions", "1",
    "--generation-plies", "0",
    "--candidate-kind", "deep-turn-search",
    "--candidate-complete-turn-max-nodes", "100000",
    "--reference-kind", "random",
  ]);

  assert.deepEqual(report.configuration.candidate, {
    kind: "deep-turn-search",
    profile: "deep-100k",
    max_turn_depth: 32,
    max_nodes: 100000,
    transposition_table_entries: 65536,
    evaluation_cache_entries: 32768,
    max_time_ms: 0,
    model_blend_percent: 0,
    replay_corrections: false,
    ranked_source_sha256:
      "f29959c4b6db6225de4e3913ee1eb020c7adf4e5363cabff545bfa275d0dce29",
  });
  const candidate = report.positions[0].evaluations[0];
  assert.equal(candidate.rank5_derived, null);
  assert.equal(candidate.deep_turn_search.profile_node_budget, 100000);
  assert.equal(candidate.deep_turn_search.requested_nodes, 100000);
  assert.ok(candidate.deep_turn_search.visited_nodes > 0);
  assert.ok(candidate.deep_turn_search.visited_nodes <= 100000);
  assert.equal(candidate.deep_turn_search.cached_continuation, false);
  assert.equal(report.summary.candidate.rank5_derived.decisions, 0);
  assert.equal(report.summary.candidate.deep_turn_search.decisions, 1);
  assert.equal(
    report.summary.candidate.deep_turn_search.requested_nodes_per_fresh_search,
    100000,
  );

  const invalid = spawnSync(arena, [
    "positions",
    "--candidate-kind", "deep-turn-search",
    "--candidate-complete-turn-max-nodes", "50000",
  ], { encoding: "utf8" });
  assert.notEqual(invalid.status, 0);
  assert.match(invalid.stderr, /100k, 200k, or 400k node profile/u);
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

  const openingBankInput = spawnSync(arena, [
    "positions", "--opening-bank", "/tmp/not-read-in-positions.tsv",
  ], { encoding: "utf8" });
  assert.notEqual(openingBankInput.status, 0);
  assert.match(openingBankInput.stderr,
    /--opening-bank is only valid in matches mode/);

  const warmup = spawnSync(arena, [
    "positions", "--warmup-decisions", "1",
  ], { encoding: "utf8" });
  assert.notEqual(warmup.status, 0);
  assert.match(warmup.stderr,
    /--warmup-decisions is only valid in matches mode/);
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
    /requires random, mcts, alpha-beta, jacek-inspired, rank5-derived, or deep-turn-search/);
});
