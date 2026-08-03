#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "papersoccer/arena.hpp"
#include "papersoccer/rules.hpp"

namespace arena = papersoccer::arena;

namespace {

void require_contains(const std::string &json, const std::string &needle,
                      const std::string &message) {
  if (json.find(needle) == std::string::npos) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_invalid_argument(Function &&function, const std::string &message) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  if (!threw) {
    throw std::runtime_error(message);
  }
}

std::string deterministic_match_summary(const std::string &json) {
  const std::size_t start = json.find("\"pair_scores\"");
  if (start == std::string::npos) {
    throw std::runtime_error("match report is missing deterministic pair data");
  }
  return json.substr(start);
}

std::string deterministic_openings(const std::string &json) {
  const std::size_t start = json.find("\"openings\"");
  const std::size_t end = json.find(",\"games\":[", start);
  if (start == std::string::npos || end == std::string::npos) {
    throw std::runtime_error("match report is missing opening data");
  }
  return json.substr(start, end - start);
}

void matches_report_has_paired_games_and_measurements() {
  arena::MatchesConfig config;
  config.seed_pairs = 1;
  config.max_plies = 8;
  config.bootstrap_samples = 32;
  config.candidate.iterations = 2;
  config.reference.iterations = 2;
  config.base_seed = std::numeric_limits<std::uint64_t>::max();

  if (config.candidate.rollout_policy !=
          papersoccer::MctsRolloutPolicy::Tactical ||
      config.candidate.leaf_policy !=
          papersoccer::MctsLeafPolicy::TacticalQuiescence ||
      !config.candidate.reuse_tree ||
      config.reference.rollout_policy !=
          papersoccer::MctsRolloutPolicy::Tactical ||
      config.reference.leaf_policy !=
          papersoccer::MctsLeafPolicy::RolloutOnly ||
      !config.reference.reuse_tree) {
    throw std::runtime_error(
        "arena defaults should compare quiescence with frozen tactical rollouts");
  }

  const std::string json = arena::run_matches_json(config);
  require_contains(json, "\"schema\":\"papersoccer.arena.v1\"",
                   "match report should identify the arena schema");
  require_contains(json, "\"mode\":\"matches\"",
                   "match report should identify its mode");
  require_contains(json, "\"games\":2",
                   "one seed pair should produce two games");
  require_contains(json, "\"opening_plies\":0",
                   "default matches should report an empty opening");
  require_contains(
      json,
      "\"warmup\":{\"decisions_per_entrant\":0,\"timed\":false,"
      "\"generation_plies\":24,\"position_generator\":"
      "\"uniform_legal_move_generator\",\"seed_derivation\":"
      "\"domain_separated_splitmix64\",\"bot_instances\":"
      "\"separate_from_measured_games\"}",
      "default matches should report that no warm-up calls were made");
  require_contains(json, "\"openings\":[]",
                   "default matches should not generate opening states");
  require_contains(json, "\"base_seed\":\"18446744073709551615\"",
                   "64-bit seeds should be represented losslessly");
  require_contains(
      json, "\"game_in_pair\":0,\"player_one\":{\"bot\":\"candidate\"",
      "the candidate should start the first paired game");
  require_contains(
      json, "\"game_in_pair\":1,\"player_one\":{\"bot\":\"reference\"",
      "the reference should start the color-swapped paired game");
  require_contains(json, "\"pair_bootstrap_95\"",
                   "match report should include the paired confidence interval");
  require_contains(json, "\"elapsed_ns\"",
                   "match report should retain decision timings");
  require_contains(json, "\"p90_ns\":",
                   "match timing should retain its p90 percentile");
  require_contains(json, "\"p99_ns\":",
                   "match timing should retain its p99 percentile");
  require_contains(json, "\"total_root_visits\"",
                   "match report should retain MCTS counters");
  require_contains(json, "\"leaf_policy\":\"tactical_quiescence\"",
                   "match report should retain the candidate leaf policy");
  require_contains(json, "\"leaf_policy\":\"rollout_only\"",
                   "match report should retain the reference leaf policy");
  require_contains(json, "\"quiescence_max_depth\":",
                   "match report should retain the tactical depth limit");
  require_contains(json, "\"quiescence_max_nodes\":",
                   "match report should retain the tactical node limit");
  require_contains(json, "\"tactical_solved_positions\":",
                   "match report should retain tactical solution counters");
  require_contains(json, "\"tactical_solution_rate\":",
                   "match report should summarize tactical solution rate");
  require_contains(json, "\"max_tactical_depth\":",
                   "match report should retain tactical search depth");

  const std::string repeated = arena::run_matches_json(config);
  if (deterministic_match_summary(json) !=
      deterministic_match_summary(repeated)) {
    throw std::runtime_error(
        "identical arena runs should reproduce pair scores and confidence bounds");
  }
}

void randomized_openings_are_shared_reproducible_and_ply_aware() {
  arena::MatchesConfig config;
  config.seed_pairs = 1;
  config.opening_plies = 4;
  config.max_plies = 8;
  config.bootstrap_samples = 32;
  config.candidate.kind = papersoccer::BotKind::Random;
  config.reference.kind = papersoccer::BotKind::Random;
  config.base_seed = 123456789;

  const std::string json = arena::run_matches_json(config);
  require_contains(json, "\"opening_plies\":4",
                   "match configuration should retain its opening length");
  require_contains(json, "\"opening_generator\":\"uniform_random\"",
                   "match configuration should identify the opening generator");
  require_contains(json, "\"opening_seed_derivation\":"
                         "\"domain_separated_splitmix64\"",
                   "match configuration should identify its independent seed stream");
  require_contains(json, "\"pair_index\":0,\"generation_seed\":\"",
                   "an opening should retain its lossless generation seed");
  require_contains(json, "\"requested_plies\":4,\"actual_plies\":4",
                   "an accepted opening should reach the exact requested ply");
  require_contains(json, "\"ply\":5",
                   "competitive decision numbering should follow the opening");

  const std::string repeated = arena::run_matches_json(config);
  if (deterministic_openings(json) != deterministic_openings(repeated)) {
    throw std::runtime_error(
        "identical arena runs should reproduce randomized openings");
  }
}

void positions_report_measures_both_bots_on_shared_positions() {
  arena::PositionsConfig config;
  config.position_count = 2;
  config.generation_plies = 0;
  config.candidate.iterations = 2;
  config.reference.iterations = 2;

  const std::string json = arena::run_positions_json(config);
  require_contains(json, "\"mode\":\"positions\"",
                   "position report should identify its mode");
  require_contains(json, "\"position_count\":2",
                   "position report should retain its sample count");
  require_contains(json, "\"bot\":\"candidate\"",
                   "position report should include the candidate result");
  require_contains(json, "\"bot\":\"reference\"",
                   "position report should include the reference result");
  require_contains(json, "\"generation_seed\":\"",
                   "position generation seeds should be represented losslessly");
  require_contains(json, "\"median_simulated_plies_per_second\"",
                   "position report should summarize rollout throughput");
  require_contains(json, "\"move_comparison\":{\"positions\":2,"
                         "\"agreements\":",
                   "position report should count candidate/reference agreements");
  require_contains(json, "\"changes\":",
                   "position report should count candidate/reference move changes");
}

void alpha_beta_is_configured_measured_and_summarized() {
  arena::PositionsConfig config;
  config.position_count = 1;
  config.generation_plies = 0;
  config.candidate.kind = papersoccer::BotKind::AlphaBeta;
  config.candidate.alpha_beta_depth = 1;
  config.candidate.alpha_beta_max_nodes = 128;
  config.candidate.alpha_beta_transposition_table_entries = 0;
  config.candidate.alpha_beta_max_search_plies = 7;
  config.reference.kind = papersoccer::BotKind::Random;

  const std::string json = arena::run_positions_json(config);
  require_contains(
      json,
      "\"candidate\":{\"kind\":\"alpha-beta\",\"max_turn_depth\":1,"
      "\"max_nodes\":128,\"transposition_table_entries\":0,"
      "\"max_search_plies\":7}",
      "position report should retain the alpha-beta search limits");
  require_contains(json, "\"mcts\":null,\"alpha_beta\":{"
                         "\"completed_turn_depth\":",
                   "alpha-beta evaluations should expose search diagnostics");
  require_contains(json, "\"principal_variation\":[",
                   "alpha-beta diagnostics should expose their principal variation");
  require_contains(json, "\"physical_ply_cutoffs\":",
                   "alpha-beta diagnostics should expose physical-ply cutoffs");
  require_contains(json, "\"physical_ply_cutoffs_sum\":",
                   "alpha-beta summaries should aggregate physical-ply cutoffs");
  require_contains(json, "\"alpha_beta\":{\"searches\":1,"
                         "\"nodes_sum\":",
                   "alpha-beta decisions should be summarized independently");
  require_contains(
      json,
      "\"completed_turn_depth_zero_searches\":0,"
      "\"completed_turn_depth_histogram\":{\"1\":1},"
      "\"attempted_turn_depth_histogram\":{\"1\":1}",
      "alpha-beta summaries should count completed and attempted depths");
  require_contains(json, "\"median_nodes_per_second\":",
                   "arena timing should report comparable node throughput");

  arena::PositionsConfig interrupted = config;
  interrupted.candidate.alpha_beta_depth = 6;
  interrupted.candidate.alpha_beta_max_nodes = 1;
  const std::string interrupted_json = arena::run_positions_json(interrupted);
  require_contains(
      interrupted_json,
      "\"completed_turn_depth_zero_searches\":1,"
      "\"completed_turn_depth_histogram\":{\"0\":1},"
      "\"attempted_turn_depth_histogram\":{\"1\":1}",
      "an interrupted first iteration should be visible as completed depth zero");
}

void jacek_inspired_is_configured_measured_and_summarized() {
  arena::PositionsConfig config;
  config.position_count = 1;
  config.generation_plies = 0;
  config.candidate.kind = papersoccer::BotKind::JacekInspired;
  config.candidate.alpha_beta_depth = 1;
  config.candidate.alpha_beta_max_nodes = 256;
  config.candidate.alpha_beta_transposition_table_entries = 0;
  config.candidate.alpha_beta_max_search_plies = 7;
  config.reference.kind = papersoccer::BotKind::Random;

  const std::string json = arena::run_positions_json(config);
  const std::string expected_config =
      "\"candidate\":{\"kind\":\"jacek-inspired\","
      "\"max_turn_depth\":1,\"max_nodes\":256,"
      "\"transposition_table_entries\":0,\"max_search_plies\":7,"
      "\"model_sha256\":\"" +
      std::string(papersoccer::JacekInspiredBot::model_sha256()) + "\"}";
  require_contains(
      json, expected_config,
      "position report should retain the Jacek-inspired search limits");
  require_contains(json, "\"mcts\":null,\"alpha_beta\":{"
                         "\"completed_turn_depth\":",
                   "Jacek-inspired evaluations should expose alpha-beta diagnostics");
  require_contains(json, "\"alpha_beta\":{\"searches\":1,\"nodes_sum\":",
                   "Jacek-inspired decisions should contribute to search summaries");
}

void rank5_derived_is_measured_with_fresh_root_diagnostics() {
  arena::PositionsConfig config;
  config.position_count = 1;
  config.generation_plies = 0;
  config.candidate.kind = papersoccer::BotKind::Rank5Derived;
  config.candidate.rank5_derived_model_blend_percent = 15;
  config.reference.kind = papersoccer::BotKind::Random;

  const std::string json = arena::run_positions_json(config);
  const std::string expected_config =
      "\"candidate\":{\"kind\":\"rank5-derived\","
      "\"profile\":\"50k-demo\",\"max_turn_depth\":32,"
      "\"max_nodes\":50000,\"transposition_table_entries\":65536,"
      "\"evaluation_cache_entries\":32768,\"max_time_ms\":0,"
      "\"model_blend_percent\":15,\"replay_corrections\":false,"
      "\"replay_book_enabled\":false,"
      "\"original_sha256\":\"" +
      std::string(papersoccer::Rank5DerivedBot::original_sha256()) + "\"}";
  require_contains(
      json, expected_config,
      "programmatic arena reports should preserve the adapted Rank5 profile");
  require_contains(
      json,
      "\"mcts\":null,\"alpha_beta\":null,\"rank5_derived\":{"
      "\"profile_node_budget\":50000,\"requested_nodes\":50000,"
      "\"visited_nodes\":",
      "Rank5Derived evaluations should expose the requested and visited nodes");
  require_contains(json, "\"completed_turn_depth\":",
                   "Rank5Derived diagnostics should expose completed depth");
  require_contains(json, "\"attempted_turn_depth\":",
                   "Rank5Derived diagnostics should expose attempted depth");
  require_contains(json, "\"planned_action_length\":",
                   "Rank5Derived diagnostics should expose complete-action length");
  require_contains(json, "\"current_edge_index\":0,"
                         "\"cached_continuation\":false,"
                         "\"cached_moves_remaining\":",
                   "Rank5Derived fresh roots should expose cache diagnostics");
  require_contains(
      json,
      "\"rank5_derived\":{\"decisions\":1,\"fresh_root_searches\":1,"
      "\"cached_continuation_edges\":0,"
      "\"requested_nodes_per_fresh_search\":50000,"
      "\"requested_nodes_sum\":50000,"
      "\"visited_nodes_sum\":",
      "Rank5Derived summaries should distinguish fresh roots from cached edges");
  require_contains(json, "\"fresh_root_timing\":{\"decisions\":1,",
                   "Rank5Derived summaries should retain fresh-root timing");
  require_contains(json, "\"all_edge_timing\":{\"decisions\":1,",
                   "Rank5Derived summaries should retain all-edge timing");

  arena::PositionsConfig invalid = config;
  invalid.candidate.rank5_derived_model_blend_percent = 101;
  require_invalid_argument(
      [&] { (void)arena::run_positions_json(invalid); },
      "programmatic arena use should reject an invalid Rank5 model blend");
}

void truncations_are_invalid_not_draws_and_completed_pairs_remain_scored() {
  arena::MatchesConfig truncated;
  truncated.seed_pairs = 1;
  truncated.max_plies = 1;
  truncated.bootstrap_samples = 32;
  truncated.warmup_decisions = 2;
  truncated.candidate.kind = papersoccer::BotKind::Random;
  truncated.reference.kind = papersoccer::BotKind::Random;

  const std::string truncated_json = arena::run_matches_json(truncated);
  require_contains(truncated_json,
                   "\"warmup\":{\"decisions_per_entrant\":2,"
                   "\"timed\":false",
                   "match reports should retain the warm-up call count");
  require_contains(truncated_json, "\"truncations\":2,\"candidate\":{"
                                   "\"overall\":{\"games\":2,\"wins\":0,"
                                   "\"losses\":0,\"truncations\":2,"
                                   "\"scored_games\":0,\"score_points\":0,"
                                   "\"score_percent\":0}",
                   "truncated games should not contribute score points");
  require_contains(truncated_json, "\"pair_scores\":[null]",
                   "a pair containing truncations should have a null score");
  require_contains(
      truncated_json,
      "\"valid_pairs\":0,\"invalid_pairs\":1,\"lower_percent\":null,"
      "\"upper_percent\":null",
      "truncated pairs should be excluded from the paired bootstrap");

  arena::MatchesConfig completed = truncated;
  completed.max_plies = 512;
  const std::string completed_json = arena::run_matches_json(completed);
  require_contains(completed_json, "\"summary\":{\"illegal_moves\":0,"
                                   "\"truncations\":0",
                   "standard random games should complete without truncation");
  require_contains(completed_json, "\"valid_pairs\":1,\"invalid_pairs\":0",
                   "a completed pair should remain eligible for bootstrap");
  if (completed_json.find("\"pair_scores\":[null]") != std::string::npos) {
    throw std::runtime_error("a completed pair should retain a numeric score");
  }
}

arena::FrozenOpening make_frozen_opening() {
  arena::FrozenOpening opening;
  opening.opening_id = "development-d4-example-state-hash";
  opening.phase = "development";
  opening.depth = 4;
  opening.generation_seed = 123456789;
  opening.state_hash = "example-state-hash";
  opening.canonical_key = "example-canonical-key";
  opening.transcript = {
      papersoccer::Move{{5, 6}}, papersoccer::Move{{6, 5}},
      papersoccer::Move{{7, 4}}, papersoccer::Move{{7, 5}}};
  opening.state = papersoccer::make_initial_state();
  for (const papersoccer::Move move : opening.transcript) {
    opening.state = papersoccer::apply_move(opening.state, move);
  }
  return opening;
}

void frozen_openings_are_replayed_and_validated_programmatically() {
  arena::MatchesConfig config;
  config.seed_pairs = 1;
  config.max_plies = 5;
  config.bootstrap_samples = 32;
  config.candidate.kind = papersoccer::BotKind::Random;
  config.reference.kind = papersoccer::BotKind::Random;
  config.frozen_openings.push_back(make_frozen_opening());

  const std::string json = arena::run_matches_json(config);
  require_contains(
      json,
      "\"opening_plies\":4,\"opening_generator\":"
      "\"frozen_uniform_legal_move_data_generation_bank\"",
      "frozen arena openings should identify their data-generation source");
  require_contains(
      json,
      "\"opening_id\":\"development-d4-example-state-hash\","
      "\"phase\":\"development\","
      "\"state_hash\":\"example-state-hash\","
      "\"canonical_key\":\"example-canonical-key\"",
      "frozen opening identity and hashes should survive JSON reporting");
  require_contains(
      json,
      "\"requested_plies\":4,\"actual_plies\":4,\"moves\":["
      "{\"x\":5,\"y\":6},{\"x\":6,\"y\":5},"
      "{\"x\":7,\"y\":4},{\"x\":7,\"y\":5}]",
      "frozen opening reports should preserve the exact transcript");

  arena::MatchesConfig pair_mismatch = config;
  pair_mismatch.seed_pairs = 2;
  require_invalid_argument(
      [&] { (void)arena::run_matches_json(pair_mismatch); },
      "frozen opening count should equal the requested pair count");

  arena::MatchesConfig generated_conflict = config;
  generated_conflict.opening_plies = 4;
  require_invalid_argument(
      [&] { (void)arena::run_matches_json(generated_conflict); },
      "frozen openings should be incompatible with generated opening plies");

  arena::MatchesConfig mismatched_state = config;
  mismatched_state.frozen_openings[0].state.ball = {0, 0};
  require_invalid_argument(
      [&] { (void)arena::run_matches_json(mismatched_state); },
      "a frozen opening state should exactly match its transcript");

  arena::MatchesConfig mismatched_rules = config;
  mismatched_rules.frozen_openings[0].state.config.width = 6;
  require_invalid_argument(
      [&] { (void)arena::run_matches_json(mismatched_rules); },
      "a frozen opening state should use the arena rules");
}

void invalid_tactical_limits_and_policy_are_rejected() {
  arena::MatchesConfig invalid_depth;
  invalid_depth.candidate.quiescence_max_depth = 0;
  require_invalid_argument(
      [&] { (void)arena::run_matches_json(invalid_depth); },
      "arena should reject a zero tactical depth limit");

  arena::MatchesConfig invalid_nodes;
  invalid_nodes.candidate.quiescence_max_nodes =
      papersoccer::MctsConfig::maximum_quiescence_max_nodes + 1U;
  require_invalid_argument(
      [&] { (void)arena::run_matches_json(invalid_nodes); },
      "arena should reject an excessive tactical node limit");

  arena::MatchesConfig invalid_policy;
  invalid_policy.candidate.rollout_policy =
      papersoccer::MctsRolloutPolicy::Uniform;
  require_invalid_argument(
      [&] { (void)arena::run_matches_json(invalid_policy); },
      "arena should require tactical rollouts for tactical quiescence");
}

void invalid_alpha_beta_settings_and_unknown_kinds_are_rejected() {
  arena::PositionsConfig zero_depth;
  zero_depth.candidate.kind = papersoccer::BotKind::AlphaBeta;
  zero_depth.candidate.alpha_beta_depth = 0;
  require_invalid_argument(
      [&] { (void)arena::run_positions_json(zero_depth); },
      "arena should reject a zero alpha-beta depth");

  arena::PositionsConfig excessive_depth;
  excessive_depth.candidate.kind = papersoccer::BotKind::AlphaBeta;
  excessive_depth.candidate.alpha_beta_depth =
      papersoccer::AlphaBetaConfig::maximum_turn_depth + 1U;
  require_invalid_argument(
      [&] { (void)arena::run_positions_json(excessive_depth); },
      "arena should reject an excessive alpha-beta depth");

  arena::PositionsConfig zero_nodes;
  zero_nodes.candidate.kind = papersoccer::BotKind::AlphaBeta;
  zero_nodes.candidate.alpha_beta_max_nodes = 0;
  require_invalid_argument(
      [&] { (void)arena::run_positions_json(zero_nodes); },
      "arena should reject a zero alpha-beta node budget");

  arena::PositionsConfig zero_search_plies;
  zero_search_plies.candidate.kind = papersoccer::BotKind::AlphaBeta;
  zero_search_plies.candidate.alpha_beta_max_search_plies = 0;
  require_invalid_argument(
      [&] { (void)arena::run_positions_json(zero_search_plies); },
      "arena should reject a zero alpha-beta physical horizon");

  arena::PositionsConfig unknown_kind;
  unknown_kind.candidate.kind = static_cast<papersoccer::BotKind>(99);
  require_invalid_argument(
      [&] { (void)arena::run_positions_json(unknown_kind); },
      "arena should reject unknown bot kinds instead of treating them as MCTS");
}

void invalid_opening_length_is_rejected() {
  arena::MatchesConfig config;
  config.opening_plies = 8;
  config.max_plies = 8;
  require_invalid_argument(
      [&] { (void)arena::run_matches_json(config); },
      "arena should reserve at least one competitive ply after an opening");
}

void build_provenance_is_machine_readable() {
  const std::string json = arena::build_provenance_json();
  require_contains(json, "\"schema\":\"papersoccer.arena-build.v1\"",
                   "arena should expose a build-provenance schema");
  require_contains(json, "\"build_type\":",
                   "arena provenance should expose its configured build type");
  require_contains(json, "\"sanitizers_enabled\":",
                   "arena provenance should expose sanitizer instrumentation");
  require_contains(json, "\"compiler_id\":",
                   "arena provenance should expose its compiler identity");
  require_contains(json, "\"configured_flags\":",
                   "arena provenance should expose configured release flags");
  require_contains(json, "\"cxx_standard\":202002",
                   "arena provenance should identify the C++20 language level");
}

}  // namespace

int main() {
  try {
    matches_report_has_paired_games_and_measurements();
    randomized_openings_are_shared_reproducible_and_ply_aware();
    positions_report_measures_both_bots_on_shared_positions();
    alpha_beta_is_configured_measured_and_summarized();
    jacek_inspired_is_configured_measured_and_summarized();
    rank5_derived_is_measured_with_fresh_root_diagnostics();
    truncations_are_invalid_not_draws_and_completed_pairs_remain_scored();
    frozen_openings_are_replayed_and_validated_programmatically();
    invalid_tactical_limits_and_policy_are_rejected();
    invalid_alpha_beta_settings_and_unknown_kinds_are_rejected();
    invalid_opening_length_is_rejected();
    build_provenance_is_machine_readable();
    std::cout << "[PASS] arena smoke test\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] arena smoke test: " << error.what() << '\n';
    return 1;
  }
}
