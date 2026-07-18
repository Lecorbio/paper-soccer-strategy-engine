#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "papersoccer/arena.hpp"

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
  require_contains(json, "\"median_nodes_per_second\":",
                   "arena timing should report comparable node throughput");
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

}  // namespace

int main() {
  try {
    matches_report_has_paired_games_and_measurements();
    randomized_openings_are_shared_reproducible_and_ply_aware();
    positions_report_measures_both_bots_on_shared_positions();
    alpha_beta_is_configured_measured_and_summarized();
    invalid_tactical_limits_and_policy_are_rejected();
    invalid_alpha_beta_settings_and_unknown_kinds_are_rejected();
    invalid_opening_length_is_rejected();
    std::cout << "[PASS] arena smoke test\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] arena smoke test: " << error.what() << '\n';
    return 1;
  }
}
