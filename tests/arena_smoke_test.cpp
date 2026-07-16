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

}  // namespace

int main() {
  try {
    matches_report_has_paired_games_and_measurements();
    positions_report_measures_both_bots_on_shared_positions();
    invalid_tactical_limits_and_policy_are_rejected();
    std::cout << "[PASS] arena smoke test\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] arena smoke test: " << error.what() << '\n';
    return 1;
  }
}
