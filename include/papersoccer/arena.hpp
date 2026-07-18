#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "papersoccer/bot.hpp"
#include "papersoccer/types.hpp"

namespace papersoccer::arena {

struct ArenaBotConfig {
  BotKind kind{BotKind::Mcts};
  std::uint32_t iterations{2000};
  double exploration{1.4142135623730951};
  MctsRolloutPolicy rollout_policy{MctsRolloutPolicy::Uniform};
  bool reuse_tree{true};
  std::size_t max_nodes{65536};
  MctsLeafPolicy leaf_policy{MctsLeafPolicy::RolloutOnly};
  std::uint32_t quiescence_max_depth{
      MctsConfig::default_quiescence_max_depth};
  std::uint32_t quiescence_max_nodes{
      MctsConfig::default_quiescence_max_nodes};
  std::uint32_t alpha_beta_depth{6};
  std::uint64_t alpha_beta_max_nodes{100'000};
  std::size_t alpha_beta_transposition_table_entries{65'536};
  std::uint32_t alpha_beta_max_search_plies{12};
};

struct MatchesConfig {
  RulesConfig rules{};
  ArenaBotConfig candidate{
      BotKind::Mcts, 2000, 1.4142135623730951,
      MctsRolloutPolicy::Tactical, true, 65536,
      MctsLeafPolicy::TacticalQuiescence};
  ArenaBotConfig reference{
      BotKind::Mcts, 2000, 1.4142135623730951,
      MctsRolloutPolicy::Tactical, true, 65536,
      MctsLeafPolicy::RolloutOnly};
  std::uint64_t base_seed{RandomBot::default_seed()};
  std::size_t seed_pairs{200};
  std::size_t max_plies{512};
  std::size_t bootstrap_samples{10000};
  std::size_t opening_plies{0};
};

struct PositionsConfig {
  RulesConfig rules{};
  ArenaBotConfig candidate{
      BotKind::Mcts, 2000, 1.4142135623730951,
      MctsRolloutPolicy::Tactical, true, 65536,
      MctsLeafPolicy::TacticalQuiescence};
  ArenaBotConfig reference{
      BotKind::Mcts, 2000, 1.4142135623730951,
      MctsRolloutPolicy::Tactical, true, 65536,
      MctsLeafPolicy::RolloutOnly};
  std::uint64_t base_seed{RandomBot::default_seed()};
  std::size_t position_count{16};
  std::size_t generation_plies{24};
};

// Both functions return a complete papersoccer.arena.v1 JSON document.
std::string run_matches_json(const MatchesConfig &config);
std::string run_positions_json(const PositionsConfig &config);

}  // namespace papersoccer::arena
