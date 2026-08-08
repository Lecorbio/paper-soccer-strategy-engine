#pragma once

#include <vector>

#include "papersoccer/bot.hpp"
#include "papersoccer/types.hpp"

namespace papersoccer::detail {

struct CompleteTurnSearchResult {
  std::vector<Move> action{};
  CompleteTurnSearchStats stats{};
};

bool supports_complete_turn_search(const RulesConfig &config) noexcept;

bool same_game_state(const GameState &left, const GameState &right);

void validate_complete_turn_action(const GameState &state,
                                   const std::vector<Move> &action);

CompleteTurnSearchResult run_complete_turn_search(
    const GameState &state, const CompleteTurnAnalysisConfig &config);

}  // namespace papersoccer::detail
