#include "papersoccer/game_review.hpp"

#include <stdexcept>
#include <utility>

#include "../bots/rank5_derived/complete_turn_search_internal.hpp"
#include "papersoccer/rules.hpp"

namespace papersoccer {
namespace {

void validate_analysis_config(const CompleteTurnAnalysisConfig &config) {
  if (config.max_turn_depth == 0 ||
      config.max_turn_depth > CompleteTurnAnalysisConfig::maximum_turn_depth ||
      config.max_nodes == 0 || config.transposition_table_entries < 2 ||
      config.evaluation_table_entries < 2) {
    throw std::invalid_argument("invalid complete-turn analysis configuration");
  }
}

}  // namespace

CompleteTurnAnalyzer::CompleteTurnAnalyzer(CompleteTurnAnalysisConfig config)
    : config_(config) {
  validate_analysis_config(config_);
}

const CompleteTurnAnalysisConfig &CompleteTurnAnalyzer::config() const
    noexcept {
  return config_;
}

CompleteTurnAnalysis CompleteTurnAnalyzer::analyze(
    const GameState &state) const {
  detail::CompleteTurnSearchResult result =
      detail::run_complete_turn_search(state, config_);
  return CompleteTurnAnalysis{std::move(result.action),
                              result.stats.root_score,
                              std::move(result.stats)};
}

DeepTurnSearchBot::DeepTurnSearchBot(std::uint64_t max_nodes)
    : DeepTurnSearchBot(CompleteTurnAnalysisConfig::deep(max_nodes)) {}

DeepTurnSearchBot::DeepTurnSearchBot(CompleteTurnAnalysisConfig config)
    : config_(config) {
  if (!config_.is_deep_profile()) {
    throw std::invalid_argument(
        "DeepTurnSearchBot requires an unmodified fixed Deep profile");
  }
}

std::string_view DeepTurnSearchBot::name() const noexcept {
  return "DeepTurnSearchBot";
}

Move DeepTurnSearchBot::choose_move(const GameState &state) {
  if (!detail::supports_complete_turn_search(state.config)) {
    clear_cache();
    throw std::invalid_argument(
        "DeepTurnSearchBot requires the standard 8x10 demo rules");
  }

  if (expected_state_.has_value() &&
      detail::same_game_state(state, *expected_state_) &&
      next_cached_move_ < cached_action_.size()) {
    const Move move = cached_action_[next_cached_move_++];
    last_search_stats_.nodes = 0;
    last_search_stats_.cached_continuation = true;
    last_search_stats_.planned_action_length = cached_action_.size();
    last_search_stats_.current_edge_index = next_cached_move_ - 1U;
    last_search_stats_.cached_moves_remaining =
        cached_action_.size() - next_cached_move_;
    if (next_cached_move_ < cached_action_.size()) {
      expected_state_ = apply_move(state, move);
    } else {
      clear_cache();
    }
    return move;
  }
  clear_cache();

  detail::CompleteTurnSearchResult result =
      detail::run_complete_turn_search(state, config_);
  cached_action_ = std::move(result.action);
  ++searches_;
  last_search_stats_ = std::move(result.stats);
  last_search_stats_.planned_action_length = cached_action_.size();
  last_search_stats_.current_edge_index = 0;
  last_search_stats_.cached_moves_remaining = cached_action_.size() - 1U;
  last_search_stats_.searches = searches_;

  const Move move = cached_action_.front();
  next_cached_move_ = 1;
  if (next_cached_move_ < cached_action_.size()) {
    expected_state_ = apply_move(state, move);
  } else {
    clear_cache();
  }
  return move;
}

const CompleteTurnAnalysisConfig &DeepTurnSearchBot::config() const noexcept {
  return config_;
}

const CompleteTurnSearchStats &DeepTurnSearchBot::last_search_stats() const
    noexcept {
  return last_search_stats_;
}

void DeepTurnSearchBot::clear_cache() noexcept {
  cached_action_.clear();
  next_cached_move_ = 0;
  expected_state_.reset();
  last_search_stats_.cached_moves_remaining = 0;
}

}  // namespace papersoccer
