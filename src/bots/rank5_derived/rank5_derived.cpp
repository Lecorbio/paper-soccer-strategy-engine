#include "papersoccer/bot.hpp"

#include <algorithm>
#include <stdexcept>

#include "papersoccer/rules.hpp"

#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "../../../submissions/codingame/bots/rank_5/bot.cpp"
#undef PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN

namespace papersoccer {
namespace {

namespace rank5 = turn_action_v2;

bool is_supported_config(const RulesConfig &config) noexcept {
  return config.width == 8 && config.height == 10 &&
         config.goal_rule == GoalRule::OpponentGoalOnly &&
         config.blocked_rule == BlockedRule::PlayerToMoveLoses;
}

bool same_state(const GameState &left, const GameState &right) {
  return left.config.width == right.config.width &&
         left.config.height == right.config.height &&
         left.config.goal_rule == right.config.goal_rule &&
         left.config.blocked_rule == right.config.blocked_rule &&
         left.ball == right.ball && left.to_move == right.to_move &&
         left.status == right.status && left.path == right.path &&
         left.used_segments == right.used_segments &&
         left.visit_count == right.visit_count;
}

void copy_search_stats(const rank5::SearchStats &source,
                       Rank5DerivedSearchStats &destination) noexcept {
  destination.completed_turn_depth = source.completed_turn_depth;
  destination.attempted_turn_depth = source.attempted_turn_depth;
  destination.nodes = source.nodes;
  destination.leaf_evaluations = source.leaf_evaluations;
  destination.terminal_nodes = source.terminal_nodes;
  destination.completed_actions = source.completed_actions;
  destination.cutoffs = source.cutoffs;
  destination.transposition_probes = source.transposition_probes;
  destination.transposition_hits = source.transposition_hits;
  destination.transposition_cutoffs = source.transposition_cutoffs;
  destination.transposition_stores = source.transposition_stores;
  destination.continuation_transposition_hits =
      source.continuation_transposition_hits;
  destination.evaluation_cache_probes = source.evaluation_cache_probes;
  destination.evaluation_cache_hits = source.evaluation_cache_hits;
  destination.terminal_bound_cutoffs = source.terminal_bound_cutoffs;
  destination.forced_edges = source.forced_edges;
  destination.root_seed_actions = source.root_seed_actions;
  destination.root_transposition_reuses =
      source.root_transposition_reuses;
  destination.max_action_edges = source.max_action_edges;
  destination.root_score = source.root_score;
  destination.budget_exhausted = source.budget_exhausted;
}

void validate_complete_action(const GameState &state,
                              const std::vector<Move> &action) {
  if (action.empty()) {
    throw std::logic_error("rank-5-derived search returned an empty action");
  }
  GameState replayed = state;
  const Player mover = state.to_move;
  for (const Move move : action) {
    if (is_terminal(replayed) || replayed.to_move != mover) {
      throw std::logic_error(
          "rank-5-derived search returned an overlong action");
    }
    const std::vector<Move> legal = legal_moves(replayed);
    if (std::find(legal.begin(), legal.end(), move) == legal.end()) {
      throw std::logic_error(
          "rank-5-derived search returned an illegal action");
    }
    replayed = apply_move(replayed, move);
  }
  if (!is_terminal(replayed) && replayed.to_move == mover) {
    throw std::logic_error(
        "rank-5-derived search returned an incomplete rebound action");
  }
}

}  // namespace

Rank5DerivedBot::Rank5DerivedBot(Rank5DerivedConfig config)
    : config_(config) {
  if (config_.max_turn_depth == 0 ||
      config_.max_turn_depth > Rank5DerivedConfig::maximum_turn_depth ||
      config_.max_nodes == 0) {
    throw std::invalid_argument("invalid rank-5-derived search configuration");
  }
  if (config_.replay_corrections) {
    throw std::invalid_argument(
        "Rank5DerivedBot does not accept transcript replay corrections");
  }
  if (config_.replay_value_blend_percent < 0 ||
      config_.replay_value_blend_percent > 100) {
    throw std::invalid_argument("invalid replay value blend percentage");
  }
}

std::string_view Rank5DerivedBot::name() const noexcept {
  return "Rank5DerivedBot";
}

Move Rank5DerivedBot::choose_move(const GameState &state) {
  if (!is_supported_config(state.config)) {
    clear_cache();
    throw std::invalid_argument(
        "Rank5DerivedBot requires the standard 8x10 demo rules");
  }

  if (expected_state_.has_value() &&
      same_state(state, *expected_state_) &&
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

  rank5::SearchConfig search_config;
  search_config.max_turn_depth = config_.max_turn_depth;
  search_config.max_nodes = config_.max_nodes;
  search_config.transposition_entries =
      config_.transposition_table_entries;
  search_config.evaluation_entries = config_.evaluation_table_entries;
  search_config.max_time_ms = config_.max_time_ms;
  search_config.absolute_deadline.reset();
  search_config.replay_value_blend_percent =
      config_.replay_value_blend_percent;

  rank5::CompleteTurnSearch search(state, search_config);
  cached_action_ = search.run();
  validate_complete_action(state, cached_action_);

  ++searches_;
  last_search_stats_ = {};
  copy_search_stats(search.stats(), last_search_stats_);
  last_search_stats_.planned_action_length = cached_action_.size();
  last_search_stats_.current_edge_index = 0;
  last_search_stats_.searches = searches_;

  const Move move = cached_action_.front();
  next_cached_move_ = 1;
  last_search_stats_.cached_moves_remaining = cached_action_.size() - 1U;
  if (next_cached_move_ < cached_action_.size()) {
    expected_state_ = apply_move(state, move);
  } else {
    clear_cache();
  }
  return move;
}

const Rank5DerivedConfig &Rank5DerivedBot::config() const noexcept {
  return config_;
}

const Rank5DerivedSearchStats &Rank5DerivedBot::last_search_stats() const
    noexcept {
  return last_search_stats_;
}

void Rank5DerivedBot::clear_cache() noexcept {
  cached_action_.clear();
  next_cached_move_ = 0;
  expected_state_.reset();
  last_search_stats_.cached_moves_remaining = 0;
}

}  // namespace papersoccer
