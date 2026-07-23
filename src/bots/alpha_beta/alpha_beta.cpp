#include "papersoccer/bot.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "alpha_beta_internal.hpp"
#include "jacek_inspired/jacek_neural_internal.hpp"
#include "mcts_internal.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

namespace papersoccer {

namespace {

constexpr int kMateScore = 1'000'000;
constexpr int kInfinity = 2'000'000;
constexpr int kMaximumEvaluation = 100'000;

constexpr int kDirectGoalWeight = 50'000;
constexpr int kGoalDistanceWeight = 120;
constexpr int kVerticalProgressWeight = 80;
constexpr int kContinuationWeight = 35;
constexpr int kMobilityWeight = 20;
constexpr int kForwardMoveWeight = 10;
constexpr int kCenterAlignmentWeight = 6;
constexpr int kTempoWeight = 15;

class SearchBudgetReached final {};

using SearchClock = std::chrono::steady_clock;

std::optional<SearchClock::time_point> search_deadline(
    const AlphaBetaConfig &config) {
  if (config.max_time_ms == 0) {
    return std::nullopt;
  }
  return SearchClock::now() + std::chrono::milliseconds(config.max_time_ms);
}

struct TranspositionEntry {
  detail::PositionKey key{};
  std::uint32_t depth{};
  int score{};
  Move best_move{};
  AlphaBetaScoreBound bound{AlphaBetaScoreBound::Exact};
  bool occupied{};
};

class TranspositionTable {
 public:
  explicit TranspositionTable(std::size_t entries) : entries_(entries) {}

  const TranspositionEntry *find(detail::PositionKey key) const noexcept {
    if (entries_.empty()) {
      return nullptr;
    }
    const TranspositionEntry &entry = entries_[index(key)];
    return entry.occupied && entry.key == key ? &entry : nullptr;
  }

  bool store(detail::PositionKey key, std::uint32_t depth, int score,
             Move best_move, AlphaBetaScoreBound bound) noexcept {
    if (entries_.empty()) {
      return false;
    }
    TranspositionEntry &entry = entries_[index(key)];
    if (entry.occupied && entry.depth > depth) {
      return false;
    }
    entry = TranspositionEntry{key, depth, score, best_move, bound, true};
    return true;
  }

 private:
  std::vector<TranspositionEntry> entries_{};

  std::size_t index(detail::PositionKey key) const noexcept {
    const std::uint64_t combined =
        key.first ^ ((key.second << 23U) | (key.second >> 41U));
    return static_cast<std::size_t>(combined % entries_.size());
  }
};

class ScopedMove {
 public:
  ScopedMove(detail::SearchPosition &position, std::uint8_t slot)
      : position_(position) {
    position_.make_move(slot);
  }

  ~ScopedMove() { position_.unmake_move(); }

  ScopedMove(const ScopedMove &) = delete;
  ScopedMove &operator=(const ScopedMove &) = delete;

 private:
  detail::SearchPosition &position_;
};

struct OrderedMove {
  std::uint8_t slot{};
  Move move{};
  int ordering_score{};
};

struct RootResult {
  Move move{};
  int score{};
  std::vector<AlphaBetaRootMove> moves{};
};

void validate_config(const AlphaBetaConfig &config) {
  if (config.max_turn_depth == 0 ||
      config.max_turn_depth > AlphaBetaConfig::maximum_turn_depth) {
    throw std::invalid_argument(
        "alpha-beta turn depth must be between one and " +
        std::to_string(AlphaBetaConfig::maximum_turn_depth));
  }
  if (config.max_nodes == 0) {
    throw std::invalid_argument("alpha-beta node budget must be greater than zero");
  }
  if (config.max_search_plies == 0 ||
      config.max_search_plies > AlphaBetaConfig::maximum_search_plies) {
    throw std::invalid_argument(
        "alpha-beta search ply limit must be between one and " +
        std::to_string(AlphaBetaConfig::maximum_search_plies));
  }
}

int player_sign(Player player) noexcept {
  return player == Player::One ? 1 : -1;
}

class AlphaBetaSearch {
 public:
  AlphaBetaSearch(const GameState &state, const AlphaBetaConfig &config,
                  detail::AlphaBetaLeafEvaluator evaluator)
      : config_(config),
        evaluator_(evaluator),
        deadline_(search_deadline(config)),
        topology_(std::make_shared<detail::SearchTopology>(state.config)),
        position_(topology_, state), table_(config.transposition_table_entries),
        distances_(topology_->vertex_count(), -1),
        queue_(topology_->vertex_count()) {
    validate_config(config_);
    if (evaluator_ == detail::AlphaBetaLeafEvaluator::JacekNeural) {
      jacek_evaluator_.emplace(topology_);
    }
  }

  Move run() {
    if (position_.is_terminal()) {
      throw std::invalid_argument("bot cannot choose move from terminal state");
    }

    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position_.legal_slots(slots);
    if (count == 0) {
      throw std::invalid_argument("bot cannot choose move without legal moves");
    }

    const std::vector<OrderedMove> fallback_order = ordered_moves(std::nullopt);
    Move committed_move = fallback_order.front().move;
    stats_.principal_variation = {committed_move};

    for (std::uint32_t depth = 1; depth <= config_.max_turn_depth; ++depth) {
      stats_.attempted_turn_depth = depth;
      try {
        RootResult result = search_root(depth);
        committed_move = result.move;
        stats_.completed_turn_depth = depth;
        stats_.root_score = result.score;
        stats_.root_moves = std::move(result.moves);
        stats_.principal_variation =
            build_principal_variation(committed_move, depth);

        if (std::abs(stats_.root_score) >=
            kMateScore -
                static_cast<int>(AlphaBetaConfig::maximum_search_plies)) {
          break;
        }
      } catch (const SearchBudgetReached &) {
        stats_.budget_exhausted = true;
        break;
      }
    }

    return committed_move;
  }

  AlphaBetaSearchStats stats() const { return stats_; }

 private:
  AlphaBetaConfig config_{};
  detail::AlphaBetaLeafEvaluator evaluator_{
      detail::AlphaBetaLeafEvaluator::Handcrafted};
  std::optional<SearchClock::time_point> deadline_{};
  std::shared_ptr<const detail::SearchTopology> topology_{};
  detail::SearchPosition position_;
  TranspositionTable table_;
  std::optional<detail::JacekNeuralEvaluator> jacek_evaluator_{};
  AlphaBetaSearchStats stats_{};
  std::vector<int> distances_{};
  std::vector<detail::SearchTopology::VertexIndex> queue_{};

  void visit_node(std::uint32_t physical_ply) {
    if (stats_.nodes >= config_.max_nodes) {
      throw SearchBudgetReached{};
    }
    if ((stats_.nodes & 31U) == 0U && deadline_.has_value() &&
        SearchClock::now() >= *deadline_) {
      throw SearchBudgetReached{};
    }
    ++stats_.nodes;
    stats_.max_physical_ply =
        std::max(stats_.max_physical_ply, physical_ply);
  }

  int terminal_score(std::uint32_t physical_ply) {
    ++stats_.terminal_nodes;
    const std::optional<Player> winning_player = position_.winner();
    if (!winning_player.has_value()) {
      throw std::logic_error("terminal alpha-beta position has no winner");
    }
    const int distance = static_cast<int>(std::min<std::uint32_t>(
        physical_ply, static_cast<std::uint32_t>(kMateScore / 2)));
    return *winning_player == Player::One ? kMateScore - distance
                                          : -kMateScore + distance;
  }

  int shortest_goal_distance(Player player) {
    std::fill(distances_.begin(), distances_.end(), -1);
    std::size_t head = 0;
    std::size_t tail = 0;
    const detail::SearchTopology::VertexIndex start = position_.ball_vertex();
    distances_[start] = 0;
    queue_[tail++] = start;

    while (head < tail) {
      const detail::SearchTopology::VertexIndex vertex = queue_[head++];
      const auto &adjacency = topology_->adjacency(vertex, player);
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const detail::SearchTopology::Arc &arc = adjacency.arcs[index];
        if (position_.edge_used(arc.edge) ||
            distances_[arc.destination] >= 0) {
          continue;
        }
        const int distance = distances_[vertex] + 1;
        distances_[arc.destination] = distance;
        if (is_attacking_goal(topology_->config(),
                              topology_->point(arc.destination), player)) {
          return distance;
        }
        queue_[tail++] = arc.destination;
      }
    }
    return static_cast<int>(topology_->vertex_count()) + 8;
  }

  int evaluate_handcrafted() {
    const Player mover = position_.to_move();
    const int sign = player_sign(mover);
    const Point ball = position_.ball();
    const RulesConfig &rules = topology_->config();

    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position_.legal_slots(slots);
    int continuations = 0;
    int forward_moves = 0;
    bool direct_goal = false;
    for (std::uint8_t index = 0; index < count; ++index) {
      const std::uint8_t slot = slots[index];
      const Point destination = position_.move_for_slot(slot).to;
      direct_goal =
          direct_goal || is_attacking_goal(rules, destination, mover);
      continuations += position_.grants_extra_turn(slot) ? 1 : 0;
      const bool forward = mover == Player::One ? destination.y < ball.y
                                                : destination.y > ball.y;
      forward_moves += forward ? 1 : 0;
    }

    const int player_one_distance = shortest_goal_distance(Player::One);
    const int player_two_distance = shortest_goal_distance(Player::Two);
    const int distance_advantage = player_two_distance - player_one_distance;
    const int vertical_progress = rules.height + 2 - 2 * ball.y;
    const int center_alignment =
        rules.width / 2 - std::abs(ball.x - rules.width / 2);

    int score = distance_advantage * kGoalDistanceWeight +
                vertical_progress * kVerticalProgressWeight;
    score += sign * (static_cast<int>(count) * kMobilityWeight +
                     continuations * kContinuationWeight +
                     forward_moves * kForwardMoveWeight +
                     center_alignment * kCenterAlignmentWeight +
                     kTempoWeight);
    if (direct_goal) {
      score += sign * kDirectGoalWeight;
    }
    return std::clamp(score, -kMaximumEvaluation, kMaximumEvaluation);
  }

  int evaluate() {
    ++stats_.leaf_evaluations;
    if (evaluator_ == detail::AlphaBetaLeafEvaluator::JacekNeural) {
      if (!jacek_evaluator_.has_value()) {
        throw std::logic_error("Jacek evaluator was not initialized");
      }
      return jacek_evaluator_->evaluate(position_);
    }
    return evaluate_handcrafted();
  }

  std::vector<OrderedMove> ordered_moves(
      std::optional<Move> preferred_move) {
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position_.legal_slots(slots);
    std::vector<OrderedMove> ordered;
    ordered.reserve(count);
    const Player mover = position_.to_move();
    const Point ball = position_.ball();

    for (std::uint8_t index = 0; index < count; ++index) {
      const std::uint8_t slot = slots[index];
      const Move move = position_.move_for_slot(slot);
      const bool continuation = position_.grants_extra_turn(slot);
      int score = continuation ? 10'000 : 0;
      const int progress = mover == Player::One ? ball.y - move.to.y
                                                : move.to.y - ball.y;
      score += progress * 100;
      score -= std::abs(move.to.x - topology_->config().width / 2) * 4;
      if (preferred_move.has_value() && move == *preferred_move) {
        score += 2'000'000;
      }

      {
        ScopedMove played(position_, slot);
        if (position_.is_terminal()) {
          const std::optional<Player> winning_player = position_.winner();
          score += winning_player == mover ? 1'000'000 : -1'000'000;
        } else {
          std::array<std::uint8_t, detail::kMaximumMoves> child_slots{};
          const int child_mobility = position_.legal_slots(child_slots);
          score += position_.to_move() == mover ? child_mobility * 12
                                                : -child_mobility * 12;
        }
      }
      ordered.push_back(OrderedMove{slot, move, score});
    }

    std::stable_sort(
        ordered.begin(), ordered.end(),
        [mover](const OrderedMove &lhs, const OrderedMove &rhs) {
          if (lhs.ordering_score != rhs.ordering_score) {
            return lhs.ordering_score > rhs.ordering_score;
          }
          if (lhs.move.to.y != rhs.move.to.y) {
            return mover == Player::One ? lhs.move.to.y < rhs.move.to.y
                                        : lhs.move.to.y > rhs.move.to.y;
          }
          return lhs.move.to.x < rhs.move.to.x;
        });
    return ordered;
  }

  const TranspositionEntry *probe(detail::PositionKey key) {
    if (config_.transposition_table_entries == 0) {
      return nullptr;
    }
    ++stats_.transposition_probes;
    const TranspositionEntry *entry = table_.find(key);
    if (entry != nullptr) {
      ++stats_.transposition_hits;
    }
    return entry;
  }

  void store(detail::PositionKey key, std::uint32_t depth, int score,
             Move best_move, AlphaBetaScoreBound bound) {
    if (table_.store(key, depth, score, best_move, bound)) {
      ++stats_.transposition_stores;
    }
  }

  bool physical_horizon_reached(std::uint32_t physical_ply) {
    if (physical_ply < config_.max_search_plies) {
      return false;
    }
    if (physical_ply >= AlphaBetaConfig::maximum_search_plies) {
      return true;
    }
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    return position_.legal_slots(slots) != 1;
  }

  int search(std::uint32_t turn_depth, std::uint32_t physical_ply,
             int alpha, int beta) {
    visit_node(physical_ply);
    if (position_.is_terminal()) {
      return terminal_score(physical_ply);
    }
    if (turn_depth == 0) {
      return evaluate();
    }
    if (physical_horizon_reached(physical_ply)) {
      ++stats_.physical_ply_cutoffs;
      return evaluate();
    }

    const detail::PositionKey key = position_.position_key();
    const int original_alpha = alpha;
    const int original_beta = beta;
    std::optional<Move> preferred_move;
    if (const TranspositionEntry *entry = probe(key)) {
      preferred_move = entry->best_move;
      if (entry->depth >= turn_depth) {
        if (entry->bound == AlphaBetaScoreBound::Exact) {
          ++stats_.transposition_cutoffs;
          return entry->score;
        }
        if (entry->bound == AlphaBetaScoreBound::Lower) {
          alpha = std::max(alpha, entry->score);
        } else {
          beta = std::min(beta, entry->score);
        }
        if (alpha >= beta) {
          ++stats_.transposition_cutoffs;
          return entry->score;
        }
      }
    }

    const Player mover = position_.to_move();
    const bool maximizing = mover == Player::One;
    int best_score = maximizing ? -kInfinity : kInfinity;
    std::optional<Move> best_move;
    const std::vector<OrderedMove> moves = ordered_moves(preferred_move);
    for (const OrderedMove &candidate : moves) {
      int score = 0;
      {
        ScopedMove played(position_, candidate.slot);
        const bool handoff =
            !position_.is_terminal() && position_.to_move() != mover;
        const std::uint32_t child_depth =
            turn_depth - (handoff ? 1U : 0U);
        score = search(child_depth, physical_ply + 1U, alpha, beta);
      }

      if (!best_move.has_value() ||
          (maximizing ? score > best_score : score < best_score)) {
        best_score = score;
        best_move = candidate.move;
      }
      if (maximizing) {
        alpha = std::max(alpha, best_score);
      } else {
        beta = std::min(beta, best_score);
      }
      if (alpha >= beta) {
        ++stats_.cutoffs;
        break;
      }
    }

    if (!best_move.has_value()) {
      throw std::logic_error("alpha-beta reached a non-terminal node without moves");
    }
    AlphaBetaScoreBound bound = AlphaBetaScoreBound::Exact;
    if (best_score <= original_alpha) {
      bound = AlphaBetaScoreBound::Upper;
    } else if (best_score >= original_beta) {
      bound = AlphaBetaScoreBound::Lower;
    }
    store(key, turn_depth, best_score, *best_move, bound);
    return best_score;
  }

  RootResult search_root(std::uint32_t turn_depth) {
    visit_node(0);
    const detail::PositionKey key = position_.position_key();
    std::optional<Move> preferred_move;
    if (const TranspositionEntry *entry = probe(key)) {
      preferred_move = entry->best_move;
    }

    const Player mover = position_.to_move();
    const bool maximizing = mover == Player::One;
    int alpha = -kInfinity;
    int beta = kInfinity;
    int best_score = maximizing ? -kInfinity : kInfinity;
    std::optional<Move> best_move;
    std::vector<AlphaBetaRootMove> root_moves;

    const std::vector<OrderedMove> moves = ordered_moves(preferred_move);
    root_moves.reserve(moves.size());
    for (const OrderedMove &candidate : moves) {
      const int child_alpha = alpha;
      const int child_beta = beta;
      int score = 0;
      {
        ScopedMove played(position_, candidate.slot);
        const bool handoff =
            !position_.is_terminal() && position_.to_move() != mover;
        const std::uint32_t child_depth =
            turn_depth - (handoff ? 1U : 0U);
        score = search(child_depth, 1, alpha, beta);
      }
      AlphaBetaScoreBound score_bound = AlphaBetaScoreBound::Exact;
      if (score <= child_alpha) {
        score_bound = AlphaBetaScoreBound::Upper;
      } else if (score >= child_beta) {
        score_bound = AlphaBetaScoreBound::Lower;
      }
      root_moves.push_back(
          AlphaBetaRootMove{candidate.move, score, score_bound});
      if (!best_move.has_value() ||
          (maximizing ? score > best_score : score < best_score)) {
        best_score = score;
        best_move = candidate.move;
      }
      if (maximizing) {
        alpha = std::max(alpha, best_score);
      } else {
        beta = std::min(beta, best_score);
      }
    }

    if (!best_move.has_value()) {
      throw std::logic_error("alpha-beta root has no legal move");
    }
    store(key, turn_depth, best_score, *best_move,
          AlphaBetaScoreBound::Exact);
    return RootResult{*best_move, best_score, std::move(root_moves)};
  }

  std::vector<Move> build_principal_variation(Move root_move,
                                               std::uint32_t turn_depth) {
    if (turn_depth == 0) {
      return {root_move};
    }
    std::vector<Move> result;
    const std::size_t root_undo_depth = position_.undo_depth();
    Move next_move = root_move;
    try {
      for (std::uint32_t physical_ply = 0;
           physical_ply < AlphaBetaConfig::maximum_search_plies &&
           !position_.is_terminal() &&
           turn_depth > 0;
           ++physical_ply) {
        std::uint8_t slot = 0;
        if (!position_.slot_for_move(next_move, slot)) {
          break;
        }
        const Player mover = position_.to_move();
        result.push_back(next_move);
        position_.make_move(slot);
        const bool handoff =
            !position_.is_terminal() && position_.to_move() != mover;
        if (handoff) {
          --turn_depth;
        }
        if (position_.is_terminal() || turn_depth == 0) {
          break;
        }
        const TranspositionEntry *entry = table_.find(position_.position_key());
        if (entry == nullptr) {
          break;
        }
        next_move = entry->best_move;
      }
    } catch (...) {
      position_.unmake_to(root_undo_depth);
      throw;
    }
    position_.unmake_to(root_undo_depth);
    return result;
  }
};

}  // namespace

namespace detail {

void validate_alpha_beta_config(const AlphaBetaConfig &config) {
  validate_config(config);
}

Move run_alpha_beta_search(const GameState &state,
                           const AlphaBetaConfig &config,
                           AlphaBetaLeafEvaluator evaluator,
                           AlphaBetaSearchStats &stats) {
  AlphaBetaSearch search(state, config, evaluator);
  const Move move = search.run();
  stats = search.stats();
  return move;
}

}  // namespace detail

AlphaBetaBot::AlphaBetaBot(AlphaBetaConfig config) : config_(config) {
  detail::validate_alpha_beta_config(config_);
}

std::string_view AlphaBetaBot::name() const noexcept { return "AlphaBetaBot"; }

Move AlphaBetaBot::choose_move(const GameState &state) {
  return detail::run_alpha_beta_search(
      state, config_, detail::AlphaBetaLeafEvaluator::Handcrafted,
      last_search_stats_);
}

const AlphaBetaConfig &AlphaBetaBot::config() const noexcept { return config_; }

const AlphaBetaSearchStats &AlphaBetaBot::last_search_stats() const noexcept {
  return last_search_stats_;
}

}  // namespace papersoccer
