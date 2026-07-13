#include "papersoccer/bot.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "papersoccer/rules.hpp"

namespace papersoccer {

namespace {

constexpr std::size_t kNoNode = std::numeric_limits<std::size_t>::max();

void validate_exploration(double exploration) {
  if (!std::isfinite(exploration) || exploration < 0.0) {
    throw std::invalid_argument("MCTS exploration must be finite and non-negative");
  }
}

std::optional<Move> immediate_winning_move(const GameState &state,
                                           const std::vector<Move> &moves) {
  for (const Move move : moves) {
    const GameState next = apply_move(state, move);
    const std::optional<Player> winning_player = winner(next);
    if (winning_player.has_value() && *winning_player == state.to_move) {
      return move;
    }
  }
  return std::nullopt;
}

}  // namespace

class MctsSearch::Impl {
 public:
  Impl(const GameState &root, std::uint64_t seed, double exploration)
      : root_(root), root_moves_(legal_moves(root)), random_state_(seed),
        exploration_(exploration) {
    validate_exploration(exploration_);
    if (root_moves_.empty()) {
      throw std::invalid_argument("bot cannot choose move without legal moves");
    }
    immediate_winning_move_ = immediate_winning_move(root_, root_moves_);

    Node root_node;
    root_node.unexpanded_moves = root_moves_;
    nodes_.push_back(std::move(root_node));
  }

  void run_iterations(std::uint32_t count) {
    if (immediate_winning_move_.has_value()) {
      return;
    }
    if (count > std::numeric_limits<std::uint32_t>::max() - iterations_) {
      throw std::overflow_error("MCTS iteration count overflow");
    }

    for (std::uint32_t i = 0; i < count; ++i) {
      run_iteration();
    }
  }

  Move best_move() const {
    if (immediate_winning_move_.has_value()) {
      return *immediate_winning_move_;
    }

    const Node &root_node = nodes_.front();
    std::size_t best_child = kNoNode;
    std::uint32_t best_visits = 0;
    double best_mean = 0.0;

    for (const Move root_move : root_moves_) {
      std::size_t matching_child = kNoNode;
      for (const std::size_t child_index : root_node.children) {
        if (nodes_[child_index].incoming_move == root_move) {
          matching_child = child_index;
          break;
        }
      }
      if (matching_child == kNoNode) {
        continue;
      }

      const Node &child = nodes_[matching_child];
      const double raw_mean = child.player_one_reward / static_cast<double>(child.visits);
      const double decision_mean =
          root_.to_move == Player::One ? raw_mean : -raw_mean;
      if (best_child == kNoNode || child.visits > best_visits ||
          (child.visits == best_visits && decision_mean > best_mean)) {
        best_child = matching_child;
        best_visits = child.visits;
        best_mean = decision_mean;
      }
    }

    if (best_child == kNoNode) {
      throw std::logic_error("MCTS search has not completed any iterations");
    }
    return nodes_[best_child].incoming_move;
  }

  SearchStats stats() const noexcept {
    if (immediate_winning_move_.has_value()) {
      const double root_value = root_.to_move == Player::One ? 1.0 : -1.0;
      return SearchStats{0, nodes_.size(), 0, root_value};
    }

    const Node &root_node = nodes_.front();
    const double root_value =
        root_node.visits == 0
            ? 0.0
            : root_node.player_one_reward / static_cast<double>(root_node.visits);
    return SearchStats{iterations_, nodes_.size(), simulated_plies_, root_value};
  }

 private:
  struct Node {
    std::size_t parent{kNoNode};
    std::vector<std::size_t> children{};
    Move incoming_move{};
    std::uint32_t visits{};
    double player_one_reward{};
    std::vector<Move> unexpanded_moves{};
  };

  GameState root_;
  std::vector<Move> root_moves_;
  std::vector<Node> nodes_;
  std::optional<Move> immediate_winning_move_;
  std::uint64_t random_state_;
  double exploration_;
  std::uint32_t iterations_{};
  std::uint64_t simulated_plies_{};

  std::uint64_t next_random() noexcept {
    random_state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = random_state_;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }

  std::size_t random_index(std::size_t upper_bound) noexcept {
    const std::uint64_t bound = static_cast<std::uint64_t>(upper_bound);
    const std::uint64_t threshold = (std::uint64_t{0} - bound) % bound;
    std::uint64_t value = 0;
    do {
      value = next_random();
    } while (value < threshold);
    return static_cast<std::size_t>(value % bound);
  }

  std::size_t select_child(std::size_t parent_index, Player decision_player) const {
    const Node &parent = nodes_[parent_index];
    const double log_parent_visits = std::log(static_cast<double>(parent.visits));
    std::size_t best_child = kNoNode;
    double best_score = 0.0;

    for (const std::size_t child_index : parent.children) {
      const Node &child = nodes_[child_index];
      const double raw_mean =
          child.player_one_reward / static_cast<double>(child.visits);
      const double exploitation =
          decision_player == Player::One ? raw_mean : -raw_mean;
      const double exploration_bonus =
          exploration_ *
          std::sqrt(log_parent_visits / static_cast<double>(child.visits));
      const double score = exploitation + exploration_bonus;
      if (best_child == kNoNode || score > best_score) {
        best_child = child_index;
        best_score = score;
      }
    }

    if (best_child == kNoNode) {
      throw std::logic_error("MCTS selection reached a node without children");
    }
    return best_child;
  }

  std::size_t expand(std::size_t parent_index, GameState &state) {
    std::vector<Move> &unexpanded = nodes_[parent_index].unexpanded_moves;
    const std::size_t move_index = random_index(unexpanded.size());
    const Move move = unexpanded[move_index];
    unexpanded[move_index] = unexpanded.back();
    unexpanded.pop_back();

    state = apply_move(state, move);

    Node child;
    child.parent = parent_index;
    child.incoming_move = move;
    child.unexpanded_moves = legal_moves(state);
    const std::size_t child_index = nodes_.size();
    nodes_.push_back(std::move(child));
    nodes_[parent_index].children.push_back(child_index);
    return child_index;
  }

  void run_iteration() {
    GameState state = root_;
    std::vector<std::size_t> visited_nodes{0};
    std::size_t node_index = 0;

    while (!is_terminal(state)) {
      if (!nodes_[node_index].unexpanded_moves.empty()) {
        node_index = expand(node_index, state);
        visited_nodes.push_back(node_index);
        break;
      }

      const std::size_t child_index = select_child(node_index, state.to_move);
      const Move move = nodes_[child_index].incoming_move;
      state = apply_move(state, move);
      node_index = child_index;
      visited_nodes.push_back(node_index);
    }

    while (!is_terminal(state)) {
      const std::vector<Move> moves = legal_moves(state);
      if (moves.empty()) {
        throw std::logic_error("MCTS rollout reached a blocked non-terminal state");
      }
      state = apply_move(state, moves[random_index(moves.size())]);
      ++simulated_plies_;
    }

    const std::optional<Player> winning_player = winner(state);
    if (!winning_player.has_value()) {
      throw std::logic_error("MCTS rollout ended without a winner");
    }
    const double reward = *winning_player == Player::One ? 1.0 : -1.0;
    for (const std::size_t visited_index : visited_nodes) {
      ++nodes_[visited_index].visits;
      nodes_[visited_index].player_one_reward += reward;
    }
    ++iterations_;
  }
};

MctsSearch::MctsSearch(const GameState &root, std::uint64_t seed, double exploration)
    : impl_(std::make_unique<Impl>(root, seed, exploration)) {}

MctsSearch::~MctsSearch() = default;

void MctsSearch::run_iterations(std::uint32_t count) { impl_->run_iterations(count); }

Move MctsSearch::best_move() const { return impl_->best_move(); }

SearchStats MctsSearch::stats() const noexcept { return impl_->stats(); }

MctsBot::MctsBot(MctsConfig config) : config_(config) {
  if (config_.iterations == 0) {
    throw std::invalid_argument("MCTS iterations must be greater than zero");
  }
  validate_exploration(config_.exploration);
}

std::string_view MctsBot::name() const noexcept { return "MctsBot"; }

Move MctsBot::choose_move(const GameState &state) {
  MctsSearch search(state, config_.seed, config_.exploration);
  search.run_iterations(config_.iterations);
  const Move move = search.best_move();
  last_search_stats_ = search.stats();
  return move;
}

SearchStats MctsBot::last_search_stats() const noexcept { return last_search_stats_; }

}  // namespace papersoccer
