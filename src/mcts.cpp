#include "papersoccer/bot.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mcts_internal.hpp"
#include "papersoccer/rules.hpp"

namespace papersoccer {

namespace {

constexpr std::size_t kNoNode = std::numeric_limits<std::size_t>::max();
constexpr std::size_t kDefaultMaximumNodes = 65536;

void validate_exploration(double exploration) {
  if (!std::isfinite(exploration) || exploration < 0.0) {
    throw std::invalid_argument("MCTS exploration must be finite and non-negative");
  }
}

void validate_max_nodes(std::size_t max_nodes) {
  if (max_nodes < 2) {
    throw std::invalid_argument("MCTS maximum node count must be at least two");
  }
}

bool same_game_state(const GameState &lhs, const GameState &rhs) {
  return detail::same_rules_config(lhs.config, rhs.config) &&
         lhs.ball == rhs.ball && lhs.to_move == rhs.to_move &&
         lhs.status == rhs.status && lhs.path == rhs.path &&
         lhs.used_segments == rhs.used_segments &&
         lhs.visit_count == rhs.visit_count;
}

MctsConfig legacy_search_config(std::uint64_t seed, double exploration) {
  MctsConfig config;
  config.seed = seed;
  config.exploration = exploration;
  config.rollout_policy = MctsRolloutPolicy::Uniform;
  config.reuse_tree = false;
  config.max_nodes = kDefaultMaximumNodes;
  return config;
}

}  // namespace

class MctsSearch::Impl {
 public:
  Impl(const GameState &root, const MctsConfig &config)
      : root_state_(root),
        topology_(std::make_shared<detail::SearchTopology>(root.config)),
        position_(topology_, root), random_state_(config.seed),
        exploration_(config.exploration), policy_(config.rollout_policy),
        max_nodes_(config.max_nodes),
        node_reserve_hint_(std::min(
            config.max_nodes, static_cast<std::size_t>(config.iterations) + 1U)) {
    validate_exploration(exploration_);
    validate_max_nodes(max_nodes_);
    initialize_root();
  }

  void run_iterations(std::uint32_t count) {
    if (immediate_winning_move_.has_value() || root_is_proven()) {
      return;
    }
    if (count > std::numeric_limits<std::uint32_t>::max() - iterations_) {
      throw std::overflow_error("MCTS iteration count overflow");
    }

    for (std::uint32_t i = 0; i < count; ++i) {
      if (root_is_proven()) {
        break;
      }
      run_iteration();
    }
  }

  Move best_move() const {
    if (immediate_winning_move_.has_value()) {
      return *immediate_winning_move_;
    }

    const Node &root = nodes_.front();
    if (solver_enabled()) {
      for (std::uint8_t move_index = 0; move_index < root_move_count_;
           ++move_index) {
        const std::size_t child = child_for_move(0, root_moves_[move_index]);
        if (child != kNoNode && nodes_[child].proven_winner.has_value() &&
            *nodes_[child].proven_winner == root.player) {
          return root_moves_[move_index];
        }
      }
    }

    bool has_non_losing_option = false;
    std::optional<Move> first_unexpanded_move;
    if (solver_enabled()) {
      for (std::uint8_t move_index = 0; move_index < root_move_count_;
           ++move_index) {
        const std::size_t child = child_for_move(0, root_moves_[move_index]);
        if (child == kNoNode) {
          has_non_losing_option = true;
          if (!first_unexpanded_move.has_value()) {
            first_unexpanded_move = root_moves_[move_index];
          }
        } else if (!nodes_[child].proven_winner.has_value() ||
                   *nodes_[child].proven_winner == root.player) {
          has_non_losing_option = true;
        }
      }
    }

    std::size_t best_child = kNoNode;
    std::uint64_t best_visits = 0;
    double best_mean = 0.0;
    for (std::uint8_t move_index = 0; move_index < root_move_count_;
         ++move_index) {
      const std::size_t child_index =
          child_for_move(0, root_moves_[move_index]);
      if (child_index == kNoNode) {
        continue;
      }
      const Node &child = nodes_[child_index];
      if (solver_enabled() && has_non_losing_option &&
          child.proven_winner.has_value() &&
          *child.proven_winner != root.player) {
        continue;
      }

      const double raw_mean =
          child.player_one_reward / static_cast<double>(child.visits);
      const double decision_mean =
          root.player == Player::One ? raw_mean : -raw_mean;
      if (best_child == kNoNode || child.visits > best_visits ||
          (child.visits == best_visits && decision_mean > best_mean)) {
        best_child = child_index;
        best_visits = child.visits;
        best_mean = decision_mean;
      }
    }

    if (best_child == kNoNode) {
      if (first_unexpanded_move.has_value()) {
        return *first_unexpanded_move;
      }
      throw std::logic_error("MCTS search has not completed any iterations");
    }
    return nodes_[best_child].incoming_move;
  }

  SearchStats stats() const noexcept {
    const Node &root = nodes_.front();
    const double root_value =
        immediate_winning_move_.has_value()
            ? (root.player == Player::One ? 1.0 : -1.0)
            : (root.visits == 0
                   ? 0.0
                   : root.player_one_reward / static_cast<double>(root.visits));

    std::size_t proven_nodes = 0;
    for (const Node &node : nodes_) {
      if (node.proven_winner.has_value()) {
        ++proven_nodes;
      }
    }

    SearchStats result;
    result.iterations = iterations_;
    result.nodes = nodes_.size();
    result.simulated_plies = simulated_plies_;
    result.root_value = root_value;
    result.total_root_visits = root.visits;
    result.reused_visits = reused_visits_;
    result.max_depth = max_depth_;
    result.proven_nodes = proven_nodes;
    result.proven_winner = root.proven_winner;
    result.rebuild_count = rebuild_count_;
    result.expansion_saturated = expansion_saturated_;
    return result;
  }

  bool rebase(const GameState &new_root) {
    if (!detail::same_rules_config(root_state_.config, new_root.config)) {
      return false;
    }
    if (is_terminal(new_root)) {
      return false;
    }
    detail::SearchPosition rebased_position(topology_, new_root);
    std::array<std::uint8_t, detail::kMaximumMoves> legal_slots{};
    if (rebased_position.legal_slots(legal_slots) == 0) {
      return false;
    }

    std::size_t target_node = 0;
    if (!same_game_state(root_state_, new_root)) {
      if (root_state_.path.size() > new_root.path.size() ||
          !std::equal(root_state_.path.begin(), root_state_.path.end(),
                      new_root.path.begin())) {
        return false;
      }

      GameState reconstructed = root_state_;
      for (std::size_t path_index = root_state_.path.size();
           path_index < new_root.path.size(); ++path_index) {
        const Move move{new_root.path[path_index]};
        std::size_t matching_child = kNoNode;
        const Node &parent = nodes_[target_node];
        for (std::uint8_t child_index = 0; child_index < parent.child_count;
             ++child_index) {
          const std::size_t candidate = parent.children[child_index];
          if (nodes_[candidate].incoming_move == move) {
            matching_child = candidate;
            break;
          }
        }
        if (matching_child == kNoNode) {
          return false;
        }
        try {
          reconstructed = apply_move(reconstructed, move);
        } catch (const std::invalid_argument &) {
          return false;
        }
        target_node = matching_child;
      }
      if (!same_game_state(reconstructed, new_root)) {
        return false;
      }
    }

    if (target_node != 0) {
      compact_to_subtree(target_node);
    }
    root_state_ = new_root;
    position_ = std::move(rebased_position);
    cache_root_moves();
    immediate_winning_move_ = find_immediate_winning_move(position_);
    if (immediate_winning_move_.has_value()) {
      nodes_.front().proven_winner = nodes_.front().player;
    }

    iterations_ = 0;
    simulated_plies_ = 0;
    max_depth_ = 0;
    expansion_saturated_ = false;
    reused_visits_ = nodes_.front().visits;
    return true;
  }

  void set_rebuild_count(std::uint64_t count) noexcept {
    rebuild_count_ = count;
  }

 private:
  struct Node {
    std::size_t parent{kNoNode};
    std::array<std::size_t, detail::kMaximumMoves> children{};
    std::array<std::uint8_t, detail::kMaximumMoves> unexpanded_slots{};
    Move incoming_move{};
    std::uint64_t visits{};
    double player_one_reward{};
    std::optional<Player> proven_winner{};
    Player player{Player::One};
    std::uint8_t incoming_slot{};
    std::uint8_t child_count{};
    std::uint8_t unexpanded_count{};
    std::uint8_t legal_move_count{};
  };

  GameState root_state_{};
  std::shared_ptr<const detail::SearchTopology> topology_{};
  detail::SearchPosition position_;
  std::vector<Node> nodes_{};
  std::array<Move, detail::kMaximumMoves> root_moves_{};
  std::uint8_t root_move_count_{};
  std::optional<Move> immediate_winning_move_{};
  std::uint64_t random_state_{};
  double exploration_{};
  MctsRolloutPolicy policy_{MctsRolloutPolicy::Uniform};
  std::size_t max_nodes_{};
  std::size_t node_reserve_hint_{};
  std::uint32_t iterations_{};
  std::uint64_t simulated_plies_{};
  std::uint64_t reused_visits_{};
  std::uint64_t rebuild_count_{};
  std::uint32_t max_depth_{};
  bool expansion_saturated_{};
  std::vector<std::size_t> visited_nodes_{};

  bool solver_enabled() const noexcept {
    return policy_ == MctsRolloutPolicy::Tactical;
  }

  bool root_is_proven() const noexcept {
    return solver_enabled() && nodes_.front().proven_winner.has_value();
  }

  void initialize_root() {
    if (position_.is_terminal()) {
      throw std::invalid_argument("bot cannot choose move from terminal state");
    }
    std::array<std::uint8_t, detail::kMaximumMoves> legal_slots{};
    if (position_.legal_slots(legal_slots) == 0) {
      throw std::invalid_argument("bot cannot choose move without legal moves");
    }

    nodes_.reserve(node_reserve_hint_);
    visited_nodes_.reserve(topology_->edge_count() + 1U);
    nodes_.push_back(make_node(position_));
    cache_root_moves();
    immediate_winning_move_ = find_immediate_winning_move(position_);
    if (immediate_winning_move_.has_value()) {
      nodes_.front().proven_winner = nodes_.front().player;
    }
  }

  Node make_node(const detail::SearchPosition &position) const {
    Node node;
    node.player = position.to_move();
    if (position.is_terminal()) {
      node.proven_winner = position.winner();
      return node;
    }
    node.legal_move_count = position.legal_slots(node.unexpanded_slots);
    node.unexpanded_count = node.legal_move_count;
    return node;
  }

  void cache_root_moves() {
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    root_move_count_ = position_.legal_slots(slots);
    for (std::uint8_t i = 0; i < root_move_count_; ++i) {
      root_moves_[i] = position_.move_for_slot(slots[i]);
    }
  }

  std::optional<Move> find_immediate_winning_move(
      detail::SearchPosition &position) {
    const Player player = position.to_move();
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position.legal_slots(slots);
    for (std::uint8_t i = 0; i < count; ++i) {
      const Move move = position.move_for_slot(slots[i]);
      position.make_move(slots[i]);
      const std::optional<Player> winning_player = position.winner();
      position.unmake_move();
      if (winning_player.has_value() && *winning_player == player) {
        return move;
      }
    }
    return std::nullopt;
  }

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

  std::size_t child_for_move(std::size_t parent_index, Move move) const noexcept {
    const Node &parent = nodes_[parent_index];
    for (std::uint8_t child_index = 0; child_index < parent.child_count;
         ++child_index) {
      const std::size_t candidate = parent.children[child_index];
      if (nodes_[candidate].incoming_move == move) {
        return candidate;
      }
    }
    return kNoNode;
  }

  std::size_t select_child(std::size_t parent_index,
                           Player decision_player) const {
    const Node &parent = nodes_[parent_index];
    if (solver_enabled()) {
      for (std::uint8_t i = 0; i < parent.child_count; ++i) {
        const std::size_t child_index = parent.children[i];
        if (nodes_[child_index].proven_winner.has_value() &&
            *nodes_[child_index].proven_winner == decision_player) {
          return child_index;
        }
      }
    }

    bool has_unknown_child = false;
    if (solver_enabled()) {
      for (std::uint8_t i = 0; i < parent.child_count; ++i) {
        if (!nodes_[parent.children[i]].proven_winner.has_value()) {
          has_unknown_child = true;
          break;
        }
      }
    }

    const double log_parent_visits =
        std::log(static_cast<double>(parent.visits));
    std::size_t best_child = kNoNode;
    double best_score = 0.0;
    for (std::uint8_t i = 0; i < parent.child_count; ++i) {
      const std::size_t child_index = parent.children[i];
      const Node &child = nodes_[child_index];
      if (solver_enabled() && has_unknown_child &&
          child.proven_winner.has_value()) {
        continue;
      }
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

  std::size_t expand(std::size_t parent_index) {
    Node &parent = nodes_[parent_index];
    const std::size_t move_index = random_index(parent.unexpanded_count);
    const std::uint8_t slot = parent.unexpanded_slots[move_index];
    parent.unexpanded_slots[move_index] =
        parent.unexpanded_slots[parent.unexpanded_count - 1U];
    --parent.unexpanded_count;

    const Move move = position_.move_for_slot(slot);
    position_.make_move(slot);
    Node child = make_node(position_);
    child.parent = parent_index;
    child.incoming_move = move;
    child.incoming_slot = slot;

    if (nodes_.size() == nodes_.capacity()) {
      const std::size_t doubled =
          nodes_.capacity() > max_nodes_ / 2U ? max_nodes_
                                             : nodes_.capacity() * 2U;
      nodes_.reserve(std::max(nodes_.size() + 1U, doubled));
    }
    const std::size_t child_index = nodes_.size();
    nodes_.push_back(std::move(child));
    nodes_[parent_index].children[nodes_[parent_index].child_count++] = child_index;
    return child_index;
  }

  bool has_immediate_win(detail::SearchPosition &position,
                         Player player) {
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position.legal_slots(slots);
    for (std::uint8_t i = 0; i < count; ++i) {
      position.make_move(slots[i]);
      const std::optional<Player> winning_player = position.winner();
      position.unmake_move();
      if (winning_player.has_value() && *winning_player == player) {
        return true;
      }
    }
    return false;
  }

  std::uint8_t choose_tactical_move(
      const std::array<std::uint8_t, detail::kMaximumMoves> &slots,
      std::uint8_t count) {
    if (count == 1) {
      return slots[0];
    }

    const Player mover = position_.to_move();
    for (std::uint8_t i = 0; i < count; ++i) {
      position_.make_move(slots[i]);
      const std::optional<Player> winning_player = position_.winner();
      position_.unmake_move();
      if (winning_player.has_value() && *winning_player == mover) {
        return slots[i];
      }
    }

    std::array<std::uint8_t, detail::kMaximumMoves> safe_slots{};
    std::uint8_t safe_count = 0;
    for (std::uint8_t i = 0; i < count; ++i) {
      position_.make_move(slots[i]);
      bool unsafe_for_mover = false;
      if (position_.is_terminal()) {
        const std::optional<Player> winning_player = position_.winner();
        unsafe_for_mover =
            !winning_player.has_value() || *winning_player != mover;
      } else if (position_.to_move() == opponent(mover)) {
        unsafe_for_mover =
            has_immediate_win(position_, position_.to_move());
      }
      position_.unmake_move();
      if (!unsafe_for_mover) {
        safe_slots[safe_count++] = slots[i];
      }
    }

    if (safe_count != 0) {
      return safe_slots[random_index(safe_count)];
    }
    return slots[random_index(count)];
  }

  std::uint8_t choose_rollout_move() {
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position_.legal_slots(slots);
    if (count == 0) {
      throw std::logic_error("MCTS rollout reached a blocked non-terminal state");
    }
    if (policy_ == MctsRolloutPolicy::Tactical) {
      return choose_tactical_move(slots, count);
    }
    return slots[random_index(count)];
  }

  void refresh_proof(std::size_t node_index) {
    Node &node = nodes_[node_index];
    if (node.proven_winner.has_value()) {
      return;
    }

    for (std::uint8_t i = 0; i < node.child_count; ++i) {
      const std::optional<Player> child_winner =
          nodes_[node.children[i]].proven_winner;
      if (child_winner.has_value() && *child_winner == node.player) {
        node.proven_winner = node.player;
        return;
      }
    }

    if (node.unexpanded_count != 0 ||
        node.child_count != node.legal_move_count || node.child_count == 0) {
      return;
    }
    for (std::uint8_t i = 0; i < node.child_count; ++i) {
      if (!nodes_[node.children[i]].proven_winner.has_value()) {
        return;
      }
    }
    node.proven_winner = opponent(node.player);
  }

  void run_iteration() {
    const std::size_t initial_undo_depth = position_.undo_depth();
    visited_nodes_.clear();
    visited_nodes_.push_back(0);
    std::size_t node_index = 0;
    std::uint32_t tree_depth = 0;

    try {
      while (!position_.is_terminal()) {
        if (nodes_[node_index].unexpanded_count != 0) {
          if (nodes_.size() >= max_nodes_) {
            expansion_saturated_ = true;
            break;
          }
          node_index = expand(node_index);
          visited_nodes_.push_back(node_index);
          ++tree_depth;
          break;
        }

        const std::size_t child_index =
            select_child(node_index, position_.to_move());
        position_.make_move(nodes_[child_index].incoming_slot);
        node_index = child_index;
        visited_nodes_.push_back(node_index);
        ++tree_depth;
      }

      while (!position_.is_terminal()) {
        position_.make_move(choose_rollout_move());
        ++simulated_plies_;
      }

      const std::optional<Player> winning_player = position_.winner();
      if (!winning_player.has_value()) {
        throw std::logic_error("MCTS rollout ended without a winner");
      }
      const double reward =
          *winning_player == Player::One ? 1.0 : -1.0;
      for (const std::size_t visited_index : visited_nodes_) {
        ++nodes_[visited_index].visits;
        nodes_[visited_index].player_one_reward += reward;
      }
      ++iterations_;
      max_depth_ = std::max(max_depth_, tree_depth);

      if (solver_enabled()) {
        for (auto visited = visited_nodes_.rbegin();
             visited != visited_nodes_.rend(); ++visited) {
          refresh_proof(*visited);
        }
      }
    } catch (...) {
      position_.unmake_to(initial_undo_depth);
      throw;
    }
    position_.unmake_to(initial_undo_depth);
  }

  void compact_to_subtree(std::size_t subtree_root) {
    std::vector<bool> retained(nodes_.size(), false);
    std::vector<std::size_t> stack{subtree_root};
    while (!stack.empty()) {
      const std::size_t node_index = stack.back();
      stack.pop_back();
      if (retained[node_index]) {
        continue;
      }
      retained[node_index] = true;
      const Node &node = nodes_[node_index];
      for (std::uint8_t i = 0; i < node.child_count; ++i) {
        stack.push_back(node.children[i]);
      }
    }

    std::vector<std::size_t> mapping(nodes_.size(), kNoNode);
    std::vector<std::size_t> old_indices{};
    old_indices.reserve(nodes_.size());
    old_indices.push_back(subtree_root);
    for (std::size_t old_index = 0; old_index < nodes_.size(); ++old_index) {
      if (retained[old_index] && old_index != subtree_root) {
        old_indices.push_back(old_index);
      }
    }

    std::vector<Node> compacted{};
    compacted.reserve(std::min(
        max_nodes_, std::max(node_reserve_hint_, old_indices.size())));
    for (const std::size_t old_index : old_indices) {
      mapping[old_index] = compacted.size();
      compacted.push_back(nodes_[old_index]);
    }
    for (std::size_t new_index = 0; new_index < compacted.size(); ++new_index) {
      const std::size_t old_index = old_indices[new_index];
      Node &node = compacted[new_index];
      if (old_index == subtree_root) {
        node.parent = kNoNode;
        node.incoming_move = Move{};
        node.incoming_slot = 0;
      } else {
        node.parent = mapping[nodes_[old_index].parent];
      }
      for (std::uint8_t i = 0; i < node.child_count; ++i) {
        node.children[i] = mapping[nodes_[old_index].children[i]];
      }
    }
    nodes_.swap(compacted);
  }
};

MctsSearch::MctsSearch(const GameState &root, std::uint64_t seed,
                       double exploration)
    : MctsSearch(root, legacy_search_config(seed, exploration)) {}

MctsSearch::MctsSearch(const GameState &root, const MctsConfig &config)
    : impl_(std::make_unique<Impl>(root, config)) {}

MctsSearch::MctsSearch(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

MctsSearch::~MctsSearch() = default;

void MctsSearch::run_iterations(std::uint32_t count) {
  impl_->run_iterations(count);
}

Move MctsSearch::best_move() const { return impl_->best_move(); }

SearchStats MctsSearch::stats() const noexcept { return impl_->stats(); }

std::unique_ptr<MctsSearch> MctsSearch::clone() const {
  return std::unique_ptr<MctsSearch>(
      new MctsSearch(std::make_unique<Impl>(*impl_)));
}

bool MctsSearch::rebase(const GameState &root) { return impl_->rebase(root); }

void MctsSearch::set_rebuild_count(std::uint64_t count) noexcept {
  impl_->set_rebuild_count(count);
}

MctsBot::MctsBot(MctsConfig config) : config_(config) {
  if (config_.iterations == 0) {
    throw std::invalid_argument("MCTS iterations must be greater than zero");
  }
  validate_exploration(config_.exploration);
  validate_max_nodes(config_.max_nodes);
}

MctsBot::MctsBot(const MctsBot &other)
    : config_(other.config_), last_search_stats_(other.last_search_stats_),
      search_(other.search_ ? other.search_->clone() : nullptr),
      rebuild_count_(other.rebuild_count_) {}

MctsBot &MctsBot::operator=(const MctsBot &other) {
  if (this != &other) {
    std::unique_ptr<MctsSearch> cloned_search =
        other.search_ ? other.search_->clone() : nullptr;
    config_ = other.config_;
    last_search_stats_ = other.last_search_stats_;
    search_ = std::move(cloned_search);
    rebuild_count_ = other.rebuild_count_;
  }
  return *this;
}

std::string_view MctsBot::name() const noexcept { return "MctsBot"; }

Move MctsBot::choose_move(const GameState &state) {
  if (!config_.reuse_tree || !search_) {
    search_ = std::make_unique<MctsSearch>(state, config_);
  } else if (!search_->rebase(state)) {
    std::unique_ptr<MctsSearch> rebuilt =
        std::make_unique<MctsSearch>(state, config_);
    ++rebuild_count_;
    search_ = std::move(rebuilt);
  }

  search_->set_rebuild_count(rebuild_count_);
  search_->run_iterations(config_.iterations);
  const Move move = search_->best_move();
  last_search_stats_ = search_->stats();
  return move;
}

SearchStats MctsBot::last_search_stats() const noexcept {
  return last_search_stats_;
}

}  // namespace papersoccer
