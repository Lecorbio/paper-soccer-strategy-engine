#include "papersoccer/game_review.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "exact_endgame_internal.hpp"
#include "../bots/mcts_internal.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

namespace papersoccer {
namespace {

constexpr int kMateScore = 1'000'000;
constexpr int kInfinity = 2'000'000;

class ExactBudgetReached final {};

enum class ExactBound { Exact, Lower, Upper };

struct ExactEntry {
  int score{};
  Move best_move{};
  ExactBound bound{ExactBound::Exact};
};

struct ExactStateKey {
  std::uint32_t used_edges{};
  std::uint32_t visited_vertices{};
  detail::SearchTopology::VertexIndex ball{};
  Player to_move{Player::One};
  Status status{Status::InProgress};

  constexpr bool operator==(const ExactStateKey &) const noexcept = default;
};

struct ExactStateKeyHash {
  std::size_t operator()(const ExactStateKey &key) const noexcept {
    std::uint64_t value = key.used_edges;
    value ^= static_cast<std::uint64_t>(key.visited_vertices) << 32U;
    value ^= static_cast<std::uint64_t>(key.ball) * 0x9e3779b97f4a7c15ULL;
    value ^= static_cast<std::uint64_t>(key.to_move) << 7U;
    value ^= static_cast<std::uint64_t>(key.status) << 11U;
    return static_cast<std::size_t>(detail::mix_position_key(value));
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

bool is_root_possession_boundary(const GameState &state) {
  if (state.path.empty() || state.path.back() != state.ball) {
    return false;
  }
  if (state.path.size() == 1U) {
    return true;
  }
  const auto visits = state.visit_count.find(state.ball);
  const int visit_count =
      visits == state.visit_count.end() ? 0 : visits->second;
  return !is_boundary_point(state.config, state.ball) && visit_count == 1;
}

struct ReachableComponent {
  std::vector<detail::SearchTopology::EdgeIndex> edges{};
  std::vector<detail::SearchTopology::VertexIndex> vertices{};
};

ReachableComponent reachable_unused_component(
    const detail::SearchTopology &topology,
    const detail::SearchPosition &position) {
  std::vector<bool> reached_vertices(topology.vertex_count(), false);
  std::vector<bool> reached_edges(topology.edge_count(), false);
  std::vector<detail::SearchTopology::VertexIndex> queue;
  queue.reserve(topology.vertex_count());
  queue.push_back(position.ball_vertex());
  reached_vertices[position.ball_vertex()] = true;

  ReachableComponent component;
  component.vertices.push_back(position.ball_vertex());

  std::size_t head = 0;
  while (head < queue.size()) {
    const auto vertex = queue[head++];
    for (const Player player : {Player::One, Player::Two}) {
      const auto &adjacency = topology.adjacency(vertex, player);
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const auto &arc = adjacency.arcs[index];
        if (position.edge_used(arc.edge)) {
          continue;
        }
        if (!reached_edges[arc.edge]) {
          reached_edges[arc.edge] = true;
          component.edges.push_back(arc.edge);
        }
        if (!reached_vertices[arc.destination]) {
          reached_vertices[arc.destination] = true;
          queue.push_back(arc.destination);
          component.vertices.push_back(arc.destination);
        }
      }
    }
  }
  return component;
}

int normalized_direction_rank(Player mover, Point from, Point to) {
  int dx = to.x - from.x;
  int dy = to.y - from.y;
  if (mover == Player::Two) {
    dx = -dx;
    dy = -dy;
  }
  constexpr std::array<Point, 8> order{{
      {0, -1}, {1, -1}, {-1, -1}, {1, 0},
      {-1, 0}, {1, 1},  {-1, 1},  {0, 1},
  }};
  for (std::size_t index = 0; index < order.size(); ++index) {
    if (order[index] == Point{dx, dy}) {
      return static_cast<int>(index);
    }
  }
  return static_cast<int>(order.size());
}

class ExactSearch {
 public:
  ExactSearch(
      detail::SearchPosition &position, std::uint64_t node_limit,
      std::vector<detail::SearchTopology::EdgeIndex> component_edges,
      std::vector<detail::SearchTopology::VertexIndex> component_vertices)
      : position_(position), component_edges_(std::move(component_edges)),
        component_vertices_(std::move(component_vertices)),
        node_limit_(node_limit) {
    if (component_edges_.size() > 32U || component_vertices_.size() > 32U) {
      throw std::logic_error(
          "exact endgame component does not fit its collision-free key");
    }
    table_.reserve(32'768);
  }

  int run() { return search(0, -kInfinity, kInfinity); }

  std::vector<Move> proving_action(Player root_mover) {
    std::vector<Move> result;
    std::uint32_t ply = 0;
    while (!position_.is_terminal() && position_.to_move() == root_mover) {
      const ExactStateKey key = state_key();
      auto found = table_.find(key);
      if (found == table_.end() || found->second.bound != ExactBound::Exact) {
        (void)search(ply, -kInfinity, kInfinity);
        found = table_.find(key);
      }
      if (found == table_.end() || found->second.bound != ExactBound::Exact) {
        throw std::logic_error(
            "exact solver could not reconstruct an exact proving action");
      }
      std::uint8_t slot = 0;
      if (!position_.slot_for_move(found->second.best_move, slot)) {
        throw std::logic_error(
            "exact solver cached an illegal proving action edge");
      }
      result.push_back(found->second.best_move);
      position_.make_move(slot);
      ++ply;
    }
    position_.unmake_to(0);
    return result;
  }

  std::uint64_t nodes() const noexcept { return nodes_; }
  std::uint64_t cache_hits() const noexcept { return cache_hits_; }

 private:
  detail::SearchPosition &position_;
  std::vector<detail::SearchTopology::EdgeIndex> component_edges_{};
  std::vector<detail::SearchTopology::VertexIndex> component_vertices_{};
  std::unordered_map<ExactStateKey, ExactEntry, ExactStateKeyHash> table_{};
  std::uint64_t node_limit_{};
  std::uint64_t nodes_{};
  std::uint64_t cache_hits_{};

  void visit() {
    if (nodes_ >= node_limit_) {
      throw ExactBudgetReached{};
    }
    ++nodes_;
  }

  ExactStateKey state_key() const noexcept {
    ExactStateKey key;
    for (std::size_t i = 0; i < component_edges_.size(); ++i) {
      if (position_.edge_used(component_edges_[i])) {
        key.used_edges |= std::uint32_t{1} << i;
      }
    }
    for (std::size_t i = 0; i < component_vertices_.size(); ++i) {
      if (position_.vertex_visited(component_vertices_[i])) {
        key.visited_vertices |= std::uint32_t{1} << i;
      }
    }
    key.ball = position_.ball_vertex();
    key.to_move = position_.to_move();
    key.status = position_.status();
    return key;
  }

  std::vector<std::uint8_t> ordered_slots() const {
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position_.legal_slots(slots);
    std::vector<std::uint8_t> ordered(slots.begin(), slots.begin() + count);
    const Player mover = position_.to_move();
    const Point from = position_.ball();
    std::stable_sort(ordered.begin(), ordered.end(), [&](std::uint8_t left,
                                                         std::uint8_t right) {
      const int left_rank = normalized_direction_rank(
          mover, from, position_.move_for_slot(left).to);
      const int right_rank = normalized_direction_rank(
          mover, from, position_.move_for_slot(right).to);
      return left_rank < right_rank;
    });
    return ordered;
  }

  int terminal_score(std::uint32_t ply) const {
    const std::optional<Player> winning_player = position_.winner();
    if (!winning_player.has_value()) {
      throw std::logic_error("standard exact endgame has no terminal winner");
    }
    const int distance = static_cast<int>(ply);
    return *winning_player == Player::One ? kMateScore - distance
                                          : -kMateScore + distance;
  }

  int search(std::uint32_t ply, int alpha, int beta) {
    visit();
    const ExactStateKey key = state_key();
    const int original_alpha = alpha;
    const int original_beta = beta;
    if (const auto found = table_.find(key); found != table_.end()) {
      ++cache_hits_;
      const ExactEntry &entry = found->second;
      if (entry.bound == ExactBound::Exact) {
        return entry.score;
      }
      if (entry.bound == ExactBound::Lower) {
        alpha = std::max(alpha, entry.score);
      } else {
        beta = std::min(beta, entry.score);
      }
      if (alpha >= beta) {
        return entry.score;
      }
    }

    if (position_.is_terminal()) {
      return terminal_score(ply);
    }

    const Player mover = position_.to_move();
    const bool maximizing = mover == Player::One;
    int best_score = maximizing ? -kInfinity : kInfinity;
    std::optional<Move> best_move;
    for (const std::uint8_t slot : ordered_slots()) {
      const Move move = position_.move_for_slot(slot);
      int score = 0;
      {
        ScopedMove played(position_, slot);
        score = search(ply + 1U, alpha, beta);
      }
      if (!best_move.has_value() ||
          (maximizing ? score > best_score : score < best_score)) {
        best_score = score;
        best_move = move;
      }
      if (maximizing) {
        alpha = std::max(alpha, best_score);
      } else {
        beta = std::min(beta, best_score);
      }
      if (alpha >= beta) {
        break;
      }
    }
    if (!best_move.has_value()) {
      throw std::logic_error("non-terminal exact node has no legal edge");
    }

    ExactBound bound = ExactBound::Exact;
    if (best_score <= original_alpha) {
      bound = ExactBound::Upper;
    } else if (best_score >= original_beta) {
      bound = ExactBound::Lower;
    }
    table_[key] = ExactEntry{best_score, *best_move, bound};
    return best_score;
  }
};

}  // namespace

ExactEndgameResult detail::solve_exact_endgame_with_limit(
    const GameState &state, std::uint64_t node_limit) {
  if (node_limit == 0) {
    throw std::invalid_argument("exact endgame node limit must be positive");
  }
  if (state.config.width != 8 || state.config.height != 10 ||
      state.config.goal_rule != GoalRule::OpponentGoalOnly ||
      state.config.blocked_rule != BlockedRule::PlayerToMoveLoses) {
    throw std::invalid_argument(
        "ExactEndgameSolver requires the standard 8x10 demo rules");
  }
  if (is_terminal(state)) {
    throw std::invalid_argument(
        "ExactEndgameSolver requires a non-terminal possession boundary");
  }
  if (!is_root_possession_boundary(state)) {
    throw std::invalid_argument(
        "ExactEndgameSolver may run only at a possession boundary");
  }

  auto topology = std::make_shared<detail::SearchTopology>(state.config);
  detail::SearchPosition position(topology, state);
  ExactEndgameResult result;
  ReachableComponent component =
      reachable_unused_component(*topology, position);
  result.reachable_edge_count = component.edges.size();
  if (result.reachable_edge_count >
      ExactEndgameSolver::maximum_reachable_edges) {
    return result;
  }

  ExactSearch search(position, node_limit, std::move(component.edges),
                     std::move(component.vertices));
  try {
    const Player root_mover = state.to_move;
    const int score = search.run();
    result.action = search.proving_action(root_mover);
    result.nodes = search.nodes();
    result.cache_hits = search.cache_hits();
    result.winner = score > 0 ? Player::One : Player::Two;
    result.distance = static_cast<std::uint32_t>(
        kMateScore - std::abs(score));
    result.status = *result.winner == root_mover
                        ? ProofStatus::ProvenWin
                        : ProofStatus::ProvenLoss;

    // Validate the compact solver's complete action with the authoritative
    // public rules implementation before describing it as a proof.
    GameState replayed = state;
    for (const Move move : result.action) {
      replayed = apply_move(replayed, move);
    }
    if (!is_terminal(replayed) && replayed.to_move == root_mover) {
      throw std::logic_error(
          "exact solver returned an incomplete proving action");
    }
    return result;
  } catch (const ExactBudgetReached &) {
    result.status = ProofStatus::Unknown;
    result.winner.reset();
    result.distance.reset();
    result.action.clear();
    result.nodes = search.nodes();
    result.cache_hits = search.cache_hits();
    result.budget_exhausted = true;
    return result;
  }
}

ExactEndgameResult ExactEndgameSolver::solve(const GameState &state) const {
  return detail::solve_exact_endgame_with_limit(state, maximum_nodes);
}

}  // namespace papersoccer
