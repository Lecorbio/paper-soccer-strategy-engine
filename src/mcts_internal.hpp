#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

namespace papersoccer::detail {

inline constexpr std::size_t kMaximumMoves = 8;

inline constexpr bool same_rules_config(const RulesConfig &lhs,
                                        const RulesConfig &rhs) noexcept {
  return lhs.width == rhs.width && lhs.height == rhs.height;
}

struct PositionKey {
  std::uint64_t first{};
  std::uint64_t second{};

  constexpr bool operator==(const PositionKey &) const noexcept = default;
};

inline constexpr std::uint64_t mix_position_key(std::uint64_t value) noexcept {
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

inline constexpr PositionKey position_key_component(std::uint64_t category,
                                                     std::uint64_t index) noexcept {
  const std::uint64_t tagged = (category << 56U) ^ index;
  return PositionKey{
      mix_position_key(tagged ^ 0x243f6a8885a308d3ULL),
      mix_position_key(tagged ^ 0x13198a2e03707344ULL),
  };
}

inline constexpr void xor_position_key(PositionKey &target,
                                       PositionKey value) noexcept {
  target.first ^= value.first;
  target.second ^= value.second;
}

class CompactBitset {
 public:
  CompactBitset() = default;
  explicit CompactBitset(std::size_t bit_count)
      : words_((bit_count + 63U) / 64U) {}

  bool test(std::size_t index) const noexcept {
    return (words_[index / 64U] & (std::uint64_t{1} << (index % 64U))) != 0;
  }

  void set(std::size_t index) noexcept {
    words_[index / 64U] |= std::uint64_t{1} << (index % 64U);
  }

  void reset(std::size_t index) noexcept {
    words_[index / 64U] &= ~(std::uint64_t{1} << (index % 64U));
  }

  bool operator==(const CompactBitset &) const noexcept = default;

 private:
  std::vector<std::uint64_t> words_{};
};

class SearchTopology {
 public:
  using VertexIndex = std::uint32_t;
  using EdgeIndex = std::uint32_t;

  struct Arc {
    VertexIndex destination{};
    EdgeIndex edge{};
    Move move{};
  };

  struct Adjacency {
    std::array<Arc, kMaximumMoves> arcs{};
    std::uint8_t count{};
  };

  explicit SearchTopology(RulesConfig config) : config_(config) {
    if (config.width < 1 || config.height < 1) {
      throw std::invalid_argument("search requires positive board dimensions");
    }

    const std::size_t regular_vertex_count =
        static_cast<std::size_t>(config.width + 1) *
        static_cast<std::size_t>(config.height + 1);
    points_.reserve(regular_vertex_count + 6U);
    point_indices_.reserve(regular_vertex_count + 6U);

    for (int y = 1; y <= config.height + 1; ++y) {
      for (int x = 0; x <= config.width; ++x) {
        add_vertex(Point{x, y});
      }
    }

    const int center_x = config.width / 2;
    for (int x = center_x - 1; x <= center_x + 1; ++x) {
      add_vertex(Point{x, 0});
      add_vertex(Point{x, config.height + 2});
    }

    player_one_adjacency_.resize(points_.size());
    player_two_adjacency_.resize(points_.size());

    edge_indices_.reserve(regular_vertex_count * 4U);
    for (VertexIndex vertex = 0; vertex < points_.size(); ++vertex) {
      if (!is_regular_point(config_, points_[vertex])) {
        continue;
      }
      build_adjacency(vertex, Player::One, player_one_adjacency_[vertex]);
      build_adjacency(vertex, Player::Two, player_two_adjacency_[vertex]);
    }
  }

  const RulesConfig &config() const noexcept { return config_; }
  std::size_t vertex_count() const noexcept { return points_.size(); }
  std::size_t edge_count() const noexcept { return edge_indices_.size(); }

  std::optional<VertexIndex> find_vertex(Point point) const noexcept {
    const auto found = point_indices_.find(point);
    if (found == point_indices_.end()) {
      return std::nullopt;
    }
    return found->second;
  }

  std::optional<EdgeIndex> find_edge(const Segment &segment) const noexcept {
    const auto found = edge_indices_.find(segment);
    if (found == edge_indices_.end()) {
      return std::nullopt;
    }
    return found->second;
  }

  Point point(VertexIndex vertex) const noexcept { return points_[vertex]; }

  const Adjacency &adjacency(VertexIndex vertex, Player player) const noexcept {
    return player == Player::One ? player_one_adjacency_[vertex]
                                 : player_two_adjacency_[vertex];
  }

 private:
  RulesConfig config_{};
  std::vector<Point> points_{};
  std::unordered_map<Point, VertexIndex, PointHash> point_indices_{};
  std::unordered_map<Segment, EdgeIndex, SegmentHash> edge_indices_{};
  std::vector<Adjacency> player_one_adjacency_{};
  std::vector<Adjacency> player_two_adjacency_{};

  void add_vertex(Point point) {
    if (point_indices_.contains(point)) {
      return;
    }
    const auto index = static_cast<VertexIndex>(points_.size());
    points_.push_back(point);
    point_indices_.emplace(point, index);
  }

  EdgeIndex edge_index(const Segment &segment) {
    const auto existing = edge_indices_.find(segment);
    if (existing != edge_indices_.end()) {
      return existing->second;
    }
    const auto index = static_cast<EdgeIndex>(edge_indices_.size());
    edge_indices_.emplace(segment, index);
    return index;
  }

  void build_adjacency(VertexIndex source, Player player, Adjacency &result) {
    const Point from = points_[source];
    for (const Point destination : neighbors(config_, from, player)) {
      const Segment segment{from, destination};
      if (is_forbidden_boundary_segment(config_, segment)) {
        continue;
      }
      const auto destination_index = find_vertex(destination);
      if (!destination_index.has_value()) {
        throw std::logic_error("search topology is missing a legal destination");
      }
      if (result.count >= kMaximumMoves) {
        throw std::logic_error("paper soccer vertex has more than eight moves");
      }
      result.arcs[result.count++] =
          Arc{*destination_index, edge_index(segment), Move{destination}};
    }
  }
};

class SearchPosition {
 public:
  using VertexIndex = SearchTopology::VertexIndex;
  using EdgeIndex = SearchTopology::EdgeIndex;

  struct Undo {
    VertexIndex previous_ball{};
    EdgeIndex edge{};
    Player previous_player{Player::One};
    Status previous_status{Status::InProgress};
    bool destination_was_visited{};
  };

  SearchPosition(std::shared_ptr<const SearchTopology> topology,
                 const GameState &state)
      : topology_(std::move(topology)), used_edges_(topology_->edge_count()),
        visited_vertices_(topology_->vertex_count()), to_move_(state.to_move),
        status_(state.status) {
    if (!same_rules_config(topology_->config(), state.config)) {
      throw std::invalid_argument("search topology does not match game state");
    }
    const auto ball = topology_->find_vertex(state.ball);
    if (!ball.has_value()) {
      throw std::invalid_argument("search game state ball is outside the board");
    }
    ball_ = *ball;

    for (const Segment &segment : state.used_segments) {
      const auto edge = topology_->find_edge(segment);
      if (edge.has_value()) {
        used_edges_.set(*edge);
        xor_position_key(position_key_, position_key_component(1, *edge));
      }
    }
    for (const auto &[point, visits] : state.visit_count) {
      if (visits <= 0) {
        continue;
      }
      const auto vertex = topology_->find_vertex(point);
      if (vertex.has_value()) {
        visited_vertices_.set(*vertex);
        xor_position_key(position_key_, position_key_component(2, *vertex));
      }
    }
    add_dynamic_key();
    undo_stack_.reserve(topology_->edge_count());
  }

  const std::shared_ptr<const SearchTopology> &topology() const noexcept {
    return topology_;
  }
  Point ball() const noexcept { return topology_->point(ball_); }
  VertexIndex ball_vertex() const noexcept { return ball_; }
  Player to_move() const noexcept { return to_move_; }
  Status status() const noexcept { return status_; }
  bool is_terminal() const noexcept { return status_ != Status::InProgress; }
  std::size_t undo_depth() const noexcept { return undo_stack_.size(); }
  PositionKey position_key() const noexcept { return position_key_; }

  bool edge_used(EdgeIndex edge) const noexcept { return used_edges_.test(edge); }
  bool vertex_visited(VertexIndex vertex) const noexcept {
    return visited_vertices_.test(vertex);
  }

  bool grants_extra_turn(std::uint8_t slot) const noexcept {
    const SearchTopology::Arc &arc =
        topology_->adjacency(ball_, to_move_).arcs[slot];
    return is_boundary_point(topology_->config(),
                             topology_->point(arc.destination)) ||
           visited_vertices_.test(arc.destination);
  }

  std::optional<Player> winner() const noexcept {
    if (status_ == Status::WonByOne) {
      return Player::One;
    }
    if (status_ == Status::WonByTwo) {
      return Player::Two;
    }
    return std::nullopt;
  }

  std::uint8_t legal_slots(
      std::array<std::uint8_t, kMaximumMoves> &slots) const noexcept {
    if (is_terminal()) {
      return 0;
    }
    const SearchTopology::Adjacency &adjacency =
        topology_->adjacency(ball_, to_move_);
    std::uint8_t count = 0;
    for (std::uint8_t slot = 0; slot < adjacency.count; ++slot) {
      if (!used_edges_.test(adjacency.arcs[slot].edge)) {
        slots[count++] = slot;
      }
    }
    return count;
  }

  Move move_for_slot(std::uint8_t slot) const noexcept {
    return topology_->adjacency(ball_, to_move_).arcs[slot].move;
  }

  bool slot_for_move(Move move, std::uint8_t &result) const noexcept {
    const SearchTopology::Adjacency &adjacency =
        topology_->adjacency(ball_, to_move_);
    for (std::uint8_t slot = 0; slot < adjacency.count; ++slot) {
      if (adjacency.arcs[slot].move == move &&
          !used_edges_.test(adjacency.arcs[slot].edge)) {
        result = slot;
        return true;
      }
    }
    return false;
  }

  void make_move(std::uint8_t slot) {
    if (is_terminal()) {
      throw std::invalid_argument("cannot move a terminal search position");
    }
    const SearchTopology::Adjacency &adjacency =
        topology_->adjacency(ball_, to_move_);
    if (slot >= adjacency.count || used_edges_.test(adjacency.arcs[slot].edge)) {
      throw std::invalid_argument("illegal compact search move");
    }

    const SearchTopology::Arc arc = adjacency.arcs[slot];
    const bool destination_was_visited = visited_vertices_.test(arc.destination);
    undo_stack_.push_back(
        Undo{ball_, arc.edge, to_move_, status_, destination_was_visited});
    remove_dynamic_key();
    used_edges_.set(arc.edge);
    xor_position_key(position_key_, position_key_component(1, arc.edge));
    visited_vertices_.set(arc.destination);
    if (!destination_was_visited) {
      xor_position_key(position_key_,
                       position_key_component(2, arc.destination));
    }
    ball_ = arc.destination;

    const Point destination = topology_->point(ball_);
    if (is_goal_point(topology_->config(), destination)) {
      status_ = to_move_ == Player::One ? Status::WonByOne : Status::WonByTwo;
      add_dynamic_key();
      return;
    }

    const bool extra_turn =
        is_boundary_point(topology_->config(), destination) ||
        destination_was_visited;
    if (!extra_turn) {
      to_move_ = opponent(to_move_);
    }
    status_ = Status::InProgress;

    std::array<std::uint8_t, kMaximumMoves> slots{};
    if (legal_slots(slots) == 0) {
      status_ = to_move_ == Player::One ? Status::WonByTwo : Status::WonByOne;
    }
    add_dynamic_key();
  }

  void unmake_move() {
    if (undo_stack_.empty()) {
      throw std::logic_error("cannot unmake the compact search root");
    }
    const Undo undo = undo_stack_.back();
    undo_stack_.pop_back();

    const VertexIndex destination = ball_;
    remove_dynamic_key();
    ball_ = undo.previous_ball;
    to_move_ = undo.previous_player;
    status_ = undo.previous_status;
    used_edges_.reset(undo.edge);
    xor_position_key(position_key_, position_key_component(1, undo.edge));
    if (!undo.destination_was_visited) {
      visited_vertices_.reset(destination);
      xor_position_key(position_key_, position_key_component(2, destination));
    }
    add_dynamic_key();
  }

  void unmake_to(std::size_t depth) {
    while (undo_stack_.size() > depth) {
      unmake_move();
    }
  }

  bool same_compact_state(const SearchPosition &other) const noexcept {
    return same_rules_config(topology_->config(), other.topology_->config()) &&
           ball_ == other.ball_ && to_move_ == other.to_move_ &&
           status_ == other.status_ && used_edges_ == other.used_edges_ &&
           visited_vertices_ == other.visited_vertices_;
  }

 private:
  std::shared_ptr<const SearchTopology> topology_{};
  CompactBitset used_edges_{};
  CompactBitset visited_vertices_{};
  VertexIndex ball_{};
  Player to_move_{Player::One};
  Status status_{Status::InProgress};
  std::vector<Undo> undo_stack_{};
  PositionKey position_key_{};

  void add_dynamic_key() noexcept {
    xor_position_key(position_key_, position_key_component(3, ball_));
    xor_position_key(position_key_,
                     position_key_component(4, static_cast<std::uint64_t>(to_move_)));
    xor_position_key(position_key_,
                     position_key_component(5, static_cast<std::uint64_t>(status_)));
  }

  void remove_dynamic_key() noexcept { add_dynamic_key(); }
};

struct TacticalProbeStats {
  std::uint64_t nodes{};
  std::uint32_t max_depth{};
  bool depth_cutoff{};
  bool node_cutoff{};
};

struct TacticalProbeResult {
  std::optional<Player> proven_winner{};
  std::optional<Move> proving_move{};
  TacticalProbeStats stats{};
};

TacticalProbeResult run_tactical_probe(SearchPosition &position,
                                        std::uint32_t max_depth,
                                        std::uint32_t max_nodes);

}  // namespace papersoccer::detail
