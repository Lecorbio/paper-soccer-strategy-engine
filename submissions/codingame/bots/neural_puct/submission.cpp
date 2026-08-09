
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC optimize("O3")
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace papersoccer {

enum class Player { One, Two };

enum class Status { InProgress, WonByOne, WonByTwo };

enum class GoalRule { OpponentGoalOnly, OwnGoalsAllowed };

enum class BlockedRule { PlayerToMoveLoses, MoverLoses };

inline constexpr Player opponent(Player player) noexcept {
  return player == Player::One ? Player::Two : Player::One;
}

struct Point {
  int x{0};
  int y{0};

  constexpr bool operator==(const Point &) const noexcept = default;
};

inline constexpr bool operator<(const Point &lhs, const Point &rhs) noexcept {
  return lhs.y < rhs.y || (lhs.y == rhs.y && lhs.x < rhs.x);
}

struct PointHash {
  std::size_t operator()(const Point &point) const noexcept {
    const auto x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(point.x));
    const auto y = static_cast<std::uint64_t>(static_cast<std::uint32_t>(point.y));
    return static_cast<std::size_t>((x << 32U) ^ y);
  }
};

struct Segment {
  Point a{};
  Point b{};

  Segment() = default;

  Segment(Point first, Point second) noexcept {
    if (second < first) {
      std::swap(first, second);
    }
    a = first;
    b = second;
  }

  constexpr bool operator==(const Segment &) const noexcept = default;
};

struct SegmentHash {
  std::size_t operator()(const Segment &segment) const noexcept {
    PointHash hash_point;
    const auto first = hash_point(segment.a);
    const auto second = hash_point(segment.b);
    return first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6U) + (first >> 2U));
  }
};

struct Move {
  Point to{};
  constexpr bool operator==(const Move &) const noexcept = default;
};

struct RulesConfig {
  int width{8};
  int height{10};
  GoalRule goal_rule{GoalRule::OpponentGoalOnly};
  BlockedRule blocked_rule{BlockedRule::PlayerToMoveLoses};
};

struct GameState {
  RulesConfig config{};
  Point ball{};
  Player to_move{Player::One};
  Status status{Status::InProgress};
  std::vector<Point> path{};
  std::unordered_set<Segment, SegmentHash> used_segments{};
  std::unordered_map<Point, int, PointHash> visit_count{};
};

}  // namespace papersoccer

namespace papersoccer {

bool is_regular_point(const RulesConfig &config, Point point);

bool is_goal_point(const RulesConfig &config, Point point);

bool is_attacking_goal(const RulesConfig &config, Point point, Player player);

bool is_boundary_point(const RulesConfig &config, Point point);

bool is_neighbor(Point from, Point to);

bool is_forbidden_boundary_segment(const RulesConfig &config, Segment segment);

std::vector<Point> neighbors(const RulesConfig &config, Point from, Player player);

}  // namespace papersoccer

namespace papersoccer {

GameState make_initial_state(const RulesConfig &config = {});

std::vector<Move> legal_moves(const GameState &state);

bool grants_extra_turn(const GameState &before, Point destination);

GameState apply_move(const GameState &state, Move move);

bool is_terminal(const GameState &state);

std::optional<Player> winner(const GameState &state);

}  // namespace papersoccer

namespace papersoccer::detail {

inline constexpr std::size_t kMaximumMoves = 8;

inline constexpr bool same_rules_config(const RulesConfig &lhs,
                                        const RulesConfig &rhs) noexcept {
  return lhs.width == rhs.width && lhs.height == rhs.height &&
         lhs.goal_rule == rhs.goal_rule &&
         lhs.blocked_rule == rhs.blocked_rule;
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
    const Player mover = to_move_;
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
      status_ =
          is_attacking_goal(topology_->config(), destination, Player::One)
              ? Status::WonByOne
              : Status::WonByTwo;
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
      const Player blocked_player =
          topology_->config().blocked_rule == BlockedRule::MoverLoses
              ? mover
              : to_move_;
      status_ = blocked_player == Player::One ? Status::WonByTwo
                                              : Status::WonByOne;
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

namespace papersoccer {

namespace {

constexpr int kNorthGoalY = 0;
constexpr int kFieldTopY = 1;

int field_bottom_y(const RulesConfig &config) { return config.height + 1; }

int south_goal_y(const RulesConfig &config) { return config.height + 2; }

int center_x(const RulesConfig &config) { return config.width / 2; }

int mouth_left_x(const RulesConfig &config) { return center_x(config) - 1; }

int mouth_right_x(const RulesConfig &config) { return center_x(config) + 1; }

bool is_mouth_x(const RulesConfig &config, int x) {
  return x >= mouth_left_x(config) && x <= mouth_right_x(config);
}

bool is_north_goal(const RulesConfig &config, Point point) {
  return is_mouth_x(config, point.x) && point.y == kNorthGoalY;
}

bool is_south_goal(const RulesConfig &config, Point point) {
  return is_mouth_x(config, point.x) && point.y == south_goal_y(config);
}

bool is_goal_mouth_point(const RulesConfig &config, Point point) {
  return is_mouth_x(config, point.x) &&
         (point.y == kFieldTopY || point.y == field_bottom_y(config));
}

bool is_north_goal_post_segment(const RulesConfig &config, Segment segment) {
  const bool touches_north_goal = is_north_goal(config, segment.a) || is_north_goal(config, segment.b);
  if (!touches_north_goal) {
    return false;
  }

  return segment.a.x == segment.b.x &&
         (segment.a.x == mouth_left_x(config) || segment.a.x == mouth_right_x(config)) &&
         ((segment.a.y == kNorthGoalY && segment.b.y == kFieldTopY) ||
          (segment.a.y == kFieldTopY && segment.b.y == kNorthGoalY));
}

bool is_south_goal_post_segment(const RulesConfig &config, Segment segment) {
  const bool touches_south_goal = is_south_goal(config, segment.a) || is_south_goal(config, segment.b);
  if (!touches_south_goal) {
    return false;
  }

  return segment.a.x == segment.b.x &&
         (segment.a.x == mouth_left_x(config) || segment.a.x == mouth_right_x(config)) &&
         ((segment.a.y == field_bottom_y(config) && segment.b.y == south_goal_y(config)) ||
          (segment.a.y == south_goal_y(config) && segment.b.y == field_bottom_y(config)));
}

}  // namespace

bool is_regular_point(const RulesConfig &config, Point point) {
  return point.x >= 0 && point.x <= config.width && point.y >= kFieldTopY &&
         point.y <= field_bottom_y(config);
}

bool is_goal_point(const RulesConfig &config, Point point) {
  return is_north_goal(config, point) || is_south_goal(config, point);
}

bool is_attacking_goal(const RulesConfig &config, Point point, Player player) {
  return player == Player::One ? is_north_goal(config, point)
                               : is_south_goal(config, point);
}

bool is_boundary_point(const RulesConfig &config, Point point) {
  if (!is_regular_point(config, point)) {
    return false;
  }

  if (point.x == 0 || point.x == config.width) {
    return true;
  }

  const bool on_goal_line =
      point.y == kFieldTopY || point.y == field_bottom_y(config);
  const bool inside_goal_opening =
      point.x > mouth_left_x(config) && point.x < mouth_right_x(config);
  return on_goal_line && !inside_goal_opening;
}

bool is_neighbor(Point from, Point to) {
  const int dx = std::abs(from.x - to.x);
  const int dy = std::abs(from.y - to.y);
  return dx <= 1 && dy <= 1 && (dx + dy) > 0;
}

bool is_forbidden_boundary_segment(const RulesConfig &config, Segment segment) {
  if (is_north_goal_post_segment(config, segment) || is_south_goal_post_segment(config, segment)) {
    return true;
  }

  if (!is_regular_point(config, segment.a) || !is_regular_point(config, segment.b)) {
    return false;
  }

  if (!(is_boundary_point(config, segment.a) && is_boundary_point(config, segment.b))) {
    return false;
  }

  const int dx = std::abs(segment.a.x - segment.b.x);
  const int dy = std::abs(segment.a.y - segment.b.y);

  if (segment.a.y == segment.b.y &&
      (segment.a.y == kFieldTopY || segment.a.y == field_bottom_y(config)) &&
      dx == 1 && dy == 0) {
    return true;
  }

  if (segment.a.x == segment.b.x && (segment.a.x == 0 || segment.a.x == config.width) &&
      dx == 0 && dy == 1) {
    return true;
  }

  return false;
}

std::vector<Point> neighbors(const RulesConfig &config, Point from, Player player) {
  std::vector<Point> result;
  if (!is_regular_point(config, from)) {
    return result;
  }

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const Point candidate{from.x + dx, from.y + dy};
      if (is_regular_point(config, candidate)) {
        result.push_back(candidate);
      }
    }
  }

  if (is_goal_mouth_point(config, from)) {
    const bool own_goals_allowed =
        config.goal_rule == GoalRule::OwnGoalsAllowed;
    if ((own_goals_allowed || player == Player::One) &&
        from.y == kFieldTopY) {
      for (int goal_x = mouth_left_x(config); goal_x <= mouth_right_x(config); ++goal_x) {
        const Point goal_point{goal_x, kNorthGoalY};
        if (is_neighbor(from, goal_point)) {
          result.push_back(goal_point);
        }
      }
    }
    if ((own_goals_allowed || player == Player::Two) &&
        from.y == field_bottom_y(config)) {
      for (int goal_x = mouth_left_x(config); goal_x <= mouth_right_x(config); ++goal_x) {
        const Point goal_point{goal_x, south_goal_y(config)};
        if (is_neighbor(from, goal_point)) {
          result.push_back(goal_point);
        }
      }
    }
  }

  return result;
}

}  // namespace papersoccer

namespace papersoccer {

GameState make_initial_state(const RulesConfig &config) {
  GameState state;
  state.config = config;
  state.ball = Point{config.width / 2, config.height / 2 + 1};
  state.to_move = Player::One;
  state.status = Status::InProgress;
  state.path = {state.ball};
  state.visit_count[state.ball] = 1;
  return state;
}

std::vector<Move> legal_moves(const GameState &state) {
  std::vector<Move> result;
  if (state.status != Status::InProgress) {
    return result;
  }

  for (const Point destination : neighbors(state.config, state.ball, state.to_move)) {
    const Segment segment{state.ball, destination};
    if (state.used_segments.find(segment) != state.used_segments.end()) {
      continue;
    }
    if (is_forbidden_boundary_segment(state.config, segment)) {
      continue;
    }
    result.push_back(Move{destination});
  }

  return result;
}

bool grants_extra_turn(const GameState &before, Point destination) {
  if (is_boundary_point(before.config, destination)) {
    return true;
  }
  const auto it = before.visit_count.find(destination);
  return it != before.visit_count.end() && it->second > 0;
}

GameState apply_move(const GameState &state, Move move) {
  if (state.status != Status::InProgress) {
    throw std::invalid_argument("cannot apply move to terminal game state");
  }

  const auto moves = legal_moves(state);
  const auto legal_it = std::find_if(
      moves.begin(), moves.end(),
      [&](const Move &candidate) { return candidate.to == move.to; });
  if (legal_it == moves.end()) {
    throw std::invalid_argument("illegal move");
  }

  GameState next = state;
  next.used_segments.insert(Segment{state.ball, move.to});
  next.ball = move.to;
  next.path.push_back(move.to);
  next.visit_count[move.to] += 1;

  if (is_goal_point(next.config, move.to)) {
    next.status = is_attacking_goal(next.config, move.to, Player::One)
                      ? Status::WonByOne
                      : Status::WonByTwo;
    return next;
  }

  const bool extra_turn = grants_extra_turn(state, move.to);
  next.to_move = extra_turn ? state.to_move : opponent(state.to_move);
  next.status = Status::InProgress;

  if (legal_moves(next).empty()) {
    const Player blocked_player =
        next.config.blocked_rule == BlockedRule::MoverLoses ? state.to_move
                                                            : next.to_move;
    next.status = blocked_player == Player::One ? Status::WonByTwo
                                                : Status::WonByOne;
  }

  return next;
}

bool is_terminal(const GameState &state) {
  return state.status != Status::InProgress;
}

std::optional<Player> winner(const GameState &state) {
  if (state.status == Status::WonByOne) {
    return Player::One;
  }
  if (state.status == Status::WonByTwo) {
    return Player::Two;
  }
  return std::nullopt;
}

}  // namespace papersoccer

namespace papersoccer::neural_puct::model_data {

inline constexpr std::string_view kFeatureSchema = "neural-puct-features-v1";
inline constexpr std::string_view kModelKind = "expert-imitation-policy-value-v1";
inline constexpr std::string_view kArtifactSha256 = "d7e5255fd2c3f7a203796829c21c8a7580b384de13942e466f4069081e2126b2";
inline constexpr std::size_t kInputCount = 1286;
inline constexpr std::size_t kEdgeCount = 316;
inline constexpr std::size_t kVertexCount = 105;
inline constexpr std::size_t kDistanceBuckets = 8;
inline constexpr std::size_t kGlobalCount = 25;
inline constexpr std::size_t kHiddenOne = 32;
inline constexpr std::size_t kHiddenTwo = 32;
inline constexpr std::size_t kPolicyCount = 8;
inline constexpr std::size_t kUnpackedWeightCount = 42464;
inline constexpr std::string_view kGoldenInput = "stride17-mod11-v1";
inline constexpr float kGoldenValue = 0.168110818F;
inline constexpr std::array<float, 8> kGoldenPolicy{{0.272144735F, 0.336739600F, -0.0838090852F, -0.297033519F, -0.173216045F, 0.120557614F, 0.0667520836F, 0.322451919F}};
inline constexpr std::array<float, 32> kW1Scales{{0.0174012762F, 0.0205736272F, 0.0194652863F, 0.0208285600F, 0.0214778781F, 0.0199870709F, 0.0196593963F, 0.0184845347F, 0.0200953092F, 0.0212085843F, 0.0263133477F, 0.0186669696F, 0.0214237925F, 0.0271626431F, 0.0193458106F, 0.0191733669F, 0.0186756756F, 0.0204434805F, 0.0259615052F, 0.0271740686F, 0.0177942012F, 0.0262929741F, 0.0228270944F, 0.0222062953F, 0.0220758207F, 0.0250374135F, 0.0240648203F, 0.0206829011F, 0.0230245292F, 0.0198020376F, 0.0204994138F, 0.0306902695F}};
inline constexpr std::array<float, 32> kW2Scales{{0.0799989179F, 0.0641811863F, 0.107928805F, 0.0769911855F, 0.0798237845F, 0.0970907807F, 0.0798741430F, 0.0864417478F, 0.0755815282F, 0.0807577521F, 0.0850005224F, 0.0959892347F, 0.0711791366F, 0.0874470919F, 0.0923933983F, 0.0657940581F, 0.114882268F, 0.0756988600F, 0.0886399224F, 0.0828943849F, 0.0772827119F, 0.0701185241F, 0.0866855308F, 0.105609663F, 0.0857571363F, 0.0783833861F, 0.0788104981F, 0.0797959343F, 0.0739284232F, 0.0824111849F, 0.0685485080F, 0.0962149948F}};
inline constexpr std::array<float, 1> kValueScales{{0.112080470F}};
inline constexpr std::array<float, 8> kPolicyScales{{0.0588041507F, 0.0496623404F, 0.0540180765F, 0.0475804396F, 0.0730587840F, 0.0583454296F, 0.0522861592F, 0.0493009984F}};
inline constexpr std::array<float, 32> kB1{{-0.0132744843F, 0.00565505307F, -0.0110954782F, -0.00248580473F, -0.00132699171F, 0.0101020746F, -0.0144269830F, -0.0118989404F, -0.0199261233F, -0.00648655649F, -0.0122459037F, -0.00935794879F, -0.00334828463F, 0.00333867222F, -0.0178488269F, -0.0156151792F, -0.0116111273F, -0.00916101597F, 0.00160551083F, 0.0118446006F, -0.0157308355F, -0.00263921358F, 0.00809761416F, -0.00707739638F, 0.00370345055F, 0.00403345935F, -0.00599675998F, -0.00229042931F, 0.00139894441F, -0.00929133222F, 0.00362811051F, 0.00103854737F}};
inline constexpr std::array<float, 32> kB2{{0.00186633924F, 0.0145721650F, 0.0135970190F, -0.00985762663F, 0.0118807144F, 0.00704419753F, -0.00324765360F, -0.00579971960F, 0.0178576782F, -0.0114460262F, -0.00900825392F, -0.00249178638F, -0.0152213918F, -0.0101867272F, -0.0150221558F, -0.0126820672F, 0.0128336735F, 0.0169966277F, -0.0107833557F, -0.0292310249F, -0.0127271535F, -0.0238281712F, 0.00522788987F, -0.0198793653F, -0.0196920205F, 0.00293123489F, 0.0152919572F, 0.0429280810F, -0.0281460546F, -0.0399783403F, 0.0231802631F, 0.0217819475F}};
inline constexpr float kValueBias = 0.00337133743F;
inline constexpr std::array<float, 8> kPolicyBias{{0.0200416669F, 0.00958309509F, -0.0138616078F, 0.0124997096F, 0.00167424197F, -0.0146839414F, -0.0458144657F, 0.0302213002F}};
inline constexpr std::string_view kEncodedPackedWeights = "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";

}  // namespace papersoccer::neural_puct::model_data

namespace papersoccer::neural_puct {

using SearchClock = std::chrono::steady_clock;

inline constexpr std::uint32_t kFirstSearchTimeMs = 650;
inline constexpr std::uint32_t kLaterSearchTimeMs = 130;
inline constexpr std::size_t kMaximumNodes = 100'000;
inline constexpr float kDefaultCpuct = 0.45F;

inline constexpr std::size_t kUsedEdgeOffset = 0;
inline constexpr std::size_t kDistanceOffset = 316;
inline constexpr std::size_t kDegreeOffset = 1156;
inline constexpr std::size_t kGlobalOffset = 1261;
inline constexpr std::size_t kFeatureCount = 1286;

static_assert(kDistanceOffset == model_data::kEdgeCount);
static_assert(kDegreeOffset ==
              kDistanceOffset + model_data::kVertexCount *
                                    model_data::kDistanceBuckets);
static_assert(kGlobalOffset == kDegreeOffset + model_data::kVertexCount);
static_assert(kFeatureCount == kGlobalOffset + model_data::kGlobalCount);
static_assert(kFeatureCount == model_data::kInputCount);

inline constexpr std::array<Point, 8> kDirectionDeltas{{
    {0, -1},
    {1, -1},
    {1, 0},
    {1, 1},
    {0, 1},
    {-1, 1},
    {-1, 0},
    {-1, -1},
}};

struct ModelOutput {
  float value{};
  std::array<float, model_data::kPolicyCount> policy_logits{};
};

class QuantizedModel {
 public:
  ModelOutput evaluate(
      const std::array<float, model_data::kInputCount> &features,
      const std::uint16_t *active_inputs = nullptr,
      std::size_t input_count = model_data::kInputCount) const {
    const std::vector<std::int8_t> &weights = decoded_weights();
    std::array<float, model_data::kHiddenOne> hidden_one = model_data::kB1;
    for (std::size_t entry = 0; entry < input_count; ++entry) {
      const std::size_t input =
          active_inputs == nullptr ? entry : active_inputs[entry];
      const float feature = features[input];
      if (feature == 0.0F) {
        continue;
      }
      const std::size_t offset = input * model_data::kHiddenOne;
      for (std::size_t output = 0; output < model_data::kHiddenOne;
           ++output) {
        hidden_one[output] +=
            feature * static_cast<float>(weights[offset + output]) *
            model_data::kW1Scales[output];
      }
    }
    for (float &value : hidden_one) {
      value = std::max(value, 0.0F);
    }

    constexpr std::size_t kSecondOffset =
        model_data::kInputCount * model_data::kHiddenOne;
    std::array<float, model_data::kHiddenTwo> hidden_two = model_data::kB2;
    for (std::size_t input = 0; input < model_data::kHiddenOne; ++input) {
      const std::size_t offset =
          kSecondOffset + input * model_data::kHiddenTwo;
      for (std::size_t output = 0; output < model_data::kHiddenTwo;
           ++output) {
        hidden_two[output] +=
            hidden_one[input] * static_cast<float>(weights[offset + output]) *
            model_data::kW2Scales[output];
      }
    }
    for (float &value : hidden_two) {
      value = std::max(value, 0.0F);
    }

    constexpr std::size_t kValueOffset =
        kSecondOffset + model_data::kHiddenOne * model_data::kHiddenTwo;
    constexpr std::size_t kPolicyOffset =
        kValueOffset + model_data::kHiddenTwo;
    float value_logit = model_data::kValueBias;
    std::array<float, model_data::kPolicyCount> policy_logits =
        model_data::kPolicyBias;
    for (std::size_t hidden = 0; hidden < model_data::kHiddenTwo; ++hidden) {
      value_logit += hidden_two[hidden] *
                     static_cast<float>(weights[kValueOffset + hidden]) *
                     model_data::kValueScales[0];
      for (std::size_t policy = 0; policy < model_data::kPolicyCount;
           ++policy) {
        policy_logits[policy] +=
            hidden_two[hidden] *
            static_cast<float>(
                weights[kPolicyOffset + hidden * model_data::kPolicyCount +
                        policy]) *
            model_data::kPolicyScales[policy];
      }
    }
    return ModelOutput{std::tanh(0.5F * value_logit), policy_logits};
  }

  static const std::vector<std::int8_t> &decoded_weights() {
    static const std::vector<std::int8_t> result = [] {
      std::vector<std::uint8_t> packed;
      packed.reserve((model_data::kUnpackedWeightCount + 1U) / 2U);
      std::uint32_t buffer = 0;
      int bits = 0;
      for (const char character : model_data::kEncodedPackedWeights) {
        if (character == '=') {
          break;
        }
        const int value = base64_value(character);
        if (value < 0) {
          throw std::logic_error("invalid neural PUCT model encoding");
        }
        buffer = (buffer << 6U) | static_cast<std::uint32_t>(value);
        bits += 6;
        if (bits >= 8) {
          bits -= 8;
          packed.push_back(static_cast<std::uint8_t>(buffer >> bits));
        }
      }
      if (packed.size() != (model_data::kUnpackedWeightCount + 1U) / 2U) {
        throw std::logic_error("invalid neural PUCT packed model size");
      }
      std::vector<std::int8_t> unpacked;
      unpacked.reserve(model_data::kUnpackedWeightCount);
      for (const std::uint8_t byte : packed) {
        for (const std::uint8_t nibble :
             {static_cast<std::uint8_t>(byte & 15U),
              static_cast<std::uint8_t>(byte >> 4U)}) {
          if (unpacked.size() == model_data::kUnpackedWeightCount) {
            break;
          }
          if (nibble == 8U) {
            throw std::logic_error("model contains asymmetric int4 value -8");
          }
          unpacked.push_back(static_cast<std::int8_t>(
              nibble >= 8U ? static_cast<int>(nibble) - 16
                           : static_cast<int>(nibble)));
        }
      }
      return unpacked;
    }();
    return result;
  }

 private:
  static int base64_value(char character) noexcept {
    if (character >= 'A' && character <= 'Z') {
      return character - 'A';
    }
    if (character >= 'a' && character <= 'z') {
      return character - 'a' + 26;
    }
    if (character >= '0' && character <= '9') {
      return character - '0' + 52;
    }
    if (character == '+') {
      return 62;
    }
    if (character == '/') {
      return 63;
    }
    return -1;
  }
};

class FeatureEncoder {
 public:
  explicit FeatureEncoder(std::shared_ptr<const detail::SearchTopology> topology)
      : topology_(std::move(topology)),
        cooperative_adjacency_(topology_->vertex_count()) {
    if (topology_->config().width != 8 || topology_->config().height != 10 ||
        topology_->edge_count() != model_data::kEdgeCount ||
        topology_->vertex_count() != model_data::kVertexCount) {
      throw std::invalid_argument(
          "neural PUCT requires the standard 8x10 topology");
    }
    initialize_topology_maps();
  }

  const std::array<float, kFeatureCount> &encode(
      const detail::SearchPosition &position) {
    if (!detail::same_rules_config(position.topology()->config(),
                                   topology_->config())) {
      throw std::invalid_argument("feature position has a different topology");
    }
    for (std::size_t index = 0; index < active_count_; ++index) {
      features_[active_indices_[index]] = 0.0F;
    }
    active_count_ = 0;
    calculate_turn_distances(position);
    const Player mover = position.to_move();

    std::size_t used_count = 0;
    std::array<std::size_t, 12> cut_used{};
    for (std::size_t canonical = 0; canonical < model_data::kEdgeCount;
         ++canonical) {
      const auto physical = physical_edge(canonical, mover);
      if (position.edge_used(physical)) {
        const std::size_t feature = kUsedEdgeOffset + canonical;
        features_[feature] = 1.0F;
        active_indices_[active_count_++] = static_cast<std::uint16_t>(feature);
        ++used_count;
        const std::int8_t cut = edge_cuts_[canonical];
        if (cut >= 0) {
          ++cut_used[static_cast<std::size_t>(cut)];
        }
      }
    }

    std::size_t visited_count = 0;
    for (std::size_t canonical = 0; canonical < model_data::kVertexCount;
         ++canonical) {
      const auto physical = physical_vertex(canonical, mover);
      visited_count += position.vertex_visited(physical) ? 1U : 0U;
      const int distance = std::clamp(distances_[physical], 0, 7);
      const std::size_t distance_feature =
          kDistanceOffset + canonical * model_data::kDistanceBuckets +
          static_cast<std::size_t>(distance);
      features_[distance_feature] = 1.0F;
      active_indices_[active_count_++] =
          static_cast<std::uint16_t>(distance_feature);

      std::size_t free_degree = 0;
      const auto &adjacency = cooperative_adjacency_[physical];
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        free_degree += position.edge_used(adjacency.arcs[index].edge) ? 0U : 1U;
      }
      features_[kDegreeOffset + canonical] =
          static_cast<float>(free_degree) / 8.0F;
    }
    for (std::size_t canonical = 0; canonical < model_data::kVertexCount;
         ++canonical) {
      const std::size_t feature = kDegreeOffset + canonical;
      if (features_[feature] != 0.0F) {
        active_indices_[active_count_++] = static_cast<std::uint16_t>(feature);
      }
    }

    const ComponentFeatures component = component_features(position);
    std::array<std::uint8_t, detail::kMaximumMoves> legal{};
    const std::uint8_t ball_degree = position.legal_slots(legal);
    const Point canonical_ball =
        mover == Player::One ? position.ball() : rotate_point(position.ball());
    const std::size_t global = kGlobalOffset;
    features_[global + 0] = static_cast<float>(canonical_ball.x) / 8.0F;
    features_[global + 1] = static_cast<float>(canonical_ball.y) / 12.0F;
    features_[global + 2] = static_cast<float>(used_count) / 316.0F;
    features_[global + 3] = static_cast<float>(visited_count) / 105.0F;
    features_[global + 4] = static_cast<float>(ball_degree) / 8.0F;
    features_[global + 5] =
        static_cast<float>(minimum_goal_distance(mover, true)) / 7.0F;
    features_[global + 6] =
        static_cast<float>(minimum_goal_distance(mover, false)) / 7.0F;
    features_[global + 7] =
        static_cast<float>(component.vertices) / 105.0F;
    features_[global + 8] =
        static_cast<float>(std::min<std::size_t>(component.safe_frontiers, 64)) /
        64.0F;
    features_[global + 9] =
        static_cast<float>(std::min<std::size_t>(component.dead_frontiers, 64)) /
        64.0F;
    features_[global + 10] =
        static_cast<float>(component.internal_edges) / 316.0F;
    features_[global + 11] = component.touches_attacking_goal ? 1.0F : 0.0F;
    features_[global + 12] = component.touches_own_goal ? 1.0F : 0.0F;

    for (std::size_t cut = 0; cut < 12; ++cut) {
      features_[global + 13 + cut] =
          cut_totals_[cut] == 0
              ? 0.0F
              : static_cast<float>(cut_used[cut]) /
                    static_cast<float>(cut_totals_[cut]);
    }
    for (std::size_t feature = global; feature < kFeatureCount; ++feature) {
      if (features_[feature] != 0.0F) {
        active_indices_[active_count_++] = static_cast<std::uint16_t>(feature);
      }
    }
    return features_;
  }

  const std::array<std::uint16_t, kFeatureCount> &active_indices() const {
    return active_indices_;
  }

  std::size_t active_count() const noexcept { return active_count_; }

  const std::array<detail::SearchTopology::VertexIndex,
                   model_data::kVertexCount> &
  rotated_vertices() const noexcept {
    return rotated_vertices_;
  }

  const std::array<detail::SearchTopology::EdgeIndex,
                   model_data::kEdgeCount> &
  rotated_edges() const noexcept {
    return rotated_edges_;
  }

 private:
  struct ComponentFeatures {
    std::size_t vertices{};
    std::size_t safe_frontiers{};
    std::size_t dead_frontiers{};
    std::size_t internal_edges{};
    bool touches_attacking_goal{};
    bool touches_own_goal{};
  };

  std::shared_ptr<const detail::SearchTopology> topology_;
  std::vector<detail::SearchTopology::Adjacency> cooperative_adjacency_;
  std::array<detail::SearchTopology::VertexIndex, model_data::kVertexCount>
      rotated_vertices_{};
  std::array<detail::SearchTopology::EdgeIndex, model_data::kEdgeCount>
      rotated_edges_{};
  std::array<Segment, model_data::kEdgeCount> edge_segments_{};
  std::array<std::int8_t, model_data::kEdgeCount> edge_cuts_{};
  std::array<std::array<detail::SearchTopology::VertexIndex, 3>, 2>
      goal_vertices_{};
  std::array<bool, model_data::kEdgeCount> observed_edges_{};
  std::array<int, model_data::kVertexCount> distances_{};
  std::deque<detail::SearchTopology::VertexIndex> distance_queue_;
  std::array<std::size_t, 12> cut_totals_{};
  std::array<float, kFeatureCount> features_{};
  std::array<std::uint16_t, kFeatureCount> active_indices_{};
  std::size_t active_count_{};

  Point rotate_point(Point point) const noexcept {
    return {topology_->config().width - point.x,
            topology_->config().height + 2 - point.y};
  }

  static bool contains_arc(const detail::SearchTopology::Adjacency &adjacency,
                           const detail::SearchTopology::Arc &candidate) {
    for (std::uint8_t index = 0; index < adjacency.count; ++index) {
      if (adjacency.arcs[index].edge == candidate.edge &&
          adjacency.arcs[index].destination == candidate.destination) {
        return true;
      }
    }
    return false;
  }

  void initialize_topology_maps() {
    using Vertex = detail::SearchTopology::VertexIndex;
    edge_cuts_.fill(-1);
    for (std::size_t source = 0; source < model_data::kVertexCount; ++source) {
      for (const Player player : {Player::One, Player::Two}) {
        const auto &adjacency = topology_->adjacency(
            static_cast<Vertex>(source), player);
        for (std::uint8_t index = 0; index < adjacency.count; ++index) {
          const auto &arc = adjacency.arcs[index];
          auto &cooperative = cooperative_adjacency_[source];
          if (!contains_arc(cooperative, arc)) {
            if (cooperative.count >= detail::kMaximumMoves) {
              throw std::logic_error("cooperative vertex exceeds degree eight");
            }
            cooperative.arcs[cooperative.count++] = arc;
          }
          if (!observed_edges_[arc.edge]) {
            edge_segments_[arc.edge] =
                Segment{topology_->point(static_cast<Vertex>(source)),
                        topology_->point(arc.destination)};
            observed_edges_[arc.edge] = true;
          }
        }
      }
    }
    if (std::find(observed_edges_.begin(), observed_edges_.end(), false) !=
        observed_edges_.end()) {
      throw std::logic_error("neural feature map missed an edge");
    }

    for (std::size_t vertex = 0; vertex < model_data::kVertexCount; ++vertex) {
      const auto rotated = topology_->find_vertex(
          rotate_point(topology_->point(static_cast<Vertex>(vertex))));
      if (!rotated.has_value()) {
        throw std::logic_error("neural topology is not rotationally symmetric");
      }
      rotated_vertices_[vertex] = *rotated;
    }
    for (std::size_t player = 0; player < goal_vertices_.size(); ++player) {
      const int goal_y = player == 0 ? 0 : topology_->config().height + 2;
      for (std::size_t offset = 0; offset < 3; ++offset) {
        const auto vertex = topology_->find_vertex(
            Point{topology_->config().width / 2 - 1 +
                      static_cast<int>(offset),
                  goal_y});
        if (!vertex.has_value()) {
          throw std::logic_error("neural topology is missing a goal vertex");
        }
        goal_vertices_[player][offset] = *vertex;
      }
    }
    for (std::size_t edge = 0; edge < model_data::kEdgeCount; ++edge) {
      const Segment segment = edge_segments_[edge];
      const auto rotated = topology_->find_edge(
          Segment{rotate_point(segment.a), rotate_point(segment.b)});
      if (!rotated.has_value()) {
        throw std::logic_error("neural edge map is not rotationally symmetric");
      }
      rotated_edges_[edge] = *rotated;
      if (std::abs(segment.a.y - segment.b.y) == 1) {
        const int cut = std::min(segment.a.y, segment.b.y);
        if (cut >= 0 && cut < 12) {
          edge_cuts_[edge] = static_cast<std::int8_t>(cut);
          ++cut_totals_[static_cast<std::size_t>(cut)];
        }
      }
    }
  }

  detail::SearchTopology::VertexIndex physical_vertex(
      std::size_t canonical, Player mover) const noexcept {
    return mover == Player::One
               ? static_cast<detail::SearchTopology::VertexIndex>(canonical)
               : rotated_vertices_[canonical];
  }

  detail::SearchTopology::EdgeIndex physical_edge(
      std::size_t canonical, Player mover) const noexcept {
    return mover == Player::One
               ? static_cast<detail::SearchTopology::EdgeIndex>(canonical)
               : rotated_edges_[canonical];
  }

  void calculate_turn_distances(const detail::SearchPosition &position) {
    constexpr int kUnreachable = std::numeric_limits<int>::max() / 4;
    distances_.fill(kUnreachable);
    distance_queue_.clear();
    distances_[position.ball_vertex()] = 0;
    distance_queue_.push_back(position.ball_vertex());
    while (!distance_queue_.empty()) {
      const auto vertex = distance_queue_.front();
      distance_queue_.pop_front();
      const int source_distance = distances_[vertex];
      const auto &adjacency = cooperative_adjacency_[vertex];
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const auto &arc = adjacency.arcs[index];
        if (position.edge_used(arc.edge)) {
          continue;
        }
        const Point destination = topology_->point(arc.destination);
        const bool zero_cost = position.vertex_visited(arc.destination) ||
                               is_boundary_point(topology_->config(),
                                                 destination) ||
                               is_goal_point(topology_->config(), destination);
        const int candidate = source_distance + (zero_cost ? 0 : 1);
        if (candidate >= distances_[arc.destination]) {
          continue;
        }
        distances_[arc.destination] = candidate;
        if (zero_cost) {
          distance_queue_.push_front(arc.destination);
        } else {
          distance_queue_.push_back(arc.destination);
        }
      }
    }
  }

  int minimum_goal_distance(Player mover, bool attacking) const {
    int result = 7;
    const Player goal_owner = attacking ? mover : opponent(mover);
    const std::size_t owner_index =
        goal_owner == Player::One ? 0U : 1U;
    for (const auto vertex : goal_vertices_[owner_index]) {
      result = std::min(result, std::clamp(distances_[vertex], 0, 7));
    }
    return result;
  }

  ComponentFeatures component_features(
      const detail::SearchPosition &position) const {
    using Vertex = detail::SearchTopology::VertexIndex;
    std::array<bool, model_data::kVertexCount> in_component{};
    std::array<bool, model_data::kVertexCount> frontier_seen{};
    std::array<bool, model_data::kEdgeCount> internal_edge_seen{};
    std::array<Vertex, model_data::kVertexCount> queue{};
    std::size_t head = 0;
    std::size_t tail = 0;
    const Player mover = position.to_move();
    in_component[position.ball_vertex()] = true;
    queue[tail++] = position.ball_vertex();
    ComponentFeatures result;

    while (head < tail) {
      const Vertex vertex = queue[head++];
      const Point point = topology_->point(vertex);
      result.touches_attacking_goal =
          result.touches_attacking_goal ||
          is_attacking_goal(topology_->config(), point, mover);
      result.touches_own_goal =
          result.touches_own_goal ||
          is_attacking_goal(topology_->config(), point, opponent(mover));
      const auto &adjacency = topology_->adjacency(vertex, mover);
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const auto &arc = adjacency.arcs[index];
        if (position.edge_used(arc.edge)) {
          continue;
        }
        const Point destination = topology_->point(arc.destination);
        const bool zero_cost = position.vertex_visited(arc.destination) ||
                               is_boundary_point(topology_->config(),
                                                 destination) ||
                               is_goal_point(topology_->config(), destination);
        if (zero_cost) {
          if (!internal_edge_seen[arc.edge]) {
            internal_edge_seen[arc.edge] = true;
            ++result.internal_edges;
          }
          if (!in_component[arc.destination]) {
            in_component[arc.destination] = true;
            queue[tail++] = arc.destination;
          }
          continue;
        }
        if (frontier_seen[arc.destination]) {
          continue;
        }
        frontier_seen[arc.destination] = true;
        bool safe = false;
        const auto &handoff =
            topology_->adjacency(arc.destination, opponent(mover));
        for (std::uint8_t child = 0; child < handoff.count; ++child) {
          if (handoff.arcs[child].edge != arc.edge &&
              !position.edge_used(handoff.arcs[child].edge)) {
            safe = true;
            break;
          }
        }
        if (safe) {
          ++result.safe_frontiers;
        } else {
          ++result.dead_frontiers;
        }
      }
    }

    result.vertices = tail;
    return result;
  }
};

std::size_t physical_direction(Point from, Point to) {
  const Point delta{to.x - from.x, to.y - from.y};
  for (std::size_t direction = 0; direction < kDirectionDeltas.size();
       ++direction) {
    if (kDirectionDeltas[direction] == delta) {
      return direction;
    }
  }
  throw std::logic_error("search produced a non-adjacent move");
}

std::size_t canonical_direction(Point from, Point to, Player mover) {
  const std::size_t physical = physical_direction(from, to);
  return mover == Player::One ? physical : (physical + 4U) & 7U;
}

float mover_value_to_player_one(float value, Player mover) noexcept {
  return mover == Player::One ? value : -value;
}

float player_one_value_for_mover(float value, Player mover) noexcept {
  return mover == Player::One ? value : -value;
}

struct SearchConfig {
  std::size_t max_nodes{kMaximumNodes};
  std::uint64_t max_simulations{std::numeric_limits<std::uint64_t>::max()};
  float c_puct{kDefaultCpuct};
  std::optional<SearchClock::time_point> absolute_deadline{};
};

struct SearchStats {
  std::uint64_t simulations{};
  std::size_t nodes{};
  std::uint64_t neural_evaluations{};
  std::uint32_t max_depth{};
  float root_value{};
  bool deadline_reached{};
  bool node_cap_reached{};
};

class NeuralPuctTrainingAccess;

class NeuralPuctSearch {
 public:
  NeuralPuctSearch(const GameState &state, SearchConfig config)
      : config_(config), root_mover_(state.to_move),
        topology_(std::make_shared<detail::SearchTopology>(state.config)),
        position_(topology_, state), features_(topology_) {
    if (config_.max_nodes == 0 || config_.max_simulations == 0 ||
        !std::isfinite(config_.c_puct) || config_.c_puct < 0.0F) {
      throw std::invalid_argument("invalid neural PUCT search configuration");
    }
    if (position_.is_terminal()) {
      throw std::invalid_argument("cannot search a terminal position");
    }
    std::array<std::uint8_t, detail::kMaximumMoves> legal{};
    if (position_.legal_slots(legal) == 0) {
      throw std::invalid_argument("cannot search a position without moves");
    }
    nodes_.reserve(std::min<std::size_t>(config_.max_nodes, 8192));
    nodes_.push_back(make_node());
    stats_.nodes = 1;
  }

  std::vector<Move> run() {
    const std::size_t root_depth = position_.undo_depth();
    const float root_value_p1 = expand(0);
    nodes_[0].visits = 1;
    nodes_[0].value_sum_p1 = root_value_p1;
    if (!nodes_[0].exact_winning_child.has_value()) {
      while (stats_.simulations < config_.max_simulations) {
        if (deadline_expired()) {
          stats_.deadline_reached = true;
          break;
        }
        simulate();
      }
    }
    position_.unmake_to(root_depth);
    stats_.nodes = nodes_.size();
    const float mean_p1 = nodes_[0].visits == 0
                              ? 0.0F
                              : nodes_[0].value_sum_p1 /
                                    static_cast<float>(nodes_[0].visits);
    stats_.root_value =
        root_mover_ == Player::One ? mean_p1 : -mean_p1;
    return extract_complete_action();
  }

  const SearchStats &stats() const noexcept { return stats_; }

  ModelOutput inspect_model_output() {
    ++stats_.neural_evaluations;
    const auto &encoded = features_.encode(position_);
    return model_.evaluate(encoded, features_.active_indices().data(),
                           features_.active_count());
  }

  const std::array<float, kFeatureCount> &inspect_features() {
    return features_.encode(position_);
  }

 private:
  friend class NeuralPuctTrainingAccess;

  static constexpr std::uint32_t kNoNode =
      std::numeric_limits<std::uint32_t>::max();

  struct Edge {
    Move move{};
    float prior{};
    float value_sum_p1{};
    std::uint32_t child{kNoNode};
    std::uint32_t visits{};
    std::uint8_t slot{};
    std::uint8_t canonical_direction{};
  };

  struct Node {
    std::array<Edge, detail::kMaximumMoves> edges{};
    float value_sum_p1{};
    std::uint32_t visits{};
    Player mover{Player::One};
    std::uint8_t edge_count{};
    bool expanded{};
    bool terminal{};
    std::optional<std::uint8_t> exact_winning_child{};
  };

  struct PathEntry {
    std::uint32_t node{};
    std::uint8_t edge{};
  };

  struct EvaluationCacheEntry {
    detail::PositionKey key{};
    ModelOutput output{};
    bool occupied{};
  };

  static constexpr std::size_t kEvaluationCacheSize = 16'384;
  static_assert((kEvaluationCacheSize & (kEvaluationCacheSize - 1U)) == 0);

  SearchConfig config_;
  Player root_mover_;
  std::shared_ptr<const detail::SearchTopology> topology_;
  detail::SearchPosition position_;
  FeatureEncoder features_;
  QuantizedModel model_;
  std::array<EvaluationCacheEntry, kEvaluationCacheSize> evaluation_cache_{};
  std::vector<Node> nodes_;
  std::vector<PathEntry> path_;
  SearchStats stats_;

  Node make_node() const {
    Node result;
    result.mover = position_.to_move();
    result.terminal = position_.is_terminal();
    return result;
  }

  bool deadline_expired() const noexcept {
    return config_.absolute_deadline.has_value() &&
           SearchClock::now() >= *config_.absolute_deadline;
  }

  float terminal_value_p1() const {
    const std::optional<Player> winner = position_.winner();
    if (!winner.has_value()) {
      throw std::logic_error("terminal neural PUCT node has no winner");
    }
    return *winner == Player::One ? 1.0F : -1.0F;
  }

  ModelOutput neural_output() {
    const detail::PositionKey key = position_.position_key();
    const std::uint64_t rotated_second =
        (key.second << 23U) | (key.second >> (64U - 23U));
    EvaluationCacheEntry &entry = evaluation_cache_[
        detail::mix_position_key(key.first ^ rotated_second) &
        (kEvaluationCacheSize - 1U)];
    if (entry.occupied && entry.key == key) {
      return entry.output;
    }
    ++stats_.neural_evaluations;
    entry.key = key;
    const auto &encoded = features_.encode(position_);
    entry.output = model_.evaluate(encoded, features_.active_indices().data(),
                                   features_.active_count());
    entry.occupied = true;
    return entry.output;
  }

  float expand(std::uint32_t node_index) {
    if (position_.is_terminal()) {
      nodes_[node_index].terminal = true;
      nodes_[node_index].expanded = true;
      return terminal_value_p1();
    }
    ModelOutput output = neural_output();
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position_.legal_slots(slots);
    if (count == 0) {
      throw std::logic_error("nonterminal neural PUCT node is blocked");
    }
    const Player mover = position_.to_move();
    const Point ball = position_.ball();
    float maximum_logit = -std::numeric_limits<float>::infinity();
    for (std::uint8_t index = 0; index < count; ++index) {
      const Move move = position_.move_for_slot(slots[index]);
      maximum_logit = std::max(
          maximum_logit,
          output.policy_logits[canonical_direction(ball, move.to, mover)]);
    }
    float denominator = 0.0F;
    Node &node = nodes_[node_index];
    node.edge_count = count;
    node.mover = mover;
    for (std::uint8_t index = 0; index < count; ++index) {
      const std::uint8_t slot = slots[index];
      const Move move = position_.move_for_slot(slot);
      const std::size_t direction = canonical_direction(ball, move.to, mover);
      const float prior = std::exp(std::clamp(
          output.policy_logits[direction] - maximum_logit, -80.0F, 0.0F));
      node.edges[index] = Edge{move, prior, 0.0F, kNoNode, 0, slot,
                               static_cast<std::uint8_t>(direction)};
      denominator += prior;
    }
    std::sort(node.edges.begin(), node.edges.begin() + count,
              [](const Edge &left, const Edge &right) {
                return left.canonical_direction < right.canonical_direction;
              });
    for (std::uint8_t index = 0; index < count; ++index) {
      node.edges[index].prior /= denominator;
      position_.make_move(node.edges[index].slot);
      const std::optional<Player> winner = position_.winner();
      position_.unmake_move();
      if (winner.has_value() && *winner == mover &&
          !node.exact_winning_child.has_value()) {
        node.exact_winning_child = index;
      }
    }
    if (node.exact_winning_child.has_value()) {
      for (std::uint8_t index = 0; index < count; ++index) {
        node.edges[index].prior =
            index == *node.exact_winning_child ? 1.0F : 0.0F;
      }
    }
    node.expanded = true;
    const float mover_value = node.exact_winning_child.has_value()
                                  ? 1.0F
                                  : std::clamp(output.value, -1.0F, 1.0F);
    return mover == Player::One ? mover_value : -mover_value;
  }

  std::uint8_t select_edge(std::uint32_t node_index) const {
    const Node &node = nodes_[node_index];
    if (node.exact_winning_child.has_value()) {
      return *node.exact_winning_child;
    }
    const float parent_sqrt =
        std::sqrt(static_cast<float>(std::max<std::uint32_t>(node.visits, 1)));
    std::uint8_t best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (std::uint8_t index = 0; index < node.edge_count; ++index) {
      const Edge &edge = node.edges[index];
      const float mean_p1 = edge.visits == 0
                                ? 0.0F
                                : edge.value_sum_p1 /
                                      static_cast<float>(edge.visits);
      const float decision_value =
          player_one_value_for_mover(mean_p1, node.mover);
      const float exploration =
          config_.c_puct * edge.prior * parent_sqrt /
          static_cast<float>(1U + edge.visits);
      const float score = decision_value + exploration;
      if (score > best_score) {
        best_score = score;
        best = index;
      }
    }
    return best;
  }

  void simulate() {
    const std::size_t root_depth = position_.undo_depth();
    path_.clear();
    std::uint32_t node_index = 0;
    std::optional<std::uint32_t> leaf_node;
    float leaf_value_p1 = 0.0F;
    std::uint32_t depth = 0;

    while (true) {
      Node &node = nodes_[node_index];
      if (position_.is_terminal()) {
        leaf_value_p1 = terminal_value_p1();
        leaf_node = node_index;
        break;
      }
      if (!node.expanded) {
        leaf_value_p1 = expand(node_index);
        leaf_node = node_index;
        break;
      }
      const std::uint8_t edge_index = select_edge(node_index);
      const std::uint8_t slot = node.edges[edge_index].slot;
      position_.make_move(slot);
      path_.push_back(PathEntry{node_index, edge_index});
      ++depth;
      const std::uint32_t child = nodes_[node_index].edges[edge_index].child;
      if (child != kNoNode) {
        node_index = child;
        continue;
      }
      if (nodes_.size() >= config_.max_nodes) {
        stats_.node_cap_reached = true;
        leaf_value_p1 = position_.is_terminal()
                            ? terminal_value_p1()
                            : mover_value_to_player_one(neural_output().value,
                                                        position_.to_move());
        break;
      }
      const std::uint32_t new_child =
          static_cast<std::uint32_t>(nodes_.size());
      nodes_.push_back(make_node());
      nodes_[node_index].edges[edge_index].child = new_child;
      node_index = new_child;
    }

    if (leaf_node.has_value()) {
      Node &leaf = nodes_[*leaf_node];
      ++leaf.visits;
      leaf.value_sum_p1 += leaf_value_p1;
    }
    for (auto iterator = path_.rbegin(); iterator != path_.rend(); ++iterator) {
      Node &parent = nodes_[iterator->node];
      Edge &edge = parent.edges[iterator->edge];
      ++edge.visits;
      edge.value_sum_p1 += leaf_value_p1;
      ++parent.visits;
      parent.value_sum_p1 += leaf_value_p1;
    }
    position_.unmake_to(root_depth);
    ++stats_.simulations;
    stats_.max_depth = std::max(stats_.max_depth, depth);
  }

  std::uint8_t visit_max_edge(const Node &node) const {
    if (node.exact_winning_child.has_value()) {
      return *node.exact_winning_child;
    }
    std::uint8_t best = 0;
    for (std::uint8_t index = 1; index < node.edge_count; ++index) {
      const Edge &candidate = node.edges[index];
      const Edge &current = node.edges[best];
      const float candidate_mean =
          candidate.visits == 0
              ? 0.0F
              : candidate.value_sum_p1 / static_cast<float>(candidate.visits);
      const float current_mean =
          current.visits == 0
              ? 0.0F
              : current.value_sum_p1 / static_cast<float>(current.visits);
      const float sign =
          player_one_value_for_mover(1.0F, node.mover);
      if (candidate.visits > current.visits ||
          (candidate.visits == current.visits &&
           (sign * candidate_mean > sign * current_mean ||
            (candidate_mean == current_mean &&
             candidate.prior > current.prior)))) {
        best = index;
      }
    }
    return best;
  }

  std::uint8_t neural_fallback_slot() {
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position_.legal_slots(slots);
    if (count == 0) {
      throw std::logic_error("neural fallback reached a blocked state");
    }
    const Player mover = position_.to_move();
    for (std::uint8_t index = 0; index < count; ++index) {
      position_.make_move(slots[index]);
      const std::optional<Player> winner = position_.winner();
      position_.unmake_move();
      if (winner.has_value() && *winner == mover) {
        return slots[index];
      }
    }
    const ModelOutput output = neural_output();
    const Point ball = position_.ball();
    std::uint8_t best = slots[0];
    std::size_t best_direction = canonical_direction(
        ball, position_.move_for_slot(best).to, mover);
    float best_logit = output.policy_logits[best_direction];
    for (std::uint8_t index = 1; index < count; ++index) {
      const std::uint8_t slot = slots[index];
      const std::size_t direction = canonical_direction(
          ball, position_.move_for_slot(slot).to, mover);
      const float logit = output.policy_logits[direction];
      if (logit > best_logit ||
          (logit == best_logit && direction < best_direction)) {
        best = slot;
        best_direction = direction;
        best_logit = logit;
      }
    }
    return best;
  }

  std::vector<Move> extract_complete_action() {
    const std::size_t root_depth = position_.undo_depth();
    std::vector<Move> action;
    action.reserve(8);
    std::optional<std::uint32_t> node_index = 0;
    for (std::size_t edge_count = 0; edge_count <= topology_->edge_count();
         ++edge_count) {
      if (position_.is_terminal() || position_.to_move() != root_mover_) {
        break;
      }
      std::uint8_t slot{};
      std::uint32_t child = kNoNode;
      if (node_index.has_value() && nodes_[*node_index].expanded &&
          nodes_[*node_index].edge_count != 0) {
        const std::uint8_t edge = visit_max_edge(nodes_[*node_index]);
        slot = nodes_[*node_index].edges[edge].slot;
        child = nodes_[*node_index].edges[edge].child;
      } else {
        slot = neural_fallback_slot();
      }
      action.push_back(position_.move_for_slot(slot));
      position_.make_move(slot);
      node_index = child == kNoNode ? std::nullopt
                                    : std::optional<std::uint32_t>(child);
    }
    if (action.empty() ||
        (!position_.is_terminal() && position_.to_move() == root_mover_)) {
      position_.unmake_to(root_depth);
      throw std::logic_error("neural PUCT failed to complete a rebound action");
    }
    position_.unmake_to(root_depth);
    return action;
  }
};

Player player_for_id(int player_id) {
  if (player_id == 0) {
    return Player::One;
  }
  if (player_id == 1) {
    return Player::Two;
  }
  throw std::invalid_argument("player id must be zero or one");
}

Move decode_direction(Point from, char direction) {
  if (direction < '0' || direction > '7') {
    throw std::invalid_argument("direction must be between zero and seven");
  }
  const Point delta =
      kDirectionDeltas[static_cast<std::size_t>(direction - '0')];
  return Move{{from.x + delta.x, from.y + delta.y}};
}

char encode_direction(Point from, Point to) {
  return static_cast<char>('0' + physical_direction(from, to));
}

void apply_encoded_turn(GameState &state, std::string_view encoded) {
  if (encoded.empty()) {
    throw std::invalid_argument("a turn cannot be empty");
  }
  GameState next = state;
  const Player mover = next.to_move;
  for (const char direction : encoded) {
    if (is_terminal(next) || next.to_move != mover) {
      throw std::invalid_argument("encoded action continues after turn end");
    }
    next = apply_move(next, decode_direction(next.ball, direction));
  }
  if (!is_terminal(next) && next.to_move == mover) {
    throw std::invalid_argument("encoded action omits a required rebound");
  }
  state = std::move(next);
}

std::string choose_complete_turn(GameState &state,
                                 std::uint32_t search_time_ms,
                                 std::size_t max_nodes = kMaximumNodes) {
  SearchConfig config;
  config.max_nodes = max_nodes;
  config.absolute_deadline =
      SearchClock::now() + std::chrono::milliseconds(search_time_ms);
  NeuralPuctSearch search(state, config);
  const std::vector<Move> action = search.run();
  const Player mover = state.to_move;
  std::string encoded;
  encoded.reserve(action.size());
  for (const Move move : action) {
    if (is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("neural PUCT action continues after turn end");
    }
    encoded.push_back(encode_direction(state.ball, move.to));
    state = apply_move(state, move);
  }
  if (!is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("neural PUCT action ends during a rebound");
  }
  return encoded;
}

}  // namespace papersoccer::neural_puct

#ifndef PAPER_SOCCER_NEURAL_PUCT_NO_MAIN
int main() {
  using namespace papersoccer;
  using namespace papersoccer::neural_puct;

  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  int player_id = -1;
  if (!(std::cin >> player_id)) {
    return 0;
  }
  Player me;
  try {
    me = player_for_id(player_id);
  } catch (const std::exception &) {
    return 1;
  }
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  RulesConfig rules;
  rules.goal_rule = GoalRule::OwnGoalsAllowed;
  rules.blocked_rule = BlockedRule::MoverLoses;
  GameState state = make_initial_state(rules);
  bool first_execution = true;

  while (true) {
    int opponent_move_length = 0;
    if (!(std::cin >> opponent_move_length)) {
      break;
    }
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string opponent_move;
    if (!std::getline(std::cin, opponent_move)) {
      break;
    }
    if (opponent_move != "-" &&
        opponent_move_length != static_cast<int>(opponent_move.size())) {
      return 1;
    }

    try {
      if (opponent_move == "-") {
        if (player_id != 0 || !first_execution) {
          return 1;
        }
      } else {
        apply_encoded_turn(state, opponent_move);
      }
      if (is_terminal(state)) {
        return 0;
      }
      if (state.to_move != me) {
        return 1;
      }
      const std::uint32_t time_ms =
          first_execution ? kFirstSearchTimeMs : kLaterSearchTimeMs;
      const std::string action = choose_complete_turn(state, time_ms);
      std::cout << action << std::endl;
      first_execution = false;
    } catch (const std::exception &) {
      return 1;
    }
  }
  return 0;
}
#endif
