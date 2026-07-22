
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC optimize("O3")
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

namespace papersoccer::selfplay_nn::replay_book {

struct Replay {
  std::uint8_t player_id;
  std::uint8_t first_turn;
  std::string_view transcript;
};

inline constexpr std::array<Replay, 24> kReplays{{
    {0, 6, "0/5/21/2/7/523/3/35/0/36350/07/5/6/6/0/5/5/631/16430/5023/21/41674/7577/2/21/3/577/55327/724101305/533/3050100/321647/66666/72/5202/243/11425056542006074772/71/714/3/66144/3/6302/330/007/2/2/2/0/0363636/7507/13/34546764750014/71/05211"},
    {1, 5, "1/1/7/6/6/3/07/05/74/53/1/2060350634/14/663/1211/4/1/03635/6060/1474553/032/17723/364/27056/56/30206577/724452/22076524144/2/0574/1313/12744/3/17563/235/350/17/1647074665/3/5/2/01/05/0367635/61/7241466335/6300/61420/742164203/02/3/24705/652/105633/6/35"},
    {0, 4, "0/6/5/5/71/34/1/033/5/5/2/5/57/1/2477/1647253/463071420/33/641000/7245/7470/063224/366111/3/0/3/17/55/520061/6/7/2/1/03/057/65/753/0/14652530/1/21/02543367432235/36/17/02545/2357/571/32/05417667070/20/1465317074612/03632/025441/1/427713527/46/360633/45/64/47230/00531657667161/65"},
    {0, 2, "0/2/7/45/7/5/71/34/2212/2/7/1/6/1/03636074/33535/7/00535/0/633/10/35/035/27635/50/241/035067/46/57/1/0/5/2310/256/102/16613165475/5331/17574/236746167/064/276313/17/60/672/3310/6/52/42756/06310/50223611/07"},
    {1, 3, "1/1/0/1/275/03674/2476/4/7/75/6143/123/617/505/3/23/6/125032/0/361745642/4/1/63/2774142564/2/03677770645/72/520224/6/03144/75/0653/13/6/4/74/61632116/50275/3552/14/7421/02716443/1/175364613/272564/3/1/756/035/725/6/614145/2/500333"},
    {1, 3, "7/6/7/53/10/34/71/45/221/2/1/35/70/54/17/43/660/33/020/641364/2755/2/166/1336/74747/5/0/5/6/3/5/31/7167/7223036135/24/3/50/241/25727/0252/75767/234/4/125635"},
    {0, 2, "0/1/0/6/33/5/3/6/5/3/6/05/21/66/7/4/6/3/161/13442/72/01/17/0/255336/5702/4202577/22/520167/5452066/6/1/7/7/0524/53/27/164453/2572461330067/2/50141016/64/146652134/632/0613574302100/164721/41675/03/5030654/47271/17"},
    {1, 19, "6/7/5/61/44/53/0/0617227/47271/053/03/5023/1/4/7/425/61434/3/0/3/65/3/41/65/67/1/6/47275253147/11/425/01/61410/63613/42/754/607/03647/16/366163071/123/4/555/234/44"},
    {1, 3, "3/1/7/7/7/2/2/505/6/6/1/7/5352/7243/2/3/6/02574/5/3/6/6171/6110300/5253/312/1/7/2/0/25252744/64136/02566/44/71134/4/27254/4/166/74/6/16/323257/50/1467016023/03663/534705/07/2050610164563353/57/006353012/3647/23/02525"},
    {0, 40, "0/3/2/3/6/5/2/4/76/5/6/4/10/55/2476317/1/221/030524745/674663071/36/071/3/0/2/21/3/5671/47/5671/4/72714/43/65212221/4/2/505/247021/2506335/2553/66/677/0174744/366311/1256650134/6413312/005325357/014271761/0270/56/57/47/22/2470/13/030522575771"},
    {0, 18, "0/5/21/3/0/3/17/55/061/4276/1/42/217/4/427570/5/4432225/35/36/57/1/4/125664130061/02745/2470/274763/25/5/0/5/230/65/57/4721/2470/5/71/335/7242476061/721/0/364/671/33/0523111/60575/6671/4/721/7533/6301/74/2221/3166614743/67/756/3552133550500/33/6631231642706/427543/07027027/4561160327417/01"},
    {1, 3, "7/6/1/44/21/7/7/422/7/455/00/21657/1/75353/10/27433/141/3/17/25/42700/256/3607/135524/665760/166352325/06/33/0/10676356144/133/3/1/6/1/44/2/71/7/42355/01/435755"},
    {0, 20, "7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474/42474176"},
    {0, 4, "0/6/5/4/5/53/61/72442/30/2/7/46413/00/2/531/316/50216/56/52076572/450120/357177/532/21/03/2/3/2500/65/441/24763072/4167474/2507/2253117/50/247656431605470/7272/350230/270/330/1/7/2/42/25/2743527/677/45017/2/46124/5201317/05/74/53611631757/65/31/30616474/60317"},
    {1, 3, "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35/7/164675/6/6/71/72524/4611232/764/763/30654/17201/3550102144444/1721/425/46/16467/5/024674/75252/53/530/1/0/711324/75/0632/3/1/3/17505675/23/2027545/2/13/3/25050/50/11356675/47/4772420613520633/10/3067425/46165"},
    {0, 4, "0/6/5/4/5/3/6/7243/5716172/23/0/3/1/2/1/35/77/43/225/4/250174700/54/661/45/05417/60316443/47411/4/60217077/4/7/53/120/543212225/017/4/167/1/75/32/036175/74235/357/41/652770/55/44717271/4/71413/5/2001"},
    {1, 1, "4/6/6/3/6/3/2/746/6/14117/7/7/71/63/003/14/6544/61613/0/7144524612/4/61713/4/53550143/01/6605/74/20553/6301021/72432/4/2/75/555/7531070/723613423/2/4/71/43/17/0522725743/425750/306064/5747/72422/13610/722744641320555/0603364"},
    {1, 1, "2/4/6/05/0/33/6/7241767/1/1/6/42/5/67/63/610/2/5453/2/10077/46114/360753114/54/1303/63/03/6455/0636/74/205721/0613316/6521312/7/43/163635/27/000/6657645/3074453/63/6113/024/11/247642/71/25721165444235/4167/03/056/666/52/06333"},
    {0, 30, "0/3/2/3/6/5/2/4/76/5/6/4/10/55/2476317/1/221/030524745/674663071/36/071/3/0/2/21/3/0/3/17/25/0567/4/7/5/4610/6/1/36075333325/4471271/065556333/63012/5/31/0632/25071/01703664723/052531636650706/66/063131241160/1/2566/357165/247020/756/46131/213/0305247575001"},
    {1, 7, "1/1/7/6/0/75/03/213/570/6433/00/063203/0365/4644/1/3/0/2535/0/35/07/216575/022/003545/77/6/7/53/0/5253/0/5252/01/2/7/43/00/164561613255/074321/7672410/1/013563335/06/52032366/03350/57/41/65/67/614474113/0/3644"},
    {1, 11, "0/0/0/6/6/5/2/5/0527271/614347/013/433/1/3/17/54/657/5/2/5/06/61434/7/23/1/7244/1/2/1/164276336/60/335/7/4/225/41675/41/66/57/7/7101/4722223535/242570310/052746/063450/166652/06/45310136/00/35335"},
    {0, 20, "0/3/2/3/6/5/2/4/76/5/6/5/63072/1/30/24/764671/1/0/2/21/3/44710/3/17/25/0527/1/03560/03475/24607/4/7/0365/0/031/0"},
    {1, 19, "7/6/0/2/1/4/1/2/0/53/01/2750363/03667/274466/0/75/0522/205335023/0/2554/1/65/7/0105354/7/53/11/6613425/0632/5/06/653/0/063135/752250300/23/31/30/7/447/10/3/17/55/775470/071/0/05327524/10/13313/0525/5023/250367/63/00707/4165633/025641231/274455/74236522702024/67/07463254/453"},
    {1, 3, "0/0/3/67/27/45/5/2/5/6143/5/717271/1/7/532/27412/41/654/74761/06344/1721/05/3075221/23/57/063633/652141/6033/35/6/3/35/27/002525457/41/65/67/57132/1/31/2530606/1/6/7/012/216/36016/17535254/35475206/7/713550/074322446/547/71/0252347/460220/64117/0642447123/300561/4535/502363"},
}};

}  // namespace papersoccer::selfplay_nn::replay_book

namespace papersoccer::selfplay_nn::neural_model {

inline constexpr std::size_t kInputCount = 423;
inline constexpr std::size_t kEdgeCount = 316;
inline constexpr std::size_t kVertexCount = 105;
inline constexpr std::size_t kHiddenOne = 16;
inline constexpr std::size_t kHiddenTwo = 16;
inline constexpr std::size_t kPolicyCount = 8;
inline constexpr float kW1Scale = 0.00299100378F;
inline constexpr float kW2Scale = 0.00976546261F;
inline constexpr float kValueScale = 0.00608489692F;
inline constexpr float kPolicyScale = 0.00418342756F;
inline constexpr std::array<float, 16> kB1{{-0.0263735000F, 0.00136247650F, 0.00624843221F, 0.0248609167F, 0.0746440738F, 0.0138767995F, -0.00574371126F, -0.129159629F, 0.0148954317F, -0.0265926160F, -0.0437100008F, -0.0349988453F, -0.0273925923F, -0.0303408410F, 0.0150259370F, 0.0280531161F}};
inline constexpr std::array<float, 16> kB2{{0.101082690F, 0.0587032400F, 0.0700766221F, -0.0365668871F, -0.0209626984F, -0.0469203182F, -0.0248493012F, 0.0700823665F, -0.0352873504F, 0.0246810019F, 0.0534638576F, 0.000144182675F, -0.0466778278F, -0.0790033340F, -0.0726257786F, -0.107910626F}};
inline constexpr float kValueBias = 0.0570807420F;
inline constexpr std::array<float, 8> kPolicyBias{{0.0418817624F, 0.0500030592F, 0.0286710933F, -0.0696171001F, -0.0954422653F, -0.0783621520F, 0.0332519822F, 0.0355964527F}};
inline constexpr std::string_view kEncodedWeights = "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";

}  // namespace papersoccer::selfplay_nn::neural_model

namespace papersoccer::selfplay_nn {

constexpr std::uint32_t kFirstSearchTimeMs = 650;
constexpr std::uint32_t kLaterSearchTimeMs = 130;
constexpr std::uint32_t kMaximumTurnDepth = 32;
constexpr std::uint64_t kMaximumNodes = 3'000'000;
constexpr std::size_t kTranspositionEntries = 262'144;
constexpr std::size_t kEvaluationEntries = 131'072;

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

constexpr std::array<Point, 8> kDirectionDeltas{{
    {0, -1},
    {1, -1},
    {1, 0},
    {1, 1},
    {0, 1},
    {-1, 1},
    {-1, 0},
    {-1, -1},
}};

class SearchBudgetReached final {};

using SearchClock = std::chrono::steady_clock;

enum class ScoreBound { Exact, Lower, Upper };

struct SearchConfig {
  std::uint32_t max_turn_depth{kMaximumTurnDepth};
  std::uint64_t max_nodes{kMaximumNodes};
  std::size_t transposition_entries{kTranspositionEntries};
  std::size_t evaluation_entries{kEvaluationEntries};
  std::uint32_t max_time_ms{};
  std::optional<SearchClock::time_point> absolute_deadline{};
  bool root_seed_endpoints{true};
  bool terminal_bound_pruning{true};
  bool root_transposition_pruning{true};
  int neural_value_blend_percent{};
  int neural_policy_order_weight{};
};

struct SearchStats {
  std::uint32_t completed_turn_depth{};
  std::uint32_t attempted_turn_depth{};
  std::uint64_t nodes{};
  std::uint64_t leaf_evaluations{};
  std::uint64_t terminal_nodes{};
  std::uint64_t completed_actions{};
  std::uint64_t cutoffs{};
  std::uint64_t transposition_probes{};
  std::uint64_t transposition_hits{};
  std::uint64_t transposition_cutoffs{};
  std::uint64_t transposition_stores{};
  std::uint64_t continuation_transposition_hits{};
  std::uint64_t evaluation_cache_probes{};
  std::uint64_t evaluation_cache_hits{};
  std::uint64_t neural_evaluations{};
  std::uint64_t neural_policy_orderings{};
  std::uint64_t terminal_bound_cutoffs{};
  std::uint64_t forced_edges{};
  std::uint64_t root_seed_actions{};
  std::uint64_t root_transposition_reuses{};
  std::uint32_t max_action_edges{};
  int root_score{};
  bool budget_exhausted{};
};

struct TranspositionEntry {
  detail::PositionKey key{};
  std::uint32_t depth{};
  int score{};
  Move best_move{};
  ScoreBound bound{ScoreBound::Exact};
  bool occupied{};
};

class TranspositionTable {
 public:
  explicit TranspositionTable(std::size_t entries) : entries_(entries) {}

  const TranspositionEntry *find(detail::PositionKey key) const noexcept {
    if (entries_.size() < 2) {
      return nullptr;
    }
    const std::size_t bucket = bucket_index(key);
    for (std::size_t way = 0; way < 2; ++way) {
      const TranspositionEntry &entry = entries_[bucket + way];
      if (entry.occupied && entry.key == key) {
        return &entry;
      }
    }
    return nullptr;
  }

  bool store(detail::PositionKey key, std::uint32_t depth, int score,
             Move best_move, ScoreBound bound) noexcept {
    if (entries_.size() < 2) {
      return false;
    }
    const std::size_t bucket = bucket_index(key);
    TranspositionEntry *victim = nullptr;
    for (std::size_t way = 0; way < 2; ++way) {
      TranspositionEntry &entry = entries_[bucket + way];
      if (entry.occupied && entry.key == key) {
        if (entry.depth > depth) {
          return false;
        }
        victim = &entry;
        break;
      }
      if (!entry.occupied || victim == nullptr || entry.depth < victim->depth) {
        victim = &entry;
      }
    }
    if (victim == nullptr || (victim->occupied && victim->depth > depth)) {
      return false;
    }
    *victim = TranspositionEntry{key, depth, score, best_move, bound, true};
    return true;
  }

 private:
  std::vector<TranspositionEntry> entries_;

  std::size_t bucket_index(detail::PositionKey key) const noexcept {
    const std::uint64_t combined =
        key.first ^ ((key.second << 23U) | (key.second >> 41U));
    const std::size_t buckets = entries_.size() / 2U;
    return 2U * static_cast<std::size_t>(combined % buckets);
  }
};

struct EvaluationEntry {
  detail::PositionKey key{};
  int score{};
  bool occupied{};
};

class EvaluationTable {
 public:
  explicit EvaluationTable(std::size_t entries) : entries_(entries) {}

  std::optional<int> find(detail::PositionKey key) const noexcept {
    if (entries_.size() < 2) {
      return std::nullopt;
    }
    const std::size_t bucket = bucket_index(key);
    for (std::size_t way = 0; way < 2; ++way) {
      const EvaluationEntry &entry = entries_[bucket + way];
      if (entry.occupied && entry.key == key) {
        return entry.score;
      }
    }
    return std::nullopt;
  }

  void store(detail::PositionKey key, int score) noexcept {
    if (entries_.size() < 2) {
      return;
    }
    const std::size_t bucket = bucket_index(key);
    for (std::size_t way = 0; way < 2; ++way) {
      EvaluationEntry &entry = entries_[bucket + way];
      if (!entry.occupied || entry.key == key) {
        entry = EvaluationEntry{key, score, true};
        return;
      }
    }
    const std::uint64_t combined = combined_key(key);
    entries_[bucket + ((combined >> 32U) & 1U)] =
        EvaluationEntry{key, score, true};
  }

 private:
  std::vector<EvaluationEntry> entries_;

  std::uint64_t combined_key(detail::PositionKey key) const noexcept {
    return key.first ^ ((key.second << 29U) | (key.second >> 35U));
  }

  std::size_t bucket_index(detail::PositionKey key) const noexcept {
    const std::size_t buckets = entries_.size() / 2U;
    return 2U * static_cast<std::size_t>(combined_key(key) % buckets);
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
  int score{};
};

struct OrderedMoveList {
  std::array<OrderedMove, detail::kMaximumMoves> values{};
  std::uint8_t count{};
};

struct ActionSearchResult {
  int score{};
  Move first_move{};
};

struct RootResult {
  int score{};
  std::vector<Move> action;
};

int player_sign(Player player) noexcept {
  return player == Player::One ? 1 : -1;
}

class CompleteTurnSearch {
 public:
  struct NeuralOutput {
    float value_logit{};
    std::array<float, neural_model::kPolicyCount> policy_logits{};
  };

  CompleteTurnSearch(const GameState &state, SearchConfig config)
      : config_(config),
        topology_(std::make_shared<detail::SearchTopology>(state.config)),
        position_(topology_, state),
        table_(config.transposition_entries),
        evaluations_(config.evaluation_entries),
        distances_(topology_->vertex_count(), -1),
        queue_(topology_->vertex_count()) {
    if (config_.max_turn_depth == 0 ||
        config_.max_turn_depth > kMaximumTurnDepth || config_.max_nodes == 0) {
      throw std::invalid_argument("invalid complete-turn search configuration");
    }
    if (config_.neural_value_blend_percent < 0 ||
        config_.neural_value_blend_percent > 100 ||
        config_.neural_policy_order_weight < 0) {
      throw std::invalid_argument("invalid neural search configuration");
    }
    if (config_.neural_value_blend_percent != 0 ||
        config_.neural_policy_order_weight != 0) {
      initialize_rotation_maps();
    }
    if (config_.absolute_deadline.has_value()) {
      deadline_ = config_.absolute_deadline;
    } else if (config_.max_time_ms != 0) {
      deadline_ =
          SearchClock::now() + std::chrono::milliseconds(config_.max_time_ms);
    }
  }

  std::vector<Move> run() {
    if (position_.is_terminal()) {
      throw std::invalid_argument("cannot search a terminal state");
    }

    std::vector<Move> committed_action = fallback_action();
    for (std::uint32_t depth = 1; depth <= config_.max_turn_depth; ++depth) {
      stats_.attempted_turn_depth = depth;
      try {
        RootResult result = search_root(depth);
        committed_action = std::move(result.action);
        stats_.completed_turn_depth = depth;
        stats_.root_score = result.score;
        if (std::abs(stats_.root_score) >=
            kMateScore - static_cast<int>(kMaximumTurnDepth)) {
          break;
        }
      } catch (const SearchBudgetReached &) {
        stats_.budget_exhausted = true;
        if (stats_.completed_turn_depth == 0 && captured_any_ &&
            !captured_action_.empty()) {
          committed_action = captured_action_;
          stats_.root_score = captured_score_;
        }
        break;
      }
    }
    return committed_action;
  }

  const SearchStats &stats() const noexcept { return stats_; }

  std::pair<float, std::array<float, neural_model::kPolicyCount>>
  neural_evaluation() {
    const NeuralOutput output = neural_output(position_.to_move());
    return {output.value_logit, output.policy_logits};
  }

 private:
  SearchConfig config_;
  std::shared_ptr<const detail::SearchTopology> topology_;
  detail::SearchPosition position_;
  TranspositionTable table_;
  EvaluationTable evaluations_;
  SearchStats stats_;
  std::vector<int> distances_;
  std::vector<detail::SearchTopology::VertexIndex> queue_;
  std::deque<detail::SearchTopology::VertexIndex> distance_queue_;
  std::vector<detail::SearchTopology::VertexIndex> rotated_vertices_;
  std::vector<detail::SearchTopology::EdgeIndex> rotated_edges_;
  std::optional<SearchClock::time_point> deadline_;
  std::vector<Move> current_action_;
  std::vector<Move> captured_action_;
  int captured_score_{};
  bool captured_any_{};

  void visit_node() {
    if (stats_.nodes >= config_.max_nodes) {
      throw SearchBudgetReached{};
    }
    if ((stats_.nodes & 15U) == 0U && deadline_.has_value() &&
        SearchClock::now() >= *deadline_) {
      throw SearchBudgetReached{};
    }
    ++stats_.nodes;
  }

  detail::PositionKey boundary_key() const noexcept {
    detail::PositionKey key = position_.position_key();
    detail::xor_position_key(
        key, detail::position_key_component(6, position_.ball_vertex()));
    return key;
  }

  int terminal_score(std::uint32_t turn_ply) {
    ++stats_.terminal_nodes;
    const std::optional<Player> winning_player = position_.winner();
    if (!winning_player.has_value()) {
      throw std::logic_error("terminal state has no winner");
    }
    const int distance = static_cast<int>(
        std::min<std::uint32_t>(turn_ply, kMaximumTurnDepth));
    return *winning_player == Player::One ? kMateScore - distance
                                          : -kMateScore + distance;
  }

  int immediate_win_score(Player mover, std::uint32_t turn_ply) const {
    const int distance = static_cast<int>(
        std::min<std::uint32_t>(turn_ply + 1U, kMaximumTurnDepth));
    return mover == Player::One ? kMateScore - distance
                                : -kMateScore + distance;
  }

  bool reached_immediate_win_bound(int score, Player mover,
                                   std::uint32_t turn_ply) const {
    return config_.terminal_bound_pruning &&
           score == immediate_win_score(mover, turn_ply);
  }

  Point rotated_point(Point point) const noexcept {
    return {topology_->config().width - point.x,
            topology_->config().height + 2 - point.y};
  }

  void initialize_rotation_maps() {
    using VertexIndex = detail::SearchTopology::VertexIndex;
    using EdgeIndex = detail::SearchTopology::EdgeIndex;
    constexpr EdgeIndex kMissingEdge =
        std::numeric_limits<EdgeIndex>::max();

    rotated_vertices_.resize(topology_->vertex_count());
    for (std::size_t vertex = 0; vertex < topology_->vertex_count(); ++vertex) {
      const auto rotated =
          topology_->find_vertex(rotated_point(topology_->point(vertex)));
      if (!rotated.has_value()) {
        throw std::logic_error("rotated vertex is missing from topology");
      }
      rotated_vertices_[vertex] = *rotated;
    }

    rotated_edges_.assign(topology_->edge_count(), kMissingEdge);
    for (std::size_t source = 0; source < topology_->vertex_count(); ++source) {
      for (const Player player : {Player::One, Player::Two}) {
        const auto &adjacency = topology_->adjacency(
            static_cast<VertexIndex>(source), player);
        for (std::uint8_t index = 0; index < adjacency.count; ++index) {
          const auto &arc = adjacency.arcs[index];
          const Segment rotated_segment{
              rotated_point(topology_->point(source)),
              rotated_point(topology_->point(arc.destination))};
          const auto rotated_edge = topology_->find_edge(rotated_segment);
          if (!rotated_edge.has_value()) {
            throw std::logic_error("rotated edge is missing from topology");
          }
          rotated_edges_[arc.edge] = *rotated_edge;
        }
      }
    }
    if (std::find(rotated_edges_.begin(), rotated_edges_.end(), kMissingEdge) !=
        rotated_edges_.end()) {
      throw std::logic_error("rotation map does not cover every edge");
    }
  }

  int shortest_goal_distance(Player player) {
    std::fill(distances_.begin(), distances_.end(), -1);
    std::size_t head = 0;
    std::size_t tail = 0;
    const auto start = position_.ball_vertex();
    distances_[start] = 0;
    queue_[tail++] = start;

    while (head < tail) {
      const auto vertex = queue_[head++];
      const auto &adjacency = topology_->adjacency(vertex, player);
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const auto &arc = adjacency.arcs[index];
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

  void calculate_goal_turn_distances(Player player) {
    constexpr int kUnreachable = 1'000'000;
    std::fill(distances_.begin(), distances_.end(), kUnreachable);
    distance_queue_.clear();
    const auto start = position_.ball_vertex();
    distances_[start] = 0;
    distance_queue_.push_front(start);

    while (!distance_queue_.empty()) {
      const auto vertex = distance_queue_.front();
      distance_queue_.pop_front();
      const auto &adjacency = topology_->adjacency(vertex, player);
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const auto &arc = adjacency.arcs[index];
        if (position_.edge_used(arc.edge)) {
          continue;
        }
        const Point destination = topology_->point(arc.destination);
        const bool continues_turn =
            is_boundary_point(topology_->config(), destination) ||
            position_.vertex_visited(arc.destination) ||
            is_goal_point(topology_->config(), destination);
        const int edge_cost = continues_turn ? 0 : 1;
        const int distance = distances_[vertex] + edge_cost;
        if (distance >= distances_[arc.destination]) {
          continue;
        }
        distances_[arc.destination] = distance;
        if (edge_cost == 0) {
          distance_queue_.push_front(arc.destination);
        } else {
          distance_queue_.push_back(arc.destination);
        }
      }
    }
  }

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

  static const std::vector<std::int8_t> &neural_weights() {
    static const std::vector<std::int8_t> weights = [] {
      std::vector<std::int8_t> decoded;
      decoded.reserve(neural_model::kInputCount * neural_model::kHiddenOne +
                      neural_model::kHiddenOne * neural_model::kHiddenTwo +
                      neural_model::kHiddenTwo +
                      neural_model::kHiddenTwo * neural_model::kPolicyCount);
      std::uint32_t buffer = 0;
      int bits = 0;
      for (const char character : neural_model::kEncodedWeights) {
        if (character == '=') {
          break;
        }
        const int value = base64_value(character);
        if (value < 0) {
          throw std::logic_error("invalid neural model encoding");
        }
        buffer = (buffer << 6U) | static_cast<std::uint32_t>(value);
        bits += 6;
        if (bits >= 8) {
          bits -= 8;
          decoded.push_back(static_cast<std::int8_t>(
              static_cast<std::uint8_t>((buffer >> bits) & 0xffU)));
        }
      }
      const std::size_t expected =
          neural_model::kInputCount * neural_model::kHiddenOne +
          neural_model::kHiddenOne * neural_model::kHiddenTwo +
          neural_model::kHiddenTwo +
          neural_model::kHiddenTwo * neural_model::kPolicyCount;
      if (decoded.size() != expected) {
        throw std::logic_error("neural model has an invalid size");
      }
      return decoded;
    }();
    return weights;
  }

  NeuralOutput neural_output(Player mover, bool include_policy = true) {
    ++stats_.neural_evaluations;
    if (topology_->edge_count() != neural_model::kEdgeCount ||
        topology_->vertex_count() != neural_model::kVertexCount) {
      throw std::logic_error("neural model topology mismatch");
    }
    const auto &weights = neural_weights();
    std::array<float, neural_model::kHiddenOne> hidden_one =
        neural_model::kB1;

    const auto accumulate_first_layer = [&](std::size_t input,
                                            float feature = 1.0F) {
      const std::size_t offset = input * neural_model::kHiddenOne;
      for (std::size_t hidden = 0;
           hidden < neural_model::kHiddenOne; ++hidden) {
        hidden_one[hidden] +=
            feature * static_cast<float>(weights[offset + hidden]) *
            neural_model::kW1Scale;
      }
    };

    for (std::size_t edge = 0; edge < topology_->edge_count(); ++edge) {
      if (!position_.edge_used(
              static_cast<detail::SearchTopology::EdgeIndex>(edge))) {
        continue;
      }
      const std::size_t normalized_edge =
          mover == Player::One ? edge : rotated_edges_[edge];
      accumulate_first_layer(normalized_edge);
    }

    calculate_goal_turn_distances(mover);
    for (std::size_t vertex = 0; vertex < topology_->vertex_count(); ++vertex) {
      const std::size_t physical_vertex =
          mover == Player::One ? vertex : rotated_vertices_[vertex];
      const int distance = std::clamp(distances_[physical_vertex], 0, 7);
      accumulate_first_layer(neural_model::kEdgeCount + vertex,
                             static_cast<float>(distance) / 7.0F);
    }
    const Point normalized_ball =
        mover == Player::One ? position_.ball() : rotated_point(position_.ball());
    accumulate_first_layer(neural_model::kEdgeCount +
                               neural_model::kVertexCount,
                           static_cast<float>(normalized_ball.x) /
                               static_cast<float>(topology_->config().width));
    accumulate_first_layer(neural_model::kEdgeCount +
                               neural_model::kVertexCount + 1U,
                           static_cast<float>(normalized_ball.y) /
                               static_cast<float>(topology_->config().height + 2));
    for (float &value : hidden_one) {
      value = std::max(value, 0.0F);
    }

    constexpr std::size_t kSecondLayerOffset =
        neural_model::kInputCount * neural_model::kHiddenOne;
    std::array<float, neural_model::kHiddenTwo> hidden_two =
        neural_model::kB2;
    for (std::size_t first = 0; first < neural_model::kHiddenOne;
         ++first) {
      for (std::size_t second = 0; second < neural_model::kHiddenTwo;
           ++second) {
        const std::size_t weight = kSecondLayerOffset +
                                   first * neural_model::kHiddenTwo +
                                   second;
        hidden_two[second] += hidden_one[first] *
                              static_cast<float>(weights[weight]) *
                              neural_model::kW2Scale;
      }
    }
    for (float &value : hidden_two) {
      value = std::max(value, 0.0F);
    }

    constexpr std::size_t kValueLayerOffset =
        kSecondLayerOffset +
        neural_model::kHiddenOne * neural_model::kHiddenTwo;
    constexpr std::size_t kPolicyLayerOffset =
        kValueLayerOffset + neural_model::kHiddenTwo;
    NeuralOutput output;
    output.value_logit = neural_model::kValueBias;
    output.policy_logits = neural_model::kPolicyBias;
    for (std::size_t hidden = 0; hidden < neural_model::kHiddenTwo;
         ++hidden) {
      output.value_logit +=
          hidden_two[hidden] *
          static_cast<float>(weights[kValueLayerOffset + hidden]) *
          neural_model::kValueScale;
      if (include_policy) {
        for (std::size_t policy = 0; policy < neural_model::kPolicyCount;
             ++policy) {
          output.policy_logits[policy] +=
              hidden_two[hidden] *
              static_cast<float>(
                  weights[kPolicyLayerOffset +
                          hidden * neural_model::kPolicyCount + policy]) *
              neural_model::kPolicyScale;
        }
      }
    }
    return output;
  }

  int evaluate() {
    ++stats_.leaf_evaluations;
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
      direct_goal = direct_goal ||
                    is_attacking_goal(rules, destination, mover);
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
    if (config_.neural_value_blend_percent != 0) {
      const float model_probability_score =
          std::tanh(neural_output(mover, false).value_logit * 0.5F);
      const int model_score = sign * static_cast<int>(
                                         std::lround(model_probability_score *
                                                     kMaximumEvaluation));
      score = (score * (100 - config_.neural_value_blend_percent) +
               model_score * config_.neural_value_blend_percent) /
              100;
    }
    return std::clamp(score, -kMaximumEvaluation, kMaximumEvaluation);
  }

  int cached_evaluate() {
    const detail::PositionKey key = boundary_key();
    ++stats_.evaluation_cache_probes;
    if (const std::optional<int> cached = evaluations_.find(key)) {
      ++stats_.evaluation_cache_hits;
      return *cached;
    }
    const int score = evaluate();
    evaluations_.store(key, score);
    return score;
  }

  OrderedMoveList ordered_moves(
      std::optional<Move> preferred = std::nullopt,
      bool use_neural_policy = false) {
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position_.legal_slots(slots);
    OrderedMoveList ordered;
    ordered.count = count;
    const Player mover = position_.to_move();
    const Point ball = position_.ball();
    const auto source = position_.ball_vertex();
    const RulesConfig &rules = topology_->config();
    std::optional<NeuralOutput> policy;
    if (use_neural_policy && config_.neural_policy_order_weight != 0) {
      policy = neural_output(mover);
      ++stats_.neural_policy_orderings;
    }

    for (std::uint8_t index = 0; index < count; ++index) {
      const std::uint8_t slot = slots[index];
      const Move move = position_.move_for_slot(slot);
      const bool extra_turn = position_.grants_extra_turn(slot);
      int score = extra_turn ? 10'000 : 0;
      const int progress = mover == Player::One ? ball.y - move.to.y
                                                : move.to.y - ball.y;
      score += progress * 100;
      score -= std::abs(move.to.x - topology_->config().width / 2) * 4;
      if (preferred.has_value() && move == *preferred) {
        score += 2'000'000;
      }
      if (policy.has_value()) {
        const Point delta{move.to.x - ball.x, move.to.y - ball.y};
        std::size_t direction = 0;
        while (direction < kDirectionDeltas.size() &&
               kDirectionDeltas[direction] != delta) {
          ++direction;
        }
        if (direction == kDirectionDeltas.size()) {
          throw std::logic_error("ordered move is not an adjacent direction");
        }
        if (mover == Player::Two) {
          direction = (direction + 4U) & 7U;
        }
        score += static_cast<int>(std::lround(
            policy->policy_logits[direction] *
            static_cast<float>(config_.neural_policy_order_weight)));
      }

      const detail::SearchTopology::Arc &played_arc =
          topology_->adjacency(source, mover).arcs[slot];
      if (is_goal_point(rules, move.to)) {
        const Player winning_player =
            is_attacking_goal(rules, move.to, Player::One) ? Player::One
                                                           : Player::Two;
        score += winning_player == mover ? 1'000'000 : -1'000'000;
      } else {
        const Player child_player = extra_turn ? mover : opponent(mover);
        const detail::SearchTopology::Adjacency &child_adjacency =
            topology_->adjacency(played_arc.destination, child_player);
        int mobility = 0;
        for (std::uint8_t child = 0; child < child_adjacency.count; ++child) {
          const auto edge = child_adjacency.arcs[child].edge;
          mobility += edge != played_arc.edge && !position_.edge_used(edge)
                          ? 1
                          : 0;
        }
        if (mobility == 0) {
          const Player blocked_player =
              rules.blocked_rule == BlockedRule::MoverLoses ? mover
                                                             : child_player;
          score += opponent(blocked_player) == mover ? 1'000'000
                                                      : -1'000'000;
        } else {
          score += child_player == mover ? mobility * 12 : -mobility * 12;
        }
      }
      ordered.values[index] = OrderedMove{slot, move, score};
    }

    const auto better = [mover](const OrderedMove &left,
                                const OrderedMove &right) {
      if (left.score != right.score) {
        return left.score > right.score;
      }
      if (left.move.to.y != right.move.to.y) {
        return mover == Player::One ? left.move.to.y < right.move.to.y
                                    : left.move.to.y > right.move.to.y;
      }
      return left.move.to.x < right.move.to.x;
    };
    for (std::uint8_t index = 1; index < count; ++index) {
      const OrderedMove value = ordered.values[index];
      std::uint8_t insertion = index;
      while (insertion > 0 &&
             better(value, ordered.values[insertion - 1U])) {
        ordered.values[insertion] = ordered.values[insertion - 1U];
        --insertion;
      }
      ordered.values[insertion] = value;
    }
    return ordered;
  }

  std::vector<Move> fallback_action() {
    const std::size_t root_depth = position_.undo_depth();
    const Player mover = position_.to_move();
    std::vector<Move> action;
    try {
      while (!position_.is_terminal() && position_.to_move() == mover) {
        const OrderedMoveList moves = ordered_moves();
        if (moves.count == 0) {
          throw std::logic_error("fallback reached a state without moves");
        }
        action.push_back(moves.values[0].move);
        position_.make_move(moves.values[0].slot);
      }
    } catch (...) {
      position_.unmake_to(root_depth);
      throw;
    }
    position_.unmake_to(root_depth);
    return action;
  }

  void record_completed_action(int score, Player root_mover) {
    ++stats_.completed_actions;
    stats_.max_action_edges = std::max<std::uint32_t>(
        stats_.max_action_edges,
        static_cast<std::uint32_t>(current_action_.size()));
    const bool better =
        !captured_any_ ||
        (root_mover == Player::One ? score > captured_score_
                                   : score < captured_score_);
    if (better) {
      captured_any_ = true;
      captured_score_ = score;
      captured_action_ = current_action_;
    }
  }

  bool reuse_cached_root_action(Player mover, std::uint32_t remaining_depth,
                                int score) {
    const std::size_t base_position_depth = position_.undo_depth();
    const std::size_t base_action_size = current_action_.size();
    try {
      while (!position_.is_terminal() && position_.to_move() == mover) {
        std::array<std::uint8_t, detail::kMaximumMoves> slots{};
        const std::uint8_t count = position_.legal_slots(slots);
        if (count == 0) {
          position_.unmake_to(base_position_depth);
          current_action_.resize(base_action_size);
          return false;
        }

        std::uint8_t slot = slots[0];
        if (count > 1) {
          const TranspositionEntry *entry = table_.find(boundary_key());
          if (entry == nullptr || entry->depth < remaining_depth ||
              entry->bound != ScoreBound::Exact ||
              !position_.slot_for_move(entry->best_move, slot)) {
            position_.unmake_to(base_position_depth);
            current_action_.resize(base_action_size);
            return false;
          }
        }
        current_action_.push_back(position_.move_for_slot(slot));
        position_.make_move(slot);
      }
      record_completed_action(score, mover);
      ++stats_.root_transposition_reuses;
      position_.unmake_to(base_position_depth);
      current_action_.resize(base_action_size);
      return true;
    } catch (...) {
      position_.unmake_to(base_position_depth);
      current_action_.resize(base_action_size);
      throw;
    }
  }

  int explore_continuation(Player mover, std::uint32_t remaining_depth,
                           std::uint32_t turn_ply, int alpha, int beta,
                           bool capture_root) {
    const std::size_t base_position_depth = position_.undo_depth();
    const std::size_t base_action_size = current_action_.size();
    try {
      visit_node();
      const detail::PositionKey key = boundary_key();
      const int original_alpha = alpha;
      const int original_beta = beta;
      std::optional<Move> preferred;
      if (const TranspositionEntry *entry = probe(key)) {
        ++stats_.continuation_transposition_hits;
        preferred = entry->best_move;
        if (capture_root && config_.root_transposition_pruning &&
            entry->depth >= remaining_depth &&
            entry->bound == ScoreBound::Exact &&
            reuse_cached_root_action(mover, remaining_depth, entry->score)) {
          ++stats_.transposition_cutoffs;
          return entry->score;
        }
        if (entry->depth >= remaining_depth) {
          if (entry->bound == ScoreBound::Exact) {
            const bool cannot_improve =
                mover == Player::One ? entry->score <= alpha
                                     : entry->score >= beta;
            if (!capture_root ||
                (config_.root_transposition_pruning && cannot_improve)) {
              ++stats_.transposition_cutoffs;
              return entry->score;
            }
          } else if (!capture_root || config_.root_transposition_pruning) {
            if (entry->bound == ScoreBound::Lower) {
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
      }

      OrderedMoveList moves = ordered_moves(preferred);
      std::optional<Move> first_forced_move;
      while (moves.count == 1) {
        const OrderedMove forced = moves.values[0];
        if (!first_forced_move.has_value()) {
          first_forced_move = forced.move;
        }
        position_.make_move(forced.slot);
        current_action_.push_back(forced.move);
        ++stats_.forced_edges;
        if (position_.is_terminal() || position_.to_move() != mover) {
          const int score =
              search(remaining_depth - 1U, turn_ply + 1U, alpha, beta);
          if (capture_root) {
            record_completed_action(score, mover);
          }
          ScoreBound bound = ScoreBound::Exact;
          if (score <= original_alpha) {
            bound = ScoreBound::Upper;
          } else if (score >= original_beta) {
            bound = ScoreBound::Lower;
          }
          store(key, remaining_depth, score, *first_forced_move, bound);
          position_.unmake_to(base_position_depth);
          current_action_.resize(base_action_size);
          return score;
        }
        visit_node();
        moves = ordered_moves();
      }

      const bool maximizing = mover == Player::One;
      int best_score = maximizing ? -kInfinity : kInfinity;
      std::optional<Move> best_move = first_forced_move;
      bool found = false;
      for (std::uint8_t index = 0; index < moves.count; ++index) {
        const OrderedMove &candidate = moves.values[index];
        int score = 0;
        {
          ScopedMove played(position_, candidate.slot);
          current_action_.push_back(candidate.move);
          if (position_.is_terminal() || position_.to_move() != mover) {
            score = search(remaining_depth - 1U, turn_ply + 1U, alpha, beta);
            if (capture_root) {
              record_completed_action(score, mover);
            }
          } else {
            score = explore_continuation(mover, remaining_depth, turn_ply,
                                         alpha, beta, capture_root);
          }
          current_action_.pop_back();
        }
        if (!found ||
            (maximizing ? score > best_score : score < best_score)) {
          best_score = score;
          if (!first_forced_move.has_value()) {
            best_move = candidate.move;
          }
        }
        found = true;
        if (maximizing) {
          alpha = std::max(alpha, best_score);
        } else {
          beta = std::min(beta, best_score);
        }
        if (reached_immediate_win_bound(best_score, mover, turn_ply)) {
          ++stats_.terminal_bound_cutoffs;
          break;
        }
        if (alpha >= beta) {
          ++stats_.cutoffs;
          break;
        }
      }
      if (!found || !best_move.has_value()) {
        throw std::logic_error("continuation node has no legal edge");
      }
      ScoreBound bound = ScoreBound::Exact;
      if (best_score <= original_alpha) {
        bound = ScoreBound::Upper;
      } else if (best_score >= original_beta) {
        bound = ScoreBound::Lower;
      }
      store(key, remaining_depth, best_score, *best_move, bound);
      position_.unmake_to(base_position_depth);
      current_action_.resize(base_action_size);
      return best_score;
    } catch (...) {
      position_.unmake_to(base_position_depth);
      current_action_.resize(base_action_size);
      throw;
    }
  }

  int seed_root_endpoint(const OrderedMove &first, std::uint32_t turn_ply,
                         int alpha, int beta, Player mover) {
    const std::size_t base_position_depth = position_.undo_depth();
    const std::size_t base_action_size = current_action_.size();
    try {
      position_.make_move(first.slot);
      current_action_.push_back(first.move);
      while (!position_.is_terminal() && position_.to_move() == mover) {
        visit_node();
        const OrderedMoveList moves = ordered_moves();
        if (moves.count == 0) {
          throw std::logic_error("root seed reached a state without moves");
        }
        position_.make_move(moves.values[0].slot);
        current_action_.push_back(moves.values[0].move);
      }
      const int score = search(0, turn_ply + 1U, alpha, beta);
      record_completed_action(score, mover);
      ++stats_.root_seed_actions;
      position_.unmake_to(base_position_depth);
      current_action_.resize(base_action_size);
      return score;
    } catch (...) {
      position_.unmake_to(base_position_depth);
      current_action_.resize(base_action_size);
      throw;
    }
  }

  ActionSearchResult search_actions(
      std::uint32_t remaining_depth, std::uint32_t turn_ply, int alpha,
      int beta, std::optional<Move> preferred, bool capture_root) {
    const Player mover = position_.to_move();
    const bool maximizing = mover == Player::One;
    int best_score = maximizing ? -kInfinity : kInfinity;
    std::optional<Move> best_move;
    OrderedMoveList moves =
        ordered_moves(preferred, capture_root || turn_ply <= 1U);

    if (capture_root && remaining_depth == 1U &&
        config_.root_seed_endpoints) {
      for (std::uint8_t index = 0; index < moves.count; ++index) {
        const int score = seed_root_endpoint(moves.values[index], turn_ply,
                                             alpha, beta, mover);
        moves.values[index].score = mover == Player::One ? score : -score;
        if (reached_immediate_win_bound(score, mover, turn_ply)) {
          ++stats_.terminal_bound_cutoffs;
          return ActionSearchResult{score, captured_action_.front()};
        }
      }
      for (std::uint8_t index = 1; index < moves.count; ++index) {
        const OrderedMove value = moves.values[index];
        std::uint8_t insertion = index;
        while (insertion > 0 &&
               value.score > moves.values[insertion - 1U].score) {
          moves.values[insertion] = moves.values[insertion - 1U];
          --insertion;
        }
        moves.values[insertion] = value;
      }
    }

    for (std::uint8_t index = 0; index < moves.count; ++index) {
      const OrderedMove &candidate = moves.values[index];
      int score = 0;
      {
        ScopedMove played(position_, candidate.slot);
        current_action_.push_back(candidate.move);
        if (position_.is_terminal() || position_.to_move() != mover) {
          score = search(remaining_depth - 1U, turn_ply + 1U, alpha, beta);
          if (capture_root) {
            record_completed_action(score, mover);
          }
        } else {
          score = explore_continuation(mover, remaining_depth, turn_ply,
                                       alpha, beta, capture_root);
        }
        current_action_.pop_back();
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
      if (reached_immediate_win_bound(best_score, mover, turn_ply)) {
        ++stats_.terminal_bound_cutoffs;
        break;
      }
      if (!capture_root && alpha >= beta) {
        ++stats_.cutoffs;
        break;
      }
    }
    if (!best_move.has_value()) {
      throw std::logic_error("turn boundary has no legal action");
    }
    return ActionSearchResult{best_score, *best_move};
  }

  const TranspositionEntry *probe(detail::PositionKey key) {
    ++stats_.transposition_probes;
    const TranspositionEntry *entry = table_.find(key);
    if (entry != nullptr) {
      ++stats_.transposition_hits;
    }
    return entry;
  }

  void store(detail::PositionKey key, std::uint32_t depth, int score,
             Move best_move, ScoreBound bound) {
    if (table_.store(key, depth, score, best_move, bound)) {
      ++stats_.transposition_stores;
    }
  }

  int search(std::uint32_t remaining_depth, std::uint32_t turn_ply,
             int alpha, int beta) {
    visit_node();
    if (position_.is_terminal()) {
      return terminal_score(turn_ply);
    }
    if (remaining_depth == 0) {
      return cached_evaluate();
    }

    const detail::PositionKey key = boundary_key();
    const int original_alpha = alpha;
    const int original_beta = beta;
    std::optional<Move> preferred;
    if (const TranspositionEntry *entry = probe(key)) {
      preferred = entry->best_move;
      if (entry->depth >= remaining_depth) {
        if (entry->bound == ScoreBound::Exact) {
          ++stats_.transposition_cutoffs;
          return entry->score;
        }
        if (entry->bound == ScoreBound::Lower) {
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

    const ActionSearchResult result = search_actions(
        remaining_depth, turn_ply, alpha, beta, preferred, false);
    ScoreBound bound = ScoreBound::Exact;
    if (result.score <= original_alpha) {
      bound = ScoreBound::Upper;
    } else if (result.score >= original_beta) {
      bound = ScoreBound::Lower;
    }
    store(key, remaining_depth, result.score, result.first_move, bound);
    return result.score;
  }

  RootResult search_root(std::uint32_t depth) {
    visit_node();
    const detail::PositionKey key = boundary_key();
    std::optional<Move> preferred;
    if (const TranspositionEntry *entry = probe(key)) {
      preferred = entry->best_move;
    }
    current_action_.clear();
    captured_action_.clear();
    captured_any_ = false;
    const ActionSearchResult result =
        search_actions(depth, 0, -kInfinity, kInfinity, preferred, true);
    if (!captured_any_ || captured_action_.empty()) {
      throw std::logic_error("root search did not complete a legal action");
    }
    store(key, depth, result.score, captured_action_.front(),
          ScoreBound::Exact);
    return RootResult{result.score, captured_action_};
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
  const Point delta{to.x - from.x, to.y - from.y};
  for (std::size_t index = 0; index < kDirectionDeltas.size(); ++index) {
    if (kDirectionDeltas[index] == delta) {
      return static_cast<char>('0' + index);
    }
  }
  throw std::invalid_argument("move is not an adjacent direction");
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

bool lookup_replay_correction(int player_id, std::string_view transcript,
                              std::string &encoded) {
  if (player_id == 1 && transcript == "1") {
    encoded = "1";
    return true;
  }
  if (player_id == 0 && transcript == "0/6") {
    encoded = "5";
    return true;
  }

  const std::size_t completed_turns =
      transcript.empty()
          ? 0U
          : 1U + static_cast<std::size_t>(
                     std::count(transcript.begin(), transcript.end(), '/'));
  for (const replay_book::Replay &replay : replay_book::kReplays) {
    if (replay.player_id != player_id ||
        completed_turns < replay.first_turn ||
        completed_turns % 2U != static_cast<std::size_t>(player_id)) {
      continue;
    }

    const std::string_view recorded = replay.transcript;
    std::size_t action_begin = 0;
    if (transcript.empty()) {
      if (replay.first_turn != 0) {
        continue;
      }
    } else {
      if (recorded.size() <= transcript.size() ||
          !recorded.starts_with(transcript) ||
          recorded[transcript.size()] != '/') {
        continue;
      }
      action_begin = transcript.size() + 1U;
    }
    const std::size_t action_end = recorded.find('/', action_begin);
    encoded.assign(recorded.substr(action_begin, action_end - action_begin));
    if (!encoded.empty()) {
      return true;
    }
  }
  return false;
}

bool try_replay_correction(GameState &state, int player_id,
                           std::string_view transcript,
                           std::string &encoded) {
  std::string candidate;
  if (!lookup_replay_correction(player_id, transcript, candidate)) {
    return false;
  }

  GameState next = state;
  try {
    apply_encoded_turn(next, candidate);
  } catch (const std::exception &) {
    return false;
  }

  state = std::move(next);
  encoded = std::move(candidate);
  return true;
}

std::string choose_complete_turn(GameState &state,
                                 std::uint32_t search_time_ms) {
  SearchConfig config;
  config.neural_value_blend_percent = 10;
  config.neural_policy_order_weight = 0;
  config.absolute_deadline =
      SearchClock::now() + std::chrono::milliseconds(search_time_ms);
  CompleteTurnSearch search(state, config);
  const std::vector<Move> action = search.run();
  if (action.empty()) {
    throw std::logic_error("search returned an empty action");
  }

  const Player mover = state.to_move;
  std::string encoded;
  encoded.reserve(action.size());
  for (const Move move : action) {
    if (is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("search action continues after turn end");
    }
    encoded.push_back(encode_direction(state.ball, move.to));
    state = apply_move(state, move);
  }
  if (!is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("search action ends before rebound completion");
  }
  return encoded;
}

}  // namespace papersoccer::selfplay_nn

#ifndef PAPER_SOCCER_SELFPLAY_NN_NO_MAIN
int main() {
  using namespace papersoccer;
  using namespace papersoccer::selfplay_nn;

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
  std::string transcript;

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
        if (!transcript.empty()) {
          transcript.push_back('/');
        }
        transcript += opponent_move;
      }
      if (is_terminal(state)) {
        return 0;
      }
      if (state.to_move != me) {
        return 1;
      }

      const std::uint32_t time_ms =
          first_execution ? kFirstSearchTimeMs : kLaterSearchTimeMs;
      std::string action;
      if (!try_replay_correction(state, player_id, transcript, action)) {
        action = choose_complete_turn(state, time_ms);
      }
      std::cout << action << std::endl;
      if (!transcript.empty()) {
        transcript.push_back('/');
      }
      transcript += action;
      first_execution = false;
    } catch (const std::exception &) {
      return 1;
    }
  }
  return 0;
}
#endif
