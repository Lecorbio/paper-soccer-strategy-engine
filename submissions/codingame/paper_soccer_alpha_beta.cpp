// Paper Soccer AlphaBeta submission for CodinGame.
// Paste this complete file into the C++ editor.
// Rebuild with: node submissions/codingame/generate_submission.mjs

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

// --- include/papersoccer/types.hpp ---
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

// --- include/papersoccer/geometry.hpp ---
namespace papersoccer {

bool is_regular_point(const RulesConfig &config, Point point);

bool is_goal_point(const RulesConfig &config, Point point);

bool is_attacking_goal(const RulesConfig &config, Point point, Player player);

bool is_boundary_point(const RulesConfig &config, Point point);

bool is_neighbor(Point from, Point to);

bool is_forbidden_boundary_segment(const RulesConfig &config, Segment segment);

std::vector<Point> neighbors(const RulesConfig &config, Point from, Player player);

}  // namespace papersoccer

// --- include/papersoccer/rules.hpp ---
namespace papersoccer {

GameState make_initial_state(const RulesConfig &config = {});

std::vector<Move> legal_moves(const GameState &state);

bool grants_extra_turn(const GameState &before, Point destination);

GameState apply_move(const GameState &state, Move move);

bool is_terminal(const GameState &state);

std::optional<Player> winner(const GameState &state);

}  // namespace papersoccer

// --- include/papersoccer/bot.hpp ---
namespace papersoccer {

enum class BotKind {
  Random,
  Mcts,
  AlphaBeta,
};

std::string_view bot_kind_name(BotKind kind) noexcept;

class Bot {
 public:
  virtual ~Bot() = default;

  virtual std::string_view name() const noexcept = 0;
  virtual Move choose_move(const GameState &state) = 0;
};

class RandomBot final : public Bot {
 public:
  static constexpr std::uint64_t default_seed() noexcept { return 0xC0FFEE1234ULL; }

  explicit RandomBot(std::uint64_t seed = default_seed()) noexcept;

  std::string_view name() const noexcept override;
  Move choose_move(const GameState &state) override;

 private:
  std::uint64_t state_;

  std::uint64_t next_random() noexcept;
};

struct AlphaBetaConfig {
  static constexpr std::uint32_t maximum_turn_depth{64};
  static constexpr std::uint32_t maximum_search_plies{512};

  std::uint32_t max_turn_depth{6};
  std::uint64_t max_nodes{100'000};
  std::size_t transposition_table_entries{65'536};
  std::uint32_t max_search_plies{12};
  std::uint32_t max_time_ms{0};
};

enum class AlphaBetaScoreBound {
  Exact,
  Lower,
  Upper,
};

struct AlphaBetaRootMove {
  Move move{};
  int score{};
  AlphaBetaScoreBound bound{AlphaBetaScoreBound::Exact};
};

struct AlphaBetaSearchStats {
  std::uint32_t completed_turn_depth{};
  std::uint32_t attempted_turn_depth{};
  std::uint64_t nodes{};
  std::uint64_t leaf_evaluations{};
  std::uint64_t terminal_nodes{};
  std::uint64_t cutoffs{};
  std::uint64_t transposition_probes{};
  std::uint64_t transposition_hits{};
  std::uint64_t transposition_cutoffs{};
  std::uint64_t transposition_stores{};
  std::uint64_t physical_ply_cutoffs{};
  std::uint32_t max_physical_ply{};
  int root_score{};
  bool budget_exhausted{};
  std::vector<Move> principal_variation{};
  std::vector<AlphaBetaRootMove> root_moves{};
};

class AlphaBetaBot final : public Bot {
 public:
  explicit AlphaBetaBot(AlphaBetaConfig config = {});

  std::string_view name() const noexcept override;
  Move choose_move(const GameState &state) override;
  const AlphaBetaConfig &config() const noexcept;
  const AlphaBetaSearchStats &last_search_stats() const noexcept;

 private:
  AlphaBetaConfig config_;
  AlphaBetaSearchStats last_search_stats_{};
};

struct BotConfig {
  BotKind kind{BotKind::Random};
  std::uint64_t seed{RandomBot::default_seed()};
  std::uint32_t mcts_iterations{2000};
  std::uint32_t alpha_beta_depth{6};
  std::uint64_t alpha_beta_max_nodes{100'000};
  std::size_t alpha_beta_transposition_table_entries{65'536};
  std::uint32_t alpha_beta_max_search_plies{12};
};

enum class MctsRolloutPolicy {
  Uniform,
  Tactical,
};

enum class MctsLeafPolicy {
  RolloutOnly,
  TacticalQuiescence,
};

struct MctsConfig {
  static constexpr std::uint32_t default_quiescence_max_depth{8};
  static constexpr std::uint32_t default_quiescence_max_nodes{256};
  static constexpr std::uint32_t maximum_quiescence_max_depth{64};
  static constexpr std::uint32_t maximum_quiescence_max_nodes{1'000'000};

  std::uint64_t seed{RandomBot::default_seed()};
  std::uint32_t iterations{2000};
  double exploration{1.4142135623730951};
  MctsRolloutPolicy rollout_policy{MctsRolloutPolicy::Tactical};
  bool reuse_tree{true};
  std::size_t max_nodes{65536};
  MctsLeafPolicy leaf_policy{MctsLeafPolicy::RolloutOnly};
  std::uint32_t quiescence_max_depth{default_quiescence_max_depth};
  std::uint32_t quiescence_max_nodes{default_quiescence_max_nodes};
};

struct SearchStats {
  std::uint32_t iterations{};
  std::size_t nodes{};
  std::uint64_t simulated_plies{};
  double root_value{};
  std::uint64_t total_root_visits{};
  std::uint64_t reused_visits{};
  std::uint32_t max_depth{};
  std::size_t proven_nodes{};
  std::optional<Player> proven_winner{};
  std::uint64_t rebuild_count{};
  bool expansion_saturated{};
  std::uint64_t tactical_probes{};
  std::uint64_t tactical_nodes{};
  std::uint64_t tactical_solved_positions{};
  std::uint64_t tactical_depth_cutoffs{};
  std::uint64_t tactical_node_cutoffs{};
  std::uint32_t max_tactical_depth{};
};

class MctsSearch {
 public:
  MctsSearch(const GameState &root, std::uint64_t seed,
             double exploration = 1.4142135623730951);
  MctsSearch(const GameState &root, const MctsConfig &config);
  ~MctsSearch();

  MctsSearch(const MctsSearch &) = delete;
  MctsSearch &operator=(const MctsSearch &) = delete;

  void run_iterations(std::uint32_t count);
  Move best_move() const;
  SearchStats stats() const noexcept;

 private:
  class Impl;
  friend class MctsBot;

  explicit MctsSearch(std::unique_ptr<Impl> impl);
  std::unique_ptr<MctsSearch> clone() const;
  bool rebase(const GameState &root);
  void set_rebuild_count(std::uint64_t count) noexcept;

  std::unique_ptr<Impl> impl_;
};

class MctsBot final : public Bot {
 public:
  explicit MctsBot(MctsConfig config = {});
  MctsBot(const MctsBot &other);
  MctsBot &operator=(const MctsBot &other);
  MctsBot(MctsBot &&) noexcept = default;
  MctsBot &operator=(MctsBot &&) noexcept = default;

  std::string_view name() const noexcept override;
  Move choose_move(const GameState &state) override;
  SearchStats last_search_stats() const noexcept;

 private:
  MctsConfig config_;
  SearchStats last_search_stats_{};
  std::unique_ptr<MctsSearch> search_{};
  std::uint64_t rebuild_count_{};
};

std::unique_ptr<Bot> make_bot(const BotConfig &config);

}  // namespace papersoccer

// --- src/bots/mcts_internal.hpp ---
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

// --- src/core/geometry.cpp ---
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

// --- src/core/rules.cpp ---
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

// --- src/bots/alpha_beta.cpp ---
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
  AlphaBetaSearch(const GameState &state, const AlphaBetaConfig &config)
      : config_(config),
        deadline_(search_deadline(config)),
        topology_(std::make_shared<detail::SearchTopology>(state.config)),
        position_(topology_, state), table_(config.transposition_table_entries),
        distances_(topology_->vertex_count(), -1),
        queue_(topology_->vertex_count()) {
    validate_config(config_);
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
  std::optional<SearchClock::time_point> deadline_{};
  std::shared_ptr<const detail::SearchTopology> topology_{};
  detail::SearchPosition position_;
  TranspositionTable table_;
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

AlphaBetaBot::AlphaBetaBot(AlphaBetaConfig config) : config_(config) {
  validate_config(config_);
}

std::string_view AlphaBetaBot::name() const noexcept { return "AlphaBetaBot"; }

Move AlphaBetaBot::choose_move(const GameState &state) {
  AlphaBetaSearch search(state, config_);
  const Move move = search.run();
  last_search_stats_ = search.stats();
  return move;
}

const AlphaBetaConfig &AlphaBetaBot::config() const noexcept { return config_; }

const AlphaBetaSearchStats &AlphaBetaBot::last_search_stats() const noexcept {
  return last_search_stats_;
}

}  // namespace papersoccer

// --- submissions/codingame/alpha_beta_adapter.cpp ---
namespace papersoccer::codingame {

constexpr std::uint32_t kFirstSearchTimeMs = 850;
constexpr std::uint32_t kLaterSearchTimeMs = 150;

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
    throw std::invalid_argument("move direction must be between zero and seven");
  }
  const Point delta = kDirectionDeltas[static_cast<std::size_t>(direction - '0')];
  return Move{{from.x + delta.x, from.y + delta.y}};
}

char encode_direction(Point from, Point to) {
  const Point delta{to.x - from.x, to.y - from.y};
  for (std::size_t index = 0; index < kDirectionDeltas.size(); ++index) {
    if (kDirectionDeltas[index] == delta) {
      return static_cast<char>('0' + index);
    }
  }
  throw std::invalid_argument("move is not an adjacent paper-soccer direction");
}

bool contains_move(const std::vector<Move> &moves, Move candidate) {
  return std::find(moves.begin(), moves.end(), candidate) != moves.end();
}

int fallback_score(const GameState &state, Move move) {
  const Player mover = state.to_move;
  const GameState next = apply_move(state, move);
  if (is_terminal(next)) {
    return winner(next) == mover ? 1'000'000 : -1'000'000;
  }

  const int progress = mover == Player::One ? state.ball.y - move.to.y
                                             : move.to.y - state.ball.y;
  const int center = state.config.width / 2;
  const int center_alignment = center - std::abs(move.to.x - center);
  const int mobility = static_cast<int>(legal_moves(next).size());
  const int mobility_score = next.to_move == mover ? mobility : -mobility;
  const int continuation = next.to_move == mover ? 25 : 0;
  return progress * 100 + center_alignment * 4 + mobility_score * 3 +
         continuation;
}

Move choose_fallback(const GameState &state,
                     const std::vector<Move> &moves) {
  if (moves.empty()) {
    throw std::invalid_argument("cannot choose a fallback without a legal move");
  }

  Move best = moves.front();
  int best_score = fallback_score(state, best);
  for (std::size_t index = 1; index < moves.size(); ++index) {
    const int score = fallback_score(state, moves[index]);
    if (score > best_score) {
      best = moves[index];
      best_score = score;
    }
  }
  return best;
}

std::string complete_turn_from_plan(GameState &state,
                                    const std::vector<Move> &plan) {
  if (is_terminal(state)) {
    throw std::invalid_argument("cannot compose a move from a terminal state");
  }

  const Player mover = state.to_move;
  std::size_t plan_index = 0;
  bool following_plan = true;
  std::string encoded;

  while (!is_terminal(state) && state.to_move == mover) {
    const std::vector<Move> moves = legal_moves(state);
    if (moves.empty()) {
      throw std::logic_error("an in-progress state must have a legal move");
    }

    std::optional<Move> selected;
    if (following_plan && plan_index < plan.size() &&
        contains_move(moves, plan[plan_index])) {
      selected = plan[plan_index++];
    } else {
      following_plan = false;
      selected = choose_fallback(state, moves);
    }

    encoded.push_back(encode_direction(state.ball, selected->to));
    state = apply_move(state, *selected);
  }

  if (encoded.empty()) {
    throw std::logic_error("a complete turn must contain at least one move");
  }
  return encoded;
}

std::string choose_complete_turn(GameState &state,
                                 std::uint32_t search_time_ms) {
  AlphaBetaConfig config;
  config.max_turn_depth = AlphaBetaConfig::maximum_turn_depth;
  config.max_nodes = 5'000'000;
  config.transposition_table_entries = 65'536;
  config.max_search_plies = 10;
  config.max_time_ms = search_time_ms;

  AlphaBetaBot bot(config);
  const Move root_move = bot.choose_move(state);
  std::vector<Move> plan = bot.last_search_stats().principal_variation;
  if (plan.empty() || !(plan.front() == root_move)) {
    plan = {root_move};
  }
  return complete_turn_from_plan(state, plan);
}

void apply_encoded_turn(GameState &state, std::string_view encoded) {
  if (encoded.empty()) {
    throw std::invalid_argument("a turn cannot be empty");
  }

  GameState next = state;
  const Player mover = next.to_move;
  for (const char direction : encoded) {
    if (is_terminal(next) || next.to_move != mover) {
      throw std::invalid_argument("turn contains moves after it has ended");
    }
    next = apply_move(next, decode_direction(next.ball, direction));
  }

  if (!is_terminal(next) && next.to_move == mover) {
    throw std::invalid_argument("turn ends before a required rebound move");
  }
  state = std::move(next);
}

}  // namespace papersoccer::codingame

#ifndef PAPER_SOCCER_CODINGAME_NO_MAIN
int main() {
  using namespace papersoccer;
  using namespace papersoccer::codingame;

  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  int player_id = -1;
  if (!(std::cin >> player_id)) {
    return 0;
  }

  Player me;
  try {
    me = player_for_id(player_id);
  } catch (const std::invalid_argument &) {
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
    if (opponent_move_length != static_cast<int>(opponent_move.size())) {
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

      const std::uint32_t search_time =
          first_execution ? kFirstSearchTimeMs : kLaterSearchTimeMs;
      const std::string action = choose_complete_turn(state, search_time);
      std::cout << action << std::endl;
      first_execution = false;
    } catch (const std::exception &) {
      return 1;
    }
  }
  return 0;
}
#endif
