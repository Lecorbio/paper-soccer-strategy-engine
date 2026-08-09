#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
#include <utility>
#include <vector>

#include "mcts_internal.hpp"
#include "papersoccer/rules.hpp"
#include "replay_book.hpp"
#include "replay_value_model.hpp"

namespace papersoccer::turn_action_v2 {

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
enum class ReboundOutcome { Unknown, Win, Loss };

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
  int replay_value_blend_percent{};
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
  std::uint64_t terminal_bound_cutoffs{};
  std::uint64_t rebound_goal_probes{};
  std::uint64_t rebound_goal_hits{};
  std::uint64_t rebound_loss_hits{};
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
    if (config_.replay_value_blend_percent < 0 ||
        config_.replay_value_blend_percent > 100) {
      throw std::invalid_argument("invalid replay value blend percentage");
    }
    if (config_.replay_value_blend_percent != 0) {
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

    std::vector<Move> rebound_goal;
    if (analyze_rebound_component(&rebound_goal) == ReboundOutcome::Win) {
      stats_.completed_actions = 1;
      stats_.max_action_edges =
          static_cast<std::uint32_t>(rebound_goal.size());
      stats_.root_score = immediate_win_score(position_.to_move(), 0);
      return rebound_goal;
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

  ReboundOutcome analyze_rebound_component(std::vector<Move> *action) {
    ++stats_.rebound_goal_probes;
    if (action != nullptr) {
      action->clear();
    }
    std::fill(distances_.begin(), distances_.end(), -1);
    const auto start = position_.ball_vertex();
    distances_[start] = static_cast<int>(start);
    std::size_t head = 0;
    std::size_t tail = 0;
    queue_[tail++] = start;
    const Player mover = position_.to_move();
    const RulesConfig &rules = topology_->config();
    bool safe_handoff = false;

    while (head < tail) {
      const auto vertex = queue_[head++];
      const auto &adjacency = topology_->adjacency(vertex, mover);
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const auto &arc = adjacency.arcs[index];
        if (position_.edge_used(arc.edge)) {
          continue;
        }
        const Point destination = topology_->point(arc.destination);
        if (is_attacking_goal(rules, destination, mover)) {
          ++stats_.rebound_goal_hits;
          if (action != nullptr) {
            distances_[arc.destination] = static_cast<int>(vertex);
            for (auto current = arc.destination; current != start;
                 current = static_cast<detail::SearchTopology::VertexIndex>(
                     distances_[current])) {
              action->push_back(Move{topology_->point(current)});
            }
            std::reverse(action->begin(), action->end());
          }
          return ReboundOutcome::Win;
        }
        if (is_goal_point(rules, destination) ||
            distances_[arc.destination] >= 0) {
          continue;
        }
        if (!is_boundary_point(rules, destination) &&
            !position_.vertex_visited(arc.destination)) {
          const auto &handoff =
              topology_->adjacency(arc.destination, opponent(mover));
          for (std::uint8_t child = 0; child < handoff.count; ++child) {
            const auto edge = handoff.arcs[child].edge;
            if (edge != arc.edge && !position_.edge_used(edge)) {
              safe_handoff = true;
              break;
            }
          }
          continue;
        }
        distances_[arc.destination] = static_cast<int>(vertex);
        queue_[tail++] = arc.destination;
      }
    }
    if (!safe_handoff && rules.blocked_rule == BlockedRule::MoverLoses) {
      ++stats_.rebound_loss_hits;
      return ReboundOutcome::Loss;
    }
    return ReboundOutcome::Unknown;
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

  static const std::vector<std::int8_t> &replay_value_weights() {
    static const std::vector<std::int8_t> weights = [] {
      std::vector<std::int8_t> decoded;
      decoded.reserve(replay_value_model::kInputCount *
                          replay_value_model::kHiddenOne +
                      replay_value_model::kHiddenOne *
                          replay_value_model::kHiddenTwo +
                      replay_value_model::kHiddenTwo);
      std::uint32_t buffer = 0;
      int bits = 0;
      for (const char character : replay_value_model::kEncodedWeights) {
        if (character == '=') {
          break;
        }
        const int value = base64_value(character);
        if (value < 0) {
          throw std::logic_error("invalid replay value model encoding");
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
          replay_value_model::kInputCount * replay_value_model::kHiddenOne +
          replay_value_model::kHiddenOne * replay_value_model::kHiddenTwo +
          replay_value_model::kHiddenTwo;
      if (decoded.size() != expected) {
        throw std::logic_error("replay value model has an invalid size");
      }
      return decoded;
    }();
    return weights;
  }

  float replay_value_logit(Player mover) {
    if (topology_->edge_count() != replay_value_model::kEdgeCount ||
        topology_->vertex_count() != replay_value_model::kVertexCount) {
      throw std::logic_error("replay value model topology mismatch");
    }
    const auto &weights = replay_value_weights();
    std::array<float, replay_value_model::kHiddenOne> hidden_one =
        replay_value_model::kB1;

    const auto accumulate_first_layer = [&](std::size_t input) {
      const std::size_t offset = input * replay_value_model::kHiddenOne;
      for (std::size_t hidden = 0;
           hidden < replay_value_model::kHiddenOne; ++hidden) {
        hidden_one[hidden] +=
            static_cast<float>(weights[offset + hidden]) *
            replay_value_model::kW1Scale;
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
      accumulate_first_layer(replay_value_model::kEdgeCount + vertex * 8U +
                             static_cast<std::size_t>(distance));
    }
    for (float &value : hidden_one) {
      value = std::max(value, 0.0F);
    }

    constexpr std::size_t kSecondLayerOffset =
        replay_value_model::kInputCount * replay_value_model::kHiddenOne;
    std::array<float, replay_value_model::kHiddenTwo> hidden_two =
        replay_value_model::kB2;
    for (std::size_t first = 0; first < replay_value_model::kHiddenOne;
         ++first) {
      for (std::size_t second = 0; second < replay_value_model::kHiddenTwo;
           ++second) {
        const std::size_t weight = kSecondLayerOffset +
                                   first * replay_value_model::kHiddenTwo +
                                   second;
        hidden_two[second] += hidden_one[first] *
                              static_cast<float>(weights[weight]) *
                              replay_value_model::kW2Scale;
      }
    }
    for (float &value : hidden_two) {
      value = std::max(value, 0.0F);
    }

    constexpr std::size_t kOutputLayerOffset =
        kSecondLayerOffset + replay_value_model::kHiddenOne *
                                 replay_value_model::kHiddenTwo;
    float output = replay_value_model::kB3;
    for (std::size_t hidden = 0; hidden < replay_value_model::kHiddenTwo;
         ++hidden) {
      output += hidden_two[hidden] *
                static_cast<float>(weights[kOutputLayerOffset + hidden]) *
                replay_value_model::kW3Scale;
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
    if (config_.replay_value_blend_percent != 0) {
      const float model_probability_score =
          std::tanh(replay_value_logit(mover) * 0.5F);
      const int model_score = sign * static_cast<int>(
                                         std::lround(model_probability_score *
                                                     kMaximumEvaluation));
      score = (score * (100 - config_.replay_value_blend_percent) +
               model_score * config_.replay_value_blend_percent) /
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
      std::optional<Move> preferred = std::nullopt) {
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    const std::uint8_t count = position_.legal_slots(slots);
    OrderedMoveList ordered;
    ordered.count = count;
    const Player mover = position_.to_move();
    const Point ball = position_.ball();
    const auto source = position_.ball_vertex();
    const RulesConfig &rules = topology_->config();

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
            // A cached root continuation only needs a reconstructed action if
            // it can improve the action already captured. Otherwise its exact
            // score is sufficient to discard this transposed path.
            const bool cannot_improve =
                mover == Player::One ? entry->score <= alpha
                                     : entry->score >= beta;
            if (!capture_root ||
                (config_.root_transposition_pruning && cannot_improve)) {
              ++stats_.transposition_cutoffs;
              return entry->score;
            }
          } else if (!capture_root || config_.root_transposition_pruning) {
            // At the root, a bound cutoff is safe without reconstructing the
            // cached suffix: the bound proves that this path cannot improve
            // the complete action which established alpha or beta.
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
    OrderedMoveList moves = ordered_moves(preferred);

    if (capture_root && remaining_depth == 1U &&
        config_.root_seed_endpoints) {
      // Touch one legal endpoint below every first edge before exhaustive DFS.
      // This changes only order and guarantees a diverse legal partial result
      // if the first iteration reaches its budget.
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
      const ReboundOutcome rebound = analyze_rebound_component(nullptr);
      if (rebound == ReboundOutcome::Win) {
        return immediate_win_score(position_.to_move(), turn_ply);
      }
      if (rebound == ReboundOutcome::Loss) {
        return immediate_win_score(opponent(position_.to_move()), turn_ply);
      }
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

    const ReboundOutcome rebound = analyze_rebound_component(nullptr);
    if (rebound == ReboundOutcome::Win) {
      return immediate_win_score(position_.to_move(), turn_ply);
    }
    if (rebound == ReboundOutcome::Loss) {
      return immediate_win_score(opponent(position_.to_move()), turn_ply);
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
    encoded.assign(
        recorded.substr(action_begin, action_end - action_begin));
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
  config.replay_value_blend_percent = 15;
  // Include construction and table initialization in the response budget.
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

}  // namespace papersoccer::turn_action_v2

#ifndef PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
int main() {
  using namespace papersoccer;
  using namespace papersoccer::turn_action_v2;

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
