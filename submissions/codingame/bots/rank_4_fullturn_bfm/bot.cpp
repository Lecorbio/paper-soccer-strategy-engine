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
#include <utility>
#include <vector>

#include "mcts_internal.hpp"
#include "papersoccer/rules.hpp"
#include "replay_book.hpp"
#include "replay_value_model.hpp"
#include "teacher_residual_model.hpp"

namespace papersoccer::turn_action_v2 {

constexpr std::uint32_t kFirstSearchTimeMs = 800;
constexpr std::uint32_t kLaterSearchTimeMs = 165;
constexpr std::uint64_t kMaximumWork = 3'000'000;
constexpr std::size_t kMaximumTreeNodes = 120'000;
constexpr std::size_t kMaximumActions = 250;
constexpr std::uint64_t kMaximumPartialPaths = 50'000;
constexpr std::uint64_t kTacticalPartialPaths = 32;
constexpr std::size_t kEvaluationEntries = 131'072;
constexpr double kExplorationConstant = 1.5;
constexpr double kFirstPlayUrgency = 0.5;

constexpr int kMateScore = 1'000'000;
constexpr int kMaximumMateDistance = 32;
constexpr int kMaximumEvaluation = 100'000;
constexpr int kDirectGoalWeight = 50'000;
constexpr int kGoalDistanceWeight = 120;
constexpr int kVerticalProgressWeight = 80;
constexpr int kContinuationWeight = 35;
constexpr int kMobilityWeight = 20;
constexpr int kForwardMoveWeight = 10;
constexpr int kCenterAlignmentWeight = 6;
constexpr int kTempoWeight = 15;
constexpr int kTeacherResidualTargetScale = 20'000;
constexpr int kTeacherResidualHardCap = 6'000;
constexpr int kTeacherResidualMinimumUsedEdges = 12;
constexpr std::size_t kTeacherResidualInputs = 24;

constexpr std::array<Point, 8> kDirectionDeltas{{
    {0, -1}, {1, -1}, {1, 0}, {1, 1},
    {0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
}};

using SearchClock = std::chrono::steady_clock;

struct SearchConfig {
  std::uint64_t max_work{kMaximumWork};
  std::size_t max_tree_nodes{kMaximumTreeNodes};
  std::size_t max_actions{kMaximumActions};
  std::uint64_t max_partial_paths{kMaximumPartialPaths};
  std::size_t evaluation_entries{kEvaluationEntries};
  std::uint32_t max_time_ms{};
  std::optional<SearchClock::time_point> absolute_deadline{};
  int replay_value_blend_percent{};
  int teacher_residual_weight_percent{};
  double exploration_constant{kExplorationConstant};
  double first_play_urgency{kFirstPlayUrgency};
};

struct SearchStats {
  std::uint64_t work{};
  std::uint64_t tree_nodes{};
  std::uint64_t expansions{};
  std::uint64_t child_evaluations{};
  std::uint64_t generator_partial_paths{};
  std::uint64_t completed_actions{};
  std::uint64_t generator_duplicates{};
  std::uint64_t tactical_actions{};
  std::uint64_t generator_truncations{};
  std::uint32_t max_deque_size{};
  std::uint32_t max_action_edges{};
  int root_score{};
  bool budget_exhausted{};
  bool deadline_reached{};
  bool node_cap_reached{};
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
    entries_[bucket + ((combined_key(key) >> 32U) & 1U)] =
        EvaluationEntry{key, score, true};
  }

 private:
  std::vector<EvaluationEntry> entries_;

  static std::uint64_t combined_key(detail::PositionKey key) noexcept {
    return key.first ^ ((key.second << 29U) | (key.second >> 35U));
  }

  std::size_t bucket_index(detail::PositionKey key) const noexcept {
    return 2U * static_cast<std::size_t>(
                    combined_key(key) % (entries_.size() / 2U));
  }
};

struct EvaluationSnapshot {
  int anchor_score{};
  int mover_sign{};
  std::array<float, kTeacherResidualInputs> features{};
};

struct ReplayValueOutput {
  float logit{};
  std::array<float, replay_value_model::kHiddenTwo> hidden{};
  std::uint16_t used_edges{};
};

int player_sign(Player player) noexcept {
  return player == Player::One ? 1 : -1;
}

Player player_for_id(int player_id) {
  if (player_id == 0) {
    return Player::One;
  }
  if (player_id == 1) {
    return Player::Two;
  }
  throw std::invalid_argument("");
}

Move decode_direction(Point from, char direction) {
  if (direction < '0' || direction > '7') {
    throw std::invalid_argument("");
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
  throw std::invalid_argument("");
}

detail::PositionKey boundary_key(
    const detail::SearchPosition &position) noexcept {
  detail::PositionKey key = position.position_key();
  detail::xor_position_key(
      key, detail::position_key_component(6, position.ball_vertex()));
  return key;
}

class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t state) : state_(state) {}

  std::uint64_t next() noexcept {
    state_ += 0x9e3779b97f4a7c15ULL;
    return detail::mix_position_key(state_);
  }

  std::size_t index(std::size_t bound) noexcept {
    if (bound <= 1) {
      return 0;
    }
    const std::uint64_t threshold =
        static_cast<std::uint64_t>(-bound) % bound;
    for (;;) {
      const std::uint64_t value = next();
      if (value >= threshold) {
        return static_cast<std::size_t>(value % bound);
      }
    }
  }

 private:
  std::uint64_t state_{};
};

enum class TacticalClass {
  ImmediateWin,
  ForcedCutoff,
  SafeHandoff,
  OpponentImmediateWin,
  TerminalLoss,
};

struct GeneratorConfig {
  std::size_t max_actions{kMaximumActions};
  std::uint64_t max_partial_paths{kMaximumPartialPaths};
  std::uint64_t max_work{kMaximumWork};
  std::optional<SearchClock::time_point> absolute_deadline{};
};

struct GeneratorStats {
  std::uint64_t explored_partial_paths{};
  std::uint64_t completed_actions{};
  std::uint64_t duplicates{};
  std::uint64_t tactical_actions{};
  std::uint64_t truncations{};
  std::uint64_t fifo_extractions{};
  std::uint64_t lifo_extractions{};
  std::uint64_t immediate_wins{};
  std::uint64_t terminal_losses{};
  std::uint64_t safe_handoffs{};
  std::uint64_t opponent_immediate_wins{};
  std::uint64_t forced_cutoffs{};
  std::uint32_t max_deque_size{};
  std::uint32_t max_action_edges{};
  bool deadline_reached{};
};

struct CompleteTurnAction {
  std::vector<Move> moves;
  detail::SearchPosition result;
  TacticalClass tactical{TacticalClass::SafeHandoff};
  std::string encoded;

  CompleteTurnAction(std::vector<Move> action,
                     detail::SearchPosition position,
                     TacticalClass classification, std::string key)
      : moves(std::move(action)), result(std::move(position)),
        tactical(classification), encoded(std::move(key)) {}
};

struct GenerationResult {
  std::vector<CompleteTurnAction> actions;
  GeneratorStats stats{};
  bool exhaustive{};
};

enum class ReboundOutcome { Unknown, Win, Loss };

class FullTurnGenerator {
 public:
  FullTurnGenerator(std::shared_ptr<const detail::SearchTopology> topology,
                    const detail::SearchPosition &root,
                    GeneratorConfig config = {})
      : topology_(std::move(topology)), root_(root), config_(config),
        parents_(topology_->vertex_count(), kUnseen),
        queue_(topology_->vertex_count()) {
    if (config_.max_actions == 0 ||
        config_.max_actions > kMaximumActions ||
        config_.max_partial_paths == 0 ||
        config_.max_partial_paths > kMaximumPartialPaths ||
        config_.max_work == 0 || config_.max_work > kMaximumWork) {
      throw std::invalid_argument("");
    }
    edge_vertices_.resize(topology_->edge_count());
    for (std::uint32_t vertex = 0; vertex < topology_->vertex_count();
         ++vertex) {
      for (const Player player : {Player::One, Player::Two}) {
        const auto &adjacency = topology_->adjacency(vertex, player);
        for (std::uint8_t index = 0; index < adjacency.count; ++index) {
          const auto &arc = adjacency.arcs[index];
          edge_vertices_[arc.edge] = {vertex, arc.destination};
        }
      }
    }
  }

  GenerationResult run() {
    if (root_.is_terminal()) {
      throw std::invalid_argument("");
    }
    retained_keys_.clear();
    deadline_reached_ = false;

    GenerationResult output;
    output.actions.reserve(config_.max_actions);
    bool truncated = false;
    if (!deadline_expired()) {
      retain_exact_goal_path(output, root_.to_move());
    }
    if (!deadline_expired()) {
      retain_bounded_handoff_paths(output);
    }
    if (!deadline_expired()) {
      retain_exact_goal_path(output, opponent(root_.to_move()));
    }
    if (deadline_reached_) {
      output.stats.deadline_reached = true;
      truncated = true;
    }

    std::deque<Partial> pending;
    if (!deadline_reached_) {
      pending.emplace_back(root_, std::vector<Move>{});
      output.stats.max_deque_size =
          std::max<std::uint32_t>(output.stats.max_deque_size, 1);
    }

    while (!pending.empty()) {
      if (output.actions.size() >= config_.max_actions ||
          output.stats.explored_partial_paths >=
              config_.max_partial_paths ||
          output.stats.explored_partial_paths >= config_.max_work) {
        truncated = true;
        break;
      }
      if (deadline_expired()) {
        truncated = true;
        output.stats.deadline_reached = true;
        break;
      }

      ++output.stats.explored_partial_paths;
      Partial partial = [&] {
        if (output.stats.explored_partial_paths % 10U == 0U) {
          ++output.stats.fifo_extractions;
          Partial value = std::move(pending.front());
          pending.pop_front();
          return value;
        }
        ++output.stats.lifo_extractions;
        Partial value = std::move(pending.back());
        pending.pop_back();
        return value;
      }();

      std::array<std::uint8_t, detail::kMaximumMoves> slots{};
      const std::uint8_t count = partial.position.legal_slots(slots);
      shuffle(slots, count, partial.position);
      for (std::uint8_t index = 0; index < count; ++index) {
        Partial child = partial;
        const Move move = child.position.move_for_slot(slots[index]);
        child.moves.push_back(move);
        child.position.make_move(slots[index]);
        if (child.position.is_terminal() ||
            child.position.to_move() != root_.to_move()) {
          retain(output, std::move(child.moves),
                 std::move(child.position));
          if (output.actions.size() >= config_.max_actions) {
            truncated = truncated || !pending.empty() || index + 1U < count;
            break;
          }
        } else {
          if (pending.size() < config_.max_partial_paths) {
            pending.push_back(std::move(child));
            output.stats.max_deque_size = std::max<std::uint32_t>(
                output.stats.max_deque_size,
                static_cast<std::uint32_t>(pending.size()));
          } else {
            truncated = true;
          }
        }
      }
      if (output.actions.size() >= config_.max_actions) {
        break;
      }
    }

    if (deadline_reached_) {
      output.stats.deadline_reached = true;
      truncated = true;
    }

    if (output.actions.empty() && !deadline_reached_) {
      Partial fallback{root_, fallback_action(root_)};
      fallback.position = root_;
      for (const Move move : fallback.moves) {
        std::uint8_t slot = 0;
        if (!fallback.position.slot_for_move(move, slot)) {
          throw std::logic_error("");
        }
        fallback.position.make_move(slot);
      }
      retain(output, std::move(fallback.moves),
             std::move(fallback.position));
      truncated = true;
    }

    std::stable_sort(output.actions.begin(), output.actions.end(),
                     [](const CompleteTurnAction &left,
                        const CompleteTurnAction &right) {
                       if (left.tactical != right.tactical) {
                         return static_cast<int>(left.tactical) <
                                static_cast<int>(right.tactical);
                       }
                       return left.encoded < right.encoded;
                     });
    output.exhaustive = pending.empty() && !truncated;
    if (truncated) {
      ++output.stats.truncations;
    }
    return output;
  }

 private:
  static constexpr int kUnseen = -2;

  struct Partial {
    detail::SearchPosition position;
    std::vector<Move> moves;

    Partial(detail::SearchPosition state, std::vector<Move> action)
        : position(std::move(state)), moves(std::move(action)) {}
  };

  std::shared_ptr<const detail::SearchTopology> topology_;
  detail::SearchPosition root_;
  GeneratorConfig config_;
  std::vector<int> parents_;
  std::vector<detail::SearchTopology::VertexIndex> queue_;
  std::vector<detail::PositionKey> retained_keys_;
  std::vector<std::pair<detail::SearchTopology::VertexIndex,
                        detail::SearchTopology::VertexIndex>> edge_vertices_;
  bool deadline_reached_{};

  bool deadline_expired() {
    if (!deadline_reached_ && config_.absolute_deadline.has_value() &&
        SearchClock::now() >= *config_.absolute_deadline) {
      deadline_reached_ = true;
    }
    return deadline_reached_;
  }

  Point transformed(Point point, unsigned symmetry) const noexcept {
    if ((symmetry & 2U) != 0U) {
      point = {topology_->config().width - point.x,
               topology_->config().height + 2 - point.y};
    }
    if ((symmetry & 1U) != 0U) {
      point.x = topology_->config().width - point.x;
    }
    return point;
  }

  static std::uint64_t packed(Point point) noexcept {
    return static_cast<std::uint64_t>(point.y * 16 + point.x);
  }

  std::uint64_t symmetry_key(const detail::SearchPosition &position,
                             unsigned symmetry) const noexcept {
    std::uint64_t key = 0;
    const auto add = [&](std::uint64_t category, std::uint64_t value) {
      key ^= detail::mix_position_key((category << 56U) ^ value ^
                                     0xd1b54a32d192ed03ULL);
    };
    for (std::uint32_t vertex = 0; vertex < topology_->vertex_count();
         ++vertex) {
      if (position.vertex_visited(vertex)) {
        add(2, packed(transformed(topology_->point(vertex), symmetry)));
      }
    }
    for (std::uint32_t edge = 0; edge < topology_->edge_count(); ++edge) {
      if (position.edge_used(edge)) {
        auto [first, second] = edge_vertices_[edge];
        std::uint64_t a = packed(transformed(topology_->point(first), symmetry));
        std::uint64_t b = packed(transformed(topology_->point(second), symmetry));
        if (a > b) std::swap(a, b);
        add(1, (a << 8U) | b);
      }
    }
    Player mover = position.to_move();
    Status status = position.status();
    if ((symmetry & 2U) != 0U) {
      mover = opponent(mover);
      if (status == Status::WonByOne) status = Status::WonByTwo;
      else if (status == Status::WonByTwo) status = Status::WonByOne;
    }
    add(3, packed(transformed(position.ball(), symmetry)));
    add(4, static_cast<std::uint64_t>(mover));
    add(5, static_cast<std::uint64_t>(status));
    return key;
  }

  std::pair<std::uint64_t, unsigned> canonical_key(
      const detail::SearchPosition &position) const noexcept {
    std::pair<std::uint64_t, unsigned> best{symmetry_key(position, 0), 0};
    for (unsigned symmetry = 1; symmetry < 4; ++symmetry) {
      const std::uint64_t key = symmetry_key(position, symmetry);
      if (key < best.first) {
        best = {key, symmetry};
      }
    }
    return best;
  }

  static unsigned canonical_direction(Point from, Point to,
                                      unsigned symmetry) {
    unsigned value =
        static_cast<unsigned>(encode_direction(from, to) - '0');
    if ((symmetry & 2U) != 0U) value = (value + 4U) & 7U;
    if ((symmetry & 1U) != 0U) value = (8U - value) & 7U;
    return value;
  }

  void shuffle(std::array<std::uint8_t, detail::kMaximumMoves> &slots,
               std::uint8_t count,
               const detail::SearchPosition &position) const {
    const auto [key, symmetry] = canonical_key(position);
    std::sort(slots.begin(), slots.begin() + count,
              [&](std::uint8_t left, std::uint8_t right) {
                return canonical_direction(
                           position.ball(), position.move_for_slot(left).to,
                           symmetry) <
                       canonical_direction(
                           position.ball(), position.move_for_slot(right).to,
                           symmetry);
              });
    SplitMix64 random(key);
    for (std::size_t remaining = count; remaining > 1; --remaining) {
      std::swap(slots[remaining - 1U],
                slots[random.index(remaining)]);
    }
    if (symmetry_key(position, 0) == symmetry_key(position, 1)) {
      for (std::uint8_t index = 0; index < count; ++index) {
        const unsigned direction = canonical_direction(
            position.ball(), position.move_for_slot(slots[index]).to, 0);
        if (direction == 0 || direction == 4) {
          std::swap(slots[0], slots[index]);
          break;
        }
      }
    }
  }

  std::string encode_action(const std::vector<Move> &moves) const {
    detail::SearchPosition position = root_;
    std::string encoded;
    encoded.reserve(moves.size());
    for (const Move move : moves) {
      encoded.push_back(encode_direction(position.ball(), move.to));
      std::uint8_t slot = 0;
      if (!position.slot_for_move(move, slot)) {
        throw std::logic_error("");
      }
      position.make_move(slot);
    }
    return encoded;
  }

  std::optional<std::vector<Move>> rebound_goal_path(
      const detail::SearchPosition &state, Player goal_owner) {
    std::fill(parents_.begin(), parents_.end(), kUnseen);
    const auto start = state.ball_vertex();
    parents_[start] = -1;
    std::size_t head = 0;
    std::size_t tail = 0;
    queue_[tail++] = start;
    const Player mover = state.to_move();
    const RulesConfig &rules = topology_->config();
    const auto &root_adjacency = topology_->adjacency(start, mover);
    const Point center_goal{rules.width / 2,
                            goal_owner == Player::One
                                ? 0
                                : rules.height + 2};
    for (std::uint8_t index = 0; index < root_adjacency.count; ++index) {
      const auto &arc = root_adjacency.arcs[index];
      if (!state.edge_used(arc.edge) &&
          topology_->point(arc.destination) == center_goal) {
        return std::vector<Move>{Move{center_goal}};
      }
    }
    const unsigned symmetry = canonical_key(state).second;

    while (head < tail) {
      if (deadline_expired()) {
        return std::nullopt;
      }
      const auto vertex = queue_[head++];
      const auto &adjacency = topology_->adjacency(vertex, mover);
      for (unsigned direction = 0; direction < 8; ++direction) {
        for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const auto &arc = adjacency.arcs[index];
        const Point destination = topology_->point(arc.destination);
        if (canonical_direction(topology_->point(vertex), destination,
                                symmetry) != direction) {
          continue;
        }
        if (state.edge_used(arc.edge)) {
          continue;
        }
        if (is_goal_point(rules, destination)) {
          const Player winner =
              is_attacking_goal(rules, destination, Player::One)
                  ? Player::One
                  : Player::Two;
          if (winner != goal_owner) {
            continue;
          }
          parents_[arc.destination] = static_cast<int>(vertex);
          std::vector<Move> path;
          for (auto current = arc.destination; current != start;
               current = static_cast<detail::SearchTopology::VertexIndex>(
                   parents_[current])) {
            path.push_back(Move{topology_->point(current)});
          }
          std::reverse(path.begin(), path.end());
          return path;
        }
        if (parents_[arc.destination] != kUnseen) {
          continue;
        }
        const bool continues =
            is_boundary_point(rules, destination) ||
            state.vertex_visited(arc.destination);
        if (!continues) {
          continue;
        }
        parents_[arc.destination] = static_cast<int>(vertex);
        queue_[tail++] = arc.destination;
        }
      }
    }
    return std::nullopt;
  }

  ReboundOutcome analyze_rebound_component(
      const detail::SearchPosition &state) {
    if (rebound_goal_path(state, state.to_move()).has_value()) {
      return ReboundOutcome::Win;
    }

    std::fill(parents_.begin(), parents_.end(), kUnseen);
    const auto start = state.ball_vertex();
    parents_[start] = -1;
    std::size_t head = 0;
    std::size_t tail = 0;
    queue_[tail++] = start;
    const Player mover = state.to_move();
    const RulesConfig &rules = topology_->config();
    bool safe_handoff = false;

    while (head < tail) {
      if (deadline_expired()) {
        return ReboundOutcome::Unknown;
      }
      const auto vertex = queue_[head++];
      const auto &adjacency = topology_->adjacency(vertex, mover);
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const auto &arc = adjacency.arcs[index];
        if (state.edge_used(arc.edge)) {
          continue;
        }
        const Point destination = topology_->point(arc.destination);
        if (is_goal_point(rules, destination) ||
            parents_[arc.destination] != kUnseen) {
          continue;
        }
        if (!is_boundary_point(rules, destination) &&
            !state.vertex_visited(arc.destination)) {
          const auto &reply =
              topology_->adjacency(arc.destination, opponent(mover));
          for (std::uint8_t child = 0; child < reply.count; ++child) {
            if (reply.arcs[child].edge != arc.edge &&
                !state.edge_used(reply.arcs[child].edge)) {
              safe_handoff = true;
              break;
            }
          }
          continue;
        }
        parents_[arc.destination] = static_cast<int>(vertex);
        queue_[tail++] = arc.destination;
      }
    }
    if (!safe_handoff &&
        rules.blocked_rule == BlockedRule::MoverLoses) {
      return ReboundOutcome::Loss;
    }
    return ReboundOutcome::Unknown;
  }

  TacticalClass classify(const detail::SearchPosition &result) {
    if (result.is_terminal()) {
      return result.winner() == root_.to_move()
                 ? TacticalClass::ImmediateWin
                 : TacticalClass::TerminalLoss;
    }
    switch (analyze_rebound_component(result)) {
      case ReboundOutcome::Win:
        return TacticalClass::OpponentImmediateWin;
      case ReboundOutcome::Loss:
        return TacticalClass::ForcedCutoff;
      case ReboundOutcome::Unknown:
        return TacticalClass::SafeHandoff;
    }
    throw std::logic_error("");
  }

  void retain(GenerationResult &output, std::vector<Move> moves,
              detail::SearchPosition result) {
    if (deadline_expired()) {
      return;
    }
    ++output.stats.completed_actions;
    output.stats.max_action_edges = std::max<std::uint32_t>(
        output.stats.max_action_edges,
        static_cast<std::uint32_t>(moves.size()));
    const detail::PositionKey key = boundary_key(result);
    if (std::find(retained_keys_.begin(), retained_keys_.end(), key) !=
        retained_keys_.end()) {
      ++output.stats.duplicates;
      return;
    }
    retained_keys_.push_back(key);
    const TacticalClass tactical = classify(result);
    if (deadline_reached_) {
      return;
    }
    switch (tactical) {
      case TacticalClass::ImmediateWin:
        ++output.stats.immediate_wins;
        ++output.stats.tactical_actions;
        break;
      case TacticalClass::ForcedCutoff:
        ++output.stats.forced_cutoffs;
        ++output.stats.tactical_actions;
        break;
      case TacticalClass::SafeHandoff:
        ++output.stats.safe_handoffs;
        break;
      case TacticalClass::OpponentImmediateWin:
        ++output.stats.opponent_immediate_wins;
        ++output.stats.tactical_actions;
        break;
      case TacticalClass::TerminalLoss:
        ++output.stats.terminal_losses;
        ++output.stats.tactical_actions;
        break;
    }
    const std::string encoded = encode_action(moves);
    output.actions.emplace_back(std::move(moves), std::move(result), tactical,
                                encoded);
  }

  void retain_bounded_handoff_paths(GenerationResult &output) {
    if (output.actions.size() >= config_.max_actions) {
      return;
    }
    const std::uint64_t available = std::min(
        config_.max_partial_paths - output.stats.explored_partial_paths,
        config_.max_work - output.stats.explored_partial_paths);
    const std::uint64_t limit =
        std::min<std::uint64_t>(kTacticalPartialPaths, available);
    std::array<std::optional<Partial>, 5> candidates;
    std::deque<Partial> pending;
    pending.emplace_back(root_, std::vector<Move>{});
    std::uint64_t extracted = 0;
    while (!pending.empty() && extracted < limit) {
      if (deadline_expired()) {
        return;
      }
      ++extracted;
      ++output.stats.explored_partial_paths;
      Partial partial = [&] {
        if (output.stats.explored_partial_paths % 10U == 0U) {
          ++output.stats.fifo_extractions;
          Partial value = std::move(pending.front());
          pending.pop_front();
          return value;
        }
        ++output.stats.lifo_extractions;
        Partial value = std::move(pending.back());
        pending.pop_back();
        return value;
      }();
      std::array<std::uint8_t, detail::kMaximumMoves> slots{};
      const std::uint8_t count = partial.position.legal_slots(slots);
      shuffle(slots, count, partial.position);
      for (std::uint8_t index = 0; index < count; ++index) {
        Partial child = partial;
        const Move move = child.position.move_for_slot(slots[index]);
        child.moves.push_back(move);
        child.position.make_move(slots[index]);
        if (child.position.is_terminal() ||
            child.position.to_move() != root_.to_move()) {
          const TacticalClass tactical = classify(child.position);
          if (deadline_reached_) {
            return;
          }
          const std::size_t slot = static_cast<std::size_t>(tactical);
          if (tactical != TacticalClass::ImmediateWin &&
              !candidates[slot].has_value()) {
            candidates[slot] = std::move(child);
          }
        } else if (pending.size() < config_.max_partial_paths) {
          pending.push_back(std::move(child));
          output.stats.max_deque_size = std::max<std::uint32_t>(
              output.stats.max_deque_size,
              static_cast<std::uint32_t>(pending.size()));
        }
      }
    }
    constexpr std::array<TacticalClass, 4> priority{{
        TacticalClass::ForcedCutoff, TacticalClass::SafeHandoff,
        TacticalClass::OpponentImmediateWin, TacticalClass::TerminalLoss}};
    for (const TacticalClass tactical : priority) {
      auto &candidate = candidates[static_cast<std::size_t>(tactical)];
      if (candidate.has_value() &&
          output.actions.size() < config_.max_actions) {
        retain(output, std::move(candidate->moves),
               std::move(candidate->position));
      }
    }
  }

  void retain_exact_goal_path(GenerationResult &output, Player goal_owner) {
    const auto path = rebound_goal_path(root_, goal_owner);
    if (deadline_reached_ || !path.has_value() ||
        output.actions.size() >= config_.max_actions) {
      return;
    }
    detail::SearchPosition result = root_;
    for (const Move move : *path) {
      std::uint8_t slot = 0;
      if (!result.slot_for_move(move, slot)) {
        throw std::logic_error("");
      }
      result.make_move(slot);
    }
    retain(output, *path, std::move(result));
  }

  std::vector<Move> fallback_action(
      const detail::SearchPosition &state) const {
    detail::SearchPosition position = state;
    const Player mover = position.to_move();
    std::vector<Move> action;
    while (!position.is_terminal() && position.to_move() == mover) {
      std::array<std::uint8_t, detail::kMaximumMoves> slots{};
      const std::uint8_t count = position.legal_slots(slots);
      if (count == 0) {
        throw std::logic_error("");
      }
      std::uint8_t best = slots[0];
      int best_score = std::numeric_limits<int>::min();
      for (std::uint8_t index = 0; index < count; ++index) {
        const Move move = position.move_for_slot(slots[index]);
        int score = position.grants_extra_turn(slots[index]) ? 10'000 : 0;
        const int progress = mover == Player::One
                                 ? position.ball().y - move.to.y
                                 : move.to.y - position.ball().y;
        score += progress * 100;
        if (is_attacking_goal(topology_->config(), move.to, mover)) {
          score += 1'000'000;
        } else if (is_goal_point(topology_->config(), move.to)) {
          score -= 1'000'000;
        }
        if (score > best_score ||
            (score == best_score && slots[index] < best)) {
          best = slots[index];
          best_score = score;
        }
      }
      action.push_back(position.move_for_slot(best));
      position.make_move(best);
    }
    return action;
  }

};

double normalized_search_value(int value, bool proven = true) noexcept {
  if (proven &&
      std::abs(value) >= kMateScore - kMaximumMateDistance) {
    return value > 0 ? 10'000.0 : -10'000.0;
  }
  return static_cast<double>(
             std::clamp(value, -kMaximumEvaluation, kMaximumEvaluation)) /
         kMaximumEvaluation;
}

double uct_selection_score(int value, Player mover,
                           std::uint32_t parent_visits,
                           std::uint32_t child_visits) noexcept {
  const double exploitation =
      player_sign(mover) * normalized_search_value(value);
  if (child_visits == 0) {
    return exploitation + kFirstPlayUrgency;
  }
  return exploitation +
         kExplorationConstant *
             std::sqrt(std::log(static_cast<double>(
                                    std::max<std::uint32_t>(1, parent_visits))) /
                       static_cast<double>(child_visits));
}

int minimax_backup(Player mover, const std::vector<int> &values) {
  if (values.empty()) {
    throw std::invalid_argument("");
  }
  return mover == Player::One
             ? *std::max_element(values.begin(), values.end())
             : *std::min_element(values.begin(), values.end());
}

class FullTurnBfmSearch {
 public:
  FullTurnBfmSearch(const GameState &state, SearchConfig config)
      : config_(config),
        topology_(std::make_shared<detail::SearchTopology>(state.config)),
        root_position_(topology_, state), position_(root_position_),
        evaluations_(config.evaluation_entries),
        distances_(topology_->vertex_count(), -1),
        queue_(topology_->vertex_count()),
        teacher_residual_root_enabled_(
            state.used_segments.size() >=
            static_cast<std::size_t>(kTeacherResidualMinimumUsedEdges)) {
    if (config_.max_work == 0 || config_.max_work > kMaximumWork ||
        config_.max_tree_nodes < 2 ||
        config_.max_tree_nodes > kMaximumTreeNodes ||
        config_.max_actions == 0 || config_.max_actions > kMaximumActions ||
        config_.max_partial_paths == 0 ||
        config_.max_partial_paths > kMaximumPartialPaths ||
        config_.evaluation_entries < 2 ||
        config_.replay_value_blend_percent < 0 ||
        config_.replay_value_blend_percent > 100 ||
        config_.teacher_residual_weight_percent < 0 ||
        config_.teacher_residual_weight_percent > 100 ||
        !std::isfinite(config_.exploration_constant) ||
        config_.exploration_constant < 0.0 ||
        !std::isfinite(config_.first_play_urgency)) {
      throw std::invalid_argument("");
    }
    initialize_rotation_maps();
    if (config_.absolute_deadline.has_value()) {
      deadline_ = config_.absolute_deadline;
    } else if (config_.max_time_ms != 0) {
      deadline_ =
          SearchClock::now() + std::chrono::milliseconds(config_.max_time_ms);
    }
    nodes_.reserve(config_.max_tree_nodes);
  }

  std::vector<Move> run() {
    if (root_position_.is_terminal()) {
      throw std::invalid_argument("");
    }

    const std::vector<Move> fallback = fallback_action(root_position_);
    if (!budget_available()) {
      return fallback;
    }
    position_ = root_position_;
    const int root_evaluation = cached_evaluate();
    if (deadline_expired()) {
      return fallback;
    }
    nodes_.emplace_back(root_position_, kNoParent, std::vector<Move>{},
                        std::string{}, root_evaluation, 0, false);
    stats_.tree_nodes = 1;
    stats_.root_score = root_evaluation;

    if (!expand_node(0)) {
      stats_.budget_exhausted = true;
      return fallback;
    }
    ++nodes_[0].visits;
    refresh_node(0);

    while (!nodes_[0].closed && budget_available()) {
      const std::optional<std::vector<std::size_t>> selected = select_path();
      if (!selected.has_value() || selected->empty()) {
        break;
      }
      const std::size_t leaf = selected->back();
      if (!expand_node(leaf)) {
        break;
      }
      for (const std::size_t index : *selected) {
        ++nodes_[index].visits;
      }
      backup_path(*selected);
    }

    if (nodes_[0].children.empty()) {
      stats_.budget_exhausted = true;
      return fallback;
    }
    const std::size_t chosen = final_child();
    stats_.root_score = nodes_[chosen].value;
    return nodes_[chosen].incoming_action;
  }

  const SearchStats &stats() const noexcept { return stats_; }

  EvaluationSnapshot evaluation_snapshot() {
    position_ = root_position_;
    return make_evaluation_snapshot();
  }

 private:
  static constexpr std::size_t kNoParent =
      std::numeric_limits<std::size_t>::max();

  struct TreeNode {
    detail::SearchPosition position;
    std::size_t parent{kNoParent};
    std::vector<std::size_t> children;
    std::vector<Move> incoming_action;
    std::string action_key;
    int value{};
    std::uint32_t visits{};
    std::uint32_t depth{};
    bool expanded{};
    bool solved{};
    bool exhaustive{};
    bool closed{};

    TreeNode(detail::SearchPosition state, std::size_t parent_index,
             std::vector<Move> action, std::string key, int score,
             std::uint32_t node_depth, bool proven)
        : position(std::move(state)), parent(parent_index),
          incoming_action(std::move(action)), action_key(std::move(key)),
          value(score), depth(node_depth), expanded(proven),
          solved(proven), exhaustive(proven), closed(proven) {}
  };

  SearchConfig config_;
  std::shared_ptr<const detail::SearchTopology> topology_;
  detail::SearchPosition root_position_;
  detail::SearchPosition position_;
  EvaluationTable evaluations_;
  SearchStats stats_;
  std::vector<int> distances_;
  std::vector<detail::SearchTopology::VertexIndex> queue_;
  std::deque<detail::SearchTopology::VertexIndex> distance_queue_;
  std::vector<detail::SearchTopology::VertexIndex> rotated_vertices_;
  std::vector<detail::SearchTopology::EdgeIndex> rotated_edges_;
  std::optional<SearchClock::time_point> deadline_;
  std::vector<TreeNode> nodes_;
  bool teacher_residual_root_enabled_{};

  bool deadline_expired() {
    if (deadline_.has_value() && SearchClock::now() >= *deadline_) {
      stats_.deadline_reached = true;
      stats_.budget_exhausted = true;
      return true;
    }
    return false;
  }

  bool budget_available() {
    if (stats_.work >= config_.max_work) {
      stats_.budget_exhausted = true;
      return false;
    }
    if (!nodes_.empty() && nodes_.size() >= config_.max_tree_nodes) {
      stats_.node_cap_reached = true;
      stats_.budget_exhausted = true;
      return false;
    }
    return !deadline_expired();
  }

  int terminal_score(const detail::SearchPosition &state,
                     std::uint32_t turn_depth) {
    const std::optional<Player> winning_player = state.winner();
    if (!winning_player.has_value()) {
      throw std::logic_error("");
    }
    return proven_score(*winning_player, turn_depth);
  }

  static int proven_score(Player winning_player,
                          std::uint32_t turn_depth) noexcept {
    const int distance = static_cast<int>(
        std::min<std::uint32_t>(turn_depth, kMaximumMateDistance));
    return winning_player == Player::One ? kMateScore - distance
                                         : -kMateScore + distance;
  }

  static bool score_wins_for(int score, Player player) noexcept {
    return std::abs(score) >= kMateScore - kMaximumMateDistance &&
           (score > 0) == (player == Player::One);
  }

  void add_generator_stats(const GeneratorStats &source) {
    stats_.generator_partial_paths += source.explored_partial_paths;
    stats_.completed_actions += source.completed_actions;
    stats_.generator_duplicates += source.duplicates;
    stats_.tactical_actions += source.tactical_actions;
    stats_.generator_truncations += source.truncations;
    stats_.max_deque_size =
        std::max(stats_.max_deque_size, source.max_deque_size);
    stats_.max_action_edges =
        std::max(stats_.max_action_edges, source.max_action_edges);
    stats_.deadline_reached =
        stats_.deadline_reached || source.deadline_reached;
  }

  bool expand_node(std::size_t node_index) {
    if (!budget_available() || nodes_[node_index].expanded ||
        nodes_[node_index].solved) {
      return false;
    }
    const std::uint64_t remaining_work = config_.max_work - stats_.work;
    GeneratorConfig generator_config;
    generator_config.max_actions = config_.max_actions;
    generator_config.max_partial_paths = config_.max_partial_paths;
    generator_config.max_work = remaining_work;
    generator_config.absolute_deadline = deadline_;
    FullTurnGenerator generator(topology_, nodes_[node_index].position,
                                generator_config);
    GenerationResult generated = generator.run();
    add_generator_stats(generated.stats);
    stats_.work += generated.stats.explored_partial_paths;

    std::size_t expected = generated.actions.size();
    if (expected == 0) {
      stats_.budget_exhausted = true;
      return false;
    }
    const bool work_fits = expected <= config_.max_work - stats_.work;
    const bool tree_fits =
        expected <= config_.max_tree_nodes - nodes_.size();
    if (!work_fits || !tree_fits) {
      const Player mover = nodes_[node_index].position.to_move();
      const auto winning = std::find_if(
          generated.actions.begin(), generated.actions.end(),
          [mover](const CompleteTurnAction &action) {
            return (action.result.is_terminal() &&
                    action.result.winner() == mover) ||
                   action.tactical == TacticalClass::ForcedCutoff;
          });
      if (winning != generated.actions.end() &&
          stats_.work < config_.max_work &&
          nodes_.size() < config_.max_tree_nodes) {
        std::iter_swap(generated.actions.begin(), winning);
        generated.actions.erase(generated.actions.begin() + 1,
                                generated.actions.end());
        generated.exhaustive = false;
        expected = 1;
      } else {
        stats_.node_cap_reached = !tree_fits;
        stats_.budget_exhausted = true;
        return false;
      }
    }

    const std::size_t original_tree_size = nodes_.size();
    std::size_t inserted = 0;
    for (CompleteTurnAction &action : generated.actions) {
      if (!budget_available()) {
        break;
      }
      ++stats_.work;
      ++stats_.child_evaluations;
      const bool terminal = action.result.is_terminal();
      const std::uint32_t child_depth = nodes_[node_index].depth + 1U;
      int value = 0;
      bool proven = terminal;
      if (terminal) {
        value = terminal_score(action.result, child_depth);
      } else if (action.tactical == TacticalClass::ForcedCutoff) {
        value = proven_score(opponent(action.result.to_move()),
                             child_depth + 1U);
        proven = true;
      } else if (action.tactical ==
                 TacticalClass::OpponentImmediateWin) {
        value = proven_score(action.result.to_move(), child_depth + 1U);
        proven = true;
      } else {
        position_ = action.result;
        value = cached_evaluate();
      }
      const std::size_t child_index = nodes_.size();
      nodes_.emplace_back(std::move(action.result), node_index,
                          std::move(action.moves), std::move(action.encoded),
                          value, child_depth, proven);
      nodes_[node_index].children.push_back(child_index);
      ++inserted;
    }

    if (inserted != expected) {
      nodes_.erase(nodes_.begin() +
                       static_cast<std::ptrdiff_t>(original_tree_size),
                   nodes_.end());
      nodes_[node_index].children.clear();
      stats_.tree_nodes = nodes_.size();
      stats_.budget_exhausted = true;
      return false;
    }

    nodes_[node_index].expanded = true;
    nodes_[node_index].exhaustive = generated.exhaustive;
    stats_.tree_nodes = nodes_.size();
    if (nodes_[node_index].expanded) {
      ++stats_.expansions;
      refresh_node(node_index);
      return true;
    }
    stats_.budget_exhausted = true;
    return false;
  }

  void refresh_node(std::size_t node_index) {
    TreeNode &node = nodes_[node_index];
    if (node.children.empty()) {
      return;
    }
    std::vector<int> child_values;
    child_values.reserve(node.children.size());
    std::vector<int> winning_values;
    winning_values.reserve(node.children.size());
    bool all_solved = node.exhaustive;
    bool all_closed = true;
    for (const std::size_t child_index : node.children) {
      const TreeNode &child = nodes_[child_index];
      child_values.push_back(child.value);
      all_solved = all_solved && child.solved;
      all_closed = all_closed && child.closed;
      if (child.solved &&
          score_wins_for(child.value, node.position.to_move())) {
        winning_values.push_back(child.value);
      }
    }
    if (!winning_values.empty()) {
      node.value =
          minimax_backup(node.position.to_move(), winning_values);
      node.solved = true;
    } else {
      node.value = minimax_backup(node.position.to_move(), child_values);
      node.solved = all_solved;
    }
    node.closed = node.solved || all_closed;
  }

  void backup_path(const std::vector<std::size_t> &path) {
    for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
      refresh_node(*iterator);
    }
  }

  double selection_score(const TreeNode &parent,
                         const TreeNode &child) const noexcept {
    const double exploitation =
        player_sign(parent.position.to_move()) *
        normalized_search_value(child.value, child.solved);
    if (child.visits == 0) {
      return exploitation + config_.first_play_urgency;
    }
    return exploitation +
           config_.exploration_constant *
               std::sqrt(std::log(static_cast<double>(
                                      std::max<std::uint32_t>(1,
                                                              parent.visits))) /
                         static_cast<double>(child.visits));
  }

  std::optional<std::vector<std::size_t>> select_path() const {
    std::vector<std::size_t> path{0};
    std::size_t current = 0;
    while (nodes_[current].expanded) {
      const TreeNode &parent = nodes_[current];
      std::optional<std::size_t> best;
      double best_score = -std::numeric_limits<double>::infinity();
      for (const std::size_t child_index : parent.children) {
        const TreeNode &child = nodes_[child_index];
        if (child.closed) {
          continue;
        }
        const double score = selection_score(parent, child);
        if (!best.has_value() || score > best_score ||
            (score == best_score &&
             child.action_key < nodes_[*best].action_key)) {
          best = child_index;
          best_score = score;
        }
      }
      if (!best.has_value()) {
        return std::nullopt;
      }
      current = *best;
      path.push_back(current);
    }
    return path;
  }

  std::size_t final_child() const {
    const TreeNode &root = nodes_[0];
    std::size_t best = root.children.front();
    double best_score = -std::numeric_limits<double>::infinity();
    for (const std::size_t child_index : root.children) {
      const TreeNode &child = nodes_[child_index];
      const double score =
          player_sign(root.position.to_move()) *
              normalized_search_value(child.value, child.solved) +
          std::log(static_cast<double>(
              std::max<std::uint32_t>(1, child.visits)));
      if (score > best_score ||
          (score == best_score &&
           child.action_key < nodes_[best].action_key)) {
        best = child_index;
        best_score = score;
      }
    }
    return best;
  }

  std::vector<Move> fallback_action(
      const detail::SearchPosition &state) const {
    detail::SearchPosition current = state;
    const Player mover = current.to_move();
    std::vector<Move> action;
    while (!current.is_terminal() && current.to_move() == mover) {
      std::array<std::uint8_t, detail::kMaximumMoves> slots{};
      const std::uint8_t count = current.legal_slots(slots);
      if (count == 0) {
        throw std::logic_error("");
      }
      std::uint8_t best = slots[0];
      int best_score = std::numeric_limits<int>::min();
      for (std::uint8_t index = 0; index < count; ++index) {
        const Move move = current.move_for_slot(slots[index]);
        int score = current.grants_extra_turn(slots[index]) ? 10'000 : 0;
        const int progress = mover == Player::One
                                 ? current.ball().y - move.to.y
                                 : move.to.y - current.ball().y;
        score += progress * 100;
        if (is_attacking_goal(topology_->config(), move.to, mover)) {
          score += 1'000'000;
        } else if (is_goal_point(topology_->config(), move.to)) {
          score -= 1'000'000;
        }
        if (score > best_score ||
            (score == best_score && slots[index] < best)) {
          best = slots[index];
          best_score = score;
        }
      }
      action.push_back(current.move_for_slot(best));
      current.make_move(best);
    }
    return action;
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
        throw std::logic_error("");
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
            throw std::logic_error("");
          }
          rotated_edges_[arc.edge] = *rotated_edge;
        }
      }
    }
    if (std::find(rotated_edges_.begin(), rotated_edges_.end(),
                  kMissingEdge) != rotated_edges_.end()) {
      throw std::logic_error("");
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
          throw std::logic_error("");
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
        throw std::logic_error("");
      }
      return decoded;
    }();
    return weights;
  }

  ReplayValueOutput replay_value_output(Player mover) {
    if (topology_->edge_count() != replay_value_model::kEdgeCount ||
        topology_->vertex_count() != replay_value_model::kVertexCount) {
      throw std::logic_error("");
    }
    const auto &weights = replay_value_weights();
    ReplayValueOutput output;
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
      ++output.used_edges;
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
      for (std::size_t second = 0;
           second < replay_value_model::kHiddenTwo; ++second) {
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
    output.hidden = hidden_two;
    output.logit = replay_value_model::kB3;
    for (std::size_t hidden = 0; hidden < replay_value_model::kHiddenTwo;
         ++hidden) {
      output.logit += hidden_two[hidden] *
                      static_cast<float>(
                          weights[kOutputLayerOffset + hidden]) *
                      replay_value_model::kW3Scale;
    }
    return output;
  }

  EvaluationSnapshot make_evaluation_snapshot() {
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
    int hand_score = distance_advantage * kGoalDistanceWeight +
                     vertical_progress * kVerticalProgressWeight;
    hand_score += sign * (static_cast<int>(count) * kMobilityWeight +
                          continuations * kContinuationWeight +
                          forward_moves * kForwardMoveWeight +
                          center_alignment * kCenterAlignmentWeight +
                          kTempoWeight);
    if (direct_goal) {
      hand_score += sign * kDirectGoalWeight;
    }
    const ReplayValueOutput replay = replay_value_output(mover);
    const float replay_probability = std::tanh(replay.logit * 0.5F);
    const int replay_score = sign * static_cast<int>(
                                        std::lround(replay_probability *
                                                    kMaximumEvaluation));
    int anchor_score = hand_score;
    if (config_.replay_value_blend_percent != 0) {
      anchor_score =
          (hand_score * (100 - config_.replay_value_blend_percent) +
           replay_score * config_.replay_value_blend_percent) /
          100;
    }
    EvaluationSnapshot snapshot;
    snapshot.anchor_score =
        std::clamp(anchor_score, -kMaximumEvaluation, kMaximumEvaluation);
    snapshot.mover_sign = sign;
    for (std::size_t hidden = 0; hidden < replay.hidden.size(); ++hidden) {
      snapshot.features[hidden] =
          std::clamp(replay.hidden[hidden] * 0.25F, 0.0F, 1.0F);
    }
    snapshot.features[8] = replay_probability;
    snapshot.features[9] =
        static_cast<float>(sign * hand_score) / kMaximumEvaluation;
    snapshot.features[10] =
        static_cast<float>(sign * snapshot.anchor_score) /
        kMaximumEvaluation;
    const Point canonical_ball =
        mover == Player::One ? ball : rotated_point(ball);
    snapshot.features[11] =
        static_cast<float>(canonical_ball.x - rules.width / 2) /
        static_cast<float>(rules.width / 2);
    snapshot.features[12] =
        static_cast<float>(rules.height + 2 - 2 * canonical_ball.y) /
        static_cast<float>(rules.height + 2);
    const int mover_distance = mover == Player::One ? player_one_distance
                                                     : player_two_distance;
    const int opponent_distance = mover == Player::One ? player_two_distance
                                                        : player_one_distance;
    snapshot.features[13] =
        static_cast<float>(std::min(mover_distance, 16)) / 16.0F;
    snapshot.features[14] =
        static_cast<float>(std::min(opponent_distance, 16)) / 16.0F;
    snapshot.features[15] =
        static_cast<float>(std::clamp(sign * distance_advantage, -16, 16)) /
        16.0F;
    snapshot.features[16] = static_cast<float>(count) / 8.0F;
    snapshot.features[17] = static_cast<float>(continuations) / 8.0F;
    snapshot.features[18] = static_cast<float>(forward_moves) / 8.0F;
    snapshot.features[19] =
        static_cast<float>(center_alignment) /
        static_cast<float>(rules.width / 2);
    snapshot.features[20] = direct_goal ? 1.0F : 0.0F;
    snapshot.features[21] = is_boundary_point(rules, ball) ? 1.0F : 0.0F;
    snapshot.features[22] = count == 1 ? 1.0F : 0.0F;
    snapshot.features[23] =
        static_cast<float>(std::min<int>(replay.used_edges, 64)) / 64.0F;
    return snapshot;
  }

  float teacher_residual_prediction(
      const std::array<float, kTeacherResidualInputs> &features) const {
    static_assert(teacher_residual_model::kInputCount ==
                  kTeacherResidualInputs);
    float prediction = teacher_residual_model::kBias;
    for (std::size_t input = 0; input < features.size(); ++input) {
      prediction += features[input] * teacher_residual_model::kWeights[input];
    }
    return std::clamp(prediction, -1.0F, 1.0F);
  }

  int evaluate() {
    const EvaluationSnapshot snapshot = make_evaluation_snapshot();
    int score = snapshot.anchor_score;
    if (config_.teacher_residual_weight_percent != 0 &&
        teacher_residual_root_enabled_) {
      const int mover_correction = static_cast<int>(std::lround(
          teacher_residual_prediction(snapshot.features) *
          kTeacherResidualTargetScale *
          config_.teacher_residual_weight_percent / 100.0F));
      const int confidence_cap =
          std::max(0, kTeacherResidualHardCap -
                          std::abs(snapshot.anchor_score) / 10);
      const int correction =
          snapshot.features[20] != 0.0F ||
                  snapshot.features[23] * 64.0F <
                      kTeacherResidualMinimumUsedEdges
              ? 0
              : snapshot.mover_sign *
                    std::clamp(mover_correction, -confidence_cap,
                               confidence_cap);
      score += correction;
    }
    return std::clamp(score, -kMaximumEvaluation, kMaximumEvaluation);
  }

  int cached_evaluate() {
    const detail::PositionKey key = boundary_key(position_);
    if (const std::optional<int> cached = evaluations_.find(key)) {
      return *cached;
    }
    const int score = evaluate();
    evaluations_.store(key, score);
    return score;
  }
};

void apply_encoded_turn(GameState &state, std::string_view encoded) {
  if (encoded.empty()) {
    throw std::invalid_argument("");
  }
  GameState next = state;
  const Player mover = next.to_move;
  for (const char direction : encoded) {
    if (is_terminal(next) || next.to_move != mover) {
      throw std::invalid_argument("");
    }
    next = apply_move(next, decode_direction(next.ball, direction));
  }
  if (!is_terminal(next) && next.to_move == mover) {
    throw std::invalid_argument("");
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
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  config.absolute_deadline =
      SearchClock::now() + std::chrono::milliseconds(search_time_ms);
  FullTurnBfmSearch search(state, config);
  const std::vector<Move> action = search.run();
  if (action.empty()) {
    throw std::logic_error("");
  }
  const Player mover = state.to_move;
  std::string encoded;
  encoded.reserve(action.size());
  for (const Move move : action) {
    if (is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("");
    }
    encoded.push_back(encode_direction(state.ball, move.to));
    state = apply_move(state, move);
  }
  if (!is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("");
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
