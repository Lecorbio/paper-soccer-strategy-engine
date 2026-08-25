#include "papersoccer/bot.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "jacek_replay_bfm/jacek_replay_bfm_internal.hpp"
#include "mcts_internal.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

namespace papersoccer {
namespace {

using SearchClock = std::chrono::steady_clock;
using SearchTopology = detail::SearchTopology;
using SearchPosition = detail::SearchPosition;
using PositionKey = detail::PositionKey;

constexpr std::size_t kMaximumActions = 250;
constexpr std::size_t kMaximumPartialPaths = 50'000;
constexpr std::size_t kMaximumTreeNodes = 1'000'000;
constexpr std::size_t kProofPaths = 64;
constexpr float kMateScore = 10'000.0F;

std::uint32_t finalization_reserve_ms(std::uint32_t budget_ms) noexcept {
  // Large searches own many heap-backed compact states.  Their destruction,
  // result validation, and action-cache setup happen after the search loop,
  // so reserve up to 25 ms inside the caller's total budget.  Tiny diagnostic
  // budgets keep almost all of their time because their trees are small.
  return std::min<std::uint32_t>(25U, budget_ms / 40U);
}

constexpr std::array<Point, 8> kDirectionDeltas{{
    {0, -1}, {1, -1}, {1, 0}, {1, 1},
    {0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
}};

bool supports_rules(const RulesConfig &rules) noexcept {
  return rules.width == 8 && rules.height == 10 &&
         rules.goal_rule == GoalRule::OwnGoalsAllowed &&
         rules.blocked_rule == BlockedRule::MoverLoses;
}

bool same_game_state(const GameState &left, const GameState &right) {
  return left.config.width == right.config.width &&
         left.config.height == right.config.height &&
         left.config.goal_rule == right.config.goal_rule &&
         left.config.blocked_rule == right.config.blocked_rule &&
         left.ball == right.ball && left.to_move == right.to_move &&
         left.status == right.status && left.path == right.path &&
         left.used_segments == right.used_segments &&
         left.visit_count == right.visit_count;
}

std::uint8_t canonical_direction_index(Point from, Point to,
                                       Player mover) {
  const Point delta{to.x - from.x, to.y - from.y};
  for (std::size_t index = 0; index < kDirectionDeltas.size(); ++index) {
    if (kDirectionDeltas[index] == delta) {
      return static_cast<std::uint8_t>(
          mover == Player::Two ? (index + 4U) % kDirectionDeltas.size()
                               : index);
    }
  }
  throw std::logic_error("Jacek replay BFM produced a non-neighbour move");
}

char encode_direction(Point from, Point to, Player mover) {
  return static_cast<char>('0' + canonical_direction_index(from, to, mover));
}

class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t state) noexcept : state_(state) {}

  std::uint64_t next() noexcept {
    state_ += 0x9e3779b97f4a7c15ULL;
    return detail::mix_position_key(state_);
  }

  std::size_t index(std::size_t bound) noexcept {
    if (bound < 2U) {
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

float terminal_value(Player winner, Player perspective,
                     std::uint32_t depth) noexcept {
  const float magnitude =
      std::max(1.0F, kMateScore - 10.0F * static_cast<float>(depth));
  return winner == perspective ? magnitude : -magnitude;
}

double uct_score(float action_value, std::uint32_t parent_visits,
                 std::uint32_t child_visits, double exploration,
                 double fpu) noexcept {
  if (child_visits == 0U) {
    return static_cast<double>(action_value) + fpu;
  }
  return static_cast<double>(action_value) +
         exploration *
             std::sqrt(std::log(static_cast<double>(
                           std::max<std::uint32_t>(1U, parent_visits))) /
                       static_cast<double>(child_visits));
}

enum class TacticalClass {
  ImmediateGoal,
  ForcedCutoff,
  SafeHandoff,
  OpponentImmediateGoal,
  OwnGoal,
};

struct TurnTactics {
  bool can_score{};
  bool can_handoff{};
};

class TacticalAnalyzer {
 public:
  explicit TacticalAnalyzer(std::shared_ptr<const SearchTopology> topology)
      : topology_(std::move(topology)) {}

  TurnTactics analyze(const SearchPosition &position) const {
    if (position.is_terminal()) {
      return {};
    }
    const Player mover = position.to_move();
    std::array<bool, detail::kReplayBfmVertices> seen{};
    std::array<SearchTopology::VertexIndex, detail::kReplayBfmVertices> queue{};
    std::size_t head = 0;
    std::size_t tail = 0;
    queue[tail++] = position.ball_vertex();
    seen[position.ball_vertex()] = true;
    TurnTactics result;

    while (head < tail) {
      const SearchTopology::VertexIndex vertex = queue[head++];
      const SearchTopology::Adjacency &adjacency =
          topology_->adjacency(vertex, mover);
      for (std::uint8_t slot = 0; slot < adjacency.count; ++slot) {
        const SearchTopology::Arc &arc = adjacency.arcs[slot];
        if (position.edge_used(arc.edge)) {
          continue;
        }
        const Point destination = topology_->point(arc.destination);
        if (is_goal_point(topology_->config(), destination)) {
          result.can_score =
              result.can_score ||
              is_attacking_goal(topology_->config(), destination, mover);
          continue;
        }
        const bool rebounds =
            position.vertex_visited(arc.destination) ||
            is_boundary_point(topology_->config(), destination);
        if (!rebounds) {
          const SearchTopology::Adjacency &replies =
              topology_->adjacency(arc.destination, opponent(mover));
          for (std::uint8_t reply = 0; reply < replies.count; ++reply) {
            if (replies.arcs[reply].edge != arc.edge &&
                !position.edge_used(replies.arcs[reply].edge)) {
              result.can_handoff = true;
              break;
            }
          }
        } else if (!seen[arc.destination]) {
          seen[arc.destination] = true;
          queue[tail++] = arc.destination;
        }
      }
    }
    return result;
  }

  std::optional<std::vector<Move>> goal_path(
      const SearchPosition &position, Player goal_owner) const {
    if (position.is_terminal()) {
      return std::nullopt;
    }
    constexpr int unseen = -2;
    std::array<int, detail::kReplayBfmVertices> parent{};
    parent.fill(unseen);
    std::array<SearchTopology::VertexIndex, detail::kReplayBfmVertices> queue{};
    std::size_t head = 0;
    std::size_t tail = 0;
    const SearchTopology::VertexIndex start = position.ball_vertex();
    parent[start] = -1;
    queue[tail++] = start;

    while (head < tail) {
      const SearchTopology::VertexIndex vertex = queue[head++];
      const SearchTopology::Adjacency &adjacency =
          topology_->adjacency(vertex, position.to_move());
      std::array<std::uint8_t, detail::kMaximumMoves> ordered_slots{};
      for (std::uint8_t slot = 0; slot < adjacency.count; ++slot) {
        ordered_slots[slot] = slot;
      }
      std::sort(
          ordered_slots.begin(), ordered_slots.begin() + adjacency.count,
          [&](std::uint8_t left, std::uint8_t right) {
            return canonical_slot_rank(vertex, left, position.to_move()) <
                   canonical_slot_rank(vertex, right, position.to_move());
          });
      for (std::uint8_t order = 0; order < adjacency.count; ++order) {
        const std::uint8_t slot = ordered_slots[order];
        const SearchTopology::Arc &arc = adjacency.arcs[slot];
        if (position.edge_used(arc.edge)) {
          continue;
        }
        const Point destination = topology_->point(arc.destination);
        if (is_goal_point(topology_->config(), destination)) {
          const Player owner =
              is_attacking_goal(topology_->config(), destination, Player::One)
                  ? Player::One
                  : Player::Two;
          if (owner != goal_owner) {
            continue;
          }
          parent[arc.destination] = static_cast<int>(vertex);
          std::vector<Move> path;
          for (SearchTopology::VertexIndex current = arc.destination;
               current != start;
               current = static_cast<SearchTopology::VertexIndex>(
                   parent[current])) {
            path.push_back(Move{topology_->point(current)});
          }
          std::reverse(path.begin(), path.end());
          return path;
        }
        const bool rebounds =
            position.vertex_visited(arc.destination) ||
            is_boundary_point(topology_->config(), destination);
        if (rebounds && parent[arc.destination] == unseen) {
          parent[arc.destination] = static_cast<int>(vertex);
          queue[tail++] = arc.destination;
        }
      }
    }
    return std::nullopt;
  }

 private:
  std::shared_ptr<const SearchTopology> topology_;

  std::uint8_t canonical_slot_rank(
      SearchTopology::VertexIndex source, std::uint8_t slot,
      Player mover) const {
    if (mover == Player::One) {
      return slot;
    }
    const SearchTopology::Adjacency &physical =
        topology_->adjacency(source, Player::Two);
    const Point source_point = topology_->point(source);
    const Point destination_point =
        topology_->point(physical.arcs[slot].destination);
    const auto canonical_source = topology_->find_vertex(
        Point{8 - source_point.x, 12 - source_point.y});
    const auto canonical_destination = topology_->find_vertex(
        Point{8 - destination_point.x, 12 - destination_point.y});
    if (!canonical_source.has_value() ||
        !canonical_destination.has_value()) {
      throw std::logic_error(
          "Jacek replay BFM cannot rotate a tactical arc");
    }
    const SearchTopology::Adjacency &canonical =
        topology_->adjacency(*canonical_source, Player::One);
    for (std::uint8_t rank = 0; rank < canonical.count; ++rank) {
      if (canonical.arcs[rank].destination == *canonical_destination) {
        return rank;
      }
    }
    throw std::logic_error(
        "Jacek replay BFM cannot rank a tactical arc");
  }
};

struct CompleteTurnAction {
  std::vector<Move> moves{};
  SearchPosition result;
  TacticalClass tactical{TacticalClass::SafeHandoff};
  std::string encoded{};
};

struct GeneratorStats {
  std::uint64_t partial_paths{};
  std::uint64_t proof_paths{};
  std::uint64_t completed_actions{};
  std::uint64_t duplicate_boundaries{};
  std::uint64_t fifo_extractions{};
  std::uint64_t lifo_extractions{};
  std::uint64_t truncations{};
  std::uint64_t action_cap_stops{};
  std::uint64_t partial_cap_stops{};
  std::uint64_t deadline_stops{};
  std::uint64_t queue_drops{};
  std::uint64_t retention_drops{};
  std::uint64_t boundary_replacements{};
  std::uint64_t tactical_shortcuts{};
  std::uint64_t fallbacks{};
  std::uint64_t frontier_resumptions{};
  std::uint64_t zero_action_resumptions{};
  std::uint64_t max_frontier_depth{};
  bool deadline_reached{};
};

struct BoundaryReplacement {
  std::size_t child{};
  std::vector<Move> moves{};
  std::string encoded{};
};

struct BoundaryReference {
  PositionKey key{};
  const SearchPosition *position{};
  std::size_t child{};
  std::string_view action_key{};
};

struct GenerationResult {
  std::vector<CompleteTurnAction> actions{};
  std::vector<BoundaryReplacement> replacements{};
  GeneratorStats stats{};
  bool exhaustive{};
  bool frontier_resumable{};
};

bool position_key_less(PositionKey left, PositionKey right) noexcept {
  return left.first < right.first ||
         (left.first == right.first && left.second < right.second);
}

class CompleteTurnGenerator {
 private:
  struct Partial {
    SearchPosition position;
    std::vector<Move> moves;
  };

  struct TraversalFrame {
    std::array<std::uint8_t, detail::kMaximumMoves> slots{};
    std::uint8_t count{};
    std::uint8_t next{};
    bool initialized{};
  };

  enum class TraversalStatus {
    Action,
    Exhausted,
    PartialCap,
    Deadline,
  };

  enum class RetainResult {
    Retained,
    Duplicate,
    Full,
  };

  struct TraversalStep {
    TraversalStatus status{TraversalStatus::Exhausted};
    std::optional<Partial> action{};
    bool count_completed{true};
  };

 public:
  CompleteTurnGenerator(std::shared_ptr<const SearchTopology> topology,
                        const SearchPosition &root,
                        detail::ReplayBfmFeatureEncoder &canonicalizer,
                        std::size_t max_partial_paths,
                        SearchClock::time_point deadline,
                        std::uint64_t seed)
      : topology_(std::move(topology)), root_(root),
        canonicalizer_(canonicalizer),
        max_partial_paths_(max_partial_paths), deadline_(deadline),
        seed_(seed), tactics_(topology_), traversal_position_(root_) {}

  GenerationResult run(std::size_t max_actions,
                       std::vector<BoundaryReference> existing) {
    max_actions_ = max_actions;
    existing_ = std::move(existing);
    std::sort(existing_.begin(), existing_.end(),
              [](const BoundaryReference &left,
                 const BoundaryReference &right) {
                return position_key_less(left.key, right.key);
              });
    retained_.clear();
    retention_drops_ = 0U;

    GenerationResult output;
    output.actions.reserve(max_actions_);
    output.stats.max_frontier_depth = traversal_frames_.size();
    if (traversal_started_ && !traversal_complete_) {
      ++output.stats.frontier_resumptions;
    }
    if (!tactical_complete_) {
      retain_goal_path(output, root_.to_move());
      retain_goal_path(output, opponent(root_.to_move()));
      retain_witnesses(output);
      tactical_complete_ = true;
    }

    // Either class is a complete proof for the mover: an immediate goal wins
    // now, while a forced cutoff hands the ball to an opponent that cannot
    // finish a legal turn.  No ordinary sibling can improve on a proven win,
    // so avoid spending the rest of the decision filling an irrelevant
    // action sample.
    const auto winning = std::min_element(
        output.actions.begin(), output.actions.end(),
        [](const CompleteTurnAction &left, const CompleteTurnAction &right) {
          if (left.tactical != right.tactical) {
            return static_cast<int>(left.tactical) <
                   static_cast<int>(right.tactical);
          }
          return left.encoded < right.encoded;
        });
    if (winning != output.actions.end() &&
        (winning->tactical == TacticalClass::ImmediateGoal ||
         winning->tactical == TacticalClass::ForcedCutoff)) {
      CompleteTurnAction proof = std::move(*winning);
      output.actions.clear();
      output.actions.push_back(std::move(proof));
      output.exhaustive = false;
      ++output.stats.truncations;
      ++output.stats.tactical_shortcuts;
      output.stats.deadline_reached = deadline_reached_;
      existing_.clear();
      retained_.clear();
      return output;
    }

    bool truncated = false;
    bool action_cap_stopped = false;
    while (!truncated) {
      TraversalStep step = next_action(output.stats);
      if (step.status == TraversalStatus::Action) {
        const RetainResult retained =
            retain(output, std::move(*step.action), false,
                   step.count_completed);
        if (retained == RetainResult::Full) {
          action_cap_stopped = true;
          truncated = true;
        }
        continue;
      }
      if (step.status == TraversalStatus::PartialCap) {
        ++output.stats.partial_cap_stops;
        truncated = true;
      } else if (step.status == TraversalStatus::Deadline) {
        ++output.stats.deadline_stops;
        output.stats.deadline_reached = true;
        truncated = true;
      }
      break;
    }

    if (output.actions.empty() && existing_.empty()) {
      Partial fallback{root_, deterministic_turn()};
      fallback.position = root_;
      for (const Move move : fallback.moves) {
        std::uint8_t slot = 0;
        if (!fallback.position.slot_for_move(move, slot)) {
          throw std::logic_error("Jacek replay BFM fallback became illegal");
        }
        fallback.position.make_move(slot);
      }
      retain(output, std::move(fallback));
      ++output.stats.fallbacks;
      truncated = true;
    }

    std::stable_sort(
        output.actions.begin(), output.actions.end(),
        [](const CompleteTurnAction &left, const CompleteTurnAction &right) {
          if (left.tactical != right.tactical) {
            return static_cast<int>(left.tactical) <
                   static_cast<int>(right.tactical);
          }
          return left.encoded < right.encoded;
        });
    output.stats.action_cap_stops = action_cap_stopped ? 1U : 0U;
    output.stats.retention_drops = retention_drops_;
    output.exhaustive =
        traversal_complete_ && !deferred_action_.has_value() && !truncated;
    output.frontier_resumable =
        !output.exhaustive && output.stats.partial_cap_stops != 0U &&
        !output.stats.deadline_reached;
    if (output.actions.empty() && output.frontier_resumable) {
      ++output.stats.zero_action_resumptions;
    }
    output.stats.deadline_reached =
        output.stats.deadline_reached || deadline_reached_;
    if (truncated) {
      ++output.stats.truncations;
    }
    existing_.clear();
    retained_.clear();
    return output;
  }

 private:
  std::shared_ptr<const SearchTopology> topology_;
  SearchPosition root_;
  detail::ReplayBfmFeatureEncoder &canonicalizer_;
  std::size_t max_actions_{};
  std::size_t max_partial_paths_{};
  SearchClock::time_point deadline_{};
  std::uint64_t seed_{};
  TacticalAnalyzer tactics_;
  SearchPosition traversal_position_;
  std::vector<Move> traversal_moves_{};
  std::vector<TraversalFrame> traversal_frames_{};
  std::optional<Partial> deferred_action_{};
  std::vector<BoundaryReference> existing_{};
  std::vector<PositionKey> retained_{};
  bool tactical_complete_{};
  bool traversal_started_{};
  bool traversal_complete_{};
  bool deadline_reached_{};
  std::uint64_t retention_drops_{};

  bool expired() {
    if (!deadline_reached_ && SearchClock::now() >= deadline_) {
      deadline_reached_ = true;
    }
    return deadline_reached_;
  }

  TraversalStep next_action(GeneratorStats &stats) {
    if (deferred_action_.has_value()) {
      Partial action = std::move(*deferred_action_);
      deferred_action_.reset();
      return {TraversalStatus::Action, std::move(action), false};
    }
    if (traversal_complete_) {
      return {TraversalStatus::Exhausted, std::nullopt};
    }
    if (!traversal_started_) {
      traversal_started_ = true;
      traversal_frames_.push_back(TraversalFrame{});
    }

    for (;;) {
      if (expired()) {
        return {TraversalStatus::Deadline, std::nullopt};
      }
      if (traversal_frames_.empty()) {
        traversal_complete_ = true;
        return {TraversalStatus::Exhausted, std::nullopt};
      }

      TraversalFrame &frame = traversal_frames_.back();
      if (!frame.initialized) {
        if (stats.partial_paths >= max_partial_paths_) {
          return {TraversalStatus::PartialCap, std::nullopt};
        }
        ++stats.partial_paths;
        ++stats.lifo_extractions;
        frame.count = traversal_position_.legal_slots(frame.slots);
        shuffle(frame.slots, frame.count, traversal_position_);
        frame.initialized = true;
        stats.max_frontier_depth = std::max<std::uint64_t>(
            stats.max_frontier_depth, traversal_frames_.size());
      }

      if (frame.next >= frame.count) {
        traversal_frames_.pop_back();
        if (traversal_frames_.empty()) {
          traversal_complete_ = true;
          return {TraversalStatus::Exhausted, std::nullopt};
        }
        traversal_position_.unmake_move();
        traversal_moves_.pop_back();
        continue;
      }

      const std::uint8_t slot = frame.slots[frame.next++];
      traversal_moves_.push_back(traversal_position_.move_for_slot(slot));
      traversal_position_.make_move(slot);
      if (traversal_position_.is_terminal() ||
          traversal_position_.to_move() != root_.to_move()) {
        Partial result{traversal_position_, traversal_moves_};
        traversal_position_.unmake_move();
        traversal_moves_.pop_back();
        return {TraversalStatus::Action, std::move(result)};
      }
      traversal_frames_.push_back(TraversalFrame{});
    }
  }

  void shuffle(std::array<std::uint8_t, detail::kMaximumMoves> &slots,
               std::uint8_t count,
               const SearchPosition &position) const noexcept {
    std::sort(slots.begin(), slots.begin() + count,
              [&](std::uint8_t left, std::uint8_t right) {
                return canonicalizer_.canonical_slot_rank(position, left) <
                       canonicalizer_.canonical_slot_rank(position, right);
              });
    const PositionKey key = canonicalizer_.canonical_position_key(position);
    SplitMix64 random(seed_ ^ key.first ^
                      detail::mix_position_key(key.second));
    for (std::size_t remaining = count; remaining > 1U; --remaining) {
      std::swap(slots[remaining - 1U], slots[random.index(remaining)]);
    }
  }

  std::string encode(const std::vector<Move> &moves) const {
    SearchPosition position = root_;
    std::string result;
    result.reserve(moves.size());
    for (const Move move : moves) {
      result.push_back(
          encode_direction(position.ball(), move.to, root_.to_move()));
      std::uint8_t slot = 0;
      if (!position.slot_for_move(move, slot)) {
        throw std::logic_error("Jacek replay BFM encoded an illegal action");
      }
      position.make_move(slot);
    }
    return result;
  }

  TacticalClass classify(const SearchPosition &result) const {
    if (result.is_terminal()) {
      return result.winner() == root_.to_move()
                 ? TacticalClass::ImmediateGoal
                 : TacticalClass::OwnGoal;
    }
    const TurnTactics next = tactics_.analyze(result);
    if (next.can_score) {
      return TacticalClass::OpponentImmediateGoal;
    }
    if (!next.can_handoff) {
      return TacticalClass::ForcedCutoff;
    }
    return TacticalClass::SafeHandoff;
  }

  RetainResult retain(GenerationResult &output, Partial partial,
                      bool replace_when_full = true,
                      bool count_completed = true) {
    output.stats.completed_actions += count_completed ? 1U : 0U;
    const PositionKey key = partial.position.position_key();
    auto existing = std::lower_bound(
        existing_.begin(), existing_.end(), key,
        [](const BoundaryReference &candidate, PositionKey expected) {
          return position_key_less(candidate.key, expected);
        });
    while (existing != existing_.end() && existing->key == key) {
      if (existing->position != nullptr &&
          existing->position->same_compact_state(partial.position)) {
        ++output.stats.duplicate_boundaries;
        std::string encoded = encode(partial.moves);
        auto replacement = std::find_if(
            output.replacements.begin(), output.replacements.end(),
            [&](const BoundaryReplacement &candidate) {
              return candidate.child == existing->child;
            });
        const std::string_view retained =
            replacement == output.replacements.end()
                ? existing->action_key
                : std::string_view(replacement->encoded);
        if (encoded < retained) {
          if (replacement == output.replacements.end()) {
            output.replacements.push_back(BoundaryReplacement{
                existing->child, std::move(partial.moves),
                std::move(encoded)});
            ++output.stats.boundary_replacements;
          } else {
            replacement->moves = std::move(partial.moves);
            replacement->encoded = std::move(encoded);
          }
        }
        return RetainResult::Duplicate;
      }
      ++existing;
    }
    for (std::size_t index = 0; index < retained_.size(); ++index) {
      if (retained_[index] != key ||
          !output.actions[index].result.same_compact_state(partial.position)) {
        continue;
      }
      ++output.stats.duplicate_boundaries;
      std::string encoded = encode(partial.moves);
      if (encoded < output.actions[index].encoded) {
        output.actions[index].moves = std::move(partial.moves);
        output.actions[index].result = std::move(partial.position);
        output.actions[index].encoded = std::move(encoded);
      }
      return RetainResult::Duplicate;
    }

    if (output.actions.size() >= max_actions_ && !replace_when_full) {
      deferred_action_ = std::move(partial);
      return RetainResult::Full;
    }

    const TacticalClass tactical = classify(partial.position);
    CompleteTurnAction candidate{
        std::move(partial.moves), std::move(partial.position), tactical, {}};
    candidate.encoded = encode(candidate.moves);
    if (output.actions.size() < max_actions_) {
      retained_.push_back(key);
      output.actions.push_back(std::move(candidate));
      return RetainResult::Retained;
    }

    const auto worse = [](const CompleteTurnAction &left,
                          const CompleteTurnAction &right) {
      if (left.tactical != right.tactical) {
        return static_cast<int>(left.tactical) >
               static_cast<int>(right.tactical);
      }
      return left.encoded > right.encoded;
    };
    std::size_t worst = 0;
    for (std::size_t index = 1; index < output.actions.size(); ++index) {
      if (worse(output.actions[index], output.actions[worst])) {
        worst = index;
      }
    }
    if (worse(output.actions[worst], candidate)) {
      retained_[worst] = key;
      output.actions[worst] = std::move(candidate);
    }
    ++retention_drops_;
    return RetainResult::Retained;
  }

  void retain_goal_path(GenerationResult &output, Player goal_owner) {
    if (expired()) {
      return;
    }
    const auto path = tactics_.goal_path(root_, goal_owner);
    if (!path.has_value()) {
      return;
    }
    Partial partial{root_, *path};
    for (const Move move : partial.moves) {
      std::uint8_t slot = 0;
      if (!partial.position.slot_for_move(move, slot)) {
        throw std::logic_error(
            "Jacek replay BFM tactical goal path became illegal");
      }
      partial.position.make_move(slot);
    }
    retain(output, std::move(partial));
  }

  void retain_witnesses(GenerationResult &output) {
    std::array<std::optional<Partial>, 3> best{};
    std::deque<Partial> pending;
    pending.push_back(Partial{root_, {}});
    while (!pending.empty() && output.stats.proof_paths < kProofPaths) {
      if (expired()) {
        break;
      }
      ++output.stats.proof_paths;
      Partial partial = std::move(pending.front());
      pending.pop_front();
      std::array<std::uint8_t, detail::kMaximumMoves> slots{};
      const std::uint8_t count = partial.position.legal_slots(slots);
      shuffle(slots, count, partial.position);
      for (std::uint8_t index = 0; index < count; ++index) {
        Partial child = partial;
        child.moves.push_back(child.position.move_for_slot(slots[index]));
        child.position.make_move(slots[index]);
        if (child.position.is_terminal() ||
            child.position.to_move() != root_.to_move()) {
          const int witness =
              static_cast<int>(classify(child.position)) -
              static_cast<int>(TacticalClass::ForcedCutoff);
          if (witness < 0 || witness >= static_cast<int>(best.size())) {
            continue;
          }
          std::optional<Partial> &retained =
              best[static_cast<std::size_t>(witness)];
          if (!retained.has_value() ||
              encode(child.moves) < encode(retained->moves)) {
            retained = std::move(child);
          }
        } else if (pending.size() < kProofPaths) {
          pending.push_back(std::move(child));
        }
      }
    }
    for (std::optional<Partial> &witness : best) {
      if (witness.has_value()) {
        retain(output, std::move(*witness));
      }
    }
  }

  std::vector<Move> deterministic_turn() const {
    SearchPosition position = root_;
    const Player mover = root_.to_move();
    std::vector<Move> result;
    while (!position.is_terminal() && position.to_move() == mover) {
      std::array<std::uint8_t, detail::kMaximumMoves> slots{};
      const std::uint8_t count = position.legal_slots(slots);
      if (count == 0U) {
        throw std::logic_error("Jacek replay BFM fallback has no legal edge");
      }
      shuffle(slots, count, position);
      result.push_back(position.move_for_slot(slots[0]));
      position.make_move(slots[0]);
    }
    return result;
  }
};

struct SearchResult {
  std::vector<Move> action{};
  JacekReplayBfmSearchStats stats{};
};

class BestFirstMinimaxSearch {
 public:
  BestFirstMinimaxSearch(const GameState &state,
                         const JacekReplayBfmConfig &config,
                         const detail::ReplayBfmModel &model)
      : config_(config), model_(model),
        topology_(std::make_shared<SearchTopology>(state.config)),
        root_(topology_, state), encoder_(topology_),
        deadline_(SearchClock::now() +
                  std::chrono::milliseconds(
                      config.max_time_ms -
                      finalization_reserve_ms(config.max_time_ms))) {
    nodes_.reserve(std::min<std::size_t>(config_.max_tree_nodes, 4096U));
  }

  SearchResult run() {
    const float root_value = evaluate(root_);
    nodes_.emplace_back(root_, no_parent(), std::vector<Move>{},
                        TacticalClass::SafeHandoff, root_.to_move(),
                        root_value, 0U, false, false, false);
    if (!expand(0U)) {
      throw std::logic_error("Jacek replay BFM could not expand its root");
    }
    refresh(0U);
    for (;;) {
      if (nodes_[0].closed) {
        stats_.termination =
            nodes_[0].solved
                ? JacekReplayBfmSearchTermination::RootSolved
                : JacekReplayBfmSearchTermination::ClosedUnsolvedRoot;
        break;
      }
      if (!budget_available()) {
        stats_.termination =
            stats_.tree_cap_reached
                ? JacekReplayBfmSearchTermination::FixedWorkCap
                : JacekReplayBfmSearchTermination::Deadline;
        break;
      }
      const auto path = select_path();
      if (!path.has_value()) {
        stats_.termination =
            JacekReplayBfmSearchTermination::NoSelectableFrontier;
        break;
      }
      for (const std::size_t index : *path) {
        ++nodes_[index].visits;
        ++nodes_[index].selection_visits;
        ++stats_.visits;
      }
      if (!expand(path->back())) {
        if (stats_.tree_cap_reached) {
          stats_.termination =
              JacekReplayBfmSearchTermination::FixedWorkCap;
        } else if (stats_.deadline_reached) {
          stats_.termination = JacekReplayBfmSearchTermination::Deadline;
        } else {
          stats_.termination =
              JacekReplayBfmSearchTermination::ExpansionFailed;
        }
        break;
      }
      backup(*path);
    }
    return make_result();
  }

 private:
  struct Node {
    SearchPosition position;
    std::size_t parent{};
    std::vector<std::size_t> children{};
    std::vector<Move> incoming_action{};
    TacticalClass tactical{TacticalClass::SafeHandoff};
    Player perspective{Player::One};
    float value{};
    float prior_value{};
    std::uint32_t visits{1};
    std::uint32_t selection_visits{};
    std::uint32_t depth{};
    bool expanded{};
    bool solved{};
    bool exhaustive{};
    bool closed{};
    std::string action_key{};
    std::unique_ptr<CompleteTurnGenerator> action_frontier{};

    Node(SearchPosition state, std::size_t parent_index,
         std::vector<Move> action, TacticalClass classification,
         Player value_perspective, float node_value,
         std::uint32_t node_depth, bool is_expanded,
         bool is_solved, bool is_exhaustive, std::string key = {})
        : position(std::move(state)), parent(parent_index),
          incoming_action(std::move(action)), tactical(classification),
          perspective(value_perspective), value(node_value),
          prior_value(node_value), depth(node_depth), expanded(is_expanded),
          solved(is_solved), exhaustive(is_exhaustive), closed(is_solved),
          action_key(std::move(key)) {}
  };

  const JacekReplayBfmConfig &config_;
  const detail::ReplayBfmModel &model_;
  std::shared_ptr<const SearchTopology> topology_;
  SearchPosition root_;
  detail::ReplayBfmFeatureEncoder encoder_;
  SearchClock::time_point deadline_{};
  std::vector<Node> nodes_{};
  JacekReplayBfmSearchStats stats_{};

  static constexpr std::size_t no_parent() noexcept {
    return std::numeric_limits<std::size_t>::max();
  }

  bool expired() {
    if (SearchClock::now() >= deadline_) {
      stats_.deadline_reached = true;
      return true;
    }
    return false;
  }

  bool budget_available() {
    if (nodes_.size() >= config_.max_tree_nodes) {
      stats_.tree_cap_reached = true;
      return false;
    }
    return !expired();
  }

  float evaluate(const SearchPosition &position) {
    ++stats_.neural_evaluations;
    return model_.evaluate(encoder_.encode(position));
  }

  float evaluate_delta(
      const SearchPosition &position,
      const detail::ReplayBfmPreparedEvaluation &base) {
    ++stats_.neural_evaluations;
    return model_.evaluate_delta(base, encoder_.encode(position));
  }

  void merge(const GeneratorStats &source,
             std::size_t retained_actions) noexcept {
    stats_.generated_actions += source.completed_actions;
    stats_.retained_actions += retained_actions;
    stats_.completed_actions += source.completed_actions;
    stats_.duplicate_boundaries += source.duplicate_boundaries;
    stats_.partial_paths += source.partial_paths;
    stats_.fifo_extractions += source.fifo_extractions;
    stats_.lifo_extractions += source.lifo_extractions;
    stats_.tactical_proofs += source.proof_paths;
    stats_.truncations += source.truncations;
    stats_.generation_action_cap_stops += source.action_cap_stops;
    stats_.generation_partial_cap_stops += source.partial_cap_stops;
    stats_.generation_deadline_stops += source.deadline_stops;
    stats_.generation_queue_drops += source.queue_drops;
    stats_.generation_retention_drops += source.retention_drops;
    stats_.generation_boundary_replacements += source.boundary_replacements;
    stats_.generation_tactical_shortcuts += source.tactical_shortcuts;
    stats_.generation_fallbacks += source.fallbacks;
    stats_.generation_frontier_resumptions += source.frontier_resumptions;
    stats_.generation_zero_action_resumptions +=
        source.zero_action_resumptions;
    stats_.generation_max_frontier_depth = std::max(
        stats_.generation_max_frontier_depth, source.max_frontier_depth);
    stats_.deadline_reached =
        stats_.deadline_reached || source.deadline_reached;
  }

  bool expand(std::size_t index) {
    if (nodes_[index].solved ||
        (nodes_[index].expanded && nodes_[index].exhaustive)) {
      return true;
    }
    if (nodes_.size() >= config_.max_tree_nodes) {
      stats_.tree_cap_reached = true;
      return false;
    }
    // A truncated complete-turn page leaves an implicit action frontier.  It
    // becomes selectable only after every materialized child drains; resume
    // its saved DFS cursor to materialize the next deterministic page of real
    // tree nodes without replaying or dropping partial turns.
    const bool widening = nodes_[index].expanded;
    if (widening &&
        std::any_of(nodes_[index].children.begin(),
                    nodes_[index].children.end(),
                    [&](std::size_t child) {
                      return !nodes_[child].closed;
                    })) {
      throw std::logic_error(
          "Jacek replay BFM widened before draining sampled children");
    }
    std::vector<BoundaryReference> existing;
    existing.reserve(nodes_[index].children.size());
    for (const std::size_t child : nodes_[index].children) {
      existing.push_back(
          BoundaryReference{nodes_[child].position.position_key(),
                            &nodes_[child].position, child,
                            nodes_[child].action_key});
    }
    const std::size_t remaining = config_.max_tree_nodes - nodes_.size();
    if (!nodes_[index].action_frontier) {
      nodes_[index].action_frontier =
          std::make_unique<CompleteTurnGenerator>(
              topology_, nodes_[index].position, encoder_,
              config_.max_partial_paths, deadline_, config_.seed);
    }
    GenerationResult generated = nodes_[index].action_frontier->run(
        std::min(config_.max_actions, remaining), std::move(existing));
    if (widening) {
      ++stats_.progressive_widenings;
    }
    for (BoundaryReplacement &replacement : generated.replacements) {
      nodes_[replacement.child].action_key =
          std::move(replacement.encoded);
      if (index == 0U) {
        nodes_[replacement.child].incoming_action =
            std::move(replacement.moves);
      }
    }
    merge(generated.stats, generated.actions.size());
    if (generated.actions.empty()) {
      nodes_[index].expanded = true;
      nodes_[index].exhaustive = generated.exhaustive;
      if (generated.exhaustive) {
        refresh(index);
      }
      ++stats_.expansions;
      stats_.tree_nodes = nodes_.size();
      return generated.exhaustive || generated.frontier_resumable;
    }

    const Player child_perspective = opponent(nodes_[index].perspective);
    const std::uint32_t child_depth = nodes_[index].depth + 1U;
    stats_.max_complete_turn_depth =
        std::max(stats_.max_complete_turn_depth, child_depth);
    const std::size_t heuristic_actions = static_cast<std::size_t>(
        std::count_if(generated.actions.begin(), generated.actions.end(),
                      [](const CompleteTurnAction &action) {
                        return !action.result.is_terminal() &&
                               action.tactical != TacticalClass::ForcedCutoff &&
                               action.tactical !=
                                   TacticalClass::OpponentImmediateGoal;
                      }));
    std::optional<detail::ReplayBfmPreparedEvaluation> prepared;
    if (heuristic_actions > 1U) {
      prepared = model_.prepare(
          encoder_.encode(nodes_[index].position, child_perspective));
    }
    const std::size_t first_child = nodes_.size();
    std::size_t materialized_actions = 0;
    for (CompleteTurnAction &action : generated.actions) {
      if (materialized_actions != 0U && expired()) {
        ++stats_.materialization_deadline_stops;
        break;
      }
      float value = 0.0F;
      bool solved = false;
      if (action.result.is_terminal()) {
        value = terminal_value(*action.result.winner(), child_perspective,
                               child_depth);
        solved = true;
      } else if (action.tactical == TacticalClass::ForcedCutoff) {
        value = terminal_value(opponent(child_perspective),
                               child_perspective, child_depth + 1U);
        solved = true;
      } else if (action.tactical ==
                 TacticalClass::OpponentImmediateGoal) {
        value = terminal_value(child_perspective, child_perspective,
                               child_depth + 1U);
        solved = true;
      } else {
        value = prepared.has_value()
                    ? evaluate_delta(action.result, *prepared)
                    : evaluate(action.result);
      }
      if (solved) {
        ++stats_.tactical_solutions;
      }
      // Only root-child transcripts can be returned to the caller.  Keeping
      // every deeper complete-turn vector adds an allocation per tree node
      // and makes deadline cleanup noticeably expensive at large searches.
      std::vector<Move> incoming_action;
      if (index == 0U) {
        incoming_action = std::move(action.moves);
      }
      nodes_.emplace_back(
          std::move(action.result), index, std::move(incoming_action),
          action.tactical, child_perspective, value, child_depth,
          false, solved, true, std::move(action.encoded));
      ++materialized_actions;
    }
    nodes_[index].children.reserve(
        nodes_[index].children.size() + materialized_actions);
    for (std::size_t child = first_child; child < nodes_.size(); ++child) {
      nodes_[index].children.push_back(child);
    }
    nodes_[index].expanded = true;
    // Unexpanded children use exhaustive=true only as a leaf sentinel.  A
    // real expansion must overwrite that sentinel; preserving it across a
    // truncated page would turn sampled losses into a false universal proof.
    nodes_[index].exhaustive =
        generated.exhaustive &&
        materialized_actions == generated.actions.size();
    ++stats_.expansions;
    stats_.tree_nodes = nodes_.size();
    refresh(index);
    return true;
  }

  void refresh(std::size_t index) {
    Node &node = nodes_[index];
    if (node.children.empty()) {
      return;
    }
    float best = -std::numeric_limits<float>::infinity();
    bool proven_win = false;
    bool all_solved = node.exhaustive;
    bool all_closed = true;
    std::size_t open_children = 0U;
    for (const std::size_t child_index : node.children) {
      const Node &child = nodes_[child_index];
      const float action_value = -child.value;
      best = std::max(best, action_value);
      proven_win = proven_win || (child.solved && action_value > 0.0F);
      all_solved = all_solved && child.solved;
      all_closed = all_closed && child.closed;
      open_children += child.closed ? 0U : 1U;
    }
    if (open_children > config_.max_actions) {
      throw std::logic_error(
          "Jacek replay BFM exceeded its sampled frontier width");
    }
    stats_.max_open_children =
        std::max(stats_.max_open_children, open_children);
    if (!node.exhaustive) {
      best = std::max(best, node.prior_value);
    }
    node.value = best;
    node.solved = proven_win || all_solved;
    node.closed = node.solved || (node.exhaustive && all_closed);
    if (node.solved || node.exhaustive) {
      node.action_frontier.reset();
    }
  }

  std::optional<std::vector<std::size_t>> select_path() const {
    std::vector<std::size_t> path{0U};
    std::size_t current = 0U;
    while (nodes_[current].expanded) {
      const Node &parent = nodes_[current];
      std::optional<std::size_t> best;
      double best_score = -std::numeric_limits<double>::infinity();
      for (const std::size_t child_index : parent.children) {
        const Node &child = nodes_[child_index];
        if (child.closed) {
          continue;
        }
        const double score =
            uct_score(-child.value, parent.selection_visits,
                      child.selection_visits, config_.exploration,
                      config_.fpu);
        if (!best.has_value() || score > best_score ||
            (score == best_score &&
             child.action_key < nodes_[*best].action_key)) {
          best = child_index;
          best_score = score;
        }
      }
      if (!best.has_value()) {
        if (!parent.solved && !parent.exhaustive) {
          return path;
        }
        return std::nullopt;
      }
      current = *best;
      path.push_back(current);
    }
    return path;
  }

  void backup(const std::vector<std::size_t> &path) {
    for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
      refresh(*iterator);
    }
  }

  SearchResult make_result() {
    if (nodes_.empty() || nodes_[0].children.empty()) {
      throw std::logic_error("Jacek replay BFM has no root action");
    }
    const Node &root = nodes_[0];
    std::size_t best = root.children.front();
    double best_score = -std::numeric_limits<double>::infinity();
    for (const std::size_t child_index : root.children) {
      const Node &child = nodes_[child_index];
      const double score =
          static_cast<double>(-child.value) +
          std::log(static_cast<double>(std::max<std::uint32_t>(
              1U, child.visits)));
      if (score > best_score ||
          (score == best_score &&
           child.action_key < nodes_[best].action_key)) {
        best = child_index;
        best_score = score;
      }
    }
    stats_.root_value = root.value;
    stats_.root_solved = root.solved;
    if (root.solved) {
      if (!std::isfinite(root.value) || root.value == 0.0F) {
        throw std::logic_error(
            "Jacek replay BFM solved root has no finite winner sign");
      }
      stats_.proven_winner =
          root.value > 0.0F ? root_.to_move() : opponent(root_.to_move());
    }
    for (const Node &node : nodes_) {
      if (node.closed && !node.solved) {
        ++stats_.closed_unsolved_nodes;
        if (!node.exhaustive) {
          ++stats_.closed_unsolved_nonexhaustive_nodes;
        }
      }
      if (!node.closed && !node.solved && !node.expanded) {
        ++stats_.open_unexpanded_nodes;
      } else if (!node.closed && !node.solved && !node.exhaustive) {
        const bool has_open_child = std::any_of(
            node.children.begin(), node.children.end(),
            [&](std::size_t child) { return !nodes_[child].closed; });
        if (!has_open_child) {
          ++stats_.implicit_action_frontiers;
        }
      }
    }
    stats_.tree_nodes = nodes_.size();
    return SearchResult{nodes_[best].incoming_action, stats_};
  }
};

void validate_action(const GameState &state,
                     const std::vector<Move> &action) {
  if (action.empty()) {
    throw std::logic_error("Jacek replay BFM returned an empty action");
  }
  GameState replayed = state;
  const Player mover = state.to_move;
  for (const Move move : action) {
    if (is_terminal(replayed) || replayed.to_move != mover) {
      throw std::logic_error("Jacek replay BFM returned an overlong action");
    }
    const std::vector<Move> legal = legal_moves(replayed);
    if (std::find(legal.begin(), legal.end(), move) == legal.end()) {
      throw std::logic_error("Jacek replay BFM returned an illegal action");
    }
    replayed = apply_move(replayed, move);
  }
  if (!is_terminal(replayed) && replayed.to_move == mover) {
    throw std::logic_error("Jacek replay BFM returned an incomplete action");
  }
}

JacekReplayBfmConfig validate_config(JacekReplayBfmConfig config) {
  if (config.model_path.empty() || config.max_time_ms == 0U ||
      config.max_tree_nodes < 2U ||
      config.max_tree_nodes > kMaximumTreeNodes ||
      config.max_actions == 0U || config.max_actions > kMaximumActions ||
      config.max_partial_paths == 0U ||
      config.max_partial_paths > kMaximumPartialPaths ||
      !std::isfinite(config.exploration) || config.exploration < 0.0 ||
      !std::isfinite(config.fpu) || config.fpu < -1.0 ||
      config.fpu > 1.0) {
    throw std::invalid_argument("invalid Jacek replay BFM configuration");
  }
  return config;
}

}  // namespace

std::string_view jacek_replay_bfm_search_termination_name(
    JacekReplayBfmSearchTermination termination) noexcept {
  switch (termination) {
    case JacekReplayBfmSearchTermination::NotStarted:
      return "not-started";
    case JacekReplayBfmSearchTermination::RootSolved:
      return "root-solved";
    case JacekReplayBfmSearchTermination::FixedWorkCap:
      return "fixed-work-cap";
    case JacekReplayBfmSearchTermination::Deadline:
      return "deadline";
    case JacekReplayBfmSearchTermination::ClosedUnsolvedRoot:
      return "closed-unsolved-root";
    case JacekReplayBfmSearchTermination::NoSelectableFrontier:
      return "no-selectable-frontier";
    case JacekReplayBfmSearchTermination::ExpansionFailed:
      return "expansion-failed";
  }
  return "invalid";
}

class JacekReplayBfmBot::Impl {
 public:
  explicit Impl(JacekReplayBfmConfig config)
      : config_(validate_config(std::move(config))),
        model_(config_.model_path) {}

  Move choose_move(const GameState &state) {
    if (!supports_rules(state.config)) {
      clear_cache();
      throw std::invalid_argument(
          "JacekReplayBfmBot requires 8x10 CodinGame rules");
    }
    if (is_terminal(state)) {
      clear_cache();
      throw std::invalid_argument(
          "JacekReplayBfmBot cannot move in a terminal state");
    }

    if (expected_state_.has_value() &&
        same_game_state(state, *expected_state_) &&
        next_cached_move_ < cached_action_.size()) {
      const std::size_t edge_index = next_cached_move_;
      const Move move = cached_action_[next_cached_move_++];
      last_stats_ = {};
      last_stats_.cached_continuation = true;
      last_stats_.planned_action_length = cached_action_.size();
      last_stats_.current_edge_index = edge_index;
      last_stats_.cached_moves_remaining =
          cached_action_.size() - next_cached_move_;
      last_stats_.searches = searches_;
      if (next_cached_move_ < cached_action_.size()) {
        expected_state_ = apply_move(state, move);
      } else {
        clear_cache();
      }
      return move;
    }
    clear_cache();

    BestFirstMinimaxSearch search(state, config_, model_);
    SearchResult result = search.run();
    validate_action(state, result.action);
    cached_action_ = std::move(result.action);
    ++searches_;
    last_stats_ = result.stats;
    last_stats_.planned_action_length = cached_action_.size();
    last_stats_.current_edge_index = 0U;
    last_stats_.cached_moves_remaining = cached_action_.size() - 1U;
    last_stats_.searches = searches_;

    const Move move = cached_action_.front();
    next_cached_move_ = 1U;
    if (next_cached_move_ < cached_action_.size()) {
      expected_state_ = apply_move(state, move);
    } else {
      clear_cache();
    }
    return move;
  }

  JacekReplayBfmSearchStats analyze_position(
      const GameState &state, std::uint64_t seed) const {
    if (!supports_rules(state.config)) {
      throw std::invalid_argument(
          "JacekReplayBfmBot requires 8x10 CodinGame rules");
    }
    if (is_terminal(state)) {
      throw std::invalid_argument(
          "JacekReplayBfmBot cannot analyze a terminal state");
    }

    JacekReplayBfmConfig analysis_config = config_;
    analysis_config.seed = seed;
    BestFirstMinimaxSearch search(state, analysis_config, model_);
    SearchResult result = search.run();
    validate_action(state, result.action);
    result.stats.planned_action_length = result.action.size();
    return result.stats;
  }

  const JacekReplayBfmConfig &config() const noexcept { return config_; }
  const JacekReplayBfmSearchStats &stats() const noexcept {
    return last_stats_;
  }
  std::string_view model_sha256() const noexcept {
    return model_.model_sha256();
  }

 private:
  JacekReplayBfmConfig config_;
  detail::ReplayBfmModel model_;
  JacekReplayBfmSearchStats last_stats_{};
  std::vector<Move> cached_action_{};
  std::size_t next_cached_move_{};
  std::optional<GameState> expected_state_{};
  std::uint64_t searches_{};

  void clear_cache() noexcept {
    cached_action_.clear();
    next_cached_move_ = 0U;
    expected_state_.reset();
  }
};

JacekReplayBfmBot::JacekReplayBfmBot(JacekReplayBfmConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

JacekReplayBfmBot::~JacekReplayBfmBot() = default;
JacekReplayBfmBot::JacekReplayBfmBot(JacekReplayBfmBot &&) noexcept = default;
JacekReplayBfmBot &JacekReplayBfmBot::operator=(
    JacekReplayBfmBot &&) noexcept = default;

std::string_view JacekReplayBfmBot::name() const noexcept {
  return "JacekReplayBfmBot";
}

Move JacekReplayBfmBot::choose_move(const GameState &state) {
  return impl_->choose_move(state);
}

JacekReplayBfmSearchStats JacekReplayBfmBot::analyze_position(
    const GameState &state, std::uint64_t seed) const {
  return impl_->analyze_position(state, seed);
}

const JacekReplayBfmConfig &JacekReplayBfmBot::config() const noexcept {
  return impl_->config();
}

const JacekReplayBfmSearchStats &
JacekReplayBfmBot::last_search_stats() const noexcept {
  return impl_->stats();
}

std::string_view JacekReplayBfmBot::model_sha256() const noexcept {
  return impl_->model_sha256();
}

std::string_view JacekReplayBfmBot::feature_schema_sha256() noexcept {
  return detail::kReplayBfmFeatureSchemaSha256;
}

}  // namespace papersoccer
