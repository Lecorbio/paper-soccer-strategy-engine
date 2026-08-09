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
#include <utility>
#include <vector>

#include "mcts_internal.hpp"
#include "neural_puct_model.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

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
    // Training uses binary cross entropy with target (value + 1) / 2.
    // Therefore 2 * sigmoid(logit) - 1, not tanh(logit), is the calibrated
    // mover-relative expectation.
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
