#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace compact_value_bfm {

inline constexpr int kVertexCount = 105;
inline constexpr int kEdgeCount = 316;
inline constexpr int kVertexCategories = 57;
inline constexpr int kFeatureCount =
    kEdgeCount + kVertexCount * kVertexCategories;
inline constexpr int kMaximumActiveFeatures = kEdgeCount + kVertexCount;
inline constexpr int kMaximumActionLength = kEdgeCount;
inline constexpr std::size_t kMaximumActions = 250;
inline constexpr std::size_t kMaximumPartialPaths = 50'000;
inline constexpr std::size_t kRootPartialPaths = 4'000;
inline constexpr std::size_t kNonrootPartialPaths = 512;
inline constexpr std::size_t kMaximumTreeNodes = 120'000;
inline constexpr std::size_t kProductionTreeNodes = 80'000;
inline constexpr std::uint64_t kMaximumExpansions = 2'000'000;
inline constexpr std::size_t kProofPaths = 64;
inline constexpr double kExploration = 0.95;
inline constexpr double kFirstPlayUrgency = 0.5;
inline constexpr double kFinalVisitWeight = 1.0;
inline constexpr float kMateScore = 10'000.0F;

static_assert(kFeatureCount == 6301);

struct Action {
  std::array<std::uint8_t, kMaximumActionLength> directions{};
  std::uint16_t length{};

  bool operator==(const Action &other) const noexcept;
  std::string text() const;
};

struct State {
  std::array<std::uint64_t, 5> used{};
  std::uint8_t ball{};
  std::uint8_t to_move{};
  std::int8_t winner{-1};
  std::uint16_t ply{};

  bool terminal() const noexcept { return winner >= 0; }
};

struct SparseFeatures {
  std::array<std::uint16_t, kMaximumActiveFeatures> indices{};
  std::uint16_t count{};

  bool operator==(const SparseFeatures &) const noexcept = default;
};

class Topology {
 public:
  struct Arc {
    std::uint16_t edge{};
    std::uint8_t destination{};
    std::uint8_t direction{};
  };

  static const Topology &get();
  int x(int vertex) const noexcept { return xs_[vertex]; }
  int y(int vertex) const noexcept { return ys_[vertex]; }
  int vertex_at(int x, int y) const noexcept;
  int degree(int vertex) const noexcept { return degrees_[vertex]; }
  Arc arc(int vertex, int index) const noexcept { return arcs_[vertex][index]; }
  int rotated_vertex(int vertex) const noexcept {
    return rotated_vertices_[vertex];
  }
  int rotated_edge(int edge) const noexcept { return rotated_edges_[edge]; }
  bool boundary(int vertex) const noexcept { return boundaries_[vertex]; }
  bool north_goal(int vertex) const noexcept { return ys_[vertex] == 0; }
  bool south_goal(int vertex) const noexcept { return ys_[vertex] == 12; }

 private:
  Topology();
  std::array<std::int8_t, kVertexCount> xs_{};
  std::array<std::int8_t, kVertexCount> ys_{};
  std::array<std::array<Arc, 8>, kVertexCount> arcs_{};
  std::array<std::uint8_t, kVertexCount> degrees_{};
  std::array<std::uint8_t, kVertexCount> rotated_vertices_{};
  std::array<std::uint16_t, kEdgeCount> rotated_edges_{};
  std::array<bool, kVertexCount> boundaries_{};
};

State initial_state();
bool same_boundary(const State &left, const State &right) noexcept;
bool edge_used(const State &state, int edge) noexcept;
bool vertex_visited(const State &state, int vertex) noexcept;
std::uint8_t legal_arcs(const State &state,
                        std::array<Topology::Arc, 8> &result) noexcept;
bool apply_edge(State &state, std::uint8_t direction) noexcept;
bool apply_action(State &state, const Action &action,
                  bool require_complete = true) noexcept;
State rotate_and_swap(const State &state);
std::uint64_t state_hash(const State &state) noexcept;
std::uint64_t canonical_state_hash(const State &state) noexcept;
SparseFeatures active_features(const State &state);
SparseFeatures active_features(const State &state, std::uint8_t perspective);

std::string sha256_hex(std::span<const std::uint8_t> bytes);
float first_activation(float value) noexcept;
float second_activation(float value) noexcept;
float fast_tanh(float value) noexcept;

struct ModelDescriptor {
  std::size_t inputs{kFeatureCount};
  std::size_t hidden_one{};
  std::size_t hidden_two{};
  float scale_one{};
  float scale_two{};
  float scale_three{};
  std::string_view packed_base64{};
  std::string_view packed_sha256{};
  bool allow_empty_bootstrap{};
};

struct PreparedEvaluation {
  SparseFeatures features{};
  std::array<std::int32_t, 12> first{};
};

class QuantizedModel {
 public:
  explicit QuantizedModel(ModelDescriptor descriptor);

  PreparedEvaluation prepare(const SparseFeatures &features) const noexcept;
  float evaluate(const SparseFeatures &features) const noexcept;
  float evaluate_delta(const PreparedEvaluation &base,
                       const SparseFeatures &features) const noexcept;
  float finish(std::array<std::int32_t, 12> first) const noexcept;
  std::span<const std::int8_t> weights() const noexcept { return weights_; }
  std::size_t hidden_one() const noexcept { return hidden_one_; }
  std::size_t hidden_two() const noexcept { return hidden_two_; }
  std::string_view payload_sha256() const noexcept { return payload_sha256_; }

 private:
  std::vector<std::int8_t> weights_{};
  std::size_t hidden_one_{};
  std::size_t hidden_two_{};
  float scale_one_{};
  float scale_two_{};
  float scale_three_{};
  std::string payload_sha256_{};

  void add_input(std::array<std::int32_t, 12> &first, std::size_t input,
                 int multiplier) const noexcept;
};

const QuantizedModel &deployment_model();

enum class TacticalClass : std::uint8_t {
  ImmediateGoal,
  ForcedCutoff,
  SafeHandoff,
  OpponentImmediateGoal,
  OwnGoal,
};

struct GeneratorConfig {
  std::size_t max_actions{kMaximumActions};
  std::size_t max_partial_paths{kRootPartialPaths};
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::time_point::max()};
  std::uint64_t shuffle_seed{0x6a09e667f3bcc909ULL};
  const Action *precomputed_emergency{};
};

struct GeneratorStats {
  std::uint64_t partial_paths{};
  std::uint64_t proof_paths{};
  std::uint64_t completed_actions{};
  std::uint64_t duplicate_boundaries{};
  std::uint64_t fifo_extractions{};
  std::uint64_t lifo_extractions{};
  std::uint64_t tactical_actions{};
  std::uint64_t truncations{};
  std::uint64_t cap_drops{};
  bool deadline_reached{};
  bool proof_truncated{};
};

struct CompleteTurnAction {
  State result{};
  Action action{};
  TacticalClass tactical{TacticalClass::SafeHandoff};
  std::uint32_t order{};
};

struct GenerationResult {
  std::vector<CompleteTurnAction> actions{};
  GeneratorStats stats{};
  bool exhaustive{};
};

Action emergency_complete_action(
    const State &state,
    std::uint64_t seed = GeneratorConfig{}.shuffle_seed);
GenerationResult generate_complete_turns(const State &state,
                                         const GeneratorConfig &config = {});

struct SearchConfig {
  std::size_t max_actions{kMaximumActions};
  std::size_t root_partial_paths{kRootPartialPaths};
  std::size_t nonroot_partial_paths{kNonrootPartialPaths};
  std::size_t max_tree_nodes{kProductionTreeNodes};
  std::uint64_t max_expansions{kMaximumExpansions};
  std::uint64_t shuffle_seed{GeneratorConfig{}.shuffle_seed};
  double exploration{kExploration};
  double fpu{kFirstPlayUrgency};
  double final_visit_weight{kFinalVisitWeight};
  const QuantizedModel *model{};
};

struct SearchStats {
  std::uint64_t expansions{};
  std::uint64_t generated_children{};
  std::uint64_t evaluated_children{};
  std::uint64_t tactical_children{};
  std::uint64_t generator_partial_paths{};
  std::uint64_t generator_proof_paths{};
  std::uint64_t duplicate_boundaries{};
  std::uint64_t fifo_extractions{};
  std::uint64_t lifo_extractions{};
  std::size_t tree_nodes{};
  std::uint32_t max_depth{};
  bool deadline_reached{};
  bool tree_cap_reached{};
  bool expansion_cap_reached{};
};

struct RootActionStat {
  Action action{};
  TacticalClass tactical{TacticalClass::SafeHandoff};
  float value{};
  float initial_value{};
  std::uint32_t visits{};
  std::uint32_t selection_visits{};
  bool solved{};
  std::uint32_t order{};
};

struct SearchResult {
  Action action{};
  float value{};
  bool solved{};
  std::vector<RootActionStat> root_actions{};
  SearchStats stats{};
};

float mate_value(int winner, int perspective, std::uint32_t depth) noexcept;
double uct_score(float value, std::uint32_t parent_visits,
                 std::uint32_t child_visits, double exploration,
                 double fpu);
double final_score(float value, std::uint32_t visits, double weight);
SearchResult search(
    const State &state, std::chrono::steady_clock::time_point deadline,
    const SearchConfig &config = {}, const Action *precomputed_emergency = nullptr);

}  // namespace compact_value_bfm
