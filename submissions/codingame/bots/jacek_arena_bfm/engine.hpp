#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace jacek_arena_bfm {

inline constexpr int kVertexCount = 105;
inline constexpr int kEdgeCount = 316;
inline constexpr int kFeatureCount = 1156;
inline constexpr int kMaximumActionLength = 316;

struct Action {
  std::array<std::uint8_t, kMaximumActionLength> directions{};
  std::uint16_t length{0};

  bool operator==(const Action &other) const noexcept;
  std::string text() const;
};

struct State {
  std::array<std::uint64_t, 5> used{};
  std::uint8_t ball{0};
  std::uint8_t to_move{0};
  std::int8_t winner{-1};
  std::uint16_t ply{0};

  bool terminal() const noexcept { return winner >= 0; }
};

enum class GeneratorStrategy {
  Fixed250NineOne,
  TacticalProgressive,
  PriorityBeam,
  HighCapRecall,
};

struct GeneratorStats {
  std::size_t partials{0};
  std::size_t completed{0};
  std::size_t duplicates{0};
  bool truncated{false};
  bool deadline_reached{false};
};

struct SearchConfig {
  GeneratorStrategy generator{GeneratorStrategy::TacticalProgressive};
  std::size_t maximum_nodes{80000};
  double exploration{0.95};
};

struct SearchResult {
  Action action{};
  std::size_t nodes{0};
  std::size_t root_actions{0};
  std::size_t generator_deadline_stops{0};
  std::uint64_t generator_microseconds{0};
  std::uint64_t maximum_generator_microseconds{0};
  double root_value{0.0};
  bool deadline_reached{false};
};

class Topology {
 public:
  struct Arc {
    std::uint16_t edge{0};
    std::uint8_t destination{0};
    std::uint8_t direction{0};
  };

  static const Topology &get();
  int x(int vertex) const noexcept { return xs_[vertex]; }
  int y(int vertex) const noexcept { return ys_[vertex]; }
  int vertex_at(int x, int y) const noexcept;
  int degree(int vertex) const noexcept { return degrees_[vertex]; }
  Arc arc(int vertex, int index) const noexcept { return arcs_[vertex][index]; }
  int rotated_vertex(int vertex) const noexcept { return rotated_vertices_[vertex]; }
  int rotated_edge(int edge) const noexcept { return rotated_edges_[edge]; }
  bool boundary(int vertex) const noexcept { return boundaries_[vertex]; }
  bool north_goal(int vertex) const noexcept;
  bool south_goal(int vertex) const noexcept;

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
bool edge_used(const State &state, int edge) noexcept;
bool vertex_visited(const State &state, int vertex) noexcept;
std::vector<Topology::Arc> legal_arcs(const State &state);
bool apply_edge(State &state, std::uint8_t direction);
bool apply_action(State &state, const Action &action, bool require_complete = true);
std::vector<Action> generate_actions(
    const State &state, GeneratorStrategy strategy, bool root,
    GeneratorStats *stats = nullptr,
    std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::time_point::max());
std::array<std::uint16_t, 421> active_features(const State &state,
                                                std::size_t &count);
float evaluate(const State &state);
SearchResult search(const State &state,
                    std::chrono::steady_clock::time_point deadline,
                    const SearchConfig &config = {});
std::uint64_t state_hash(const State &state) noexcept;
State rotate_and_swap(const State &state);

}  // namespace jacek_arena_bfm
