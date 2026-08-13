#define JACEK_ARENA_BFM_NO_MAIN
#include "submission.cpp"

#include <iostream>
#include <stdexcept>

namespace jab = jacek_arena_bfm;

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

void test_topology() {
  const auto &topology = jab::Topology::get();
  require(topology.vertex_at(4, 6) >= 0, "missing center");
  require(topology.vertex_at(0, 0) < 0, "invalid corner goal vertex");
  std::array<bool, jab::kEdgeCount> edges{};
  int unique = 0;
  for (int vertex = 0; vertex < jab::kVertexCount; ++vertex) {
    require(topology.degree(vertex) <= 8, "degree exceeds eight");
    require(topology.rotated_vertex(topology.rotated_vertex(vertex)) == vertex,
            "vertex rotation is not involutive");
    for (int index = 0; index < topology.degree(vertex); ++index) {
      const int edge = topology.arc(vertex, index).edge;
      if (!edges[edge]) {
        edges[edge] = true;
        ++unique;
      }
      require(topology.rotated_edge(topology.rotated_edge(edge)) == edge,
              "edge rotation is not involutive");
    }
  }
  require(unique == jab::kEdgeCount, "topology is not exactly 316 edges");
}

void test_initial_actions() {
  jab::State state = jab::initial_state();
  require(jab::legal_arcs(state).size() == 8, "center must have eight moves");
  for (const auto strategy : {
           jab::GeneratorStrategy::Fixed250NineOne,
           jab::GeneratorStrategy::TacticalProgressive,
           jab::GeneratorStrategy::PriorityBeam,
           jab::GeneratorStrategy::HighCapRecall}) {
    const auto actions = jab::generate_actions(state, strategy, true);
    require(actions.size() == 8, "initial position must have eight actions");
    for (const auto &action : actions) {
      jab::State copy = state;
      require(jab::apply_action(copy, action), "generated action is illegal");
      require(copy.to_move == 1 || copy.terminal(),
              "generated action does not complete the turn");
    }
  }
}

void test_rebound_completion_and_deduplication() {
  jab::State state = jab::initial_state();
  for (const std::uint8_t direction : {0, 1, 3, 4, 6, 7}) {
    require(jab::apply_edge(state, direction), "rebound setup edge failed");
  }
  require(state.to_move == 1, "setup did not hand off");
  const auto actions = jab::generate_actions(
      state, jab::GeneratorStrategy::HighCapRecall, true);
  std::vector<std::uint64_t> hashes;
  for (const auto &action : actions) {
    jab::State copy = state;
    require(jab::apply_action(copy, action), "rebound action is incomplete");
    const auto hash = jab::state_hash(copy);
    require(std::find(hashes.begin(), hashes.end(), hash) == hashes.end(),
            "duplicate action boundary");
    hashes.push_back(hash);
  }
}

void test_features_and_symmetry() {
  jab::State state = jab::initial_state();
  std::size_t first_count = 0;
  const auto first = jab::active_features(state, first_count);
  require(first_count == 105, "initial feature count must be 105");
  require(*std::max_element(first.begin(), first.begin() + first_count) <
              jab::kFeatureCount,
          "feature index out of range");
  require(jab::apply_edge(state, 1), "feature setup move failed");
  jab::State rotated = jab::rotate_and_swap(state);
  std::size_t left_count = 0;
  std::size_t right_count = 0;
  auto left = jab::active_features(state, left_count);
  auto right = jab::active_features(rotated, right_count);
  require(left_count == right_count, "symmetry feature count differs");
  std::sort(left.begin(), left.begin() + left_count);
  std::sort(right.begin(), right.begin() + right_count);
  require(std::equal(left.begin(), left.begin() + left_count, right.begin()),
          "mover-relative features violate color symmetry");
}

void mark_used(jab::State &state, int edge) {
  state.used[edge >> 6] |= 1ULL << (edge & 63);
}

void test_goal_own_goal_and_forced_block() {
  const auto &topology = jab::Topology::get();
  jab::State near_goal = jab::initial_state();
  near_goal.ball = static_cast<std::uint8_t>(topology.vertex_at(4, 1));
  near_goal.to_move = 0;
  jab::State north = near_goal;
  require(jab::apply_edge(north, 0) && north.terminal() && north.winner == 0,
          "attacking goal was not a win");
  jab::State own = near_goal;
  own.to_move = 1;
  require(jab::apply_edge(own, 0) && own.terminal() && own.winner == 0,
          "own goal was not a loss");

  for (const auto strategy : {
           jab::GeneratorStrategy::Fixed250NineOne,
           jab::GeneratorStrategy::TacticalProgressive,
           jab::GeneratorStrategy::PriorityBeam,
           jab::GeneratorStrategy::HighCapRecall}) {
    const auto actions = jab::generate_actions(near_goal, strategy, true);
    const bool keeps_goal = std::any_of(
        actions.begin(), actions.end(), [&](const jab::Action &action) {
          jab::State successor = near_goal;
          return jab::apply_action(successor, action) && successor.terminal() &&
                 successor.winner == 0 &&
                 (topology.north_goal(successor.ball) ||
                  topology.south_goal(successor.ball));
        });
    require(keeps_goal, "generator lost exact goal witness");
    jab::State own_source = near_goal;
    own_source.to_move = 1;
    const auto own_actions = jab::generate_actions(own_source, strategy, true);
    const bool keeps_own_goal = std::any_of(
        own_actions.begin(), own_actions.end(), [&](const jab::Action &action) {
          jab::State successor = own_source;
          return jab::apply_action(successor, action) && successor.terminal() &&
                 successor.winner == 0 &&
                 (topology.north_goal(successor.ball) ||
                  topology.south_goal(successor.ball));
        });
    require(keeps_own_goal, "generator lost own-goal witness");
  }

  jab::State blocked = jab::initial_state();
  const auto only = topology.arc(blocked.ball, 0);
  for (int index = 1; index < topology.degree(blocked.ball); ++index) {
    mark_used(blocked, topology.arc(blocked.ball, index).edge);
  }
  for (int index = 0; index < topology.degree(only.destination); ++index) {
    const int edge = topology.arc(only.destination, index).edge;
    if (edge != only.edge) mark_used(blocked, edge);
  }
  require(jab::legal_arcs(blocked).size() == 1, "forced block setup is not forced");
  for (const auto strategy : {
           jab::GeneratorStrategy::Fixed250NineOne,
           jab::GeneratorStrategy::TacticalProgressive,
           jab::GeneratorStrategy::PriorityBeam,
           jab::GeneratorStrategy::HighCapRecall}) {
    const auto actions = jab::generate_actions(blocked, strategy, true);
    require(actions.size() == 1, "forced block witness was lost");
    jab::State successor = blocked;
    require(jab::apply_action(successor, actions.front()) && successor.terminal() &&
                successor.winner == 1,
            "blocked mover did not lose");
  }
}

void test_search_legality() {
  jab::State state = jab::initial_state();
  const auto result = jab::search(
      state, std::chrono::steady_clock::now() + std::chrono::milliseconds(20),
      jab::SearchConfig{jab::GeneratorStrategy::TacticalProgressive, 2000, 0.95});
  require(result.action.length > 0, "search returned empty action");
  require(jab::apply_action(state, result.action), "search returned illegal action");
}

void test_packed_model_decoder() {
  const auto bytes = jab::decode_base64("AP+Afw==", 4);
  require(bytes.size() == 4, "base64 decoder rejected valid payload");
  require(bytes[0] == 0 && bytes[1] == -1 && bytes[2] == -128 &&
              bytes[3] == 127,
          "base64 decoder changed signed int8 weights");
  require(jab::decode_base64("AA==", 2).empty(),
          "base64 decoder accepted a wrong declared count");
}

}  // namespace

int main() {
  try {
    test_topology();
    test_initial_actions();
    test_rebound_completion_and_deduplication();
    test_features_and_symmetry();
    test_goal_own_goal_and_forced_block();
    test_packed_model_decoder();
    test_search_legality();
    std::cout << "jacek_arena_bfm submission tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek_arena_bfm submission test failed: " << error.what()
              << '\n';
    return 1;
  }
}
