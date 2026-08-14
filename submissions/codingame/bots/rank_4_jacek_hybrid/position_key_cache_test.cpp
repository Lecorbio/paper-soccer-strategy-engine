#include "mcts_internal.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "papersoccer/rules.hpp"

namespace ps = papersoccer;
namespace detail = papersoccer::hybrid_detail;

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::uint64_t next_random(std::uint64_t &state) noexcept {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

detail::PositionKey direct_key(
    const std::shared_ptr<const detail::SearchTopology> &topology,
    const ps::GameState &state) {
  detail::PositionKey key{};
  for (const ps::Segment &segment : state.used_segments) {
    if (const auto edge = topology->find_edge(segment); edge.has_value()) {
      detail::xor_position_key(key, detail::position_key_component(1, *edge));
    }
  }
  for (const auto &[point, visits] : state.visit_count) {
    if (visits <= 0) {
      continue;
    }
    if (const auto vertex = topology->find_vertex(point); vertex.has_value()) {
      detail::xor_position_key(
          key, detail::position_key_component(2, *vertex));
    }
  }
  const auto ball = topology->find_vertex(state.ball);
  require(ball.has_value(), "ball is missing from topology");
  detail::xor_position_key(key, detail::position_key_component(3, *ball));
  detail::xor_position_key(
      key, detail::position_key_component(
               4, static_cast<std::uint64_t>(state.to_move)));
  detail::xor_position_key(
      key, detail::position_key_component(
               5, static_cast<std::uint64_t>(state.status)));
  return key;
}

void verify_position(const detail::SearchPosition &position,
                     const ps::GameState &state) {
  const auto &topology = position.topology();
  const detail::PositionKey direct = direct_key(topology, state);
  require(position.position_key() == direct, "cached position key mismatch");

  detail::PositionKey cached_boundary = position.position_key();
  detail::xor_position_key(
      cached_boundary, topology->boundary_key(position.ball_vertex()));
  detail::PositionKey direct_boundary = direct;
  detail::xor_position_key(
      direct_boundary,
      detail::position_key_component(6, position.ball_vertex()));
  require(cached_boundary == direct_boundary, "cached boundary key mismatch");
}

void verify_components(const detail::SearchTopology &topology) {
  for (std::size_t edge = 0; edge < topology.edge_count(); ++edge) {
    const auto index = static_cast<detail::SearchTopology::EdgeIndex>(edge);
    require(topology.edge_key(index) ==
                detail::position_key_component(1, edge),
            "cached edge component mismatch");
  }
  for (std::size_t vertex = 0; vertex < topology.vertex_count(); ++vertex) {
    const auto index = static_cast<detail::SearchTopology::VertexIndex>(vertex);
    require(topology.visited_key(index) ==
                detail::position_key_component(2, vertex),
            "cached visited component mismatch");
    require(topology.ball_key(index) ==
                detail::position_key_component(3, vertex),
            "cached ball component mismatch");
    require(topology.boundary_key(index) ==
                detail::position_key_component(6, vertex),
            "cached boundary component mismatch");
  }
}

}  // namespace

int main() {
  require(detail::player_key_component(ps::Player::One) ==
              detail::position_key_component(4, 0),
          "player-one component mismatch");
  require(detail::player_key_component(ps::Player::Two) ==
              detail::position_key_component(4, 1),
          "player-two component mismatch");
  require(detail::status_key_component(ps::Status::InProgress) ==
              detail::position_key_component(5, 0),
          "in-progress component mismatch");
  require(detail::status_key_component(ps::Status::WonByOne) ==
              detail::position_key_component(5, 1),
          "won-by-one component mismatch");
  require(detail::status_key_component(ps::Status::WonByTwo) ==
              detail::position_key_component(5, 2),
          "won-by-two component mismatch");

  constexpr std::array<std::pair<int, int>, 5> dimensions{{
      {1, 1}, {2, 3}, {5, 4}, {8, 10}, {12, 7},
  }};
  constexpr std::array<ps::GoalRule, 2> goal_rules{{
      ps::GoalRule::OpponentGoalOnly, ps::GoalRule::OwnGoalsAllowed,
  }};
  constexpr std::array<ps::BlockedRule, 2> blocked_rules{{
      ps::BlockedRule::PlayerToMoveLoses, ps::BlockedRule::MoverLoses,
  }};

  std::uint64_t random = 0xd1b54a32d192ed03ULL;
  std::size_t configurations = 0;
  std::size_t operations = 0;
  for (const auto [width, height] : dimensions) {
    for (const ps::GoalRule goal_rule : goal_rules) {
      for (const ps::BlockedRule blocked_rule : blocked_rules) {
        const ps::RulesConfig config{width, height, goal_rule, blocked_rule};
        auto topology = std::make_shared<detail::SearchTopology>(config);
        verify_components(*topology);

        const ps::GameState root_state = ps::make_initial_state(config);
        detail::SearchPosition position(topology, root_state);
        const detail::PositionKey root_key = position.position_key();
        std::vector<ps::GameState> states{root_state};
        verify_position(position, states.back());

        for (unsigned operation = 0; operation < 128; ++operation) {
          std::array<std::uint8_t, detail::kMaximumMoves> slots{};
          const std::uint8_t count = position.legal_slots(slots);
          const bool can_unmake = position.undo_depth() != 0;
          if (can_unmake &&
              (count == 0 || (next_random(random) & 3U) == 0U)) {
            position.unmake_move();
            states.pop_back();
          } else if (count != 0) {
            const std::uint8_t slot = slots[next_random(random) % count];
            const ps::Move move = position.move_for_slot(slot);
            states.push_back(ps::apply_move(states.back(), move));
            position.make_move(slot);
          }
          verify_position(position, states.back());
          ++operations;
        }

        position.unmake_to(0);
        require(position.position_key() == root_key,
                "root key was not restored");
        verify_position(position, root_state);
        ++configurations;
      }
    }
  }

  std::cout << "position_key_cache configurations=" << configurations
            << " operations=" << operations << '\n';
}
