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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define private public
#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "bot.cpp"
#undef PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#undef private

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;
#if defined(CANDIDATE_COMPONENT_CACHE)
namespace detail = papersoccer::hybrid_detail;
#else
namespace detail = papersoccer::detail;
#endif

namespace {

ps::RulesConfig rules() {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed,
          ps::BlockedRule::MoverLoses};
}

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

std::uint64_t next_random(std::uint64_t &state) {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

std::vector<ps::GameState> corpus(std::size_t target) {
  std::vector<ps::GameState> states;
  states.reserve(target + 8);
  std::uint64_t random = 0x4f1bbcdc676f2b31ULL;
  while (states.size() < target) {
    ps::GameState state = ps::make_initial_state(rules());
    while (!ps::is_terminal(state) && states.size() < target) {
      states.push_back(state);
      const std::vector<ps::Move> legal = ps::legal_moves(state);
      require(!legal.empty(), "live state has no legal move");
      state = ps::apply_move(state, legal[next_random(random) % legal.size()]);
    }
  }
  for (const std::string_view transcript : {
           "6/1", "4/3/6/4/3/0", "0/6/5/4/5/53/61/0633",
           "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35"}) {
    ps::GameState state = ps::make_initial_state(rules());
    std::size_t begin = 0;
    while (begin <= transcript.size()) {
      const std::size_t slash = transcript.find('/', begin);
      const std::size_t end = slash == std::string_view::npos
                                  ? transcript.size() : slash;
      cg::apply_encoded_turn(state, transcript.substr(begin, end - begin));
      if (slash == std::string_view::npos) break;
      begin = slash + 1;
    }
    if (!ps::is_terminal(state)) states.push_back(std::move(state));
  }
  return states;
}

detail::PositionKey canonical_key(
    const std::shared_ptr<const detail::SearchTopology> &topology,
    const ps::GameState &state) {
  detail::PositionKey key{};
  for (const ps::Segment &segment : state.used_segments) {
    const auto edge = topology->find_edge(segment);
    if (edge.has_value()) {
      detail::xor_position_key(key, detail::position_key_component(1, *edge));
    }
  }
  for (const auto &[point, visits] : state.visit_count) {
    if (visits <= 0) continue;
    const auto vertex = topology->find_vertex(point);
    if (vertex.has_value()) {
      detail::xor_position_key(key,
                               detail::position_key_component(2, *vertex));
    }
  }
  const auto ball = topology->find_vertex(state.ball);
  require(ball.has_value(), "ball missing from topology");
  detail::xor_position_key(key, detail::position_key_component(3, *ball));
  detail::xor_position_key(
      key, detail::position_key_component(
               4, static_cast<std::uint64_t>(state.to_move)));
  detail::xor_position_key(
      key, detail::position_key_component(
               5, static_cast<std::uint64_t>(state.status)));
  return key;
}

std::uint64_t combined_tt(detail::PositionKey key) {
  return key.first ^ ((key.second << 23U) | (key.second >> 41U));
}

std::uint64_t combined_eval(detail::PositionKey key) {
  return key.first ^ ((key.second << 29U) | (key.second >> 35U));
}

struct Audit {
  cg::TranspositionTable tt_small{4096};
  cg::TranspositionTable tt_production{262144};
  cg::EvaluationTable eval_small{2048};
  cg::EvaluationTable eval_production{131072};
  std::uint64_t events{};
  std::uint64_t digest_one{0x243f6a8885a308d3ULL};
  std::uint64_t digest_two{0x13198a2e03707344ULL};

  void observe(const detail::SearchPosition &position,
               const ps::GameState &state) {
    const detail::PositionKey actual = position.position_key();
    const detail::PositionKey expected = canonical_key(position.topology(), state);
    require(actual == expected, "compact position key differs from formula");

    const std::size_t tt_small_expected =
        2U * static_cast<std::size_t>(combined_tt(actual) % (4096U / 2U));
    const std::size_t tt_production_expected =
        2U * static_cast<std::size_t>(combined_tt(actual) % (262144U / 2U));
    const std::size_t eval_small_expected =
        2U * static_cast<std::size_t>(combined_eval(actual) % (2048U / 2U));
    const std::size_t eval_production_expected =
        2U * static_cast<std::size_t>(combined_eval(actual) % (131072U / 2U));
    require(tt_small.bucket_index(actual) == tt_small_expected,
            "small TT placement mismatch");
    require(tt_production.bucket_index(actual) == tt_production_expected,
            "production TT placement mismatch");
    require(eval_small.bucket_index(actual) == eval_small_expected,
            "small evaluation placement mismatch");
    require(eval_production.bucket_index(actual) == eval_production_expected,
            "production evaluation placement mismatch");

    const std::uint64_t placement =
        tt_small_expected ^ (tt_production_expected << 12U) ^
        (eval_small_expected << 30U) ^ (eval_production_expected << 41U);
    digest_one = detail::mix_position_key(
        digest_one ^ actual.first ^ placement ^ events);
    digest_two = detail::mix_position_key(
        digest_two ^ actual.second ^ (placement << 7U) ^ (events << 1U));
    ++events;
  }
};

void exhaustive(detail::SearchPosition &position, const ps::GameState &state,
                unsigned depth, Audit &audit) {
  audit.observe(position, state);
  if (depth == 0 || position.is_terminal()) return;
  std::array<std::uint8_t, detail::kMaximumMoves> slots{};
  const std::uint8_t count = position.legal_slots(slots);
  for (std::uint8_t index = 0; index < count; ++index) {
    const detail::PositionKey before = position.position_key();
    const std::size_t undo_depth = position.undo_depth();
    const ps::Move move = position.move_for_slot(slots[index]);
    const ps::GameState next = ps::apply_move(state, move);
    position.make_move(slots[index]);
    exhaustive(position, next, depth - 1U, audit);
    position.unmake_move();
    require(position.undo_depth() == undo_depth, "undo depth not restored");
    require(position.position_key() == before, "key not restored after unmake");
    audit.observe(position, state);
  }
}

}  // namespace

int main() {
  const std::vector<ps::GameState> states = corpus(1000);
  require(states.size() == 1004, "unexpected corpus size");
  Audit audit;

#if defined(CANDIDATE_COMPONENT_CACHE)
  require(detail::player_key_component(ps::Player::One) ==
              detail::position_key_component(4, 0),
          "constexpr player-one key component mismatch");
  require(detail::player_key_component(ps::Player::Two) ==
              detail::position_key_component(4, 1),
          "constexpr player-two key component mismatch");
  require(detail::status_key_component(ps::Status::InProgress) ==
              detail::position_key_component(5, 0),
          "constexpr in-progress key component mismatch");
  require(detail::status_key_component(ps::Status::WonByOne) ==
              detail::position_key_component(5, 1),
          "constexpr won-by-one key component mismatch");
  require(detail::status_key_component(ps::Status::WonByTwo) ==
              detail::position_key_component(5, 2),
          "constexpr won-by-two key component mismatch");

  constexpr std::array<std::pair<int, int>, 5> dimensions{{
      {1, 1}, {2, 3}, {5, 4}, {8, 10}, {12, 7},
  }};
  constexpr std::array<ps::GoalRule, 2> goal_rules{{
      ps::GoalRule::OpponentGoalOnly, ps::GoalRule::OwnGoalsAllowed,
  }};
  constexpr std::array<ps::BlockedRule, 2> blocked_rules{{
      ps::BlockedRule::PlayerToMoveLoses, ps::BlockedRule::MoverLoses,
  }};
  std::uint64_t matrix_seed = 0xd1b54a32d192ed03ULL;
  for (const auto [width, height] : dimensions) {
    for (const ps::GoalRule goal_rule : goal_rules) {
      for (const ps::BlockedRule blocked_rule : blocked_rules) {
        const ps::RulesConfig config{width, height, goal_rule, blocked_rule};
        auto topology = std::make_shared<detail::SearchTopology>(config);
        for (std::size_t edge = 0; edge < topology->edge_count(); ++edge) {
          require(topology->edge_key(static_cast<std::uint32_t>(edge)) ==
                      detail::position_key_component(1, edge),
                  "matrix cached edge component mismatch");
        }
        for (std::size_t vertex = 0; vertex < topology->vertex_count();
             ++vertex) {
          const auto index = static_cast<std::uint32_t>(vertex);
          require(topology->visited_key(index) ==
                      detail::position_key_component(2, vertex),
                  "matrix cached visited component mismatch");
          require(topology->ball_key(index) ==
                      detail::position_key_component(3, vertex),
                  "matrix cached ball component mismatch");
          require(topology->boundary_key(index) ==
                      detail::position_key_component(6, vertex),
                  "matrix cached boundary component mismatch");
        }
        const ps::GameState root_state = ps::make_initial_state(config);
        detail::SearchPosition position(topology, root_state);
        const detail::PositionKey root_key = position.position_key();
        std::vector<ps::GameState> stack{root_state};
        require(root_key == canonical_key(topology, root_state),
                "matrix root key mismatch");
        for (unsigned operation = 0; operation < 128; ++operation) {
          std::array<std::uint8_t, detail::kMaximumMoves> slots{};
          const std::uint8_t count = position.legal_slots(slots);
          const bool can_unmake = position.undo_depth() != 0;
          if (can_unmake &&
              (count == 0 || (next_random(matrix_seed) & 3U) == 0U)) {
            position.unmake_move();
            stack.pop_back();
          } else if (count != 0) {
            const std::uint8_t slot = slots[next_random(matrix_seed) % count];
            const ps::Move move = position.move_for_slot(slot);
            stack.push_back(ps::apply_move(stack.back(), move));
            position.make_move(slot);
          }
          require(position.position_key() == canonical_key(topology, stack.back()),
                  "matrix make/unmake key mismatch");
        }
        position.unmake_to(0);
        require(position.position_key() == root_key,
                "matrix root key was not restored");
      }
    }
  }
#endif

  for (std::size_t index = 0; index < states.size(); ++index) {
    auto topology = std::make_shared<detail::SearchTopology>(states[index].config);
    detail::SearchPosition position(topology, states[index]);
    const detail::PositionKey root = position.position_key();
    audit.observe(position, states[index]);

    std::vector<ps::GameState> stack{states[index]};
    std::uint64_t random = 0x9e3779b97f4a7c15ULL ^ index;
    for (unsigned operation = 0; operation < 128; ++operation) {
      std::array<std::uint8_t, detail::kMaximumMoves> slots{};
      const std::uint8_t count = position.legal_slots(slots);
      const bool can_unmake = position.undo_depth() != 0;
      const bool choose_unmake = can_unmake &&
          (count == 0 || (next_random(random) & 3U) == 0U);
      if (choose_unmake) {
        position.unmake_move();
        stack.pop_back();
      } else if (count != 0) {
        const std::uint8_t slot = slots[next_random(random) % count];
        const ps::Move move = position.move_for_slot(slot);
        stack.push_back(ps::apply_move(stack.back(), move));
        position.make_move(slot);
      }
      audit.observe(position, stack.back());
    }
    position.unmake_to(0);
    require(position.position_key() == root, "random walk root key not restored");
    audit.observe(position, states[index]);

    cg::SearchConfig config;
    config.max_nodes = 1;
    config.transposition_entries = 2;
    config.evaluation_entries = 2;
    cg::CompleteTurnSearch search(states[index], config);
    detail::PositionKey boundary = canonical_key(search.position_.topology(),
                                                 states[index]);
    const auto ball = search.position_.ball_vertex();
    detail::xor_position_key(
        boundary, detail::position_key_component(6, ball));
    require(search.boundary_key() == boundary, "boundary key mismatch");
  }

  auto topology = std::make_shared<detail::SearchTopology>(rules());
  detail::SearchPosition initial(topology, ps::make_initial_state(rules()));
  exhaustive(initial, ps::make_initial_state(rules()), 6, audit);
  require(initial.undo_depth() == 0, "exhaustive walk did not restore depth");
  audit.observe(initial, ps::make_initial_state(rules()));

  std::cout << "states=" << states.size() << " events=" << audit.events
            << " digest=" << audit.digest_one << ':' << audit.digest_two
            << " vertices=" << topology->vertex_count()
            << " edges=" << topology->edge_count() << '\n';
}
