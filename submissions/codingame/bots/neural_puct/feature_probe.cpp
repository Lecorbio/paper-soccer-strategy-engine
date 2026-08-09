#define PAPER_SOCCER_NEURAL_PUCT_NO_MAIN
#include "submission.cpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>

namespace ps = papersoccer;
namespace np = papersoccer::neural_puct;

namespace {

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

ps::Point rotate(ps::Point point) { return {8 - point.x, 12 - point.y}; }
ps::Point reflect(ps::Point point) { return {8 - point.x, point.y}; }

template <typename Transform>
ps::GameState transform_state(const ps::GameState &state, Transform transform,
                              bool swap_player) {
  ps::GameState result = state;
  result.ball = transform(state.ball);
  result.to_move = swap_player ? ps::opponent(state.to_move) : state.to_move;
  result.path.clear();
  for (const ps::Point point : state.path) {
    result.path.push_back(transform(point));
  }
  result.used_segments.clear();
  for (const ps::Segment segment : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{transform(segment.a), transform(segment.b)});
  }
  result.visit_count.clear();
  for (const auto &[point, visits] : state.visit_count) {
    result.visit_count[transform(point)] = visits;
  }
  return result;
}

std::array<float, np::kFeatureCount> encode(const ps::GameState &state) {
  auto topology = std::make_shared<ps::detail::SearchTopology>(state.config);
  ps::detail::SearchPosition position(topology, state);
  np::FeatureEncoder encoder(topology);
  return encoder.encode(position);
}

void print_fixture(std::string_view name, const ps::GameState &state) {
  const auto features = encode(state);
  std::cout << name;
  for (const float feature : features) {
    std::cout << ',' << std::hexfloat << feature;
  }
  std::cout << '\n';
}

}  // namespace

int main() {
  const ps::GameState initial = ps::make_initial_state(codingame_rules());
  const ps::GameState initial_rotated =
      transform_state(initial, rotate, true);

  ps::GameState transcript = initial;
  np::apply_encoded_turn(transcript, "0");
  np::apply_encoded_turn(transcript, "5");
  np::apply_encoded_turn(transcript, "21");
  const ps::GameState transcript_reflected =
      transform_state(transcript, reflect, false);

  print_fixture("initial_player_one", initial);
  print_fixture("initial_player_two_rotated", initial_rotated);
  print_fixture("transcript_0_5_21", transcript);
  print_fixture("transcript_0_3_67_reflected", transcript_reflected);
  return 0;
}
