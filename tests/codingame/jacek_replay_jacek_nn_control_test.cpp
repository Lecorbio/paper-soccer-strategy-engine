#include "jacek_replay_bfm_gate_internal.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "papersoccer/rules.hpp"

namespace {

papersoccer::Move decode(papersoccer::Point from, char direction) {
  constexpr std::array<papersoccer::Point, 8> deltas{{
      {0, -1}, {1, -1}, {1, 0}, {1, 1},
      {0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
  }};
  const papersoccer::Point delta =
      deltas.at(static_cast<std::size_t>(direction - '0'));
  return {{from.x + delta.x, from.y + delta.y}};
}

void apply_turn(papersoccer::GameState &state, std::string_view encoded) {
  const papersoccer::Player mover = state.to_move;
  for (const char direction : encoded) {
    state = papersoccer::apply_move(state, decode(state.ball, direction));
  }
  if (!papersoccer::is_terminal(state) && state.to_move == mover) {
    throw std::runtime_error("fixture ends during a mandatory rebound");
  }
}

std::string encode_action(const papersoccer::GameState &state,
                          const std::vector<papersoccer::Move> &action) {
  constexpr std::array<papersoccer::Point, 8> deltas{{
      {0, -1}, {1, -1}, {1, 0}, {1, 1},
      {0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
  }};
  papersoccer::GameState replay = state;
  std::string encoded;
  for (const papersoccer::Move move : action) {
    const papersoccer::Point delta{move.to.x - replay.ball.x,
                                   move.to.y - replay.ball.y};
    const auto found = std::find(deltas.begin(), deltas.end(), delta);
    if (found == deltas.end()) {
      throw std::runtime_error("adapter returned a non-adjacent move");
    }
    encoded.push_back(
        static_cast<char>('0' + std::distance(deltas.begin(), found)));
    replay = papersoccer::apply_move(replay, move);
  }
  return encoded;
}

}  // namespace

int main() {
  using namespace papersoccer;
  GameState state = make_initial_state(
      {8, 10, GoalRule::OwnGoalsAllowed, BlockedRule::MoverLoses});
  const std::string prefix = "0/5/21/2/7/523";
  std::size_t begin = 0;
  while (begin < prefix.size()) {
    const std::size_t separator = prefix.find('/', begin);
    const std::size_t end = separator == std::string::npos
                                ? prefix.size()
                                : separator;
    apply_turn(state, std::string_view(prefix).substr(begin, end - begin));
    if (separator == std::string::npos) break;
    begin = separator + 1;
  }
  if (state.to_move != Player::One) {
    throw std::runtime_error("fixture does not return to Player One");
  }
  jacek_replay_gate::ControlConfig config;
  config.time_ms = 1;
  config.work_cap = 16;
  const jacek_replay_gate::ControlDecision decision =
      jacek_replay_gate::choose_jacek_nn_control(state, prefix, config);
  if (!decision.replay_correction || decision.work != 0 ||
      encode_action(state, decision.action) != "3") {
    throw std::runtime_error("known jacek_nn replay correction was not used");
  }
  apply_turn(state, "3");
  return 0;
}
