#define PAPER_SOCCER_JACEK_NATIVE_BFM_NO_MAIN
#include "submission.cpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace candidate = papersoccer::jacek_native_bfm;

namespace {

inline constexpr double kFirstSafetyCeilingMs = 900.0;
inline constexpr double kLaterSafetyCeilingMs = 180.0;

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

std::uint64_t splitmix64(std::uint64_t &state) noexcept {
  state += 0x9e3779b97f4a7c15ULL;
  std::uint64_t value = state;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

void apply_procedural_turn(ps::GameState &state, std::uint64_t &seed) {
  if (ps::is_terminal(state)) {
    throw std::logic_error("procedural actor received a terminal state");
  }
  const ps::Player mover = state.to_move;
  std::size_t edges = 0;
  while (!ps::is_terminal(state) && state.to_move == mover) {
    const std::vector<ps::Move> legal = ps::legal_moves(state);
    if (legal.empty()) {
      throw std::logic_error("in-progress state has no legal move");
    }
    const std::size_t index =
        static_cast<std::size_t>(splitmix64(seed) % legal.size());
    state = ps::apply_move(state, legal[index]);
    if (++edges > 316U) {
      throw std::logic_error("procedural turn exceeded the board edge count");
    }
  }
}

ps::GameState first_response_state(ps::Player player) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  if (player == ps::Player::Two) {
    std::uint64_t seed = 0x2dc88b2f56cc8b17ULL;
    apply_procedural_turn(state, seed);
  }
  if (ps::is_terminal(state) || state.to_move != player) {
    throw std::logic_error("could not construct the first-response state");
  }
  return state;
}

ps::GameState later_response_state(ps::Player player) {
  const int target_turns = player == ps::Player::One ? 8 : 9;
  for (std::uint64_t attempt = 0; attempt < 4'096; ++attempt) {
    ps::GameState state = ps::make_initial_state(codingame_rules());
    std::uint64_t seed = 0x71bfc745d8a13e29ULL +
                         attempt * 0xd1b54a32d192ed03ULL;
    for (int turn = 0; turn < target_turns && !ps::is_terminal(state); ++turn) {
      apply_procedural_turn(state, seed);
    }
    if (!ps::is_terminal(state) && state.to_move == player &&
        state.used_segments.size() >= static_cast<std::size_t>(target_turns)) {
      return state;
    }
  }
  throw std::logic_error("could not construct a nonterminal later state");
}

double timed_choice(ps::GameState &state, std::size_t own_decision) {
  const auto start = std::chrono::steady_clock::now();
  const std::string action =
      candidate::choose_complete_turn(state, own_decision);
  const auto finish = std::chrono::steady_clock::now();
  if (action.empty()) {
    throw std::logic_error("timing probe produced an empty action");
  }
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2 || (std::string_view(argv[1]) != "0" &&
                    std::string_view(argv[1]) != "1")) {
    std::cerr << "usage: timing_probe PLAYER_ID(0|1)\n";
    return 2;
  }
  const int player_id = argv[1][0] - '0';
  const ps::Player player =
      player_id == 0 ? ps::Player::One : ps::Player::Two;
  ps::GameState first_state = first_response_state(player);
  const double first = timed_choice(first_state, 0);
  ps::GameState later_state = later_response_state(player);
  const double later = timed_choice(later_state, 1);
  std::cout << "player=" << player_id << " first_ms=" << first
            << " later_ms=" << later << '\n';
  return first < kFirstSafetyCeilingMs && later < kLaterSafetyCeilingMs ? 0 : 1;
}
