#include "engine.hpp"
#include "model.hpp"

#include <chrono>
#include <iostream>
#include <string>

namespace jacek_arena_bfm {
namespace {

bool parse_action(const std::string &text, Action &action) {
  action = {};
  if (text.empty() || text.size() > kMaximumActionLength) return false;
  for (const char character : text) {
    if (character < '0' || character > '7') return false;
    action.directions[action.length++] =
        static_cast<std::uint8_t>(character - '0');
  }
  return true;
}

[[maybe_unused]] int run_protocol() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  int my_id = -1;
  if (!(std::cin >> my_id) || (my_id != 0 && my_id != 1)) return 1;
  State state = initial_state();
  bool first_decision = true;
  for (;;) {
    int opponent_length = -1;
    std::string opponent_text;
    if (!(std::cin >> opponent_length >> opponent_text)) break;
    if (opponent_text == "-") {
      if (my_id != 0 || state.ply != 0) return 2;
    } else {
      if (opponent_length < 1 ||
          static_cast<std::size_t>(opponent_length) != opponent_text.size() ||
          state.to_move == my_id) return 3;
      Action opponent_action;
      if (!parse_action(opponent_text, opponent_action) ||
          !apply_action(state, opponent_action)) return 4;
    }
    if (state.terminal()) break;
    if (state.to_move != my_id) return 5;
    const auto started = std::chrono::steady_clock::now();
    // Leave substantial wall-clock headroom for the CodinGame protocol and
    // slower shared runners. Search additionally reserves its own finalization
    // time and can interrupt complete-turn generation at the deadline.
    const int milliseconds = first_decision ? 780 : 128;
    const auto deadline = started + std::chrono::milliseconds(milliseconds);
    const SearchResult result = search(state, deadline);
    if (result.action.length == 0 || !apply_action(state, result.action)) return 6;
    std::cout << result.action.text() << '\n' << std::flush;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    std::cerr << "jacek_arena_bfm nodes=" << result.nodes
              << " root_actions=" << result.root_actions
              << " generator_us=" << result.generator_microseconds
              << " generator_max_us=" << result.maximum_generator_microseconds
              << " generator_deadline_stops="
              << result.generator_deadline_stops
              << " deadline_reached=" << result.deadline_reached
              << " ms=" << elapsed << " value=" << result.root_value
              << " model=" << model::kIdentity << '\n';
    first_decision = false;
    if (state.terminal()) break;
  }
  return 0;
}

}  // namespace
}  // namespace jacek_arena_bfm

#ifndef JACEK_ARENA_BFM_NO_MAIN
int main() { return jacek_arena_bfm::run_protocol(); }
#endif
