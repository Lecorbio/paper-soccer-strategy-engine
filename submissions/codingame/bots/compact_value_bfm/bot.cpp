#include "engine.hpp"
#include "model.hpp"

#include <chrono>
#include <iostream>
#include <limits>
#include <string>

namespace compact_value_bfm {
namespace {

bool parse_action(const std::string &text, Action &action) noexcept {
  action = {};
  if (text.empty() || text.size() > action.directions.size()) return false;
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
  int player_id = -1;
  if (!(std::cin >> player_id) || (player_id != 0 && player_id != 1)) return 1;
  State state = initial_state();
  bool first_decision = true;
  std::uint32_t decision = 0;
  for (;;) {
    int opponent_length = -1;
    std::string opponent_text;
    if (!(std::cin >> opponent_length >> opponent_text)) break;
    if (opponent_text == "-") {
      if (player_id != 0 || state.ply != 0 || opponent_length != 1) return 2;
    } else {
      if (opponent_length < 1 ||
          static_cast<std::size_t>(opponent_length) != opponent_text.size() ||
          state.to_move == player_id) return 3;
      Action opponent;
      if (!parse_action(opponent_text, opponent) ||
          !apply_action(state, opponent)) return 4;
    }
    if (state.terminal()) break;
    if (state.to_move != player_id) return 5;

    // Establish a complete legal response before any timed work.
    const Action emergency = emergency_complete_action(state);
    if (emergency.length == 0) return 6;
    const int budget_ms = first_decision ? 800 : 155;
    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started + std::chrono::milliseconds(budget_ms);
    SearchResult result;
    try {
      result = search(state, deadline, SearchConfig{}, &emergency);
    } catch (const std::exception &) {
      result.action = emergency;
      result.stats.deadline_reached = true;
    }
    if (result.action.length == 0 || !apply_action(state, result.action)) return 7;
    std::cout << result.action.text() << '\n' << std::flush;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    std::cerr << "CVB d=" << decision
              << " n=" << result.stats.tree_nodes
              << " x=" << result.stats.expansions
              << " dl=" << result.stats.deadline_reached
              << " ms=" << elapsed
              << " v=" << result.value
              << " m=" << model::kIdentity << '\n';
    ++decision;
    first_decision = false;
    if (state.terminal()) break;
  }
  return 0;
}

}  // namespace
}  // namespace compact_value_bfm

#ifndef COMPACT_VALUE_BFM_NO_MAIN
int main() { return compact_value_bfm::run_protocol(); }
#endif
