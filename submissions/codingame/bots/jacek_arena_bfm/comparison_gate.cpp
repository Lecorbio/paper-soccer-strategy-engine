#define JACEK_ARENA_BFM_NO_MAIN
#include "submission.cpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace jab = jacek_arena_bfm;

int main() {
  try {
    jab::State state = jab::initial_state();
    std::size_t checked = 0;
    std::size_t exact_winning_positions = 0;
    for (int sample = 0; sample < 96; ++sample) {
      if (state.terminal() || sample % 19 == 0) state = jab::initial_state();
      const jab::State rotated = jab::rotate_and_swap(state);
      if (std::abs(jab::evaluate(state) - jab::evaluate(rotated)) > 1.0e-5F) {
        throw std::runtime_error("color symmetry gate failed");
      }
      const auto reference = jab::generate_actions(
          state, jab::GeneratorStrategy::HighCapRecall, true);
      if (reference.empty()) throw std::runtime_error("recall generator is empty");
      bool exact_win = false;
      for (const auto &action : reference) {
        jab::State successor = state;
        if (!jab::apply_action(successor, action)) {
          throw std::runtime_error("recall generator produced illegal action");
        }
        exact_win = exact_win ||
            (successor.terminal() && successor.winner == state.to_move);
      }
      const auto tactical = jab::generate_actions(
          state, jab::GeneratorStrategy::TacticalProgressive, true);
      bool tactical_exact_win = false;
      for (const auto &action : tactical) {
        jab::State successor = state;
        if (!jab::apply_action(successor, action)) {
          throw std::runtime_error("tactical generator produced illegal action");
        }
        tactical_exact_win = tactical_exact_win ||
            (successor.terminal() && successor.winner == state.to_move);
      }
      if (exact_win && !tactical_exact_win) {
        std::ostringstream message;
        message << "tactical generator dropped exact winning action sample="
                << sample << " state_hash=" << jab::state_hash(state)
                << " reference_actions=" << reference.size()
                << " tactical_actions=" << tactical.size();
        throw std::runtime_error(message.str());
      }
      if (exact_win) ++exact_winning_positions;
      // This is a tactical-retention invariant, not a wall-clock benchmark.
      // A 3 ms deadline became vacuous once search reserved 6 ms for safe
      // finalization.  An unbounded deadline plus a fixed 512-node cap makes
      // the evidence deterministic while retaining finite work.
      const auto result = jab::search(
          state, std::chrono::steady_clock::time_point::max(),
          jab::SearchConfig{jab::GeneratorStrategy::TacticalProgressive,
                            512, 0.95});
      jab::State selected = state;
      if (result.action.length == 0 ||
          !jab::apply_action(selected, result.action)) {
        throw std::runtime_error("search legality gate failed");
      }
      if (exact_win &&
          !(selected.terminal() && selected.winner == state.to_move)) {
        std::ostringstream message;
        message << "exact winning action was displaced sample=" << sample
                << " state_hash=" << jab::state_hash(state)
                << " reference_actions=" << reference.size()
                << " tactical_actions=" << tactical.size()
                << " search_root_actions=" << result.root_actions
                << " search_nodes=" << result.nodes;
        throw std::runtime_error(message.str());
      }
      ++checked;
      const std::size_t advance =
          static_cast<std::size_t>(jab::state_hash(state) + sample * 13U) %
          reference.size();
      if (!jab::apply_action(state, reference[advance])) {
        throw std::runtime_error("procedural advance failed");
      }
    }
    std::cout << "jacek_arena_bfm comparison gate passed positions="
              << checked
              << " exact_winning_positions=" << exact_winning_positions
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek_arena_bfm comparison gate failed: " << error.what()
              << '\n';
    return 1;
  }
}
