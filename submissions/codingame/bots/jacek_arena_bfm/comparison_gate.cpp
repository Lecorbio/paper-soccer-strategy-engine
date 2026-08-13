#define JACEK_ARENA_BFM_NO_MAIN
#include "submission.cpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace jab = jacek_arena_bfm;

int main() {
  try {
    jab::State state = jab::initial_state();
    std::size_t checked = 0;
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
      const auto result = jab::search(
          state, std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(3),
          jab::SearchConfig{jab::GeneratorStrategy::TacticalProgressive,
                            512, 0.95});
      jab::State selected = state;
      if (result.action.length == 0 ||
          !jab::apply_action(selected, result.action)) {
        throw std::runtime_error("search legality gate failed");
      }
      if (exact_win &&
          !(selected.terminal() && selected.winner == state.to_move)) {
        throw std::runtime_error("exact winning action was displaced");
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
              << checked << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek_arena_bfm comparison gate failed: " << error.what()
              << '\n';
    return 1;
  }
}
