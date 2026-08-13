#define JACEK_ARENA_BFM_NO_MAIN
#include "submission.cpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace jab = jacek_arena_bfm;

namespace {

jab::State scratch_position(int turns) {
  jab::State state = jab::initial_state();
  for (int turn = 0; turn < turns && !state.terminal(); ++turn) {
    const auto actions = jab::generate_actions(
        state, jab::GeneratorStrategy::PriorityBeam, true);
    if (actions.empty()) break;
    const std::size_t selected =
        static_cast<std::size_t>(jab::state_hash(state) + turn * 17U) %
        actions.size();
    if (!jab::apply_action(state, actions[selected])) {
      throw std::runtime_error("scratch construction failed");
    }
  }
  return state.terminal() ? jab::initial_state() : state;
}

void measure(const char *label, const jab::State &source, int budget_ms) {
  const auto start = std::chrono::steady_clock::now();
  jab::State state = source;
  const auto result = jab::search(
      state, start + std::chrono::milliseconds(budget_ms));
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count();
  if (result.action.length == 0 || !jab::apply_action(state, result.action)) {
    throw std::runtime_error("timed search returned illegal action");
  }
  std::cout << label << " budget_ms=" << budget_ms
            << " elapsed_us=" << elapsed << " nodes=" << result.nodes
            << " root_actions=" << result.root_actions << '\n';
}

}  // namespace

int main() {
  try {
    const jab::State first = jab::initial_state();
    const jab::State later = scratch_position(10);
    measure("player0_first", first, 800);
    measure("player1_first", jab::rotate_and_swap(first), 800);
    measure("player0_later", later.to_move == 0 ? later
                                                 : jab::rotate_and_swap(later),
            155);
    measure("player1_later", later.to_move == 1 ? later
                                                 : jab::rotate_and_swap(later),
            155);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "timing_probe: " << error.what() << '\n';
    return 1;
  }
}
