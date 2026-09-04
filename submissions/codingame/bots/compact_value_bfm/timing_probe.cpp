#define COMPACT_VALUE_BFM_NO_MAIN
#include "submission.cpp"

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace cv = compact_value_bfm;

namespace {

cv::State procedural_state(int turns) {
  cv::State state = cv::initial_state();
  cv::GeneratorConfig config;
  config.max_actions = 32;
  config.max_partial_paths = 256;
  for (int turn = 0; turn < turns && !state.terminal(); ++turn) {
    const auto generated = cv::generate_complete_turns(state, config);
    if (generated.actions.empty()) break;
    const std::size_t selected =
        (cv::canonical_state_hash(state) + static_cast<std::uint64_t>(turn) * 17U) %
        generated.actions.size();
    if (!cv::apply_action(state, generated.actions[selected].action)) {
      throw std::runtime_error("procedural timing state");
    }
  }
  return state.terminal() ? cv::initial_state() : state;
}

void measure(const char *label, const cv::State &source, int budget_ms) {
  cv::State state = source;
  const cv::Action emergency = cv::emergency_complete_action(state);
  const auto started = std::chrono::steady_clock::now();
  const auto result = cv::search(
      state, started + std::chrono::milliseconds(budget_ms),
      cv::SearchConfig{}, &emergency);
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - started).count();
  if (result.action.length == 0 || !cv::apply_action(state, result.action)) {
    throw std::runtime_error("timed BFM returned an illegal action");
  }
  std::cout << label << " budget_ms=" << budget_ms
            << " elapsed_us=" << elapsed
            << " nodes=" << result.stats.tree_nodes
            << " expansions=" << result.stats.expansions
            << " deadline=" << result.stats.deadline_reached << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  try {
    int player = -1;
    if (argc == 2) player = std::stoi(argv[1]);
    if (argc > 2 || player < -1 || player > 1) {
      throw std::invalid_argument("usage: timing_probe [0|1]");
    }
    const cv::State initial = cv::initial_state();
    cv::State later = procedural_state(10);
    if (player < 0 || player == 0) {
      measure("player0_first", initial, 800);
      measure("player0_later", later.to_move == 0 ? later
                                                   : cv::rotate_and_swap(later),
              155);
      measure("player0_later_initial", initial, 155);
    }
    if (player < 0 || player == 1) {
      measure("player1_first", cv::rotate_and_swap(initial), 800);
      measure("player1_later", later.to_move == 1 ? later
                                                   : cv::rotate_and_swap(later),
              155);
      measure("player1_later_initial", cv::rotate_and_swap(initial), 155);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "compact_value_bfm timing probe failure: " << error.what()
              << '\n';
    return 1;
  }
}
