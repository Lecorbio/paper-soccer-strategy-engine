#define COMPACT_VALUE_BFM_NO_MAIN
#include "submission.cpp"

#include <iostream>
#include <stdexcept>

namespace cv = compact_value_bfm;

namespace {

cv::Action rotated_action(cv::Action action) {
  for (std::size_t index = 0; index < action.length; ++index) {
    action.directions[index] =
        static_cast<std::uint8_t>((action.directions[index] + 4U) & 7U);
  }
  return action;
}

}  // namespace

int main() {
  try {
    cv::State state = cv::initial_state();
    std::size_t checked = 0;
    for (int sample = 0; sample < 24; ++sample) {
      if (state.terminal() || sample % 11 == 0) state = cv::initial_state();
      const cv::State rotated = cv::rotate_and_swap(state);
      cv::SearchConfig config;
      config.max_actions = 32;
      config.root_partial_paths = 256;
      config.nonroot_partial_paths = 64;
      config.max_tree_nodes = 512;
      config.max_expansions = 16;
      const cv::Action emergency = cv::emergency_complete_action(state);
      const cv::Action rotated_emergency = rotated_action(emergency);
      const auto left = cv::search(
          state, std::chrono::steady_clock::time_point::max(), config,
          &emergency);
      const auto right = cv::search(
          rotated, std::chrono::steady_clock::time_point::max(), config,
          &rotated_emergency);
      cv::State left_after = state;
      cv::State right_after = rotated;
      if (!cv::apply_action(left_after, left.action) ||
          !cv::apply_action(right_after, right.action) ||
          !cv::same_boundary(cv::rotate_and_swap(left_after), right_after)) {
        throw std::runtime_error(
            "fixed-work color rotation mismatch sample=" +
            std::to_string(sample) + " left=" + left.action.text() +
            " right=" + right.action.text());
      }
      ++checked;
      cv::GeneratorConfig generator;
      generator.max_actions = 64;
      generator.max_partial_paths = 512;
      const auto actions = cv::generate_complete_turns(state, generator);
      if (actions.actions.empty()) throw std::runtime_error("empty advance");
      const std::size_t advance =
          (cv::canonical_state_hash(state) + sample * 29U) % actions.actions.size();
      if (!cv::apply_action(state, actions.actions[advance].action)) {
        throw std::runtime_error("illegal procedural advance");
      }
    }
    std::cout << "compact_value_bfm comparison smoke passed positions="
              << checked << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "compact_value_bfm comparison smoke failure: "
              << error.what() << '\n';
    return 1;
  }
}
