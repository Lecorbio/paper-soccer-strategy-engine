#define COMPACT_VALUE_BFM_NO_MAIN
#include "submission.cpp"

#include <iostream>
#include <string>

namespace cv = compact_value_bfm;

int main() {
  cv::State state = cv::initial_state();
  std::string line;
  while (std::getline(std::cin, line)) {
    state = cv::initial_state();
    std::size_t begin = 0;
    bool valid = true;
    while (begin < line.size()) {
      const std::size_t end = line.find('/', begin);
      const std::string text = line.substr(
          begin, end == std::string::npos ? line.size() - begin : end - begin);
      cv::Action action;
      for (const char character : text) {
        if (character < '0' || character > '7' ||
            action.length >= action.directions.size()) {
          valid = false;
          break;
        }
        action.directions[action.length++] =
            static_cast<std::uint8_t>(character - '0');
      }
      if (!valid || !cv::apply_action(state, action)) {
        valid = false;
        break;
      }
      if (end == std::string::npos) break;
      begin = end + 1U;
    }
    if (!valid || state.terminal()) {
      std::cout << "invalid\n";
      continue;
    }
    const auto features = cv::active_features(state);
    for (std::size_t index = 0; index < features.count; ++index) {
      if (index != 0) std::cout << ',';
      std::cout << features.indices[index];
    }
    std::cout << '\n';
  }
  return 0;
}
