#define COMPACT_VALUE_BFM_NO_MAIN
#include "submission.cpp"

#include <bit>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace cv = compact_value_bfm;

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    cv::SparseFeatures features;
    std::istringstream input(line);
    std::string field;
    int previous = -1;
    bool valid = true;
    while (std::getline(input, field, ',')) {
      try {
        const int value = std::stoi(field);
        if (value <= previous || value < 0 || value >= cv::kFeatureCount ||
            features.count >= features.indices.size()) {
          valid = false;
          break;
        }
        previous = value;
        features.indices[features.count++] = static_cast<std::uint16_t>(value);
      } catch (const std::exception &) {
        valid = false;
        break;
      }
    }
    if (!valid || features.count == 0) {
      std::cout << "invalid\n";
      continue;
    }
    const float value = cv::deployment_model().evaluate(features);
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(value) << std::dec << '\n';
  }
  return 0;
}
