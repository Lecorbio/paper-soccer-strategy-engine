#include "papersoccer/codingame_referee.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string require_value(int &index, int argc, char **argv) {
  if (++index >= argc) {
    throw std::invalid_argument("missing command-line value");
  }
  return argv[index];
}

std::vector<std::string> split_actions(std::string_view value) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t slash = value.find('/', start);
    result.emplace_back(value.substr(start, slash == std::string_view::npos
                                               ? value.size() - start
                                               : slash - start));
    if (slash == std::string_view::npos) break;
    start = slash + 1;
  }
  if (result.size() == 1 && result.front().empty()) result.clear();
  return result;
}

int player_number(std::optional<papersoccer::Player> player) {
  if (!player) return -1;
  return *player == papersoccer::Player::One ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
  using papersoccer::codingame::RefereeConfig;
  try {
    RefereeConfig config;
    std::optional<std::string> transcript;
    std::optional<int> expected_winner;
    bool allow_incomplete = false;
    for (int index = 1; index < argc; ++index) {
      const std::string_view option = argv[index];
      if (option == "--player-one") config.player_one_executable = require_value(index, argc, argv);
      else if (option == "--player-two") config.player_two_executable = require_value(index, argc, argv);
      else if (option == "--player-one-id") config.player_one_id = require_value(index, argc, argv);
      else if (option == "--player-two-id") config.player_two_id = require_value(index, argc, argv);
      else if (option == "--first-timeout-ms") config.first_timeout_ms = std::stoul(require_value(index, argc, argv));
      else if (option == "--later-timeout-ms") config.later_timeout_ms = std::stoul(require_value(index, argc, argv));
      else if (option == "--validate-transcript") transcript = require_value(index, argc, argv);
      else if (option == "--expected-winner") expected_winner = std::stoi(require_value(index, argc, argv));
      else if (option == "--allow-incomplete") allow_incomplete = true;
      else throw std::invalid_argument("unknown command-line option: " + std::string(option));
    }
    if (transcript) {
      const auto result = papersoccer::codingame::validate_transcript(
          split_actions(*transcript), allow_incomplete);
      const int winner = player_number(result.winner);
      if (expected_winner && winner != *expected_winner) {
        throw std::invalid_argument("transcript winner does not match expectation");
      }
      std::cout << "{\"schema\":\"papersoccer.codingame-transcript-validation.v1\","
                << "\"terminal\":" << (result.terminal ? "true" : "false")
                << ",\"winnerPlayer\":";
      if (winner < 0) std::cout << "null"; else std::cout << winner;
      std::cout << ",\"terminalReason\":";
      if (result.terminal_reason) {
        std::cout << '\"' << *result.terminal_reason << '\"';
      } else {
        std::cout << "null";
      }
      std::cout << ",\"acceptedActionCount\":" << result.action_count
                << ",\"edgeCount\":" << result.edge_count << "}\n";
      return 0;
    }
    std::cout << papersoccer::codingame::run_match_json(config) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "referee error: " << error.what() << '\n';
    return 2;
  }
}
