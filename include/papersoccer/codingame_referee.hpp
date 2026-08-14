#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "papersoccer/types.hpp"

namespace papersoccer::codingame {

struct RefereeConfig {
  std::string player_one_executable;
  std::string player_two_executable;
  std::string player_one_id;
  std::string player_two_id;
  std::uint32_t first_timeout_ms{1000};
  std::uint32_t later_timeout_ms{200};
  std::size_t output_limit_bytes{65536};
  std::size_t stderr_limit_bytes{65536};
};

struct ReplayResult {
  bool terminal{false};
  std::optional<Player> winner;
  std::optional<std::string> terminal_reason;
  std::size_t action_count{0};
  std::size_t edge_count{0};
};

ReplayResult validate_transcript(const std::vector<std::string> &actions,
                                 bool allow_incomplete = false);

std::string run_match_json(const RefereeConfig &config);

}  // namespace papersoccer::codingame
