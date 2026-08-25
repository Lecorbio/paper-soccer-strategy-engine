#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "papersoccer/bot.hpp"

namespace papersoccer::jacek_replay_search_teacher {

std::uint64_t derive_search_seed(std::string_view campaign_id,
                                 std::string_view position_id,
                                 std::size_t max_tree_nodes);

void validate_model_identity(std::string_view actual_sha256,
                             std::string_view expected_sha256);

float direct_teacher_value(const JacekReplayBfmSearchStats &stats,
                           Player mover);

}  // namespace papersoccer::jacek_replay_search_teacher
