#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string_view>

#include "papersoccer/bot.hpp"

namespace papersoccer::jacek_replay_search_teacher {

inline constexpr std::string_view kCompleteTurnActionGroupSchema =
    "papersoccer.jacek-replay-complete-turn-action-group.v1";

std::uint64_t derive_search_seed(std::string_view campaign_id,
                                 std::string_view position_id,
                                 std::size_t max_tree_nodes);

void validate_model_identity(std::string_view actual_sha256,
                             std::string_view expected_sha256);

float direct_teacher_value(const JacekReplayBfmSearchStats &stats,
                           Player mover,
                           std::size_t expected_tree_nodes);

float direct_action_teacher_value(
    const JacekReplayBfmRootActionAnalysis &action, Player mover);

float successor_frame_teacher_value(
    const JacekReplayBfmRootActionAnalysis &action, Player parent_mover,
    Player successor_mover);

int run(int argc, char **argv, std::istream &input, std::ostream &output);

}  // namespace papersoccer::jacek_replay_search_teacher
