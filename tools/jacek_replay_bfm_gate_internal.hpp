#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "papersoccer/types.hpp"

namespace papersoccer::jacek_replay_gate {

struct ControlConfig {
  std::uint32_t time_ms{980};
  std::uint64_t work_cap{3'000'000};
  std::size_t tree_cap{100'000};
};

struct ControlDecision {
  std::vector<Move> action;
  std::uint64_t work{};
  std::size_t tree_nodes{};
  std::uint64_t neural_evaluations{};
  std::uint32_t depth{};
  float root_value{};
  bool deadline_reached{};
  bool cap_reached{};
  bool replay_correction{};
};

ControlDecision choose_rank4_control(const GameState &state,
                                     const ControlConfig &config);
ControlDecision choose_neural_puct_control(const GameState &state,
                                           const ControlConfig &config);
ControlDecision choose_jacek_nn_control(const GameState &state,
                                        std::string_view transcript,
                                        const ControlConfig &config);

}  // namespace papersoccer::jacek_replay_gate
