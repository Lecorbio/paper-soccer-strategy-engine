#pragma once

#include <cstdint>
#include <vector>

#include "papersoccer/types.hpp"

namespace papersoccer::rank4_jacek_gate {

struct EngineConfig {
  std::uint64_t max_nodes{3'000'000};
  std::uint32_t time_ms{};
  std::uint8_t exact_proof_mask{};
};

struct EngineDecision {
  std::vector<Move> moves;
  std::uint64_t nodes{};
  std::uint32_t completed_depth{};
  std::uint32_t attempted_depth{};
  std::uint64_t rebound_goal_probes{};
  std::uint64_t rebound_goal_hits{};
  std::uint64_t rebound_loss_hits{};
  std::uint64_t root_rebound_probes{};
  std::uint64_t root_rebound_win_hits{};
  std::uint64_t root_rebound_loss_hits{};
  std::uint64_t leaf_rebound_probes{};
  std::uint64_t leaf_rebound_win_hits{};
  std::uint64_t leaf_rebound_loss_hits{};
  std::uint64_t exchange_ply1_probes{};
  std::uint64_t exchange_ply1_win_hits{};
  std::uint64_t exchange_ply1_loss_hits{};
  std::uint64_t exchange_ply1_cutoffs{};
  std::uint64_t exchange_ply2_probes{};
  std::uint64_t exchange_ply2_win_hits{};
  std::uint64_t exchange_ply2_loss_hits{};
  std::uint64_t exchange_ply2_cutoffs{};
  bool budget_exhausted{};
};

EngineDecision choose_rank4(const GameState &state,
                            const EngineConfig &config);
EngineDecision choose_hybrid(const GameState &state,
                             const EngineConfig &config);
EngineDecision choose_hybrid_control(const GameState &state,
                                     const EngineConfig &config);

}  // namespace papersoccer::rank4_jacek_gate
