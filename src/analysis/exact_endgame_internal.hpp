#pragma once

#include <cstdint>

#include "papersoccer/game_review.hpp"

namespace papersoccer::detail {

// Private deterministic harness used to exercise budget exhaustion. Production
// callers always go through ExactEndgameSolver's fixed 250,000-node limit.
ExactEndgameResult solve_exact_endgame_with_limit(const GameState &state,
                                                  std::uint64_t node_limit);

}  // namespace papersoccer::detail
