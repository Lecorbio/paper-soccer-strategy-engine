#pragma once

#include <cstddef>
#include <vector>

#include "papersoccer/types.hpp"

namespace papersoccer {

struct PlayedMove {
  std::size_t ply{};
  Player player{Player::One};
  Point from{};
  Point to{};
  bool extra_turn{false};
  Status status_after{Status::InProgress};

  constexpr bool operator==(const PlayedMove &) const noexcept = default;
};

class Match {
 public:
  explicit Match(const RulesConfig &config = {});

  const GameState &state() const noexcept;
  const std::vector<PlayedMove> &history() const noexcept;

  std::vector<Move> legal_moves() const;
  PlayedMove play(Move move);

 private:
  GameState state_;
  std::vector<PlayedMove> history_;
};

}  // namespace papersoccer
