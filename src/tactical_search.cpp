#include "mcts_internal.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>

namespace papersoccer::detail {

namespace {

class ScopedMove {
 public:
  ScopedMove(SearchPosition &position, std::uint8_t slot)
      : position_(position), undo_depth_(position.undo_depth()) {
    position_.make_move(slot);
  }

  ~ScopedMove() { position_.unmake_to(undo_depth_); }

  ScopedMove(const ScopedMove &) = delete;
  ScopedMove &operator=(const ScopedMove &) = delete;

 private:
  SearchPosition &position_;
  std::size_t undo_depth_{};
};

std::uint8_t legal_move_count(const SearchPosition &position) noexcept {
  std::array<std::uint8_t, kMaximumMoves> slots{};
  return position.legal_slots(slots);
}

bool has_direct_tactic(SearchPosition &position) {
  if (position.is_terminal()) {
    return false;
  }

  std::array<std::uint8_t, kMaximumMoves> slots{};
  const std::uint8_t count = position.legal_slots(slots);
  // This is one-ply defensive lookahead, not the probe itself. An
  // unconstrained rebound is extended when it becomes the actual frontier,
  // but does not make every preceding quiet position noisy.
  if (count == 1) {
    return true;
  }
  for (std::uint8_t index = 0; index < count; ++index) {
    ScopedMove move(position, slots[index]);
    if (position.is_terminal() || legal_move_count(position) == 1) {
      return true;
    }
  }
  return false;
}

bool is_tactically_noisy(SearchPosition &position) {
  if (position.is_terminal()) {
    return true;
  }

  std::array<std::uint8_t, kMaximumMoves> slots{};
  const std::uint8_t count = position.legal_slots(slots);
  if (count == 1) {
    return true;
  }

  const Player mover = position.to_move();
  for (std::uint8_t index = 0; index < count; ++index) {
    ScopedMove move(position, slots[index]);
    if (position.is_terminal() || position.to_move() == mover) {
      return true;
    }

    const std::uint8_t reply_count = legal_move_count(position);
    if (reply_count == 1) {
      return true;
    }

    if (position.to_move() == opponent(mover) &&
        has_direct_tactic(position)) {
      return true;
    }
  }
  return false;
}

class TacticalProbe {
 public:
  TacticalProbe(std::uint32_t max_depth, std::uint32_t max_nodes)
      : max_depth_(max_depth), max_nodes_(max_nodes) {}

  TacticalProbeResult run(SearchPosition &position) {
    TacticalProbeResult result;
    const Proof proof = search(position, 0);
    result.proven_winner = proof.winner;
    result.proving_move = proof.move;
    result.stats = stats_;
    return result;
  }

 private:
  struct Proof {
    std::optional<Player> winner{};
    std::optional<Move> move{};
  };

  std::uint32_t max_depth_{};
  std::uint32_t max_nodes_{};
  TacticalProbeStats stats_{};

  Proof search(SearchPosition &position, std::uint32_t depth) {
    if (stats_.nodes >= max_nodes_) {
      stats_.node_cutoff = true;
      return {};
    }

    ++stats_.nodes;
    stats_.max_depth = std::max(stats_.max_depth, depth);

    if (position.is_terminal()) {
      return Proof{position.winner(), std::nullopt};
    }
    if (depth >= max_depth_) {
      stats_.depth_cutoff = true;
      return {};
    }
    if (!is_tactically_noisy(position)) {
      return {};
    }

    const Player mover = position.to_move();
    const Player opposing_player = opponent(mover);
    std::array<std::uint8_t, kMaximumMoves> slots{};
    const std::uint8_t count = position.legal_slots(slots);
    bool every_move_loses = count != 0;

    for (std::uint8_t index = 0; index < count; ++index) {
      const Move candidate_move = position.move_for_slot(slots[index]);
      Proof child;
      {
        ScopedMove move(position, slots[index]);
        child = search(position, depth + 1U);
      }

      if (child.winner.has_value() && *child.winner == mover) {
        // A single proven winning alternative is a complete existential proof.
        return Proof{mover, candidate_move};
      }
      if (!child.winner.has_value() || *child.winner != opposing_player) {
        every_move_loses = false;
      }
      if (stats_.node_cutoff) {
        return {};
      }
    }

    if (every_move_loses) {
      // This is reached only after every authoritative legal alternative has
      // been searched and proven for the opponent.
      return Proof{opposing_player, std::nullopt};
    }
    return {};
  }
};

void restore_and_verify(SearchPosition &position,
                        const SearchPosition &original,
                        std::size_t original_undo_depth) {
  position.unmake_to(original_undo_depth);
  if (position.undo_depth() != original_undo_depth ||
      !position.same_compact_state(original)) {
    throw std::logic_error(
        "tactical probe failed to restore the compact position");
  }
}

}  // namespace

TacticalProbeResult run_tactical_probe(SearchPosition &position,
                                        std::uint32_t max_depth,
                                        std::uint32_t max_nodes) {
  if (max_depth == 0) {
    throw std::invalid_argument(
        "tactical probe maximum depth must be greater than zero");
  }
  if (max_nodes == 0) {
    throw std::invalid_argument(
        "tactical probe maximum node count must be greater than zero");
  }

  const SearchPosition original = position;
  const std::size_t original_undo_depth = position.undo_depth();
  try {
    TacticalProbe probe(max_depth, max_nodes);
    TacticalProbeResult result = probe.run(position);
    restore_and_verify(position, original, original_undo_depth);
    return result;
  } catch (...) {
    const std::exception_ptr failure = std::current_exception();
    restore_and_verify(position, original, original_undo_depth);
    std::rethrow_exception(failure);
  }
}

}  // namespace papersoccer::detail
