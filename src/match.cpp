#include "papersoccer/match.hpp"

#include <utility>

#include "papersoccer/rules.hpp"

namespace papersoccer {

Match::Match(const RulesConfig &config) : state_(make_initial_state(config)) {}

const GameState &Match::state() const noexcept { return state_; }

const std::vector<PlayedMove> &Match::history() const noexcept { return history_; }

std::vector<Move> Match::legal_moves() const { return papersoccer::legal_moves(state_); }

PlayedMove Match::play(Move move) {
  const Player player = state_.to_move;
  const Point from = state_.ball;
  const bool extra_turn = grants_extra_turn(state_, move.to);
  GameState next = apply_move(state_, move);

  const PlayedMove played{
      history_.size() + 1,
      player,
      from,
      move.to,
      extra_turn,
      next.status,
  };

  // Record the move before replacing the state. If growing the history throws,
  // the match still points at the unchanged pre-move state.
  history_.push_back(played);
  state_ = std::move(next);
  return played;
}

}  // namespace papersoccer
