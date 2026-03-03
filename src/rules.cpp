#include "papersoccer/rules.hpp"

#include <algorithm>
#include <stdexcept>

#include "papersoccer/geometry.hpp"

namespace papersoccer {

GameState make_initial_state(const RulesConfig &config) {
  GameState state;
  state.config = config;
  state.ball = Point{config.width / 2, config.height / 2};
  state.to_move = Player::One;
  state.status = Status::InProgress;
  state.path = {state.ball};
  state.visit_count[state.ball] = 1;
  return state;
}

std::vector<Move> legal_moves(const GameState &state) {
  std::vector<Move> result;
  if (state.status != Status::InProgress) {
    return result;
  }

  for (const Point destination : neighbors(state.config, state.ball, state.to_move)) {
    const Segment segment{state.ball, destination};
    if (state.used_segments.find(segment) != state.used_segments.end()) {
      continue;
    }
    if (is_forbidden_boundary_segment(state.config, segment)) {
      continue;
    }
    result.push_back(Move{destination});
  }

  return result;
}

bool grants_extra_turn(const GameState &before, Point destination) {
  if (is_boundary_point(before.config, destination)) {
    return true;
  }
  const auto it = before.visit_count.find(destination);
  return it != before.visit_count.end() && it->second > 0;
}

GameState apply_move(const GameState &state, Move move) {
  if (state.status != Status::InProgress) {
    throw std::invalid_argument("cannot apply move to terminal game state");
  }

  const auto moves = legal_moves(state);
  const auto legal_it = std::find_if(
      moves.begin(), moves.end(),
      [&](const Move &candidate) { return candidate.to == move.to; });
  if (legal_it == moves.end()) {
    throw std::invalid_argument("illegal move");
  }

  GameState next = state;
  next.used_segments.insert(Segment{state.ball, move.to});
  next.ball = move.to;
  next.path.push_back(move.to);
  next.visit_count[move.to] += 1;

  if (is_goal_point(next.config, move.to)) {
    next.status = (state.to_move == Player::One) ? Status::WonByOne : Status::WonByTwo;
    return next;
  }

  const bool extra_turn = grants_extra_turn(state, move.to);
  next.to_move = extra_turn ? state.to_move : opponent(state.to_move);
  next.status = Status::InProgress;

  if (legal_moves(next).empty()) {
    next.status = (next.to_move == Player::One) ? Status::WonByTwo : Status::WonByOne;
  }

  return next;
}

bool is_terminal(const GameState &state) {
  return state.status != Status::InProgress;
}

std::optional<Player> winner(const GameState &state) {
  if (state.status == Status::WonByOne) {
    return Player::One;
  }
  if (state.status == Status::WonByTwo) {
    return Player::Two;
  }
  return std::nullopt;
}

}  // namespace papersoccer
