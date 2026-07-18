#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "papersoccer/bot.hpp"
#include "papersoccer/rules.hpp"

namespace papersoccer::codingame {

constexpr std::uint32_t kFirstSearchTimeMs = 850;
constexpr std::uint32_t kLaterSearchTimeMs = 150;

constexpr std::array<Point, 8> kDirectionDeltas{{
    {0, -1},
    {1, -1},
    {1, 0},
    {1, 1},
    {0, 1},
    {-1, 1},
    {-1, 0},
    {-1, -1},
}};

Player player_for_id(int player_id) {
  if (player_id == 0) {
    return Player::One;
  }
  if (player_id == 1) {
    return Player::Two;
  }
  throw std::invalid_argument("player id must be zero or one");
}

Move decode_direction(Point from, char direction) {
  if (direction < '0' || direction > '7') {
    throw std::invalid_argument("move direction must be between zero and seven");
  }
  const Point delta = kDirectionDeltas[static_cast<std::size_t>(direction - '0')];
  return Move{{from.x + delta.x, from.y + delta.y}};
}

char encode_direction(Point from, Point to) {
  const Point delta{to.x - from.x, to.y - from.y};
  for (std::size_t index = 0; index < kDirectionDeltas.size(); ++index) {
    if (kDirectionDeltas[index] == delta) {
      return static_cast<char>('0' + index);
    }
  }
  throw std::invalid_argument("move is not an adjacent paper-soccer direction");
}

bool contains_move(const std::vector<Move> &moves, Move candidate) {
  return std::find(moves.begin(), moves.end(), candidate) != moves.end();
}

int fallback_score(const GameState &state, Move move) {
  const Player mover = state.to_move;
  const GameState next = apply_move(state, move);
  if (is_terminal(next)) {
    return winner(next) == mover ? 1'000'000 : -1'000'000;
  }

  const int progress = mover == Player::One ? state.ball.y - move.to.y
                                             : move.to.y - state.ball.y;
  const int center = state.config.width / 2;
  const int center_alignment = center - std::abs(move.to.x - center);
  const int mobility = static_cast<int>(legal_moves(next).size());
  const int mobility_score = next.to_move == mover ? mobility : -mobility;
  const int continuation = next.to_move == mover ? 25 : 0;
  return progress * 100 + center_alignment * 4 + mobility_score * 3 +
         continuation;
}

Move choose_fallback(const GameState &state,
                     const std::vector<Move> &moves) {
  if (moves.empty()) {
    throw std::invalid_argument("cannot choose a fallback without a legal move");
  }

  Move best = moves.front();
  int best_score = fallback_score(state, best);
  for (std::size_t index = 1; index < moves.size(); ++index) {
    const int score = fallback_score(state, moves[index]);
    if (score > best_score) {
      best = moves[index];
      best_score = score;
    }
  }
  return best;
}

std::string complete_turn_from_plan(GameState &state,
                                    const std::vector<Move> &plan) {
  if (is_terminal(state)) {
    throw std::invalid_argument("cannot compose a move from a terminal state");
  }

  const Player mover = state.to_move;
  std::size_t plan_index = 0;
  bool following_plan = true;
  std::string encoded;

  while (!is_terminal(state) && state.to_move == mover) {
    const std::vector<Move> moves = legal_moves(state);
    if (moves.empty()) {
      throw std::logic_error("an in-progress state must have a legal move");
    }

    std::optional<Move> selected;
    if (following_plan && plan_index < plan.size() &&
        contains_move(moves, plan[plan_index])) {
      selected = plan[plan_index++];
    } else {
      following_plan = false;
      selected = choose_fallback(state, moves);
    }

    encoded.push_back(encode_direction(state.ball, selected->to));
    state = apply_move(state, *selected);
  }

  if (encoded.empty()) {
    throw std::logic_error("a complete turn must contain at least one move");
  }
  return encoded;
}

std::string choose_complete_turn(GameState &state,
                                 std::uint32_t search_time_ms) {
  AlphaBetaConfig config;
  config.max_turn_depth = AlphaBetaConfig::maximum_turn_depth;
  config.max_nodes = 5'000'000;
  config.transposition_table_entries = 65'536;
  config.max_search_plies = 10;
  config.max_time_ms = search_time_ms;

  AlphaBetaBot bot(config);
  const Move root_move = bot.choose_move(state);
  std::vector<Move> plan = bot.last_search_stats().principal_variation;
  if (plan.empty() || !(plan.front() == root_move)) {
    plan = {root_move};
  }
  return complete_turn_from_plan(state, plan);
}

void apply_encoded_turn(GameState &state, std::string_view encoded) {
  if (encoded.empty()) {
    throw std::invalid_argument("a turn cannot be empty");
  }

  GameState next = state;
  const Player mover = next.to_move;
  for (const char direction : encoded) {
    if (is_terminal(next) || next.to_move != mover) {
      throw std::invalid_argument("turn contains moves after it has ended");
    }
    next = apply_move(next, decode_direction(next.ball, direction));
  }

  if (!is_terminal(next) && next.to_move == mover) {
    throw std::invalid_argument("turn ends before a required rebound move");
  }
  state = std::move(next);
}

}  // namespace papersoccer::codingame

#ifndef PAPER_SOCCER_CODINGAME_NO_MAIN
int main() {
  using namespace papersoccer;
  using namespace papersoccer::codingame;

  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  int player_id = -1;
  if (!(std::cin >> player_id)) {
    return 0;
  }

  Player me;
  try {
    me = player_for_id(player_id);
  } catch (const std::invalid_argument &) {
    return 1;
  }
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  RulesConfig rules;
  rules.goal_rule = GoalRule::OwnGoalsAllowed;
  rules.blocked_rule = BlockedRule::MoverLoses;
  GameState state = make_initial_state(rules);
  bool first_execution = true;

  while (true) {
    int opponent_move_length = 0;
    if (!(std::cin >> opponent_move_length)) {
      break;
    }
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string opponent_move;
    if (!std::getline(std::cin, opponent_move)) {
      break;
    }
    if (opponent_move_length != static_cast<int>(opponent_move.size())) {
      return 1;
    }

    try {
      if (opponent_move == "-") {
        if (player_id != 0 || !first_execution) {
          return 1;
        }
      } else {
        apply_encoded_turn(state, opponent_move);
      }

      if (is_terminal(state)) {
        return 0;
      }
      if (state.to_move != me) {
        return 1;
      }

      const std::uint32_t search_time =
          first_execution ? kFirstSearchTimeMs : kLaterSearchTimeMs;
      const std::string action = choose_complete_turn(state, search_time);
      std::cout << action << std::endl;
      first_execution = false;
    } catch (const std::exception &) {
      return 1;
    }
  }
  return 0;
}
#endif
