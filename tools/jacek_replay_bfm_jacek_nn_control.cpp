#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 jacek_replay_jacek_nn_control_engine
#include "../submissions/codingame/bots/jacek_nn/bot.cpp"
#undef turn_action_v2

#include "jacek_replay_bfm_gate_internal.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

namespace papersoccer::jacek_replay_gate {
namespace {

std::vector<Move> decode_action(const GameState &state,
                                std::string_view encoded) {
  namespace engine = papersoccer::jacek_replay_jacek_nn_control_engine;
  if (encoded.empty()) {
    throw std::logic_error("jacek_nn returned an empty action");
  }
  GameState replay = state;
  const Player mover = replay.to_move;
  std::vector<Move> action;
  action.reserve(encoded.size());
  for (const char direction : encoded) {
    if (direction < '0' || direction > '7') {
      throw std::logic_error("jacek_nn returned an invalid direction");
    }
    if (is_terminal(replay) || replay.to_move != mover) {
      throw std::logic_error("jacek_nn returned an overlong action");
    }
    const Point delta = engine::kDirectionDeltas[
        static_cast<std::size_t>(direction - '0')];
    const Move move{{replay.ball.x + delta.x, replay.ball.y + delta.y}};
    action.push_back(move);
    replay = apply_move(replay, move);
  }
  if (!is_terminal(replay) && replay.to_move == mover) {
    throw std::logic_error(
        "jacek_nn stopped during a mandatory rebound");
  }
  return action;
}

}  // namespace

ControlDecision choose_jacek_nn_control(const GameState &state,
                                        std::string_view transcript,
                                        const ControlConfig &config) {
  namespace engine = papersoccer::jacek_replay_jacek_nn_control_engine;
  ControlDecision result;
  GameState replay = state;
  std::string correction;
  const int player_id = state.to_move == Player::One ? 0 : 1;
  if (engine::try_replay_correction(replay, player_id, transcript,
                                    correction)) {
    result.action = decode_action(state, correction);
    result.replay_correction = true;
    return result;
  }

  engine::SearchConfig search_config;
  search_config.max_nodes = config.work_cap;
  search_config.replay_value_blend_percent = 15;
  search_config.teacher_residual_weight_percent = 50;
  if (config.time_ms != 0) {
    search_config.absolute_deadline =
        engine::SearchClock::now() +
        std::chrono::milliseconds(config.time_ms);
  }
  engine::CompleteTurnSearch search(state, search_config);
  result.action = search.run();
  const engine::SearchStats &stats = search.stats();
  result.work = stats.nodes;
  result.neural_evaluations = stats.teacher_residual_evaluations;
  result.depth = stats.completed_turn_depth;
  result.root_value = static_cast<float>(stats.root_score) / 100'000.0F;
  result.deadline_reached = stats.budget_exhausted;
  result.cap_reached = stats.nodes >= config.work_cap;
  return result;
}

}  // namespace papersoccer::jacek_replay_gate
