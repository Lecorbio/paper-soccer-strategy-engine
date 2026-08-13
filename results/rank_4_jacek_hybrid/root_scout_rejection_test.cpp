#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING
#define private public
#include "source_payloads/1192d91be6526924a7ae5930e12d6b44ddd6848931522c8073ed07d2e43a932c.source"
#undef private
#undef PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

ps::GameState make_clean_state_at(ps::Point point, ps::Player player) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  state.ball = point;
  state.to_move = player;
  state.status = ps::Status::InProgress;
  state.path = {point};
  state.used_segments.clear();
  state.visit_count.clear();
  state.visit_count[point] = 1;
  return state;
}

void block_edges_except(ps::GameState &state, ps::Point from,
                        const std::vector<ps::Point> &allowed) {
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const ps::Point destination{from.x + dx, from.y + dy};
      if (std::find(allowed.begin(), allowed.end(), destination) ==
          allowed.end()) {
        state.used_segments.insert(ps::Segment{from, destination});
      }
    }
  }
}

ps::GameState root_child_loss_choice(ps::Player mover) {
  const bool player_one = mover == ps::Player::One;
  const int forward = player_one ? -1 : 1;
  const ps::Point root{4, 6};
  const ps::Point reply_trap{player_one ? 5 : 3, 6 + forward};
  const ps::Point quiet{player_one ? 3 : 5, 6 + forward};
  const ps::Point closed_rebound{reply_trap.x, reply_trap.y + forward};

  ps::GameState state = make_clean_state_at(root, mover);
  state.visit_count[closed_rebound] = 1;
  block_edges_except(state, root, {reply_trap, quiet});
  block_edges_except(state, reply_trap, {root, closed_rebound});
  block_edges_except(state, closed_rebound, {reply_trap});
  return state;
}

struct Decision {
  std::string action;
  int score{};
  cg::SearchStats stats;
};

Decision fixed_depth_decision(const ps::GameState &state,
                              cg::SearchConfig config) {
  cg::CompleteTurnSearch search(state, config);
  const cg::RootResult result = search.run_fixed_depth_for_test(1);
  ps::GameState replay = state;
  std::string action;
  for (const ps::Move move : result.action) {
    action.push_back(cg::encode_direction(replay.ball, move.to));
    replay = ps::apply_move(replay, move);
  }
  return Decision{action, result.score, search.stats()};
}

cg::SearchConfig proof_config() {
  cg::SearchConfig config;
  config.max_nodes = 500'000;
  config.max_time_ms = 0;
  config.root_seed_endpoints = false;
  config.exact_rebound_proof = true;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  return config;
}

void scout_must_not_poison_the_exact_leaf_cache() {
  const ps::GameState state = root_child_loss_choice(ps::Player::One);
  cg::SearchConfig control_config = proof_config();
  const Decision control = fixed_depth_decision(state, control_config);

  cg::SearchConfig scout_config = control_config;
  scout_config.root_tactical_scout = true;
  scout_config.scout_max_work = 512;
  scout_config.scout_max_actions = 64;
  scout_config.scout_max_time_ms = 0;
  const Decision scout = fixed_depth_decision(state, scout_config);

  std::cout << "cache_poison control=" << control.action << '/' << control.score
            << " proof_loss_hits=" << control.stats.rebound_loss_hits
            << " scout=" << scout.action << '/' << scout.score
            << " proof_loss_hits=" << scout.stats.rebound_loss_hits
            << " cache_hits=" << scout.stats.evaluation_cache_hits << '\n';
  require(scout.action == control.action && scout.score == control.score &&
              scout.stats.rebound_loss_hits > 0,
          "root scout heuristic cache entries bypassed an exact leaf proof");
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    cg::apply_encoded_turn(state, transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1;
  }
  return state;
}

int exact_leaf_score(const ps::GameState &state, std::string_view action,
                     cg::SearchConfig config) {
  cg::CompleteTurnSearch leaf_search(state, config);
  const ps::Player mover = state.to_move;
  for (const char direction : action) {
    const ps::Move move =
        cg::decode_direction(leaf_search.position_.ball(), direction);
    std::uint8_t slot = 0;
    require(leaf_search.position_.slot_for_move(move, slot),
            "captured scout action must replay in the compact search state");
    leaf_search.position_.make_move(slot);
  }
  require(leaf_search.position_.is_terminal() ||
              leaf_search.position_.to_move() != mover,
          "captured scout action must be a complete turn");
  if (leaf_search.position_.is_terminal()) {
    return leaf_search.terminal_score(1);
  }
  const cg::ReboundOutcome proof =
      leaf_search.analyze_rebound_component(nullptr);
  if (proof == cg::ReboundOutcome::Win) {
    return leaf_search.immediate_win_score(leaf_search.position_.to_move(), 1);
  }
  if (proof == cg::ReboundOutcome::Loss) {
    return leaf_search.immediate_win_score(
        ps::opponent(leaf_search.position_.to_move()), 1);
  }
  return leaf_search.evaluate();
}

void scout_must_not_return_a_stale_captured_action() {
  constexpr std::string_view transcript =
      "6/5/4/1/4/2/4/2/1/6/7/70/1/1/1";
  const ps::GameState state = reconstruct(transcript);
  cg::SearchConfig control_config = proof_config();
  control_config.evaluation_entries = 0;
  const Decision control = fixed_depth_decision(state, control_config);

  cg::SearchConfig scout_config = control_config;
  scout_config.root_tactical_scout = true;
  scout_config.scout_max_work = 512;
  scout_config.scout_max_actions = 64;
  scout_config.scout_max_time_ms = 0;
  const Decision scout = fixed_depth_decision(state, scout_config);
  const int returned_action_score =
      exact_leaf_score(state, scout.action, scout_config);

  std::cout << "stale_capture transcript=" << transcript
            << " control=" << control.action << '/' << control.score
            << " scout=" << scout.action << '/' << scout.score
            << " returned_action_score=" << returned_action_score
            << " scout_endpoints=" << scout.stats.scout_endpoints << '\n';
  require(returned_action_score == scout.score,
          "root scout returned a captured heuristic action whose exact leaf "
          "score does not match the root minimax score");
}

}  // namespace

int main() {
  int failures = 0;
  for (const auto &[name, test] :
       std::vector<std::pair<const char *, void (*)()>>{
           {"scout_must_not_poison_the_exact_leaf_cache",
            scout_must_not_poison_the_exact_leaf_cache},
           {"scout_must_not_return_a_stale_captured_action",
            scout_must_not_return_a_stale_captured_action},
       }) {
    try {
      test();
      std::cout << "PASS " << name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "FAIL " << name << ": " << error.what() << '\n';
    }
  }
  return failures == 0 ? 0 : 1;
}
