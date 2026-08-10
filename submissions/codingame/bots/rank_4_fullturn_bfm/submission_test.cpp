#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "bot.cpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_invalid_argument(Function &&function, const std::string &message) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, message);
}

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

ps::GameState make_clean_state_at(ps::Point point,
                                  ps::Player player = ps::Player::One) {
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

bool same_state(const ps::GameState &left, const ps::GameState &right) {
  return left.config.width == right.config.width &&
         left.config.height == right.config.height &&
         left.config.goal_rule == right.config.goal_rule &&
         left.config.blocked_rule == right.config.blocked_rule &&
         left.ball == right.ball && left.to_move == right.to_move &&
         left.status == right.status && left.path == right.path &&
         left.used_segments == right.used_segments &&
         left.visit_count == right.visit_count;
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

ps::GameState forced_two_move_rebound() {
  const ps::Point root{4, 4};
  const ps::Point rebound{4, 3};
  const ps::Point handoff{4, 2};
  ps::GameState state = make_clean_state_at(root);
  state.visit_count[rebound] = 1;
  block_edges_except(state, root, {rebound});
  block_edges_except(state, rebound, {root, handoff});
  return state;
}

ps::GameState forced_angled_rebound() {
  const ps::Point root{4, 6};
  const ps::Point rebound{5, 5};
  const ps::Point handoff{6, 5};
  ps::GameState state = make_clean_state_at(root);
  state.visit_count[rebound] = 1;
  block_edges_except(state, root, {rebound});
  block_edges_except(state, rebound, {root, handoff});
  return state;
}

ps::GameState forced_cutoff_choice() {
  const ps::Point root{4, 6};
  const ps::Point cutoff{4, 5};
  const ps::Point closed_rebound{4, 4};
  const ps::Point ordinary{5, 6};
  ps::GameState state = make_clean_state_at(root);
  state.visit_count[closed_rebound] = 1;
  block_edges_except(state, root, {cutoff, ordinary});
  block_edges_except(state, cutoff, {root, closed_rebound});
  block_edges_except(state, closed_rebound, {cutoff});
  return state;
}

ps::GameState path_sensitive_cutoff_choice() {
  ps::GameState state = make_clean_state_at({4, 6});
  for (const ps::Point point :
       {ps::Point{4, 5}, ps::Point{5, 6}, ps::Point{4, 4},
        ps::Point{6, 5}}) {
    state.visit_count[point] = 1;
  }
  const std::vector<ps::Segment> open{
      {{4, 6}, {4, 5}}, {{4, 6}, {5, 6}}, {{4, 5}, {4, 4}},
      {{5, 6}, {4, 5}}, {{5, 6}, {6, 5}}, {{4, 4}, {5, 4}},
      {{6, 5}, {5, 4}}, {{4, 5}, {3, 5}}, {{3, 5}, {2, 5}},
      {{2, 5}, {1, 5}},
  };
  const ps::detail::SearchTopology topology(state.config);
  for (std::uint32_t vertex = 0; vertex < topology.vertex_count(); ++vertex) {
    for (const ps::Player player : {ps::Player::One, ps::Player::Two}) {
      const auto &adjacency = topology.adjacency(vertex, player);
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const ps::Segment edge{topology.point(vertex),
                               adjacency.arcs[index].move.to};
        if (std::find(open.begin(), open.end(), edge) == open.end()) {
          state.used_segments.insert(edge);
        }
      }
    }
  }
  return state;
}

ps::GameState opponent_win_choice() {
  const ps::Point root{4, 9};
  const ps::Point dangerous{4, 10};
  const ps::Point safe{5, 9};
  ps::GameState state = make_clean_state_at(root);
  block_edges_except(state, root, {dangerous, safe});
  return state;
}

ps::GameState dense_rebound_state() {
  ps::GameState state = make_clean_state_at({4, 6});
  for (int x = 2; x <= 6; ++x) {
    for (int y = 4; y <= 8; ++y) {
      state.visit_count[{x, y}] = 1;
    }
  }
  return state;
}

ps::GameState closed_dense_rebound_state() {
  ps::GameState state = make_clean_state_at({4, 6});
  const auto inside = [](ps::Point point) {
    return point.x >= 0 && point.x <= 8 &&
           point.y >= 1 && point.y <= 11;
  };
  for (int x = 0; x <= 8; ++x) {
    for (int y = 1; y <= 11; ++y) {
      const ps::Point point{x, y};
      state.visit_count[point] = 1;
      for (const ps::Point destination :
           ps::neighbors(state.config, point, state.to_move)) {
        if (!inside(destination)) {
          state.used_segments.insert(ps::Segment{point, destination});
        }
      }
    }
  }
  return state;
}

ps::GameState symmetric_handoff_choice() {
  ps::GameState state = make_clean_state_at({4, 6});
  block_edges_except(state, state.ball, {{5, 6}, {3, 6}});
  return state;
}

ps::GameState developed_synthetic_state() {
  ps::GameState state = make_clean_state_at({3, 6});
  const std::array<ps::Segment, 12> used{{
      ps::Segment{{1, 2}, {2, 2}}, ps::Segment{{2, 2}, {3, 2}},
      ps::Segment{{3, 2}, {4, 2}}, ps::Segment{{4, 2}, {5, 2}},
      ps::Segment{{5, 2}, {6, 2}}, ps::Segment{{6, 2}, {7, 2}},
      ps::Segment{{1, 2}, {1, 3}}, ps::Segment{{2, 2}, {2, 3}},
      ps::Segment{{3, 2}, {3, 3}}, ps::Segment{{4, 2}, {4, 3}},
      ps::Segment{{5, 2}, {5, 3}}, ps::Segment{{6, 2}, {6, 3}},
  }};
  for (const ps::Segment segment : used) {
    state.used_segments.insert(segment);
    ++state.visit_count[segment.a];
    ++state.visit_count[segment.b];
  }
  return state;
}

struct ReplayedAction {
  ps::GameState state;
  std::string encoded;
};

ReplayedAction replay_complete_action(const ps::GameState &root,
                                      const std::vector<ps::Move> &moves) {
  require(!moves.empty(), "A generated complete turn cannot be empty.");
  ps::GameState replayed = root;
  const ps::Player mover = root.to_move;
  std::string encoded;
  encoded.reserve(moves.size());
  for (const ps::Move move : moves) {
    require(!ps::is_terminal(replayed) && replayed.to_move == mover,
            "A generated action continued after its turn ended.");
    const std::vector<ps::Move> legal = ps::legal_moves(replayed);
    require(std::find(legal.begin(), legal.end(), move) != legal.end(),
            "A generated action contains an illegal primitive move.");
    encoded.push_back(cg::encode_direction(replayed.ball, move.to));
    replayed = ps::apply_move(replayed, move);
  }
  require(ps::is_terminal(replayed) || replayed.to_move != mover,
          "A generated action stopped during a mandatory rebound.");
  return {std::move(replayed), std::move(encoded)};
}

cg::GenerationResult generate(const ps::GameState &state,
                              std::size_t max_actions = 250,
                              std::size_t max_partial_paths =
                                  cg::kMaximumPartialPaths) {
  auto topology = std::make_shared<ps::detail::SearchTopology>(state.config);
  const ps::detail::SearchPosition root(topology, state);
  cg::GeneratorConfig config;
  config.max_actions = max_actions;
  config.max_partial_paths = max_partial_paths;
  cg::FullTurnGenerator generator(topology, root, config);
  return generator.run();
}

std::vector<std::string> encoded_actions(const ps::GameState &state,
                                         const cg::GenerationResult &result) {
  std::vector<std::string> actions;
  actions.reserve(result.actions.size());
  for (const auto &action : result.actions) {
    actions.push_back(replay_complete_action(state, action.moves).encoded);
  }
  return actions;
}

std::vector<std::string> sorted_encoded_actions(
    const ps::GameState &state, const cg::GenerationResult &result) {
  std::vector<std::string> actions = encoded_actions(state, result);
  std::sort(actions.begin(), actions.end());
  return actions;
}

bool same_generator_stats(const cg::GeneratorStats &left,
                          const cg::GeneratorStats &right) {
  return left.explored_partial_paths == right.explored_partial_paths &&
         left.completed_actions == right.completed_actions &&
         left.duplicates == right.duplicates &&
         left.tactical_actions == right.tactical_actions &&
         left.truncations == right.truncations &&
         left.max_deque_size == right.max_deque_size &&
         left.fifo_extractions == right.fifo_extractions &&
         left.lifo_extractions == right.lifo_extractions &&
         left.immediate_wins == right.immediate_wins &&
         left.terminal_losses == right.terminal_losses &&
         left.safe_handoffs == right.safe_handoffs &&
         left.opponent_immediate_wins == right.opponent_immediate_wins &&
         left.forced_cutoffs == right.forced_cutoffs;
}

ps::Point rotate_point(ps::Point point) {
  return {8 - point.x, 12 - point.y};
}

ps::Point reflect_point(ps::Point point) { return {8 - point.x, point.y}; }

template <typename Transform>
ps::GameState transform_state(const ps::GameState &state, Transform transform,
                              bool swap_players) {
  ps::GameState result = state;
  result.ball = transform(state.ball);
  result.to_move = swap_players ? ps::opponent(state.to_move) : state.to_move;
  if (swap_players) {
    if (state.status == ps::Status::WonByOne) {
      result.status = ps::Status::WonByTwo;
    } else if (state.status == ps::Status::WonByTwo) {
      result.status = ps::Status::WonByOne;
    }
  }
  result.path.clear();
  for (const ps::Point point : state.path) {
    result.path.push_back(transform(point));
  }
  result.used_segments.clear();
  for (const ps::Segment &edge : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{transform(edge.a), transform(edge.b)});
  }
  result.visit_count.clear();
  for (const auto &[point, count] : state.visit_count) {
    result.visit_count[transform(point)] = count;
  }
  return result;
}

ps::GameState rotate_and_swap(const ps::GameState &state) {
  return transform_state(state, rotate_point, true);
}

ps::GameState reflect(const ps::GameState &state) {
  return transform_state(state, reflect_point, false);
}

std::string rotate_action(std::string action) {
  for (char &direction : action) {
    direction = static_cast<char>('0' + (direction - '0' + 4) % 8);
  }
  return action;
}

std::string reflect_action(std::string action) {
  constexpr std::string_view reflected = "07654321";
  for (char &direction : action) {
    direction = reflected[static_cast<std::size_t>(direction - '0')];
  }
  return action;
}

void directions_round_trip_exactly() {
  const ps::Point origin{4, 6};
  constexpr std::array<ps::Point, 8> expected{{
      {4, 5}, {5, 5}, {5, 6}, {5, 7},
      {4, 7}, {3, 7}, {3, 6}, {3, 5},
  }};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const char direction = static_cast<char>('0' + index);
    require(cg::decode_direction(origin, direction).to == expected[index],
            "Direction decoding should match the CodinGame compass table.");
    require(cg::encode_direction(origin, expected[index]) == direction,
            "Direction encoding should invert every valid direction.");
  }
  require_invalid_argument(
      [&] { (void)cg::decode_direction(origin, '8'); },
      "Direction eight must never be accepted or emitted.");
}

void encoded_turn_validation_is_atomic() {
  ps::GameState rebound = forced_two_move_rebound();
  const ps::GameState snapshot = rebound;
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(rebound, "0"); },
      "An encoded turn may not stop during a mandatory rebound.");
  require(same_state(rebound, snapshot),
          "A rejected rebound prefix must not partially mutate state.");

  cg::apply_encoded_turn(rebound, "00");
  require(rebound.ball == ps::Point{4, 2} &&
              rebound.to_move == ps::Player::Two,
          "A complete encoded rebound should hand possession over.");

  ps::GameState initial = ps::make_initial_state(codingame_rules());
  const ps::GameState initial_snapshot = initial;
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(initial, "00"); },
      "An encoded turn may not include the opponent's next move.");
  require(same_state(initial, initial_snapshot),
          "A rejected overlong turn must not partially mutate state.");
}

void generator_emits_only_complete_legal_turns_and_exact_results() {
  const ps::GameState state = ps::make_initial_state(codingame_rules());
  const cg::GenerationResult result = generate(state, 32, 2'000);
  require(result.exhaustive && result.actions.size() == 8,
          "The empty-board root should enumerate its eight one-edge turns.");

  auto topology = std::make_shared<ps::detail::SearchTopology>(state.config);
  for (const auto &action : result.actions) {
    const ReplayedAction replayed = replay_complete_action(state, action.moves);
    const ps::detail::SearchPosition expected(topology, replayed.state);
    require(action.result.same_compact_state(expected),
            "A generated endpoint must equal an authoritative replay.");
    ps::GameState decoded = state;
    cg::apply_encoded_turn(decoded, replayed.encoded);
    require(same_state(decoded, replayed.state),
            "The encoded generated action must replay atomically and exactly.");
  }
  require(result.stats.completed_actions ==
                  result.actions.size() + result.stats.duplicates &&
              result.stats.safe_handoffs == result.actions.size() &&
              result.stats.truncations == 0,
          "Exhaustive one-edge generation should account exactly for retained and duplicate actions.");
}

void generator_finishes_mandatory_rebounds_without_prefixes() {
  const ps::GameState state = forced_two_move_rebound();
  const cg::GenerationResult result = generate(state, 16, 256);
  require(result.exhaustive && result.actions.size() == 1,
          "The forced rebound fixture has exactly one complete turn.");
  const ReplayedAction replayed =
      replay_complete_action(state, result.actions.front().moves);
  require(replayed.encoded == "00" && replayed.state.ball == ps::Point{4, 2} &&
              replayed.state.to_move == ps::Player::Two,
          "The generator must retain the complete rebound chain, not its prefix.");
}

void goals_own_goals_and_dead_ends_are_terminal_actions() {
  ps::GameState win = make_clean_state_at({4, 1}, ps::Player::One);
  block_edges_except(win, win.ball, {{4, 0}});
  const cg::GenerationResult winning = generate(win, 8, 128);
  require(winning.actions.size() == 1 && winning.stats.immediate_wins == 1 &&
              winning.stats.terminal_losses == 0,
          "An attacking goal should be retained and classified as an immediate win.");
  const ReplayedAction won =
      replay_complete_action(win, winning.actions.front().moves);
  require(won.encoded == "0" && ps::winner(won.state) == ps::Player::One,
          "The attacking-goal action should terminate for the mover.");

  const ps::GameState symmetric_goal =
      make_clean_state_at({4, 1}, ps::Player::One);
  const cg::GenerationResult centered = generate(symmetric_goal, 1, 128);
  require(replay_complete_action(symmetric_goal,
                                 centered.actions.front().moves)
                  .encoded == "0",
          "A cap-one symmetric goal choice must retain the reflection-fixed center goal.");

  ps::GameState own_goal = make_clean_state_at({4, 11}, ps::Player::One);
  block_edges_except(own_goal, own_goal.ball, {{4, 12}});
  const cg::GenerationResult losing_goal = generate(own_goal, 8, 128);
  require(losing_goal.actions.size() == 1 &&
              losing_goal.stats.immediate_wins == 0 &&
              losing_goal.stats.terminal_losses == 1,
          "An own goal must be classified as a terminal loss, never a win.");
  const ReplayedAction lost =
      replay_complete_action(own_goal, losing_goal.actions.front().moves);
  require(lost.encoded == "4" && ps::winner(lost.state) == ps::Player::Two,
          "The own-goal action should award the opponent.");

  ps::GameState dead_end = make_clean_state_at({4, 6}, ps::Player::One);
  block_edges_except(dead_end, {4, 6}, {{4, 5}});
  block_edges_except(dead_end, {4, 5}, {{4, 6}});
  const cg::GenerationResult blocked = generate(dead_end, 8, 128);
  require(blocked.actions.size() == 1 && blocked.stats.terminal_losses == 1 &&
              ps::winner(replay_complete_action(
                             dead_end, blocked.actions.front().moves)
                             .state) == ps::Player::Two,
          "A landing that blocks the mover must be emitted as a terminal loss.");
}

void tactical_actions_survive_tiny_action_caps() {
  ps::GameState goal_choice = make_clean_state_at({4, 1}, ps::Player::One);
  block_edges_except(goal_choice, goal_choice.ball, {{4, 0}, {5, 2}});
  const cg::GenerationResult goal = generate(goal_choice, 1, 256);
  require(goal.actions.size() == 1 && goal.stats.immediate_wins >= 1 &&
              goal.stats.tactical_actions >= 1 &&
              replay_complete_action(goal_choice, goal.actions.front().moves)
                      .encoded ==
                  "0",
          "A proven goal must displace an ordinary action at cap one.");

  const ps::GameState cutoff_choice = forced_cutoff_choice();
  const cg::GenerationResult cutoff = generate(cutoff_choice, 1, 512);
  require(cutoff.actions.size() == 1 && cutoff.stats.forced_cutoffs >= 1 &&
              cutoff.stats.tactical_actions >= 1 &&
              replay_complete_action(cutoff_choice,
                                     cutoff.actions.front().moves)
                      .encoded ==
                  "0",
          "A forced opponent cutoff must displace an ordinary handoff.");

  const ps::GameState path_sensitive = path_sensitive_cutoff_choice();
  const cg::GenerationResult path_cutoff = generate(path_sensitive, 1, 2'048);
  require(path_cutoff.actions.size() == 1 &&
              path_cutoff.stats.forced_cutoffs == 1 &&
              replay_complete_action(path_sensitive,
                                     path_cutoff.actions.front().moves)
                      .encoded == "2702",
          "Tactical retention must distinguish paths that consume different rebound edges.");

  const ps::GameState danger_choice = opponent_win_choice();
  const cg::GenerationResult safe = generate(danger_choice, 1, 512);
  require(safe.actions.size() == 1 &&
              safe.stats.safe_handoffs >= 1 &&
              replay_complete_action(danger_choice, safe.actions.front().moves)
                      .encoded ==
                  "2",
          "A safe handoff must displace a move allowing an immediate reply goal.");
  const cg::GenerationResult classified = generate(danger_choice, 8, 512);
  require(classified.stats.opponent_immediate_wins >= 1 &&
              classified.stats.safe_handoffs >= 1 &&
              classified.stats.tactical_actions >= 1,
          "Uncapped generation must classify both safe and losing handoffs.");
}

void generator_cap_deduplication_and_mixed_deque_schedule_are_exact() {
  const ps::GameState state = dense_rebound_state();
  const cg::GenerationResult result =
      generate(state, 250, cg::kMaximumPartialPaths);
  require(result.actions.size() == 250 && !result.exhaustive &&
              result.stats.truncations > 0,
          "A branching rebound state should hit the exact 250-action cap.");

  const std::vector<std::string> encodings = encoded_actions(state, result);
  const std::set<std::string> unique(encodings.begin(), encodings.end());
  require(unique.size() == encodings.size(),
          "The capped action set must not expose duplicates.");
  require(result.stats.explored_partial_paths >= 10 &&
              result.stats.fifo_extractions ==
                  result.stats.explored_partial_paths / 10 &&
              result.stats.lifo_extractions +
                      result.stats.fifo_extractions ==
                  result.stats.explored_partial_paths &&
              result.stats.lifo_extractions >
                  result.stats.fifo_extractions,
          "Generation must pop FIFO every tenth extraction and LIFO otherwise.");
  require(result.stats.max_deque_size > 1,
          "The mixed schedule fixture should exercise a branching deque.");
}

void generator_is_deterministic_for_a_canonical_position() {
  const ps::GameState state = dense_rebound_state();
  const cg::GenerationResult first = generate(state, 64, 20'000);
  const cg::GenerationResult second = generate(state, 64, 20'000);
  require(encoded_actions(state, first) == encoded_actions(state, second) &&
              first.exhaustive == second.exhaustive &&
              same_generator_stats(first.stats, second.stats),
          "A canonical position must reproduce action order and generator stats.");
  require(first.actions.size() == second.actions.size(),
          "Deterministic generation changed the action count.");
  for (std::size_t index = 0; index < first.actions.size(); ++index) {
    require(first.actions[index].tactical == second.actions[index].tactical,
            "Deterministic generation changed a tactical classification.");
  }
}

void generator_respects_reflection_and_player_rotation() {
  const ps::GameState initial = ps::make_initial_state(codingame_rules());
  const std::string fixed = replay_complete_action(
      initial, generate(initial, 1, 128).actions.front().moves).encoded;
  require(reflect_action(fixed) == fixed,
          "A cap-one symmetric root must retain a reflection-fixed action.");

  ps::GameState branching_goal =
      make_clean_state_at({4, 2}, ps::Player::One);
  branching_goal.visit_count[{3, 1}] = 1;
  branching_goal.visit_count[{5, 1}] = 1;
  branching_goal.used_segments.insert({{0, 5}, {1, 5}});
  block_edges_except(branching_goal, {4, 2}, {{3, 1}, {5, 1}});
  block_edges_except(branching_goal, {3, 1}, {{4, 2}, {3, 0}});
  block_edges_except(branching_goal, {5, 1}, {{4, 2}, {5, 0}});
  const std::string goal_path = replay_complete_action(
      branching_goal,
      generate(branching_goal, 1, 128).actions.front().moves).encoded;
  const ps::GameState reflected_goal = reflect(branching_goal);
  require(reflect_action(goal_path) ==
              replay_complete_action(
                  reflected_goal,
                  generate(reflected_goal, 1, 128).actions.front().moves)
                  .encoded,
          "Canonical rebound-goal BFS must reflect its capped path choice.");

  const ps::GameState angled = forced_angled_rebound();
  const ps::GameState reflected_state = reflect(angled);
  std::vector<std::string> expected_reflection =
      sorted_encoded_actions(angled, generate(angled, 16, 256));
  for (std::string &action : expected_reflection) {
    action = reflect_action(std::move(action));
  }
  std::sort(expected_reflection.begin(), expected_reflection.end());
  require(expected_reflection == sorted_encoded_actions(
                                     reflected_state,
                                     generate(reflected_state, 16, 256)),
          "Horizontal reflection should reflect every generated complete turn.");

  const ps::GameState vertical = forced_two_move_rebound();
  const ps::GameState rotated_state = rotate_and_swap(vertical);
  std::vector<std::string> expected_rotation =
      sorted_encoded_actions(vertical, generate(vertical, 16, 256));
  for (std::string &action : expected_rotation) {
    action = rotate_action(std::move(action));
  }
  std::sort(expected_rotation.begin(), expected_rotation.end());
  require(expected_rotation == sorted_encoded_actions(
                                   rotated_state,
                                   generate(rotated_state, 16, 256)),
          "A 180-degree player rotation should rotate every complete turn.");

  ps::GameState dense = dense_rebound_state();
  dense.visit_count[{1, 3}] = 1;
  std::vector<std::string> expected_dense_reflection =
      sorted_encoded_actions(dense, generate(dense, 64));
  for (std::string &action : expected_dense_reflection) {
    action = reflect_action(std::move(action));
  }
  std::sort(expected_dense_reflection.begin(),
            expected_dense_reflection.end());
  const std::vector<std::string> actual_dense_reflection =
      sorted_encoded_actions(reflect(dense), generate(reflect(dense), 64));
  if (expected_dense_reflection != actual_dense_reflection) {
    for (const std::string &action : expected_dense_reflection) {
      if (!std::binary_search(actual_dense_reflection.begin(),
                              actual_dense_reflection.end(), action)) {
        std::cerr << "missing reflected action=" << action << '\n';
        break;
      }
    }
    for (const std::string &action : actual_dense_reflection) {
      if (!std::binary_search(expected_dense_reflection.begin(),
                              expected_dense_reflection.end(), action)) {
        std::cerr << "extra reflected action=" << action << '\n';
        break;
      }
    }
  }
  require(expected_dense_reflection == actual_dense_reflection,
          "A capped dense sample must remain reflection-equivariant.");

  std::vector<std::string> expected_dense_rotation =
      sorted_encoded_actions(dense, generate(dense, 64));
  for (std::string &action : expected_dense_rotation) {
    action = rotate_action(std::move(action));
  }
  std::sort(expected_dense_rotation.begin(), expected_dense_rotation.end());
  require(expected_dense_rotation ==
              sorted_encoded_actions(rotate_and_swap(dense),
                                     generate(rotate_and_swap(dense), 64)),
          "A capped dense sample must remain player-rotation-equivariant.");
}

void minimax_backup_and_uct_scores_use_player_one_coordinates() {
  const std::vector<int> values{-7, 11, 3};
  require(cg::minimax_backup(ps::Player::One, values) == 11,
          "Player One must maximize absolute evaluation values.");
  require(cg::minimax_backup(ps::Player::Two, values) == -7,
          "Player Two must minimize absolute evaluation values.");
  require_invalid_argument(
      [&] {
        const std::vector<int> empty;
        (void)cg::minimax_backup(ps::Player::One, empty);
      },
      "Minimax backup needs at least one child.");

  const double p1_high =
      cg::uct_selection_score(100, ps::Player::One, 64, 4);
  const double p1_low =
      cg::uct_selection_score(-100, ps::Player::One, 64, 4);
  const double p2_high =
      cg::uct_selection_score(100, ps::Player::Two, 64, 4);
  const double p2_low =
      cg::uct_selection_score(-100, ps::Player::Two, 64, 4);
  require(p1_high > p1_low && p2_low > p2_high,
          "UCT exploitation must maximize for Player One and minimize for Player Two.");
  require(cg::uct_selection_score(7, ps::Player::One, 64, 4) ==
              cg::uct_selection_score(7, ps::Player::One, 64, 4),
          "Identical UCT children must produce an exact deterministic tie.");
  require(cg::uct_selection_score(0, ps::Player::One, 64, 1) >
              cg::uct_selection_score(0, ps::Player::One, 64, 16),
          "Less-visited children must receive a larger UCT exploration term.");
  require(cg::normalized_search_value(cg::kMateScore - 1, true) ==
                  10'000.0 &&
              cg::normalized_search_value(cg::kMateScore - 1, false) ==
                  1.0,
          "Only solved mate scores may receive terminal UCT dominance.");
}

void final_choice_breaks_a_real_equal_value_tie_lexicographically() {
  const ps::GameState state = symmetric_handoff_choice();
  const cg::GenerationResult generated = generate(state, 8, 512);
  require(generated.exhaustive && generated.actions.size() == 2,
          "The tie fixture must have exactly two complete handoffs.");

  std::vector<std::pair<std::string, int>> evaluated;
  for (const auto &action : generated.actions) {
    const ReplayedAction replayed = replay_complete_action(state, action.moves);
    cg::SearchConfig child_config;
    child_config.max_work = 1;
    child_config.max_tree_nodes = 2;
    child_config.max_actions = 2;
    child_config.max_partial_paths = 32;
    child_config.replay_value_blend_percent = 0;
    child_config.teacher_residual_weight_percent = 0;
    cg::FullTurnBfmSearch child(replayed.state, child_config);
    evaluated.emplace_back(replayed.encoded,
                           child.evaluation_snapshot().anchor_score);
  }
  std::sort(evaluated.begin(), evaluated.end());
  require(evaluated[0].first == "2" && evaluated[1].first == "6" &&
              evaluated[0].second == evaluated[1].second,
          "The tie fixture must present equal-valued east/west actions.");

  cg::SearchConfig config;
  config.max_work = generated.stats.explored_partial_paths +
                    generated.actions.size();
  config.max_tree_nodes = 8;
  config.max_actions = 8;
  config.max_partial_paths = 512;
  config.replay_value_blend_percent = 0;
  config.teacher_residual_weight_percent = 0;
  cg::FullTurnBfmSearch search(state, config);
  const std::vector<ps::Move> chosen = search.run();
  require(replay_complete_action(state, chosen).encoded == "2",
          "An actual equal-value final-choice tie must choose the lexicographically smallest action.");
}

void evaluator_is_invariant_under_player_rotation() {
  const ps::GameState state = developed_synthetic_state();
  const ps::GameState rotated = rotate_and_swap(state);
  cg::SearchConfig config;
  config.max_work = 64;
  config.max_tree_nodes = 128;
  config.max_actions = 32;
  config.max_partial_paths = 2'000;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  cg::FullTurnBfmSearch original_search(state, config);
  cg::FullTurnBfmSearch rotated_search(rotated, config);
  const cg::EvaluationSnapshot original =
      original_search.evaluation_snapshot();
  const cg::EvaluationSnapshot transformed =
      rotated_search.evaluation_snapshot();
  require(original.anchor_score == -transformed.anchor_score &&
              original.mover_sign == -transformed.mover_sign,
          "Player rotation should negate the absolute evaluator coordinates.");
  for (std::size_t index = 0; index < original.features.size(); ++index) {
    require(std::abs(original.features[index] - transformed.features[index]) <
                1e-5F,
            "Player rotation changed a mover-relative evaluator feature.");
  }
}

void fixed_work_bfm_is_deterministic_and_evaluates_each_child() {
  const ps::GameState state = developed_synthetic_state();
  cg::SearchConfig config;
  config.max_work = 128;
  config.max_tree_nodes = 256;
  config.max_actions = 64;
  config.max_partial_paths = 4'000;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;

  cg::FullTurnBfmSearch first(state, config);
  cg::FullTurnBfmSearch second(state, config);
  const std::vector<ps::Move> first_action = first.run();
  const std::vector<ps::Move> second_action = second.run();
  const cg::SearchStats &first_stats = first.stats();
  const cg::SearchStats &second_stats = second.stats();
  require(first_action == second_action &&
              first_stats.work == second_stats.work &&
              first_stats.tree_nodes == second_stats.tree_nodes &&
              first_stats.expansions == second_stats.expansions &&
              first_stats.child_evaluations ==
                  second_stats.child_evaluations &&
              first_stats.root_score == second_stats.root_score &&
              first_stats.budget_exhausted == second_stats.budget_exhausted,
          "A fixed-work BFM search must reproduce its action and core stats.");
  require(first_stats.work <= config.max_work &&
              first_stats.tree_nodes <= config.max_tree_nodes &&
              first_stats.child_evaluations + 1 ==
                  first_stats.tree_nodes &&
              first_stats.expansions > 0 &&
              first_stats.expansions <= first_stats.child_evaluations,
          "Every newly expanded complete-turn child must be evaluated once within bounds.");
  (void)replay_complete_action(state, first_action);
}

void interrupted_bfm_still_returns_a_complete_turn() {
  const ps::GameState state = forced_two_move_rebound();
  cg::SearchConfig config;
  config.max_work = 1;
  config.max_tree_nodes = 9;
  config.max_actions = 8;
  config.max_partial_paths = 64;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  cg::FullTurnBfmSearch search(state, config);
  const std::vector<ps::Move> action = search.run();
  require(search.stats().budget_exhausted &&
              replay_complete_action(state, action).encoded == "00",
          "A one-work search must return its legal, rebound-complete fallback.");
}

void bfm_reports_and_obeys_a_reached_tree_node_cap() {
  const ps::GameState state = ps::make_initial_state(codingame_rules());
  cg::SearchConfig config;
  config.max_work = 64;
  config.max_tree_nodes = 2;
  config.max_actions = 32;
  config.max_partial_paths = 2'000;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  cg::FullTurnBfmSearch search(state, config);
  const std::vector<ps::Move> action = search.run();
  require(search.stats().node_cap_reached &&
              search.stats().budget_exhausted &&
              search.stats().tree_nodes == 1 &&
              search.stats().child_evaluations == 0 &&
              search.stats().expansions == 0,
          "A complete child batch that cannot fit must not be partially inserted.");
  (void)replay_complete_action(state, action);
}

void bfm_keeps_a_proven_cutoff_when_the_full_batch_cannot_fit() {
  const ps::GameState state = forced_cutoff_choice();
  cg::SearchConfig config;
  config.max_work = 5'000;
  config.max_tree_nodes = 2;
  config.max_actions = 32;
  config.max_partial_paths = 512;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  cg::FullTurnBfmSearch search(state, config);
  const std::vector<ps::Move> action = search.run();
  require(search.stats().tree_nodes == 2 &&
              replay_complete_action(state, action).encoded == "0",
          "An atomic batch limit must not discard a proved winning cutoff.");
}

void generator_freezes_50000_and_enforces_the_partial_path_cap() {
  const cg::GeneratorConfig defaults;
  require(cg::kMaximumPartialPaths == 50'000 &&
              defaults.max_partial_paths == cg::kMaximumPartialPaths,
          "The production generator configuration must freeze the 50,000-partial ceiling.");
  const ps::GameState state = closed_dense_rebound_state();
  constexpr std::uint64_t kFocusedLimit = 1'000;
  const cg::GenerationResult result =
      generate(state, cg::kMaximumActions, kFocusedLimit);
  require(result.stats.explored_partial_paths ==
                  kFocusedLimit &&
              result.stats.truncations == 1 && !result.exhaustive &&
              result.actions.size() < cg::kMaximumActions,
          "The dense closed rebound fixture must stop at its configured partial-path cap before the action cap (partials=" +
              std::to_string(result.stats.explored_partial_paths) +
              ", actions=" + std::to_string(result.actions.size()) + ").");
  for (const auto &action : result.actions) {
    (void)replay_complete_action(state, action.moves);
  }
}

bool same_game_state(const ps::GameState &left,
                     const ps::GameState &right) {
  return left.config.width == right.config.width &&
         left.config.height == right.config.height &&
         left.config.goal_rule == right.config.goal_rule &&
         left.config.blocked_rule == right.config.blocked_rule &&
         left.ball == right.ball && left.to_move == right.to_move &&
         left.status == right.status && left.path == right.path &&
         left.used_segments == right.used_segments &&
         left.visit_count == right.visit_count;
}

class TestContinuationCache {
 public:
  void store(const ps::GameState &root, std::vector<ps::Move> action) {
    clear();
    expected_ = root;
    action_ = std::move(action);
  }

  std::optional<ps::Move> take(const ps::GameState &actual) {
    if (!expected_.has_value() || next_ >= action_.size() ||
        !same_game_state(actual, *expected_)) {
      clear();
      return std::nullopt;
    }
    const ps::Move move = action_[next_++];
    const ps::GameState next = ps::apply_move(actual, move);
    if (next_ < action_.size()) {
      expected_ = next;
    } else {
      clear();
    }
    return move;
  }

  std::size_t remaining() const { return action_.size() - next_; }

 private:
  void clear() {
    action_.clear();
    expected_.reset();
    next_ = 0;
  }

  std::vector<ps::Move> action_;
  std::optional<ps::GameState> expected_;
  std::size_t next_{};
};

void continuation_cache_requires_an_exact_expected_state() {
  const ps::GameState root = forced_two_move_rebound();
  const cg::GenerationResult generated = generate(root, 8, 128);
  require(generated.actions.size() == 1 &&
              generated.actions.front().moves.size() == 2,
          "The continuation-cache fixture needs a two-edge turn.");
  const std::vector<ps::Move> action = generated.actions.front().moves;

  TestContinuationCache cache;
  cache.store(root, action);
  const std::optional<ps::Move> first = cache.take(root);
  require(first.has_value() && *first == action[0],
          "The exact root should release the first cached edge.");
  const ps::GameState expected = ps::apply_move(root, *first);

  ps::GameState mismatch = expected;
  ++mismatch.visit_count[{1, 1}];
  require(!cache.take(mismatch).has_value(),
          "Any expected-state difference must reject the continuation.");
  require(!cache.take(expected).has_value(),
          "A mismatch must invalidate the remainder, not merely skip one call.");

  cache.store(root, action);
  const std::optional<ps::Move> restarted = cache.take(root);
  require(restarted.has_value(), "Restoring a valid plan should restart caching.");
  const ps::GameState after_first = ps::apply_move(root, *restarted);
  const std::optional<ps::Move> second = cache.take(after_first);
  require(second.has_value() && *second == action[1],
          "The exact predicted rebound state should release the next edge.");
  const ps::GameState finished = ps::apply_move(after_first, *second);
  require(!cache.take(finished).has_value(),
          "A consumed complete turn must leave no stale continuation.");
}

void invalid_generator_limits_are_rejected() {
  const ps::GameState state = forced_two_move_rebound();
  auto topology = std::make_shared<ps::detail::SearchTopology>(state.config);
  const ps::detail::SearchPosition root(topology, state);
  cg::GeneratorConfig config;
  config.max_actions = 0;
  require_invalid_argument(
      [&] {
        cg::FullTurnGenerator generator(topology, root, config);
        (void)generator.run();
      },
      "A zero complete-action cap must be rejected.");
  config.max_actions = 1;
  config.max_partial_paths = 0;
  require_invalid_argument(
      [&] {
        cg::FullTurnGenerator generator(topology, root, config);
        (void)generator.run();
      },
      "A zero partial-path cap must be rejected.");
  config.max_partial_paths = cg::kMaximumPartialPaths + 1;
  require_invalid_argument(
      [&] {
        cg::FullTurnGenerator generator(topology, root, config);
        (void)generator.run();
      },
      "A partial-path limit above the production hard cap must be rejected.");
  config.max_partial_paths = 1;
  config.max_work = cg::kMaximumWork + 1;
  require_invalid_argument(
      [&] {
        cg::FullTurnGenerator generator(topology, root, config);
        (void)generator.run();
      },
      "A work limit above the production hard cap must be rejected.");
}

}  // namespace

int main() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"directions_round_trip_exactly", directions_round_trip_exactly},
      {"encoded_turn_validation_is_atomic",
       encoded_turn_validation_is_atomic},
      {"generator_emits_only_complete_legal_turns_and_exact_results",
       generator_emits_only_complete_legal_turns_and_exact_results},
      {"generator_finishes_mandatory_rebounds_without_prefixes",
       generator_finishes_mandatory_rebounds_without_prefixes},
      {"goals_own_goals_and_dead_ends_are_terminal_actions",
       goals_own_goals_and_dead_ends_are_terminal_actions},
      {"tactical_actions_survive_tiny_action_caps",
       tactical_actions_survive_tiny_action_caps},
      {"generator_cap_deduplication_and_mixed_deque_schedule_are_exact",
       generator_cap_deduplication_and_mixed_deque_schedule_are_exact},
      {"generator_is_deterministic_for_a_canonical_position",
       generator_is_deterministic_for_a_canonical_position},
      {"generator_respects_reflection_and_player_rotation",
       generator_respects_reflection_and_player_rotation},
      {"minimax_backup_and_uct_scores_use_player_one_coordinates",
       minimax_backup_and_uct_scores_use_player_one_coordinates},
      {"final_choice_breaks_a_real_equal_value_tie_lexicographically",
       final_choice_breaks_a_real_equal_value_tie_lexicographically},
      {"evaluator_is_invariant_under_player_rotation",
       evaluator_is_invariant_under_player_rotation},
      {"fixed_work_bfm_is_deterministic_and_evaluates_each_child",
       fixed_work_bfm_is_deterministic_and_evaluates_each_child},
      {"interrupted_bfm_still_returns_a_complete_turn",
       interrupted_bfm_still_returns_a_complete_turn},
      {"bfm_reports_and_obeys_a_reached_tree_node_cap",
       bfm_reports_and_obeys_a_reached_tree_node_cap},
      {"bfm_keeps_a_proven_cutoff_when_the_full_batch_cannot_fit",
       bfm_keeps_a_proven_cutoff_when_the_full_batch_cannot_fit},
      {"generator_freezes_50000_and_enforces_the_partial_path_cap",
       generator_freezes_50000_and_enforces_the_partial_path_cap},
      {"continuation_cache_requires_an_exact_expected_state",
       continuation_cache_requires_an_exact_expected_state},
      {"invalid_generator_limits_are_rejected",
       invalid_generator_limits_are_rejected},
  };

  int failures = 0;
  for (const TestCase &test : tests) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": " << error.what() << '\n';
    }
  }
  std::cout << '\n' << (tests.size() - static_cast<std::size_t>(failures))
            << '/' << tests.size() << " submission tests passed.\n";
  return failures == 0 ? 0 : 1;
}
