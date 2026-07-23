#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "mcts_internal.hpp"
#include "papersoccer/bot.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

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

bool contains_move(const std::vector<ps::Move> &moves, ps::Point point) {
  return std::any_of(moves.begin(), moves.end(),
                     [&](const ps::Move &move) { return move.to == point; });
}

bool contains_point(std::initializer_list<ps::Point> points, ps::Point point) {
  return std::find(points.begin(), points.end(), point) != points.end();
}

bool same_stats(const ps::SearchStats &lhs, const ps::SearchStats &rhs) {
  return lhs.iterations == rhs.iterations && lhs.nodes == rhs.nodes &&
         lhs.simulated_plies == rhs.simulated_plies &&
         lhs.root_value == rhs.root_value &&
         lhs.total_root_visits == rhs.total_root_visits &&
         lhs.reused_visits == rhs.reused_visits &&
         lhs.max_depth == rhs.max_depth &&
         lhs.proven_nodes == rhs.proven_nodes &&
         lhs.proven_winner == rhs.proven_winner &&
         lhs.rebuild_count == rhs.rebuild_count &&
         lhs.expansion_saturated == rhs.expansion_saturated &&
         lhs.tactical_probes == rhs.tactical_probes &&
         lhs.tactical_nodes == rhs.tactical_nodes &&
         lhs.tactical_solved_positions == rhs.tactical_solved_positions &&
         lhs.tactical_depth_cutoffs == rhs.tactical_depth_cutoffs &&
         lhs.tactical_node_cutoffs == rhs.tactical_node_cutoffs &&
         lhs.max_tactical_depth == rhs.max_tactical_depth;
}

bool has_no_tactical_probe_work(const ps::SearchStats &stats) {
  return stats.tactical_probes == 0 && stats.tactical_nodes == 0 &&
         stats.tactical_solved_positions == 0 &&
         stats.tactical_depth_cutoffs == 0 &&
         stats.tactical_node_cutoffs == 0 &&
         stats.max_tactical_depth == 0;
}

ps::GameState make_clean_state_at(ps::Point point,
                                  ps::Player to_move = ps::Player::One,
                                  const ps::RulesConfig &config = {}) {
  ps::GameState state = ps::make_initial_state(config);
  state.ball = point;
  state.to_move = to_move;
  state.status = ps::Status::InProgress;
  state.path = {point};
  state.used_segments.clear();
  state.visit_count.clear();
  state.visit_count[point] = 1;
  return state;
}

void block_edges_except(ps::GameState &state, ps::Point from,
                        std::initializer_list<ps::Point> allowed) {
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) {
        continue;
      }

      const ps::Point destination{from.x + dx, from.y + dy};
      if (!contains_point(allowed, destination)) {
        state.used_segments.insert(ps::Segment{from, destination});
      }
    }
  }
}

ps::GameState make_player_two_choice(bool second_branch_wins_for_player_two) {
  const ps::Point root{4, 5};
  const ps::Point player_one_branch{3, 5};
  const ps::Point player_two_branch{5, 5};
  const ps::Point player_one_destination{2, 5};
  const ps::Point player_two_destination{6, 5};

  ps::GameState state = make_clean_state_at(root, ps::Player::Two);
  if (second_branch_wins_for_player_two) {
    state.visit_count[player_two_destination] = 1;
  }

  block_edges_except(state, root, {player_one_branch, player_two_branch});
  block_edges_except(state, player_one_branch,
                     {root, player_one_destination});
  block_edges_except(state, player_two_branch,
                     {root, player_two_destination});
  block_edges_except(state, player_one_destination, {player_one_branch});
  block_edges_except(state, player_two_destination, {player_two_branch});
  return state;
}

ps::GameState make_rebound_choice() {
  const ps::Point root{4, 6};
  const ps::Point rebound{4, 5};
  const ps::Point winning_destination{3, 5};
  const ps::Point losing_destination{5, 5};

  ps::GameState state = make_clean_state_at(root, ps::Player::One);
  state.visit_count[rebound] = 1;
  state.visit_count[losing_destination] = 1;

  block_edges_except(state, root, {rebound});
  block_edges_except(state, rebound,
                     {root, winning_destination, losing_destination});
  block_edges_except(state, winning_destination, {rebound});
  block_edges_except(state, losing_destination, {rebound});
  return state;
}

ps::GameState make_goal_blocking_choice(ps::Player player) {
  const bool player_one = player == ps::Player::One;
  const int branch_y = player_one ? 10 : 2;
  const ps::Point root{4, branch_y};
  const ps::Point safe{3, branch_y};
  const ps::Point unsafe{4, player_one ? 11 : 1};
  const ps::Point opponent_goal{4, player_one ? 12 : 0};
  const ps::Point unsafe_escape{5, branch_y};
  const ps::Point unsafe_finish{6, branch_y};
  const ps::Point safe_step{2, branch_y};
  const ps::Point safe_finish{1, branch_y};

  ps::GameState state = make_clean_state_at(root, player);
  block_edges_except(state, root, {safe, unsafe});
  block_edges_except(state, unsafe,
                     {root, opponent_goal, unsafe_escape});
  block_edges_except(state, unsafe_escape, {unsafe, unsafe_finish});
  block_edges_except(state, unsafe_finish, {unsafe_escape});
  block_edges_except(state, safe, {root, safe_step});
  block_edges_except(state, safe_step, {safe, safe_finish});
  block_edges_except(state, safe_finish, {safe_step});
  return state;
}

ps::GameState make_self_trap_choice() {
  const ps::Point root{4, 5};
  const ps::Point trap{3, 5};
  const ps::Point safe{5, 5};
  const ps::Point safe_step{6, 5};
  const ps::Point safe_finish{7, 5};

  ps::GameState state = make_clean_state_at(root, ps::Player::One);
  state.visit_count[trap] = 1;
  block_edges_except(state, root, {trap, safe});
  block_edges_except(state, trap, {root});
  block_edges_except(state, safe, {root, safe_step});
  block_edges_except(state, safe_step, {safe, safe_finish});
  block_edges_except(state, safe_finish, {safe_step});
  return state;
}

ps::GameState make_forced_rebound_goal(ps::Player player) {
  const bool player_one = player == ps::Player::One;
  const ps::Point root{4, player_one ? 3 : 9};
  const ps::Point first_rebound{4, player_one ? 2 : 10};
  const ps::Point second_rebound{4, player_one ? 1 : 11};
  const ps::Point goal{4, player_one ? 0 : 12};

  ps::GameState state = make_clean_state_at(root, player);
  state.visit_count[first_rebound] = 1;
  state.visit_count[second_rebound] = 1;
  block_edges_except(state, root, {first_rebound});
  block_edges_except(state, first_rebound, {root, second_rebound});
  block_edges_except(state, second_rebound, {first_rebound, goal});
  return state;
}

ps::GameState make_rebound_goal_choice() {
  const ps::Point root{4, 3};
  const ps::Point winning_rebound{4, 2};
  const ps::Point second_rebound{4, 1};
  const ps::Point goal{4, 0};
  const ps::Point quiet_alternative{5, 3};

  ps::GameState state = make_clean_state_at(root, ps::Player::One);
  state.visit_count[winning_rebound] = 1;
  state.visit_count[second_rebound] = 1;
  block_edges_except(state, root, {winning_rebound, quiet_alternative});
  block_edges_except(state, winning_rebound, {root, second_rebound});
  block_edges_except(state, second_rebound, {winning_rebound, goal});
  return state;
}

ps::GameState make_delayed_trap(ps::Player player) {
  const bool player_one = player == ps::Player::One;
  const ps::Point root{4, player_one ? 5 : 7};
  const ps::Point rebound{4, player_one ? 4 : 8};
  const ps::Point trapped_reply{4, player_one ? 3 : 9};

  ps::GameState state = make_clean_state_at(root, player);
  state.visit_count[rebound] = 1;
  block_edges_except(state, root, {rebound});
  block_edges_except(state, rebound, {root, trapped_reply});
  block_edges_except(state, trapped_reply, {rebound});
  return state;
}

ps::GameState make_every_reply_loses(ps::Player losing_player) {
  const bool player_two_loses = losing_player == ps::Player::Two;
  const ps::Point root{4, player_two_loses ? 3 : 9};
  const ps::Point left{3, player_two_loses ? 2 : 10};
  const ps::Point right{5, player_two_loses ? 2 : 10};
  const ps::Point rebound{4, player_two_loses ? 1 : 11};
  const ps::Point goal{4, player_two_loses ? 0 : 12};

  ps::GameState state = make_clean_state_at(root, losing_player);
  state.visit_count[rebound] = 1;
  block_edges_except(state, root, {left, right});
  block_edges_except(state, left, {root, rebound});
  block_edges_except(state, right, {root, rebound});
  block_edges_except(state, rebound, {left, right, goal});
  return state;
}

ps::GameState make_rebound_attack_with_one_escape() {
  const ps::Point root{4, 5};
  const ps::Point attacking_rebound{4, 4};
  const ps::Point defense{4, 3};
  const ps::Point losing_left{3, 2};
  const ps::Point losing_center{4, 2};
  const ps::Point safe_rebound{5, 3};
  const ps::Point safe_second_rebound{6, 3};
  const ps::Point safe_exit{7, 4};
  const ps::Point goal_rebound{4, 1};
  const ps::Point goal{4, 0};

  ps::GameState state = make_clean_state_at(root, ps::Player::One);
  state.visit_count[attacking_rebound] = 1;
  state.visit_count[safe_rebound] = 1;
  state.visit_count[safe_second_rebound] = 1;
  state.visit_count[goal_rebound] = 1;
  block_edges_except(state, root, {attacking_rebound});
  block_edges_except(state, attacking_rebound, {root, defense});
  block_edges_except(state, defense,
                     {attacking_rebound, losing_left, losing_center,
                      safe_rebound});
  block_edges_except(state, losing_left, {defense, goal_rebound});
  block_edges_except(state, losing_center, {defense, goal_rebound});
  block_edges_except(state, goal_rebound,
                     {losing_left, losing_center, goal});
  block_edges_except(state, safe_rebound,
                     {defense, safe_second_rebound});
  block_edges_except(state, safe_second_rebound,
                     {safe_rebound, safe_exit});
  return state;
}

ps::GameState state_after_path(std::initializer_list<ps::Point> path) {
  ps::GameState state = ps::make_initial_state();
  for (const ps::Point point : path) {
    state = ps::apply_move(state, ps::Move{point});
  }
  return state;
}

ps::detail::TacticalProbeResult probe_state(
    const ps::GameState &state, std::uint32_t max_depth = 8,
    std::uint32_t max_nodes = 256) {
  auto topology =
      std::make_shared<ps::detail::SearchTopology>(state.config);
  ps::detail::SearchPosition position(topology, state);
  const ps::detail::SearchPosition original(position);
  const std::size_t original_depth = position.undo_depth();
  const ps::detail::TacticalProbeResult result =
      ps::detail::run_tactical_probe(position, max_depth, max_nodes);
  require(position.undo_depth() == original_depth &&
              position.same_compact_state(original),
          "A tactical probe must restore the exact compact position.");
  return result;
}

ps::MctsConfig config_with(std::uint64_t seed, std::uint32_t iterations,
                           double exploration) {
  ps::MctsConfig config;
  config.seed = seed;
  config.iterations = iterations;
  config.exploration = exploration;
  return config;
}

ps::MctsConfig reference_config(std::uint64_t seed,
                                std::uint32_t iterations,
                                double exploration = 1.4142135623730951) {
  ps::MctsConfig config = config_with(seed, iterations, exploration);
  config.rollout_policy = ps::MctsRolloutPolicy::Uniform;
  config.reuse_tree = false;
  config.max_nodes = 65536;
  return config;
}

ps::MctsConfig tactical_config(std::uint64_t seed,
                               std::uint32_t iterations,
                               bool reuse_tree = false,
                               std::size_t max_nodes = 65536) {
  ps::MctsConfig config = config_with(seed, iterations, 1.0);
  config.rollout_policy = ps::MctsRolloutPolicy::Tactical;
  config.reuse_tree = reuse_tree;
  config.max_nodes = max_nodes;
  return config;
}

ps::MctsConfig quiescence_config(std::uint64_t seed,
                                 std::uint32_t iterations,
                                 bool reuse_tree = false,
                                 std::size_t max_nodes = 65536) {
  ps::MctsConfig config =
      tactical_config(seed, iterations, reuse_tree, max_nodes);
  config.leaf_policy = ps::MctsLeafPolicy::TacticalQuiescence;
  return config;
}

void require_compact_position_matches(const ps::GameState &state) {
  auto topology =
      std::make_shared<ps::detail::SearchTopology>(state.config);
  ps::detail::SearchPosition compact(topology, state);
  ps::detail::SearchPosition root_copy(topology, state);

  require(compact.to_move() == state.to_move &&
              compact.status() == state.status &&
              compact.winner() == ps::winner(state),
          "The compact position should preserve the public state metadata.");

  std::array<std::uint8_t, ps::detail::kMaximumMoves> slots{};
  const std::uint8_t slot_count = compact.legal_slots(slots);
  const std::vector<ps::Move> public_moves = ps::legal_moves(state);
  require(slot_count == public_moves.size(),
          "Compact and public move generation should return the same count.");

  for (std::size_t index = 0; index < public_moves.size(); ++index) {
    require(compact.move_for_slot(slots[index]) == public_moves[index],
            "Compact move generation should preserve public legal-move order.");

    const std::size_t depth = compact.undo_depth();
    compact.make_move(slots[index]);
    const ps::GameState public_next =
        ps::apply_move(state, public_moves[index]);
    const ps::detail::SearchPosition compact_public_next(topology, public_next);
    require(compact.same_compact_state(compact_public_next),
            "Compact make_move should match the authoritative public successor.");
    compact.unmake_to(depth);
    require(compact.same_compact_state(root_copy),
            "Compact unmake should exactly restore the root position.");
  }
}

struct MctsGameTrace {
  std::vector<ps::Point> path{};
  std::vector<ps::SearchStats> searches{};
};

MctsGameTrace play_reusing_tactical_game(
    ps::MctsLeafPolicy leaf_policy = ps::MctsLeafPolicy::RolloutOnly) {
  ps::MctsConfig player_one_config = tactical_config(149, 32, true, 4096);
  ps::MctsConfig player_two_config = tactical_config(151, 32, true, 4096);
  player_one_config.leaf_policy = leaf_policy;
  player_two_config.leaf_policy = leaf_policy;
  ps::MctsBot player_one(player_one_config);
  ps::MctsBot player_two(player_two_config);
  ps::GameState state = ps::make_initial_state();
  MctsGameTrace trace;

  for (std::size_t ply = 0; !ps::is_terminal(state); ++ply) {
    require(ply < 512,
            "A deterministic tactical self-play game should terminate.");
    ps::MctsBot &bot = state.to_move == ps::Player::One
                           ? player_one
                           : player_two;
    state = ps::apply_move(state, bot.choose_move(state));
    trace.searches.push_back(bot.last_search_stats());
  }
  trace.path = state.path;
  return trace;
}

void mcts_bot_chooses_a_legal_move_and_reports_its_name() {
  const ps::GameState state = ps::make_initial_state();
  ps::MctsBot bot(config_with(7, 64, 1.0));

  const ps::Move move = bot.choose_move(state);

  require(bot.name() == "MctsBot", "MctsBot should expose its stable public name.");
  require(contains_move(ps::legal_moves(state), move.to),
          "MctsBot must always return a legal move.");
}

void mcts_defaults_use_promoted_tactical_search() {
  const ps::MctsConfig config;
  require(config.rollout_policy == ps::MctsRolloutPolicy::Tactical &&
              config.reuse_tree && config.max_nodes == 65536 &&
              config.leaf_policy == ps::MctsLeafPolicy::RolloutOnly &&
              config.quiescence_max_depth == 8 &&
              config.quiescence_max_nodes == 256,
          "The promoted MCTS defaults should use tactical rollouts, reuse, and "
          "the fixed node bound while keeping quiescence experimental.");
}

void identical_searches_produce_identical_moves_and_statistics() {
  const ps::GameState state = ps::make_initial_state();
  const ps::MctsConfig config = config_with(123456789, 128, 1.25);
  ps::MctsBot first(config);
  ps::MctsBot second(config);

  const ps::Move first_move = first.choose_move(state);
  const ps::Move second_move = second.choose_move(state);

  require(first_move == second_move,
          "Identical MCTS state, seed, and configuration should choose the same move.");
  require(same_stats(first.last_search_stats(), second.last_search_stats()),
          "Identical MCTS searches should report identical statistics.");
}

void incremental_batches_match_one_complete_run() {
  const ps::GameState state = ps::make_initial_state();
  constexpr std::uint64_t seed = 0x123456789ABCDEF0ULL;
  constexpr double exploration = 1.125;

  ps::MctsSearch complete(state, seed, exploration);
  complete.run_iterations(2000);

  ps::MctsSearch incremental(state, seed, exploration);
  for (int batch = 0; batch < 20; ++batch) {
    incremental.run_iterations(100);
  }

  require(complete.best_move() == incremental.best_move(),
          "Running MCTS in batches must not change its selected move.");
  require(same_stats(complete.stats(), incremental.stats()),
          "Running MCTS in batches must preserve every deterministic statistic.");
  require(complete.stats().iterations == 2000,
          "The complete incremental search should contain exactly 2,000 iterations.");
}

void mcts_bot_honors_the_configured_iteration_count() {
  const ps::GameState state = ps::make_initial_state();
  ps::MctsBot bot(config_with(19, 37, 0.75));

  (void)bot.choose_move(state);

  require(bot.last_search_stats().iterations == 37,
          "MctsBot should run exactly its configured number of iterations.");
}

void mcts_bot_selects_immediate_wins_for_either_player() {
  ps::MctsBot player_one(config_with(31, 64, 1.0));
  const ps::GameState north_mouth =
      make_clean_state_at(ps::Point{3, 1}, ps::Player::One);
  require(player_one.choose_move(north_mouth) == ps::Move{{4, 0}},
          "Player One should take the unique immediate north-goal win.");
  require(player_one.last_search_stats().iterations == 0,
          "An immediate Player One win should be selected before sampling.");

  ps::MctsSearch incremental(north_mouth, 31, 1.0);
  incremental.run_iterations(64);
  require(incremental.best_move() == ps::Move{{4, 0}} &&
              incremental.stats().iterations == 0,
          "The incremental API should also resolve an immediate win without sampling.");

  ps::MctsBot player_two(config_with(37, 64, 1.0));
  const ps::GameState south_mouth =
      make_clean_state_at(ps::Point{5, 11}, ps::Player::Two);
  require(player_two.choose_move(south_mouth) == ps::Move{{4, 12}},
          "Player Two should take the unique immediate south-goal win.");
  require(player_two.last_search_stats().iterations == 0,
          "An immediate Player Two win should be selected before sampling.");
}

void player_two_minimizes_player_one_reward() {
  const ps::Point player_one_branch{3, 5};
  const ps::Point player_two_branch{5, 5};
  const ps::GameState state = make_player_two_choice(true);
  const std::vector<ps::Move> root_moves = ps::legal_moves(state);

  require(root_moves.size() == 2 && root_moves[0].to == player_one_branch &&
              root_moves[1].to == player_two_branch,
          "The Player Two fixture should expose its two branches in legal-move order.");
  require(ps::apply_move(state, root_moves[0]).status == ps::Status::InProgress &&
              ps::apply_move(state, root_moves[1]).status == ps::Status::InProgress,
          "Neither Player Two root move should bypass search as an immediate win.");

  ps::MctsSearch search(state, 17, 0.0);
  search.run_iterations(2);
  require(search.best_move() == ps::Move{player_two_branch},
          "With equal visits, Player Two should prefer the lower Player One mean value.");

  search.run_iterations(62);
  require(search.best_move() == ps::Move{player_two_branch},
          "Player Two selection nodes should keep choosing the branch that wins for Player Two.");
  require(search.stats().root_value < -0.9,
          "A forced Player Two win should have a strongly negative Player One root value.");
}

void final_move_ties_use_existing_legal_move_order() {
  const ps::Point first_legal_move{3, 5};
  const ps::GameState state = make_player_two_choice(false);
  ps::MctsSearch search(state, 17, 0.0);

  search.run_iterations(2);

  require(search.best_move() == ps::Move{first_legal_move},
          "Equal root visits and means should be resolved by existing legal-move order.");
}

void rebounds_keep_the_same_player_perspective() {
  const ps::Point rebound{4, 5};
  const ps::GameState state = make_rebound_choice();
  const std::vector<ps::Move> root_moves = ps::legal_moves(state);

  require(root_moves.size() == 1 && root_moves.front().to == rebound,
          "The rebound fixture should have one forced root move.");
  const ps::GameState after_rebound = ps::apply_move(state, root_moves.front());
  require(after_rebound.status == ps::Status::InProgress &&
              after_rebound.to_move == ps::Player::One,
          "The forced first move should rebound to another Player One decision.");

  ps::MctsSearch search(state, 23, 0.0);
  search.run_iterations(64);

  require(search.stats().root_value > 0.9,
          "A forced winning rebound line must not be negated as though turns alternated.");
}

void mcts_bot_rejects_terminal_states() {
  ps::MctsBot bot(config_with(41, 8, 1.0));
  ps::GameState terminal = ps::make_initial_state();
  terminal.status = ps::Status::WonByOne;

  require_invalid_argument([&] { (void)bot.choose_move(terminal); },
                           "MctsBot must reject terminal states.");
}

void mcts_bot_rejects_blocked_nonterminal_states() {
  ps::MctsBot bot(config_with(43, 8, 1.0));
  ps::GameState blocked = ps::make_initial_state();
  const std::vector<ps::Move> moves = ps::legal_moves(blocked);
  for (const ps::Move &move : moves) {
    blocked.used_segments.insert(ps::Segment{blocked.ball, move.to});
  }

  require(blocked.status == ps::Status::InProgress &&
              ps::legal_moves(blocked).empty(),
          "The blocked fixture should be nonterminal while exposing no legal moves.");
  require_invalid_argument([&] { (void)bot.choose_move(blocked); },
                           "MctsBot must reject blocked states without legal moves.");
}

void invalid_mcts_configurations_are_rejected() {
  require_invalid_argument(
      [] { (void)ps::MctsBot(config_with(47, 0, 1.0)); },
      "MctsBot should reject a zero iteration budget.");
  require_invalid_argument(
      [] { (void)ps::MctsBot(config_with(47, 1, -0.01)); },
      "MctsBot should reject a negative exploration constant.");
  require_invalid_argument(
      [] {
        (void)ps::MctsBot(config_with(
            47, 1, std::numeric_limits<double>::infinity()));
      },
      "MctsBot should reject an infinite exploration constant.");
  require_invalid_argument(
      [] {
        (void)ps::MctsBot(config_with(
            47, 1, std::numeric_limits<double>::quiet_NaN()));
      },
      "MctsBot should reject a NaN exploration constant.");
  ps::MctsConfig too_few_nodes = config_with(47, 1, 1.0);
  too_few_nodes.max_nodes = 1;
  require_invalid_argument(
      [&] { (void)ps::MctsBot(too_few_nodes); },
      "MctsBot should reject a tree bound smaller than two nodes.");

  ps::MctsConfig zero_quiescence_depth = config_with(47, 1, 1.0);
  zero_quiescence_depth.quiescence_max_depth = 0;
  require_invalid_argument(
      [&] { (void)ps::MctsBot(zero_quiescence_depth); },
      "MctsBot should reject a zero quiescence depth even while disabled.");
  ps::MctsConfig excessive_quiescence_depth = config_with(47, 1, 1.0);
  excessive_quiescence_depth.quiescence_max_depth =
      ps::MctsConfig::maximum_quiescence_max_depth + 1U;
  require_invalid_argument(
      [&] { (void)ps::MctsBot(excessive_quiescence_depth); },
      "MctsBot should reject an excessive quiescence depth.");
  ps::MctsConfig zero_quiescence_nodes = config_with(47, 1, 1.0);
  zero_quiescence_nodes.quiescence_max_nodes = 0;
  require_invalid_argument(
      [&] { (void)ps::MctsBot(zero_quiescence_nodes); },
      "MctsBot should reject a zero quiescence node budget even while disabled.");
  ps::MctsConfig excessive_quiescence_nodes = config_with(47, 1, 1.0);
  excessive_quiescence_nodes.quiescence_max_nodes =
      ps::MctsConfig::maximum_quiescence_max_nodes + 1U;
  require_invalid_argument(
      [&] { (void)ps::MctsBot(excessive_quiescence_nodes); },
      "MctsBot should reject an excessive quiescence node budget.");
  ps::MctsConfig uniform_quiescence = reference_config(47, 1);
  uniform_quiescence.leaf_policy =
      ps::MctsLeafPolicy::TacticalQuiescence;
  require_invalid_argument(
      [&] { (void)ps::MctsBot(uniform_quiescence); },
      "Tactical quiescence should require the Tactical fallback rollout.");
  ps::MctsConfig unknown_leaf_policy = config_with(47, 1, 1.0);
  unknown_leaf_policy.leaf_policy = static_cast<ps::MctsLeafPolicy>(99);
  require_invalid_argument(
      [&] { (void)ps::MctsBot(unknown_leaf_policy); },
      "MctsBot should reject an unknown leaf policy.");

  const ps::GameState state = ps::make_initial_state();
  require_invalid_argument(
      [&] { (void)ps::MctsSearch(state, 47, -1.0); },
      "The incremental search API should reject invalid exploration constants too.");
}

void compact_search_position_matches_authoritative_rules() {
  const ps::RulesConfig configurations[]{
      {},
      {6, 8},
      {10, 12},
      {8, 10, ps::GoalRule::OwnGoalsAllowed, ps::BlockedRule::MoverLoses},
  };
  for (const ps::RulesConfig config : configurations) {
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
      ps::RandomBot bot(seed * 0x9e3779b97f4a7c15ULL);
      ps::GameState state = ps::make_initial_state(config);
      for (std::size_t ply = 0; ply < 512; ++ply) {
        require_compact_position_matches(state);
        if (ps::is_terminal(state)) {
          break;
        }
        state = ps::apply_move(state, bot.choose_move(state));
      }
      require(ps::is_terminal(state),
              "Seeded differential games should reach a terminal position.");
    }
  }

  const ps::RulesConfig codingame_config{
      8, 10, ps::GoalRule::OwnGoalsAllowed, ps::BlockedRule::MoverLoses};
  require_compact_position_matches(
      make_clean_state_at({4, 1}, ps::Player::Two, codingame_config));
  require_compact_position_matches(
      make_clean_state_at({4, 11}, ps::Player::One, codingame_config));

  ps::GameState blocked =
      make_clean_state_at({4, 6}, ps::Player::One, codingame_config);
  block_edges_except(blocked, {4, 5}, {{4, 6}});
  require_compact_position_matches(blocked);
}

void tactical_probe_proves_multi_rebound_goals_for_both_players() {
  for (const ps::Player player : {ps::Player::One, ps::Player::Two}) {
    ps::GameState state = make_forced_rebound_goal(player);
    for (int rebound = 0; rebound < 2; ++rebound) {
      const std::vector<ps::Move> moves = ps::legal_moves(state);
      require(moves.size() == 1,
              "The rebound-goal fixture should remain forced.");
      state = ps::apply_move(state, moves.front());
      require(!ps::is_terminal(state) && state.to_move == player,
              "Consecutive rebounds must retain the actual player to move.");
    }

    const ps::detail::TacticalProbeResult result =
        probe_state(make_forced_rebound_goal(player));
    require(result.proven_winner == player &&
                result.proving_move ==
                    ps::legal_moves(make_forced_rebound_goal(player)).front() &&
                result.stats.nodes == 4 &&
                result.stats.max_depth == 3 &&
                !result.stats.depth_cutoff && !result.stats.node_cutoff,
            "The probe should prove a three-move same-player rebound goal.");
  }
}

void tactical_probe_handles_immediate_and_delayed_traps() {
  for (const ps::Player player : {ps::Player::One, ps::Player::Two}) {
    const ps::Point root{4, player == ps::Player::One ? 5 : 7};
    const ps::Point trap{3, root.y};
    ps::GameState immediate = make_clean_state_at(root, player);
    block_edges_except(immediate, root, {trap});
    block_edges_except(immediate, trap, {root});
    require(ps::winner(ps::apply_move(immediate, ps::Move{trap})) == player,
            "The immediate trap should block the opponent after one move.");
    require(probe_state(immediate).proven_winner == player,
            "The probe should prove an immediate terminal trap.");

    const ps::GameState delayed = make_delayed_trap(player);
    const ps::GameState after_rebound =
        ps::apply_move(delayed, ps::legal_moves(delayed).front());
    require(!ps::is_terminal(after_rebound) &&
                after_rebound.to_move == player,
            "The delayed trap should first require a same-player rebound.");
    require(probe_state(delayed).proven_winner == player,
            "The probe should prove the delayed trapping sequence.");
  }
}

void tactical_probe_proves_a_loss_only_after_every_reply_loses() {
  for (const ps::Player losing_player : {ps::Player::One, ps::Player::Two}) {
    const ps::GameState state = make_every_reply_loses(losing_player);
    require(ps::legal_moves(state).size() == 2,
            "The all-replies-lose fixture should expose two defenses.");
    for (const ps::Move move : ps::legal_moves(state)) {
      const ps::GameState child = ps::apply_move(state, move);
      const auto child_result = probe_state(child);
      require(child_result.proven_winner == ps::opponent(losing_player),
              "Every individual defense should retain the opponent's win.");
    }
    require(probe_state(state).proven_winner == ps::opponent(losing_player),
            "A loss should be proven after all authoritative replies lose.");
  }
}

void tactical_probe_keeps_a_rebound_attack_unknown_when_one_escape_survives() {
  const ps::GameState attack = make_rebound_attack_with_one_escape();
  const ps::detail::TacticalProbeResult attack_result = probe_state(attack);
  require(!attack_result.proven_winner.has_value(),
          "A tempting rebound attack must remain unknown when the defender can escape.");

  ps::GameState defense = attack;
  for (int move = 0; move < 2; ++move) {
    const std::vector<ps::Move> forced = ps::legal_moves(defense);
    require(forced.size() == 1,
            "The attack prefix should contain two forced moves.");
    defense = ps::apply_move(defense, forced.front());
  }
  require(defense.to_move == ps::Player::Two &&
              ps::legal_moves(defense).size() == 3,
          "The defender should face three replies after the rebound combination.");

  for (const ps::Move reply : ps::legal_moves(defense)) {
    const ps::GameState child = ps::apply_move(defense, reply);
    const ps::detail::TacticalProbeResult child_result = probe_state(child);
    if (reply == ps::Move{{5, 3}}) {
      ps::GameState escaped = child;
      require(escaped.to_move == ps::Player::Two &&
                  ps::legal_moves(escaped).size() == 1,
              "The safe defense should begin with a forced Player Two rebound.");
      escaped = ps::apply_move(escaped, ps::legal_moves(escaped).front());
      require(escaped.to_move == ps::Player::Two &&
                  ps::legal_moves(escaped).size() == 1,
              "The forced defense should preserve Player Two across two levels.");
      escaped = ps::apply_move(escaped, ps::legal_moves(escaped).front());
      require(!ps::is_terminal(escaped) &&
                  escaped.to_move == ps::Player::One,
              "The defensive rebound chain should escape into a quiet position.");
      require(!child_result.proven_winner.has_value(),
              "The multi-rebound defensive escape should reach a quiet unknown state.");
    } else {
      require(child_result.proven_winner == ps::Player::One,
              "Each tempting alternative should lose to the forced goal sequence.");
    }
  }

  require(!probe_state(defense).proven_winner.has_value(),
          "One surviving defense must prevent a false attacker proof.");
}

void tactical_probe_preserves_the_escape_from_an_arena_failure() {
  // Minimized from the frozen arena report's pair 103 loss: Tactical chose
  // (7,3), while (5,3) was the only defense not refuted by the forcing search.
  const ps::GameState state = state_after_path({
      {5, 5}, {6, 5}, {7, 4}, {6, 4}, {5, 3}, {5, 4}, {4, 3}, {5, 3},
      {4, 4}, {4, 3}, {3, 2}, {2, 3}, {1, 2}, {2, 1}, {3, 2}, {3, 1},
      {2, 2}, {3, 2}, {4, 1}, {5, 1}, {6, 2}, {6, 1}, {5, 2}, {6, 2},
  });
  const std::vector<ps::Move> moves = ps::legal_moves(state);
  const std::vector<ps::Move> expected{
      {{5, 3}}, {{6, 3}}, {{7, 1}}, {{7, 2}}, {{7, 3}},
  };
  require(state.ball == ps::Point{6, 2} &&
              state.to_move == ps::Player::Two && moves == expected,
          "The arena regression should retain its five defenses in legal order.");

  require(!probe_state(state, 8, 2048).proven_winner.has_value(),
          "The surviving arena defense must keep the root result Unknown.");
  require(!probe_state(ps::apply_move(state, moves[0]), 8, 2048)
               .proven_winner.has_value(),
          "The (5,3) escape should remain unrefuted.");
  require(probe_state(ps::apply_move(state, moves[1]), 8, 2048)
              .proven_winner == ps::Player::One &&
              probe_state(ps::apply_move(state, moves[2]), 8, 2048)
                      .proven_winner == ps::Player::One,
          "The probe should refute representative losing alternatives.");
}

void tactical_probe_cutoffs_are_unknown_and_restore_the_position() {
  const ps::GameState forcing = make_forced_rebound_goal(ps::Player::One);
  const ps::detail::TacticalProbeResult depth_limited =
      probe_state(forcing, 2, 256);
  require(!depth_limited.proven_winner.has_value() &&
              depth_limited.stats.depth_cutoff &&
              !depth_limited.stats.node_cutoff &&
              depth_limited.stats.max_depth == 2,
          "A nonterminal depth boundary should return Unknown.");

  const ps::detail::TacticalProbeResult node_limited =
      probe_state(forcing, 8, 2);
  require(!node_limited.proven_winner.has_value() &&
              node_limited.stats.node_cutoff &&
              node_limited.stats.nodes == 2,
          "Exhausting the tactical node budget should return Unknown without overshoot.");

  const ps::detail::TacticalProbeResult quiet =
      probe_state(ps::make_initial_state());
  require(!quiet.proven_winner.has_value() && quiet.stats.nodes == 1 &&
              !quiet.stats.depth_cutoff && !quiet.stats.node_cutoff,
          "A quiet position should return Unknown without reporting a cutoff.");
}

void quiescence_proofs_propagate_from_the_mcts_frontier() {
  const ps::GameState state = make_forced_rebound_goal(ps::Player::One);
  ps::MctsBot bot(quiescence_config(173, 64));

  require(bot.choose_move(state) == ps::legal_moves(state).front(),
          "The integrated search should follow the forced rebound.");
  const ps::SearchStats stats = bot.last_search_stats();
  require(stats.iterations == 1 && stats.proven_winner == ps::Player::One &&
              stats.proven_nodes >= 2 && stats.simulated_plies == 0 &&
              stats.tactical_probes == 1 &&
              stats.tactical_solved_positions == 1 &&
              stats.tactical_nodes == 3 && stats.max_tactical_depth == 2,
          "A frontier proof should supply the exact reward, mark the node, and stop the root.");
}

void a_quiescence_proven_root_retains_its_proving_move() {
  const ps::GameState state = make_rebound_goal_choice();
  const ps::Move winning_move{{4, 2}};
  bool exercised_saturated_root = false;

  for (std::uint64_t seed = 1; seed <= 64; ++seed) {
    ps::MctsConfig config = quiescence_config(seed, 2, false, 2);
    ps::MctsBot bot(config);
    const ps::Move chosen = bot.choose_move(state);
    const ps::SearchStats stats = bot.last_search_stats();
    if (!stats.expansion_saturated) {
      continue;
    }
    exercised_saturated_root = true;
    require(stats.proven_winner == ps::Player::One &&
                chosen == winning_move,
            "A saturated root must choose the move identified by its tactical proof.");
    break;
  }

  require(exercised_saturated_root,
          "The principal-move fixture should exercise a saturated proven root.");
}

void unknown_quiescence_probes_do_not_consume_rollout_rng() {
  const ps::GameState state = ps::make_initial_state();
  ps::MctsConfig rollout_config = tactical_config(179, 1);
  ps::MctsConfig probe_config = rollout_config;
  probe_config.leaf_policy = ps::MctsLeafPolicy::TacticalQuiescence;
  ps::MctsBot rollout_only(rollout_config);
  ps::MctsBot with_probe(probe_config);

  const ps::Move rollout_move = rollout_only.choose_move(state);
  const ps::Move probe_move = with_probe.choose_move(state);
  const ps::SearchStats rollout_stats = rollout_only.last_search_stats();
  const ps::SearchStats probe_stats = with_probe.last_search_stats();
  require(probe_move == rollout_move &&
              probe_stats.iterations == rollout_stats.iterations &&
              probe_stats.nodes == rollout_stats.nodes &&
              probe_stats.simulated_plies == rollout_stats.simulated_plies &&
              probe_stats.root_value == rollout_stats.root_value &&
              probe_stats.total_root_visits == rollout_stats.total_root_visits,
          "An unknown probe should leave the existing Tactical rollout RNG stream unchanged.");
  require(probe_stats.tactical_probes == 1 &&
              probe_stats.tactical_solved_positions == 0 &&
              has_no_tactical_probe_work(rollout_stats),
          "Probe work should be counted separately and remain zero when disabled.");
}

void uniform_non_reusing_search_preserves_reference_fixtures() {
  {
    const ps::GameState state = ps::make_initial_state();
    ps::MctsBot bot(reference_config(123456789, 64));
    require(bot.choose_move(state) == ps::Move{{3, 6}},
            "The optimized uniform search should preserve the 64-iteration opening move.");
    const ps::SearchStats stats = bot.last_search_stats();
    require(stats.iterations == 64 && stats.nodes == 65 &&
                stats.simulated_plies == 3000 && stats.root_value == -0.125,
            "The optimized uniform search should preserve its opening counters.");
  }

  {
    const ps::GameState state = ps::make_initial_state();
    ps::MctsBot bot(reference_config(0x123456789ABCDEF0ULL, 2000));
    require(bot.choose_move(state) == ps::Move{{3, 5}},
            "The optimized uniform search should preserve the 2,000-iteration opening move.");
    const ps::SearchStats stats = bot.last_search_stats();
    require(stats.iterations == 2000 && stats.nodes == 2001 &&
                stats.simulated_plies == 76582 && stats.root_value == 0.004,
            "The optimized uniform search should preserve its long opening counters.");
  }

  {
    ps::GameState state = ps::make_initial_state();
    for (const ps::Point destination :
         std::vector<ps::Point>{{4, 5}, {3, 5}, {4, 4}, {3, 3}}) {
      state = ps::apply_move(state, ps::Move{destination});
    }
    ps::MctsBot bot(reference_config(777, 2000));
    require(bot.choose_move(state) == ps::Move{{4, 2}},
            "The optimized uniform search should preserve the midgame move.");
    const ps::SearchStats stats = bot.last_search_stats();
    require(stats.iterations == 2000 && stats.nodes == 1894 &&
                stats.simulated_plies == 63122 && stats.root_value == 0.164,
            "The optimized uniform search should preserve its midgame counters.");
  }
}

void uniform_non_reusing_game_preserves_reference_path() {
  const std::vector<ps::Point> expected{
      {4, 6}, {4, 7}, {3, 6}, {3, 7}, {4, 7}, {3, 8},
      {2, 8}, {2, 9}, {1, 10}, {1, 11}, {0, 10}, {1, 9},
      {2, 8}, {3, 7}, {2, 6}, {2, 7}, {3, 6}, {3, 5},
      {3, 4}, {3, 3}, {2, 2}, {3, 1}, {4, 0},
  };
  ps::MctsBot player_one(reference_config(17, 64));
  ps::RandomBot player_two(23);
  ps::GameState state = ps::make_initial_state();
  while (!ps::is_terminal(state)) {
    ps::Bot &bot = state.to_move == ps::Player::One
                       ? static_cast<ps::Bot &>(player_one)
                       : static_cast<ps::Bot &>(player_two);
    state = ps::apply_move(state, bot.choose_move(state));
  }
  require(state.path == expected,
          "Uniform non-reusing play should preserve the complete reference match.");
}

void tactical_rollout_only_preserves_frozen_reference_fixtures() {
  {
    const ps::GameState state = ps::make_initial_state();
    ps::MctsBot bot(tactical_config(101, 64));
    require(bot.choose_move(state) == ps::Move{{4, 5}},
            "Tactical RolloutOnly should preserve its frozen opening move.");
    const ps::SearchStats stats = bot.last_search_stats();
    require(stats.iterations == 64 && stats.nodes == 65 &&
                stats.simulated_plies == 4438 && stats.root_value == 0.0 &&
                stats.total_root_visits == 64 && stats.reused_visits == 0 &&
                stats.max_depth == 3 && stats.proven_nodes == 0 &&
                !stats.proven_winner.has_value() && stats.rebuild_count == 0 &&
                !stats.expansion_saturated && has_no_tactical_probe_work(stats),
            "Disabling the probe should preserve every frozen Tactical counter.");
  }

  const std::vector<ps::Point> expected{
      {4, 6}, {5, 5}, {6, 4}, {6, 5}, {6, 6}, {7, 5}, {6, 4},
      {7, 3}, {6, 3}, {7, 2}, {7, 1}, {6, 2}, {6, 3}, {5, 4},
      {4, 3}, {3, 4}, {4, 5}, {4, 6}, {4, 7}, {3, 7}, {4, 6},
      {5, 7}, {4, 7}, {5, 6}, {6, 7}, {6, 6}, {5, 6}, {6, 5},
      {5, 4}, {5, 3}, {6, 4}, {7, 4}, {7, 3}, {6, 2}, {5, 1},
      {4, 0},
  };
  const MctsGameTrace trace = play_reusing_tactical_game();
  require(trace.path == expected,
          "Tactical RolloutOnly should preserve its complete frozen game.");
  for (const ps::SearchStats &stats : trace.searches) {
    require(has_no_tactical_probe_work(stats),
            "The frozen reference game must perform no tactical probe work.");
  }
}

void tactical_rollouts_block_immediate_goals_for_both_players() {
  for (const ps::Player player : {ps::Player::One, ps::Player::Two}) {
    const ps::GameState state = make_goal_blocking_choice(player);
    const std::vector<ps::Move> moves = ps::legal_moves(state);
    require(moves.size() == 2 && moves.front() == ps::Move{{3, state.ball.y}},
            "The goal-blocking fixture should expose its safe move first.");

    ps::MctsBot bot(tactical_config(101, 2));
    require(bot.choose_move(state) == moves.front(),
            "Tactical rollouts should reject a move that gives the opponent an immediate goal.");
  }
}

void tactical_rollouts_avoid_terminal_self_traps() {
  const ps::GameState state = make_self_trap_choice();
  const std::vector<ps::Move> moves = ps::legal_moves(state);
  require(moves.size() == 2 && moves.front() == ps::Move{{3, 5}},
          "The self-trap fixture should expose the losing rebound first.");
  require(ps::winner(ps::apply_move(state, moves.front())) == ps::Player::Two,
          "The marked rebound should trap Player One immediately.");

  ps::MctsBot bot(tactical_config(103, 1));
  require(bot.choose_move(state) == ps::Move{{5, 5}},
          "The solver should avoid a proven loss while a safe branch remains unexpanded.");
}

void immediate_traps_are_selected_without_sampling() {
  const ps::Point root{4, 5};
  const ps::Point trapping_move{3, 5};
  const ps::Point alternative{5, 5};
  ps::GameState state = make_clean_state_at(root, ps::Player::One);
  block_edges_except(state, root, {trapping_move, alternative});
  block_edges_except(state, trapping_move, {root});
  block_edges_except(state, alternative, {root, ps::Point{6, 5}});

  ps::MctsBot bot(tactical_config(107, 64));
  require(bot.choose_move(state) == ps::Move{trapping_move},
          "MCTS should immediately take a move that leaves the opponent blocked.");
  require(bot.last_search_stats().iterations == 0 &&
              bot.last_search_stats().proven_winner == ps::Player::One,
          "An immediate trapping win should be proven without sampling.");
}

void tactical_solver_proves_forced_wins_and_losses() {
  {
    const ps::GameState state = make_player_two_choice(true);
    ps::MctsBot bot(tactical_config(109, 64));
    require(bot.choose_move(state) == ps::Move{{5, 5}},
            "The tactical solver should choose Player Two's forced win.");
    const ps::SearchStats stats = bot.last_search_stats();
    require(stats.iterations < 64 &&
                stats.proven_winner == ps::Player::Two &&
                stats.proven_nodes >= 3,
            "The root should stop early once Player Two's win is proven.");
  }

  {
    const ps::GameState state = make_player_two_choice(false);
    ps::MctsBot bot(tactical_config(113, 64));
    (void)bot.choose_move(state);
    const ps::SearchStats stats = bot.last_search_stats();
    require(stats.iterations < 64 &&
                stats.proven_winner == ps::Player::One,
            "The tactical solver should prove when every Player Two branch loses.");
  }
  {
    const ps::GameState state = make_rebound_choice();
    ps::MctsBot bot(tactical_config(117, 64));
    require(bot.choose_move(state) == ps::Move{{4, 5}},
            "The solver should follow the forced rebound into Player One's "
            "winning decision.");
    const ps::SearchStats stats = bot.last_search_stats();
    require(stats.iterations < 64 &&
                stats.proven_winner == ps::Player::One,
            "A forced rebound line should propagate its proven winner without "
            "alternating perspective.");
  }
}

void mcts_bot_reuses_reachable_subtrees() {
  ps::MctsConfig config = reference_config(127, 128);
  config.reuse_tree = true;
  ps::MctsBot bot(config);
  ps::GameState state = ps::make_initial_state();

  const ps::Move first = bot.choose_move(state);
  const ps::SearchStats first_stats = bot.last_search_stats();
  require(first_stats.iterations == 128 && first_stats.reused_visits == 0 &&
              first_stats.total_root_visits == 128,
          "The first call should build a new 128-iteration root.");

  state = ps::apply_move(state, first);
  const ps::Move second = bot.choose_move(state);
  const ps::SearchStats second_stats = bot.last_search_stats();
  require(contains_move(ps::legal_moves(state), second.to),
          "A re-rooted search must still return a legal move.");
  require(second_stats.iterations == 128 && second_stats.reused_visits > 0 &&
              second_stats.total_root_visits ==
                  second_stats.reused_visits + second_stats.iterations &&
              second_stats.rebuild_count == 0,
          "A reachable child should retain visits and add exactly the configured new work.");
}

void mcts_bot_rebuilds_when_the_played_branch_was_not_expanded() {
  ps::MctsConfig config = reference_config(131, 1);
  config.reuse_tree = true;
  ps::MctsBot bot(config);
  const ps::GameState root = ps::make_initial_state();
  const ps::Move expanded = bot.choose_move(root);
  const std::vector<ps::Move> moves = ps::legal_moves(root);
  const auto different = std::find_if(
      moves.begin(), moves.end(),
      [&](const ps::Move move) { return move != expanded; });
  require(different != moves.end(),
          "The initial state should have an unexpanded alternative.");

  const ps::GameState external = ps::apply_move(root, *different);
  (void)bot.choose_move(external);
  const ps::SearchStats stats = bot.last_search_stats();
  require(stats.iterations == 1 && stats.reused_visits == 0 &&
              stats.rebuild_count == 1,
          "An unexpanded external move should trigger one deterministic rebuild.");
}

void mcts_bot_rebuilds_for_an_unrelated_state() {
  ps::MctsConfig config = reference_config(139, 16);
  config.reuse_tree = true;
  ps::MctsBot bot(config);
  (void)bot.choose_move(ps::make_initial_state());

  const ps::GameState unrelated =
      make_clean_state_at(ps::Point{4, 4}, ps::Player::Two);
  const ps::Move move = bot.choose_move(unrelated);
  const ps::SearchStats stats = bot.last_search_stats();
  require(contains_move(ps::legal_moves(unrelated), move.to),
          "A rebuilt search on an unrelated state should return a legal move.");
  require(stats.iterations == 16 && stats.reused_visits == 0 &&
              stats.rebuild_count == 1,
          "An unrelated state should discard the old tree and count one rebuild.");
}

void reusing_tactical_games_are_fully_reproducible() {
  const MctsGameTrace first = play_reusing_tactical_game();
  const MctsGameTrace second = play_reusing_tactical_game();
  require(first.path == second.path,
          "Identical seeds should reproduce a complete tree-reusing game.");
  require(first.searches.size() == second.searches.size(),
          "Reproduced games should contain the same number of searches.");
  for (std::size_t index = 0; index < first.searches.size(); ++index) {
    require(same_stats(first.searches[index], second.searches[index]),
            "Every deterministic search counter should reproduce across games.");
  }
}

void reusing_quiescence_games_are_fully_reproducible() {
  const MctsGameTrace first = play_reusing_tactical_game(
      ps::MctsLeafPolicy::TacticalQuiescence);
  const MctsGameTrace second = play_reusing_tactical_game(
      ps::MctsLeafPolicy::TacticalQuiescence);
  require(first.path == second.path &&
              first.searches.size() == second.searches.size(),
          "Identical quiescence seeds should reproduce a complete game.");
  bool observed_probe = false;
  for (std::size_t index = 0; index < first.searches.size(); ++index) {
    require(same_stats(first.searches[index], second.searches[index]),
            "Every deterministic quiescence counter should reproduce.");
    observed_probe =
        observed_probe || first.searches[index].tactical_probes != 0;
  }
  require(observed_probe,
          "The reproducibility game should exercise the tactical probe.");
}

void copied_mcts_bots_preserve_live_search_state() {
  ps::MctsConfig config = reference_config(157, 128);
  config.reuse_tree = true;
  ps::MctsBot original(config);
  ps::GameState state = ps::make_initial_state();
  state = ps::apply_move(state, original.choose_move(state));

  ps::MctsBot copied(original);
  ps::MctsBot assigned(reference_config(999, 1));
  assigned = original;

  const ps::Move original_move = original.choose_move(state);
  const ps::SearchStats original_stats = original.last_search_stats();
  const ps::Move copied_move = copied.choose_move(state);
  const ps::Move assigned_move = assigned.choose_move(state);

  require(copied_move == original_move && assigned_move == original_move,
          "Copies of a live bot should preserve its retained tree and random stream.");
  require(same_stats(copied.last_search_stats(), original_stats) &&
              same_stats(assigned.last_search_stats(), original_stats),
          "Copied bots should reproduce every deterministic search counter.");
}

void mcts_node_bound_saturates_without_stopping_iterations() {
  ps::MctsConfig config = reference_config(137, 64);
  config.max_nodes = 2;
  ps::MctsBot bot(config);
  const ps::GameState state = ps::make_initial_state();
  const ps::Move move = bot.choose_move(state);
  const ps::SearchStats stats = bot.last_search_stats();

  require(contains_move(ps::legal_moves(state), move.to),
          "A saturated tree should still return a legal move.");
  require(stats.iterations == 64 && stats.nodes == 2 &&
              stats.expansion_saturated,
          "A full tree should finish all iterations without allocating past its bound.");
}

}  // namespace

int run_mcts_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };

  const std::vector<TestCase> tests{
      {"mcts_bot_chooses_a_legal_move_and_reports_its_name",
       mcts_bot_chooses_a_legal_move_and_reports_its_name},
      {"mcts_defaults_use_promoted_tactical_search",
       mcts_defaults_use_promoted_tactical_search},
      {"identical_searches_produce_identical_moves_and_statistics",
       identical_searches_produce_identical_moves_and_statistics},
      {"incremental_batches_match_one_complete_run",
       incremental_batches_match_one_complete_run},
      {"mcts_bot_honors_the_configured_iteration_count",
       mcts_bot_honors_the_configured_iteration_count},
      {"mcts_bot_selects_immediate_wins_for_either_player",
       mcts_bot_selects_immediate_wins_for_either_player},
      {"player_two_minimizes_player_one_reward",
       player_two_minimizes_player_one_reward},
      {"final_move_ties_use_existing_legal_move_order",
       final_move_ties_use_existing_legal_move_order},
      {"rebounds_keep_the_same_player_perspective",
       rebounds_keep_the_same_player_perspective},
      {"mcts_bot_rejects_terminal_states", mcts_bot_rejects_terminal_states},
      {"mcts_bot_rejects_blocked_nonterminal_states",
       mcts_bot_rejects_blocked_nonterminal_states},
      {"invalid_mcts_configurations_are_rejected",
       invalid_mcts_configurations_are_rejected},
      {"compact_search_position_matches_authoritative_rules",
       compact_search_position_matches_authoritative_rules},
      {"tactical_probe_proves_multi_rebound_goals_for_both_players",
       tactical_probe_proves_multi_rebound_goals_for_both_players},
      {"tactical_probe_handles_immediate_and_delayed_traps",
       tactical_probe_handles_immediate_and_delayed_traps},
      {"tactical_probe_proves_a_loss_only_after_every_reply_loses",
       tactical_probe_proves_a_loss_only_after_every_reply_loses},
      {"tactical_probe_keeps_a_rebound_attack_unknown_when_one_escape_survives",
       tactical_probe_keeps_a_rebound_attack_unknown_when_one_escape_survives},
      {"tactical_probe_preserves_the_escape_from_an_arena_failure",
       tactical_probe_preserves_the_escape_from_an_arena_failure},
      {"tactical_probe_cutoffs_are_unknown_and_restore_the_position",
       tactical_probe_cutoffs_are_unknown_and_restore_the_position},
      {"quiescence_proofs_propagate_from_the_mcts_frontier",
       quiescence_proofs_propagate_from_the_mcts_frontier},
      {"a_quiescence_proven_root_retains_its_proving_move",
       a_quiescence_proven_root_retains_its_proving_move},
      {"unknown_quiescence_probes_do_not_consume_rollout_rng",
       unknown_quiescence_probes_do_not_consume_rollout_rng},
      {"uniform_non_reusing_search_preserves_reference_fixtures",
       uniform_non_reusing_search_preserves_reference_fixtures},
      {"uniform_non_reusing_game_preserves_reference_path",
       uniform_non_reusing_game_preserves_reference_path},
      {"tactical_rollout_only_preserves_frozen_reference_fixtures",
       tactical_rollout_only_preserves_frozen_reference_fixtures},
      {"tactical_rollouts_block_immediate_goals_for_both_players",
       tactical_rollouts_block_immediate_goals_for_both_players},
      {"tactical_rollouts_avoid_terminal_self_traps",
       tactical_rollouts_avoid_terminal_self_traps},
      {"immediate_traps_are_selected_without_sampling",
       immediate_traps_are_selected_without_sampling},
      {"tactical_solver_proves_forced_wins_and_losses",
       tactical_solver_proves_forced_wins_and_losses},
      {"mcts_bot_reuses_reachable_subtrees",
       mcts_bot_reuses_reachable_subtrees},
      {"mcts_bot_rebuilds_when_the_played_branch_was_not_expanded",
       mcts_bot_rebuilds_when_the_played_branch_was_not_expanded},
      {"mcts_bot_rebuilds_for_an_unrelated_state",
       mcts_bot_rebuilds_for_an_unrelated_state},
      {"reusing_tactical_games_are_fully_reproducible",
       reusing_tactical_games_are_fully_reproducible},
      {"reusing_quiescence_games_are_fully_reproducible",
       reusing_quiescence_games_are_fully_reproducible},
      {"copied_mcts_bots_preserve_live_search_state",
       copied_mcts_bots_preserve_live_search_state},
      {"mcts_node_bound_saturates_without_stopping_iterations",
       mcts_node_bound_saturates_without_stopping_iterations},
  };

  int failures = 0;
  for (const TestCase &test : tests) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << "\n";
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": " << error.what() << "\n";
    } catch (...) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": unknown error\n";
    }
  }

  std::cout << "\n" << (tests.size() - static_cast<std::size_t>(failures)) << "/"
            << tests.size() << " MCTS tests passed.\n";
  return failures == 0 ? 0 : 1;
}
