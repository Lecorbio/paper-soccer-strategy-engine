#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
         lhs.root_value == rhs.root_value;
}

ps::GameState make_clean_state_at(ps::Point point,
                                  ps::Player to_move = ps::Player::One) {
  ps::GameState state = ps::make_initial_state();
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

ps::MctsConfig config_with(std::uint64_t seed, std::uint32_t iterations,
                           double exploration) {
  ps::MctsConfig config;
  config.seed = seed;
  config.iterations = iterations;
  config.exploration = exploration;
  return config;
}

void mcts_bot_chooses_a_legal_move_and_reports_its_name() {
  const ps::GameState state = ps::make_initial_state();
  ps::MctsBot bot(config_with(7, 64, 1.0));

  const ps::Move move = bot.choose_move(state);

  require(bot.name() == "MctsBot", "MctsBot should expose its stable public name.");
  require(contains_move(ps::legal_moves(state), move.to),
          "MctsBot must always return a legal move.");
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

  const ps::GameState state = ps::make_initial_state();
  require_invalid_argument(
      [&] { (void)ps::MctsSearch(state, 47, -1.0); },
      "The incremental search API should reject invalid exploration constants too.");
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
