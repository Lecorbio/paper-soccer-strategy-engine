#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mcts_internal.hpp"
#include "papersoccer/bot.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
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

bool contains_point(const std::vector<ps::Point> &points, ps::Point point) {
  return std::find(points.begin(), points.end(), point) != points.end();
}

ps::GameState make_clean_state_at(ps::Point point,
                                  ps::Player player = ps::Player::One) {
  ps::GameState state = ps::make_initial_state();
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
      if (dx == 0 && dy == 0) continue;
      const ps::Point destination{from.x + dx, from.y + dy};
      if (!contains_point(allowed, destination)) {
        state.used_segments.insert(ps::Segment{from, destination});
      }
    }
  }
}

ps::GameState state_after_path(const std::vector<ps::Point> &path) {
  ps::GameState state = ps::make_initial_state();
  for (const ps::Point destination : path) {
    state = ps::apply_move(state, ps::Move{destination});
  }
  return state;
}

const std::vector<ps::Point> &replay_path_before_ply_34() {
  static const std::vector<ps::Point> path{
      {5, 5}, {5, 6}, {4, 6}, {3, 5}, {2, 6}, {1, 5}, {0, 6},
      {1, 7}, {2, 6}, {2, 5}, {1, 6}, {1, 5}, {0, 4}, {1, 3},
      {2, 4}, {3, 3}, {3, 4}, {4, 3}, {5, 4}, {6, 3}, {7, 2},
      {6, 1}, {5, 2}, {5, 1}, {6, 2}, {5, 2}, {4, 1}, {3, 1},
      {2, 2}, {3, 2}, {2, 1}, {2, 2}, {1, 3},
  };
  return path;
}

const std::vector<ps::Point> &replay_path_before_ply_69() {
  static const std::vector<ps::Point> path = [] {
    std::vector<ps::Point> result = replay_path_before_ply_34();
    const std::vector<ps::Point> suffix{
        {0, 3}, {1, 4}, {2, 4}, {3, 4}, {4, 4}, {5, 5}, {6, 6},
        {7, 5}, {8, 6}, {7, 7}, {6, 6}, {6, 5}, {5, 6}, {5, 7},
        {6, 6}, {5, 6}, {4, 5}, {4, 6}, {5, 7}, {5, 8}, {4, 7},
        {4, 8}, {3, 7}, {2, 8}, {1, 7}, {1, 6}, {2, 6}, {3, 6},
        {3, 7}, {4, 7}, {5, 7}, {4, 8}, {3, 9}, {2, 8}, {2, 7},
    };
    result.insert(result.end(), suffix.begin(), suffix.end());
    return result;
  }();
  return path;
}

void alpha_beta_has_a_stable_name_and_chooses_legally() {
  ps::AlphaBetaConfig config;
  config.max_turn_depth = 3;
  config.max_nodes = 20'000;
  ps::AlphaBetaBot bot(config);
  const ps::GameState state = ps::make_initial_state();
  const ps::Move move = bot.choose_move(state);
  require(bot.name() == "AlphaBetaBot", "AlphaBetaBot should expose its name.");
  require(contains_move(ps::legal_moves(state), move.to),
          "AlphaBetaBot must choose a legal move.");
  const auto &stats = bot.last_search_stats();
  require(stats.completed_turn_depth >= 1 &&
              stats.completed_turn_depth <= stats.attempted_turn_depth &&
              stats.attempted_turn_depth <= config.max_turn_depth,
          "Iterative-depth counters should be ordered.");
  require(!stats.principal_variation.empty() &&
              stats.principal_variation.front() == move,
          "The principal variation should begin with the chosen move.");
  const auto selected = std::find_if(
      stats.root_moves.begin(), stats.root_moves.end(),
      [&](const ps::AlphaBetaRootMove &root_move) {
        return root_move.move == move;
      });
  require(selected != stats.root_moves.end() &&
              selected->bound == ps::AlphaBetaScoreBound::Exact,
          "The selected root move should expose an exact score.");
}

void alpha_beta_is_deterministic() {
  ps::AlphaBetaConfig config;
  config.max_turn_depth = 4;
  config.max_nodes = 30'000;
  ps::AlphaBetaBot first(config);
  ps::AlphaBetaBot second(config);
  const ps::GameState state = ps::make_initial_state();
  const ps::Move first_move = first.choose_move(state);
  const ps::Move second_move = second.choose_move(state);
  const auto &lhs = first.last_search_stats();
  const auto &rhs = second.last_search_stats();
  require(first_move == second_move && lhs.nodes == rhs.nodes &&
              lhs.leaf_evaluations == rhs.leaf_evaluations &&
              lhs.terminal_nodes == rhs.terminal_nodes &&
              lhs.cutoffs == rhs.cutoffs &&
              lhs.transposition_hits == rhs.transposition_hits &&
              lhs.root_score == rhs.root_score &&
              lhs.principal_variation == rhs.principal_variation,
          "Identical alpha-beta searches should be deterministic.");
}

void alpha_beta_is_color_reflection_symmetric() {
  ps::AlphaBetaConfig config;
  config.max_turn_depth = 3;
  config.max_nodes = 30'000;
  ps::GameState player_one = ps::make_initial_state();
  ps::GameState player_two = player_one;
  player_two.to_move = ps::Player::Two;

  ps::AlphaBetaBot first(config);
  ps::AlphaBetaBot second(config);
  const ps::Move north = first.choose_move(player_one);
  const ps::Move south = second.choose_move(player_two);
  require(north.to.x == south.to.x &&
              north.to.y + south.to.y == player_one.config.height + 2 &&
              first.last_search_stats().root_score ==
                  -second.last_search_stats().root_score,
          "Color-swapped searches should choose reflected moves and scores.");
}

void alpha_beta_takes_immediate_goals_for_both_players() {
  ps::AlphaBetaConfig config;
  config.max_turn_depth = 2;
  config.max_nodes = 5'000;
  ps::AlphaBetaBot bot(config);
  const ps::GameState player_one = make_clean_state_at({4, 1}, ps::Player::One);
  require(ps::is_goal_point(player_one.config, bot.choose_move(player_one).to),
          "Player One should take an immediate north goal.");
  const ps::GameState player_two = make_clean_state_at({4, 11}, ps::Player::Two);
  require(ps::is_goal_point(player_two.config, bot.choose_move(player_two).to),
          "Player Two should take an immediate south goal.");
}

void alpha_beta_depth_one_follows_a_rebound_turn_to_goal() {
  const ps::Point root{4, 3};
  const ps::Point rebound{4, 2};
  const ps::Point second_rebound{4, 1};
  const ps::Point goal{4, 0};
  const ps::Point quiet{5, 3};
  ps::GameState state = make_clean_state_at(root);
  state.visit_count[rebound] = 1;
  state.visit_count[second_rebound] = 1;
  block_edges_except(state, root, {rebound, quiet});
  block_edges_except(state, rebound, {root, second_rebound});
  block_edges_except(state, second_rebound, {rebound, goal});
  ps::AlphaBetaConfig config;
  config.max_turn_depth = 1;
  config.max_nodes = 5'000;
  config.max_search_plies = 6;
  ps::AlphaBetaBot bot(config);
  require(bot.choose_move(state).to == rebound,
          "Depth one should see a complete same-player rebound goal.");
  require(bot.last_search_stats().root_score > 900'000,
          "The rebound goal should receive a mate-band score.");
}

void alpha_beta_rejects_invalid_states_and_configuration() {
  ps::GameState terminal = ps::make_initial_state();
  terminal.status = ps::Status::WonByOne;
  ps::AlphaBetaBot bot;
  require_invalid_argument([&] { (void)bot.choose_move(terminal); },
                           "AlphaBetaBot should reject terminal states.");
  ps::AlphaBetaConfig invalid;
  invalid.max_turn_depth = 0;
  require_invalid_argument([&] { (void)ps::AlphaBetaBot(invalid); },
                           "Zero turn depth should be rejected.");
  invalid = ps::AlphaBetaConfig{};
  invalid.max_nodes = 0;
  require_invalid_argument([&] { (void)ps::AlphaBetaBot(invalid); },
                           "A zero node budget should be rejected.");
  invalid = ps::AlphaBetaConfig{};
  invalid.max_search_plies = 0;
  require_invalid_argument([&] { (void)ps::AlphaBetaBot(invalid); },
                           "A zero ply limit should be rejected.");
}

void alpha_beta_tiny_budget_returns_a_deterministic_fallback() {
  ps::AlphaBetaConfig config;
  config.max_nodes = 1;
  ps::AlphaBetaBot first(config);
  ps::AlphaBetaBot second(config);
  const ps::GameState state = ps::make_initial_state();
  const ps::Move first_move = first.choose_move(state);
  require(first_move == second.choose_move(state) &&
              contains_move(ps::legal_moves(state), first_move.to),
          "A tiny budget should return a deterministic legal fallback.");
  require(first.last_search_stats().budget_exhausted &&
              first.last_search_stats().nodes == 1 &&
              first.last_search_stats().completed_turn_depth == 0,
          "The tiny search should report its interrupted first iteration.");
}

void alpha_beta_reports_a_physical_ply_horizon() {
  const ps::Point root{4, 3};
  const ps::Point rebound{4, 2};
  const ps::Point goal_mouth{4, 1};
  const ps::Point alternative{5, 2};
  ps::GameState state = make_clean_state_at(root);
  state.visit_count[rebound] = 1;
  state.visit_count[goal_mouth] = 1;
  block_edges_except(state, root, {rebound});
  block_edges_except(state, rebound, {root, goal_mouth, alternative});

  ps::AlphaBetaConfig config;
  config.max_turn_depth = 1;
  config.max_nodes = 1'000;
  config.max_search_plies = 1;
  ps::AlphaBetaBot bot(config);
  (void)bot.choose_move(state);
  require(bot.last_search_stats().physical_ply_cutoffs > 0,
          "A truncated rebound turn should report its physical-ply horizon.");
}

void alpha_beta_extends_forced_play_beyond_the_soft_ply_horizon() {
  const std::vector<ps::Point> chain{
      {4, 6}, {3, 5}, {2, 6}, {1, 5}, {0, 4}, {1, 3},
      {2, 4}, {3, 3}, {4, 2}, {5, 2}, {4, 1}, {4, 0},
  };
  ps::GameState state = make_clean_state_at(chain.front());
  for (std::size_t index = 0; index + 1 < chain.size(); ++index) {
    state.visit_count[chain[index]] = 1;
  }
  for (std::size_t index = 0; index + 1 < chain.size(); ++index) {
    std::vector<ps::Point> allowed;
    if (index != 0) {
      allowed.push_back(chain[index - 1]);
    }
    allowed.push_back(chain[index + 1]);
    block_edges_except(state, chain[index], allowed);
  }

  ps::AlphaBetaConfig config;
  config.max_turn_depth = 1;
  config.max_nodes = 100'000;
  config.max_search_plies = 10;
  ps::AlphaBetaBot bot(config);
  require(bot.choose_move(state).to == chain[1],
          "The forced rebound line should have one legal root move.");
  const ps::AlphaBetaSearchStats &stats = bot.last_search_stats();
  require(stats.root_score > 900'000 && stats.terminal_nodes > 0 &&
              stats.max_physical_ply == 11 &&
              stats.physical_ply_cutoffs == 0,
          "A forced line should extend past the soft horizon and prove its goal.");
}

void alpha_beta_transposition_table_preserves_the_result() {
  ps::AlphaBetaConfig without_table;
  without_table.max_turn_depth = 3;
  without_table.max_nodes = 200'000;
  without_table.max_search_plies = 8;
  without_table.transposition_table_entries = 0;
  ps::AlphaBetaConfig with_table = without_table;
  with_table.transposition_table_entries = 65'536;
  ps::AlphaBetaBot first(without_table);
  ps::AlphaBetaBot second(with_table);
  const ps::GameState state = ps::make_initial_state();
  require(first.choose_move(state) == second.choose_move(state) &&
              first.last_search_stats().root_score ==
                  second.last_search_stats().root_score,
          "The transposition table must preserve the minimax result.");
  require(second.last_search_stats().transposition_hits > 0,
          "The enabled table should record useful hits.");
}

void compact_position_hash_restores_after_unmake() {
  const ps::GameState state = ps::make_initial_state();
  auto topology = std::make_shared<ps::detail::SearchTopology>(state.config);
  ps::detail::SearchPosition position(topology, state);
  const ps::detail::PositionKey root_key = position.position_key();
  std::array<std::uint8_t, ps::detail::kMaximumMoves> slots{};
  require(position.legal_slots(slots) > 1,
          "The initial compact position should have alternatives.");
  position.make_move(slots[0]);
  require(position.position_key() != root_key,
          "Making a move should change the compact hash.");
  position.unmake_move();
  require(position.position_key() == root_key,
          "Unmaking should exactly restore the compact hash.");
}

void alpha_beta_avoids_the_replay_ply_34_blunder() {
  const ps::GameState state = state_after_path(replay_path_before_ply_34());
  require(state.ball == ps::Point{1, 3} && state.to_move == ps::Player::Two,
          "The ply-34 fixture should reconstruct its expected root.");
  ps::AlphaBetaBot bot;
  require(bot.choose_move(state).to == ps::Point{0, 2},
          "Alpha-beta should choose the only unrefuted ply-34 defense.");
}

void alpha_beta_avoids_the_replay_ply_69_handoffs() {
  const ps::GameState state = state_after_path(replay_path_before_ply_69());
  require(state.ball == ps::Point{2, 7} && state.to_move == ps::Player::Two,
          "The ply-69 fixture should reconstruct its expected root.");
  ps::AlphaBetaBot bot;
  const ps::Move move = bot.choose_move(state);
  require(contains_point({{1, 6}, {1, 7}, {2, 6}, {3, 6}, {3, 7}}, move.to),
          "Alpha-beta should retain possession instead of choosing a losing handoff.");
}

void alpha_beta_factory_creates_the_requested_bot() {
  ps::BotConfig config;
  config.kind = ps::BotKind::AlphaBeta;
  config.alpha_beta_depth = 2;
  config.alpha_beta_max_nodes = 5'000;
  config.alpha_beta_transposition_table_entries = 1'024;
  config.alpha_beta_max_search_plies = 7;
  std::unique_ptr<ps::Bot> bot = ps::make_bot(config);
  const auto *alpha_beta = dynamic_cast<const ps::AlphaBetaBot *>(bot.get());
  require(alpha_beta != nullptr && bot->name() == "AlphaBetaBot" &&
              alpha_beta->config().max_turn_depth == 2 &&
              alpha_beta->config().max_nodes == 5'000 &&
              alpha_beta->config().transposition_table_entries == 1'024 &&
              alpha_beta->config().max_search_plies == 7 &&
              ps::bot_kind_name(ps::BotKind::AlphaBeta) == "AlphaBetaBot",
          "The public factory should construct and configure AlphaBetaBot.");
}

}  // namespace

int run_alpha_beta_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"alpha_beta_has_a_stable_name_and_chooses_legally",
       alpha_beta_has_a_stable_name_and_chooses_legally},
      {"alpha_beta_is_deterministic", alpha_beta_is_deterministic},
      {"alpha_beta_is_color_reflection_symmetric",
       alpha_beta_is_color_reflection_symmetric},
      {"alpha_beta_takes_immediate_goals_for_both_players",
       alpha_beta_takes_immediate_goals_for_both_players},
      {"alpha_beta_depth_one_follows_a_rebound_turn_to_goal",
       alpha_beta_depth_one_follows_a_rebound_turn_to_goal},
      {"alpha_beta_rejects_invalid_states_and_configuration",
       alpha_beta_rejects_invalid_states_and_configuration},
      {"alpha_beta_tiny_budget_returns_a_deterministic_fallback",
       alpha_beta_tiny_budget_returns_a_deterministic_fallback},
      {"alpha_beta_reports_a_physical_ply_horizon",
       alpha_beta_reports_a_physical_ply_horizon},
      {"alpha_beta_extends_forced_play_beyond_the_soft_ply_horizon",
       alpha_beta_extends_forced_play_beyond_the_soft_ply_horizon},
      {"alpha_beta_transposition_table_preserves_the_result",
       alpha_beta_transposition_table_preserves_the_result},
      {"compact_position_hash_restores_after_unmake",
       compact_position_hash_restores_after_unmake},
      {"alpha_beta_avoids_the_replay_ply_34_blunder",
       alpha_beta_avoids_the_replay_ply_34_blunder},
      {"alpha_beta_avoids_the_replay_ply_69_handoffs",
       alpha_beta_avoids_the_replay_ply_69_handoffs},
      {"alpha_beta_factory_creates_the_requested_bot",
       alpha_beta_factory_creates_the_requested_bot},
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
            << tests.size() << " alpha-beta tests passed.\n";
  return failures == 0 ? 0 : 1;
}
