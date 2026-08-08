#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
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

bool contains_move(const std::vector<ps::Move> &moves, ps::Move move) {
  return std::find(moves.begin(), moves.end(), move) != moves.end();
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

ps::GameState forced_two_edge_rebound() {
  const ps::Point root{4, 4};
  const ps::Point rebound{4, 3};
  const ps::Point handoff{4, 2};
  ps::GameState state = ps::make_initial_state();
  state.ball = root;
  state.path = {root};
  state.used_segments.clear();
  state.visit_count.clear();
  state.visit_count[root] = 1;
  state.visit_count[rebound] = 1;
  block_edges_except(state, root, {rebound});
  block_edges_except(state, rebound, {root, handoff});
  return state;
}

void fixed_node_search_is_legal_and_deterministic() {
  const ps::GameState state = ps::make_initial_state();
  ps::Rank5DerivedBot first;
  ps::Rank5DerivedBot second;
  const ps::Move first_move = first.choose_move(state);
  const ps::Move second_move = second.choose_move(state);
  const auto &left = first.last_search_stats();
  const auto &right = second.last_search_stats();

  require(contains_move(ps::legal_moves(state), first_move),
          "The rank-5-derived bot must choose a legal edge.");
  require(first_move == second_move && left.nodes == right.nodes &&
              left.completed_turn_depth == right.completed_turn_depth &&
              left.root_score == right.root_score &&
              left.budget_exhausted == right.budget_exhausted,
          "Equal fixed-node searches should be deterministic.");
}

void rebound_edges_are_cached_and_match_a_fresh_search() {
  const ps::GameState root = forced_two_edge_rebound();
  ps::Rank5DerivedBot cached;

  const ps::Move first = cached.choose_move(root);
  require(first == ps::Move{ps::Point{4, 3}} &&
              cached.last_search_stats().searches == 1 &&
              cached.last_search_stats().cached_moves_remaining == 1 &&
              cached.last_search_stats().planned_action_length == 2 &&
              cached.last_search_stats().current_edge_index == 0,
          "The first edge should retain the mandatory rebound continuation.");

  const ps::GameState rebound = ps::apply_move(root, first);
  ps::Rank5DerivedBot fresh;
  const ps::Move expected = fresh.choose_move(rebound);
  const ps::Move second = cached.choose_move(rebound);
  require(second == expected && second == ps::Move{ps::Point{4, 2}} &&
              cached.last_search_stats().cached_continuation &&
              cached.last_search_stats().nodes == 0 &&
              cached.last_search_stats().searches == 1 &&
              cached.last_search_stats().cached_moves_remaining == 0 &&
              cached.last_search_stats().planned_action_length == 2 &&
              cached.last_search_stats().current_edge_index == 1,
          "An exact rebound state should reuse the complete-turn result.");
}

void any_expected_state_difference_invalidates_the_cache() {
  ps::Rank5DerivedBot bot;
  const ps::GameState root = forced_two_edge_rebound();
  const ps::Move first = bot.choose_move(root);
  ps::GameState different = ps::apply_move(root, first);
  different.path.push_back(ps::Point{0, 0});

  const ps::Move move = bot.choose_move(different);
  require(move == ps::Move{ps::Point{4, 2}} &&
              !bot.last_search_stats().cached_continuation &&
              bot.last_search_stats().searches == 2,
          "A non-identical state must discard rather than reuse a cached edge.");
}

void undo_to_the_original_root_invalidates_the_cache() {
  ps::Rank5DerivedBot bot;
  const ps::GameState root = forced_two_edge_rebound();
  (void)bot.choose_move(root);

  const ps::Move replayed = bot.choose_move(root);
  require(replayed == ps::Move{ps::Point{4, 3}} &&
              !bot.last_search_stats().cached_continuation &&
              bot.last_search_stats().searches == 2,
          "Undoing to the root must force a new complete-turn search.");
}

void incompatible_rules_are_rejected() {
  ps::Rank5DerivedBot bot;

  ps::RulesConfig wrong_width;
  wrong_width.width = 10;
  require_invalid_argument(
      [&] { (void)bot.choose_move(ps::make_initial_state(wrong_width)); },
      "A non-demo board width must be rejected.");

  ps::RulesConfig wrong_height;
  wrong_height.height = 12;
  require_invalid_argument(
      [&] { (void)bot.choose_move(ps::make_initial_state(wrong_height)); },
      "A non-demo board height must be rejected.");

  ps::RulesConfig wrong_goal;
  wrong_goal.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  require_invalid_argument(
      [&] { (void)bot.choose_move(ps::make_initial_state(wrong_goal)); },
      "Own-goals rules must be rejected independently.");

  ps::RulesConfig wrong_blocking;
  wrong_blocking.blocked_rule = ps::BlockedRule::MoverLoses;
  require_invalid_argument(
      [&] { (void)bot.choose_move(ps::make_initial_state(wrong_blocking)); },
      "Mover-loses blocking must be rejected independently.");
}

void public_metadata_has_frozen_defaults() {
  ps::Rank5DerivedBot bot;
  const ps::CompleteTurnAnalysisConfig &config = bot.config();
  require(config.max_turn_depth == 32 && config.max_nodes == 50'000 &&
              config.transposition_table_entries == 65'536 &&
              config.evaluation_table_entries == 32'768 &&
              config.is_fast_profile() && config.profile_name() == "fast-50k" &&
              ps::Rank5DerivedBot::profile_name() == "50k-demo" &&
              ps::Rank5DerivedBot::original_artifact_name() == "rank_5" &&
              ps::Rank5DerivedBot::original_submission_id() == "41015554" &&
              ps::Rank5DerivedBot::original_sha256().size() == 64 &&
              ps::Rank5DerivedBot::original_rank == 5 &&
              ps::Rank5DerivedBot::original_field_size == 206 &&
              bot.name() == "Rank5DerivedBot" &&
              ps::bot_kind_name(ps::BotKind::Rank5Derived) ==
                  "Rank5DerivedBot",
          "Rank-5-derived metadata should expose the fixed-node defaults.");
}

void fixed_profile_cannot_be_reconfigured() {
  static_assert(std::is_default_constructible_v<ps::Rank5DerivedBot>);
  static_assert(!std::is_constructible_v<ps::Rank5DerivedBot,
                                         ps::CompleteTurnAnalysisConfig>);
  static_assert(!std::is_constructible_v<ps::Rank5DerivedBot,
                                         ps::Rank5DerivedConfig>);

  ps::CompleteTurnAnalysisConfig custom =
      ps::CompleteTurnAnalysisConfig::fast();
  custom.max_nodes = 1;
  require(custom.profile_name() == "custom-analysis" &&
              !custom.is_fast_profile(),
          "Configurable work must identify itself as custom analysis, not "
          "the fixed Rank5Derived profile.");
}

void factory_ignores_seed_and_unknown_kinds_still_fail() {
  ps::BotConfig first_config;
  first_config.kind = ps::BotKind::Rank5Derived;
  first_config.seed = 1;
  ps::BotConfig second_config = first_config;
  second_config.seed = 999;
  const std::unique_ptr<ps::Bot> first = ps::make_bot(first_config);
  const std::unique_ptr<ps::Bot> second = ps::make_bot(second_config);
  const auto *first_rank5 =
      dynamic_cast<const ps::Rank5DerivedBot *>(first.get());
  const auto *second_rank5 =
      dynamic_cast<const ps::Rank5DerivedBot *>(second.get());
  require(first_rank5 != nullptr && second_rank5 != nullptr &&
              first_rank5->config().max_nodes == 50'000 &&
              second_rank5->config().max_nodes == 50'000,
          "The factory should create the fixed configuration regardless of seed.");

  require(ps::bot_kind_name(static_cast<ps::BotKind>(999)) == "UnknownBot",
          "Unknown bot kinds should retain their public fallback name.");
  require_invalid_argument(
      [] {
        ps::BotConfig unknown;
        unknown.kind = static_cast<ps::BotKind>(999);
        (void)ps::make_bot(unknown);
      },
      "The factory must reject an unknown bot kind.");
}

}  // namespace

int run_rank5_derived_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"fixed_node_search_is_legal_and_deterministic",
       fixed_node_search_is_legal_and_deterministic},
      {"rebound_edges_are_cached_and_match_a_fresh_search",
       rebound_edges_are_cached_and_match_a_fresh_search},
      {"any_expected_state_difference_invalidates_the_cache",
       any_expected_state_difference_invalidates_the_cache},
      {"undo_to_the_original_root_invalidates_the_cache",
       undo_to_the_original_root_invalidates_the_cache},
      {"incompatible_rules_are_rejected", incompatible_rules_are_rejected},
      {"public_metadata_has_frozen_defaults",
       public_metadata_has_frozen_defaults},
      {"fixed_profile_cannot_be_reconfigured",
       fixed_profile_cannot_be_reconfigured},
      {"factory_ignores_seed_and_unknown_kinds_still_fail",
       factory_ignores_seed_and_unknown_kinds_still_fail},
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
  std::cout << "\n" << tests.size() - static_cast<std::size_t>(failures)
            << "/" << tests.size() << " rank-5-derived tests passed.\n";
  return failures == 0 ? 0 : 1;
}

#ifdef PAPERSOCCER_STANDALONE_RANK5_DERIVED_TEST
int main() { return run_rank5_derived_tests(); }
#endif
