#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank4_reference_engine
#include "../rank_4/bot.cpp"
#undef turn_action_v2

#if defined(__GNUC__) && !defined(__clang__)
namespace papersoccer::hybrid_engine {
namespace replay_book {
using namespace ::papersoccer::rank4_reference_engine::replay_book;
}
namespace replay_value_model {
using namespace ::papersoccer::rank4_reference_engine::replay_value_model;
}
namespace teacher_residual_model {
using namespace ::papersoccer::rank4_reference_engine::teacher_residual_model;
}
}  // namespace papersoccer::hybrid_engine
#endif

#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 hybrid_engine
#include "bot.cpp"
#undef turn_action_v2

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace reference = papersoccer::rank4_reference_engine;
namespace hybrid = papersoccer::hybrid_engine;

namespace {

struct Decision {
  std::string action;
  reference::SearchStats reference_stats;
  hybrid::SearchStats hybrid_stats;
};

[[noreturn]] void fail(std::string_view message) {
  throw std::runtime_error(std::string(message));
}

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

std::vector<std::string_view> split_transcript(std::string_view transcript) {
  std::vector<std::string_view> turns;
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    if (end == begin) {
      fail("empty transcript action");
    }
    turns.push_back(transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  return turns;
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  for (const std::string_view action : split_transcript(transcript)) {
    reference::apply_encoded_turn(state, action);
  }
  return state;
}

std::pair<std::string, reference::SearchStats> choose_reference(
    const ps::GameState &state, std::uint64_t max_nodes) {
  reference::SearchConfig config;
  config.max_nodes = max_nodes;
  config.max_time_ms = 0;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  reference::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  if (moves.empty()) {
    fail("fixed-work search returned no action");
  }
  ps::GameState replay = state;
  std::string encoded;
  for (const ps::Move move : moves) {
    encoded.push_back(reference::encode_direction(replay.ball, move.to));
    replay = ps::apply_move(replay, move);
  }
  return {encoded, search.stats()};
}

std::pair<std::string, hybrid::SearchStats> choose_hybrid(
    const ps::GameState &state, std::uint64_t max_nodes) {
  hybrid::SearchConfig config;
  config.max_nodes = max_nodes;
  config.max_time_ms = 0;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  hybrid::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  if (moves.empty()) {
    fail("fixed-work search returned no action");
  }
  ps::GameState replay = state;
  std::string encoded;
  for (const ps::Move move : moves) {
    encoded.push_back(hybrid::encode_direction(replay.ball, move.to));
    replay = ps::apply_move(replay, move);
  }
  return {encoded, search.stats()};
}

bool same_stats(const reference::SearchStats &left,
                const hybrid::SearchStats &right) {
  return left.completed_turn_depth == right.completed_turn_depth &&
         left.attempted_turn_depth == right.attempted_turn_depth &&
         left.nodes == right.nodes &&
         left.leaf_evaluations == right.leaf_evaluations &&
         left.terminal_nodes == right.terminal_nodes &&
         left.completed_actions == right.completed_actions &&
         left.cutoffs == right.cutoffs &&
         left.transposition_probes == right.transposition_probes &&
         left.transposition_hits == right.transposition_hits &&
         left.transposition_cutoffs == right.transposition_cutoffs &&
         left.transposition_stores == right.transposition_stores &&
         left.continuation_transposition_hits ==
             right.continuation_transposition_hits &&
         left.evaluation_cache_probes == right.evaluation_cache_probes &&
         left.evaluation_cache_hits == right.evaluation_cache_hits &&
         left.teacher_residual_evaluations ==
             right.teacher_residual_evaluations &&
         left.terminal_bound_cutoffs == right.terminal_bound_cutoffs &&
         left.forced_edges == right.forced_edges &&
         left.root_seed_actions == right.root_seed_actions &&
         left.root_transposition_reuses == right.root_transposition_reuses &&
         left.max_action_edges == right.max_action_edges &&
         left.root_score == right.root_score &&
         left.budget_exhausted == right.budget_exhausted;
}

void fixed_work_delta_is_audited() {
  constexpr std::array<std::string_view, 8> transcripts{{
      "",
      "0/6",
      "7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474",
      "0/6/5/4/5/53/61/0633",
      "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35",
      "0/2/7/45/7/5/71/34/2212/2/7/1/6/1/03636074/33535",
      "7/6/7/53/10/34/71/45/221/2/1/35/70/54/17/43/660/33",
      "0/0/3/67/27/45/5/2/5/6143/5/717271/1/7/532/27412/41/654",
  }};
  constexpr std::array<std::uint64_t, 4> node_budgets{{1, 251, 2'000,
                                                        10'000}};

  std::size_t exact_decisions = 0;
  std::size_t traversal_only_deltas = 0;
  std::size_t action_deltas = 0;
  for (const std::string_view transcript : transcripts) {
    const ps::GameState state = reconstruct(transcript);
    if (ps::is_terminal(state)) {
      fail("parity transcript unexpectedly terminal");
    }
    for (const std::uint64_t node_budget : node_budgets) {
      const auto [reference_action, reference_stats] =
          choose_reference(state, node_budget);
      const auto [hybrid_action, hybrid_stats] =
          choose_hybrid(state, node_budget);
      ps::GameState reference_after = state;
      ps::GameState hybrid_after = state;
      reference::apply_encoded_turn(reference_after, reference_action);
      hybrid::apply_encoded_turn(hybrid_after, hybrid_action);
      if (reference_action != hybrid_action) {
        ++action_deltas;
      } else if (!same_stats(reference_stats, hybrid_stats)) {
        ++traversal_only_deltas;
      } else {
        ++exact_decisions;
      }
    }
  }
  if (exact_decisions != 19U || traversal_only_deltas != 13U ||
      action_deltas != 0U) {
    fail("fixed-work delta panel changed unexpectedly");
  }
  std::cout << "rank_4 compatibility exact=" << exact_decisions
            << " traversal_only_deltas=" << traversal_only_deltas
            << " action_deltas=" << action_deltas << '\n';
}

ps::Point rotate_point(ps::Point point) {
  return {8 - point.x, 12 - point.y};
}

ps::GameState rotate_and_swap(const ps::GameState &state) {
  ps::GameState rotated = state;
  rotated.ball = rotate_point(state.ball);
  rotated.to_move = ps::opponent(state.to_move);
  rotated.path.clear();
  for (const ps::Point point : state.path) {
    rotated.path.push_back(rotate_point(point));
  }
  rotated.used_segments.clear();
  for (const ps::Segment &edge : state.used_segments) {
    rotated.used_segments.insert(
        ps::Segment{rotate_point(edge.a), rotate_point(edge.b)});
  }
  rotated.visit_count.clear();
  for (const auto &[point, count] : state.visit_count) {
    rotated.visit_count[rotate_point(point)] = count;
  }
  return rotated;
}

void mover_relative_tie_is_an_intentional_rank4_delta() {
  const ps::GameState rotated = rotate_and_swap(reconstruct("6/1"));
  const auto [reference_action, reference_stats] =
      choose_reference(rotated, 1);
  const auto [hybrid_action, hybrid_stats] = choose_hybrid(rotated, 1);
  if (reference_action != "05" || hybrid_action != "03") {
    fail("the frozen Rank-4 horizontal-tie delta changed");
  }
  if (reference_stats.nodes != hybrid_stats.nodes ||
      reference_stats.completed_turn_depth !=
          hybrid_stats.completed_turn_depth ||
      reference_stats.root_score != hybrid_stats.root_score) {
    fail("the horizontal-tie delta changed work or value semantics");
  }
}

void replay_inputs_are_identical() {
  if (reference::replay_book::kReplays.size() !=
      hybrid::replay_book::kReplays.size()) {
    fail("replay book size differs from rank_4");
  }
  for (std::size_t index = 0;
       index < reference::replay_book::kReplays.size(); ++index) {
    const auto &left = reference::replay_book::kReplays[index];
    const auto &right = hybrid::replay_book::kReplays[index];
    if (left.player_id != right.player_id ||
        left.first_turn != right.first_turn ||
        left.transcript != right.transcript) {
      fail("replay book content differs from rank_4");
    }
  }
  if (reference::replay_value_model::kEncodedWeights !=
          hybrid::replay_value_model::kEncodedWeights ||
      reference::teacher_residual_model::kWeights !=
          hybrid::teacher_residual_model::kWeights ||
      reference::teacher_residual_model::kBias !=
          hybrid::teacher_residual_model::kBias) {
    fail("model parameters differ from rank_4");
  }
}

}  // namespace

int main() {
  try {
    replay_inputs_are_identical();
    fixed_work_delta_is_audited();
    mover_relative_tie_is_an_intentional_rank4_delta();
    std::cout << "rank_4_jacek_hybrid bounded-delta parity passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "rank_4_jacek_hybrid bounded-delta parity failed: "
              << error.what() << '\n';
    return 1;
  }
}
