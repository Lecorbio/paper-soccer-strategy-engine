#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define PAPER_SOCCER_CONSERVATIVE_FRONTIER_PROOF_TESTING
#include "bot.cpp"
#undef PAPER_SOCCER_CONSERVATIVE_FRONTIER_PROOF_TESTING

namespace papersoccer::leaf_reference_engine {
namespace replay_value_model =
    ::papersoccer::turn_action_v2::replay_value_model;
namespace replay_book = ::papersoccer::turn_action_v2::replay_book;
}

#define PAPER_SOCCER_CONSERVATIVE_FRONTIER_LEAF_REFERENCE
#define turn_action_v2 leaf_reference_engine
#include "bot.cpp"
#undef turn_action_v2
#undef PAPER_SOCCER_CONSERVATIVE_FRONTIER_LEAF_REFERENCE

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;
namespace reference = papersoccer::leaf_reference_engine;

namespace {

constexpr std::string_view kSelectedTranscript =
    "7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474";
constexpr std::string_view kSelectedAction = "42474176";

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

ps::GameState layered_goal_choice(ps::Player mover) {
  const bool player_one = mover == ps::Player::One;
  const int bad_step = player_one ? 1 : -1;
  const ps::Point root{4, 6};
  const ps::Point safe{4, 6 - bad_step};
  const ps::Point bad{4, 6 + bad_step};
  const ps::Point goal{4, player_one ? 12 : 0};
  ps::GameState state = make_clean_state_at(root, mover);
  block_edges_except(state, root, {safe, bad});

  ps::Point previous = bad;
  for (int y = bad.y + bad_step; y != goal.y; y += bad_step) {
    const ps::Point current{4, y};
    state.visit_count[current] = 1;
    block_edges_except(state, current, {previous, {4, y + bad_step}});
    previous = current;
  }
  block_edges_except(state, bad, {root, {4, bad.y + bad_step}});
  return state;
}

struct FixedSearchResult {
  std::string action;
  std::uint32_t completed_depth{};
  int score{};
  std::uint64_t nodes{};
  std::uint64_t goal_hits{};
  std::uint64_t loss_hits{};
  std::uint64_t transposition_hits{};
};

template <typename Search, typename Config>
FixedSearchResult run_fixed_search(const ps::GameState &state, Config config) {
  const ps::GameState before = state;
  Search search(state, config);
  const std::vector<ps::Move> moves = search.run();
  require(same_state(state, before),
          "Search construction and execution must not mutate its input.");

  ps::GameState replayed = state;
  std::string action;
  for (const ps::Move move : moves) {
    action.push_back(cg::encode_direction(replayed.ball, move.to));
    replayed = ps::apply_move(replayed, move);
  }
  const auto &stats = search.stats();
  return {action,
          stats.completed_turn_depth,
          stats.root_score,
          stats.nodes,
          stats.rebound_goal_hits,
          stats.rebound_loss_hits,
          stats.transposition_hits};
}

std::vector<std::string_view> split_transcript(std::string_view transcript) {
  std::vector<std::string_view> turns;
  std::size_t start = 0;
  while (start <= transcript.size()) {
    const std::size_t separator = transcript.find('/', start);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    require(end != start, "Transcript contains an empty turn.");
    turns.push_back(transcript.substr(start, end - start));
    if (separator == std::string_view::npos) {
      break;
    }
    start = separator + 1;
  }
  return turns;
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  for (const std::string_view turn : split_transcript(transcript)) {
    cg::apply_encoded_turn(state, turn);
  }
  return state;
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

void own_goals_and_blocked_mover_follow_contest_rules() {
  ps::GameState north = make_clean_state_at({4, 1}, ps::Player::Two);
  north = ps::apply_move(north, {{4, 0}});
  require(ps::winner(north) == ps::Player::One,
          "Player Two's north own goal should award Player One.");

  ps::GameState south = make_clean_state_at({4, 11}, ps::Player::One);
  south = ps::apply_move(south, {{4, 12}});
  require(ps::winner(south) == ps::Player::Two,
          "Player One's south own goal should award Player Two.");

  ps::GameState blocked = make_clean_state_at({4, 6}, ps::Player::One);
  block_edges_except(blocked, {4, 5}, {{4, 6}});
  blocked = ps::apply_move(blocked, {{4, 5}});
  require(ps::winner(blocked) == ps::Player::Two,
          "A blocked landing should immediately defeat the mover.");
}

void encoded_turn_validation_is_atomic() {
  ps::GameState rebound = forced_two_move_rebound();
  const ps::GameState snapshot = rebound;
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(rebound, "0"); },
      "A turn may not stop during a mandatory rebound.");
  require(same_state(rebound, snapshot),
          "A rejected early turn should not partially mutate state.");

  cg::apply_encoded_turn(rebound, "00");
  require(rebound.ball == ps::Point{4, 2} &&
              rebound.to_move == ps::Player::Two,
          "A complete rebound sequence should hand possession over.");

  ps::GameState initial = ps::make_initial_state(codingame_rules());
  const ps::GameState initial_snapshot = initial;
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(initial, "00"); },
      "A turn may not include an opponent's next move.");
  require(same_state(initial, initial_snapshot),
          "A rejected extra move should not partially mutate state.");
}

void timed_search_returns_a_complete_replayable_action() {
  const ps::GameState original = ps::make_initial_state(codingame_rules());
  ps::GameState chosen = original;
  const std::string action = cg::choose_complete_turn(chosen, 1);
  require(!action.empty(), "Timed search should return an action.");

  ps::GameState replayed = original;
  cg::apply_encoded_turn(replayed, action);
  require(same_state(chosen, replayed),
          "The timed action should reconstruct the chosen state.");
}

void normal_fallback_finishes_mandatory_rebounds() {
  const ps::GameState original = forced_two_move_rebound();
  ps::GameState chosen = original;
  const std::string action = cg::choose_complete_turn(chosen, 1);
  require(action == "00",
          "The normal V2 path should finish the forced rebound action.");

  ps::GameState replayed = original;
  cg::apply_encoded_turn(replayed, action);
  require(same_state(chosen, replayed) &&
              replayed.to_move == ps::Player::Two,
          "The fallback action should be legal and rebound-complete.");
}

void interrupted_search_preserves_a_complete_action() {
  const ps::GameState state = forced_two_move_rebound();
  cg::SearchConfig config;
  config.max_nodes = 1;
  config.max_time_ms = 0;
  cg::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  require(search.stats().budget_exhausted && !moves.empty(),
          "A one-node search should return its complete fallback action.");

  ps::GameState replayed = state;
  std::string encoded;
  for (const ps::Move move : moves) {
    encoded.push_back(cg::encode_direction(replayed.ball, move.to));
    replayed = ps::apply_move(replayed, move);
  }
  ps::GameState verified = state;
  cg::apply_encoded_turn(verified, encoded);
  require(same_state(replayed, verified),
          "The interrupted search fallback should replay exactly.");
}

std::pair<std::string, cg::SearchStats> search_with_one_node(
    const ps::GameState &state) {
  cg::SearchConfig config;
  config.max_turn_depth = 1;
  config.max_nodes = 1;
  config.max_time_ms = 0;
  cg::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  ps::GameState replayed = state;
  std::string encoded;
  for (const ps::Move move : moves) {
    encoded.push_back(cg::encode_direction(replayed.ball, move.to));
    replayed = ps::apply_move(replayed, move);
  }
  ps::GameState verified = state;
  cg::apply_encoded_turn(verified, encoded);
  require(same_state(replayed, verified),
          "A rebound-component proof action should replay exactly.");
  return {encoded, search.stats()};
}

void rebound_component_proofs_are_exact_and_symmetric() {
  const ps::Point north_root{4, 2};
  const ps::Point north_center{4, 1};

  ps::GameState through_visited = make_clean_state_at(north_root);
  through_visited.visit_count[north_center] = 1;
  block_edges_except(through_visited, north_root, {north_center});
  const auto [visited_action, visited_stats] =
      search_with_one_node(through_visited);
  require(visited_action.size() == 2 && visited_action.front() == '0' &&
              visited_stats.rebound_goal_hits == 1,
          "A visited center mouth should prove a two-edge north goal.");
  ps::GameState visited_result = through_visited;
  cg::apply_encoded_turn(visited_result, visited_action);
  require(ps::winner(visited_result) == ps::Player::One,
          "The proven north rebound path should score for Player One.");

  ps::GameState through_fresh = make_clean_state_at(north_root);
  block_edges_except(through_fresh, north_root, {north_center});
  const auto [fresh_action, fresh_stats] = search_with_one_node(through_fresh);
  require(fresh_action == "0" && fresh_stats.rebound_goal_hits == 0,
          "An unvisited center mouth must hand off instead of proving a goal.");

  ps::GameState used_entry = make_clean_state_at(north_root);
  used_entry.visit_count[north_center] = 1;
  block_edges_except(used_entry, north_root, {north_center, {4, 3}});
  used_entry.used_segments.insert(ps::Segment{north_root, north_center});
  const auto [used_action, used_stats] = search_with_one_node(used_entry);
  require(used_action == "4" && used_stats.rebound_goal_hits == 0,
          "A used center-entry edge must not prove a rebound goal.");

  ps::GameState through_post = make_clean_state_at(north_root);
  block_edges_except(through_post, north_root, {{3, 1}});
  const auto [post_action, post_stats] = search_with_one_node(through_post);
  require(post_action == "71" && post_stats.rebound_goal_hits == 1,
          "An unvisited boundary post should still continue into goal.");

  const ps::Point south_root{4, 10};
  const ps::Point south_center{4, 11};
  ps::GameState south = make_clean_state_at(south_root, ps::Player::Two);
  south.visit_count[south_center] = 1;
  block_edges_except(south, south_root, {south_center});
  const auto [south_action, south_stats] = search_with_one_node(south);
  require(south_action.size() == 2 && south_action.front() == '4' &&
              south_stats.rebound_goal_hits == 1,
          "The rebound-goal proof should rotate for Player Two.");
  ps::GameState south_result = south;
  cg::apply_encoded_turn(south_result, south_action);
  require(ps::winner(south_result) == ps::Player::Two,
          "The proven south rebound path should score for Player Two.");

  ps::GameState own_goal = make_clean_state_at(north_root, ps::Player::Two);
  own_goal.visit_count[north_center] = 1;
  block_edges_except(own_goal, north_root, {north_center});
  const auto [own_action, own_stats] = search_with_one_node(own_goal);
  require(!own_action.empty() && own_stats.rebound_goal_hits == 0,
          "A rebound path into the mover's own goal must not be a proof.");

  const ps::Point trapped_root{4, 4};
  const ps::Point trapped_rebound{4, 3};
  ps::GameState trapped = make_clean_state_at(trapped_root);
  trapped.visit_count[trapped_rebound] = 1;
  block_edges_except(trapped, trapped_root, {trapped_rebound});
  block_edges_except(trapped, trapped_rebound, {trapped_root});
  const auto [trapped_action, trapped_stats] = search_with_one_node(trapped);
  require(trapped_action == "0" && trapped_stats.rebound_loss_hits >= 1,
          "A closed rebound component should prove the mover trapped.");
  ps::GameState trapped_result = trapped;
  cg::apply_encoded_turn(trapped_result, trapped_action);
  require(ps::winner(trapped_result) == ps::Player::Two,
          "The closed rebound component should defeat its mover.");
}

void safe_handoff_frontier_term_is_unknown_only_and_symmetric() {
  const ps::GameState north_state =
      make_clean_state_at({3, 5}, ps::Player::One);
  cg::SearchConfig config;
  config.max_turn_depth = 1;
  config.max_nodes = 1'000;
  config.max_time_ms = 0;
  config.replay_value_blend_percent = 0;
  cg::CompleteTurnSearch north(north_state, config);
  const auto [north_outcome, north_frontier] =
      north.inspect_rebound_frontier();
  const int north_without = north.inspect_evaluation(0);
  const int north_with = north.inspect_evaluation(north_frontier);

  const ps::GameState south_state =
      make_clean_state_at({5, 7}, ps::Player::Two);
  cg::CompleteTurnSearch south(south_state, config);
  const auto [south_outcome, south_frontier] =
      south.inspect_rebound_frontier();
  const int south_without = south.inspect_evaluation(0);
  const int south_with = south.inspect_evaluation(south_frontier);

  require(north_outcome == cg::ReboundOutcome::Unknown &&
              south_outcome == cg::ReboundOutcome::Unknown,
          "The frontier heuristic must apply only after an exact Unknown result.");
  require(north_frontier == 8 && south_frontier == 8,
          "A clean interior and its rotation must expose eight endpoints.");
  require(north_with - north_without ==
              static_cast<int>(north_frontier) *
                  cg::kSafeHandoffFrontierWeight &&
              south_with - south_without ==
                  -static_cast<int>(south_frontier) *
                      cg::kSafeHandoffFrontierWeight,
          "The frontier term must rotate by mover sign without changing weight.");

  const ps::Point root{4, 5};
  const ps::Point left_component{3, 5};
  const ps::Point upper_component{4, 4};
  const ps::Point shared_frontier{3, 4};
  ps::GameState converging = make_clean_state_at(root);
  converging.visit_count[left_component] = 1;
  converging.visit_count[upper_component] = 1;
  block_edges_except(converging, root,
                     {left_component, upper_component});
  block_edges_except(converging, left_component,
                     {root, shared_frontier});
  block_edges_except(converging, upper_component,
                     {root, shared_frontier});
  cg::CompleteTurnSearch converging_search(converging, config);
  const auto [converging_outcome, converging_frontier] =
      converging_search.inspect_rebound_frontier();
  require(converging_outcome == cg::ReboundOutcome::Unknown &&
              converging_frontier == 1,
          "Converging component arcs must count their shared endpoint once.");
}

void rebound_goal_is_proven_at_depth_zero() {
  ps::GameState state = make_clean_state_at({4, 10}, ps::Player::One);
  block_edges_except(state, {4, 10}, {{4, 11}});

  cg::SearchConfig config;
  config.max_turn_depth = 1;
  config.max_nodes = 1'000;
  config.max_time_ms = 0;
  config.root_seed_endpoints = false;
  cg::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();

  require(moves.size() == 1 &&
              cg::encode_direction(state.ball, moves.front().to) == '4',
          "The forced fresh landing should be the complete root action.");
  require(search.stats().completed_turn_depth == 1 &&
              search.stats().root_score == -999'998 &&
              search.stats().rebound_goal_hits > 0,
          "The opponent's direct rebound goal must be scored at depth zero.");

  ps::GameState rotated = make_clean_state_at({4, 2}, ps::Player::Two);
  block_edges_except(rotated, {4, 2}, {{4, 1}});
  cg::CompleteTurnSearch rotated_search(rotated, config);
  const std::vector<ps::Move> rotated_moves = rotated_search.run();
  require(rotated_moves.size() == 1 &&
              cg::encode_direction(rotated.ball, rotated_moves.front().to) ==
                  '0',
          "The rotated fresh landing should be the complete root action.");
  require(rotated_search.stats().completed_turn_depth == 1 &&
              rotated_search.stats().root_score == 999'998 &&
              rotated_search.stats().rebound_goal_hits > 0,
          "The rotated depth-zero proof must have the opposite score.");
}

void positive_depth_proof_matches_leaf_reference() {
  std::array<FixedSearchResult, 2> rotated_results{};
  std::size_t rotation = 0;
  for (const ps::Player mover : {ps::Player::One, ps::Player::Two}) {
    const ps::GameState state = layered_goal_choice(mover);

    cg::SearchConfig candidate_config;
    candidate_config.max_turn_depth = 2;
    candidate_config.max_nodes = 1'000'000;
    candidate_config.max_time_ms = 0;
    candidate_config.root_seed_endpoints = false;
    const FixedSearchResult candidate =
        run_fixed_search<cg::CompleteTurnSearch>(state, candidate_config);

    reference::SearchConfig reference_config;
    reference_config.max_turn_depth = 2;
    reference_config.max_nodes = 1'000'000;
    reference_config.max_time_ms = 0;
    reference_config.root_seed_endpoints = false;
    const FixedSearchResult leaf_reference =
        run_fixed_search<reference::CompleteTurnSearch>(state,
                                                        reference_config);

    cg::SearchConfig no_table_config = candidate_config;
    no_table_config.transposition_entries = 0;
    const FixedSearchResult no_table =
        run_fixed_search<cg::CompleteTurnSearch>(state, no_table_config);

    const std::string expected_action =
        mover == ps::Player::One ? "0" : "4";
    require(candidate.completed_depth == 2 &&
                leaf_reference.completed_depth == 2,
            "The crafted branch must complete a positive-depth iteration.");
    require(candidate.action == expected_action &&
                candidate.action == leaf_reference.action &&
                candidate.score == leaf_reference.score,
            "Frontier proof must preserve the exact fixed-depth branch result.");
    require(candidate.transposition_hits > 0 &&
                no_table.completed_depth == candidate.completed_depth &&
                no_table.action == candidate.action &&
                no_table.score == candidate.score,
            "TT-first placement must preserve the no-table minimax result.");
    require(candidate.goal_hits >= 2 &&
                candidate.nodes < leaf_reference.nodes,
            "The positive-depth proof should prune the losing goal branch.");
    rotated_results[rotation++] = candidate;
  }
  require(rotated_results[0].score == -rotated_results[1].score &&
              rotated_results[0].action == "0" &&
              rotated_results[1].action == "4",
          "The exact proof result and selected action must rotate by player.");
}

void selected_replay_correction_activates_exactly() {
  ps::GameState actual = reconstruct(kSelectedTranscript);
  ps::GameState expected = actual;
  cg::apply_encoded_turn(expected, kSelectedAction);

  std::string encoded = "sentinel";
  require(cg::try_replay_correction(actual, 0, kSelectedTranscript, encoded),
          "The independently accepted correction should activate.");
  require(encoded == kSelectedAction,
          "The accepted correction should return the frozen action.");
  require(same_state(actual, expected),
          "The accepted correction should reach the directly replayed state.");
}


void complete_replay_book_is_legal_and_exact() {
  for (const cg::replay_book::Replay &replay : cg::replay_book::kReplays) {
    ps::GameState state = ps::make_initial_state(codingame_rules());
    std::string prefix;
    std::size_t turn = 0;
    std::size_t begin = 0;
    while (begin < replay.transcript.size()) {
      const std::size_t separator = replay.transcript.find('/', begin);
      const std::size_t end = separator == std::string_view::npos
                                  ? replay.transcript.size()
                                  : separator;
      const std::string_view action = replay.transcript.substr(begin, end - begin);
      require(!action.empty(), "A replay-book action cannot be empty.");
      if (turn >= replay.first_turn && turn % 2U == replay.player_id) {
        ps::GameState actual = state;
        ps::GameState expected = state;
        cg::apply_encoded_turn(expected, action);
        std::string encoded = "sentinel";
        require(cg::try_replay_correction(actual, replay.player_id, prefix,
                                          encoded),
                "Every eligible replay-book decision should activate.");
        require(encoded == action && same_state(actual, expected),
                "Every replay-book decision should be exact and legal.");
      }
      cg::apply_encoded_turn(state, action);
      if (!prefix.empty()) {
        prefix.push_back('/');
      }
      prefix.append(action);
      ++turn;
      if (separator == std::string_view::npos) {
        break;
      }
      begin = separator + 1U;
    }
  }
}

void rejected_replay_corrections_are_absent() {
  constexpr std::array<std::string_view, 3> rejected{{
      "7/6/0/5/71/335",
      "7/6/0/5/71/335/01/44/21/2/0/3/17/554",
      "7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474/"
      "42474177/65",
  }};
  for (const std::string_view transcript : rejected) {
    ps::GameState state = reconstruct(transcript);
    const ps::GameState before = state;
    std::string encoded = "sentinel";
    require(!cg::try_replay_correction(state, 0, transcript, encoded),
            "An independently rejected correction activated.");
    require(same_state(state, before) && encoded == "sentinel",
            "A rejected correction should preserve state and output.");
  }
}

void replay_correction_fallback_is_safe() {
  ps::GameState wrong_player = reconstruct(kSelectedTranscript);
  const ps::GameState wrong_player_before = wrong_player;
  std::string encoded = "sentinel";
  require(!cg::try_replay_correction(
              wrong_player, 1, kSelectedTranscript, encoded),
          "The correction should not activate for Player 1.");
  require(same_state(wrong_player, wrong_player_before) &&
              encoded == "sentinel",
          "Wrong-player lookup should preserve state and output.");

  ps::GameState unknown = ps::make_initial_state(codingame_rules());
  const ps::GameState unknown_before = unknown;
  require(!cg::try_replay_correction(unknown, 0, "not/a/transcript", encoded),
          "An unknown transcript should not activate.");
  require(same_state(unknown, unknown_before) && encoded == "sentinel",
          "Unknown lookup should preserve state and output.");

  ps::GameState terminal = reconstruct(kSelectedTranscript);
  terminal.status = ps::Status::WonByOne;
  const ps::GameState terminal_before = terminal;
  require(!cg::try_replay_correction(
              terminal, 0, kSelectedTranscript, encoded),
          "An illegal correction should fail its copy-before-apply guard.");
  require(same_state(terminal, terminal_before) && encoded == "sentinel",
          "An illegal correction should preserve state and output.");
}

}  // namespace

int main() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"directions_round_trip_exactly", directions_round_trip_exactly},
      {"own_goals_and_blocked_mover_follow_contest_rules",
       own_goals_and_blocked_mover_follow_contest_rules},
      {"encoded_turn_validation_is_atomic",
       encoded_turn_validation_is_atomic},
      {"timed_search_returns_a_complete_replayable_action",
       timed_search_returns_a_complete_replayable_action},
      {"normal_fallback_finishes_mandatory_rebounds",
       normal_fallback_finishes_mandatory_rebounds},
      {"interrupted_search_preserves_a_complete_action",
       interrupted_search_preserves_a_complete_action},
      {"rebound_component_proofs_are_exact_and_symmetric",
       rebound_component_proofs_are_exact_and_symmetric},
      {"safe_handoff_frontier_term_is_unknown_only_and_symmetric",
       safe_handoff_frontier_term_is_unknown_only_and_symmetric},
      {"rebound_goal_is_proven_at_depth_zero",
       rebound_goal_is_proven_at_depth_zero},
      {"positive_depth_proof_matches_leaf_reference",
       positive_depth_proof_matches_leaf_reference},
      {"selected_replay_correction_activates_exactly",
       selected_replay_correction_activates_exactly},
      {"complete_replay_book_is_legal_and_exact",
       complete_replay_book_is_legal_and_exact},
      {"rejected_replay_corrections_are_absent",
       rejected_replay_corrections_are_absent},
      {"replay_correction_fallback_is_safe",
       replay_correction_fallback_is_safe},
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
