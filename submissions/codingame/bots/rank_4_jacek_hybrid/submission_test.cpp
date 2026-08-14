#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING
#include "submission.cpp"
#undef PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

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

ps::GameState two_ply_exchange_component(ps::Player mover, bool win) {
  const int forward = mover == ps::Player::One ? -1 : 1;
  const ps::Point root{4, 6};
  const ps::Point reply{4, 6 + forward};
  const ps::Point counterturn{4, 6 + 2 * forward};

  ps::GameState state = make_clean_state_at(root, mover);
  block_edges_except(state, root, {reply});
  block_edges_except(state, reply, {root, counterturn});

  if (win) {
    const int goal_y = mover == ps::Player::One ? 0 : 12;
    ps::Point previous = counterturn;
    for (int y = counterturn.y + forward; y != goal_y; y += forward) {
      const ps::Point current{4, y};
      state.visit_count[current] = 1;
      block_edges_except(state, current, {previous, {4, y + forward}});
      previous = current;
    }
    block_edges_except(
        state, counterturn,
        {reply, {counterturn.x, counterturn.y + forward}});
  } else {
    const ps::Point closed_rebound{counterturn.x,
                                   counterturn.y + forward};
    state.visit_count[closed_rebound] = 1;
    block_edges_except(state, counterturn, {reply, closed_rebound});
    block_edges_except(state, closed_rebound, {counterturn});
  }
  return state;
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
  if (transcript.empty()) {
    return state;
  }
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
  require(cg::kOperationalExactProofs == 7,
          "Operational search must use the selected proof mask 7.");
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

ps::Point rotate_point(ps::Point point) {
  return {8 - point.x, 12 - point.y};
}

ps::GameState rotate_and_swap(const ps::GameState &state) {
  ps::GameState result = state;
  result.ball = rotate_point(state.ball);
  result.to_move = ps::opponent(state.to_move);
  result.path.clear();
  for (const ps::Point point : state.path) {
    result.path.push_back(rotate_point(point));
  }
  result.used_segments.clear();
  for (const ps::Segment &edge : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{rotate_point(edge.a), rotate_point(edge.b)});
  }
  result.visit_count.clear();
  for (const auto &[point, count] : state.visit_count) {
    result.visit_count[rotate_point(point)] = count;
  }
  return result;
}

char rotate_direction(char direction) {
  return static_cast<char>('0' + (direction - '0' + 4) % 8);
}

std::string rotate_action(std::string_view action) {
  std::string rotated;
  rotated.reserve(action.size());
  for (const char direction : action) {
    rotated.push_back(rotate_direction(direction));
  }
  return rotated;
}

void sole_legal_edge_bypasses_move_scoring() {
  ps::GameState base = make_clean_state_at({4, 4});
  block_edges_except(base, {4, 4}, {{4, 3}});
  for (const ps::GameState &state : {base, rotate_and_swap(base)}) {
    cg::SearchConfig config;
    config.max_nodes = 1;
    config.max_time_ms = 0;
    cg::CompleteTurnSearch search(state, config);
    const cg::OrderedMoveList moves = search.ordered_moves_for_test();
    require(moves.count == 1,
            "Sole-edge scoring witness does not have exactly one move.");
    require(moves.values[0].score == 0,
            "A sole legal edge unexpectedly ran heuristic move scoring.");
    const std::vector<ps::Move> legal = ps::legal_moves(state);
    require(std::find(legal.begin(), legal.end(), moves.values[0].move) !=
                legal.end(),
            "Sole-edge bypass returned an illegal move.");
  }
}

struct FixedNodeDecision {
  std::string action;
  cg::SearchStats stats;
};

FixedNodeDecision configured_decision(const ps::GameState &state,
                                      cg::SearchConfig config) {
  cg::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  ps::GameState replay = state;
  std::string action;
  for (const ps::Move move : moves) {
    action.push_back(cg::encode_direction(replay.ball, move.to));
    replay = ps::apply_move(replay, move);
  }
  return FixedNodeDecision{action, search.stats()};
}

FixedNodeDecision fixed_node_decision(const ps::GameState &state,
                                      std::uint64_t max_nodes) {
  cg::SearchConfig config;
  config.max_nodes = max_nodes;
  config.max_time_ms = 0;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  return configured_decision(state, config);
}

FixedNodeDecision fixed_depth_decision(const ps::GameState &state,
                                       cg::SearchConfig config,
                                       std::uint32_t depth) {
  cg::CompleteTurnSearch search(state, config);
  const cg::RootResult result = search.run_fixed_depth_for_test(depth);
  ps::GameState replay = state;
  std::string action;
  for (const ps::Move move : result.action) {
    action.push_back(cg::encode_direction(replay.ball, move.to));
    replay = ps::apply_move(replay, move);
  }
  return FixedNodeDecision{action, search.stats()};
}

bool same_stats(const cg::SearchStats &left, const cg::SearchStats &right) {
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
         left.rebound_goal_probes == right.rebound_goal_probes &&
         left.rebound_goal_hits == right.rebound_goal_hits &&
         left.rebound_loss_hits == right.rebound_loss_hits &&
         left.root_rebound_probes == right.root_rebound_probes &&
         left.root_rebound_win_hits == right.root_rebound_win_hits &&
         left.root_rebound_loss_hits == right.root_rebound_loss_hits &&
         left.leaf_rebound_probes == right.leaf_rebound_probes &&
         left.leaf_rebound_win_hits == right.leaf_rebound_win_hits &&
         left.leaf_rebound_loss_hits == right.leaf_rebound_loss_hits &&
         left.exchange_ply1_probes == right.exchange_ply1_probes &&
         left.exchange_ply1_win_hits == right.exchange_ply1_win_hits &&
         left.exchange_ply1_loss_hits == right.exchange_ply1_loss_hits &&
         left.exchange_ply1_cutoffs == right.exchange_ply1_cutoffs &&
         left.exchange_ply2_probes == right.exchange_ply2_probes &&
         left.exchange_ply2_win_hits == right.exchange_ply2_win_hits &&
         left.exchange_ply2_loss_hits == right.exchange_ply2_loss_hits &&
         left.exchange_ply2_cutoffs == right.exchange_ply2_cutoffs &&
         left.forced_edges == right.forced_edges &&
         left.root_seed_actions == right.root_seed_actions &&
         left.root_transposition_reuses == right.root_transposition_reuses &&
         left.max_action_edges == right.max_action_edges &&
         left.root_score == right.root_score &&
         left.budget_exhausted == right.budget_exhausted;
}

bool proof_stats_are_zero(const cg::SearchStats &stats) {
  return stats.rebound_goal_probes == 0 && stats.rebound_goal_hits == 0 &&
         stats.rebound_loss_hits == 0 && stats.root_rebound_probes == 0 &&
         stats.root_rebound_win_hits == 0 &&
         stats.root_rebound_loss_hits == 0 &&
         stats.leaf_rebound_probes == 0 &&
         stats.leaf_rebound_win_hits == 0 &&
         stats.leaf_rebound_loss_hits == 0 &&
         stats.exchange_ply1_probes == 0 &&
         stats.exchange_ply1_win_hits == 0 &&
         stats.exchange_ply1_loss_hits == 0 &&
         stats.exchange_ply1_cutoffs == 0 &&
         stats.exchange_ply2_probes == 0 &&
         stats.exchange_ply2_win_hits == 0 &&
         stats.exchange_ply2_loss_hits == 0 &&
         stats.exchange_ply2_cutoffs == 0;
}

bool proof_stats_are_consistent(const cg::SearchStats &stats) {
  return stats.rebound_goal_probes ==
             stats.root_rebound_probes + stats.leaf_rebound_probes +
                 stats.exchange_ply1_probes +
                 stats.exchange_ply2_probes &&
         stats.rebound_goal_hits ==
             stats.root_rebound_win_hits + stats.leaf_rebound_win_hits +
                 stats.exchange_ply1_win_hits +
                 stats.exchange_ply2_win_hits &&
         stats.rebound_loss_hits ==
             stats.root_rebound_loss_hits + stats.leaf_rebound_loss_hits +
                 stats.exchange_ply1_loss_hits +
                 stats.exchange_ply2_loss_hits;
}

bool disabled_scope_stats_are_zero(const cg::SearchStats &stats,
                                   std::uint8_t mask) {
  const bool root_zero =
      stats.root_rebound_probes == 0 && stats.root_rebound_win_hits == 0 &&
      stats.root_rebound_loss_hits == 0;
  const bool leaf_zero =
      stats.leaf_rebound_probes == 0 && stats.leaf_rebound_win_hits == 0 &&
      stats.leaf_rebound_loss_hits == 0;
  const bool ply_one_zero =
      stats.exchange_ply1_probes == 0 &&
      stats.exchange_ply1_win_hits == 0 &&
      stats.exchange_ply1_loss_hits == 0 &&
      stats.exchange_ply1_cutoffs == 0;
  const bool ply_two_zero =
      stats.exchange_ply2_probes == 0 &&
      stats.exchange_ply2_win_hits == 0 &&
      stats.exchange_ply2_loss_hits == 0 &&
      stats.exchange_ply2_cutoffs == 0;
  return ((mask & cg::kExactProofRootGoal) != 0 || root_zero) &&
         ((mask & cg::kExactProofLeafBoundary) != 0 || leaf_zero) &&
         ((mask & cg::kExactProofPlyOne) != 0 || ply_one_zero) &&
         ((mask & cg::kExactProofPlyTwo) != 0 || ply_two_zero);
}

void mover_relative_ties_are_rotation_equivariant() {
  struct Witness {
    std::string_view transcript;
    std::uint64_t max_nodes;
  };
  constexpr std::array<Witness, 3> witnesses{{
      {"6/1", 1},
      {"4/3/6/4/3/0", 16},
      {"4/3/6/4/3/0", 64},
  }};

  for (const Witness &witness : witnesses) {
    const ps::GameState state = reconstruct(witness.transcript);
    const ps::GameState rotated = rotate_and_swap(state);
    const FixedNodeDecision original =
        fixed_node_decision(state, witness.max_nodes);
    const FixedNodeDecision transformed =
        fixed_node_decision(rotated, witness.max_nodes);
    require(rotate_action(original.action) == transformed.action,
            "A paired fixed-node action changed under player rotation.");
    require(original.stats.root_score == -transformed.stats.root_score,
            "A paired fixed-node root score did not negate under rotation.");
    require(original.stats.nodes == transformed.stats.nodes &&
                original.stats.completed_turn_depth ==
                    transformed.stats.completed_turn_depth &&
                original.stats.attempted_turn_depth ==
                    transformed.stats.attempted_turn_depth &&
                original.stats.completed_actions ==
                    transformed.stats.completed_actions &&
                original.stats.cutoffs == transformed.stats.cutoffs &&
                original.stats.transposition_hits ==
                    transformed.stats.transposition_hits,
            "A paired fixed-node traversal changed under player rotation.");
  }

  const FixedNodeDecision fallback =
      fixed_node_decision(reconstruct("6/1"), 1);
  const FixedNodeDecision rotated_fallback =
      fixed_node_decision(rotate_and_swap(reconstruct("6/1")), 1);
  require(fallback.action == "47" && rotated_fallback.action == "03",
          "The frozen horizontal-tie witness no longer selects its canonical "
          "paired fallbacks.");
}

void exact_proof_disabled_is_strict_parity() {
  constexpr std::array<std::string_view, 8> transcripts{{
      "",
      "0/6",
      "6/1",
      "0/6/5/4/5/53/61/0633",
      "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35",
      "0/2/7/45/7/5/71/34/2212/2/7/1/6/1/03636074/33535",
      "7/6/7/53/10/34/71/45/221/2/1/35/70/54/17/43/660/33",
      "0/0/3/67/27/45/5/2/5/6143/5/717271/1/7/532/27412/41/654",
  }};
  constexpr std::array<std::uint64_t, 4> budgets{{1, 251, 2'000,
                                                   10'000}};

  for (const std::string_view transcript : transcripts) {
    const ps::GameState state = reconstruct(transcript);
    for (const std::uint64_t budget : budgets) {
      cg::SearchConfig implicit;
      implicit.max_nodes = budget;
      implicit.max_time_ms = 0;
      implicit.replay_value_blend_percent = 15;
      implicit.teacher_residual_weight_percent = 100;
      cg::SearchConfig explicit_off = implicit;
      explicit_off.exact_proof_mask = 0;
      const FixedNodeDecision default_result =
          configured_decision(state, implicit);
      const FixedNodeDecision off_result =
          configured_decision(state, explicit_off);
      require(default_result.action == off_result.action &&
                  same_stats(default_result.stats, off_result.stats) &&
                  proof_stats_are_zero(off_result.stats),
              "Explicitly disabled proof changed a default-off decision.");
    }
  }
}

void exact_root_rebound_goal_is_safe_and_symmetric() {
  const ps::Point north_root{4, 2};
  const ps::Point north_mouth{4, 1};
  ps::GameState north = make_clean_state_at(north_root);
  north.visit_count[north_mouth] = 1;
  block_edges_except(north, north_root, {north_mouth});

  cg::SearchConfig config;
  config.max_nodes = 1;
  config.max_time_ms = 0;
  config.exact_proof_mask = cg::kExactProofRootGoal;
  const FixedNodeDecision original = configured_decision(north, config);
  const ps::GameState south = rotate_and_swap(north);
  const FixedNodeDecision rotated = configured_decision(south, config);
  require(original.action.size() == 2 &&
              rotate_action(original.action) == rotated.action &&
              original.stats.root_score == -rotated.stats.root_score &&
              original.stats.rebound_goal_hits == 1 &&
              rotated.stats.rebound_goal_hits == 1,
          "The exact root rebound goal must rotate and swap exactly.");

  ps::GameState north_result = north;
  cg::apply_encoded_turn(north_result, original.action);
  ps::GameState south_result = south;
  cg::apply_encoded_turn(south_result, rotated.action);
  require(ps::winner(north_result) == ps::Player::One &&
              ps::winner(south_result) == ps::Player::Two,
          "Each exact root witness must replay to the mover's goal.");

  ps::GameState fresh = make_clean_state_at(north_root);
  block_edges_except(fresh, north_root, {north_mouth});
  const FixedNodeDecision unknown = configured_decision(fresh, config);
  require(unknown.action == "0" && unknown.stats.rebound_goal_hits == 0 &&
              unknown.stats.budget_exhausted,
          "A fresh mouth must remain Unknown and use the safe fallback.");

  ps::GameState own_goal = make_clean_state_at(north_root,
                                               ps::Player::Two);
  own_goal.visit_count[north_mouth] = 1;
  block_edges_except(own_goal, north_root, {north_mouth});
  const FixedNodeDecision own = configured_decision(own_goal, config);
  require(!own.action.empty() && own.stats.rebound_goal_hits == 0,
          "A route to the mover's own goal must not be a Win proof.");
}

void exact_leaf_and_exchange_proofs_are_symmetric() {
  for (const ps::Player mover : {ps::Player::One, ps::Player::Two}) {
    cg::SearchConfig config;
    config.max_nodes = 1'000'000;
    config.max_time_ms = 0;
    config.root_seed_endpoints = false;
    config.exact_proof_mask = cg::kExactProofLeafBoundary;

    const FixedNodeDecision leaf =
        fixed_depth_decision(layered_goal_choice(mover), config, 1);
    require(leaf.stats.completed_turn_depth == 1 &&
                leaf.stats.leaf_rebound_probes > 0 &&
                leaf.stats.leaf_rebound_win_hits > 0 &&
                leaf.stats.root_rebound_probes == 0 &&
                leaf.stats.exchange_ply1_probes == 0 &&
                leaf.stats.exchange_ply2_probes == 0,
            "The isolated depth-zero boundary proof did not classify.");

    config.exact_proof_mask = cg::kExactProofPlyOne;

    const FixedNodeDecision win =
        fixed_depth_decision(layered_goal_choice(mover), config, 2);
    require(win.stats.completed_turn_depth == 2 &&
                win.stats.exchange_ply1_win_hits > 0 &&
                win.stats.exchange_ply1_cutoffs ==
                    win.stats.exchange_ply1_win_hits +
                        win.stats.exchange_ply1_loss_hits,
            "The first-reply Win proof did not classify and cut off.");

    const FixedNodeDecision loss =
        fixed_depth_decision(root_child_loss_choice(mover), config, 2);
    const std::string expected_loss_action =
        mover == ps::Player::One ? "1" : "5";
    const int expected_loss_score =
        mover == ps::Player::One ? 999'998 : -999'998;
    require(loss.action == expected_loss_action &&
                loss.stats.root_score == expected_loss_score &&
                loss.stats.exchange_ply1_loss_hits > 0 &&
                loss.stats.exchange_ply1_cutoffs ==
                    loss.stats.exchange_ply1_win_hits +
                        loss.stats.exchange_ply1_loss_hits,
            "The first-reply Loss proof did not preserve the exact mate.");

    config.exact_proof_mask = cg::kExactProofPlyTwo;
    for (const bool is_win : {true, false}) {
      const FixedNodeDecision exchange = fixed_depth_decision(
          two_ply_exchange_component(mover, is_win), config, 3);
      const std::string expected = mover == ps::Player::One ? "0" : "4";
      const int expected_score =
          (mover == ps::Player::One) == is_win ? 999'997 : -999'997;
      require(exchange.action == expected &&
                  exchange.stats.root_score == expected_score &&
                  exchange.stats.exchange_ply1_probes == 0 &&
                  exchange.stats.exchange_ply2_probes > 0 &&
                  exchange.stats.exchange_ply2_cutoffs ==
                      exchange.stats.exchange_ply2_win_hits +
                          exchange.stats.exchange_ply2_loss_hits &&
                  (is_win ? exchange.stats.exchange_ply2_win_hits > 0
                          : exchange.stats.exchange_ply2_loss_hits > 0),
              "The counterturn proof did not preserve the exact mate.");
    }
  }
}

void every_exact_proof_mask_is_legal_and_symmetric() {
  ps::GameState root_goal = make_clean_state_at({4, 2});
  root_goal.visit_count[{4, 1}] = 1;
  block_edges_except(root_goal, {4, 2}, {{4, 1}});
  const std::vector<ps::GameState> witnesses{
      root_goal,
      layered_goal_choice(ps::Player::One),
      root_child_loss_choice(ps::Player::One),
      two_ply_exchange_component(ps::Player::One, true),
      two_ply_exchange_component(ps::Player::One, false),
      reconstruct("0/6/5/4/5/53/61/0633"),
  };

  for (std::uint8_t mask = 0; mask <= cg::kAllExactProofs; ++mask) {
    for (const ps::GameState &state : witnesses) {
      cg::SearchConfig config;
      config.max_nodes = 4'000;
      config.max_time_ms = 0;
      config.exact_proof_mask = mask;
      config.replay_value_blend_percent = 15;
      config.teacher_residual_weight_percent = 100;
      const FixedNodeDecision original = configured_decision(state, config);
      const ps::GameState rotated_state = rotate_and_swap(state);
      const FixedNodeDecision rotated =
          configured_decision(rotated_state, config);
      require(rotate_action(original.action) == rotated.action &&
                  original.stats.root_score == -rotated.stats.root_score,
              "An exact-proof mask changed under player rotation.");

      ps::GameState original_after = state;
      cg::apply_encoded_turn(original_after, original.action);
      ps::GameState rotated_after = rotated_state;
      cg::apply_encoded_turn(rotated_after, rotated.action);
      require(proof_stats_are_consistent(original.stats) &&
                  proof_stats_are_consistent(rotated.stats) &&
                  disabled_scope_stats_are_zero(original.stats, mask) &&
                  disabled_scope_stats_are_zero(rotated.stats, mask),
              "An exact-proof mask leaked work or miscounted its scope.");
    }
  }

  cg::SearchConfig invalid;
  invalid.exact_proof_mask = static_cast<std::uint8_t>(
      cg::kAllExactProofs + 1U);
  require_invalid_argument(
      [&] { cg::CompleteTurnSearch search(witnesses.front(), invalid); },
      "An unknown exact-proof mask bit must be rejected.");
}

void teacher_residual_is_player_rotation_invariant() {
  const ps::GameState state = reconstruct("0/6/5/4/5/53/61/0633");
  const ps::GameState rotated = rotate_and_swap(state);
  cg::SearchConfig config;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  cg::CompleteTurnSearch original_search(state, config);
  cg::CompleteTurnSearch rotated_search(rotated, config);
  const cg::EvaluationSnapshot original =
      original_search.evaluation_snapshot();
  const cg::EvaluationSnapshot transformed =
      rotated_search.evaluation_snapshot();
  require(original.anchor_score == -transformed.anchor_score,
          "Player rotation should negate the absolute anchor score.");
  require(original.mover_sign == -transformed.mover_sign,
          "Player rotation should swap the mover sign.");
  for (std::size_t index = 0; index < original.features.size(); ++index) {
    require(std::abs(original.features[index] - transformed.features[index]) <
                1e-5F,
            "Player rotation changed a mover-relative residual feature.");
  }
}

void teacher_residual_obeys_the_root_phase_gate() {
  cg::SearchConfig config;
  config.max_nodes = 2'000;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;

  cg::CompleteTurnSearch opening_search(reconstruct("0/6"), config);
  (void)opening_search.run();
  require(opening_search.stats().teacher_residual_evaluations == 0,
          "The residual must stay disabled in verified opening states.");

  cg::CompleteTurnSearch developed_search(
      reconstruct("0/6/5/4/5/53/61/0633"), config);
  (void)developed_search.run();
  require(developed_search.stats().teacher_residual_evaluations > 0,
          "The residual should participate after the phase threshold.");
}

void fixed_node_search_is_deterministic() {
  const ps::GameState state = reconstruct("0/6/5/4/5/53/61/0633");
  cg::SearchConfig config;
  config.max_nodes = 5'000;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  cg::CompleteTurnSearch first(state, config);
  cg::CompleteTurnSearch second(state, config);
  const std::vector<ps::Move> first_moves = first.run();
  const std::vector<ps::Move> second_moves = second.run();
  require(first_moves == second_moves &&
              first.stats().nodes == second.stats().nodes &&
              first.stats().completed_turn_depth ==
                  second.stats().completed_turn_depth &&
              first.stats().root_score == second.stats().root_score,
          "Fixed-node residual search must be deterministic.");
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
      {"mover_relative_ties_are_rotation_equivariant",
       mover_relative_ties_are_rotation_equivariant},
      {"exact_proof_disabled_is_strict_parity",
       exact_proof_disabled_is_strict_parity},
      {"exact_root_rebound_goal_is_safe_and_symmetric",
       exact_root_rebound_goal_is_safe_and_symmetric},
      {"sole_legal_edge_bypasses_move_scoring",
       sole_legal_edge_bypasses_move_scoring},
      {"exact_leaf_and_exchange_proofs_are_symmetric",
       exact_leaf_and_exchange_proofs_are_symmetric},
      {"every_exact_proof_mask_is_legal_and_symmetric",
       every_exact_proof_mask_is_legal_and_symmetric},
      {"teacher_residual_is_player_rotation_invariant",
       teacher_residual_is_player_rotation_invariant},
      {"teacher_residual_obeys_the_root_phase_gate",
       teacher_residual_obeys_the_root_phase_gate},
      {"fixed_node_search_is_deterministic",
       fixed_node_search_is_deterministic},
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
