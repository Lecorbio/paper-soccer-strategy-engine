#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING
#include "../submissions/codingame/bots/rank_4_jacek_hybrid/submission.cpp"
#undef PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING

#include <array>
#include <bit>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

struct OracleResult {
  cg::ReboundOutcome outcome{cg::ReboundOutcome::Unknown};
  std::set<ps::Point> safe_endpoints;
  std::uint32_t raw_safe_arcs{};
};

struct CandidateResult {
  cg::ReboundOutcome outcome{cg::ReboundOutcome::Unknown};
  std::uint32_t safe_endpoints{};
  cg::SearchStats stats{};
};

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

ps::RulesConfig rules(ps::BlockedRule blocked = ps::BlockedRule::MoverLoses) {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed, blocked};
}

ps::GameState clean_state(ps::Point ball, ps::Player mover,
                          ps::BlockedRule blocked =
                              ps::BlockedRule::MoverLoses) {
  ps::GameState state = ps::make_initial_state(rules(blocked));
  state.ball = ball;
  state.to_move = mover;
  state.status = ps::Status::InProgress;
  state.path = {ball};
  state.used_segments.clear();
  state.visit_count.clear();
  state.visit_count[ball] = 1;
  return state;
}

bool was_visited(const ps::GameState &state, ps::Point point) {
  const auto found = state.visit_count.find(point);
  return found != state.visit_count.end() && found->second > 0;
}

bool edge_available(const ps::GameState &state, ps::Point from,
                    ps::Point to) {
  const ps::Segment edge{from, to};
  return !ps::is_forbidden_boundary_segment(state.config, edge) &&
         !state.used_segments.contains(edge);
}

OracleResult oracle_rebound_component(const ps::GameState &state) {
  OracleResult result;
  const ps::Player mover = state.to_move;
  std::set<ps::Point> component_seen{state.ball};
  std::vector<ps::Point> queue{state.ball};
  std::size_t head = 0;
  while (head < queue.size()) {
    const ps::Point vertex = queue[head++];
    for (const ps::Point destination :
         ps::neighbors(state.config, vertex, mover)) {
      if (!edge_available(state, vertex, destination)) {
        continue;
      }
      if (ps::is_attacking_goal(state.config, destination, mover)) {
        result.outcome = cg::ReboundOutcome::Win;
        return result;
      }
      if (ps::is_goal_point(state.config, destination)) {
        continue;
      }
      const bool continues =
          ps::is_boundary_point(state.config, destination) ||
          was_visited(state, destination);
      if (!continues) {
        const ps::Segment incoming{vertex, destination};
        bool safe = false;
        for (const ps::Point child :
             ps::neighbors(state.config, destination, ps::opponent(mover))) {
          const ps::Segment reply{destination, child};
          if (reply == incoming ||
              ps::is_forbidden_boundary_segment(state.config, reply) ||
              state.used_segments.contains(reply)) {
            continue;
          }
          safe = true;
          break;
        }
        if (safe) {
          ++result.raw_safe_arcs;
          result.safe_endpoints.insert(destination);
        }
        continue;
      }
      if (component_seen.insert(destination).second) {
        queue.push_back(destination);
      }
    }
  }
  if (result.safe_endpoints.empty() &&
      state.config.blocked_rule == ps::BlockedRule::MoverLoses) {
    result.outcome = cg::ReboundOutcome::Loss;
  }
  return result;
}

CandidateResult candidate_rebound_component(const ps::GameState &state) {
  cg::SearchConfig config;
  config.max_nodes = 1;
  config.max_time_ms = 0;
  config.exact_proof_mask = cg::kExactProofLeafBoundary;
  cg::CompleteTurnSearch search(state, config);
  std::vector<ps::Move> ignored;
  std::uint32_t safe_endpoints = 0;
  const cg::ReboundOutcome outcome =
      search.analyze_rebound_component_for_test(false, ignored,
                                                &safe_endpoints);
  return {outcome, safe_endpoints, search.stats()};
}

void require_matches_oracle(const ps::GameState &state,
                            std::string_view label) {
  const OracleResult oracle = oracle_rebound_component(state);
  const CandidateResult candidate = candidate_rebound_component(state);
  require(candidate.outcome == oracle.outcome,
          std::string(label) + ": proof outcome disagrees with oracle");
  if (oracle.outcome != cg::ReboundOutcome::Win) {
    require(candidate.safe_endpoints == oracle.safe_endpoints.size(),
            std::string(label) + ": unique frontier count disagrees");
  }
  require(candidate.stats.rebound_goal_probes == 1,
          std::string(label) + ": frontier must reuse exactly one proof scan");
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
  for (const ps::Segment edge : state.used_segments) {
    rotated.used_segments.insert(
        ps::Segment{rotate_point(edge.a), rotate_point(edge.b)});
  }
  rotated.visit_count.clear();
  for (const auto &[point, visits] : state.visit_count) {
    rotated.visit_count[rotate_point(point)] = visits;
  }
  return rotated;
}

void exhaustive_local_oracle_agreement() {
  constexpr std::array<ps::Point, 8> neighbors{{
      {4, 5}, {5, 5}, {5, 6}, {5, 7},
      {4, 7}, {3, 7}, {3, 6}, {3, 5},
  }};
  std::uint32_t checked = 0;
  std::uint32_t wins = 0;
  std::uint32_t losses = 0;
  std::uint32_t unknowns = 0;
  for (const ps::BlockedRule blocked : {ps::BlockedRule::MoverLoses,
                                       ps::BlockedRule::PlayerToMoveLoses}) {
    for (const ps::Player mover : {ps::Player::One, ps::Player::Two}) {
      for (int encoded = 0; encoded < 81; ++encoded) {
        ps::GameState state = clean_state({4, 6}, mover, blocked);
        int trits = encoded;
        for (std::size_t index = 0; index < neighbors.size(); ++index) {
          const int state_code = index < 4 ? trits % 3 : 0;
          if (index < 4) {
            trits /= 3;
          }
          if (state_code == 0) {
            state.used_segments.insert(ps::Segment{{4, 6}, neighbors[index]});
            state.visit_count[neighbors[index]] = 1;
          } else if (state_code == 1) {
            state.visit_count[neighbors[index]] = 1;
          }
        }
        require_matches_oracle(state, "local trit state");
        const CandidateResult result = candidate_rebound_component(state);
        wins += result.outcome == cg::ReboundOutcome::Win;
        losses += result.outcome == cg::ReboundOutcome::Loss;
        unknowns += result.outcome == cg::ReboundOutcome::Unknown;
        ++checked;

        const ps::GameState rotated = rotate_and_swap(state);
        require_matches_oracle(rotated, "rotated local trit state");
        const CandidateResult paired = candidate_rebound_component(rotated);
        require(result.outcome == paired.outcome &&
                    result.safe_endpoints == paired.safe_endpoints,
                "Local frontier proof did not rotate and swap exactly");
        ++checked;
      }
    }
  }
  require(checked == 648 && wins == 0 && losses > 0 && unknowns > 0,
          "The exhaustive local matrix lost classification coverage");
}

void exhaustive_goal_oracle_agreement() {
  std::uint32_t checked = 0;
  std::uint32_t wins = 0;
  std::uint32_t losses = 0;
  for (const ps::BlockedRule blocked : {ps::BlockedRule::MoverLoses,
                                       ps::BlockedRule::PlayerToMoveLoses}) {
    for (const ps::Player mover : {ps::Player::One, ps::Player::Two}) {
      for (unsigned goal_mask = 0; goal_mask < 8; ++goal_mask) {
        ps::GameState state = clean_state({4, 1}, mover, blocked);
        for (const ps::Point destination :
             ps::neighbors(state.config, state.ball, mover)) {
          if (ps::is_goal_point(state.config, destination)) {
            const unsigned bit = static_cast<unsigned>(destination.x - 3);
            if ((goal_mask & (1U << bit)) != 0) {
              state.used_segments.insert(
                  ps::Segment{state.ball, destination});
            }
          } else {
            state.used_segments.insert(ps::Segment{state.ball, destination});
            state.visit_count[destination] = 1;
          }
        }
        require_matches_oracle(state, "goal mask state");
        const CandidateResult result = candidate_rebound_component(state);
        wins += result.outcome == cg::ReboundOutcome::Win;
        losses += result.outcome == cg::ReboundOutcome::Loss;
        ++checked;
      }
    }
  }
  require(checked == 32 && wins > 0 && losses > 0,
          "The goal matrix lost Win/Loss coverage");
}

ps::GameState duplicate_frontier_state(ps::Player mover) {
  ps::GameState state = clean_state({4, 6}, mover);
  state.visit_count[{3, 5}] = 1;
  state.visit_count[{5, 5}] = 1;
  return state;
}

void duplicate_frontier_is_counted_once() {
  for (const ps::Player mover : {ps::Player::One, ps::Player::Two}) {
    const ps::GameState state = duplicate_frontier_state(mover);
    const OracleResult oracle = oracle_rebound_component(state);
    const CandidateResult candidate = candidate_rebound_component(state);
    require(oracle.outcome == cg::ReboundOutcome::Unknown &&
                oracle.raw_safe_arcs > oracle.safe_endpoints.size(),
            "The dedup witness no longer has converging frontier arcs");
    require(candidate.safe_endpoints == oracle.safe_endpoints.size(),
            "A converging fresh endpoint was counted more than once");

    const CandidateResult paired =
        candidate_rebound_component(rotate_and_swap(state));
    require(candidate.outcome == paired.outcome &&
                candidate.safe_endpoints == paired.safe_endpoints,
            "Deduplication changed under rotation");
  }
}

int leaf_score(const ps::GameState &state, cg::SearchConfig config,
               cg::SearchStats *stats = nullptr) {
  cg::CompleteTurnSearch search(state, config);
  const int score = search.leaf_score_for_test(5);
  if (stats != nullptr) {
    *stats = search.stats();
  }
  return score;
}

void unknown_leaf_bonus_is_exact_and_mover_relative() {
  for (const int replay_blend : {0, 15, 100}) {
    for (const ps::Player mover : {ps::Player::One, ps::Player::Two}) {
      const ps::GameState state = duplicate_frontier_state(mover);
      const CandidateResult frontier = candidate_rebound_component(state);
      require(frontier.outcome == cg::ReboundOutcome::Unknown,
              "Bonus witness must remain Unknown");

      cg::SearchConfig baseline;
      baseline.max_nodes = 100;
      baseline.max_time_ms = 0;
      baseline.exact_proof_mask = 0;
      baseline.replay_value_blend_percent = replay_blend;
      baseline.teacher_residual_weight_percent = 100;
      cg::SearchConfig candidate = baseline;
      candidate.exact_proof_mask = cg::kExactProofLeafBoundary;

      const int unadjusted = leaf_score(state, baseline);
      cg::SearchStats candidate_stats;
      const int adjusted = leaf_score(state, candidate, &candidate_stats);
      const int sign = mover == ps::Player::One ? 1 : -1;
      const int expected_delta =
          sign * static_cast<int>(frontier.safe_endpoints) *
          cg::kSafeHandoffFrontierWeight * (100 - replay_blend) / 100;
      require(adjusted ==
                  std::clamp(unadjusted + expected_delta,
                             -cg::kMaximumEvaluation,
                             cg::kMaximumEvaluation),
              "Unknown leaf did not receive the exact scaled +10 bonus");
      require(candidate_stats.leaf_rebound_probes == 1 &&
                  candidate_stats.leaf_rebound_win_hits == 0 &&
                  candidate_stats.leaf_rebound_loss_hits == 0 &&
                  candidate_stats.rebound_goal_probes == 1 &&
                  candidate_stats.leaf_evaluations == 1,
              "Unknown leaf bonus escaped its one-scan proof boundary");

      const ps::GameState rotated = rotate_and_swap(state);
      const int rotated_adjusted = leaf_score(rotated, candidate);
      require(adjusted == -rotated_adjusted,
              "Adjusted leaf score did not negate under rotation");
    }
  }
}

ps::GameState exact_win_state() {
  ps::GameState state = clean_state({4, 2}, ps::Player::One);
  state.visit_count[{4, 1}] = 1;
  return state;
}

ps::GameState exact_loss_state() {
  const ps::Point root{4, 6};
  const ps::Point rebound{4, 5};
  ps::GameState state = clean_state(root, ps::Player::One);
  state.visit_count[rebound] = 1;
  for (const ps::Point destination :
       ps::neighbors(state.config, root, state.to_move)) {
    if (destination != rebound) {
      state.used_segments.insert(ps::Segment{root, destination});
      state.visit_count[destination] = 1;
    }
  }
  for (const ps::Point destination :
       ps::neighbors(state.config, rebound, state.to_move)) {
    if (destination != root) {
      state.used_segments.insert(ps::Segment{rebound, destination});
      state.visit_count[destination] = 1;
    }
  }
  return state;
}

void exact_mates_bypass_the_feature() {
  cg::SearchConfig config;
  config.max_nodes = 100;
  config.max_time_ms = 0;
  config.exact_proof_mask = cg::kExactProofLeafBoundary;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;

  std::size_t witness = 0;
  for (const auto &[state, expected] :
       std::array<std::pair<ps::GameState, int>, 4>{{
           {exact_win_state(), cg::kMateScore - 6},
           {rotate_and_swap(exact_win_state()), -cg::kMateScore + 6},
           {exact_loss_state(), -cg::kMateScore + 6},
           {rotate_and_swap(exact_loss_state()), cg::kMateScore - 6},
       }}) {
    cg::SearchStats stats;
    const int actual = leaf_score(state, config, &stats);
    require(actual == expected,
            "Frontier feature changed exact mate witness " +
                std::to_string(witness) + " (actual=" +
                std::to_string(actual) + ", expected=" +
                std::to_string(expected) + ")");
    require(stats.leaf_evaluations == 0 &&
                stats.leaf_rebound_probes == 1 &&
                stats.leaf_rebound_win_hits + stats.leaf_rebound_loss_hits ==
                    1,
            "Exact mate reached heuristic evaluation");
    ++witness;
  }
}

void adjusted_leaf_value_is_cached_after_proof() {
  const ps::GameState state = duplicate_frontier_state(ps::Player::One);
  cg::SearchConfig config;
  config.max_nodes = 100;
  config.max_time_ms = 0;
  config.transposition_entries = 0;
  config.evaluation_entries = 64;
  config.exact_proof_mask = cg::kExactProofLeafBoundary;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  cg::CompleteTurnSearch search(state, config);
  const int first = search.leaf_score_for_test(3);
  const cg::SearchStats after_first = search.stats();
  const int second = search.leaf_score_for_test(3);
  const cg::SearchStats after_second = search.stats();
  require(first == second,
          "Evaluation cache changed the adjusted frontier value");
  require(after_first.leaf_rebound_probes == 1 &&
              after_second.leaf_rebound_probes == 1 &&
              after_second.evaluation_cache_hits ==
                  after_first.evaluation_cache_hits + 1 &&
              after_second.leaf_evaluations == after_first.leaf_evaluations,
          "Cache hit repeated the proof scan or heuristic evaluation");
}

bool actions_rotate(const std::vector<ps::Move> &left,
                    const std::vector<ps::Move> &right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (rotate_point(left[index].to) != right[index].to) {
      return false;
    }
  }
  return true;
}

void transposition_and_fixed_work_are_stable() {
  const ps::GameState state = duplicate_frontier_state(ps::Player::One);
  cg::SearchConfig config;
  config.max_nodes = 200'000;
  config.max_time_ms = 0;
  config.transposition_entries = 4'096;
  config.evaluation_entries = 2'048;
  config.root_seed_endpoints = false;
  config.exact_proof_mask = cg::kExactProofLeafBoundary;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;

  cg::CompleteTurnSearch reused(state, config);
  const cg::RootResult first = reused.run_fixed_depth_for_test(2);
  const cg::SearchStats after_first = reused.stats();
  const cg::RootResult second = reused.run_fixed_depth_for_test(2);
  const cg::SearchStats after_second = reused.stats();
  require(first.score == second.score && first.action == second.action &&
              after_second.transposition_hits >
                  after_first.transposition_hits &&
              after_second.transposition_cutoffs >
                  after_first.transposition_cutoffs,
          "TT reuse changed the frontier score/action or failed to activate");

  cg::CompleteTurnSearch fresh(state, config);
  const cg::RootResult repeated = fresh.run_fixed_depth_for_test(2);
  require(first.score == repeated.score && first.action == repeated.action &&
              fresh.stats().nodes == after_first.nodes &&
              fresh.stats().leaf_rebound_probes ==
                  after_first.leaf_rebound_probes,
          "Fresh fixed-depth frontier work was not deterministic");

  cg::SearchConfig fixed_nodes = config;
  fixed_nodes.max_nodes = 4'000;
  cg::CompleteTurnSearch left(state, fixed_nodes);
  cg::CompleteTurnSearch left_again(state, fixed_nodes);
  const std::vector<ps::Move> left_action = left.run();
  const std::vector<ps::Move> repeated_action = left_again.run();
  require(left_action == repeated_action &&
              left.stats().root_score == left_again.stats().root_score &&
              left.stats().nodes == left_again.stats().nodes &&
              left.stats().completed_turn_depth ==
                  left_again.stats().completed_turn_depth &&
              left.stats().leaf_rebound_probes ==
                  left_again.stats().leaf_rebound_probes,
          "Fixed-node frontier search was not deterministic");

  const ps::GameState rotated_state = rotate_and_swap(state);
  cg::CompleteTurnSearch right(rotated_state, fixed_nodes);
  const std::vector<ps::Move> right_action = right.run();
  require(actions_rotate(left_action, right_action) &&
              left.stats().root_score == -right.stats().root_score &&
              left.stats().nodes == right.stats().nodes &&
              left.stats().completed_turn_depth ==
                  right.stats().completed_turn_depth,
          "Fixed-node frontier search changed under rotation");
}

}  // namespace

int main() {
  struct Test {
    const char *name;
    void (*function)();
  };
  constexpr std::array tests{
      Test{"exhaustive_local_oracle_agreement",
           exhaustive_local_oracle_agreement},
      Test{"exhaustive_goal_oracle_agreement",
           exhaustive_goal_oracle_agreement},
      Test{"duplicate_frontier_is_counted_once",
           duplicate_frontier_is_counted_once},
      Test{"unknown_leaf_bonus_is_exact_and_mover_relative",
           unknown_leaf_bonus_is_exact_and_mover_relative},
      Test{"exact_mates_bypass_the_feature", exact_mates_bypass_the_feature},
      Test{"adjusted_leaf_value_is_cached_after_proof",
           adjusted_leaf_value_is_cached_after_proof},
      Test{"transposition_and_fixed_work_are_stable",
           transposition_and_fixed_work_are_stable},
  };
  int failures = 0;
  for (const Test &test : tests) {
    try {
      test.function();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": " << error.what() << '\n';
    }
  }
  std::cout << (tests.size() - static_cast<std::size_t>(failures)) << '/'
            << tests.size() << " frontier semantic tests passed.\n";
  return failures == 0 ? 0 : 1;
}
