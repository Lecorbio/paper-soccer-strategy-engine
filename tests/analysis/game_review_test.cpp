#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../src/analysis/exact_endgame_internal.hpp"
#include "papersoccer/game_review.hpp"
#include "papersoccer/geometry.hpp"
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

bool contains_segment(const std::vector<ps::Segment> &segments,
                      ps::Segment candidate) {
  return std::find(segments.begin(), segments.end(), candidate) !=
         segments.end();
}

void isolate_unused_edges(ps::GameState &state,
                          const std::vector<ps::Segment> &allowed) {
  state.used_segments.clear();
  for (int y = 1; y <= state.config.height + 1; ++y) {
    for (int x = 0; x <= state.config.width; ++x) {
      const ps::Point from{x, y};
      for (const ps::Player player : {ps::Player::One, ps::Player::Two}) {
        for (const ps::Point to : ps::neighbors(state.config, from, player)) {
          const ps::Segment edge{from, to};
          if (!ps::is_forbidden_boundary_segment(state.config, edge) &&
              !contains_segment(allowed, edge)) {
            state.used_segments.insert(edge);
          }
        }
      }
    }
  }
}

ps::GameState isolated_state(ps::Point ball, ps::Player mover,
                             const std::vector<ps::Segment> &allowed,
                             const std::vector<ps::Point> &visited = {}) {
  ps::GameState state = ps::make_initial_state();
  state.ball = ball;
  state.to_move = mover;
  state.status = ps::Status::InProgress;
  state.path = {ball};
  state.visit_count.clear();
  state.visit_count[ball] = 1;
  for (const ps::Point point : visited) {
    state.visit_count[point] = 1;
  }
  isolate_unused_edges(state, allowed);
  return state;
}

ps::GameState forced_two_edge_rebound() {
  const ps::Point root{4, 4};
  const ps::Point rebound{4, 3};
  const ps::Point handoff{4, 2};
  return isolated_state(root, ps::Player::One,
                        {ps::Segment{root, rebound},
                         ps::Segment{rebound, handoff}},
                        {rebound});
}

ps::GameState forced_two_edge_rebound_for_two() {
  const ps::Point root{4, 8};
  const ps::Point rebound{4, 9};
  const ps::Point handoff{4, 10};
  return isolated_state(root, ps::Player::Two,
                        {ps::Segment{root, rebound},
                         ps::Segment{rebound, handoff}},
                        {rebound});
}

ps::GameState four_edge_diamond(ps::Player mover) {
  const bool rotated = mover == ps::Player::Two;
  const auto point = [rotated](int x, int y) {
    return ps::Point{x, rotated ? 12 - y : y};
  };
  const ps::Point root = point(4, 5);
  const ps::Point left = point(3, 4);
  const ps::Point right = point(5, 4);
  const ps::Point far = point(4, 3);
  return isolated_state(root, mover,
                        {ps::Segment{root, left},
                         ps::Segment{root, right},
                         ps::Segment{left, far},
                         ps::Segment{right, far}});
}

ps::GameState north_goal_tie() {
  const ps::Point root{4, 1};
  return isolated_state(root, ps::Player::One,
                        {ps::Segment{root, {4, 0}},
                         ps::Segment{root, {5, 0}}});
}

ps::GameState south_goal_tie() {
  const ps::Point root{4, 11};
  return isolated_state(root, ps::Player::Two,
                        {ps::Segment{root, {4, 12}},
                         ps::Segment{root, {3, 12}}});
}

struct BruteResult {
  ps::Player winner{ps::Player::One};
  std::uint32_t distance{};
};

BruteResult brute_force(const ps::GameState &state) {
  if (ps::is_terminal(state)) {
    return BruteResult{*ps::winner(state), 0};
  }
  const ps::Player mover = state.to_move;
  std::optional<BruteResult> best;
  for (const ps::Move move : ps::legal_moves(state)) {
    BruteResult candidate = brute_force(ps::apply_move(state, move));
    ++candidate.distance;
    if (!best.has_value()) {
      best = candidate;
      continue;
    }
    const bool candidate_wins = candidate.winner == mover;
    const bool best_wins = best->winner == mover;
    if ((candidate_wins && !best_wins) ||
        (candidate_wins && best_wins &&
         candidate.distance < best->distance) ||
        (!candidate_wins && !best_wins &&
         candidate.distance > best->distance)) {
      best = candidate;
    }
  }
  if (!best.has_value()) {
    throw std::logic_error("brute-force fixture has no terminal continuation");
  }
  return *best;
}

ps::GameReviewConfig fast_review_config() {
  ps::GameReviewConfig config;
  config.mode = ps::ReviewMode::Fast;
  config.fast_calibration = {
      "test-fast-calibration-v1", "fast-50k", 0.0, 0.00001};
  return config;
}

ps::GameReviewConfig deep_review_config() {
  ps::GameReviewConfig config = fast_review_config();
  config.mode = ps::ReviewMode::Deep;
  config.deep_profile = ps::CompleteTurnAnalysisConfig::deep(100'000);
  config.deep_calibration = {
      "test-deep-calibration-v1", "deep-100k", 0.0, 0.00001};
  return config;
}

void complete_turn_analysis_is_legal_and_deterministic() {
  ps::CompleteTurnAnalysisConfig config =
      ps::CompleteTurnAnalysisConfig::fast();
  config.max_turn_depth = 2;
  config.max_nodes = 5'000;
  config.transposition_table_entries = 2'048;
  config.evaluation_table_entries = 1'024;
  require(config.profile_name() == "custom-analysis",
          "A modified analyzer config must not claim a fixed profile.");

  const ps::GameState root = ps::make_initial_state();
  const ps::CompleteTurnAnalysis first =
      ps::CompleteTurnAnalyzer(config).analyze(root);
  const ps::CompleteTurnAnalysis second =
      ps::CompleteTurnAnalyzer(config).analyze(root);
  require(first.action == second.action && first.root_score == second.root_score &&
              first.stats.nodes == second.stats.nodes,
          "Equal fixed-node complete-turn analyses must be deterministic.");

  ps::GameState replayed = root;
  const ps::Player mover = root.to_move;
  for (const ps::Move move : first.action) {
    const auto legal = ps::legal_moves(replayed);
    require(std::find(legal.begin(), legal.end(), move) != legal.end(),
            "Every recommended complete-action edge must be legal.");
    replayed = ps::apply_move(replayed, move);
  }
  require(ps::is_terminal(replayed) || replayed.to_move != mover,
          "A recommended action must reach terminal state or handoff.");
}

void deep_profiles_and_rebound_cache_are_strict() {
  for (const std::uint64_t nodes : {100'000ULL, 200'000ULL, 400'000ULL}) {
    const auto config = ps::CompleteTurnAnalysisConfig::deep(nodes);
    require(config.is_deep_profile() && config.max_nodes == nodes,
            "Every allowed Deep node profile must be identified as fixed.");
  }
  require_invalid_argument(
      [] { (void)ps::CompleteTurnAnalysisConfig::deep(50'000); },
      "Deep factory must reject non-candidate node budgets.");
  ps::CompleteTurnAnalysisConfig mutated =
      ps::CompleteTurnAnalysisConfig::deep(100'000);
  mutated.evaluation_table_entries = 10;
  require_invalid_argument(
      [&] { (void)ps::DeepTurnSearchBot(mutated); },
      "Deep bot must reject a mutated Deep profile.");

  const ps::GameState root = forced_two_edge_rebound();
  ps::DeepTurnSearchBot bot(100'000);
  const ps::Move first = bot.choose_move(root);
  const ps::GameState rebound = ps::apply_move(root, first);
  const ps::Move second = bot.choose_move(rebound);
  require(first == ps::Move{{4, 3}} && second == ps::Move{{4, 2}} &&
              bot.last_search_stats().cached_continuation &&
              bot.last_search_stats().nodes == 0 &&
              bot.last_search_stats().searches == 1,
          "Deep bot must cache the rest of a rebound action.");

  (void)bot.choose_move(root);
  require(!bot.last_search_stats().cached_continuation &&
              bot.last_search_stats().searches == 2,
          "Undo-like input must invalidate a Deep rebound cache.");

  ps::DeepTurnSearchBot mismatch(100'000);
  const ps::Move cached_first = mismatch.choose_move(root);
  ps::GameState different = ps::apply_move(root, cached_first);
  different.path.push_back({0, 0});
  (void)mismatch.choose_move(different);
  require(!mismatch.last_search_stats().cached_continuation &&
              mismatch.last_search_stats().searches == 2,
          "Any unrelated state mismatch must invalidate a Deep cache.");
}

void exact_solver_agrees_with_brute_force_and_distance() {
  const ps::GameState win = north_goal_tie();
  const BruteResult brute_win = brute_force(win);
  const ps::ExactEndgameResult exact_win = ps::ExactEndgameSolver{}.solve(win);
  require(exact_win.status == ps::ProofStatus::ProvenWin &&
              exact_win.winner == brute_win.winner &&
              exact_win.distance == brute_win.distance &&
              exact_win.distance == 1 &&
              exact_win.action == std::vector<ps::Move>{{{4, 0}}},
          "Exact solver must agree with exhaustive search and deterministic ties.");

  const ps::Point root{4, 4};
  const ps::Point visited{4, 3};
  const ps::GameState loss = isolated_state(
      root, ps::Player::One, {ps::Segment{root, visited}}, {visited});
  const BruteResult brute_loss = brute_force(loss);
  const ps::ExactEndgameResult exact_loss =
      ps::ExactEndgameSolver{}.solve(loss);
  require(exact_loss.status == ps::ProofStatus::ProvenLoss &&
              exact_loss.winner == brute_loss.winner &&
              exact_loss.distance == brute_loss.distance &&
              exact_loss.distance == 1,
          "Exact solver must report a forced blocked loss and edge distance.");

  for (const ps::GameState &fixture :
       {forced_two_edge_rebound(), forced_two_edge_rebound_for_two(),
        four_edge_diamond(ps::Player::One),
        four_edge_diamond(ps::Player::Two)}) {
    const BruteResult brute = brute_force(fixture);
    const ps::ExactEndgameResult exact = ps::ExactEndgameSolver{}.solve(fixture);
    const ps::ProofStatus expected =
        brute.winner == fixture.to_move ? ps::ProofStatus::ProvenWin
                                        : ps::ProofStatus::ProvenLoss;
    require(exact.status == expected && exact.winner == brute.winner &&
                exact.distance == brute.distance && !exact.action.empty() &&
                !exact.budget_exhausted,
            "Exact multi-edge proofs must agree with exhaustive minimax.");

    ps::GameState replayed = fixture;
    const ps::Player possession_player = fixture.to_move;
    for (const ps::Move move : exact.action) {
      const auto legal = ps::legal_moves(replayed);
      require(std::find(legal.begin(), legal.end(), move) != legal.end(),
              "Every proving-action edge must be legal.");
      replayed = ps::apply_move(replayed, move);
    }
    require(ps::is_terminal(replayed) ||
                replayed.to_move != possession_player,
            "An exact proving action must contain the complete possession.");
  }

  const ps::ExactEndgameResult rebound =
      ps::ExactEndgameSolver{}.solve(forced_two_edge_rebound());
  require(rebound.status == ps::ProofStatus::ProvenWin &&
              rebound.distance == 2 && rebound.action.size() == 2 &&
              rebound.action[0] == ps::Move{{4, 3}} &&
              rebound.action[1] == ps::Move{{4, 2}},
          "A rebound proof must reconstruct both edges and distance two.");

  for (const ps::Player mover : {ps::Player::One, ps::Player::Two}) {
    const ps::GameState diamond = four_edge_diamond(mover);
    const ps::ExactEndgameResult exact = ps::ExactEndgameSolver{}.solve(diamond);
    require(exact.status == ps::ProofStatus::ProvenWin &&
                exact.distance == 4 && exact.reachable_edge_count == 4,
            "A branching four-edge component must prove the exhaustive win distance.");
  }
}

void exact_solver_is_symmetric_and_rejects_non_boundaries() {
  const ps::ExactEndgameResult north =
      ps::ExactEndgameSolver{}.solve(north_goal_tie());
  const ps::ExactEndgameResult south =
      ps::ExactEndgameSolver{}.solve(south_goal_tie());
  require(north.status == ps::ProofStatus::ProvenWin &&
              south.status == ps::ProofStatus::ProvenWin &&
              north.winner == ps::Player::One &&
              south.winner == ps::Player::Two &&
              north.distance == south.distance &&
              north.action.front().to == ps::Point{4, 0} &&
              south.action.front().to == ps::Point{4, 12},
          "Color-swapped rotation must preserve proof and deterministic tie choice.");

  ps::GameState rebound_root = forced_two_edge_rebound();
  const ps::GameState rebound =
      ps::apply_move(rebound_root, ps::Move{{4, 3}});
  require_invalid_argument(
      [&] { (void)ps::ExactEndgameSolver{}.solve(rebound); },
      "The exact oracle must reject a state inside a rebound possession.");

  ps::RulesConfig wrong_rules;
  wrong_rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  require_invalid_argument(
      [&] {
        (void)ps::ExactEndgameSolver{}.solve(
            ps::make_initial_state(wrong_rules));
      },
      "The exact oracle must reject non-demo rules.");
}

void exact_solver_eligibility_and_budget_never_fake_a_proof() {
  static_assert(ps::ExactEndgameSolver::maximum_nodes == 250'000);
  const ps::ExactEndgameResult opening =
      ps::ExactEndgameSolver{}.solve(ps::make_initial_state());
  require(opening.status == ps::ProofStatus::Unknown &&
              !opening.winner.has_value() &&
              !opening.distance.has_value() && opening.action.empty() &&
              opening.reachable_edge_count >
                  ps::ExactEndgameSolver::maximum_reachable_edges &&
              !opening.budget_exhausted && opening.nodes == 0,
          "An ineligible large component must return Unknown without searching.");

  const ps::ExactEndgameResult exhausted =
      ps::detail::solve_exact_endgame_with_limit(north_goal_tie(), 1);
  require(exhausted.status == ps::ProofStatus::Unknown &&
              !exhausted.winner.has_value() &&
              !exhausted.distance.has_value() && exhausted.action.empty() &&
              exhausted.reachable_edge_count == 2 &&
              exhausted.budget_exhausted &&
              exhausted.nodes == 1,
          "Budget exhaustion must clear every partial proof field.");
}

void grade_bands_overrides_borderline_and_unclear_are_exact() {
  const auto grade = [](double loss) {
    ps::PossessionGradingInput input;
    input.estimated_loss_percentage_points = loss;
    return ps::grade_possession(input).grade;
  };
  require(grade(0.0) == ps::PossessionGrade::Best &&
              grade(1.999) == ps::PossessionGrade::Best &&
              grade(2.0) == ps::PossessionGrade::Good &&
              grade(4.999) == ps::PossessionGrade::Good &&
              grade(5.0) == ps::PossessionGrade::Inaccuracy &&
              grade(9.999) == ps::PossessionGrade::Inaccuracy &&
              grade(10.0) == ps::PossessionGrade::Mistake &&
              grade(19.999) == ps::PossessionGrade::Mistake &&
              grade(20.0) == ps::PossessionGrade::Blunder,
          "Every fixed estimated-loss grade boundary must be exact.");

  ps::PossessionGradingInput borderline;
  borderline.estimated_loss_percentage_points = 1.0;
  require(ps::grade_possession(borderline).borderline,
          "A result one point from a threshold must be borderline.");
  borderline.estimated_loss_percentage_points = 0.99;
  require(!ps::grade_possession(borderline).borderline,
          "A result more than one point from every threshold is not borderline.");
  borderline.deterministic_engine_estimate = false;
  borderline.estimated_loss_percentage_points = 2.0;
  require(!ps::grade_possession(borderline).borderline,
          "An exact proof must not claim heuristic threshold uncertainty.");

  ps::PossessionGradingInput override;
  override.forced = true;
  require(ps::grade_possession(override).grade == ps::PossessionGrade::Forced,
          "Every forced decision in a possession must yield Forced.");
  override.winning_terminal = true;
  require(ps::grade_possession(override).grade == ps::PossessionGrade::Best,
          "A winning terminal action must override Forced with Best.");
  override.winning_terminal = false;
  override.before_proof = ps::ProofStatus::ProvenWin;
  override.after_proof = ps::ProofStatus::ProvenLoss;
  require(ps::grade_possession(override).grade == ps::PossessionGrade::Blunder,
          "Surrendering a proven win must be Blunder.");
  override.after_proof = ps::ProofStatus::Unknown;
  require(ps::grade_possession(override).grade == ps::PossessionGrade::Forced,
          "An unresolved after-boundary search must not fabricate surrender proof.");
  override.before_proof = ps::ProofStatus::Unknown;
  override.after_proof = ps::ProofStatus::ProvenLoss;
  require(ps::grade_possession(override).grade == ps::PossessionGrade::Blunder,
          "Newly allowing a proven loss must be Blunder.");

  ps::PossessionGradingInput matched;
  matched.estimated_loss_percentage_points = 40.0;
  matched.action_matched = true;
  require(ps::grade_possession(matched).grade == ps::PossessionGrade::Best,
          "Matching the recommended complete action must be Best.");
  matched.required_search_completed = false;
  require(ps::grade_possession(matched).grade == ps::PossessionGrade::Unclear,
          "A required search with no completed depth must be Unclear.");
  matched.winning_terminal = true;
  require(ps::grade_possession(matched).grade == ps::PossessionGrade::Best,
          "A winning terminal action must remain Best without a heuristic depth.");

  ps::PossessionGradingInput disagreement;
  disagreement.estimated_loss_percentage_points = 12.0;
  disagreement.fast_grade = ps::PossessionGrade::Best;
  require(ps::grade_possession(disagreement).grade ==
              ps::PossessionGrade::Unclear,
          "A two-band Fast/Deep disagreement must be Unclear.");
  disagreement.estimated_loss_percentage_points = 4.0;
  require(ps::grade_possession(disagreement).grade == ps::PossessionGrade::Good,
          "A one-band Fast/Deep refinement remains reportable.");
}

void review_validates_declared_moves_and_outcomes() {
  ps::GameReviewSession review(fast_review_config());
  const ps::DeclaredReviewMove wrong{
      2, ps::Player::One, {4, 6}, {4, 5}, false,
      ps::Status::InProgress};
  require_invalid_argument(
      [&] { review.append_move(wrong); },
      "A declared ply mismatch must be rejected through Match validation.");

  const ps::DeclaredReviewMove correct{
      1, ps::Player::One, {4, 6}, {4, 5}, false,
      ps::Status::InProgress};
  review.append_move(correct);
  require_invalid_argument(
      [&] { review.append_move(ps::Move{{8, 8}}); },
      "An illegal imported edge must be rejected.");
  require_invalid_argument(
      [&] {
        review.finalize(ps::DeclaredReviewOutcome{
            ps::Status::WonByOne, ps::Player::One, false});
      },
      "An inconsistent declared winner/status must be rejected.");
  review.finalize(ps::DeclaredReviewOutcome{
      ps::Status::InProgress, std::nullopt, true});
  require(review.snapshot().source_replay.size() == 1 &&
              review.snapshot().possessions.size() == 1,
          "Rejected declared input must not mutate the retained replay.");

  ps::GameReviewConfig mismatched = fast_review_config();
  mismatched.fast_calibration.profile_name = "deep-100k";
  require_invalid_argument(
      [&] { (void)ps::GameReviewSession(mismatched); },
      "A calibration from a different search profile must be rejected.");
}

void review_groups_rebounds_terminal_and_truncated_possessions() {
  ps::GameReviewSession grouped(fast_review_config());
  grouped.append_move(ps::Move{{4, 5}});  // P1 handoff.
  grouped.append_move(ps::Move{{5, 6}});  // P2 handoff.
  grouped.append_move(ps::Move{{4, 6}});  // P1 rebound at the start.
  grouped.append_move(ps::Move{{5, 5}});  // P1 handoff.
  grouped.finalize(ps::DeclaredReviewOutcome{
      ps::Status::InProgress, std::nullopt, true});
  const auto &grouped_view = grouped.snapshot();
  require(grouped_view.possessions.size() == 3 &&
              grouped_view.possessions[0].first_ply == 0 &&
              grouped_view.possessions[0].edge_count == 1 &&
              grouped_view.possessions[2].first_ply == 2 &&
              grouped_view.possessions[2].edge_count == 2 &&
              !grouped_view.possessions[2].truncated,
          "Consecutive rebound edges by one player must form one possession.");

  ps::GameReviewSession truncated(fast_review_config());
  truncated.append_move(ps::Move{{4, 5}});
  truncated.append_move(ps::Move{{5, 6}});
  truncated.append_move(ps::Move{{4, 6}});
  truncated.finalize(ps::DeclaredReviewOutcome{
      ps::Status::InProgress, std::nullopt, true});
  require(truncated.snapshot().possessions.size() == 3 &&
              truncated.snapshot().possessions.back().truncated,
          "A replay ending during a required rebound must mark its possession truncated.");
  try {
    (void)truncated.step();
    (void)truncated.step();
    (void)truncated.step();
  } catch (const std::exception &error) {
    throw std::runtime_error("truncated review analysis failed: " +
                             std::string(error.what()));
  }
  require(truncated.snapshot().possessions.back().grade ==
              ps::PossessionGrade::Unclear,
          "A truncated possession must remain Unclear.");

  ps::GameReviewSession terminal(fast_review_config());
  std::size_t terminal_ply = 0;
  for (const ps::Point to : std::vector<ps::Point>{
           {4, 5}, {4, 4}, {4, 3}, {4, 2},
           {4, 1}, {3, 2}, {2, 3}, {3, 3},
           {4, 2}, {3, 1}, {4, 0}}) {
    ++terminal_ply;
    try {
      terminal.append_move(ps::Move{to});
    } catch (const std::exception &error) {
      throw std::runtime_error("terminal fixture failed at ply " +
                               std::to_string(terminal_ply) + ": " +
                               error.what());
    }
  }
  terminal.finalize(ps::DeclaredReviewOutcome{
      ps::Status::WonByOne, ps::Player::One, false});
  const auto &terminal_view = terminal.snapshot();
  require(terminal_view.possessions.size() == 9 &&
              terminal_view.possessions.back().terminal &&
              terminal_view.possessions.back().edge_count == 3 &&
              !terminal_view.possessions.back().truncated,
          "A terminal rebound sequence must remain one complete possession.");
}

void review_orients_both_players_and_refines_fast_before_deep() {
  ps::GameReviewSession review(deep_review_config());
  review.append_move(ps::Move{{4, 5}});
  review.append_move(ps::Move{{5, 4}});
  review.finalize(ps::DeclaredReviewOutcome{
      ps::Status::InProgress, std::nullopt, true});
  require(review.snapshot().total_steps == 4,
          "Deep review must schedule a Fast pass and a Deep refinement.");

  require(review.step() && review.snapshot().completed_steps == 1,
          "The first Deep-review step must publish a Fast preview.");
  require(review.step() && review.snapshot().completed_steps == 2,
          "All Fast previews must precede Deep refinements.");
  const auto fast_second = review.snapshot().possessions[1];

  ps::GameState player_two_boundary = ps::apply_move(
      ps::make_initial_state(), ps::Move{{4, 5}});
  const int raw_player_one_score =
      ps::CompleteTurnAnalyzer(ps::CompleteTurnAnalysisConfig::fast())
          .analyze(player_two_boundary)
          .root_score;
  require(fast_second.player == ps::Player::Two &&
              fast_second.before.oriented_score == -raw_player_one_score,
          "Player Two review scores must be oriented to the possession player.");

  while (review.step()) {
  }
  require(review.snapshot().complete &&
              review.snapshot().completed_steps == 4 &&
              review.snapshot().possessions[0].fast_grade.has_value() &&
              review.snapshot().possessions[1].fast_grade.has_value(),
          "Deep review must retain Fast grades after deterministic refinement.");

  ps::GameReviewSession cancelled(deep_review_config());
  cancelled.append_move(ps::Move{{4, 5}});
  cancelled.finalize(ps::DeclaredReviewOutcome{
      ps::Status::InProgress, std::nullopt, true});
  cancelled.cancel();
  require(!cancelled.step() && cancelled.snapshot().cancelled &&
              !cancelled.snapshot().complete,
          "Cancellation must stop future synchronous review steps.");
}

}  // namespace

int run_game_review_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"complete_turn_analysis_is_legal_and_deterministic",
       complete_turn_analysis_is_legal_and_deterministic},
      {"deep_profiles_and_rebound_cache_are_strict",
       deep_profiles_and_rebound_cache_are_strict},
      {"exact_solver_agrees_with_brute_force_and_distance",
       exact_solver_agrees_with_brute_force_and_distance},
      {"exact_solver_is_symmetric_and_rejects_non_boundaries",
       exact_solver_is_symmetric_and_rejects_non_boundaries},
      {"exact_solver_eligibility_and_budget_never_fake_a_proof",
       exact_solver_eligibility_and_budget_never_fake_a_proof},
      {"grade_bands_overrides_borderline_and_unclear_are_exact",
       grade_bands_overrides_borderline_and_unclear_are_exact},
      {"review_validates_declared_moves_and_outcomes",
       review_validates_declared_moves_and_outcomes},
      {"review_groups_rebounds_terminal_and_truncated_possessions",
       review_groups_rebounds_terminal_and_truncated_possessions},
      {"review_orients_both_players_and_refines_fast_before_deep",
       review_orients_both_players_and_refines_fast_before_deep},
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
            << "/" << tests.size() << " game-review tests passed.\n";
  return failures == 0 ? 0 : 1;
}

#ifdef PAPERSOCCER_STANDALONE_GAME_REVIEW_TEST
int main() { return run_game_review_tests(); }
#endif
