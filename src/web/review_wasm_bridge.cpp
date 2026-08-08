#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <emscripten/emscripten.h>
#include <emscripten/heap.h>

#include "papersoccer/game_review.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

std::unique_ptr<ps::GameReviewSession> review_session;
std::unique_ptr<ps::Match> validation_match;
std::optional<ps::GameReviewConfig> review_config;
std::string snapshot_cache;
std::string last_error;
bool declared_truncated{false};
bool finalized{false};

std::unique_ptr<ps::Match> sandbox_match;
std::size_t sandbox_source_possession{};
std::size_t sandbox_boundary_ply{};
std::size_t sandbox_recommended_edges{};
std::vector<ps::Move> sandbox_recommended_action;
std::string sandbox_snapshot_cache;

std::unique_ptr<ps::Match> analysis_probe_match;
std::optional<ps::CompleteTurnAnalysisConfig> analysis_probe_profile;
std::optional<ps::CompleteTurnAnalysis> analysis_probe_result;
std::string analysis_probe_result_cache;
std::string analysis_probe_error;
bool analysis_probe_ran{false};

template <typename Config>
Config make_locked_config(ps::ReviewMode mode) {
  if constexpr (requires { Config::locked(mode); }) {
    return Config::locked(mode);
  } else {
    throw std::runtime_error(
        "the frozen validation calibration lock is not available");
  }
}

std::string_view player_name(ps::Player player) noexcept {
  return player == ps::Player::One ? "one" : "two";
}

std::string_view status_name(ps::Status status) noexcept {
  switch (status) {
    case ps::Status::InProgress:
      return "inProgress";
    case ps::Status::WonByOne:
      return "wonByOne";
    case ps::Status::WonByTwo:
      return "wonByTwo";
  }
  return "unknown";
}

std::string_view mode_name(ps::ReviewMode mode) noexcept {
  return mode == ps::ReviewMode::Deep ? "deep" : "fast";
}

std::string_view proof_name(ps::ProofStatus proof) noexcept {
  switch (proof) {
    case ps::ProofStatus::Unknown:
      return "Unknown";
    case ps::ProofStatus::ProvenWin:
      return "ProvenWin";
    case ps::ProofStatus::ProvenLoss:
      return "ProvenLoss";
  }
  return "Unknown";
}

std::string_view grade_name(ps::PossessionGrade grade) noexcept {
  switch (grade) {
    case ps::PossessionGrade::Forced:
      return "Forced";
    case ps::PossessionGrade::Best:
      return "Best";
    case ps::PossessionGrade::Good:
      return "Good";
    case ps::PossessionGrade::Inaccuracy:
      return "Inaccuracy";
    case ps::PossessionGrade::Mistake:
      return "Mistake";
    case ps::PossessionGrade::Blunder:
      return "Blunder";
    case ps::PossessionGrade::Unclear:
      return "Unclear";
  }
  return "Unclear";
}

std::string_view confidence_state(
    const ps::PossessionReview &possession) noexcept {
  if (possession.grade == ps::PossessionGrade::Unclear) {
    return "unclear";
  }
  if (possession.before.exact &&
      (possession.terminal || possession.after.exact)) {
    return "exact";
  }
  if (possession.borderline) {
    return "borderline-estimate";
  }
  return "deterministic-estimate";
}

std::string_view probe_candidate_name(std::uint64_t max_nodes) {
  switch (max_nodes) {
    case 100'000:
      return "deep-turn-search-100k";
    case 200'000:
      return "deep-turn-search-200k";
    case 400'000:
      return "deep-turn-search-400k";
    default:
      throw std::invalid_argument(
          "analysis probe requires a 100k, 200k, or 400k candidate");
  }
}

bool parse_player(int value, ps::Player &player) noexcept {
  if (value == 1) {
    player = ps::Player::One;
    return true;
  }
  if (value == 2) {
    player = ps::Player::Two;
    return true;
  }
  return false;
}

bool parse_status(int value, ps::Status &status) noexcept {
  if (value == 0) {
    status = ps::Status::InProgress;
    return true;
  }
  if (value == 1) {
    status = ps::Status::WonByOne;
    return true;
  }
  if (value == 2) {
    status = ps::Status::WonByTwo;
    return true;
  }
  return false;
}

std::optional<ps::Player> winner_for_status(ps::Status status) noexcept {
  if (status == ps::Status::WonByOne) {
    return ps::Player::One;
  }
  if (status == ps::Status::WonByTwo) {
    return ps::Player::Two;
  }
  return std::nullopt;
}

void write_point(std::ostream &out, ps::Point point) {
  out << "{\"x\":" << point.x << ",\"y\":" << point.y << '}';
}

void write_optional_player(std::ostream &out,
                           std::optional<ps::Player> player) {
  if (player.has_value()) {
    out << '"' << player_name(*player) << '"';
  } else {
    out << "null";
  }
}

void write_played_move(std::ostream &out, const ps::PlayedMove &move) {
  out << "{\"ply\":" << move.ply << ",\"player\":\""
      << player_name(move.player) << "\",\"from\":";
  write_point(out, move.from);
  out << ",\"to\":";
  write_point(out, move.to);
  out << ",\"extraTurn\":" << (move.extra_turn ? "true" : "false")
      << ",\"statusAfter\":\"" << status_name(move.status_after) << "\"}";
}

void write_action(std::ostream &out, const std::vector<ps::Move> &action) {
  out << '[';
  for (std::size_t i = 0; i < action.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    write_point(out, action[i].to);
  }
  out << ']';
}

void write_search_stats(std::ostream &out,
                        const ps::CompleteTurnSearchStats &stats) {
  out << "{\"completedDepth\":" << stats.completed_turn_depth
      << ",\"attemptedDepth\":" << stats.attempted_turn_depth
      << ",\"nodes\":" << stats.nodes
      << ",\"leafEvaluations\":" << stats.leaf_evaluations
      << ",\"terminalNodes\":" << stats.terminal_nodes
      << ",\"completedActions\":" << stats.completed_actions
      << ",\"cutoffs\":" << stats.cutoffs
      << ",\"transpositionProbes\":" << stats.transposition_probes
      << ",\"transpositionHits\":" << stats.transposition_hits
      << ",\"transpositionCutoffs\":" << stats.transposition_cutoffs
      << ",\"transpositionStores\":" << stats.transposition_stores
      << ",\"continuationTranspositionHits\":"
      << stats.continuation_transposition_hits
      << ",\"evaluationCacheProbes\":" << stats.evaluation_cache_probes
      << ",\"evaluationCacheHits\":" << stats.evaluation_cache_hits
      << ",\"terminalBoundCutoffs\":" << stats.terminal_bound_cutoffs
      << ",\"forcedEdges\":" << stats.forced_edges
      << ",\"rootSeedActions\":" << stats.root_seed_actions
      << ",\"rootTranspositionReuses\":"
      << stats.root_transposition_reuses
      << ",\"maxActionEdges\":" << stats.max_action_edges
      << ",\"rootScore\":" << stats.root_score
      << ",\"budgetExhausted\":"
      << (stats.budget_exhausted ? "true" : "false") << '}';
}

void write_boundary(std::ostream &out,
                    const ps::BoundaryReviewDiagnostics &diagnostic) {
  out << "{\"exact\":" << (diagnostic.exact ? "true" : "false")
      << ",\"proof\":\"" << proof_name(diagnostic.proof)
      << "\",\"provenWinner\":";
  write_optional_player(out, diagnostic.proven_winner);
  out << ",\"proofDistance\":";
  if (diagnostic.proof_distance.has_value()) {
    out << *diagnostic.proof_distance;
  } else {
    out << "null";
  }
  out << ",\"reachableEdgeCount\":" << diagnostic.reachable_edge_count
      << ",\"oracleNodes\":" << diagnostic.oracle_nodes
      << ",\"oracleCacheHits\":" << diagnostic.oracle_cache_hits
      << ",\"oracleBudgetExhausted\":"
      << (diagnostic.oracle_budget_exhausted ? "true" : "false")
      << ",\"orientedScore\":" << diagnostic.oriented_score
      << ",\"estimatedWinChance\":"
      << diagnostic.estimated_win_chance << ",\"search\":";
  write_search_stats(out, diagnostic.search);
  out << '}';
}

void write_profile(std::ostream &out,
                   const ps::CompleteTurnAnalysisConfig &profile) {
  out << "{\"identity\":\"" << profile.profile_name()
      << "\",\"maxTurnDepth\":" << profile.max_turn_depth
      << ",\"maxNodes\":" << profile.max_nodes
      << ",\"transpositionEntries\":"
      << profile.transposition_table_entries
      << ",\"evaluationCacheEntries\":"
      << profile.evaluation_table_entries
      << ",\"wallClock\":false,\"replayCorrections\":false,"
         "\"learnedValueBlendPercent\":0}";
}

std::string serialize_analysis_probe_result() {
  if (!analysis_probe_match || !analysis_probe_profile.has_value() ||
      !analysis_probe_result.has_value() || !analysis_probe_ran) {
    throw std::runtime_error("no completed analysis probe is available");
  }

  const ps::GameState &state = analysis_probe_match->state();
  const ps::CompleteTurnAnalysisConfig &profile = *analysis_probe_profile;
  const ps::CompleteTurnAnalysis &result = *analysis_probe_result;
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "{\"schema\":\"papersoccer.analysis-probe.v1\",";
  out << "\"analyzerIdentity\":\"complete-turn-analysis-fresh-search-v1\",";
  out << "\"candidateIdentity\":\""
      << probe_candidate_name(profile.max_nodes) << "\",\"profile\":";
  write_profile(out, profile);
  out << ",\"openingPlies\":" << analysis_probe_match->history().size()
      << ",\"position\":{\"ball\":";
  write_point(out, state.ball);
  out << ",\"toMove\":\"" << player_name(state.to_move)
      << "\",\"status\":\"" << status_name(state.status)
      << "\",\"possessionBoundary\":true},\"action\":";
  write_action(out, result.action);
  out << ",\"completeAction\":true,\"rootScore\":" << result.root_score
      << ",\"diagnostics\":";
  write_search_stats(out, result.stats);
  out << '}';
  return out.str();
}

void write_calibration(std::ostream &out,
                       const ps::GameReviewCalibration &calibration) {
  out << "{\"identity\":\"" << calibration.identity
      << "\",\"profileIdentity\":\"" << calibration.profile_name
      << "\",\"mapping\":\"logistic\",\"intercept\":"
      << calibration.intercept << ",\"scoreCoefficient\":"
      << calibration.score_coefficient
      << ",\"evidenceProfileIdentity\":\""
      << calibration.evidence_profile_identity
      << "\",\"profileSha256\":\"" << calibration.profile_sha256
      << "\",\"mappingSha256\":\"" << calibration.mapping_sha256
      << "\"}";
}

std::string serialize_snapshot() {
  if (!review_session || !review_config.has_value()) {
    throw std::runtime_error("no Game Review session has been started");
  }
  const ps::GameReviewSnapshot &snapshot = review_session->snapshot();
  const ps::CompleteTurnAnalysisConfig &selected_profile =
      snapshot.mode == ps::ReviewMode::Deep ? review_config->deep_profile
                                            : review_config->fast_profile;
  const ps::GameReviewCalibration &selected_calibration =
      snapshot.mode == ps::ReviewMode::Deep ? review_config->deep_calibration
                                            : review_config->fast_calibration;

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "{\"schema\":\"papersoccer.review-session.v1\",\"mode\":\""
      << mode_name(snapshot.mode) << "\",\"finalized\":"
      << (snapshot.finalized ? "true" : "false") << ",\"cancelled\":"
      << (snapshot.cancelled ? "true" : "false") << ",\"complete\":"
      << (snapshot.complete ? "true" : "false")
      << ",\"completedPossessions\":" << snapshot.completed_steps
      << ",\"totalPossessions\":" << snapshot.total_steps
      << ",\"replayStatus\":\"" << status_name(snapshot.replay_status)
      << "\",\"truncated\":" << (declared_truncated ? "true" : "false")
      << ",\"analyzer\":{\"identity\":\"complete-turn-game-review-v1\","
         "\"judgment\":\"deterministic-engine-estimate\"},\"profile\":";
  write_profile(out, selected_profile);
  out << ",\"calibration\":";
  write_calibration(out, selected_calibration);
  out << ",\"oracle\":{\"identity\":\"exact-endgame-18-edges-v1\","
         "\"maximumReachableEdges\":"
      << ps::ExactEndgameSolver::maximum_reachable_edges
      << ",\"maximumNodes\":" << ps::ExactEndgameSolver::maximum_nodes
      << ",\"wallClock\":false},\"rankedSource\":{"
         "\"name\":\""
      << ps::Rank5DerivedBot::original_artifact_name()
      << "\",\"submissionId\":\""
      << ps::Rank5DerivedBot::original_submission_id()
      << "\",\"sha256\":\"" << ps::Rank5DerivedBot::original_sha256()
      << "\",\"rank\":" << ps::Rank5DerivedBot::original_rank
      << ",\"fieldSize\":" << ps::Rank5DerivedBot::original_field_size
      << "},\"possessions\":[";

  for (std::size_t i = 0; i < snapshot.possessions.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const ps::PossessionReview &possession = snapshot.possessions[i];
    out << "{\"possession\":" << possession.possession
        << ",\"startPly\":" << possession.first_ply + 1U
        << ",\"endPly\":" << possession.first_ply + possession.edge_count
        << ",\"edgeCount\":" << possession.edge_count
        << ",\"player\":\"" << player_name(possession.player)
        << "\",\"terminal\":" << (possession.terminal ? "true" : "false")
        << ",\"truncated\":" << (possession.truncated ? "true" : "false")
        << ",\"forced\":" << (possession.forced ? "true" : "false")
        << ",\"analyzed\":" << (possession.analyzed ? "true" : "false")
        << ",\"recommendedActionMatched\":";
    if (!possession.analyzed || possession.truncated ||
        possession.recommended_action.empty()) {
      out << "null";
    } else {
      out << (possession.first_divergence.has_value() ? "false" : "true");
    }
    out << ",\"playedAction\":";
    write_action(out, possession.played_action);
    out << ",\"recommendedAction\":";
    write_action(out, possession.recommended_action);
    out << ",\"firstDivergenceEdge\":";
    if (possession.first_divergence.has_value()) {
      out << *possession.first_divergence;
    } else {
      out << "null";
    }
    out << ",\"grade\":\"" << grade_name(possession.grade)
        << "\",\"estimatedLossPercentagePoints\":";
    if (!possession.analyzed || possession.truncated) {
      out << "null";
    } else {
      out << possession.estimated_loss_percentage_points;
    }
    out << ",\"borderline\":"
        << (possession.borderline ? "true" : "false")
        << ",\"confidenceState\":\"" << confidence_state(possession)
        << '"'
        << ",\"proof\":\"" << proof_name(possession.proof)
        << "\",\"deterministicEngineEstimate\":"
        << (possession.deterministic_engine_estimate ? "true" : "false")
        << ",\"fastGrade\":";
    if (possession.fast_grade.has_value()) {
      out << '"' << grade_name(*possession.fast_grade) << '"';
    } else {
      out << "null";
    }
    out << ",\"before\":";
    write_boundary(out, possession.before);
    out << ",\"after\":";
    write_boundary(out, possession.after);
    out << '}';
  }
  out << "]}";
  return out.str();
}

std::string serialize_sandbox_snapshot() {
  if (!sandbox_match) {
    throw std::runtime_error("no try-line sandbox is active");
  }

  const ps::GameState &state = sandbox_match->state();
  const std::optional<ps::Player> winning_player = ps::winner(state);
  const std::vector<ps::Move> legal = sandbox_match->legal_moves();
  const std::vector<ps::PlayedMove> &history = sandbox_match->history();
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "{\"schema\":\"papersoccer.review-sandbox.v1\",\"active\":true"
      << ",\"sourcePossession\":" << sandbox_source_possession
      << ",\"boundaryPly\":" << sandbox_boundary_ply
      << ",\"recommendedEdges\":" << sandbox_recommended_edges
      << ",\"recommendedAction\":";
  write_action(out, sandbox_recommended_action);
  out << ",\"state\":{\"ball\":";
  write_point(out, state.ball);
  out << ",\"toMove\":\"" << player_name(state.to_move)
      << "\",\"status\":\"" << status_name(state.status)
      << "\",\"winner\":";
  write_optional_player(out, winning_player);
  out << "},\"legalMoves\":[";
  for (std::size_t i = 0; i < legal.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const ps::Move &move = legal[i];
    out << "{\"id\":" << i << ",\"to\":";
    write_point(out, move.to);
    out << ",\"extraTurn\":"
        << (ps::grants_extra_turn(state, move.to) ? "true" : "false")
        << '}';
  }
  out << "],\"replay\":{\"schema\":\"papersoccer.replay.v2\","
         "\"rules\":{\"width\":8,\"height\":10},\"players\":{},"
         "\"start\":{\"x\":4,\"y\":6},\"status\":\""
      << status_name(state.status) << "\",\"winner\":";
  write_optional_player(out, winning_player);
  out << ",\"truncated\":"
      << (state.status == ps::Status::InProgress ? "true" : "false")
      << ",\"moves\":[";
  for (std::size_t i = 0; i < history.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    write_played_move(out, history[i]);
  }
  out << "]}}";
  return out.str();
}

bool ensure_active() {
  if (!review_session || !validation_match) {
    last_error = "no Game Review session has been started";
    return false;
  }
  return true;
}

template <typename Callback>
int command(Callback callback) {
  try {
    callback();
    snapshot_cache.clear();
    last_error.clear();
    return 1;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 0;
  }
}

template <typename Callback>
int analysis_probe_command(Callback callback) {
  try {
    callback();
    analysis_probe_result_cache.clear();
    analysis_probe_error.clear();
    return 1;
  } catch (const std::exception &error) {
    analysis_probe_error = error.what();
    return 0;
  }
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int ps_review_start(int mode_value) {
  return command([&] {
    ps::ReviewMode mode;
    if (mode_value == 0) {
      mode = ps::ReviewMode::Fast;
    } else if (mode_value == 1) {
      mode = ps::ReviewMode::Deep;
    } else {
      throw std::invalid_argument("review mode must be Fast or Deep");
    }
    ps::GameReviewConfig config =
        make_locked_config<ps::GameReviewConfig>(mode);
    auto replacement = std::make_unique<ps::GameReviewSession>(config);
    auto validator = std::make_unique<ps::Match>();
    review_session = std::move(replacement);
    validation_match = std::move(validator);
    review_config = std::move(config);
    declared_truncated = false;
    finalized = false;
    sandbox_match.reset();
    sandbox_recommended_action.clear();
    sandbox_snapshot_cache.clear();
  });
}

EMSCRIPTEN_KEEPALIVE int ps_review_append_move(
    std::uint32_t declared_ply, int declared_player, int from_x, int from_y,
    int to_x, int to_y, int declared_extra_turn, int declared_status) {
  if (!ensure_active()) {
    return 0;
  }
  return command([&] {
    if (finalized) {
      throw std::invalid_argument("cannot append a move after review finalization");
    }
    const ps::GameState &state = validation_match->state();
    if (ps::is_terminal(state)) {
      throw std::invalid_argument("the replay contains a move after the game ended");
    }
    if (declared_ply != validation_match->history().size() + 1U) {
      throw std::invalid_argument("the replay ply sequence is not contiguous");
    }
    ps::Player player;
    if (!parse_player(declared_player, player) || player != state.to_move) {
      throw std::invalid_argument("the replay declares the wrong player to move");
    }
    if (state.ball != ps::Point{from_x, from_y}) {
      throw std::invalid_argument("the replay move starts at the wrong point");
    }
    if (declared_extra_turn != 0 && declared_extra_turn != 1) {
      throw std::invalid_argument("the replay move must declare extraTurn as a boolean");
    }
    ps::Status status;
    if (!parse_status(declared_status, status)) {
      throw std::invalid_argument("the replay move declares an unknown status");
    }

    const ps::Move move{ps::Point{to_x, to_y}};
    const bool extra_turn = ps::grants_extra_turn(state, move.to);
    const ps::GameState next = ps::apply_move(state, move);
    if (extra_turn != (declared_extra_turn != 0)) {
      throw std::invalid_argument("the replay move has an inconsistent extraTurn value");
    }
    if (next.status != status) {
      throw std::invalid_argument("the replay move has an inconsistent statusAfter value");
    }

    review_session->append_move(ps::DeclaredReviewMove{
        declared_ply,
        player,
        ps::Point{from_x, from_y},
        move.to,
        declared_extra_turn != 0,
        status,
    });
    validation_match->play(move);
  });
}

EMSCRIPTEN_KEEPALIVE int ps_review_finalize(
    int declared_status, int declared_winner, int truncated_value) {
  if (!ensure_active()) {
    return 0;
  }
  return command([&] {
    if (finalized) {
      throw std::invalid_argument("the Game Review session is already finalized");
    }
    ps::Status status;
    if (!parse_status(declared_status, status)) {
      throw std::invalid_argument("the replay declares an unknown outcome");
    }
    if (truncated_value != 0 && truncated_value != 1) {
      throw std::invalid_argument("the replay truncated field must be a boolean");
    }
    if (status != validation_match->state().status) {
      throw std::invalid_argument("the replay outcome disagrees with the C++ rules");
    }
    const std::optional<ps::Player> winning_player = winner_for_status(status);
    const int expected_winner = !winning_player.has_value()
                                    ? 0
                                    : (*winning_player == ps::Player::One ? 1 : 2);
    if (declared_winner != expected_winner) {
      throw std::invalid_argument("the replay winner disagrees with its outcome");
    }
    if (status == ps::Status::InProgress && truncated_value == 0) {
      throw std::invalid_argument(
          "an unfinished replay must explain that it was truncated");
    }
    if (status != ps::Status::InProgress && truncated_value != 0) {
      throw std::invalid_argument("a terminal replay cannot be marked truncated");
    }
    declared_truncated = truncated_value != 0;
    review_session->finalize(ps::DeclaredReviewOutcome{
        status,
        winning_player,
        truncated_value != 0,
    });
    finalized = true;
  });
}

EMSCRIPTEN_KEEPALIVE int ps_review_step() {
  if (!ensure_active()) {
    return 0;
  }
  try {
    if (!finalized) {
      throw std::invalid_argument("the Game Review replay is not finalized");
    }
    (void)review_session->step();
    snapshot_cache.clear();
    last_error.clear();
    return review_session->snapshot().complete ? 2 : 1;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 0;
  }
}

EMSCRIPTEN_KEEPALIVE const char *ps_review_snapshot_json() {
  if (!ensure_active()) {
    return nullptr;
  }
  try {
    snapshot_cache = serialize_snapshot();
    last_error.clear();
    return snapshot_cache.c_str();
  } catch (const std::exception &error) {
    last_error = error.what();
    return nullptr;
  }
}

EMSCRIPTEN_KEEPALIVE int ps_review_cancel() {
  if (!review_session) {
    return 0;
  }
  review_session->cancel();
  snapshot_cache.clear();
  last_error.clear();
  return 1;
}

EMSCRIPTEN_KEEPALIVE const char *ps_review_last_error() {
  return last_error.c_str();
}

EMSCRIPTEN_KEEPALIVE std::uint32_t ps_review_heap_bytes() {
  return static_cast<std::uint32_t>(emscripten_get_heap_size());
}

EMSCRIPTEN_KEEPALIVE int ps_review_sandbox_start(
    std::uint32_t possession_index) {
  if (!ensure_active()) {
    return 0;
  }
  return command([&] {
    const ps::GameReviewSnapshot &snapshot = review_session->snapshot();
    if (!snapshot.complete || snapshot.cancelled) {
      throw std::invalid_argument(
          "try-line sandbox requires a completed Game Review");
    }
    if (possession_index >= snapshot.possessions.size()) {
      throw std::out_of_range("try-line possession index is out of range");
    }
    const ps::PossessionReview &possession =
        snapshot.possessions[possession_index];
    if (possession.recommended_action.empty()) {
      throw std::invalid_argument(
          "the selected possession has no complete recommended action");
    }

    auto replacement = std::make_unique<ps::Match>();
    const std::vector<ps::PlayedMove> &source = validation_match->history();
    if (possession.first_ply > source.size()) {
      throw std::logic_error("try-line boundary is outside the source replay");
    }
    for (std::size_t i = 0; i < possession.first_ply; ++i) {
      replacement->play(ps::Move{source[i].to});
    }
    ps::GameState recommendation_check = replacement->state();
    const ps::Player recommendation_player = recommendation_check.to_move;
    for (const ps::Move move : possession.recommended_action) {
      if (ps::is_terminal(recommendation_check) ||
          recommendation_check.to_move != recommendation_player) {
        throw std::logic_error(
            "the recommendation is not one complete legal possession");
      }
      const std::vector<ps::Move> legal = ps::legal_moves(recommendation_check);
      if (std::find(legal.begin(), legal.end(), move) == legal.end()) {
        throw std::logic_error(
            "the recommendation contains an illegal sandbox edge");
      }
      recommendation_check = ps::apply_move(recommendation_check, move);
    }
    if (!ps::is_terminal(recommendation_check) &&
        recommendation_check.to_move == recommendation_player) {
      throw std::logic_error(
          "the recommendation ended before its rebound possession completed");
    }

    sandbox_source_possession = possession_index;
    sandbox_boundary_ply = possession.first_ply;
    sandbox_recommended_edges = possession.recommended_action.size();
    sandbox_recommended_action = possession.recommended_action;
    sandbox_match = std::move(replacement);
    sandbox_snapshot_cache.clear();
  });
}

EMSCRIPTEN_KEEPALIVE int ps_review_sandbox_play(int to_x, int to_y) {
  if (!sandbox_match) {
    last_error = "no try-line sandbox is active";
    return 0;
  }
  return command([&] {
    if (ps::is_terminal(sandbox_match->state())) {
      throw std::invalid_argument("the try-line sandbox is already terminal");
    }
    sandbox_match->play(ps::Move{ps::Point{to_x, to_y}});
    sandbox_snapshot_cache.clear();
  });
}

EMSCRIPTEN_KEEPALIVE const char *ps_review_sandbox_snapshot_json() {
  try {
    sandbox_snapshot_cache = serialize_sandbox_snapshot();
    last_error.clear();
    return sandbox_snapshot_cache.c_str();
  } catch (const std::exception &error) {
    last_error = error.what();
    return nullptr;
  }
}

EMSCRIPTEN_KEEPALIVE int ps_review_sandbox_close() {
  sandbox_match.reset();
  sandbox_recommended_action.clear();
  sandbox_snapshot_cache.clear();
  last_error.clear();
  return 1;
}

// This intentionally separate C ABI exists only to measure the three frozen
// DeepTurnSearch candidates before one of them is selected and calibrated.
// It never constructs a GameReviewSession and cannot emit review grades.
EMSCRIPTEN_KEEPALIVE int ps_analysis_probe_start(
    std::uint32_t candidate_max_nodes) {
  return analysis_probe_command([&] {
    ps::CompleteTurnAnalysisConfig profile =
        ps::CompleteTurnAnalysisConfig::deep(candidate_max_nodes);
    analysis_probe_match = std::make_unique<ps::Match>();
    analysis_probe_profile = profile;
    analysis_probe_result.reset();
    analysis_probe_ran = false;
  });
}

EMSCRIPTEN_KEEPALIVE int ps_analysis_probe_append_move(int to_x, int to_y) {
  if (!analysis_probe_match || !analysis_probe_profile.has_value()) {
    analysis_probe_error = "no analysis probe has been started";
    return 0;
  }
  return analysis_probe_command([&] {
    if (analysis_probe_ran) {
      throw std::invalid_argument(
          "cannot append an opening move after the analysis probe ran");
    }
    if (ps::is_terminal(analysis_probe_match->state())) {
      throw std::invalid_argument(
          "the analysis probe opening extends past a terminal state");
    }
    analysis_probe_match->play(ps::Move{ps::Point{to_x, to_y}});
  });
}

EMSCRIPTEN_KEEPALIVE int ps_analysis_probe_run() {
  if (!analysis_probe_match || !analysis_probe_profile.has_value()) {
    analysis_probe_error = "no analysis probe has been started";
    return 0;
  }
  return analysis_probe_command([&] {
    if (analysis_probe_ran) {
      throw std::invalid_argument(
          "the analysis probe requires a fresh start for every search");
    }
    if (ps::is_terminal(analysis_probe_match->state())) {
      throw std::invalid_argument(
          "the analysis probe cannot search a terminal opening");
    }
    if (!analysis_probe_match->history().empty() &&
        analysis_probe_match->history().back().extra_turn) {
      throw std::invalid_argument(
          "the analysis probe requires a possession-boundary opening");
    }

    ps::CompleteTurnAnalyzer analyzer(*analysis_probe_profile);
    analysis_probe_result = analyzer.analyze(analysis_probe_match->state());
    analysis_probe_ran = true;
  });
}

EMSCRIPTEN_KEEPALIVE const char *ps_analysis_probe_result_json() {
  try {
    analysis_probe_result_cache = serialize_analysis_probe_result();
    analysis_probe_error.clear();
    return analysis_probe_result_cache.c_str();
  } catch (const std::exception &error) {
    analysis_probe_error = error.what();
    return nullptr;
  }
}

EMSCRIPTEN_KEEPALIVE const char *ps_analysis_probe_last_error() {
  return analysis_probe_error.c_str();
}

}  // extern "C"
