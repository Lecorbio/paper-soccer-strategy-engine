#include "papersoccer/game_review.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "papersoccer/rules.hpp"

#if PAPERSOCCER_GAME_REVIEW_LOCK_PRESENT
#include "../../benchmarks/game_review_gate/game_review_calibration_lock.hpp"
#endif

namespace papersoccer {
namespace {

void validate_calibration(const GameReviewCalibration &calibration,
                          const CompleteTurnAnalysisConfig &profile) {
  if (calibration.identity.empty()) {
    throw std::invalid_argument("game-review calibration identity is empty");
  }
  if (calibration.profile_name != profile.profile_name()) {
    throw std::invalid_argument(
        "game-review calibration does not match its search profile");
  }
  if (!std::isfinite(calibration.intercept) ||
      !std::isfinite(calibration.score_coefficient) ||
      calibration.score_coefficient <= 0.0) {
    throw std::invalid_argument(
        "game-review calibration coefficients are invalid");
  }
}

ProofStatus proof_for_player(std::optional<Player> winning_player,
                             Player perspective) noexcept {
  if (!winning_player.has_value()) {
    return ProofStatus::Unknown;
  }
  return *winning_player == perspective ? ProofStatus::ProvenWin
                                        : ProofStatus::ProvenLoss;
}

int orient_score(int player_one_score, Player perspective) noexcept {
  return perspective == Player::One ? player_one_score : -player_one_score;
}

std::optional<std::size_t> first_divergence(
    const std::vector<Move> &played, const std::vector<Move> &recommended) {
  const std::size_t shared = std::min(played.size(), recommended.size());
  for (std::size_t index = 0; index < shared; ++index) {
    if (!(played[index] == recommended[index])) {
      return index;
    }
  }
  if (played.size() != recommended.size()) {
    return shared;
  }
  return std::nullopt;
}

bool is_borderline_loss(double loss) noexcept {
  for (const double threshold : {2.0, 5.0, 10.0, 20.0}) {
    if (std::abs(loss - threshold) <= 1.0) {
      return true;
    }
  }
  return false;
}

PossessionGrade estimated_grade(double loss) noexcept {
  if (loss < 2.0) {
    return PossessionGrade::Best;
  }
  if (loss < 5.0) {
    return PossessionGrade::Good;
  }
  if (loss < 10.0) {
    return PossessionGrade::Inaccuracy;
  }
  if (loss < 20.0) {
    return PossessionGrade::Mistake;
  }
  return PossessionGrade::Blunder;
}

std::optional<int> grade_band(PossessionGrade grade) noexcept {
  switch (grade) {
    case PossessionGrade::Forced:
    case PossessionGrade::Best:
      return 0;
    case PossessionGrade::Good:
      return 1;
    case PossessionGrade::Inaccuracy:
      return 2;
    case PossessionGrade::Mistake:
      return 3;
    case PossessionGrade::Blunder:
      return 4;
    case PossessionGrade::Unclear:
      return std::nullopt;
  }
  return std::nullopt;
}

bool declared_move_matches(const PlayedMove &played,
                           const DeclaredReviewMove &declared) noexcept {
  return played.ply == declared.ply && played.player == declared.player &&
         played.from == declared.from && played.to == declared.to &&
         played.extra_turn == declared.extra_turn &&
         played.status_after == declared.status_after;
}

}  // namespace

double GameReviewCalibration::estimated_win_chance(
    int oriented_score) const noexcept {
  const double logit =
      intercept + score_coefficient * static_cast<double>(oriented_score);
  if (logit >= 0.0) {
    const double tail = std::exp(-logit);
    return 100.0 / (1.0 + tail);
  }
  const double head = std::exp(logit);
  return 100.0 * head / (1.0 + head);
}

bool GameReviewConfig::has_locked_profile() noexcept {
  return PAPERSOCCER_GAME_REVIEW_LOCK_PRESENT != 0;
}

GameReviewConfig GameReviewConfig::locked(ReviewMode requested_mode) {
#if PAPERSOCCER_GAME_REVIEW_LOCK_PRESENT
  const auto calibration = [](const game_review_lock::Calibration &locked) {
    return GameReviewCalibration{
        locked.identity,
        locked.search_profile_name,
        locked.raw_intercept,
        locked.raw_score_coefficient,
        locked.evidence_profile_id,
        locked.profile_sha256,
        locked.mapping_sha256,
    };
  };
  GameReviewConfig config;
  config.mode = requested_mode;
  config.fast_profile = CompleteTurnAnalysisConfig::fast();
  config.deep_profile = CompleteTurnAnalysisConfig::deep(
      game_review_lock::selected_deep_nodes);
  config.fast_calibration = calibration(game_review_lock::fast_calibration);
  config.deep_calibration = calibration(game_review_lock::deep_calibration);
  return config;
#else
  (void)requested_mode;
  throw std::runtime_error(
      "the frozen validation calibration lock is not available");
#endif
}

PossessionGradingResult grade_possession(
    const PossessionGradingInput &input) noexcept {
  PossessionGradingResult result;
  result.borderline = input.deterministic_engine_estimate &&
                      is_borderline_loss(
                          input.estimated_loss_percentage_points);
  if (input.winning_terminal) {
    result.grade = PossessionGrade::Best;
  } else if (!input.required_search_completed) {
    result.grade = PossessionGrade::Unclear;
  } else if (input.before_proof != ProofStatus::ProvenLoss &&
             input.after_proof == ProofStatus::ProvenLoss) {
    result.grade = PossessionGrade::Blunder;
  } else if (input.forced) {
    result.grade = PossessionGrade::Forced;
  } else if (input.action_matched) {
    result.grade = PossessionGrade::Best;
  } else {
    result.grade = estimated_grade(input.estimated_loss_percentage_points);
  }

  if (input.fast_grade.has_value()) {
    const std::optional<int> fast_band = grade_band(*input.fast_grade);
    const std::optional<int> deep_band = grade_band(result.grade);
    if (!fast_band.has_value() || !deep_band.has_value() ||
        std::abs(*fast_band - *deep_band) >= 2) {
      result.grade = PossessionGrade::Unclear;
    }
  }
  return result;
}

class GameReviewSession::Impl {
 public:
  Impl(GameReviewConfig requested_config, RulesConfig rules)
      : config(std::move(requested_config)), match(rules) {
    if (!config.fast_profile.is_fast_profile()) {
      throw std::invalid_argument(
          "Game Review Fast mode requires the fixed fast-50k profile");
    }
    validate_calibration(config.fast_calibration, config.fast_profile);
    if (config.mode == ReviewMode::Deep) {
      if (!config.deep_profile.is_deep_profile()) {
        throw std::invalid_argument(
            "Game Review Deep mode requires a fixed Deep profile");
      }
      validate_calibration(config.deep_calibration, config.deep_profile);
    }
    states.push_back(match.state());
    view.mode = config.mode;
  }

  struct PossessionState {
    GameState before{};
    GameState after{};
  };

  GameReviewConfig config;
  Match match;
  std::vector<GameState> states{};
  std::vector<PossessionState> possession_states{};
  GameReviewSnapshot view{};
  std::size_t work_cursor{};

  void append(Move move) {
    require_appendable();
    match.play(move);
    view.source_replay.push_back(move);
    states.push_back(match.state());
  }

  void append(const DeclaredReviewMove &declared) {
    require_appendable();
    const PlayedMove played = match.play(Move{declared.to});
    if (!declared_move_matches(played, declared)) {
      (void)match.undo();
      throw std::invalid_argument(
          "declared replay move does not match authoritative Match history");
    }
    view.source_replay.push_back(Move{declared.to});
    states.push_back(match.state());
  }

  void finish(std::optional<DeclaredReviewOutcome> declared) {
    if (view.finalized) {
      throw std::logic_error("game-review replay is already finalized");
    }
    if (view.cancelled) {
      throw std::logic_error("cannot finalize a cancelled game review");
    }

    if (declared.has_value()) {
      const Status actual_status = match.state().status;
      const std::optional<Player> actual_winner = winner(match.state());
      const bool actually_truncated = actual_status == Status::InProgress;
      if (declared->status != actual_status ||
          declared->winner != actual_winner ||
          declared->truncated != actually_truncated) {
        throw std::invalid_argument(
            "declared replay outcome is inconsistent with authoritative Match state");
      }
    }

    partition_possessions();
    view.replay_status = match.state().status;
    view.finalized = true;
    view.total_steps = view.possessions.size() *
                       (config.mode == ReviewMode::Deep ? 2U : 1U);
    view.complete = view.total_steps == 0;
  }

  bool do_step() {
    if (!view.finalized) {
      throw std::logic_error("game-review replay must be finalized before analysis");
    }
    if (view.cancelled || view.complete) {
      return false;
    }

    const std::size_t count = view.possessions.size();
    const bool deep_stage = config.mode == ReviewMode::Deep &&
                            work_cursor >= count;
    const std::size_t possession_index = work_cursor % count;
    if (!deep_stage) {
      analyze_into(possession_index, config.fast_profile,
                   config.fast_calibration, false);
    } else {
      analyze_into(possession_index, config.deep_profile,
                   config.deep_calibration, true);
    }
    ++work_cursor;
    view.completed_steps = work_cursor;
    view.complete = work_cursor >= view.total_steps;
    return true;
  }

 private:
  void require_appendable() const {
    if (view.finalized) {
      throw std::logic_error("cannot append to a finalized game review");
    }
    if (view.cancelled) {
      throw std::logic_error("cannot append to a cancelled game review");
    }
  }

  void partition_possessions() {
    const std::vector<PlayedMove> &history = match.history();
    std::size_t first = 0;
    while (first < history.size()) {
      const Player mover = history[first].player;
      std::size_t last = first;
      while (last + 1U < history.size() &&
             history[last + 1U].player == mover &&
             history[last].status_after == Status::InProgress) {
        ++last;
      }

      PossessionReview review;
      review.possession = view.possessions.size();
      review.first_ply = first;
      review.edge_count = last - first + 1U;
      review.player = mover;
      review.terminal =
          history[last].status_after != Status::InProgress;
      review.truncated = last + 1U == history.size() && !review.terminal &&
                         history[last].extra_turn;
      review.forced = true;
      review.played_action.reserve(review.edge_count);
      for (std::size_t ply = first; ply <= last; ++ply) {
        review.played_action.push_back(Move{history[ply].to});
        review.forced = review.forced && legal_moves(states[ply]).size() == 1U;
      }
      view.possessions.push_back(std::move(review));
      possession_states.push_back(PossessionState{states[first],
                                                  states[last + 1U]});
      first = last + 1U;
    }
  }

  BoundaryReviewDiagnostics analyze_boundary(
      const GameState &state, Player perspective,
      const CompleteTurnAnalysisConfig &profile,
      const GameReviewCalibration &calibration,
      std::vector<Move> *recommended) const {
    BoundaryReviewDiagnostics diagnostics;
    if (is_terminal(state)) {
      diagnostics.exact = true;
      diagnostics.proven_winner = winner(state);
      diagnostics.proof =
          proof_for_player(diagnostics.proven_winner, perspective);
      diagnostics.estimated_win_chance =
          diagnostics.proof == ProofStatus::ProvenWin ? 100.0 : 0.0;
      return diagnostics;
    }

    const ExactEndgameResult exact = ExactEndgameSolver{}.solve(state);
    diagnostics.reachable_edge_count = exact.reachable_edge_count;
    diagnostics.oracle_nodes = exact.nodes;
    diagnostics.oracle_cache_hits = exact.cache_hits;
    diagnostics.oracle_budget_exhausted = exact.budget_exhausted;
    if (exact.status != ProofStatus::Unknown) {
      diagnostics.exact = true;
      diagnostics.proven_winner = exact.winner;
      diagnostics.proof = proof_for_player(exact.winner, perspective);
      diagnostics.proof_distance = exact.distance;
      diagnostics.estimated_win_chance =
          diagnostics.proof == ProofStatus::ProvenWin ? 100.0 : 0.0;
      if (recommended != nullptr) {
        *recommended = exact.action;
      }
      return diagnostics;
    }

    const CompleteTurnAnalysis analysis =
        CompleteTurnAnalyzer(profile).analyze(state);
    diagnostics.oriented_score = orient_score(analysis.root_score, perspective);
    diagnostics.estimated_win_chance =
        calibration.estimated_win_chance(diagnostics.oriented_score);
    diagnostics.search = analysis.stats;
    if (recommended != nullptr) {
      *recommended = analysis.action;
    }
    return diagnostics;
  }

  void analyze_into(std::size_t index,
                    const CompleteTurnAnalysisConfig &profile,
                    const GameReviewCalibration &calibration,
                    bool deep_stage) {
    PossessionReview result = view.possessions[index];
    if (result.truncated) {
      result.analyzed = true;
      if (deep_stage) {
        result.fast_grade = view.possessions[index].grade;
      }
      result.grade = PossessionGrade::Unclear;
      view.possessions[index] = std::move(result);
      return;
    }

    const PossessionState &position = possession_states[index];
    result.before = analyze_boundary(position.before, result.player, profile,
                                     calibration,
                                     &result.recommended_action);
    result.after = analyze_boundary(position.after, result.player, profile,
                                    calibration, nullptr);
    result.analyzed = true;
    result.proof = result.before.proof;
    result.first_divergence =
        first_divergence(result.played_action, result.recommended_action);
    result.estimated_loss_percentage_points = std::clamp(
        result.before.estimated_win_chance -
            result.after.estimated_win_chance,
        0.0, 100.0);
    result.deterministic_engine_estimate =
        !(result.before.exact && result.after.exact);
    if (deep_stage) {
      result.fast_grade = view.possessions[index].grade;
    }
    const bool required_search_completed =
        (result.before.exact ||
         result.before.search.completed_turn_depth != 0) &&
        (result.terminal || result.after.exact ||
         result.after.search.completed_turn_depth != 0);
    const PossessionGradingResult grading = grade_possession({
        result.estimated_loss_percentage_points,
        result.forced,
        !result.first_divergence.has_value(),
        result.terminal && result.after.proof == ProofStatus::ProvenWin,
        required_search_completed,
        result.deterministic_engine_estimate,
        result.before.proof,
        result.after.proof,
        result.fast_grade,
    });
    result.grade = grading.grade;
    result.borderline = grading.borderline;
    view.possessions[index] = std::move(result);
  }
};

GameReviewSession::GameReviewSession(GameReviewConfig config,
                                     RulesConfig rules)
    : impl_(std::make_unique<Impl>(std::move(config), rules)) {}

GameReviewSession::~GameReviewSession() = default;
GameReviewSession::GameReviewSession(GameReviewSession &&) noexcept = default;
GameReviewSession &GameReviewSession::operator=(GameReviewSession &&) noexcept =
    default;

void GameReviewSession::append_move(Move move) { impl_->append(move); }

void GameReviewSession::append_move(const DeclaredReviewMove &move) {
  impl_->append(move);
}

void GameReviewSession::finalize() { impl_->finish(std::nullopt); }

void GameReviewSession::finalize(
    const DeclaredReviewOutcome &declared_outcome) {
  impl_->finish(declared_outcome);
}

bool GameReviewSession::step() { return impl_->do_step(); }

void GameReviewSession::cancel() noexcept {
  impl_->view.cancelled = true;
  impl_->view.complete = false;
}

const GameReviewSnapshot &GameReviewSession::snapshot() const noexcept {
  return impl_->view;
}

}  // namespace papersoccer
