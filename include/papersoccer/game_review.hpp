#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "papersoccer/bot.hpp"
#include "papersoccer/match.hpp"
#include "papersoccer/types.hpp"

namespace papersoccer {

enum class ReviewMode { Fast, Deep };

// ProofStatus is always oriented to the player whose possession is being
// reviewed, rather than to Player One or to the player in the resulting state.
enum class ProofStatus { Unknown, ProvenWin, ProvenLoss };

enum class PossessionGrade {
  Forced,
  Best,
  Good,
  Inaccuracy,
  Mistake,
  Blunder,
  Unclear,
};

struct PossessionGradingInput {
  double estimated_loss_percentage_points{};
  bool forced{};
  bool action_matched{};
  bool winning_terminal{};
  bool required_search_completed{true};
  bool deterministic_engine_estimate{true};
  ProofStatus before_proof{ProofStatus::Unknown};
  ProofStatus after_proof{ProofStatus::Unknown};
  std::optional<PossessionGrade> fast_grade{};
};

struct PossessionGradingResult {
  PossessionGrade grade{PossessionGrade::Unclear};
  bool borderline{};
};

PossessionGradingResult grade_possession(
    const PossessionGradingInput &input) noexcept;

struct CompleteTurnAnalysis {
  std::vector<Move> action{};
  int root_score{};  // Positive means better for Player One.
  CompleteTurnSearchStats stats{};
};

class CompleteTurnAnalyzer {
 public:
  explicit CompleteTurnAnalyzer(CompleteTurnAnalysisConfig config);

  const CompleteTurnAnalysisConfig &config() const noexcept;
  CompleteTurnAnalysis analyze(const GameState &state) const;

 private:
  CompleteTurnAnalysisConfig config_;
};

struct ExactEndgameResult {
  ProofStatus status{ProofStatus::Unknown};  // Relative to state.to_move.
  std::optional<Player> winner{};
  std::optional<std::uint32_t> distance{};  // Physical edges to terminal.
  std::vector<Move> action{};               // Complete current possession.
  std::uint64_t nodes{};
  std::uint64_t cache_hits{};
  std::size_t reachable_edge_count{};
  bool budget_exhausted{};
};

class ExactEndgameSolver {
 public:
  static constexpr std::size_t maximum_reachable_edges{18};
  static constexpr std::uint64_t maximum_nodes{250'000};

  ExactEndgameResult solve(const GameState &state) const;
};

// A calibration is valid only for the exact profile named here. The review
// session rejects a profile mismatch instead of silently reusing coefficients.
struct GameReviewCalibration {
  std::string identity{};
  std::string profile_name{};
  double intercept{};
  double score_coefficient{};
  std::string evidence_profile_identity{};
  std::string profile_sha256{};
  std::string mapping_sha256{};

  double estimated_win_chance(int oriented_score) const noexcept;
};

struct GameReviewConfig {
  ReviewMode mode{ReviewMode::Fast};
  CompleteTurnAnalysisConfig fast_profile{
      CompleteTurnAnalysisConfig::fast()};
  CompleteTurnAnalysisConfig deep_profile{
      CompleteTurnAnalysisConfig::deep(100'000)};
  GameReviewCalibration fast_calibration{};
  GameReviewCalibration deep_calibration{};

  static GameReviewConfig locked(ReviewMode mode);
  static bool has_locked_profile() noexcept;
};

struct BoundaryReviewDiagnostics {
  bool exact{};
  ProofStatus proof{ProofStatus::Unknown};
  std::optional<Player> proven_winner{};
  std::optional<std::uint32_t> proof_distance{};
  std::size_t reachable_edge_count{};
  std::uint64_t oracle_nodes{};
  std::uint64_t oracle_cache_hits{};
  bool oracle_budget_exhausted{};
  int oriented_score{};
  double estimated_win_chance{};
  CompleteTurnSearchStats search{};
};

struct PossessionReview {
  std::size_t possession{};  // Zero based.
  std::size_t first_ply{};   // Zero-based edge index in the source replay.
  std::size_t edge_count{};
  Player player{Player::One};
  bool terminal{};
  bool truncated{};
  bool forced{};
  bool analyzed{};
  std::vector<Move> played_action{};
  std::vector<Move> recommended_action{};
  std::optional<std::size_t> first_divergence{};  // Within the possession.
  PossessionGrade grade{PossessionGrade::Unclear};
  double estimated_loss_percentage_points{};
  bool borderline{};
  ProofStatus proof{ProofStatus::Unknown};
  BoundaryReviewDiagnostics before{};
  BoundaryReviewDiagnostics after{};
  bool deterministic_engine_estimate{true};
  std::optional<PossessionGrade> fast_grade{};
};

struct GameReviewSnapshot {
  ReviewMode mode{ReviewMode::Fast};
  bool finalized{};
  bool cancelled{};
  bool complete{};
  std::size_t completed_steps{};
  std::size_t total_steps{};
  Status replay_status{Status::InProgress};
  std::vector<Move> source_replay{};
  std::vector<PossessionReview> possessions{};
};

struct DeclaredReviewMove {
  std::size_t ply{};  // One based, matching PlayedMove::ply.
  Player player{Player::One};
  Point from{};
  Point to{};
  bool extra_turn{};
  Status status_after{Status::InProgress};
};

struct DeclaredReviewOutcome {
  Status status{Status::InProgress};
  std::optional<Player> winner{};
  bool truncated{};
};

class GameReviewSession {
 public:
  explicit GameReviewSession(GameReviewConfig config,
                             RulesConfig rules = {});
  ~GameReviewSession();

  GameReviewSession(const GameReviewSession &) = delete;
  GameReviewSession &operator=(const GameReviewSession &) = delete;
  GameReviewSession(GameReviewSession &&) noexcept;
  GameReviewSession &operator=(GameReviewSession &&) noexcept;

  void append_move(Move move);
  void append_move(const DeclaredReviewMove &move);
  void finalize();
  void finalize(const DeclaredReviewOutcome &declared_outcome);

  // Performs at most one synchronous possession analysis. It returns true if
  // work was performed. Cancellation is observed between these calls.
  bool step();
  void cancel() noexcept;

  const GameReviewSnapshot &snapshot() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class DeepTurnSearchBot final : public Bot {
 public:
  explicit DeepTurnSearchBot(std::uint64_t max_nodes);
  explicit DeepTurnSearchBot(CompleteTurnAnalysisConfig config);

  std::string_view name() const noexcept override;
  Move choose_move(const GameState &state) override;
  const CompleteTurnAnalysisConfig &config() const noexcept;
  const CompleteTurnSearchStats &last_search_stats() const noexcept;

 private:
  CompleteTurnAnalysisConfig config_;
  CompleteTurnSearchStats last_search_stats_{};
  std::vector<Move> cached_action_{};
  std::size_t next_cached_move_{};
  std::optional<GameState> expected_state_{};
  std::uint64_t searches_{};

  void clear_cache() noexcept;
};

}  // namespace papersoccer
