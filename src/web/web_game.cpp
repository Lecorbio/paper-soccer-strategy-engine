#include "papersoccer/web_game.hpp"

#include <chrono>
#include <iomanip>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "papersoccer/game_review.hpp"
#include "papersoccer/rules.hpp"

namespace papersoccer {

namespace {

using Clock = std::chrono::steady_clock;

std::string_view player_to_json(Player player) noexcept {
  return player == Player::One ? "one" : "two";
}

std::string_view status_to_json(Status status) noexcept {
  switch (status) {
    case Status::InProgress:
      return "inProgress";
    case Status::WonByOne:
      return "wonByOne";
    case Status::WonByTwo:
      return "wonByTwo";
  }
  return "unknown";
}

void write_point(std::ostream &out, Point point) {
  out << "{\"x\":" << point.x << ",\"y\":" << point.y << "}";
}

void write_winner(std::ostream &out, const GameState &state) {
  const std::optional<Player> winning_player = winner(state);
  if (winning_player.has_value()) {
    out << '"' << player_to_json(*winning_player) << '"';
  } else {
    out << "null";
  }
}

void write_bot(std::ostream &out, const BotConfig &config) {
  if (config.kind == BotKind::DeepTurnSearch) {
    const CompleteTurnAnalysisConfig profile =
        GameReviewConfig::locked(ReviewMode::Deep).deep_profile;
    out << "{\"kind\":\"DeepTurnSearchBot\",\"profile\":\""
        << profile.profile_name() << "\",\"maxNodes\":"
        << profile.max_nodes
        << ",\"maxTurnDepth\":" << profile.max_turn_depth
        << ",\"transpositionEntries\":"
        << profile.transposition_table_entries
        << ",\"evaluationCacheEntries\":"
        << profile.evaluation_table_entries
        << ",\"wallClock\":false,\"replayCorrections\":false,"
           "\"learnedValueBlendPercent\":0}";
    return;
  }
  if (config.kind == BotKind::Rank5Derived) {
    out << "{\"kind\":\"Rank5DerivedBot\",\"profile\":\""
        << Rank5DerivedBot::profile_name() << "\",\"maxNodes\":"
        << Rank5DerivedConfig::profile_max_nodes
        << ",\"modelBlendPercent\":"
        << Rank5DerivedConfig::default_replay_value_blend_percent
        << ",\"replayBookEnabled\":"
        << (Rank5DerivedConfig::default_replay_corrections ? "true" : "false")
        << ",\"originalArtifact\":{\"name\":\""
        << Rank5DerivedBot::original_artifact_name() << "\",\"rank\":"
        << Rank5DerivedBot::original_rank << ",\"fieldSize\":"
        << Rank5DerivedBot::original_field_size << ",\"submissionId\":\""
        << Rank5DerivedBot::original_submission_id() << "\",\"sha256\":\""
        << Rank5DerivedBot::original_sha256() << "\"}}";
    return;
  }
  out << "{\"kind\":\"" << bot_kind_name(config.kind) << "\",\"seed\":\""
      << config.seed << '"';
  if (config.kind == BotKind::Mcts) {
    out << ",\"iterations\":" << config.mcts_iterations;
  } else if (config.kind == BotKind::AlphaBeta ||
             config.kind == BotKind::JacekInspired) {
    out << ",\"depth\":" << config.alpha_beta_depth;
    if (config.kind == BotKind::JacekInspired) {
      out << ",\"maxNodes\":" << config.alpha_beta_max_nodes
          << ",\"maxTimeMs\":" << config.alpha_beta_max_time_ms
          << ",\"modelSha256\":\""
          << JacekInspiredBot::model_sha256() << '"';
    }
  }
  out << '}';
}

std::uint64_t elapsed_nanoseconds(Clock::time_point start,
                                  Clock::time_point end) noexcept {
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return elapsed <= 0 ? 0 : static_cast<std::uint64_t>(elapsed);
}

void write_player(std::ostream &out, Player player, Player human_player,
                  const BotConfig &bot_config) {
  if (player == human_player) {
    out << "{\"kind\":\"Human\"}";
    return;
  }
  write_bot(out, bot_config);
}

void write_played_move(std::ostream &out, const PlayedMove &move) {
  out << "{\"ply\":" << move.ply << ",\"player\":\""
      << player_to_json(move.player) << "\",\"from\":";
  write_point(out, move.from);
  out << ",\"to\":";
  write_point(out, move.to);
  out << ",\"extraTurn\":" << (move.extra_turn ? "true" : "false")
      << ",\"statusAfter\":\"" << status_to_json(move.status_after) << "\"}";
}

void write_bot_search(std::ostream &out,
                      const WebBotSearchDiagnostic &search) {
  out << "{\"ply\":" << search.ply << ",\"player\":\""
      << player_to_json(search.player) << "\",\"chosenMove\":{\"from\":";
  write_point(out, search.from);
  out << ",\"to\":";
  write_point(out, search.chosen_move.to);
  out << "},\"decisionTimeNs\":" << search.decision_time_ns;
  if (search.deep_turn_search_stats.has_value()) {
    const CompleteTurnSearchStats &stats = *search.deep_turn_search_stats;
    const CompleteTurnAnalysisConfig profile =
        GameReviewConfig::locked(ReviewMode::Deep).deep_profile;
    out << ",\"searchType\":\"deepTurnSearch\""
        << ",\"requestedNodes\":" << profile.max_nodes
        << ",\"visitedNodes\":" << stats.nodes
        << ",\"completedDepth\":" << stats.completed_turn_depth
        << ",\"attemptedDepth\":" << stats.attempted_turn_depth
        << ",\"rootScore\":" << stats.root_score
        << ",\"budgetExhausted\":"
        << (stats.budget_exhausted ? "true" : "false")
        << ",\"plannedActionLength\":" << stats.planned_action_length
        << ",\"currentEdgeIndex\":" << stats.current_edge_index
        << ",\"cachedContinuation\":"
        << (stats.cached_continuation ? "true" : "false") << '}';
    return;
  }
  if (search.rank5_derived_stats.has_value()) {
    const Rank5DerivedSearchStats &stats = *search.rank5_derived_stats;
    out << ",\"searchType\":\"rank5Derived\""
        << ",\"requestedNodes\":" << Rank5DerivedConfig::profile_max_nodes
        << ",\"visitedNodes\":" << stats.nodes
        << ",\"completedDepth\":" << stats.completed_turn_depth
        << ",\"attemptedDepth\":" << stats.attempted_turn_depth
        << ",\"rootScore\":" << stats.root_score
        << ",\"budgetExhausted\":"
        << (stats.budget_exhausted ? "true" : "false")
        << ",\"plannedActionLength\":" << stats.planned_action_length
        << ",\"currentEdgeIndex\":" << stats.current_edge_index
        << ",\"cachedContinuation\":"
        << (stats.cached_continuation ? "true" : "false") << '}';
    return;
  }
  if (search.alpha_beta_stats.has_value()) {
    const AlphaBetaSearchStats &stats = *search.alpha_beta_stats;
    out << ",\"searchType\":\"alphaBeta\""
        << ",\"requestedDepth\":" << search.requested_turn_depth
        << ",\"completedDepth\":" << stats.completed_turn_depth
        << ",\"attemptedDepth\":" << stats.attempted_turn_depth
        << ",\"nodes\":" << stats.nodes
        << ",\"leafEvaluations\":" << stats.leaf_evaluations
        << ",\"terminalNodes\":" << stats.terminal_nodes
        << ",\"cutoffs\":" << stats.cutoffs
        << ",\"transpositionProbes\":" << stats.transposition_probes
        << ",\"transpositionHits\":" << stats.transposition_hits
        << ",\"maxPhysicalPly\":" << stats.max_physical_ply
        << ",\"rootScore\":" << stats.root_score
        << ",\"rootScoreValid\":"
        << (stats.completed_turn_depth > 0 ? "true" : "false")
        << ",\"budgetExhausted\":"
        << (stats.budget_exhausted ? "true" : "false") << '}';
    return;
  }
  out << ",\"requestedIterations\":" << search.requested_iterations
      << ",\"completedIterations\":" << search.stats.iterations
      << ",\"nodes\":" << search.stats.nodes
      << ",\"simulatedPlies\":" << search.stats.simulated_plies
      << ",\"totalRootVisits\":" << search.stats.total_root_visits
      << ",\"reusedVisits\":" << search.stats.reused_visits
      << ",\"maxDepth\":" << search.stats.max_depth
      << ",\"provenNodes\":" << search.stats.proven_nodes
      << ",\"provenWinner\":";
  if (search.stats.proven_winner.has_value()) {
    out << '"' << player_to_json(*search.stats.proven_winner) << '"';
  } else {
    out << "null";
  }
  out << ",\"tacticalProbes\":" << search.stats.tactical_probes
      << ",\"tacticalNodes\":" << search.stats.tactical_nodes
      << ",\"tacticalSolvedPositions\":"
      << search.stats.tactical_solved_positions
      << ",\"tacticalDepthCutoffs\":"
      << search.stats.tactical_depth_cutoffs
      << ",\"tacticalNodeCutoffs\":" << search.stats.tactical_node_cutoffs
      << ",\"maxTacticalDepth\":" << search.stats.max_tactical_depth
      << ",\"rebuildCount\":" << search.stats.rebuild_count
      << ",\"expansionSaturated\":"
      << (search.stats.expansion_saturated ? "true" : "false")
      << ",\"rootValue\":" << search.stats.root_value << '}';
}

void write_bot_searches(std::ostream &out,
                        const std::vector<WebBotSearchDiagnostic> &searches) {
  out << '[';
  for (std::size_t i = 0; i < searches.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    write_bot_search(out, searches[i]);
  }
  out << ']';
}

std::optional<WebBotSearchDiagnostic> make_bot_search_diagnostic(
    const Bot &bot, const BotConfig &config, const PlayedMove &played,
    Point from, Move chosen, Clock::time_point start, Clock::time_point end) {
  if (const auto *mcts = dynamic_cast<const MctsBot *>(&bot)) {
    return WebBotSearchDiagnostic{
        played.ply,
        played.player,
        from,
        chosen,
        config.mcts_iterations,
        elapsed_nanoseconds(start, end),
        mcts->last_search_stats(),
    };
  }
  if (const auto *jacek = dynamic_cast<const JacekInspiredBot *>(&bot)) {
    return WebBotSearchDiagnostic{
        played.ply,
        played.player,
        from,
        chosen,
        0,
        elapsed_nanoseconds(start, end),
        SearchStats{},
        config.alpha_beta_depth,
        jacek->last_search_stats(),
    };
  }
  if (const auto *rank5 = dynamic_cast<const Rank5DerivedBot *>(&bot)) {
    return WebBotSearchDiagnostic{
        played.ply,
        played.player,
        from,
        chosen,
        0,
        elapsed_nanoseconds(start, end),
        SearchStats{},
        0,
        std::nullopt,
        rank5->last_search_stats(),
    };
  }
  if (const auto *deep = dynamic_cast<const DeepTurnSearchBot *>(&bot)) {
    WebBotSearchDiagnostic diagnostic{
        played.ply,
        played.player,
        from,
        chosen,
        0,
        elapsed_nanoseconds(start, end),
    };
    diagnostic.deep_turn_search_stats = deep->last_search_stats();
    return diagnostic;
  }
  return std::nullopt;
}

void write_diagnostics(
    std::ostream &out, const BotConfig &bot_config,
    const std::vector<WebBotSearchDiagnostic> &bot_searches) {
  out << "{\"schema\":\"papersoccer.bot-search-diagnostics.v1\","
         "\"botConfiguration\":";
  write_bot(out, bot_config);
  out << ",\"lastBotSearch\":";
  if (bot_searches.empty()) {
    out << "null";
  } else {
    write_bot_search(out, bot_searches.back());
  }
  out << ",\"botSearches\":";
  write_bot_searches(out, bot_searches);
  out << '}';
}

template <typename PlayerWriter>
void write_replay(std::ostream &out, const GameState &state,
                  const std::vector<PlayedMove> &history, bool truncated,
                  PlayerWriter write_player_metadata) {
  const Point start = state.path.empty()
                          ? Point{state.config.width / 2, state.config.height / 2 + 1}
                          : state.path.front();

  out << "{\"schema\":\"papersoccer.replay.v2\",\"rules\":{"
      << "\"width\":" << state.config.width << ",\"height\":"
      << state.config.height << "},\"players\":{\"one\":";
  write_player_metadata(out, Player::One);
  out << ",\"two\":";
  write_player_metadata(out, Player::Two);
  out << "},\"start\":";
  write_point(out, start);
  out << ",\"status\":\"" << status_to_json(state.status)
      << "\",\"winner\":";
  write_winner(out, state);
  out << ",\"truncated\":" << (truncated ? "true" : "false")
      << ",\"moves\":[";

  for (std::size_t i = 0; i < history.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    write_played_move(out, history[i]);
  }
  out << "]}";
}

}  // namespace

std::string_view web_game_error_code_name(WebGameErrorCode code) noexcept {
  switch (code) {
    case WebGameErrorCode::StaleSession:
      return "stale_session";
    case WebGameErrorCode::StaleRevision:
      return "stale_revision";
    case WebGameErrorCode::WrongTurn:
      return "wrong_turn";
    case WebGameErrorCode::TerminalGame:
      return "terminal_game";
    case WebGameErrorCode::MoveOutOfRange:
      return "move_out_of_range";
    case WebGameErrorCode::NoLegalMoves:
      return "no_legal_moves";
    case WebGameErrorCode::NoMovesToUndo:
      return "no_moves_to_undo";
    case WebGameErrorCode::ReplayComplete:
      return "replay_complete";
  }
  return "unknown";
}

WebGameSession::WebGameSession(Player human_player, BotConfig bot_config,
                               const RulesConfig &config, std::uint32_t session_id)
    : match_(config),
      human_player_(human_player),
      bot_config_(bot_config),
      bot_(make_bot(bot_config_)),
      session_id_(session_id) {
  if (human_player != Player::One && human_player != Player::Two) {
    throw std::invalid_argument("human player must be Player::One or Player::Two");
  }
}

WebGameSession::WebGameSession(Player human_player, std::uint64_t bot_seed,
                               const RulesConfig &config, std::uint32_t session_id)
    : WebGameSession(human_player,
                     BotConfig{BotKind::Random, bot_seed}, config,
                     session_id) {}

const Match &WebGameSession::match() const noexcept { return match_; }

Player WebGameSession::human_player() const noexcept { return human_player_; }

Player WebGameSession::bot_player() const noexcept { return opponent(human_player_); }

const BotConfig &WebGameSession::bot_config() const noexcept { return bot_config_; }

std::uint64_t WebGameSession::bot_seed() const noexcept { return bot_config_.seed; }

std::uint32_t WebGameSession::session_id() const noexcept { return session_id_; }

std::uint64_t WebGameSession::revision() const noexcept { return revision_; }

const std::vector<WebBotSearchDiagnostic> &WebGameSession::bot_searches() const
    noexcept {
  return bot_searches_;
}

std::string WebGameSession::snapshot_json() const {
  const GameState &state = match_.state();
  const std::vector<Move> legal = match_.legal_moves();

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "{\"schema\":\"papersoccer.web-session.v1\",\"sessionId\":" << session_id_
      << ",\"revision\":" << revision_
      << ",\"humanPlayer\":\"" << player_to_json(human_player_) << "\",\"state\":{"
      << "\"ball\":";
  write_point(out, state.ball);
  out << ",\"toMove\":\"" << player_to_json(state.to_move) << "\",\"status\":\""
      << status_to_json(state.status) << "\",\"winner\":";
  write_winner(out, state);
  out << "},\"legalMoves\":[";

  for (std::size_t i = 0; i < legal.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << "{\"id\":" << i << ",\"to\":";
    write_point(out, legal[i].to);
    out << ",\"extraTurn\":"
        << (grants_extra_turn(state, legal[i].to) ? "true" : "false") << '}';
  }

  out << "],\"replay\":";
  write_replay(out, state, match_.history(), false,
               [this](std::ostream &replay_out, Player player) {
                 write_player(replay_out, player, human_player_, bot_config_);
               });
  out << ",\"diagnostics\":";
  write_diagnostics(out, bot_config_, bot_searches_);
  out << '}';
  return out.str();
}

std::string WebGameSession::human_match_json() const {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "{\"schema\":\"papersoccer.human-match.v1\",\"replay\":";
  write_replay(out, match_.state(), match_.history(), false,
               [this](std::ostream &replay_out, Player player) {
                 write_player(replay_out, player, human_player_, bot_config_);
               });
  out << ",\"botConfiguration\":";
  write_bot(out, bot_config_);
  out << ",\"botSearches\":";
  write_bot_searches(out, bot_searches_);
  out << '}';
  return out.str();
}

WebGameCommandResult WebGameSession::play_human(std::uint64_t expected_revision,
                                                std::size_t move_id) {
  const WebGameCommandResult validation = validate_command(expected_revision, human_player_);
  if (!validation.ok()) {
    return validation;
  }

  const std::vector<Move> legal = match_.legal_moves();
  if (move_id >= legal.size()) {
    return failure(WebGameErrorCode::MoveOutOfRange,
                   "move ID is not present in the current legal-move snapshot");
  }

  const PlayedMove played = match_.play(legal[move_id]);
  ++revision_;
  return success(played);
}

WebGameCommandResult WebGameSession::play_bot(std::uint64_t expected_revision) {
  const WebGameCommandResult validation = validate_command(expected_revision, bot_player());
  if (!validation.ok()) {
    return validation;
  }

  if (match_.legal_moves().empty()) {
    return failure(WebGameErrorCode::NoLegalMoves,
                   "the bot cannot move because the game has no legal moves");
  }

  const Point from = match_.state().ball;
  const auto start = Clock::now();
  const Move chosen = bot_->choose_move(match_.state());
  const auto end = Clock::now();
  const PlayedMove played = match_.play(chosen);
  if (auto diagnostic = make_bot_search_diagnostic(
          *bot_, bot_config_, played, from, chosen, start, end)) {
    bot_searches_.push_back(std::move(*diagnostic));
  }
  ++revision_;
  return success(played);
}

WebGameCommandResult WebGameSession::undo(std::uint64_t expected_revision) {
  if (expected_revision != revision_) {
    return failure(WebGameErrorCode::StaleRevision,
                   "command revision does not match the current game revision");
  }

  const std::optional<PlayedMove> undone = match_.undo();
  if (!undone.has_value()) {
    return failure(WebGameErrorCode::NoMovesToUndo,
                   "the game is already at its initial position");
  }

  const std::size_t retained_plies = match_.history().size();
  while (!bot_searches_.empty() && bot_searches_.back().ply > retained_plies) {
    bot_searches_.pop_back();
  }
  ++revision_;
  return success(*undone);
}

WebGameCommandResult WebGameSession::validate_command(
    std::uint64_t expected_revision, Player expected_player) const {
  if (expected_revision != revision_) {
    return failure(WebGameErrorCode::StaleRevision,
                   "command revision does not match the current game revision");
  }
  if (is_terminal(match_.state())) {
    return failure(WebGameErrorCode::TerminalGame,
                   "cannot play a move after the game has finished");
  }
  if (match_.state().to_move != expected_player) {
    return failure(WebGameErrorCode::WrongTurn,
                   "the requested controller does not own the current turn");
  }
  return WebGameCommandResult{revision_, std::nullopt, std::nullopt};
}

WebGameCommandResult WebGameSession::success(PlayedMove move) const {
  return WebGameCommandResult{revision_, move, std::nullopt};
}

WebGameCommandResult WebGameSession::failure(WebGameErrorCode code,
                                             std::string message) const {
  return WebGameCommandResult{
      revision_,
      std::nullopt,
      WebGameError{code, std::move(message)},
  };
}

WebBotReplaySession::WebBotReplaySession(BotConfig player_one,
                                         BotConfig player_two,
                                         std::size_t max_plies,
                                         const RulesConfig &config,
                                         std::uint32_t session_id)
    : match_(config),
      player_one_config_(player_one),
      player_two_config_(player_two),
      player_one_bot_(make_bot(player_one_config_)),
      player_two_bot_(make_bot(player_two_config_)),
      max_plies_(max_plies),
      session_id_(session_id) {
  done_ = is_terminal(match_.state()) || max_plies_ == 0;
  truncated_ = max_plies_ == 0 && !is_terminal(match_.state());
}

const Match &WebBotReplaySession::match() const noexcept { return match_; }

const BotConfig &WebBotReplaySession::player_one_config() const noexcept {
  return player_one_config_;
}

const BotConfig &WebBotReplaySession::player_two_config() const noexcept {
  return player_two_config_;
}

std::size_t WebBotReplaySession::max_plies() const noexcept { return max_plies_; }

std::uint32_t WebBotReplaySession::session_id() const noexcept {
  return session_id_;
}

std::uint64_t WebBotReplaySession::revision() const noexcept { return revision_; }

bool WebBotReplaySession::done() const noexcept { return done_; }

bool WebBotReplaySession::truncated() const noexcept { return truncated_; }

const std::vector<WebBotSearchDiagnostic> &WebBotReplaySession::bot_searches()
    const noexcept {
  return bot_searches_;
}

std::string WebBotReplaySession::snapshot_json() const {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "{\"schema\":\"papersoccer.bot-replay-session.v1\",\"sessionId\":"
      << session_id_ << ",\"revision\":" << revision_ << ",\"done\":"
      << (done_ ? "true" : "false") << ",\"replay\":";
  write_replay(out, match_.state(), match_.history(), truncated_,
               [this](std::ostream &replay_out, Player player) {
                 write_bot(replay_out, player == Player::One
                                           ? player_one_config_
                                           : player_two_config_);
               });
  out << ",\"botSearches\":";
  write_bot_searches(out, bot_searches_);
  out << '}';
  return out.str();
}

WebGameCommandResult WebBotReplaySession::play_next(
    std::uint64_t expected_revision) {
  if (expected_revision != revision_) {
    return failure(WebGameErrorCode::StaleRevision,
                   "command revision does not match the current replay revision");
  }
  if (done_) {
    return failure(WebGameErrorCode::ReplayComplete,
                   "cannot play a move after replay generation has finished");
  }
  if (is_terminal(match_.state())) {
    done_ = true;
    return failure(WebGameErrorCode::ReplayComplete,
                   "cannot play a move after replay generation has finished");
  }
  if (match_.legal_moves().empty()) {
    return failure(WebGameErrorCode::NoLegalMoves,
                   "the bot cannot move because the game has no legal moves");
  }

  const bool player_one = match_.state().to_move == Player::One;
  Bot &bot = player_one ? *player_one_bot_ : *player_two_bot_;
  const BotConfig &config =
      player_one ? player_one_config_ : player_two_config_;
  const Point from = match_.state().ball;
  const auto start = Clock::now();
  const Move chosen = bot.choose_move(match_.state());
  const auto end = Clock::now();
  const PlayedMove played = match_.play(chosen);
  if (auto diagnostic = make_bot_search_diagnostic(
          bot, config, played, from, chosen, start, end)) {
    bot_searches_.push_back(std::move(*diagnostic));
  }
  ++revision_;

  if (is_terminal(match_.state())) {
    done_ = true;
  } else if (match_.history().size() >= max_plies_) {
    done_ = true;
    truncated_ = true;
  }
  return success(played);
}

WebGameCommandResult WebBotReplaySession::success(PlayedMove move) const {
  return WebGameCommandResult{revision_, move, std::nullopt};
}

WebGameCommandResult WebBotReplaySession::failure(WebGameErrorCode code,
                                                  std::string message) const {
  return WebGameCommandResult{
      revision_,
      std::nullopt,
      WebGameError{code, std::move(message)},
  };
}

}  // namespace papersoccer
