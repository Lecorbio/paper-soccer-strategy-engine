#include "papersoccer/web_game.hpp"

#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "papersoccer/rules.hpp"

namespace papersoccer {

namespace {

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

void write_player(std::ostream &out, Player player, Player human_player,
                  std::uint64_t bot_seed) {
  if (player == human_player) {
    out << "{\"kind\":\"Human\"}";
    return;
  }
  out << "{\"kind\":\"RandomBot\",\"seed\":\"" << bot_seed << "\"}";
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
  }
  return "unknown";
}

WebGameSession::WebGameSession(Player human_player, std::uint64_t bot_seed,
                               const RulesConfig &config, std::uint32_t session_id)
    : match_(config),
      human_player_(human_player),
      bot_seed_(bot_seed),
      bot_(bot_seed),
      session_id_(session_id) {
  if (human_player != Player::One && human_player != Player::Two) {
    throw std::invalid_argument("human player must be Player::One or Player::Two");
  }
}

const Match &WebGameSession::match() const noexcept { return match_; }

Player WebGameSession::human_player() const noexcept { return human_player_; }

Player WebGameSession::bot_player() const noexcept { return opponent(human_player_); }

std::uint64_t WebGameSession::bot_seed() const noexcept { return bot_seed_; }

std::uint32_t WebGameSession::session_id() const noexcept { return session_id_; }

std::uint64_t WebGameSession::revision() const noexcept { return revision_; }

std::string WebGameSession::snapshot_json() const {
  const GameState &state = match_.state();
  const std::vector<Move> legal = match_.legal_moves();
  const std::vector<PlayedMove> &history = match_.history();
  const Point start = state.path.empty()
                          ? Point{state.config.width / 2, state.config.height / 2 + 1}
                          : state.path.front();

  std::ostringstream out;
  out.imbue(std::locale::classic());
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

  out << "],\"replay\":{\"schema\":\"papersoccer.replay.v2\",\"rules\":{"
      << "\"width\":" << state.config.width << ",\"height\":" << state.config.height
      << "},\"players\":{\"one\":";
  write_player(out, Player::One, human_player_, bot_seed_);
  out << ",\"two\":";
  write_player(out, Player::Two, human_player_, bot_seed_);
  out << "},\"start\":";
  write_point(out, start);
  out << ",\"status\":\"" << status_to_json(state.status) << "\",\"winner\":";
  write_winner(out, state);
  out << ",\"truncated\":false,\"moves\":[";

  for (std::size_t i = 0; i < history.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    write_played_move(out, history[i]);
  }

  out << "]}}";
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

  const Move chosen = bot_.choose_move(match_.state());
  const PlayedMove played = match_.play(chosen);
  ++revision_;
  return success(played);
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

}  // namespace papersoccer
