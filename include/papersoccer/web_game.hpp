#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "papersoccer/bot.hpp"
#include "papersoccer/match.hpp"

namespace papersoccer {

enum class WebGameErrorCode {
  StaleSession,
  StaleRevision,
  WrongTurn,
  TerminalGame,
  MoveOutOfRange,
  NoLegalMoves,
};

std::string_view web_game_error_code_name(WebGameErrorCode code) noexcept;

struct WebGameError {
  WebGameErrorCode code{WebGameErrorCode::WrongTurn};
  std::string message{};
};

struct WebGameCommandResult {
  std::uint64_t revision{};
  std::optional<PlayedMove> move{};
  std::optional<WebGameError> error{};

  bool ok() const noexcept { return !error.has_value(); }
};

class WebGameSession {
 public:
  WebGameSession(Player human_player, std::uint64_t bot_seed,
                 const RulesConfig &config = {}, std::uint32_t session_id = 0);

  const Match &match() const noexcept;
  Player human_player() const noexcept;
  Player bot_player() const noexcept;
  std::uint64_t bot_seed() const noexcept;
  std::uint32_t session_id() const noexcept;
  std::uint64_t revision() const noexcept;

  std::string snapshot_json() const;

  WebGameCommandResult play_human(std::uint64_t expected_revision,
                                  std::size_t move_id);
  WebGameCommandResult play_bot(std::uint64_t expected_revision);

 private:
  WebGameCommandResult validate_command(std::uint64_t expected_revision,
                                        Player expected_player) const;
  WebGameCommandResult success(PlayedMove move) const;
  WebGameCommandResult failure(WebGameErrorCode code, std::string message) const;

  Match match_;
  Player human_player_;
  std::uint64_t bot_seed_;
  RandomBot bot_;
  std::uint32_t session_id_;
  std::uint64_t revision_{0};
};

}  // namespace papersoccer
