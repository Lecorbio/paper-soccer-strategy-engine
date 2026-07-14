#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
  ReplayComplete,
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

struct WebBotSearchDiagnostic {
  std::size_t ply{};
  Player player{Player::One};
  Point from{};
  Move chosen_move{};
  std::uint32_t requested_iterations{};
  std::uint64_t decision_time_ns{};
  SearchStats stats{};
};

class WebGameSession {
 public:
  WebGameSession(Player human_player, BotConfig bot_config,
                 const RulesConfig &config = {}, std::uint32_t session_id = 0);
  WebGameSession(Player human_player, std::uint64_t bot_seed,
                 const RulesConfig &config = {}, std::uint32_t session_id = 0);

  const Match &match() const noexcept;
  Player human_player() const noexcept;
  Player bot_player() const noexcept;
  const BotConfig &bot_config() const noexcept;
  std::uint64_t bot_seed() const noexcept;
  std::uint32_t session_id() const noexcept;
  std::uint64_t revision() const noexcept;
  const std::vector<WebBotSearchDiagnostic> &bot_searches() const noexcept;

  std::string snapshot_json() const;
  std::string human_match_json() const;

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
  BotConfig bot_config_;
  std::unique_ptr<Bot> bot_;
  std::uint32_t session_id_;
  std::uint64_t revision_{0};
  std::vector<WebBotSearchDiagnostic> bot_searches_{};
};

class WebBotReplaySession {
 public:
  WebBotReplaySession(BotConfig player_one, BotConfig player_two,
                      std::size_t max_plies = 512,
                      const RulesConfig &config = {},
                      std::uint32_t session_id = 0);

  const Match &match() const noexcept;
  const BotConfig &player_one_config() const noexcept;
  const BotConfig &player_two_config() const noexcept;
  std::size_t max_plies() const noexcept;
  std::uint32_t session_id() const noexcept;
  std::uint64_t revision() const noexcept;
  bool done() const noexcept;
  bool truncated() const noexcept;

  std::string snapshot_json() const;
  WebGameCommandResult play_next(std::uint64_t expected_revision);

 private:
  WebGameCommandResult success(PlayedMove move) const;
  WebGameCommandResult failure(WebGameErrorCode code, std::string message) const;

  Match match_;
  BotConfig player_one_config_;
  BotConfig player_two_config_;
  std::unique_ptr<Bot> player_one_bot_;
  std::unique_ptr<Bot> player_two_bot_;
  std::size_t max_plies_;
  std::uint32_t session_id_;
  std::uint64_t revision_{0};
  bool done_{false};
  bool truncated_{false};
};

}  // namespace papersoccer
