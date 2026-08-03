#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <emscripten/emscripten.h>

#include "papersoccer/web_game.hpp"

namespace ps = papersoccer;

namespace {

std::unique_ptr<ps::WebGameSession> session;
std::unique_ptr<ps::WebBotReplaySession> bot_replay_session;
std::string snapshot_cache;
std::string human_match_cache;
std::string bot_replay_snapshot_cache;
std::string last_error;
std::uint32_t next_session_id{1};
constexpr std::uint64_t kWebJacekMaxNodes{20'000};
constexpr std::uint32_t kWebJacekMaxTimeMs{0};
constexpr std::uint64_t kWebRank5DerivedMaxNodes{
    ps::Rank5DerivedConfig::profile_max_nodes};

bool parse_seed(std::string_view input, std::uint64_t &value) noexcept {
  if (input.empty()) {
    return false;
  }

  std::uint64_t parsed = 0;
  for (const char character : input) {
    if (character < '0' || character > '9') {
      return false;
    }
    const auto digit = static_cast<std::uint64_t>(character - '0');
    if (parsed > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
      return false;
    }
    parsed = parsed * 10U + digit;
  }

  value = parsed;
  return true;
}

bool parse_bot_kind(int input, ps::BotKind &kind) noexcept {
  if (input == 0) {
    kind = ps::BotKind::Random;
    return true;
  }
  if (input == 1) {
    kind = ps::BotKind::Mcts;
    return true;
  }
  if (input == 2) {
    kind = ps::BotKind::AlphaBeta;
    return true;
  }
  if (input == 3) {
    kind = ps::BotKind::JacekInspired;
    return true;
  }
  if (input == 4) {
    kind = ps::BotKind::Rank5Derived;
    return true;
  }
  return false;
}

bool parse_bot_config(const char *seed_text, int kind_value,
                      std::uint32_t search_setting, ps::BotConfig &config) {
  if (seed_text == nullptr || !parse_seed(seed_text, config.seed)) {
    last_error = "the bot seed must be an unsigned 64-bit decimal integer";
    return false;
  }
  if (!parse_bot_kind(kind_value, config.kind)) {
    last_error =
        "the bot kind must be 0 (RandomBot), 1 (MctsBot), "
        "2 (AlphaBetaBot), 3 (JacekInspiredBot), or 4 (Rank5DerivedBot)";
    return false;
  }
  if (config.kind == ps::BotKind::Rank5Derived) {
    if (search_setting != kWebRank5DerivedMaxNodes) {
      last_error =
          "Rank5DerivedBot uses the fixed 50000-node browser profile";
      return false;
    }
    config.seed = 0;
    return true;
  }
  if (config.kind == ps::BotKind::AlphaBeta ||
      config.kind == ps::BotKind::JacekInspired) {
    if (search_setting == 0 ||
        search_setting > ps::AlphaBetaConfig::maximum_turn_depth) {
      last_error = std::string(ps::bot_kind_name(config.kind)) +
                   " depth must be between 1 and 64";
      return false;
    }
    config.alpha_beta_depth = search_setting;
    if (config.kind == ps::BotKind::JacekInspired) {
      config.alpha_beta_max_nodes = kWebJacekMaxNodes;
      config.alpha_beta_max_time_ms = kWebJacekMaxTimeMs;
    }
  } else {
    if (search_setting == 0) {
      last_error = "new simulations per bot move must be greater than zero";
      return false;
    }
    config.mcts_iterations = search_setting;
  }
  return true;
}

void set_command_error(const ps::WebGameCommandResult &result) {
  if (!result.error.has_value()) {
    last_error = "the C++ game engine rejected the command";
    return;
  }
  last_error = std::string(ps::web_game_error_code_name(result.error->code)) +
               ": " + result.error->message;
}

std::uint32_t issue_session_id() noexcept {
  const std::uint32_t issued = next_session_id;
  ++next_session_id;
  if (next_session_id == 0) {
    next_session_id = 1;
  }
  return issued;
}

bool validate_session(std::uint32_t expected_session_id) {
  if (!session) {
    last_error = "no live game has been started";
    return false;
  }
  if (session->session_id() != expected_session_id) {
    last_error = "stale_session: command belongs to a replaced game session";
    return false;
  }
  return true;
}

bool validate_bot_replay_session(std::uint32_t expected_session_id) {
  if (!bot_replay_session) {
    last_error = "no bot replay has been started";
    return false;
  }
  if (bot_replay_session->session_id() != expected_session_id) {
    last_error = "stale_session: command belongs to a replaced bot replay session";
    return false;
  }
  return true;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int ps_start_game(const char *seed_text, int human_player,
                                      int bot_kind,
                                      std::uint32_t search_setting) {
  ps::BotConfig bot_config;
  if (!parse_bot_config(seed_text, bot_kind, search_setting, bot_config)) {
    return 0;
  }
  if (human_player != 1 && human_player != 2) {
    last_error = "the human player must be 1 or 2";
    return 0;
  }

  const ps::Player human =
      human_player == 2 ? ps::Player::Two : ps::Player::One;
  std::unique_ptr<ps::WebGameSession> replacement;
  try {
    replacement = std::make_unique<ps::WebGameSession>(
        human, bot_config, ps::RulesConfig{}, issue_session_id());
  } catch (const std::exception &error) {
    last_error = std::string("could not start the game: ") + error.what();
    return 0;
  }
  session = std::move(replacement);
  snapshot_cache.clear();
  human_match_cache.clear();
  last_error.clear();
  return 1;
}

EMSCRIPTEN_KEEPALIVE int ps_play_human(std::uint32_t expected_session_id,
                                      std::uint32_t expected_revision,
                                      std::uint32_t move_id) {
  if (!validate_session(expected_session_id)) {
    return 0;
  }

  const ps::WebGameCommandResult result =
      session->play_human(expected_revision, static_cast<std::size_t>(move_id));
  if (!result.ok()) {
    set_command_error(result);
    return 0;
  }

  snapshot_cache.clear();
  human_match_cache.clear();
  last_error.clear();
  return 1;
}

EMSCRIPTEN_KEEPALIVE int ps_play_bot(std::uint32_t expected_session_id,
                                    std::uint32_t expected_revision) {
  if (!validate_session(expected_session_id)) {
    return 0;
  }

  const ps::WebGameCommandResult result = session->play_bot(expected_revision);
  if (!result.ok()) {
    set_command_error(result);
    return 0;
  }

  snapshot_cache.clear();
  human_match_cache.clear();
  last_error.clear();
  return 1;
}

EMSCRIPTEN_KEEPALIVE int ps_undo(std::uint32_t expected_session_id,
                                std::uint32_t expected_revision) {
  if (!validate_session(expected_session_id)) {
    return 0;
  }

  const ps::WebGameCommandResult result = session->undo(expected_revision);
  if (!result.ok()) {
    set_command_error(result);
    return 0;
  }

  snapshot_cache.clear();
  human_match_cache.clear();
  last_error.clear();
  return 1;
}

EMSCRIPTEN_KEEPALIVE int ps_start_bot_replay(
    const char *one_seed, int one_kind, std::uint32_t one_search_setting,
    const char *two_seed, int two_kind, std::uint32_t two_search_setting,
    std::uint32_t max_plies) {
  ps::BotConfig player_one;
  if (!parse_bot_config(one_seed, one_kind, one_search_setting, player_one)) {
    return 0;
  }
  ps::BotConfig player_two;
  if (!parse_bot_config(two_seed, two_kind, two_search_setting, player_two)) {
    return 0;
  }

  std::unique_ptr<ps::WebBotReplaySession> replacement;
  try {
    replacement = std::make_unique<ps::WebBotReplaySession>(
        player_one, player_two, static_cast<std::size_t>(max_plies),
        ps::RulesConfig{}, issue_session_id());
  } catch (const std::exception &error) {
    last_error = std::string("could not start the bot replay: ") + error.what();
    return 0;
  }
  bot_replay_session = std::move(replacement);
  bot_replay_snapshot_cache.clear();
  last_error.clear();
  return 1;
}

EMSCRIPTEN_KEEPALIVE int ps_play_bot_replay(
    std::uint32_t expected_session_id, std::uint32_t expected_revision) {
  if (!validate_bot_replay_session(expected_session_id)) {
    return 0;
  }

  const ps::WebGameCommandResult result =
      bot_replay_session->play_next(expected_revision);
  if (!result.ok()) {
    set_command_error(result);
    return 0;
  }

  bot_replay_snapshot_cache.clear();
  last_error.clear();
  return 1;
}

EMSCRIPTEN_KEEPALIVE const char *ps_snapshot_json() {
  if (!session) {
    last_error = "no live game has been started";
    return nullptr;
  }

  snapshot_cache = session->snapshot_json();
  return snapshot_cache.c_str();
}

EMSCRIPTEN_KEEPALIVE const char *ps_human_match_json() {
  if (!session) {
    last_error = "no live game has been started";
    return nullptr;
  }

  human_match_cache = session->human_match_json();
  return human_match_cache.c_str();
}

EMSCRIPTEN_KEEPALIVE const char *ps_bot_replay_snapshot_json() {
  if (!bot_replay_session) {
    last_error = "no bot replay has been started";
    return nullptr;
  }

  bot_replay_snapshot_cache = bot_replay_session->snapshot_json();
  return bot_replay_snapshot_cache.c_str();
}

EMSCRIPTEN_KEEPALIVE const char *ps_last_error() { return last_error.c_str(); }

}  // extern "C"
