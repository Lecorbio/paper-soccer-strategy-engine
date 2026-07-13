#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include <emscripten/emscripten.h>

#include "papersoccer/web_game.hpp"

namespace ps = papersoccer;

namespace {

std::unique_ptr<ps::WebGameSession> session;
std::string snapshot_cache;
std::string last_error;
std::uint32_t next_session_id{1};

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

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int ps_start_game(const char *seed_text, int human_player) {
  std::uint64_t seed = 0;
  if (seed_text == nullptr || !parse_seed(seed_text, seed)) {
    last_error = "the bot seed must be an unsigned 64-bit decimal integer";
    return 0;
  }
  if (human_player != 1 && human_player != 2) {
    last_error = "the human player must be 1 or 2";
    return 0;
  }

  const ps::Player human =
      human_player == 2 ? ps::Player::Two : ps::Player::One;
  session = std::make_unique<ps::WebGameSession>(
      human, seed, ps::RulesConfig{}, issue_session_id());
  snapshot_cache.clear();
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

EMSCRIPTEN_KEEPALIVE const char *ps_last_error() { return last_error.c_str(); }

}  // extern "C"
