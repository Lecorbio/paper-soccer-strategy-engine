#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "papersoccer/rules.hpp"
#include "papersoccer/web_game.hpp"

namespace ps = papersoccer;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_error(const ps::WebGameCommandResult &result,
                   ps::WebGameErrorCode expected_code,
                   const std::string &message) {
  require(!result.ok(), message + " (command unexpectedly succeeded)");
  require(result.error.has_value(), message + " (missing error details)");
  require(result.error->code == expected_code, message + " (wrong error code)");
  require(!result.move.has_value(), message + " (rejected command reported a move)");
}

void snapshot_is_a_complete_versioned_view_of_the_cpp_session() {
  const std::uint64_t seed = std::numeric_limits<std::uint64_t>::max();
  const ps::WebGameSession session(ps::Player::One, seed, ps::RulesConfig{}, 37);
  const std::string snapshot = session.snapshot_json();

  require(snapshot.find("\"schema\":\"papersoccer.web-session.v1\"") !=
              std::string::npos,
          "The snapshot should identify the web-session schema.");
  require(snapshot.find("\"sessionId\":37") != std::string::npos,
          "The snapshot should identify the originating game session.");
  require(snapshot.find("\"revision\":0") != std::string::npos,
          "The initial snapshot should carry revision zero.");
  require(snapshot.find("\"humanPlayer\":\"one\"") != std::string::npos,
          "The snapshot should identify the human side.");
  require(snapshot.find("\"ball\":{\"x\":4,\"y\":6}") != std::string::npos,
          "The live state should contain the authoritative ball position.");
  require(snapshot.find("\"legalMoves\":[{\"id\":0,\"to\":{\"x\":3,\"y\":5},"
                        "\"extraTurn\":false}") != std::string::npos,
          "Legal moves should have stable snapshot-local IDs and rebound metadata.");
  require(snapshot.find("\"schema\":\"papersoccer.replay.v2\"") !=
              std::string::npos,
          "The snapshot should embed a replay.v2 document.");
  require(snapshot.find("\"one\":{\"kind\":\"Human\"}") != std::string::npos,
          "The replay should describe the human controller.");
  require(snapshot.find("\"two\":{\"kind\":\"RandomBot\",\"seed\":"
                        "\"18446744073709551615\"}") != std::string::npos,
          "The replay should serialize the full uint64 bot seed as a decimal string.");
  require(snapshot.find("\"moves\":[]") != std::string::npos,
          "A new session should embed an empty replay history.");
}

void successful_human_move_advances_revision_and_replay() {
  ps::WebGameSession session(ps::Player::One, 17);

  const ps::WebGameCommandResult result = session.play_human(0, 3);

  require(result.ok(), "A current, legal human command should succeed.");
  require(result.revision == 1 && session.revision() == 1,
          "A successful command should advance and report the revision.");
  require(result.move.has_value(), "A successful command should return played-move metadata.");
  require(result.move->player == ps::Player::One &&
              result.move->from == ps::Point{4, 6} &&
              result.move->to == ps::Point{4, 5},
          "Move ID 3 should resolve through the current C++ legal-move ordering.");

  const std::string snapshot = session.snapshot_json();
  require(snapshot.find("\"revision\":1") != std::string::npos,
          "The next snapshot should expose the new revision.");
  require(snapshot.find("\"toMove\":\"two\"") != std::string::npos,
          "The authoritative state should pass the turn to Player 2.");
  require(snapshot.find("\"moves\":[{\"ply\":1,\"player\":\"one\","
                        "\"from\":{\"x\":4,\"y\":6},\"to\":{\"x\":4,\"y\":5},"
                        "\"extraTurn\":false,\"statusAfter\":\"inProgress\"}]") !=
              std::string::npos,
          "The embedded replay should be generated from the C++ match history.");
}

void player_two_snapshot_assigns_the_bot_to_player_one() {
  const ps::WebGameSession session(ps::Player::Two, 31);
  const std::string snapshot = session.snapshot_json();

  require(snapshot.find("\"humanPlayer\":\"two\"") != std::string::npos,
          "The live snapshot should preserve Player 2 as the human side.");
  require(snapshot.find("\"one\":{\"kind\":\"RandomBot\",\"seed\":\"31\"}") !=
              std::string::npos,
          "The entered bot seed should belong to Player 1 when the human is Player 2.");
  require(snapshot.find("\"two\":{\"kind\":\"Human\"}") != std::string::npos,
          "The embedded replay should preserve Player 2 as human.");
  require(session.bot_player() == ps::Player::One && session.bot_seed() == 31,
          "Typed session metadata should match the serialized controller assignment.");
}

void stale_revision_is_rejected_without_advancing_the_bot_rng() {
  ps::WebGameSession rejected_first(ps::Player::Two, 17);
  ps::WebGameSession control(ps::Player::Two, 17);
  const std::string before = rejected_first.snapshot_json();

  const ps::WebGameCommandResult stale = rejected_first.play_bot(9);
  require_error(stale, ps::WebGameErrorCode::StaleRevision,
                "A stale bot command should be rejected");
  require(rejected_first.snapshot_json() == before,
          "A stale command should not change any serialized session state.");

  const ps::WebGameCommandResult after_rejection = rejected_first.play_bot(0);
  const ps::WebGameCommandResult expected = control.play_bot(0);
  require(after_rejection.ok() && expected.ok(),
          "Both current bot commands should succeed.");
  require(after_rejection.move == expected.move,
          "Rejecting a stale bot command must not consume a random value.");
}

void wrong_controller_is_rejected_without_mutation() {
  ps::WebGameSession session(ps::Player::One, 23);
  const std::string before = session.snapshot_json();

  const ps::WebGameCommandResult bot_result = session.play_bot(0);
  require_error(bot_result, ps::WebGameErrorCode::WrongTurn,
                "The bot should not move during the human's turn");
  require(session.snapshot_json() == before,
          "A wrong-controller command should leave the session unchanged.");

  const ps::WebGameCommandResult human_result = session.play_human(0, 3);
  require(human_result.ok(), "The human's current legal move should still succeed.");
  const std::string after_human = session.snapshot_json();
  const ps::WebGameCommandResult second_human = session.play_human(1, 0);
  require_error(second_human, ps::WebGameErrorCode::WrongTurn,
                "The human should not move during the bot's turn");
  require(session.snapshot_json() == after_human,
          "A rejected second human move should not mutate the session.");
}

void out_of_range_move_id_is_rejected_without_mutation() {
  ps::WebGameSession session(ps::Player::One, 29);
  const std::string before = session.snapshot_json();

  const ps::WebGameCommandResult result = session.play_human(0, 8);

  require_error(result, ps::WebGameErrorCode::MoveOutOfRange,
                "A move ID outside the current legal list should be rejected");
  require(session.snapshot_json() == before,
          "An out-of-range move ID should leave state, history, and revision unchanged.");
}

void bot_command_uses_the_cpp_random_bot_and_returns_one_ply() {
  ps::WebGameSession session(ps::Player::Two, 17);

  const ps::WebGameCommandResult result = session.play_bot(0);

  require(result.ok() && result.move.has_value(),
          "The bot should make one move when it owns the turn.");
  require(result.move->player == ps::Player::One &&
              result.move->to == ps::Point{4, 5},
          "Seed 17 should use the existing deterministic C++ RandomBot sequence.");
  require(result.revision == 1 && session.match().history().size() == 1,
          "A bot command should apply exactly one ply, leaving rebound sequencing to callers.");
}

void terminal_session_rejects_both_controller_commands() {
  ps::WebGameSession session(ps::Player::One, 41);
  std::size_t plies = 0;

  while (!ps::is_terminal(session.match().state())) {
    ps::WebGameCommandResult result;
    if (session.match().state().to_move == session.human_player()) {
      result = session.play_human(session.revision(), 0);
    } else {
      result = session.play_bot(session.revision());
    }
    require(result.ok(), "The deterministic test game should accept every current command.");
    ++plies;
    require(plies <= 512, "The deterministic test game should terminate.");
  }

  const std::string terminal_snapshot = session.snapshot_json();
  const ps::WebGameCommandResult human = session.play_human(session.revision(), 0);
  const ps::WebGameCommandResult bot = session.play_bot(session.revision());

  require_error(human, ps::WebGameErrorCode::TerminalGame,
                "A terminal session should reject human moves");
  require_error(bot, ps::WebGameErrorCode::TerminalGame,
                "A terminal session should reject bot moves");
  require(session.snapshot_json() == terminal_snapshot,
          "Terminal command rejections should not mutate the completed game.");
  require(terminal_snapshot.find("\"legalMoves\":[]") != std::string::npos,
          "A terminal snapshot should expose no legal destinations.");
}

void error_codes_have_stable_bridge_names() {
  require(ps::web_game_error_code_name(ps::WebGameErrorCode::StaleSession) ==
                  "stale_session" &&
              ps::web_game_error_code_name(ps::WebGameErrorCode::StaleRevision) ==
              "stale_revision" &&
              ps::web_game_error_code_name(ps::WebGameErrorCode::WrongTurn) ==
                  "wrong_turn" &&
              ps::web_game_error_code_name(ps::WebGameErrorCode::TerminalGame) ==
                  "terminal_game" &&
              ps::web_game_error_code_name(ps::WebGameErrorCode::MoveOutOfRange) ==
                  "move_out_of_range" &&
              ps::web_game_error_code_name(ps::WebGameErrorCode::NoLegalMoves) ==
                  "no_legal_moves",
          "Every command error should have a stable machine-readable bridge name.");
}

void invalid_human_player_is_rejected() {
  bool threw = false;
  try {
    (void)ps::WebGameSession(static_cast<ps::Player>(99), 17);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "A session should reject values outside the Player enum.");
}

}  // namespace

int run_web_game_session_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };

  const std::vector<TestCase> tests{
      {"snapshot_is_a_complete_versioned_view_of_the_cpp_session",
       snapshot_is_a_complete_versioned_view_of_the_cpp_session},
      {"successful_human_move_advances_revision_and_replay",
       successful_human_move_advances_revision_and_replay},
      {"player_two_snapshot_assigns_the_bot_to_player_one",
       player_two_snapshot_assigns_the_bot_to_player_one},
      {"stale_revision_is_rejected_without_advancing_the_bot_rng",
       stale_revision_is_rejected_without_advancing_the_bot_rng},
      {"wrong_controller_is_rejected_without_mutation",
       wrong_controller_is_rejected_without_mutation},
      {"out_of_range_move_id_is_rejected_without_mutation",
       out_of_range_move_id_is_rejected_without_mutation},
      {"bot_command_uses_the_cpp_random_bot_and_returns_one_ply",
       bot_command_uses_the_cpp_random_bot_and_returns_one_ply},
      {"terminal_session_rejects_both_controller_commands",
       terminal_session_rejects_both_controller_commands},
      {"error_codes_have_stable_bridge_names", error_codes_have_stable_bridge_names},
      {"invalid_human_player_is_rejected", invalid_human_player_is_rejected},
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

  std::cout << "\n" << (tests.size() - static_cast<std::size_t>(failures)) << "/"
            << tests.size() << " web game session tests passed.\n";
  return failures == 0 ? 0 : 1;
}

#ifdef PAPERSOCCER_STANDALONE_WEB_GAME_SESSION_TEST_MAIN
int main() { return run_web_game_session_tests(); }
#endif
