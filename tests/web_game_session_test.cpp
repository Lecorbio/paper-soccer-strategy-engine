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
  require(snapshot.find("\"diagnostics\":{\"schema\":"
                        "\"papersoccer.bot-search-diagnostics.v1\"") !=
              std::string::npos,
          "The snapshot should expose a separately versioned diagnostics section.");
  require(snapshot.find("\"botConfiguration\":{\"kind\":\"RandomBot\","
                        "\"seed\":\"18446744073709551615\"}") !=
              std::string::npos &&
              snapshot.find("\"lastBotSearch\":null,\"botSearches\":[]") !=
                  std::string::npos,
          "RandomBot should preserve its active config without fabricating MCTS stats.");
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

void configured_bot_kind_is_used_and_serialized() {
  const ps::BotConfig config{
      ps::BotKind::Mcts,
      std::numeric_limits<std::uint64_t>::max(),
      8,
  };
  ps::WebGameSession session(ps::Player::Two, config, ps::RulesConfig{}, 43);
  std::unique_ptr<ps::Bot> control = ps::make_bot(config);
  const ps::Move expected = control->choose_move(session.match().state());

  require(ps::bot_kind_name(ps::BotKind::Random) == "RandomBot" &&
              ps::bot_kind_name(ps::BotKind::Mcts) == "MctsBot" &&
              ps::bot_kind_name(ps::BotKind::AlphaBeta) == "AlphaBetaBot",
          "Bot kinds should expose stable replay names.");
  require(session.bot_config().kind == ps::BotKind::Mcts &&
              session.bot_config().seed == config.seed &&
              session.bot_config().mcts_iterations == 8,
          "The web session should expose the complete selected bot config.");

  const std::string before = session.snapshot_json();
  require(before.find("\"one\":{\"kind\":\"MctsBot\",\"seed\":"
                      "\"18446744073709551615\",\"iterations\":8}") !=
              std::string::npos,
          "The replay should serialize MCTS kind, exact seed, and iteration budget.");
  require(before.find("\"two\":{\"kind\":\"Human\"}") != std::string::npos,
          "The selected human side should remain human in MCTS replay metadata.");

  const ps::WebGameCommandResult result = session.play_bot(0);
  require(result.ok() && result.move.has_value() && result.move->to == expected.to,
          "The live web session should move with the configured MctsBot.");

  require(session.bot_searches().size() == 1,
          "Each successful MCTS decision should append one search diagnostic.");
  const ps::WebBotSearchDiagnostic &search = session.bot_searches().front();
  require(search.ply == result.move->ply &&
              search.player == result.move->player &&
              search.from == result.move->from &&
              search.chosen_move.to == result.move->to &&
              search.requested_iterations == config.mcts_iterations &&
              search.stats.iterations <= search.requested_iterations,
          "The search diagnostic should identify the played decision and its work budget.");

  const std::string after = session.snapshot_json();
  require(after.find("\"lastBotSearch\":{\"ply\":1,\"player\":\"one\"") !=
              std::string::npos &&
              after.find("\"requestedIterations\":8") != std::string::npos &&
              after.find("\"completedIterations\":" +
                         std::to_string(search.stats.iterations)) !=
                  std::string::npos &&
              after.find("\"decisionTimeNs\":") != std::string::npos &&
              after.find("\"nodes\":") != std::string::npos &&
              after.find("\"simulatedPlies\":") != std::string::npos &&
              after.find("\"totalRootVisits\":") != std::string::npos &&
              after.find("\"reusedVisits\":") != std::string::npos &&
              after.find("\"maxDepth\":") != std::string::npos &&
              after.find("\"provenNodes\":") != std::string::npos &&
              after.find("\"provenWinner\":") != std::string::npos &&
              after.find("\"tacticalProbes\":") != std::string::npos &&
              after.find("\"tacticalNodes\":") != std::string::npos &&
              after.find("\"tacticalSolvedPositions\":") !=
                  std::string::npos &&
              after.find("\"tacticalDepthCutoffs\":") !=
                  std::string::npos &&
              after.find("\"tacticalNodeCutoffs\":") !=
                  std::string::npos &&
              after.find("\"maxTacticalDepth\":") != std::string::npos &&
              after.find("\"rebuildCount\":") != std::string::npos &&
              after.find("\"expansionSaturated\":") != std::string::npos &&
              after.find("\"rootValue\":") != std::string::npos,
          "The web snapshot should serialize every requested MCTS counter and timing field.");
}

void configured_alpha_beta_bot_is_used_and_serialized() {
  ps::BotConfig config;
  config.kind = ps::BotKind::AlphaBeta;
  config.seed = std::numeric_limits<std::uint64_t>::max();
  config.alpha_beta_depth = 1;
  config.alpha_beta_max_nodes = 1'000;

  ps::WebGameSession session(ps::Player::Two, config, ps::RulesConfig{8, 1}, 47);
  std::unique_ptr<ps::Bot> control = ps::make_bot(config);
  const ps::Move expected = control->choose_move(session.match().state());

  require(session.bot_config().kind == ps::BotKind::AlphaBeta &&
              session.bot_config().seed == config.seed &&
              session.bot_config().alpha_beta_depth == 1,
          "The web session should retain the selected AlphaBetaBot depth.");

  const std::string before = session.snapshot_json();
  require(before.find("\"one\":{\"kind\":\"AlphaBetaBot\",\"seed\":"
                      "\"18446744073709551615\",\"depth\":1}") !=
              std::string::npos,
          "The replay should serialize AlphaBetaBot kind, exact seed, and depth.");
  require(before.find("\"botConfiguration\":{\"kind\":\"AlphaBetaBot\","
                      "\"seed\":\"18446744073709551615\",\"depth\":1}") !=
              std::string::npos,
          "Diagnostics metadata should retain the active AlphaBetaBot configuration.");

  const ps::WebGameCommandResult result = session.play_bot(0);
  require(result.ok() && result.move.has_value() && result.move->to == expected.to,
          "The live web session should move with the configured AlphaBetaBot.");
  require(session.bot_searches().empty(),
          "Alpha-beta decisions should not be mislabeled as MCTS diagnostics.");
}

void proven_root_can_complete_fewer_iterations_than_requested() {
  const ps::BotConfig config{ps::BotKind::Mcts, 47, 64};
  ps::WebGameSession session(ps::Player::Two, config, ps::RulesConfig{8, 1});

  const ps::WebGameCommandResult result = session.play_bot(0);

  require(result.ok() && session.bot_searches().size() == 1,
          "The immediate-goal fixture should produce one recorded MCTS decision.");
  const ps::WebBotSearchDiagnostic &search = session.bot_searches().front();
  require(search.requested_iterations == 64 &&
              search.stats.iterations < search.requested_iterations &&
              search.stats.proven_winner == ps::Player::One,
          "A proven root should be allowed to stop before its requested work budget.");
}

void rebound_bot_turns_record_separate_searches() {
  const ps::BotConfig config{ps::BotKind::Mcts, 1, 1};
  ps::WebGameSession session(ps::Player::One, config);

  require(session.play_human(0, 0).ok(),
          "The first fixture human move should succeed.");
  require(session.play_bot(1).ok(),
          "The first fixture bot move should succeed.");
  require(session.play_human(2, 0).ok(),
          "The second fixture human move should succeed.");
  const ps::WebGameCommandResult rebound = session.play_bot(3);
  require(rebound.ok() && rebound.move->extra_turn,
          "The fixture's second bot decision should grant a rebound turn.");
  const ps::WebGameCommandResult follow_up = session.play_bot(4);

  require(follow_up.ok() && session.bot_searches().size() == 3,
          "A bot rebound should be followed by a separately recorded search.");
  const ps::WebBotSearchDiagnostic &first_rebound = session.bot_searches()[1];
  const ps::WebBotSearchDiagnostic &second_rebound = session.bot_searches()[2];
  require(first_rebound.ply == 4 && second_rebound.ply == 5 &&
              first_rebound.player == ps::Player::Two &&
              second_rebound.player == ps::Player::Two,
          "Consecutive rebound decisions should retain distinct plies for the same player.");

  const std::string first_snapshot = session.snapshot_json();
  const std::string second_snapshot = session.snapshot_json();
  const std::size_t history_start = first_snapshot.find("\"botSearches\":[");
  const std::size_t first = first_snapshot.find("\"ply\":2", history_start);
  const std::size_t second = first_snapshot.find("\"ply\":4", first + 1);
  const std::size_t third = first_snapshot.find("\"ply\":5", second + 1);
  require(first_snapshot == second_snapshot &&
              first_snapshot.find("\"lastBotSearch\":{\"ply\":5") !=
                  std::string::npos &&
              history_start != std::string::npos && first < second &&
              second < third,
          "Snapshot reads should retain ordered rebound-search history and its latest entry.");
}

void diagnostics_survive_snapshot_reads_and_export_losslessly() {
  const ps::BotConfig config{
      ps::BotKind::Mcts,
      std::numeric_limits<std::uint64_t>::max(),
      8,
  };
  ps::WebGameSession session(ps::Player::Two, config);
  require(session.play_bot(0).ok(),
          "The export fixture should record an initial bot decision.");

  const std::string first_snapshot = session.snapshot_json();
  const std::string second_snapshot = session.snapshot_json();
  require(first_snapshot == second_snapshot && session.bot_searches().size() == 1,
          "Ordinary snapshot reads should preserve the complete stored history.");

  const std::string exported = session.human_match_json();
  require(exported.find("{\"schema\":\"papersoccer.human-match.v1\","
                        "\"replay\":{\"schema\":\"papersoccer.replay.v2\"") ==
              0 &&
              exported.find("\"botConfiguration\":{\"kind\":\"MctsBot\","
                            "\"seed\":\"18446744073709551615\","
                            "\"iterations\":8}") != std::string::npos &&
              exported.find("\"botSearches\":[{\"ply\":1") !=
                  std::string::npos &&
              exported.find("\"tacticalProbes\":0") !=
                  std::string::npos &&
              exported.find("\"maxTacticalDepth\":0") !=
                  std::string::npos,
          "Human-match export should wrap the standard replay, lossless config, and history.");
}

void bot_factory_rejects_unknown_kinds() {
  ps::BotConfig config;
  config.kind = static_cast<ps::BotKind>(99);

  bool threw = false;
  try {
    (void)ps::make_bot(config);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "The shared bot factory should reject unknown bot kinds.");
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

void zero_length_bot_replay_is_immediately_truncated() {
  const ps::BotConfig player_one{
      ps::BotKind::Random,
      std::numeric_limits<std::uint64_t>::max(),
      8,
  };
  const ps::BotConfig player_two{ps::BotKind::Mcts, 29, 8};
  ps::WebBotReplaySession replay(player_one, player_two, 0,
                                 ps::RulesConfig{}, 71);

  require(replay.done() && replay.truncated() && replay.revision() == 0 &&
              replay.match().history().empty(),
          "A zero-ply replay should finish immediately without choosing a move.");
  require(replay.max_plies() == 0 &&
              replay.player_one_config().seed == player_one.seed &&
              replay.player_two_config().kind == ps::BotKind::Mcts,
          "A replay session should expose its independent configurations and limit.");

  const std::string snapshot = replay.snapshot_json();
  require(snapshot.find("\"schema\":\"papersoccer.bot-replay-session.v1\"") !=
              std::string::npos,
          "A bot replay snapshot should identify its session schema.");
  require(snapshot.find("\"sessionId\":71,\"revision\":0,\"done\":true") !=
              std::string::npos,
          "The initial snapshot should expose replay identity, revision, and completion.");
  require(snapshot.find("\"one\":{\"kind\":\"RandomBot\",\"seed\":"
                      "\"18446744073709551615\"}") != std::string::npos,
          "Player One metadata should preserve its independently selected full seed.");
  require(snapshot.find("\"two\":{\"kind\":\"MctsBot\",\"seed\":\"29\","
                      "\"iterations\":8}") != std::string::npos,
          "Player Two metadata should preserve its independent MCTS config.");
  require(snapshot.find("\"status\":\"inProgress\",\"winner\":null,"
                      "\"truncated\":true,\"moves\":[]") != std::string::npos,
          "A zero-ply replay document should be in progress but explicitly truncated.");

  require_error(replay.play_next(0), ps::WebGameErrorCode::ReplayComplete,
                "A completed zero-ply replay should reject generation commands");
}

void bot_replay_steps_one_ply_per_revision_and_truncates_at_limit() {
  const ps::BotConfig player_one{ps::BotKind::Random, 17, 8};
  const ps::BotConfig player_two{ps::BotKind::Random, 18, 8};
  ps::WebBotReplaySession replay(player_one, player_two, 2,
                                 ps::RulesConfig{}, 73);
  ps::WebBotReplaySession control(player_one, player_two, 2);

  const std::string initial = replay.snapshot_json();
  require_error(replay.play_next(9), ps::WebGameErrorCode::StaleRevision,
                "A stale bot-replay command should be rejected");
  require(replay.snapshot_json() == initial,
          "Rejecting a stale replay command should not consume bot randomness.");

  const ps::WebGameCommandResult first = replay.play_next(0);
  const ps::WebGameCommandResult expected_first = control.play_next(0);
  require(first.ok() && expected_first.ok() && first.move == expected_first.move,
          "A current replay command should choose exactly one deterministic bot move.");
  require(first.revision == 1 && replay.revision() == 1 && !replay.done() &&
              replay.match().history().size() == 1,
          "The first replay move should advance one revision without reaching the limit.");

  const ps::WebGameCommandResult second = replay.play_next(1);
  require(second.ok() && second.revision == 2 && replay.done() && replay.truncated(),
          "The second in-progress move should finish and truncate a two-ply replay.");
  require(replay.match().history().size() == 2,
          "Reaching the replay limit should retain exactly the configured move count.");
  const std::string complete = replay.snapshot_json();
  require(complete.find("\"revision\":2,\"done\":true") != std::string::npos &&
              complete.find("\"truncated\":true,\"moves\":[") !=
                  std::string::npos,
          "A limited replay snapshot should expose its final revision and truncation.");

  require_error(replay.play_next(2), ps::WebGameErrorCode::ReplayComplete,
                "A replay at its ply limit should reject further commands");
  require(replay.snapshot_json() == complete,
          "Rejecting a command after completion should not mutate the replay.");
}

void bounded_mcts_bot_replay_uses_the_selected_player_config() {
  const ps::BotConfig player_one{ps::BotKind::Mcts, 31, 8};
  const ps::BotConfig player_two{ps::BotKind::Random, 37, 8};
  ps::WebBotReplaySession replay(player_one, player_two, 1);
  std::unique_ptr<ps::Bot> control = ps::make_bot(player_one);
  const ps::Move expected = control->choose_move(replay.match().state());

  const ps::WebGameCommandResult result = replay.play_next(0);
  require(result.ok() && result.move.has_value() &&
              result.move->player == ps::Player::One &&
              result.move->to == expected.to,
          "The replay should use Player One's selected bounded MctsBot.");
  require(replay.done() && replay.truncated(),
          "A one-ply nonterminal MCTS replay should truncate at its bound.");
}

void bot_replay_runs_to_a_terminal_untruncated_game() {
  ps::WebBotReplaySession replay(
      ps::BotConfig{ps::BotKind::Random, 17, 8},
      ps::BotConfig{ps::BotKind::Random, 18, 8}, 512);

  while (!replay.done()) {
    const ps::WebGameCommandResult result = replay.play_next(replay.revision());
    require(result.ok(), "Every current self-play command should succeed.");
    require(replay.revision() <= 512,
            "The deterministic self-play fixture should finish within its limit.");
  }

  require(ps::is_terminal(replay.match().state()) && !replay.truncated() &&
              replay.revision() == replay.match().history().size(),
          "A naturally completed bot replay should be terminal and untruncated.");
  const std::string snapshot = replay.snapshot_json();
  require(snapshot.find("\"done\":true") != std::string::npos &&
              snapshot.find("\"truncated\":false") != std::string::npos &&
              snapshot.find("\"winner\":null") == std::string::npos,
          "A terminal replay snapshot should expose completion and a winner.");
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
                  "no_legal_moves" &&
              ps::web_game_error_code_name(ps::WebGameErrorCode::ReplayComplete) ==
                  "replay_complete",
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
      {"configured_bot_kind_is_used_and_serialized",
       configured_bot_kind_is_used_and_serialized},
      {"configured_alpha_beta_bot_is_used_and_serialized",
       configured_alpha_beta_bot_is_used_and_serialized},
      {"proven_root_can_complete_fewer_iterations_than_requested",
       proven_root_can_complete_fewer_iterations_than_requested},
      {"rebound_bot_turns_record_separate_searches",
       rebound_bot_turns_record_separate_searches},
      {"diagnostics_survive_snapshot_reads_and_export_losslessly",
       diagnostics_survive_snapshot_reads_and_export_losslessly},
      {"bot_factory_rejects_unknown_kinds", bot_factory_rejects_unknown_kinds},
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
      {"zero_length_bot_replay_is_immediately_truncated",
       zero_length_bot_replay_is_immediately_truncated},
      {"bot_replay_steps_one_ply_per_revision_and_truncates_at_limit",
       bot_replay_steps_one_ply_per_revision_and_truncates_at_limit},
      {"bounded_mcts_bot_replay_uses_the_selected_player_config",
       bounded_mcts_bot_replay_uses_the_selected_player_config},
      {"bot_replay_runs_to_a_terminal_untruncated_game",
       bot_replay_runs_to_a_terminal_untruncated_game},
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
