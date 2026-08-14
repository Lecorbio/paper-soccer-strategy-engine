#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "papersoccer/codingame_referee.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

#ifndef PAPERSOCCER_CODINGAME_FAKE_BOT
#error "PAPERSOCCER_CODINGAME_FAKE_BOT must name the fake bot executable"
#endif

namespace {

namespace fs = std::filesystem;
using papersoccer::GameState;
using papersoccer::Move;
using papersoccer::Player;
using papersoccer::Point;
using papersoccer::codingame::RefereeConfig;

constexpr Point kDeltas[] = {
    {0, -1}, {1, -1}, {1, 0}, {1, 1},
    {0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void require_contains(std::string_view value, std::string_view needle,
                      std::string_view message) {
  if (value.find(needle) == std::string_view::npos) {
    throw std::runtime_error(std::string(message) + ": missing `" +
                             std::string(needle) + "` in " +
                             std::string(value));
  }
}

class TempDirectory {
 public:
  TempDirectory() {
    std::string pattern =
        (fs::temp_directory_path() / "papersoccer-referee-test-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  TempDirectory(const TempDirectory &) = delete;
  TempDirectory &operator=(const TempDirectory &) = delete;

  ~TempDirectory() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }

  const fs::path &path() const { return path_; }

  fs::path link(std::string_view name) const {
    const fs::path result = path_ / name;
    fs::create_symlink(fs::path(PAPERSOCCER_CODINGAME_FAKE_BOT), result);
    return result;
  }

 private:
  fs::path path_;
};

std::string run_fake_match(std::string_view first_mode,
                           std::string_view second_mode,
                           std::uint32_t first_timeout = 1000,
                           std::uint32_t later_timeout = 500,
                           std::size_t stdout_limit = 65536,
                           std::size_t stderr_limit = 65536) {
  TempDirectory directory;
  RefereeConfig config;
  config.player_one_executable =
      directory.link(std::string(first_mode) + "-player-one").string();
  config.player_two_executable =
      directory.link(std::string(second_mode) + "-player-two").string();
  config.player_one_id = "left";
  config.player_two_id = "right";
  config.first_timeout_ms = first_timeout;
  config.later_timeout_ms = later_timeout;
  config.output_limit_bytes = stdout_limit;
  config.stderr_limit_bytes = stderr_limit;
  return papersoccer::codingame::run_match_json(config);
}

void require_forfeit(const std::string &json, std::string_view loser,
                     std::string_view winner,
                     std::string_view classification) {
  require_contains(json, "\"reason\":\"forfeit\"", "match must be a forfeit");
  require_contains(json, "\"winnerId\":\"" + std::string(winner) + "\"",
                   "winner must be the opponent");
  require_contains(json, "\"botId\":\"" + std::string(loser) + "\"",
                   "forfeit must identify the failing bot");
  require_contains(json,
                   "\"classification\":\"" +
                       std::string(classification) + "\"",
                   "forfeit classification must be preserved");
}

void require_invalid(const std::vector<std::string> &actions,
                     std::string_view classification) {
  try {
    (void)papersoccer::codingame::validate_transcript(actions, true);
  } catch (const std::invalid_argument &error) {
    require_contains(error.what(), classification,
                     "unexpected transcript failure classification");
    return;
  }
  throw std::runtime_error("invalid transcript was accepted");
}

char encode_direction(Point from, Point to) {
  const Point delta{to.x - from.x, to.y - from.y};
  for (std::size_t index = 0; index < std::size(kDeltas); ++index) {
    if (delta == kDeltas[index]) {
      return static_cast<char>('0' + index);
    }
  }
  throw std::runtime_error("non-neighbor move in generated transcript");
}

std::uint64_t splitmix64(std::uint64_t &state) {
  std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

struct GeneratedTranscript {
  std::vector<std::string> actions;
  Player winner{};
  Player terminal_mover{};
  bool goal{false};
};

GeneratedTranscript find_transcript(
    const std::function<bool(const GeneratedTranscript &)> &predicate) {
  const papersoccer::RulesConfig rules{
      8, 10, papersoccer::GoalRule::OwnGoalsAllowed,
      papersoccer::BlockedRule::MoverLoses};
  for (std::uint64_t seed = 1; seed <= 20000; ++seed) {
    std::uint64_t random_state = seed;
    GameState state = papersoccer::make_initial_state(rules);
    GeneratedTranscript generated;
    for (std::size_t turn = 0; turn < 512 && !papersoccer::is_terminal(state);
         ++turn) {
      const Player mover = state.to_move;
      std::string action;
      while (!papersoccer::is_terminal(state) && state.to_move == mover) {
        const std::vector<Move> moves = papersoccer::legal_moves(state);
        require(!moves.empty(), "in-progress generated state must have a move");
        const Move selected = moves[splitmix64(random_state) % moves.size()];
        action.push_back(encode_direction(state.ball, selected.to));
        state = papersoccer::apply_move(state, selected);
      }
      generated.actions.push_back(std::move(action));
      if (papersoccer::is_terminal(state)) {
        generated.winner = *papersoccer::winner(state);
        generated.terminal_mover = mover;
        generated.goal = papersoccer::is_goal_point(state.config, state.ball);
      }
    }
    if (papersoccer::is_terminal(state) && predicate(generated)) {
      return generated;
    }
  }
  throw std::runtime_error("could not generate requested terminal transcript");
}

void transcript_validation_covers_rebounds_goals_own_goals_and_blocks() {
  const GeneratedTranscript normal_goal = find_transcript([](const auto &game) {
    return game.goal && game.winner == game.terminal_mover;
  });
  const GeneratedTranscript own_goal = find_transcript([](const auto &game) {
    return game.goal && game.winner != game.terminal_mover;
  });
  const GeneratedTranscript blocked = find_transcript([](const auto &game) {
    return !game.goal;
  });

  for (const GeneratedTranscript *game : {&normal_goal, &own_goal, &blocked}) {
    const auto replay = papersoccer::codingame::validate_transcript(game->actions);
    require(replay.terminal, "generated transcript must terminate");
    require(replay.winner == game->winner,
            "authoritative transcript replay must preserve winner");
    require(replay.action_count == game->actions.size(),
            "replay must count complete turns");
    require(replay.terminal_reason == (game->goal ? "goal" : "blocked_mover"),
            "authoritative transcript replay must classify its terminal edge");
  }

  require(normal_goal.goal && normal_goal.winner == normal_goal.terminal_mover,
          "normal goal fixture must be an attacking goal");
  require(own_goal.goal && own_goal.winner != own_goal.terminal_mover,
          "own-goal fixture must preserve the opponent as winner");
  require(!blocked.goal && blocked.winner != blocked.terminal_mover,
          "blocked-mover fixture must preserve the opponent as winner");

  const GeneratedTranscript rebound = find_transcript([](const auto &game) {
    for (const std::string &action : game.actions) {
      if (action.size() > 1) return true;
    }
    return false;
  });
  const auto replay = papersoccer::codingame::validate_transcript(rebound.actions);
  std::size_t expected_edges = 0;
  bool saw_rebound = false;
  for (const std::string &action : rebound.actions) {
    expected_edges += action.size();
    saw_rebound = saw_rebound || action.size() > 1;
  }
  require(saw_rebound && replay.edge_count == expected_edges,
          "multi-edge rebound must be accepted atomically and counted");

  const auto incomplete =
      papersoccer::codingame::validate_transcript({"0"}, true);
  require(!incomplete.terminal && !incomplete.winner &&
              !incomplete.terminal_reason,
          "a nonterminal prefix must not claim a winner or terminal reason");
}

void transcript_validation_rejects_every_action_boundary_violation() {
  require_invalid({""}, "empty-output");
  require_invalid({"x"}, "invalid-character");
  require_invalid({"0", "4"}, "illegal-action");
  require_invalid({"00"}, "output-after-handoff");
  require_invalid({"0", "2", "4", "6"}, "incomplete-rebound");

  GeneratedTranscript terminal = find_transcript([](const auto &) { return true; });
  terminal.actions.back().push_back('0');
  require_invalid(terminal.actions, "output-after-terminal");

  try {
    (void)papersoccer::codingame::validate_transcript({"0"});
  } catch (const std::invalid_argument &error) {
    require_contains(error.what(), "does not reach a terminal state",
                     "strict replay must reject prefixes");
    return;
  }
  throw std::runtime_error("strict replay accepted a nonterminal prefix");
}

void processes_receive_both_player_ids_and_complete_a_legal_game() {
  const std::string json = run_fake_match("require-id0", "require-id1");
  require_contains(json, "\"schema\":\"papersoccer.codingame-match.v1\"",
                   "native match schema must be versioned");
  require_contains(json, "\"forfeit\":null",
                   "legal game must not be scored as a forfeit");
  require(json.find("\"reason\":\"goal\"") != std::string::npos ||
              json.find("\"reason\":\"blocked_mover\"") != std::string::npos,
          "two legal bots must reach a natural terminal state");
  require_contains(json, "\"playerOne\":{\"id\":\"left\",\"player\":0",
                   "player zero identity must be serialized");
  require_contains(json, "\"playerTwo\":{\"id\":\"right\",\"player\":1",
                   "player one identity must be serialized");
}

void malformed_empty_crash_and_output_limits_forfeit() {
  require_forfeit(run_fake_match("malformed", "legal"), "left", "right",
                  "invalid-character");
  require_forfeit(run_fake_match("empty", "legal"), "left", "right",
                  "empty-output");
  require_forfeit(run_fake_match("crash", "legal"), "left", "right", "crash");
  require_forfeit(run_fake_match("stdout-flood", "legal", 500, 500, 128),
                  "left", "right", "stdout-overflow");
  require_forfeit(
      run_fake_match("stderr-flood", "legal", 500, 500, 65536, 128),
      "left", "right", "stderr-overflow");
}

void illegal_rebound_and_extra_output_forfeit_atomically() {
  require_forfeit(run_fake_match("opening-north", "reused"), "right", "left",
                  "illegal-action");
  require_forfeit(run_fake_match("script-a", "incomplete-b"), "right", "left",
                  "incomplete-rebound");

  const std::string after_handoff = run_fake_match("extra-handoff", "legal");
  require_forfeit(after_handoff, "left", "right", "output-after-handoff");
  require_contains(after_handoff, "\"accepted\":false", "bad turn must be rejected");
  require_contains(after_handoff, "\"moves\":[]",
                   "atomic rejection must not expose a speculative legal prefix");

  const std::string after_terminal =
      run_fake_match("terminal-extra", "terminal-extra");
  require_contains(after_terminal, "\"reason\":\"forfeit\"",
                   "output following a terminal edge must forfeit");
  require_contains(after_terminal,
                   "\"classification\":\"output-after-terminal\"",
                   "terminal overrun must retain its classification");
}

void queued_protocol_output_is_never_reused_as_a_future_response() {
  const std::string before_prompt =
      run_fake_match("preprompt-empty", "legal");
  require_forfeit(before_prompt, "left", "right", "output-after-handoff");
  require_contains(before_prompt, "\"action\":\"\"",
                   "an unsolicited empty line must be preserved");

  const std::string partial_before_prompt =
      run_fake_match("preprompt-partial", "legal");
  require_forfeit(partial_before_prompt, "left", "right",
                  "output-after-handoff");
  require_contains(partial_before_prompt, "\"action\":\"0\"",
                   "unsolicited partial bytes must be preserved");

  const std::string double_handoff =
      run_fake_match("double-handoff", "legal");
  require_forfeit(double_handoff, "left", "right",
                  "output-after-handoff");
  require_contains(double_handoff, "\"accepted\":false",
                   "a response followed by a queued response is rejected");
  require_contains(double_handoff, "\"moves\":[]",
                   "queued handoff output must reject atomically");

  const std::string partial_handoff =
      run_fake_match("partial-handoff", "legal");
  require_forfeit(partial_handoff, "left", "right",
                  "output-after-handoff");
  require_contains(partial_handoff, "\"accepted\":false",
                   "partial bytes after a response must reject atomically");

  const std::string double_terminal =
      run_fake_match("double-terminal", "double-terminal");
  require_contains(double_terminal, "\"reason\":\"forfeit\"",
                   "a second line after a terminal response must forfeit");
  require_contains(double_terminal,
                   "\"classification\":\"output-after-terminal\"",
                   "terminal framing violation must retain its classification");
  require_contains(double_terminal, "\"accepted\":false",
                   "terminal response plus extra line must reject atomically");

  const std::string terminal_partial =
      run_fake_match("terminal-partial", "terminal-partial");
  require_contains(terminal_partial,
                   "\"classification\":\"output-after-terminal\"",
                   "even unterminated bytes after a terminal response must forfeit");

  const std::string terminal_flood =
      run_fake_match("terminal-flood", "terminal-flood", 500, 500, 128);
  require_contains(terminal_flood,
                   "\"classification\":\"stdout-overflow\"",
                   "terminal stdout flooding must retain overflow precedence");
}

void first_and_later_decision_deadlines_are_independent() {
  const std::string first = run_fake_match("sleep-first", "legal", 20, 300);
  require_forfeit(first, "left", "right", "timeout");
  require_contains(first, "\"deadlineMillis\":20",
                   "first response must use the first deadline");

  const std::string later = run_fake_match("sleep-later", "legal", 300, 20);
  require_forfeit(later, "left", "right", "timeout");
  require_contains(later, "\"deadlineMillis\":300",
                   "first decision must use the long deadline");
  require_contains(later, "\"deadlineMillis\":20",
                   "subsequent decision must use the short deadline");
}

void bots_run_in_a_clean_environment_and_empty_temporary_directory() {
  const std::string json = run_fake_match("clean-context-a", "clean-context-b");
  require_contains(json, "\"forfeit\":null",
                   "clean context checks must not forfeit");
  require(json.find("\"reason\":\"goal\"") != std::string::npos ||
              json.find("\"reason\":\"blocked_mover\"") != std::string::npos,
          "sandbox context checks must pass for both bots");
}

void bots_inherit_only_standard_protocol_descriptors() {
  const std::string json = run_fake_match("fd-audit-a", "fd-audit-b");
  require_contains(json, "\"forfeit\":null",
                   "each executable must see only stdin, stdout, and stderr");
}

void per_bot_process_groups_isolate_opponents() {
  const std::string json = run_fake_match("kill-own-group", "legal");
  require_forfeit(json, "left", "right", "crash");
}

void require_process_gone(pid_t pid, std::string_view message) {
  for (int attempt = 0; attempt < 200; ++attempt) {
    if (::kill(pid, 0) < 0 && errno == ESRCH) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  (void)::kill(pid, SIGKILL);
  throw std::runtime_error(std::string(message));
}

void require_file_appears(const fs::path &path, std::string_view message) {
  for (int attempt = 0; attempt < 200; ++attempt) {
    if (fs::exists(path)) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  throw std::runtime_error(std::string(message));
}

void direct_exit_is_detected_while_descendant_holds_pipes() {
  TempDirectory directory;
  RefereeConfig config;
  config.player_one_executable =
      directory.link("fork-holds-fds-player-one").string();
  config.player_two_executable = directory.link("legal-player-two").string();
  config.player_one_id = "left";
  config.player_two_id = "right";
  config.first_timeout_ms = 1000;

  const auto started = std::chrono::steady_clock::now();
  const std::string json = papersoccer::codingame::run_match_json(config);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  require_forfeit(json, "left", "right", "crash");
  require(elapsed < std::chrono::milliseconds(500),
          "direct exit must not wait for descendant-held pipe EOF");

  std::ifstream pid_input(directory.path() / "holder.pid");
  pid_t holder = -1;
  pid_input >> holder;
  require(holder > 0, "pipe-holding descendant must record its pid");
  require_process_gone(holder,
                       "cleanup left the pipe-holding descendant alive");
}

void populated_bot_working_directories_are_removed_recursively() {
  TempDirectory directory;
  RefereeConfig config;
  config.player_one_executable =
      directory.link("temp-artifact-player-one").string();
  config.player_two_executable = directory.link("legal-player-two").string();
  config.player_one_id = "left";
  config.player_two_id = "right";
  const std::string json = papersoccer::codingame::run_match_json(config);
  require_forfeit(json, "left", "right", "invalid-character");

  std::ifstream path_input(directory.path() / "working-directory.path");
  std::string working_directory;
  std::getline(path_input, working_directory);
  require(!working_directory.empty(), "fake bot must record its working directory");
  require(!fs::exists(working_directory),
          "referee must recursively remove the exact bot working directory");
}

void process_launch_failure_is_infrastructure_not_a_scored_match() {
  TempDirectory directory;
  RefereeConfig config;
  config.player_one_executable = (directory.path() / "missing").string();
  config.player_two_executable = directory.link("legal").string();
  config.player_one_id = "left";
  config.player_two_id = "right";
  try {
    (void)papersoccer::codingame::run_match_json(config);
  } catch (const std::runtime_error &error) {
    require_contains(error.what(), "cannot exec bot",
                     "launch failure must surface as infrastructure error");
    return;
  }
  throw std::runtime_error("missing executable produced a scored match");
}

void launch_failure_reaps_all_started_supervisors() {
  TempDirectory directory;
  RefereeConfig config;
  config.player_one_executable = directory.link("legal-first").string();
  config.player_two_executable = (directory.path() / "missing").string();
  config.player_one_id = "left";
  config.player_two_id = "right";
  try {
    (void)papersoccer::codingame::run_match_json(config);
  } catch (const std::runtime_error &) {
    int status = 0;
    errno = 0;
    const pid_t remaining = ::waitpid(-1, &status, WNOHANG);
    require(remaining < 0 && errno == ECHILD,
            "launch unwind must reap both bot supervisors");
    return;
  }
  throw std::runtime_error("missing second executable unexpectedly launched");
}

void referee_death_triggers_sentinel_tree_cleanup() {
  TempDirectory directory;
  RefereeConfig config;
  config.player_one_executable =
      directory.link("process-tree-hang-player-one").string();
  config.player_two_executable = directory.link("legal-player-two").string();
  config.player_one_id = "left";
  config.player_two_id = "right";

  const pid_t referee = ::fork();
  require(referee >= 0, "must fork isolated referee fixture");
  if (referee == 0) {
    try {
      (void)papersoccer::codingame::run_match_json(config);
    } catch (...) {
    }
    _exit(0);
  }

  const fs::path pid_path = directory.path() / "grandchild.pid";
  try {
    require_file_appears(pid_path,
                         "hanging process tree must record its grandchild");
  } catch (...) {
    (void)::kill(referee, SIGKILL);
    (void)::waitpid(referee, nullptr, 0);
    throw;
  }
  std::ifstream pid_input(pid_path);
  pid_t grandchild = -1;
  pid_input >> grandchild;
  require(grandchild > 0, "hanging fixture must record a valid grandchild pid");

  (void)::kill(referee, SIGKILL);
  while (::waitpid(referee, nullptr, 0) < 0 && errno == EINTR) {
  }
  require_process_gone(
      grandchild,
      "sentinel did not kill the bot tree after referee watchdog death");
}

void match_cleanup_terminates_bot_descendants() {
  TempDirectory directory;
  RefereeConfig config;
  config.player_one_executable =
      directory.link("process-tree-player-one").string();
  config.player_two_executable = directory.link("legal-player-two").string();
  config.player_one_id = "left";
  config.player_two_id = "right";

  const std::string json = papersoccer::codingame::run_match_json(config);
  require_forfeit(json, "left", "right", "invalid-character");

  std::ifstream pid_input(directory.path() / "grandchild.pid");
  pid_t grandchild = -1;
  pid_input >> grandchild;
  require(grandchild > 0, "process-tree bot must record its grandchild pid");

  require_process_gone(grandchild,
                       "match cleanup left a TERM-ignoring bot grandchild alive");
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string_view, std::function<void()>>> tests = {
      {"transcript validation covers rebounds, goals, own goals, and blocks",
       transcript_validation_covers_rebounds_goals_own_goals_and_blocks},
      {"transcript validation rejects action-boundary violations",
       transcript_validation_rejects_every_action_boundary_violation},
      {"processes receive both player IDs and complete a legal game",
       processes_receive_both_player_ids_and_complete_a_legal_game},
      {"malformed, empty, crash, and output limits forfeit",
       malformed_empty_crash_and_output_limits_forfeit},
      {"illegal, rebound, and extra output forfeit atomically",
       illegal_rebound_and_extra_output_forfeit_atomically},
      {"queued protocol output is never reused as a future response",
       queued_protocol_output_is_never_reused_as_a_future_response},
      {"first and later decision deadlines are independent",
       first_and_later_decision_deadlines_are_independent},
      {"bots run in clean environment and temporary directory",
       bots_run_in_a_clean_environment_and_empty_temporary_directory},
      {"bots inherit only standard protocol descriptors",
       bots_inherit_only_standard_protocol_descriptors},
      {"per-bot process groups isolate opponents",
       per_bot_process_groups_isolate_opponents},
      {"direct exit is detected while descendants hold pipes",
       direct_exit_is_detected_while_descendant_holds_pipes},
      {"populated bot working directories are removed recursively",
       populated_bot_working_directories_are_removed_recursively},
      {"process launch failure is infrastructure",
       process_launch_failure_is_infrastructure_not_a_scored_match},
      {"launch failure reaps all started supervisors",
       launch_failure_reaps_all_started_supervisors},
      {"match cleanup terminates bot descendants",
       match_cleanup_terminates_bot_descendants},
      {"referee death triggers sentinel tree cleanup",
       referee_death_triggers_sentinel_tree_cleanup},
  };

  int failures = 0;
  for (const auto &[name, test] : tests) {
    try {
      test();
      std::cout << "PASS " << name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
  }
  if (failures != 0) {
    std::cerr << failures << " native referee test(s) failed\n";
    return 1;
  }
  return 0;
}
