#include <algorithm>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#include "papersoccer/rules.hpp"

namespace {

namespace fs = std::filesystem;
using papersoccer::GameState;
using papersoccer::Move;
using papersoccer::Player;
using papersoccer::Point;

constexpr std::string_view kDirections = "01234567";
constexpr Point kDeltas[] = {
    {0, -1}, {1, -1}, {1, 0}, {1, 1},
    {0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
};

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.substr(0, prefix.size()) == prefix;
}

char direction(Point from, Point to) {
  const Point delta{to.x - from.x, to.y - from.y};
  for (std::size_t index = 0; index < std::size(kDeltas); ++index) {
    if (delta == kDeltas[index]) {
      return kDirections[index];
    }
  }
  std::abort();
}

void apply_action(GameState &state, std::string_view action) {
  for (const char encoded : action) {
    if (encoded < '0' || encoded > '7') {
      throw std::invalid_argument("invalid fake-bot action");
    }
    const std::size_t index = static_cast<std::size_t>(encoded - '0');
    state = papersoccer::apply_move(
        state, Move{{state.ball.x + kDeltas[index].x,
                     state.ball.y + kDeltas[index].y}});
  }
}

std::string choose_action(GameState state) {
  const Player mover = state.to_move;
  std::string action;
  while (!papersoccer::is_terminal(state) && state.to_move == mover) {
    const std::vector<Move> moves = papersoccer::legal_moves(state);
    if (moves.empty()) {
      std::abort();
    }
    // The selection depends only on shared game state, so both persistent fake
    // processes independently reconstruct exactly the same transcript.
    const std::size_t index =
        (state.used_segments.size() * 17U + state.path.size() * 7U) % moves.size();
    action.push_back(direction(state.ball, moves[index].to));
    state = papersoccer::apply_move(state, moves[index]);
  }
  return action;
}

bool has_clean_environment_and_directory() {
  if (std::getenv("HOME") != nullptr || std::getenv("PATH") != nullptr ||
      std::getenv("USER") != nullptr) {
    return false;
  }
  const char *language = std::getenv("LANG");
  const char *locale = std::getenv("LC_ALL");
  const char *timezone = std::getenv("TZ");
  if (language == nullptr || std::string_view(language) != "C" ||
      locale == nullptr || std::string_view(locale) != "C" ||
      timezone == nullptr || std::string_view(timezone) != "UTC") {
    return false;
  }
  return fs::directory_iterator(fs::current_path()) == fs::directory_iterator{};
}

bool has_only_standard_descriptors() {
  struct rlimit limit {};
  if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) return false;
  const int maximum = static_cast<int>(std::min<rlim_t>(limit.rlim_cur, 4096));
  for (int fd = 3; fd < maximum; ++fd) {
    errno = 0;
    if (::fcntl(fd, F_GETFD) >= 0 || errno != EBADF) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  const fs::path invoked_as = argc > 0 ? fs::absolute(argv[0]) : fs::path{};
  const std::string mode = invoked_as.filename().string();

  if (starts_with(mode, "process-tree-eager")) {
    const pid_t grandchild = ::fork();
    if (grandchild < 0) return 4;
    if (grandchild == 0) {
      std::signal(SIGTERM, SIG_IGN);
      for (;;) ::pause();
    }
    std::ofstream(invoked_as.parent_path() / "grandchild.pid")
        << grandchild << '\n';
  }

  int player = -1;
  if (!(std::cin >> player)) {
    return 3;
  }
  if ((starts_with(mode, "require-id0") && player != 0) ||
      (starts_with(mode, "require-id1") && player != 1)) {
    std::cout << "x" << std::endl;
    return 0;
  }
  if (starts_with(mode, "clean-context") &&
      !has_clean_environment_and_directory()) {
    std::cout << "x" << std::endl;
    return 0;
  }
  if (starts_with(mode, "fd-audit") && !has_only_standard_descriptors()) {
    std::cout << "x" << std::endl;
    return 0;
  }
  if (starts_with(mode, "preprompt-empty")) {
    const char unsolicited = '\n';
    (void)::write(STDOUT_FILENO, &unsolicited, 1);
  }
  if (starts_with(mode, "preprompt-partial")) {
    const char unsolicited = '0';
    (void)::write(STDOUT_FILENO, &unsolicited, 1);
  }

  if (starts_with(mode, "process-tree") &&
      !starts_with(mode, "process-tree-eager")) {
    const pid_t grandchild = ::fork();
    if (grandchild < 0) {
      return 4;
    }
    if (grandchild == 0) {
      std::signal(SIGTERM, SIG_IGN);
      for (;;) {
        ::pause();
      }
    }
    std::ofstream(invoked_as.parent_path() / "grandchild.pid") << grandchild << '\n';
  }

  GameState state = papersoccer::make_initial_state(
      {8, 10, papersoccer::GoalRule::OwnGoalsAllowed,
       papersoccer::BlockedRule::MoverLoses});
  std::size_t decision = 0;
  int count = 0;
  std::string opponent_action;
  while (std::cin >> count >> opponent_action) {
    (void)count;
    if (opponent_action != "-") {
      apply_action(state, opponent_action);
    }

    if (starts_with(mode, "crash")) {
      _exit(42);
    }
    if (starts_with(mode, "fork-holds-fds")) {
      const pid_t grandchild = ::fork();
      if (grandchild < 0) return 4;
      if (grandchild == 0) {
        std::signal(SIGTERM, SIG_IGN);
        for (;;) ::pause();
      }
      std::ofstream(invoked_as.parent_path() / "holder.pid")
          << grandchild << '\n';
      _exit(42);
    }
    if (starts_with(mode, "kill-own-group")) {
      (void)::kill(0, SIGKILL);
      _exit(42);
    }
    if (starts_with(mode, "process-tree-hang")) {
      for (;;) ::pause();
    }
    if (starts_with(mode, "temp-artifact")) {
      const fs::path working = fs::current_path();
      std::ofstream(invoked_as.parent_path() / "working-directory.path")
          << working.string() << '\n';
      fs::create_directories(working / "nested");
      std::ofstream(working / "nested" / "artifact.txt") << "left behind\n";
      std::cout << "x" << std::endl;
      return 0;
    }
    if (starts_with(mode, "empty")) {
      std::cout << std::endl;
      return 0;
    }
    if (starts_with(mode, "malformed")) {
      std::cout << "x" << std::endl;
      return 0;
    }
    if (starts_with(mode, "stdout-flood")) {
      std::cout << std::string(8192, '0') << std::flush;
      for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (starts_with(mode, "stderr-flood")) {
      std::cerr << std::string(8192, 'e') << std::flush;
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    if (starts_with(mode, "sleep-first") && decision == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }
    if (starts_with(mode, "sleep-later") && decision > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }

    std::string action;
    if (starts_with(mode, "extra-handoff") && decision == 0) {
      action = "00";
    } else if (starts_with(mode, "opening-north") && decision == 0) {
      action = "0";
    } else if (starts_with(mode, "reused") && decision == 0) {
      action = "4";
    } else if (starts_with(mode, "script-a") && decision == 0) {
      action = "0";
    } else if (starts_with(mode, "script-a") && decision == 1) {
      action = "4";
    } else if (starts_with(mode, "incomplete-b") && decision == 0) {
      action = "2";
    } else if (starts_with(mode, "incomplete-b") && decision == 1) {
      action = "6";
    } else if (starts_with(mode, "process-tree")) {
      action = "x";
    } else {
      action = choose_action(state);
    }

    GameState accepted = state;
    bool locally_valid = true;
    try {
      apply_action(accepted, action);
    } catch (...) {
      locally_valid = false;
    }
    if (starts_with(mode, "terminal-extra") && locally_valid &&
        papersoccer::is_terminal(accepted)) {
      action.push_back('0');
      locally_valid = false;
    }
    if (starts_with(mode, "double-handoff") && decision == 0) {
      const std::string framed = action + "\n" + action + "\n";
      (void)::write(STDOUT_FILENO, framed.data(), framed.size());
      for (;;) ::pause();
    }
    if (starts_with(mode, "partial-handoff") && decision == 0) {
      const std::string framed = action + "\n0";
      (void)::write(STDOUT_FILENO, framed.data(), framed.size());
      for (;;) ::pause();
    }
    if (starts_with(mode, "double-terminal") && locally_valid &&
        papersoccer::is_terminal(accepted)) {
      const std::string framed = action + "\n" + action + "\n";
      (void)::write(STDOUT_FILENO, framed.data(), framed.size());
      for (;;) ::pause();
    }
    if (starts_with(mode, "terminal-flood") && locally_valid &&
        papersoccer::is_terminal(accepted)) {
      const std::string framed = action + "\n" + std::string(8192, '0');
      (void)::write(STDOUT_FILENO, framed.data(), framed.size());
      for (;;) ::pause();
    }
    if (starts_with(mode, "terminal-partial") && locally_valid &&
        papersoccer::is_terminal(accepted)) {
      const std::string framed = action + "\n0";
      (void)::write(STDOUT_FILENO, framed.data(), framed.size());
      for (;;) ::pause();
    }
    std::cout << action << std::endl;
    if (locally_valid) {
      state = std::move(accepted);
    }
    ++decision;
  }
  return 0;
}
