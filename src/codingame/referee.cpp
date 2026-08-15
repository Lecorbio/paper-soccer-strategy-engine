#include "papersoccer/codingame_referee.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <optional>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

namespace papersoccer::codingame {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<Point, 8> kDeltas{{
    {0, -1}, {1, -1}, {1, 0}, {1, 1},
    {0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
}};

struct Child {
  pid_t pid{-1};
  pid_t supervisor_pid{-1};
  pid_t process_group{-1};
  int input{-1};
  int output{-1};
  int error{-1};
  int control{-1};
  int status{-1};
  std::string stdout_buffer;
  std::string stderr_buffer;
  std::vector<char> status_buffer;
  bool exit_observed{false};
  int exit_status{0};
  std::string executable;
  std::string id;
  int player{0};
  bool first_decision{true};
  std::string working_directory;
};

struct MoveRecord {
  int direction{0};
  Point from{};
  Point to{};
  bool extra_turn{false};
  std::string status_after;
};

struct TurnRecord {
  std::size_t turn{0};
  std::string bot_id;
  int player{0};
  std::string opponent_action;
  std::optional<std::string> action;
  bool accepted{false};
  std::uint64_t duration_micros{0};
  std::uint32_t deadline_millis{0};
  std::optional<std::string> failure;
  std::vector<MoveRecord> moves;
};

struct MatchResult {
  std::vector<TurnRecord> turns;
  std::string winner_id;
  std::string loser_id;
  std::string reason;
  std::optional<std::string> forfeit_id;
  std::optional<std::string> classification;
  std::string detail;
  std::uint64_t total_micros{0};
};

RulesConfig contest_rules() {
  return RulesConfig{8, 10, GoalRule::OwnGoalsAllowed, BlockedRule::MoverLoses};
}

void close_fd(int &fd) {
  if (fd >= 0) {
    while (::close(fd) < 0 && errno == EINTR) {
    }
    fd = -1;
  }
}

class OwnedFd {
 public:
  OwnedFd() = default;
  explicit OwnedFd(int value) : value_(value) {}
  OwnedFd(const OwnedFd &) = delete;
  OwnedFd &operator=(const OwnedFd &) = delete;
  OwnedFd(OwnedFd &&other) noexcept : value_(std::exchange(other.value_, -1)) {}
  OwnedFd &operator=(OwnedFd &&other) noexcept {
    if (this != &other) {
      close_fd(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  ~OwnedFd() { close_fd(value_); }
  int get() const { return value_; }
  int release() { return std::exchange(value_, -1); }
  void reset() { close_fd(value_); }

 private:
  int value_{-1};
};

struct PipePair {
  OwnedFd read;
  OwnedFd write;
};

void set_close_on_exec(int fd) {
  const int flags = ::fcntl(fd, F_GETFD, 0);
  if (flags < 0 || ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0) {
    throw std::runtime_error("cannot configure close-on-exec pipe");
  }
}

PipePair make_cloexec_pipe() {
  int descriptors[2]{-1, -1};
#if defined(__linux__) && defined(O_CLOEXEC)
  if (::pipe2(descriptors, O_CLOEXEC) < 0) {
    throw std::runtime_error("cannot create bot pipe");
  }
#else
  if (::pipe(descriptors) < 0) {
    throw std::runtime_error("cannot create bot pipe");
  }
  try {
    set_close_on_exec(descriptors[0]);
    set_close_on_exec(descriptors[1]);
  } catch (...) {
    ::close(descriptors[0]);
    ::close(descriptors[1]);
    throw;
  }
#endif
  return PipePair{OwnedFd(descriptors[0]), OwnedFd(descriptors[1])};
}

void close_all_except(std::initializer_list<int> retained) {
  struct rlimit limit {};
  rlim_t maximum = 1024;
  if (::getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY) {
    maximum = limit.rlim_cur;
  } else {
    const long configured = ::sysconf(_SC_OPEN_MAX);
    if (configured > 0) maximum = static_cast<rlim_t>(configured);
  }
  for (int fd = 3; static_cast<rlim_t>(fd) < maximum; ++fd) {
    if (std::find(retained.begin(), retained.end(), fd) == retained.end()) {
      (void)::close(fd);
    }
  }
}

void remove_working_directory(const std::string &path) noexcept {
  if (path.empty()) return;
  std::error_code ignored;
  (void)std::filesystem::remove_all(path, ignored);
}

void set_nonblocking(int fd) {
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    throw std::runtime_error("cannot configure nonblocking pipe");
  }
}

void write_all(int fd, std::string_view data) {
  while (!data.empty()) {
    const ssize_t count = ::write(fd, data.data(), data.size());
    if (count > 0) {
      data.remove_prefix(static_cast<std::size_t>(count));
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      throw std::runtime_error("bot input pipe closed");
    }
  }
}

void drain_fd(int fd, std::string &buffer, std::size_t limit, bool &overflow,
              bool &eof) {
  std::array<char, 4096> chunk{};
  for (;;) {
    const ssize_t count = ::read(fd, chunk.data(), chunk.size());
    if (count > 0) {
      const auto amount = static_cast<std::size_t>(count);
      if (buffer.size() + amount > limit) {
        overflow = true;
        return;
      } else {
        buffer.append(chunk.data(), amount);
      }
      continue;
    }
    if (count == 0) {
      eof = true;
      return;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return;
    }
    eof = true;
    return;
  }
}

enum class SupervisorMessageKind : std::int32_t {
  Ready = 1,
  Exited = 2,
  SetupError = 3,
  CleanupDone = 4,
};

struct SupervisorMessage {
  SupervisorMessageKind kind{};
  std::int32_t value{0};
  pid_t bot_pid{-1};
  pid_t process_group{-1};
};

static_assert(sizeof(SupervisorMessage) <= 64);

void write_supervisor_message(int fd, SupervisorMessage message) noexcept {
  const char *data = reinterpret_cast<const char *>(&message);
  std::size_t remaining = sizeof(message);
  while (remaining != 0) {
    const ssize_t count = ::write(fd, data, remaining);
    if (count > 0) {
      data += count;
      remaining -= static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      return;
    }
  }
}

void signal_group(pid_t group, int signal) noexcept {
  if (group > 1) (void)::kill(-group, signal);
}

void supervise_bot(int input_read, int output_write, int error_write,
                   int control_read, int status_write, int exec_read,
                   int exec_write, const std::string &executable,
                   const std::string &working_directory) noexcept {
  if (::setsid() < 0) {
    write_supervisor_message(
        status_write,
        {SupervisorMessageKind::SetupError, errno, -1, -1});
    _exit(127);
  }

  const pid_t bot = ::fork();
  if (bot < 0) {
    write_supervisor_message(
        status_write,
        {SupervisorMessageKind::SetupError, errno, -1, -1});
    _exit(127);
  }
  if (bot == 0) {
    if (::setpgid(0, 0) < 0) {
      const int saved = errno;
      (void)::write(exec_write, &saved, sizeof(saved));
      _exit(127);
    }
    std::signal(SIGPIPE, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGCHLD, SIG_DFL);
    sigset_t empty_mask;
    sigemptyset(&empty_mask);
    (void)::sigprocmask(SIG_SETMASK, &empty_mask, nullptr);
    if (::dup2(input_read, STDIN_FILENO) < 0 ||
        ::dup2(output_write, STDOUT_FILENO) < 0 ||
        ::dup2(error_write, STDERR_FILENO) < 0) {
      const int saved = errno;
      (void)::write(exec_write, &saved, sizeof(saved));
      _exit(127);
    }
    close_all_except({exec_write});
    struct rlimit descriptor_limit { 64, 64 };
    if (::setrlimit(RLIMIT_NOFILE, &descriptor_limit) < 0 ||
        ::chdir(working_directory.c_str()) != 0) {
      const int saved = errno;
      (void)::write(exec_write, &saved, sizeof(saved));
      _exit(127);
    }
    char *const argv[] = {const_cast<char *>(executable.c_str()), nullptr};
    char language[] = "LANG=C";
    char locale[] = "LC_ALL=C";
    char timezone[] = "TZ=UTC";
    std::string temporary_environment = "TMPDIR=" + working_directory;
    char *const environment[] = {language, locale, timezone,
                                 temporary_environment.data(), nullptr};
    ::execve(executable.c_str(), argv, environment);
    const int saved = errno;
    (void)::write(exec_write, &saved, sizeof(saved));
    _exit(127);
  }

  close_all_except({control_read, status_write, exec_read});
  if (::setpgid(bot, bot) < 0) {
    const int saved = errno;
    if (saved != ESRCH &&
        !((saved == EACCES || saved == EPERM) && ::getpgid(bot) == bot)) {
      (void)::kill(bot, SIGKILL);
      (void)::waitpid(bot, nullptr, 0);
      write_supervisor_message(
          status_write,
          {SupervisorMessageKind::SetupError, saved, bot, bot});
      _exit(127);
    }
  }

  int exec_error = 0;
  ssize_t exec_count;
  do {
    exec_count = ::read(exec_read, &exec_error, sizeof(exec_error));
  } while (exec_count < 0 && errno == EINTR);
  (void)::close(exec_read);
  if (exec_count != 0) {
    if (exec_count < 0) exec_error = errno;
    signal_group(bot, SIGKILL);
    (void)::waitpid(bot, nullptr, 0);
    write_supervisor_message(
        status_write,
        {SupervisorMessageKind::SetupError, exec_error, bot, bot});
    _exit(127);
  }

  write_supervisor_message(
      status_write, {SupervisorMessageKind::Ready, 0, bot, bot});
  bool bot_reaped = false;
  int bot_status = 0;
  bool exit_reported = false;
  bool shutdown_requested = false;
  for (;;) {
    if (!bot_reaped) {
      const pid_t waited = ::waitpid(bot, &bot_status, WNOHANG);
      if (waited == bot || (waited < 0 && errno == ECHILD)) {
        bot_reaped = true;
        if (!exit_reported) {
          write_supervisor_message(
              status_write,
              {SupervisorMessageKind::Exited, bot_status, bot, bot});
          exit_reported = true;
        }
        shutdown_requested = true;
      }
    }
    if (!shutdown_requested) {
      pollfd control{control_read, POLLIN | POLLHUP, 0};
      const int polled = ::poll(&control, 1, 10);
      if (polled < 0 && errno == EINTR) continue;
      if (polled < 0) shutdown_requested = true;
      if (polled > 0 && control.revents) {
        char discard[32];
        const ssize_t count = ::read(control_read, discard, sizeof(discard));
        if (count == 0 || (count < 0 && errno != EINTR)) {
          shutdown_requested = true;
        }
      }
    }
    if (!shutdown_requested) continue;

    signal_group(bot, SIGTERM);
    for (int attempt = 0; attempt < 20; ++attempt) {
      if (!bot_reaped) {
        const pid_t waited = ::waitpid(bot, &bot_status, WNOHANG);
        if (waited == bot || (waited < 0 && errno == ECHILD)) {
          bot_reaped = true;
        }
      }
      if (::kill(-bot, 0) < 0 && errno == ESRCH) break;
      ::usleep(1000);
    }
    signal_group(bot, SIGKILL);
    if (!bot_reaped) {
      while (::waitpid(bot, &bot_status, 0) < 0 && errno == EINTR) {
      }
    }
    write_supervisor_message(
        status_write,
        {SupervisorMessageKind::CleanupDone, 0, bot, bot});
    _exit(0);
  }
}

SupervisorMessage read_supervisor_message_blocking(int fd) {
  SupervisorMessage message{};
  char *data = reinterpret_cast<char *>(&message);
  std::size_t remaining = sizeof(message);
  while (remaining != 0) {
    const ssize_t count = ::read(fd, data, remaining);
    if (count > 0) {
      data += count;
      remaining -= static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      throw std::runtime_error("bot supervisor exited during setup");
    }
  }
  return message;
}

void terminate_child(Child &child);

Child spawn_child(const std::string &executable, const std::string &id,
                  int player) {
  PipePair input_pipe = make_cloexec_pipe();
  PipePair output_pipe = make_cloexec_pipe();
  PipePair error_pipe = make_cloexec_pipe();
  PipePair control_pipe = make_cloexec_pipe();
  PipePair status_pipe = make_cloexec_pipe();
  PipePair exec_pipe = make_cloexec_pipe();
  const char *temporary_root = std::getenv("TMPDIR");
  std::string working_template =
      temporary_root != nullptr && temporary_root[0] != '\0'
          ? temporary_root
          : "/tmp";
  if (working_template.back() != '/') {
    working_template.push_back('/');
  }
  working_template += "papersoccer-bot-XXXXXX";
  if (::mkdtemp(working_template.data()) == nullptr) {
    throw std::runtime_error("cannot create bot working directory");
  }
  const pid_t supervisor = ::fork();
  if (supervisor < 0) {
    remove_working_directory(working_template);
    throw std::runtime_error("cannot fork bot supervisor");
  }
  if (supervisor == 0) {
    supervise_bot(input_pipe.read.get(), output_pipe.write.get(),
                  error_pipe.write.get(), control_pipe.read.get(),
                  status_pipe.write.get(), exec_pipe.read.get(),
                  exec_pipe.write.get(), executable, working_template);
  }

  input_pipe.read.reset();
  output_pipe.write.reset();
  error_pipe.write.reset();
  control_pipe.read.reset();
  status_pipe.write.reset();
  exec_pipe.read.reset();
  exec_pipe.write.reset();

  Child child;
  child.supervisor_pid = supervisor;
  child.input = input_pipe.write.release();
  child.output = output_pipe.read.release();
  child.error = error_pipe.read.release();
  child.control = control_pipe.write.release();
  child.status = status_pipe.read.release();
  child.executable = executable;
  child.id = id;
  child.player = player;
  child.working_directory = working_template;
  try {
    const SupervisorMessage setup = read_supervisor_message_blocking(child.status);
    if (setup.kind != SupervisorMessageKind::Ready || setup.bot_pid <= 1 ||
        setup.process_group != setup.bot_pid) {
      const int error = setup.kind == SupervisorMessageKind::SetupError
                            ? setup.value
                            : EPROTO;
      throw std::runtime_error("cannot exec bot: " +
                               std::string(std::strerror(error)));
    }
    child.pid = setup.bot_pid;
    child.process_group = setup.process_group;
    set_nonblocking(child.output);
    set_nonblocking(child.error);
    set_nonblocking(child.status);
    write_all(child.input, std::to_string(player) + "\n");
  } catch (...) {
    terminate_child(child);
    throw;
  }
  return child;
}

void terminate_child(Child &child) {
  close_fd(child.input);
  signal_group(child.process_group, SIGTERM);
  close_fd(child.control);
  if (child.supervisor_pid > 0) {
    for (int attempt = 0; attempt < 100; ++attempt) {
      const pid_t result = ::waitpid(child.supervisor_pid, nullptr, WNOHANG);
      if (result == child.supervisor_pid || (result < 0 && errno == ECHILD)) {
        child.supervisor_pid = -1;
        break;
      }
      ::usleep(1000);
    }
    signal_group(child.process_group, SIGKILL);
    if (child.supervisor_pid > 0) {
      (void)::kill(child.supervisor_pid, SIGKILL);
      while (::waitpid(child.supervisor_pid, nullptr, 0) < 0 && errno == EINTR) {
      }
      child.supervisor_pid = -1;
    }
  }
  child.pid = -1;
  child.process_group = -1;
  close_fd(child.output);
  close_fd(child.error);
  close_fd(child.status);
  if (!child.working_directory.empty()) {
    remove_working_directory(child.working_directory);
    child.working_directory.clear();
  }
}

struct ChildCleanup {
  Child &first;
  Child &second;
  ~ChildCleanup() {
    terminate_child(first);
    terminate_child(second);
  }
};

struct ReadResult {
  std::optional<std::string> line;
  std::string classification;
  std::string detail;
  std::uint64_t duration_micros{0};
  bool extra_complete_line{false};
};

struct QueuedOutput {
  std::optional<std::string> classification;
  std::string detail;
  bool stdout_eof{false};
};

void drain_supervisor_status(Child &child) {
  std::array<char, 128> bytes{};
  bool eof = false;
  for (;;) {
    const ssize_t count = ::read(child.status, bytes.data(), bytes.size());
    if (count > 0) {
      child.status_buffer.insert(child.status_buffer.end(), bytes.begin(),
                                 bytes.begin() + count);
    } else if (count == 0) {
      eof = true;
      break;
    } else if (errno == EINTR) {
      continue;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
      break;
    } else {
      throw std::runtime_error("cannot read bot supervisor status");
    }
  }
  while (child.status_buffer.size() >= sizeof(SupervisorMessage)) {
    SupervisorMessage message{};
    std::memcpy(&message, child.status_buffer.data(), sizeof(message));
    child.status_buffer.erase(
        child.status_buffer.begin(),
        child.status_buffer.begin() + static_cast<std::ptrdiff_t>(sizeof(message)));
    if (message.bot_pid != child.pid ||
        message.process_group != child.process_group) {
      throw std::runtime_error("bot supervisor reported inconsistent identity");
    }
    if (message.kind == SupervisorMessageKind::Exited) {
      child.exit_observed = true;
      child.exit_status = message.value;
    } else if (message.kind != SupervisorMessageKind::CleanupDone) {
      throw std::runtime_error("bot supervisor reported an invalid state");
    }
  }
  if (eof && !child.exit_observed) {
    throw std::runtime_error("bot supervisor exited unexpectedly");
  }
}

QueuedOutput drain_queued_output(Child &child, std::size_t stdout_limit,
                                 std::size_t stderr_limit) {
  bool stdout_overflow = false;
  bool stderr_overflow = false;
  bool stdout_eof = false;
  bool stderr_eof = false;
  drain_fd(child.output, child.stdout_buffer, stdout_limit, stdout_overflow,
           stdout_eof);
  drain_fd(child.error, child.stderr_buffer, stderr_limit, stderr_overflow,
           stderr_eof);
  drain_supervisor_status(child);
  if (stdout_overflow) {
    return {std::string("stdout-overflow"), "stdout limit exceeded",
            stdout_eof};
  }
  if (stderr_overflow) {
    return {std::string("stderr-overflow"), "stderr limit exceeded",
            stdout_eof};
  }
  return {std::nullopt, "", stdout_eof};
}

ReadResult read_line(Child &child, std::uint32_t timeout_ms,
                     std::size_t stdout_limit, std::size_t stderr_limit) {
  const auto started = Clock::now();
  const auto deadline = started + std::chrono::milliseconds(timeout_ms);
  for (;;) {
    const std::size_t newline = child.stdout_buffer.find('\n');
    if (newline != std::string::npos) {
      std::string line = child.stdout_buffer.substr(0, newline);
      child.stdout_buffer.erase(0, newline + 1);
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          Clock::now() - started);
      if (elapsed > std::chrono::milliseconds(timeout_ms)) {
        return {std::nullopt, "timeout", "decision deadline exceeded",
                static_cast<std::uint64_t>(elapsed.count()), false};
      }
      return {std::move(line), "", "",
              static_cast<std::uint64_t>(elapsed.count()),
              child.stdout_buffer.find('\n') != std::string::npos};
    }
    if (child.exit_observed) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          Clock::now() - started);
      return {std::nullopt, "crash", "bot process exited",
              static_cast<std::uint64_t>(elapsed.count()), false};
    }
    const auto now = Clock::now();
    if (now >= deadline) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          now - started);
      return {std::nullopt, "timeout", "decision deadline exceeded",
              static_cast<std::uint64_t>(elapsed.count()), false};
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now);
    std::array<pollfd, 3> fds{{{child.output, POLLIN | POLLHUP, 0},
                               {child.error, POLLIN | POLLHUP, 0},
                               {child.status, POLLIN | POLLHUP, 0}}};
    const int wait_ms = static_cast<int>(std::max<std::int64_t>(1, remaining.count()));
    const int polled = ::poll(fds.data(), fds.size(), wait_ms);
    if (polled < 0 && errno == EINTR) {
      continue;
    }
    if (polled < 0) {
      throw std::runtime_error("poll failed while reading bot output");
    }
    bool stdout_overflow = false;
    bool stderr_overflow = false;
    bool stdout_eof = false;
    bool stderr_eof = false;
    if (fds[0].revents) {
      drain_fd(child.output, child.stdout_buffer, stdout_limit,
               stdout_overflow, stdout_eof);
    }
    if (fds[1].revents) {
      drain_fd(child.error, child.stderr_buffer, stderr_limit,
               stderr_overflow, stderr_eof);
    }
    if (fds[2].revents) {
      drain_supervisor_status(child);
    }
    if (stdout_overflow) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          Clock::now() - started);
      return {std::nullopt, "stdout-overflow", "stdout limit exceeded",
              static_cast<std::uint64_t>(elapsed.count()), false};
    }
    if (stderr_overflow) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          Clock::now() - started);
      return {std::nullopt, "stderr-overflow", "stderr limit exceeded",
              static_cast<std::uint64_t>(elapsed.count()), false};
    }
    if (stdout_eof) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          Clock::now() - started);
      return {std::nullopt, "crash", "bot exited before a complete response",
              static_cast<std::uint64_t>(elapsed.count()), false};
    }
  }
}

Move decode(Point from, char direction) {
  if (direction < '0' || direction > '7') {
    throw std::invalid_argument("invalid-character");
  }
  const Point delta = kDeltas[static_cast<std::size_t>(direction - '0')];
  return Move{{from.x + delta.x, from.y + delta.y}};
}

GameState apply_complete_turn(const GameState &state, std::string_view action,
                              std::vector<MoveRecord> *moves) {
  if (action.empty()) {
    throw std::invalid_argument("empty-output");
  }
  GameState next = state;
  const Player mover = state.to_move;
  for (const char character : action) {
    if (is_terminal(next)) {
      throw std::invalid_argument("output-after-terminal");
    }
    if (next.to_move != mover) {
      throw std::invalid_argument("output-after-handoff");
    }
    const Point from = next.ball;
    const Move move = decode(from, character);
    try {
      next = apply_move(next, move);
    } catch (const std::invalid_argument &) {
      throw std::invalid_argument("illegal-action");
    }
    if (moves != nullptr) {
      std::string status = "ongoing";
      if (next.status == Status::WonByOne) {
        status = "player_0_wins";
      } else if (next.status == Status::WonByTwo) {
        status = "player_1_wins";
      }
      moves->push_back(MoveRecord{
          static_cast<int>(character - '0'), from, next.ball,
          !is_terminal(next) && next.to_move == mover, std::move(status)});
    }
  }
  if (!is_terminal(next) && next.to_move == mover) {
    throw std::invalid_argument("incomplete-rebound");
  }
  return next;
}

std::string json_escape(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (character < 0x20U) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(character) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(character);
        }
    }
  }
  out << '"';
  return out.str();
}

std::string player_id(const RefereeConfig &config, Player player) {
  return player == Player::One ? config.player_one_id : config.player_two_id;
}

std::string executable_name(std::string_view value) {
  const std::size_t separator = value.find_last_of('/');
  return std::string(separator == std::string_view::npos
                         ? value
                         : value.substr(separator + 1));
}

std::string serialize(const RefereeConfig &config, const MatchResult &result) {
  std::ostringstream out;
  out << "{\"schema\":\"papersoccer.codingame-match.v1\",\"participants\":{"
      << "\"playerOne\":{\"id\":" << json_escape(config.player_one_id)
      << ",\"player\":0,\"executable\":"
      << json_escape(executable_name(config.player_one_executable))
      << "},\"playerTwo\":{\"id\":" << json_escape(config.player_two_id)
      << ",\"player\":1,\"executable\":"
      << json_escape(executable_name(config.player_two_executable))
      << "}},\"rules\":{\"width\":8,\"height\":10,\"goalRule\":\"OwnGoalsAllowed\","
      << "\"blockedRule\":\"MoverLoses\"},\"timeouts\":{\"firstMillis\":"
      << config.first_timeout_ms << ",\"laterMillis\":" << config.later_timeout_ms
      << "},\"actions\":[";
  for (std::size_t index = 0; index < result.turns.size(); ++index) {
    const TurnRecord &turn = result.turns[index];
    if (index) out << ',';
    out << "{\"turn\":" << turn.turn << ",\"botId\":" << json_escape(turn.bot_id)
        << ",\"player\":" << turn.player << ",\"opponentAction\":"
        << json_escape(turn.opponent_action) << ",\"action\":";
    if (turn.action) out << json_escape(*turn.action); else out << "null";
    out << ",\"accepted\":" << (turn.accepted ? "true" : "false")
        << ",\"durationMicros\":" << turn.duration_micros
        << ",\"deadlineMillis\":" << turn.deadline_millis
        << ",\"failureClassification\":";
    if (turn.failure) out << json_escape(*turn.failure); else out << "null";
    out << ",\"moves\":[";
    for (std::size_t move = 0; move < turn.moves.size(); ++move) {
      if (move) out << ',';
      const MoveRecord &record = turn.moves[move];
      out << "{\"direction\":" << record.direction
          << ",\"from\":{\"x\":" << record.from.x << ",\"y\":" << record.from.y
          << "},\"to\":{\"x\":" << record.to.x << ",\"y\":" << record.to.y
          << "},\"extraTurn\":" << (record.extra_turn ? "true" : "false")
          << ",\"statusAfter\":" << json_escape(record.status_after) << '}';
    }
    out << "]}";
  }
  out << "],\"outcome\":{\"winnerId\":" << json_escape(result.winner_id)
      << ",\"loserId\":" << json_escape(result.loser_id)
      << ",\"reason\":" << json_escape(result.reason) << ",\"forfeit\":";
  if (result.forfeit_id) {
    out << "{\"botId\":" << json_escape(*result.forfeit_id)
        << ",\"classification\":" << json_escape(*result.classification)
        << ",\"detail\":" << json_escape(result.detail) << '}';
  } else {
    out << "null";
  }
  std::size_t one_count = 0;
  std::size_t two_count = 0;
  std::uint64_t one_total = 0;
  std::uint64_t two_total = 0;
  std::uint64_t one_max = 0;
  std::uint64_t two_max = 0;
  for (const TurnRecord &turn : result.turns) {
    if (turn.player == 0) {
      ++one_count;
      one_total += turn.duration_micros;
      one_max = std::max(one_max, turn.duration_micros);
    } else {
      ++two_count;
      two_total += turn.duration_micros;
      two_max = std::max(two_max, turn.duration_micros);
    }
  }
  out << "},\"timings\":{\"totalMicros\":" << (one_total + two_total)
      << ",\"playerOne\":{\"decisions\":" << one_count << ",\"totalMicros\":" << one_total
      << ",\"maxMicros\":" << one_max
      << "},\"playerTwo\":{\"decisions\":" << two_count << ",\"totalMicros\":" << two_total
      << ",\"maxMicros\":" << two_max
      << "}},\"provenance\":{\"refereeVersion\":\"papersoccer-codingame-referee-v1\"}}";
  return out.str();
}

}  // namespace

ReplayResult validate_transcript(const std::vector<std::string> &actions,
                                 bool allow_incomplete) {
  GameState state = make_initial_state(contest_rules());
  std::size_t edges = 0;
  for (const std::string &action : actions) {
    state = apply_complete_turn(state, action, nullptr);
    edges += action.size();
  }
  if (!allow_incomplete && !is_terminal(state)) {
    throw std::invalid_argument("transcript does not reach a terminal state");
  }
  std::optional<std::string> terminal_reason;
  if (is_terminal(state)) {
    terminal_reason = is_goal_point(state.config, state.ball)
                          ? "goal"
                          : "blocked_mover";
  }
  return ReplayResult{is_terminal(state), winner(state),
                      std::move(terminal_reason), actions.size(), edges};
}

std::string run_match_json(const RefereeConfig &config) {
  if (config.player_one_executable.empty() || config.player_two_executable.empty() ||
      config.player_one_id.empty() || config.player_two_id.empty() ||
      config.player_one_id == config.player_two_id) {
    throw std::invalid_argument("invalid referee configuration");
  }
  std::signal(SIGPIPE, SIG_IGN);
  const auto match_started = Clock::now();
  Child first = spawn_child(config.player_one_executable, config.player_one_id, 0);
  Child second;
  try {
    second = spawn_child(config.player_two_executable, config.player_two_id, 1);
  } catch (...) {
    terminate_child(first);
    throw;
  }
  ChildCleanup cleanup{first, second};
  GameState state = make_initial_state(contest_rules());
  std::string previous = "-";
  MatchResult result;
  for (std::size_t turn_index = 0; !is_terminal(state); ++turn_index) {
    Child &current = state.to_move == Player::One ? first : second;
    const std::uint32_t timeout = current.first_decision
                                      ? config.first_timeout_ms
                                      : config.later_timeout_ms;
    const std::string prompt = previous == "-"
                                   ? "1\n-\n"
                                   : std::to_string(previous.size()) + "\n" + previous + "\n";
    const QueuedOutput queued =
        drain_queued_output(current, config.output_limit_bytes,
                            config.stderr_limit_bytes);
    std::optional<std::string> queued_action;
    std::optional<std::string> queued_failure = queued.classification;
    std::string queued_detail = queued.detail;
    if (!queued_failure) {
      const std::size_t newline = current.stdout_buffer.find('\n');
      if (newline != std::string::npos) {
        queued_action = current.stdout_buffer.substr(0, newline);
        current.stdout_buffer.erase(0, newline + 1);
        if (!queued_action->empty() && queued_action->back() == '\r') {
          queued_action->pop_back();
        }
        queued_failure = "output-after-handoff";
        queued_detail = "complete response was emitted before its prompt";
      } else if (!current.stdout_buffer.empty()) {
        queued_action = current.stdout_buffer;
        current.stdout_buffer.clear();
        queued_failure = "output-after-handoff";
        queued_detail = "partial response was emitted before its prompt";
      } else if (current.exit_observed || queued.stdout_eof) {
        queued_failure = "crash";
        queued_detail = "bot exited before its prompt";
      }
    }
    if (queued_failure) {
      TurnRecord turn{turn_index, current.id, current.player, previous,
                      queued_action, false, 0, timeout, queued_failure, {}};
      result.turns.push_back(std::move(turn));
      result.forfeit_id = current.id;
      result.classification = *queued_failure;
      result.detail = std::move(queued_detail);
      break;
    }
    try {
      write_all(current.input, prompt);
    } catch (const std::exception &error) {
      TurnRecord turn{turn_index, current.id, current.player, previous, std::nullopt,
                      false, 0, timeout, std::string("crash"), {}};
      result.turns.push_back(std::move(turn));
      result.forfeit_id = current.id;
      result.classification = "crash";
      result.detail = error.what();
      break;
    }
    ReadResult response = read_line(current, timeout, config.output_limit_bytes,
                                    config.stderr_limit_bytes);
    TurnRecord turn{turn_index, current.id, current.player, previous, response.line,
                    false, response.duration_micros, timeout, std::nullopt, {}};
    if (!response.line) {
      turn.failure = response.classification;
      result.turns.push_back(std::move(turn));
      result.forfeit_id = current.id;
      result.classification = response.classification;
      result.detail = response.detail;
      break;
    }
    try {
      GameState next = apply_complete_turn(state, *response.line, &turn.moves);
      std::optional<std::string> framing_failure;
      std::string framing_detail;
      if (response.extra_complete_line || !current.stdout_buffer.empty()) {
        framing_failure = is_terminal(next) ? "output-after-terminal"
                                            : "output-after-handoff";
        framing_detail = "additional output was emitted for one prompt";
      } else if (is_terminal(next)) {
        const QueuedOutput terminal_output =
            drain_queued_output(current, config.output_limit_bytes,
                                config.stderr_limit_bytes);
        if (terminal_output.classification) {
          framing_failure = terminal_output.classification;
          framing_detail = terminal_output.detail;
        } else if (!current.stdout_buffer.empty()) {
          framing_failure = "output-after-terminal";
          framing_detail = "output was emitted after the terminal response";
        }
      }
      if (framing_failure) {
        turn.moves.clear();
        turn.failure = *framing_failure;
        result.turns.push_back(std::move(turn));
        result.forfeit_id = current.id;
        result.classification = *framing_failure;
        result.detail = std::move(framing_detail);
        break;
      }
      turn.accepted = true;
      result.turns.push_back(std::move(turn));
      state = std::move(next);
      previous = *response.line;
      current.first_decision = false;
    } catch (const std::invalid_argument &error) {
      turn.moves.clear();
      turn.failure = error.what();
      result.turns.push_back(std::move(turn));
      result.forfeit_id = current.id;
      result.classification = error.what();
      result.detail = "response rejected atomically";
      break;
    }
  }
  if (result.forfeit_id) {
    result.loser_id = *result.forfeit_id;
    result.winner_id = result.loser_id == config.player_one_id
                           ? config.player_two_id
                           : config.player_one_id;
    result.reason = "forfeit";
  } else {
    const Player winning_player = winner(state).value();
    result.winner_id = player_id(config, winning_player);
    result.loser_id = player_id(config, opponent(winning_player));
    result.reason = is_goal_point(state.config, state.ball)
                        ? "goal"
                        : "blocked_mover";
  }
  result.total_micros = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - match_started).count());
  return serialize(config, result);
}

}  // namespace papersoccer::codingame
