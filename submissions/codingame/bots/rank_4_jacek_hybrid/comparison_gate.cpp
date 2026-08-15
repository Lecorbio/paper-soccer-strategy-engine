#include "comparison_gate_engine.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#include "papersoccer/rules.hpp"
#include "opening_bank_internal.hpp"

namespace ps = papersoccer;
namespace gate = papersoccer::rank4_jacek_gate;
namespace opening_bank = papersoccer::opening_bank;

namespace {

using Clock = std::chrono::steady_clock;

enum class Profile { FixedNodes, ActualClock };
enum class Engine { Candidate, Reference };
enum class ReferenceEngine { Rank4, HybridControl };

struct Config {
  Profile profile{Profile::FixedNodes};
  ReferenceEngine reference_engine{ReferenceEngine::Rank4};
  std::vector<std::filesystem::path> banks;
  std::string expected_role{"development"};
  std::vector<std::uint64_t> expected_seeds;
  std::vector<std::string> expected_sha256;
  std::vector<std::uint64_t> expected_bytes;
  std::vector<int> expected_depths;
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
  std::vector<std::uint64_t> generated_pairs;
  std::vector<int> play_depths;
#endif
  int max_turns{240};
  std::uint64_t candidate_nodes{30'000};
  std::uint64_t reference_nodes{30'000};
  std::uint32_t candidate_first_ms{800};
  std::uint32_t candidate_later_ms{155};
  std::uint32_t reference_first_ms{800};
  std::uint32_t reference_later_ms{155};
  std::uint32_t operational_first_ms{1'000};
  std::uint32_t operational_later_ms{200};
  std::uint8_t candidate_exact_proof_mask{};
  std::uint8_t reference_exact_proof_mask{};
  bool self_test{};
};

struct Invocation {
  gate::EngineDecision decision;
  double milliseconds{};
  bool first{};
  bool exception{};
  bool illegal{};
  bool hard_timeout{};
  bool soft_overrun{};
};

struct EngineTotals {
  std::uint64_t invocations{};
  std::uint64_t searches{};
  std::uint64_t nodes{};
  std::uint64_t completed_depth{};
  std::uint64_t attempted_depth{};
  std::uint64_t budget_exhaustions{};
  std::uint64_t illegal_actions{};
  std::uint64_t operational_failures{};
  std::uint64_t exceptions{};
  std::uint64_t hard_timeouts{};
  std::uint64_t soft_overruns{};
  std::uint32_t maximum_completed_depth{};
  std::uint32_t maximum_attempted_depth{};
  std::vector<std::uint64_t> node_samples;
  std::vector<double> first_times;
  std::vector<double> later_times;
  std::uint64_t rebound_goal_probes{};
  std::uint64_t rebound_goal_hits{};
  std::uint64_t rebound_loss_hits{};
  std::uint64_t root_rebound_probes{};
  std::uint64_t root_rebound_win_hits{};
  std::uint64_t root_rebound_loss_hits{};
  std::uint64_t leaf_rebound_probes{};
  std::uint64_t leaf_rebound_win_hits{};
  std::uint64_t leaf_rebound_loss_hits{};
  std::uint64_t exchange_ply1_probes{};
  std::uint64_t exchange_ply1_win_hits{};
  std::uint64_t exchange_ply1_loss_hits{};
  std::uint64_t exchange_ply1_cutoffs{};
  std::uint64_t exchange_ply2_probes{};
  std::uint64_t exchange_ply2_win_hits{};
  std::uint64_t exchange_ply2_loss_hits{};
  std::uint64_t exchange_ply2_cutoffs{};
};

struct ColorTotals {
  int games{};
  int candidate_wins{};
  int reference_wins{};
  int unfinished{};
  int failed{};
};

struct Summary {
  int games{};
  int candidate_wins{};
  int reference_wins{};
  int unfinished{};
  int failed{};
#if defined(PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING)
  int candidate_sweeps{};
  int reference_sweeps{};
  int split_pairs{};
  int unresolved_pairs{};
#endif
  std::array<ColorTotals, 2> colors{};
  EngineTotals candidate;
  EngineTotals reference;
};

struct GameResult {
  std::optional<int> winner;
  int turns{};
  bool unfinished{};
  bool failed{};
  EngineTotals candidate;
  EngineTotals reference;
};

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

int player_id(ps::Player player) {
  return player == ps::Player::One ? 0 : 1;
}

template <typename UInt>
UInt parse_unsigned(std::string_view raw, std::string_view label,
                    bool allow_zero = false) {
  static_assert(std::is_unsigned_v<UInt>);
  UInt value{};
  const char *begin = raw.data();
  const char *end = begin + raw.size();
  const auto [position, error] = std::from_chars(begin, end, value);
  if (raw.empty() || raw.front() == '-' || error != std::errc{} ||
      position != end || (!allow_zero && value == 0)) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  return value;
}

int parse_int(std::string_view raw, std::string_view label,
              bool allow_zero = false) {
  const unsigned int value =
      parse_unsigned<unsigned int>(raw, label, allow_zero);
  if (value > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string(label) + " is too large");
  }
  return static_cast<int>(value);
}

bool parse_bool(std::string_view raw, std::string_view label) {
  const unsigned int value =
      parse_unsigned<unsigned int>(raw, label, true);
  if (value > 1U) {
    throw std::invalid_argument(std::string(label) + " must be 0 or 1");
  }
  return value != 0;
}

std::uint8_t parse_proof_mask(std::string_view raw,
                              std::string_view label) {
  const unsigned int value =
      parse_unsigned<unsigned int>(raw, label, true);
  if (value > 15U) {
    throw std::invalid_argument(std::string(label) +
                                " must be between 0 and 15");
  }
  return static_cast<std::uint8_t>(value);
}

std::vector<int> parse_depths(std::string_view raw) {
  std::vector<int> depths;
  std::size_t begin = 0;
  while (begin <= raw.size()) {
    const std::size_t separator = raw.find(',', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? raw.size()
                                : separator;
    const int depth =
        parse_int(raw.substr(begin, end - begin), "opening depth", true);
    if (std::find(depths.begin(), depths.end(), depth) != depths.end()) {
      throw std::invalid_argument("opening depths must be unique");
    }
    depths.push_back(depth);
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  if (depths.empty() || depths.size() > 64U) {
    throw std::invalid_argument("opening depth list is invalid");
  }
  return depths;
}

std::vector<std::uint64_t> parse_seeds(std::string_view raw) {
  std::vector<std::uint64_t> seeds;
  std::size_t begin = 0;
  while (begin <= raw.size()) {
    const std::size_t separator = raw.find(',', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? raw.size()
                                : separator;
    seeds.push_back(parse_unsigned<std::uint64_t>(
        raw.substr(begin, end - begin), "expected seed", true));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  if (seeds.empty() || seeds.size() > 64U) {
    throw std::invalid_argument("expected seed list is invalid");
  }
  return seeds;
}

#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
std::vector<std::uint64_t> parse_generated_pairs(std::string_view raw) {
  std::vector<std::uint64_t> pairs;
  std::size_t begin = 0;
  while (begin <= raw.size()) {
    const std::size_t separator = raw.find(',', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? raw.size()
                                : separator;
    pairs.push_back(parse_unsigned<std::uint64_t>(
        raw.substr(begin, end - begin), "generated pair count"));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  if (pairs.empty() || pairs.size() > 64U) {
    throw std::invalid_argument("generated pair-count list is invalid");
  }
  return pairs;
}
#endif

#if !defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
std::vector<std::uint64_t> parse_expected_bytes(std::string_view raw) {
  std::vector<std::uint64_t> sizes;
  std::size_t begin = 0;
  while (begin <= raw.size()) {
    const std::size_t separator = raw.find(',', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? raw.size()
                                : separator;
    sizes.push_back(parse_unsigned<std::uint64_t>(
        raw.substr(begin, end - begin), "expected bank bytes"));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  if (sizes.empty() || sizes.size() > 64U) {
    throw std::invalid_argument("expected bank byte list is invalid");
  }
  return sizes;
}

std::vector<std::string> parse_sha256(std::string_view raw) {
  std::vector<std::string> hashes;
  std::size_t begin = 0;
  while (begin <= raw.size()) {
    const std::size_t separator = raw.find(',', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? raw.size()
                                : separator;
    const std::string_view hash = raw.substr(begin, end - begin);
    const bool valid = hash.size() == 64U &&
                       std::all_of(hash.begin(), hash.end(), [](char value) {
                         return (value >= '0' && value <= '9') ||
                                (value >= 'a' && value <= 'f');
                       });
    if (!valid) {
      throw std::invalid_argument("expected bank SHA-256 is invalid");
    }
    hashes.emplace_back(hash);
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  return hashes;
}
#endif

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants{{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
}};

constexpr std::uint32_t rotate_right(std::uint32_t value,
                                     unsigned int amount) {
  return (value >> amount) | (value << (32U - amount));
}

std::string sha256(std::string_view raw) {
  std::vector<std::uint8_t> bytes(raw.begin(), raw.end());
  const std::uint64_t bit_count =
      static_cast<std::uint64_t>(bytes.size()) * 8U;
  bytes.push_back(0x80U);
  while (bytes.size() % 64U != 56U) {
    bytes.push_back(0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    bytes.push_back(static_cast<std::uint8_t>(bit_count >> shift));
  }
  std::array<std::uint32_t, 8> state{{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  }};
  for (std::size_t offset = 0; offset < bytes.size(); offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      words[index] =
          (static_cast<std::uint32_t>(bytes[offset + 4U * index]) << 24U) |
          (static_cast<std::uint32_t>(bytes[offset + 4U * index + 1U]) << 16U) |
          (static_cast<std::uint32_t>(bytes[offset + 4U * index + 2U]) << 8U) |
          static_cast<std::uint32_t>(bytes[offset + 4U * index + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t one = rotate_right(words[index - 15U], 7U) ^
                                rotate_right(words[index - 15U], 18U) ^
                                (words[index - 15U] >> 3U);
      const std::uint32_t zero = rotate_right(words[index - 2U], 17U) ^
                                 rotate_right(words[index - 2U], 19U) ^
                                 (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + one + words[index - 7U] + zero;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum_one = rotate_right(e, 6U) ^
                                    rotate_right(e, 11U) ^
                                    rotate_right(e, 25U);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temporary_one =
          h + sum_one + choice + kSha256RoundConstants[index] + words[index];
      const std::uint32_t sum_zero = rotate_right(a, 2U) ^
                                     rotate_right(a, 13U) ^
                                     rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary_two = sum_zero + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary_one;
      d = c;
      c = b;
      b = a;
      a = temporary_one + temporary_two;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::hex << std::setfill('0');
  for (const std::uint32_t value : state) {
    out << std::setw(8) << value;
  }
  return out.str();
}

class UniqueFd {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int value) : value_(value) {}
  ~UniqueFd() { reset(); }

  UniqueFd(const UniqueFd &) = delete;
  UniqueFd &operator=(const UniqueFd &) = delete;

  UniqueFd(UniqueFd &&other) noexcept : value_(other.release()) {}
  UniqueFd &operator=(UniqueFd &&other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] int release() noexcept {
    const int released = value_;
    value_ = -1;
    return released;
  }
  void reset(int replacement = -1) noexcept {
    if (value_ >= 0) {
      ::close(value_);
    }
    value_ = replacement;
  }

 private:
  int value_{-1};
};

[[noreturn]] void throw_system_error(std::string_view operation) {
  const int saved_errno = errno;
  throw std::runtime_error(std::string(operation) + ": " +
                           std::strerror(saved_errno));
}

UniqueFd duplicate_fd(int descriptor) {
  const int duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
  if (duplicate < 0) {
    throw_system_error("could not duplicate directory descriptor");
  }
  return UniqueFd(duplicate);
}

struct StatIdentity {
  dev_t device{};
  ino_t inode{};
  mode_t mode{};
  nlink_t links{};
  off_t size{};
  timespec modified{};
  timespec changed{};
};

timespec modified_time(const struct stat &status) noexcept {
#if defined(__APPLE__)
  return status.st_mtimespec;
#else
  return status.st_mtim;
#endif
}

timespec changed_time(const struct stat &status) noexcept {
#if defined(__APPLE__)
  return status.st_ctimespec;
#else
  return status.st_ctim;
#endif
}

StatIdentity stat_identity(const struct stat &status) noexcept {
  return StatIdentity{status.st_dev, status.st_ino, status.st_mode,
                      status.st_nlink, status.st_size,
                      modified_time(status), changed_time(status)};
}

bool same_time(const timespec &left, const timespec &right) noexcept {
  return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

bool same_identity(const StatIdentity &left,
                   const StatIdentity &right) noexcept {
  return left.device == right.device && left.inode == right.inode &&
         left.mode == right.mode && left.links == right.links &&
         left.size == right.size &&
         same_time(left.modified, right.modified) &&
         same_time(left.changed, right.changed);
}

struct stat descriptor_status(int descriptor, std::string_view label) {
  struct stat status {};
  if (::fstat(descriptor, &status) != 0) {
    throw_system_error(label);
  }
  return status;
}

struct stat pathname_status(int parent, const std::string &name,
                            std::string_view label) {
  struct stat status {};
  if (::fstatat(parent, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
    throw_system_error(label);
  }
  return status;
}

std::vector<std::string> canonical_path_components(
    std::string_view raw_path) {
  if (raw_path.empty() || raw_path.front() == '/' ||
      raw_path.back() == '/') {
    throw std::invalid_argument("opening bank path must be root-relative");
  }
  std::vector<std::string> components;
  std::size_t begin = 0;
  while (begin < raw_path.size()) {
    const std::size_t separator = raw_path.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? raw_path.size()
                                : separator;
    const std::string_view component = raw_path.substr(begin, end - begin);
    if (component.empty() || component == "." || component == ".." ||
        component.find('\0') != std::string_view::npos) {
      throw std::invalid_argument(
          "opening bank path has a noncanonical component");
    }
    components.emplace_back(component);
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  if (components.empty()) {
    throw std::invalid_argument("opening bank path is empty");
  }
  return components;
}

struct DirectoryChain {
  std::vector<UniqueFd> descriptors;
  std::vector<StatIdentity> identities;
  std::vector<std::string> components;
  std::string basename;
};

DirectoryChain traverse_parent_directory(int root_descriptor,
                                         std::string_view raw_path) {
  const std::vector<std::string> path =
      canonical_path_components(raw_path);
  DirectoryChain chain;
  chain.descriptors.push_back(duplicate_fd(root_descriptor));
  const struct stat root_status = descriptor_status(
      chain.descriptors.front().get(), "could not stat repository root");
  if (!S_ISDIR(root_status.st_mode)) {
    throw std::invalid_argument("repository root is not a directory");
  }
  chain.identities.push_back(stat_identity(root_status));
  for (std::size_t index = 0; index + 1U < path.size(); ++index) {
    const int parent = chain.descriptors.back().get();
    const struct stat pathname = pathname_status(
        parent, path[index], "could not stat opening bank directory");
    if (!S_ISDIR(pathname.st_mode)) {
      throw std::invalid_argument(
          "opening bank intermediate component is not a directory");
    }
    const int opened = ::openat(parent, path[index].c_str(),
                                O_DIRECTORY | O_RDONLY | O_CLOEXEC |
                                    O_NOFOLLOW);
    if (opened < 0) {
      throw_system_error("could not open opening bank directory");
    }
    UniqueFd descriptor(opened);
    const struct stat opened_status = descriptor_status(
        descriptor.get(), "could not stat opened bank directory");
    if (!S_ISDIR(opened_status.st_mode) ||
        !same_identity(stat_identity(pathname),
                       stat_identity(opened_status))) {
      throw std::invalid_argument(
          "opening bank directory changed during traversal");
    }
    chain.components.push_back(path[index]);
    chain.identities.push_back(stat_identity(opened_status));
    chain.descriptors.push_back(std::move(descriptor));
  }
  chain.basename = path.back();
  return chain;
}

void recheck_directory_chain(const DirectoryChain &chain) {
  for (std::size_t index = 0; index < chain.descriptors.size(); ++index) {
    const struct stat descriptor = descriptor_status(
        chain.descriptors[index].get(),
        "could not restat opening bank directory");
    if (!same_identity(chain.identities[index],
                       stat_identity(descriptor))) {
      throw std::invalid_argument(
          "opening bank directory descriptor changed during read");
    }
    if (index != 0) {
      const struct stat pathname = pathname_status(
          chain.descriptors[index - 1U].get(),
          chain.components[index - 1U],
          "could not restat opening bank directory path");
      if (!S_ISDIR(pathname.st_mode) ||
          !same_identity(chain.identities[index],
                         stat_identity(pathname))) {
        throw std::invalid_argument(
            "opening bank directory pathname changed during read");
      }
    }
  }
}

UniqueFd open_repository_root() {
  const int descriptor =
      ::open(".", O_DIRECTORY | O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    throw_system_error("could not pin repository root");
  }
  return UniqueFd(descriptor);
}

struct SafeBankRead {
  std::string bytes;
  opening_bank::Bank bank;
};

using SafeReadHook =
    std::function<void(int parent_descriptor, const std::string &basename)>;

class SafeBankReader {
 public:
  explicit SafeBankReader(UniqueFd root) : root_(std::move(root)) {
    const struct stat status =
        descriptor_status(root_.get(), "could not stat safe-reader root");
    if (!S_ISDIR(status.st_mode)) {
      throw std::invalid_argument("safe-reader root is not a directory");
    }
  }

  SafeBankRead read_bank(const std::filesystem::path &path,
                         std::optional<std::uint64_t> expected_bytes,
                         const std::string &expected_sha256,
                         const SafeReadHook &after_open = {}) {
    const std::string raw_path = path.generic_string();
    if (std::find(consumed_paths_.begin(), consumed_paths_.end(), raw_path) !=
        consumed_paths_.end()) {
      throw std::invalid_argument(
          "opening bank path may be read only once");
    }
    consumed_paths_.push_back(raw_path);
    DirectoryChain chain = traverse_parent_directory(root_.get(), raw_path);
    const int parent = chain.descriptors.back().get();
    const struct stat pathname_before = pathname_status(
        parent, chain.basename, "could not stat opening bank");
    if (!S_ISREG(pathname_before.st_mode)) {
      throw std::invalid_argument("opening bank is not a regular file");
    }
    const int opened = ::openat(parent, chain.basename.c_str(),
                                O_RDONLY | O_NONBLOCK | O_CLOEXEC |
                                    O_NOFOLLOW);
    if (opened < 0) {
      throw_system_error("could not open opening bank");
    }
    UniqueFd descriptor(opened);
    const struct stat descriptor_before = descriptor_status(
        descriptor.get(), "could not stat opened bank");
    const StatIdentity initial_identity = stat_identity(pathname_before);
    if (!same_identity(initial_identity,
                       stat_identity(descriptor_before))) {
      throw std::invalid_argument(
          "opening bank pathname changed during open");
    }
    if (!S_ISREG(descriptor_before.st_mode) ||
        (descriptor_before.st_mode & 07777) != 0444 ||
        descriptor_before.st_nlink != 1) {
      throw std::invalid_argument(
          "opening bank must be a regular mode-0444 single-link file");
    }
    if (descriptor_before.st_size <= 0) {
      throw std::invalid_argument("opening bank size must be positive");
    }
    const std::uint64_t actual_bytes =
        static_cast<std::uint64_t>(descriptor_before.st_size);
    if (expected_bytes.has_value() && *expected_bytes != actual_bytes) {
      throw std::invalid_argument("opening bank byte count mismatch");
    }
    if (actual_bytes >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
      throw std::invalid_argument("opening bank is too large to read");
    }
    if (after_open) {
      after_open(parent, chain.basename);
    }
    std::string bytes(static_cast<std::size_t>(actual_bytes), '\0');
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      const std::size_t count = std::min(
          bytes.size() - offset,
          static_cast<std::size_t>(
              std::numeric_limits<ssize_t>::max()));
      const ssize_t read_count =
          ::read(descriptor.get(), bytes.data() + offset, count);
      if (read_count < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw_system_error("could not read opening bank");
      }
      if (read_count == 0) {
        throw std::invalid_argument("opening bank ended before expected size");
      }
      offset += static_cast<std::size_t>(read_count);
    }
    if (sha256(bytes) != expected_sha256) {
      throw std::invalid_argument("opening bank file SHA-256 mismatch");
    }
    opening_bank::Bank bank =
        opening_bank::parse_bank_text(bytes, raw_path);
    const struct stat descriptor_after = descriptor_status(
        descriptor.get(), "could not restat opened bank");
    const struct stat pathname_after = pathname_status(
        parent, chain.basename, "could not restat opening bank path");
    if (!same_identity(initial_identity,
                       stat_identity(descriptor_after)) ||
        !same_identity(initial_identity, stat_identity(pathname_after))) {
      throw std::invalid_argument("opening bank changed during read");
    }
    recheck_directory_chain(chain);
    return SafeBankRead{std::move(bytes), std::move(bank)};
  }

 private:
  UniqueFd root_;
  std::vector<std::string> consumed_paths_;
};

void print_usage() {
  std::cout
      << "usage: comparison_gate [options]\n"
         "  --profile nodes|clock\n"
         "  --reference-engine rank4|hybrid-control\n"
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
         "  --expected-role development\n"
         "  --expected-depths N,N,N,N\n"
         "  --expected-seeds N,N,N,N (aligned with expected depths)\n"
         "  --generated-pairs N,N,N,N (aligned with expected depths)\n"
         "  --play-depths N[,N...] (subset of expected depths)\n"
#else
         "  --bank PATH (repeat for each preregistered TSV)\n"
         "  --expected-role development|validation|test\n"
         "  --expected-seeds N[,N...] (aligned with expected depths)\n"
         "  --expected-sha256 H[,H...] (aligned with bank paths)\n"
         "  --expected-bytes N[,N...] (aligned with bank paths)\n"
         "  --expected-depths N[,N...]\n"
#endif
         "  --max-turns N\n"
         "  --candidate-nodes N\n"
         "  --reference-nodes N\n"
         "  --candidate-first-ms N\n"
         "  --candidate-later-ms N\n"
         "  --reference-first-ms N\n"
         "  --reference-later-ms N\n"
         "  --operational-first-ms N\n"
         "  --operational-later-ms N\n"
         "  --candidate-exact-proof 0|1 (compatibility alias)\n"
         "  --candidate-exact-proof-mask 0..15\n"
         "  --reference-exact-proof-mask 0..15\n"
         "  --self-test\n";
}

Config parse_options(int argc, char **argv) {
  Config config;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--help" || option == "-h") {
      print_usage();
      std::exit(0);
    }
    if (option == "--self-test") {
      config.self_test = true;
      continue;
    }
    if (!option.starts_with("--")) {
      throw std::invalid_argument("unexpected positional argument: " +
                                  std::string(option));
    }
    if (++index >= argc) {
      throw std::invalid_argument("missing value after " +
                                  std::string(option));
    }
    const std::string_view value(argv[index]);
    if (option == "--profile") {
      if (value == "nodes") {
        config.profile = Profile::FixedNodes;
      } else if (value == "clock") {
        config.profile = Profile::ActualClock;
      } else {
        throw std::invalid_argument("profile must be nodes or clock");
      }
    } else if (option == "--reference-engine") {
      if (value == "rank4") {
        config.reference_engine = ReferenceEngine::Rank4;
      } else if (value == "hybrid-control") {
        config.reference_engine = ReferenceEngine::HybridControl;
      } else {
        throw std::invalid_argument(
            "reference engine must be rank4 or hybrid-control");
      }
    } else if (option == "--bank") {
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
      throw std::invalid_argument(
          "--bank is forbidden for the generated-bank campaign");
#else
      config.banks.emplace_back(value);
#endif
    } else if (option == "--expected-role") {
      config.expected_role = value;
    } else if (option == "--expected-seeds") {
      config.expected_seeds = parse_seeds(value);
    } else if (option == "--expected-sha256") {
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
      throw std::invalid_argument(
          "--expected-sha256 is forbidden for generated banks");
#else
      config.expected_sha256 = parse_sha256(value);
#endif
    } else if (option == "--expected-bytes") {
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
      throw std::invalid_argument(
          "--expected-bytes is forbidden for generated banks");
#else
      config.expected_bytes = parse_expected_bytes(value);
#endif
    } else if (option == "--expected-depths") {
      config.expected_depths = parse_depths(value);
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
    } else if (option == "--generated-pairs") {
      config.generated_pairs = parse_generated_pairs(value);
    } else if (option == "--play-depths") {
      config.play_depths = parse_depths(value);
#endif
    } else if (option == "--max-turns") {
      config.max_turns = parse_int(value, option);
    } else if (option == "--candidate-nodes") {
      config.candidate_nodes =
          parse_unsigned<std::uint64_t>(value, option);
    } else if (option == "--reference-nodes") {
      config.reference_nodes =
          parse_unsigned<std::uint64_t>(value, option);
    } else if (option == "--candidate-first-ms") {
      config.candidate_first_ms =
          parse_unsigned<std::uint32_t>(value, option);
    } else if (option == "--candidate-later-ms") {
      config.candidate_later_ms =
          parse_unsigned<std::uint32_t>(value, option);
    } else if (option == "--reference-first-ms") {
      config.reference_first_ms =
          parse_unsigned<std::uint32_t>(value, option);
    } else if (option == "--reference-later-ms") {
      config.reference_later_ms =
          parse_unsigned<std::uint32_t>(value, option);
    } else if (option == "--operational-first-ms") {
      config.operational_first_ms =
          parse_unsigned<std::uint32_t>(value, option);
    } else if (option == "--operational-later-ms") {
      config.operational_later_ms =
          parse_unsigned<std::uint32_t>(value, option);
    } else if (option == "--candidate-exact-proof") {
      config.candidate_exact_proof_mask =
          parse_bool(value, option) ? 15U : 0U;
    } else if (option == "--candidate-exact-proof-mask") {
      config.candidate_exact_proof_mask = parse_proof_mask(value, option);
    } else if (option == "--reference-exact-proof-mask") {
      config.reference_exact_proof_mask = parse_proof_mask(value, option);
    } else {
      throw std::invalid_argument("unknown option: " +
                                  std::string(option));
    }
  }
  if (!opening_bank::valid_phase(config.expected_role)) {
    throw std::invalid_argument("expected role is invalid");
  }
  if (config.expected_depths.empty()) {
    throw std::invalid_argument("--expected-depths is required");
  }
  if (config.expected_seeds.size() != config.expected_depths.size()) {
    throw std::invalid_argument(
        "expected seed count must match expected depth count");
  }
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
  if (!config.banks.empty() || !config.expected_sha256.empty() ||
      !config.expected_bytes.empty()) {
    throw std::invalid_argument(
        "generated-bank campaign cannot accept file-bank metadata");
  }
  if (config.expected_role != "development") {
    throw std::invalid_argument(
        "generated-bank campaign role must be development");
  }
  if (config.expected_depths.size() != 4U ||
      config.expected_seeds.size() != 4U ||
      config.generated_pairs.size() != 4U) {
    throw std::invalid_argument(
        "generated-bank campaign requires exactly four aligned depths, seeds, and pair counts");
  }
  if (config.play_depths.empty()) {
    throw std::invalid_argument("--play-depths is required");
  }
  for (const int depth : config.play_depths) {
    if (std::find(config.expected_depths.begin(),
                  config.expected_depths.end(), depth) ==
        config.expected_depths.end()) {
      throw std::invalid_argument(
          "play depths must be a subset of generated depths");
    }
  }
#else
  if (config.banks.empty()) {
    throw std::invalid_argument("at least one --bank is required");
  }
  if (config.expected_sha256.size() != config.banks.size()) {
    throw std::invalid_argument(
        "expected SHA-256 count must match bank path count");
  }
  if (!config.expected_bytes.empty() &&
      config.expected_bytes.size() != config.banks.size()) {
    throw std::invalid_argument(
        "expected byte count must match bank path count");
  }
#endif
#if !defined(PAPER_SOCCER_GATE_RANK4_SLOT_HAS_EXACT_PROOF)
  if (config.reference_engine == ReferenceEngine::Rank4 &&
      config.reference_exact_proof_mask != 0) {
    throw std::invalid_argument(
        "Rank-4 reference does not expose an exact proof mask");
  }
#endif
  return config;
}

bool same_state(const ps::GameState &left, const ps::GameState &right) {
  return left.config.width == right.config.width &&
         left.config.height == right.config.height &&
         left.config.goal_rule == right.config.goal_rule &&
         left.config.blocked_rule == right.config.blocked_rule &&
         left.ball == right.ball && left.to_move == right.to_move &&
         left.status == right.status && left.path == right.path &&
         left.used_segments == right.used_segments &&
         left.visit_count == right.visit_count;
}

#if !defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
std::vector<SafeBankRead> read_bank_contents(const Config &config) {
  SafeBankReader reader(open_repository_root());
  std::vector<SafeBankRead> contents;
  contents.reserve(config.banks.size());
  for (std::size_t index = 0; index < config.banks.size(); ++index) {
    const std::optional<std::uint64_t> expected_bytes =
        config.expected_bytes.empty()
            ? std::nullopt
            : std::optional<std::uint64_t>(config.expected_bytes[index]);
    contents.push_back(reader.read_bank(config.banks[index], expected_bytes,
                                        config.expected_sha256[index]));
  }
  return contents;
}
#endif

void validate_banks(const Config &config,
                    const std::vector<opening_bank::Bank> &banks) {
  opening_bank::validate_disjoint(banks);
  if (banks.size() != config.expected_depths.size()) {
    throw std::invalid_argument(
        "bank count does not match expected depth count");
  }
  std::vector<std::pair<int, std::uint64_t>> actual;
  actual.reserve(banks.size());
  for (const opening_bank::Bank &bank : banks) {
    if (bank.phase != config.expected_role) {
      throw std::invalid_argument("opening bank role mismatch");
    }
    if (bank.depth >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("opening bank depth is too large");
    }
    actual.emplace_back(static_cast<int>(bank.depth), bank.generator_seed);
  }
  std::vector<std::pair<int, std::uint64_t>> expected;
  expected.reserve(config.expected_depths.size());
  for (std::size_t index = 0; index < config.expected_depths.size(); ++index) {
    expected.emplace_back(config.expected_depths[index],
                          config.expected_seeds[index]);
  }
  std::sort(actual.begin(), actual.end());
  std::sort(expected.begin(), expected.end());
  if (actual != expected) {
    throw std::invalid_argument("opening bank depth/seed set mismatch");
  }
}

#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
std::vector<opening_bank::Bank> generate_all_banks(const Config &config) {
  std::vector<opening_bank::Bank> generated;
  generated.reserve(config.expected_depths.size());
  for (std::size_t index = 0; index < config.expected_depths.size(); ++index) {
    if (config.generated_pairs[index] >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
      throw std::invalid_argument("generated pair count is too large");
    }
    const std::size_t depth =
        static_cast<std::size_t>(config.expected_depths[index]);
    const std::size_t pairs =
        static_cast<std::size_t>(config.generated_pairs[index]);
    opening_bank::Bank bank = opening_bank::generate_bank(
        config.expected_role, depth, pairs, config.expected_seeds[index],
        generated);
    if (bank.phase != config.expected_role || bank.depth != depth ||
        bank.pairs != pairs ||
        bank.generator_seed != config.expected_seeds[index] ||
        bank.records.size() != pairs) {
      throw std::runtime_error(
          "generated opening bank does not match its aligned specification");
    }
    const std::string rendered = opening_bank::render_bank(bank);
    opening_bank::Bank reparsed = opening_bank::parse_bank_text(
        rendered, "generated-development-depth-" +
                      std::to_string(config.expected_depths[index]));
    if (reparsed != bank) {
      throw std::runtime_error(
          "generated opening bank failed render/parse roundtrip");
    }
    generated.push_back(std::move(reparsed));
  }
  validate_banks(config, generated);
  return generated;
}

std::vector<opening_bank::Bank> select_generated_banks(
    const Config &config,
    const std::vector<opening_bank::Bank> &generated) {
  std::vector<opening_bank::Bank> selected;
  selected.reserve(config.play_depths.size());
  for (const int play_depth : config.play_depths) {
    const auto bank = std::find_if(
        generated.begin(), generated.end(),
        [play_depth](const opening_bank::Bank &candidate) {
          return candidate.depth == static_cast<std::size_t>(play_depth);
        });
    if (bank == generated.end()) {
      throw std::runtime_error("generated play depth was not found");
    }
    selected.push_back(*bank);
  }
  opening_bank::validate_disjoint(selected);
  return selected;
}
#endif

std::vector<opening_bank::Bank> load_banks(const Config &config) {
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
  return select_generated_banks(config, generate_all_banks(config));
#else
  std::vector<SafeBankRead> contents = read_bank_contents(config);
  std::vector<opening_bank::Bank> banks;
  banks.reserve(contents.size());
  for (SafeBankRead &content : contents) {
    banks.push_back(std::move(content.bank));
  }
  validate_banks(config, banks);
  return banks;
#endif
}

ps::GameState codingame_opening(
    const opening_bank::OpeningRecord &record) {
  ps::GameState state = opening_bank::replay_transcript(record.moves);
  // The generic preregistration infrastructure uses the demo terminal-rule
  // labels. The accepted bank states are non-terminal, so switching to the
  // public CodinGame terminal rules preserves every edge, visit, ball, and
  // mover while making subsequent games use the exact target rules.
  state.config = codingame_rules();
  state.status = ps::Status::InProgress;
  if (ps::is_terminal(state) || ps::legal_moves(state).empty()) {
    throw std::invalid_argument(
        "preregistered opening is not live under CodinGame rules");
  }
  return state;
}

std::uint32_t soft_budget(const Config &config, Engine engine, bool first) {
  if (config.profile != Profile::ActualClock) {
    return 0;
  }
  if (engine == Engine::Candidate) {
    return first ? config.candidate_first_ms : config.candidate_later_ms;
  }
  return first ? config.reference_first_ms : config.reference_later_ms;
}

std::uint64_t node_budget(const Config &config, Engine engine) {
  return engine == Engine::Candidate ? config.candidate_nodes
                                     : config.reference_nodes;
}

bool apply_complete_action(ps::GameState &state,
                           const gate::EngineDecision &decision) {
  if (decision.moves.empty()) {
    return false;
  }
  ps::GameState next = state;
  const ps::Player mover = next.to_move;
  try {
    for (const ps::Move move : decision.moves) {
      if (ps::is_terminal(next) || next.to_move != mover) {
        return false;
      }
      next = ps::apply_move(next, move);
    }
  } catch (const std::exception &) {
    return false;
  }
  if (!ps::is_terminal(next) && next.to_move == mover) {
    return false;
  }
  state = std::move(next);
  return true;
}

Invocation invoke(Engine engine, ps::GameState &state, bool first,
                  const Config &config) {
  Invocation result;
  result.first = first;
  const std::uint32_t budget_ms = soft_budget(config, engine, first);
  const std::uint8_t proof_mask =
      engine == Engine::Candidate
          ? config.candidate_exact_proof_mask
          : config.reference_exact_proof_mask;
  const gate::EngineConfig engine_config{
      node_budget(config, engine), budget_ms, proof_mask};
  const Clock::time_point started = Clock::now();
  try {
    if (engine == Engine::Candidate) {
      result.decision = gate::choose_hybrid(state, engine_config);
    } else if (config.reference_engine == ReferenceEngine::Rank4) {
      result.decision = gate::choose_rank4(state, engine_config);
    } else {
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
      result.decision = gate::choose_hybrid_control(state, engine_config);
#else
      result.decision = gate::choose_hybrid(state, engine_config);
#endif
    }
  } catch (const std::exception &) {
    result.exception = true;
  } catch (...) {
    result.exception = true;
  }
  result.milliseconds =
      std::chrono::duration<double, std::milli>(Clock::now() - started)
          .count();
  if (!result.exception) {
    result.illegal = !apply_complete_action(state, result.decision);
  }
  if (config.profile == Profile::ActualClock) {
    const std::uint32_t hard_limit =
        first ? config.operational_first_ms : config.operational_later_ms;
    result.hard_timeout = result.milliseconds >= hard_limit;
    result.soft_overrun = result.milliseconds > budget_ms;
  }
  return result;
}

void add_invocation(EngineTotals &totals, const Invocation &invocation) {
  ++totals.invocations;
  (invocation.first ? totals.first_times : totals.later_times)
      .push_back(invocation.milliseconds);
  totals.illegal_actions += invocation.illegal ? 1U : 0U;
  totals.exceptions += invocation.exception ? 1U : 0U;
  totals.hard_timeouts += invocation.hard_timeout ? 1U : 0U;
  totals.soft_overruns += invocation.soft_overrun ? 1U : 0U;
  totals.operational_failures +=
      invocation.exception || invocation.hard_timeout ? 1U : 0U;
  if (invocation.exception) {
    return;
  }
  ++totals.searches;
  const gate::EngineDecision &decision = invocation.decision;
  totals.nodes += decision.nodes;
  totals.completed_depth += decision.completed_depth;
  totals.attempted_depth += decision.attempted_depth;
  totals.budget_exhaustions += decision.budget_exhausted ? 1U : 0U;
  totals.maximum_completed_depth =
      std::max(totals.maximum_completed_depth, decision.completed_depth);
  totals.maximum_attempted_depth =
      std::max(totals.maximum_attempted_depth, decision.attempted_depth);
  totals.node_samples.push_back(decision.nodes);
  totals.rebound_goal_probes += decision.rebound_goal_probes;
  totals.rebound_goal_hits += decision.rebound_goal_hits;
  totals.rebound_loss_hits += decision.rebound_loss_hits;
  totals.root_rebound_probes += decision.root_rebound_probes;
  totals.root_rebound_win_hits += decision.root_rebound_win_hits;
  totals.root_rebound_loss_hits += decision.root_rebound_loss_hits;
  totals.leaf_rebound_probes += decision.leaf_rebound_probes;
  totals.leaf_rebound_win_hits += decision.leaf_rebound_win_hits;
  totals.leaf_rebound_loss_hits += decision.leaf_rebound_loss_hits;
  totals.exchange_ply1_probes += decision.exchange_ply1_probes;
  totals.exchange_ply1_win_hits += decision.exchange_ply1_win_hits;
  totals.exchange_ply1_loss_hits += decision.exchange_ply1_loss_hits;
  totals.exchange_ply1_cutoffs += decision.exchange_ply1_cutoffs;
  totals.exchange_ply2_probes += decision.exchange_ply2_probes;
  totals.exchange_ply2_win_hits += decision.exchange_ply2_win_hits;
  totals.exchange_ply2_loss_hits += decision.exchange_ply2_loss_hits;
  totals.exchange_ply2_cutoffs += decision.exchange_ply2_cutoffs;
}

bool failed(const Invocation &invocation) {
  return invocation.exception || invocation.illegal ||
         invocation.hard_timeout;
}

GameResult play(const ps::GameState &opening, int candidate_player,
                const Config &config) {
  ps::GameState state = opening;
  GameResult result;
  std::array<unsigned int, 2> responses{};
  while (!ps::is_terminal(state) && result.turns < config.max_turns) {
    const bool candidate_turn = player_id(state.to_move) == candidate_player;
    const Engine engine =
        candidate_turn ? Engine::Candidate : Engine::Reference;
    unsigned int &engine_responses =
        responses[engine == Engine::Candidate ? 0U : 1U];
    const Invocation invocation =
        invoke(engine, state, engine_responses == 0, config);
    ++engine_responses;
    add_invocation(candidate_turn ? result.candidate : result.reference,
                   invocation);
    if (failed(invocation)) {
      result.failed = true;
      return result;
    }
    ++result.turns;
  }
  if (const std::optional<ps::Player> winner = ps::winner(state)) {
    result.winner = player_id(*winner);
  } else {
    result.unfinished = true;
  }
  return result;
}

void merge_engine(EngineTotals &into, const EngineTotals &from) {
  into.invocations += from.invocations;
  into.searches += from.searches;
  into.nodes += from.nodes;
  into.completed_depth += from.completed_depth;
  into.attempted_depth += from.attempted_depth;
  into.budget_exhaustions += from.budget_exhaustions;
  into.illegal_actions += from.illegal_actions;
  into.operational_failures += from.operational_failures;
  into.exceptions += from.exceptions;
  into.hard_timeouts += from.hard_timeouts;
  into.soft_overruns += from.soft_overruns;
  into.maximum_completed_depth =
      std::max(into.maximum_completed_depth, from.maximum_completed_depth);
  into.maximum_attempted_depth =
      std::max(into.maximum_attempted_depth, from.maximum_attempted_depth);
  into.node_samples.insert(into.node_samples.end(), from.node_samples.begin(),
                           from.node_samples.end());
  into.first_times.insert(into.first_times.end(), from.first_times.begin(),
                          from.first_times.end());
  into.later_times.insert(into.later_times.end(), from.later_times.begin(),
                          from.later_times.end());
  into.rebound_goal_probes += from.rebound_goal_probes;
  into.rebound_goal_hits += from.rebound_goal_hits;
  into.rebound_loss_hits += from.rebound_loss_hits;
  into.root_rebound_probes += from.root_rebound_probes;
  into.root_rebound_win_hits += from.root_rebound_win_hits;
  into.root_rebound_loss_hits += from.root_rebound_loss_hits;
  into.leaf_rebound_probes += from.leaf_rebound_probes;
  into.leaf_rebound_win_hits += from.leaf_rebound_win_hits;
  into.leaf_rebound_loss_hits += from.leaf_rebound_loss_hits;
  into.exchange_ply1_probes += from.exchange_ply1_probes;
  into.exchange_ply1_win_hits += from.exchange_ply1_win_hits;
  into.exchange_ply1_loss_hits += from.exchange_ply1_loss_hits;
  into.exchange_ply1_cutoffs += from.exchange_ply1_cutoffs;
  into.exchange_ply2_probes += from.exchange_ply2_probes;
  into.exchange_ply2_win_hits += from.exchange_ply2_win_hits;
  into.exchange_ply2_loss_hits += from.exchange_ply2_loss_hits;
  into.exchange_ply2_cutoffs += from.exchange_ply2_cutoffs;
}

void add_game(Summary &summary, const GameResult &game,
              int candidate_player) {
  ++summary.games;
  ColorTotals &color = summary.colors[candidate_player];
  ++color.games;
  merge_engine(summary.candidate, game.candidate);
  merge_engine(summary.reference, game.reference);
  if (game.failed) {
    ++summary.failed;
    ++color.failed;
  } else if (game.unfinished || !game.winner.has_value()) {
    ++summary.unfinished;
    ++color.unfinished;
  } else if (*game.winner == candidate_player) {
    ++summary.candidate_wins;
    ++color.candidate_wins;
  } else {
    ++summary.reference_wins;
    ++color.reference_wins;
  }
}

#if defined(PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING)
enum class PairOutcome { CandidateWin, ReferenceWin, Unresolved };

PairOutcome pair_outcome(const GameResult &game, int candidate_player) {
  if (game.failed || game.unfinished || !game.winner.has_value()) {
    return PairOutcome::Unresolved;
  }
  return *game.winner == candidate_player ? PairOutcome::CandidateWin
                                           : PairOutcome::ReferenceWin;
}

void add_opening_pair(Summary &summary, const GameResult &candidate_player0,
                      const GameResult &candidate_player1) {
  const PairOutcome first = pair_outcome(candidate_player0, 0);
  const PairOutcome second = pair_outcome(candidate_player1, 1);
  if (first == PairOutcome::CandidateWin &&
      second == PairOutcome::CandidateWin) {
    ++summary.candidate_sweeps;
  } else if (first == PairOutcome::ReferenceWin &&
             second == PairOutcome::ReferenceWin) {
    ++summary.reference_sweeps;
  } else if (first != PairOutcome::Unresolved &&
             second != PairOutcome::Unresolved) {
    ++summary.split_pairs;
  } else {
    ++summary.unresolved_pairs;
  }
  add_game(summary, candidate_player0, 0);
  add_game(summary, candidate_player1, 1);
}

void validate_pair_accounting(const Summary &summary) {
  if (summary.games < 0 || summary.games % 2 != 0) {
    throw std::runtime_error("paired-opening game count is invalid");
  }
  const int opening_pairs = summary.games / 2;
  if (summary.candidate_sweeps + summary.reference_sweeps +
          summary.split_pairs + summary.unresolved_pairs !=
      opening_pairs) {
    throw std::runtime_error("paired-opening category accounting mismatch");
  }
  if (summary.candidate_wins !=
      2 * summary.candidate_sweeps + summary.split_pairs) {
    throw std::runtime_error("candidate wins do not reconcile with pairs");
  }
  if (summary.reference_wins !=
      2 * summary.reference_sweeps + summary.split_pairs) {
    throw std::runtime_error("reference wins do not reconcile with pairs");
  }
}
#endif
void merge_summary(Summary &into, const Summary &from) {
  into.games += from.games;
  into.candidate_wins += from.candidate_wins;
  into.reference_wins += from.reference_wins;
  into.unfinished += from.unfinished;
  into.failed += from.failed;
#if defined(PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING)
  into.candidate_sweeps += from.candidate_sweeps;
  into.reference_sweeps += from.reference_sweeps;
  into.split_pairs += from.split_pairs;
  into.unresolved_pairs += from.unresolved_pairs;
#endif
  for (std::size_t color = 0; color < into.colors.size(); ++color) {
    into.colors[color].games += from.colors[color].games;
    into.colors[color].candidate_wins +=
        from.colors[color].candidate_wins;
    into.colors[color].reference_wins +=
        from.colors[color].reference_wins;
    into.colors[color].unfinished += from.colors[color].unfinished;
    into.colors[color].failed += from.colors[color].failed;
  }
  merge_engine(into.candidate, from.candidate);
  merge_engine(into.reference, from.reference);
}

template <typename Value>
Value percentile99(const std::vector<Value> &samples) {
  if (samples.empty()) {
    return Value{};
  }
  std::vector<Value> ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  const std::size_t rank = static_cast<std::size_t>(
      std::ceil(0.99 * static_cast<double>(ordered.size())));
  return ordered[std::max<std::size_t>(1U, rank) - 1U];
}

template <typename Value>
Value maximum(const std::vector<Value> &samples) {
  return samples.empty()
             ? Value{}
             : *std::max_element(samples.begin(), samples.end());
}

double average(std::uint64_t total, std::uint64_t count) {
  return count == 0 ? 0.0
                    : static_cast<double>(total) /
                          static_cast<double>(count);
}

void print_engine(std::string_view prefix, const EngineTotals &totals) {
  std::cout << ' ' << prefix << "_invocations=" << totals.invocations
            << ' ' << prefix << "_searches=" << totals.searches
            << ' ' << prefix << "_illegal=" << totals.illegal_actions
            << ' ' << prefix << "_operational="
            << totals.operational_failures
            << ' ' << prefix << "_exceptions=" << totals.exceptions
            << ' ' << prefix << "_hard_timeouts=" << totals.hard_timeouts
            << ' ' << prefix << "_soft_overruns=" << totals.soft_overruns
            << ' ' << prefix << "_nodes=" << totals.nodes
            << ' ' << prefix << "_nodes_avg="
            << average(totals.nodes, totals.searches)
            << ' ' << prefix << "_nodes_p99="
            << percentile99(totals.node_samples)
            << ' ' << prefix << "_nodes_max=" << maximum(totals.node_samples)
            << ' ' << prefix << "_depth_avg="
            << average(totals.completed_depth, totals.searches)
            << ' ' << prefix << "_depth_max="
            << totals.maximum_completed_depth
            << ' ' << prefix << "_attempted_depth_avg="
            << average(totals.attempted_depth, totals.searches)
            << ' ' << prefix << "_attempted_depth_max="
            << totals.maximum_attempted_depth
            << ' ' << prefix << "_exhaustions="
            << totals.budget_exhaustions
            << ' ' << prefix << "_first_ms_p99="
            << percentile99(totals.first_times)
            << ' ' << prefix << "_first_ms_max="
            << maximum(totals.first_times)
            << ' ' << prefix << "_later_ms_p99="
            << percentile99(totals.later_times)
            << ' ' << prefix << "_later_ms_max="
            << maximum(totals.later_times);
}

void print_proof(std::string_view prefix, const EngineTotals &totals) {
  std::cout << ' ' << prefix << "_proof_rebound="
            << totals.rebound_goal_probes << '/' << totals.rebound_goal_hits
            << '/' << totals.rebound_loss_hits
            << ' ' << prefix << "_proof_root="
            << totals.root_rebound_probes << '/'
            << totals.root_rebound_win_hits << '/'
            << totals.root_rebound_loss_hits
            << ' ' << prefix << "_proof_leaf="
            << totals.leaf_rebound_probes << '/'
            << totals.leaf_rebound_win_hits << '/'
            << totals.leaf_rebound_loss_hits
            << ' ' << prefix << "_proof_ply1="
            << totals.exchange_ply1_probes << '/'
            << totals.exchange_ply1_win_hits << '/'
            << totals.exchange_ply1_loss_hits << '/'
            << totals.exchange_ply1_cutoffs
            << ' ' << prefix << "_proof_ply2="
            << totals.exchange_ply2_probes << '/'
            << totals.exchange_ply2_win_hits << '/'
            << totals.exchange_ply2_loss_hits << '/'
            << totals.exchange_ply2_cutoffs;
}

void print_summary(std::string_view kind, std::uint64_t bank,
                   const Summary &summary) {
  std::cout << std::fixed << std::setprecision(3)
            << kind << " bank=";
  if (kind == "summary") {
    std::cout << "all";
  } else {
    std::cout << bank;
  }
  std::cout << " games=" << summary.games
            << " candidate_wins=" << summary.candidate_wins
            << " reference_wins=" << summary.reference_wins
            << " unfinished=" << summary.unfinished
            << " failed=" << summary.failed;
#if defined(PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING)
  std::cout << " candidate_sweeps=" << summary.candidate_sweeps
            << " reference_sweeps=" << summary.reference_sweeps
            << " split_pairs=" << summary.split_pairs
            << " unresolved_pairs=" << summary.unresolved_pairs;
#endif
  for (int color = 0; color < 2; ++color) {
    const ColorTotals &totals = summary.colors[color];
    std::cout << " candidate_p" << color << '='
              << totals.candidate_wins << '/' << totals.reference_wins << '/'
              << totals.unfinished << '/' << totals.failed << '/'
              << totals.games;
  }
  print_engine("candidate", summary.candidate);
  print_engine("reference", summary.reference);
  print_proof("candidate", summary.candidate);
  print_proof("reference", summary.reference);
  std::cout << '\n';
}

void print_configuration(const Config &config) {
  std::cout << "configuration profile="
      << (config.profile == Profile::FixedNodes ? "nodes" : "clock")
            << " reference_engine="
            << (config.reference_engine == ReferenceEngine::Rank4
                    ? "rank4"
                    : "hybrid-control")
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
            << " bank_count=" << config.play_depths.size()
#else
            << " bank_count=" << config.banks.size()
#endif
            << " expected_role=" << config.expected_role
            << " bank_validation=schema,header,role,depth,seed,replay,state-sha256,canonical-sha256,disjoint"
            << " max_turns=" << config.max_turns
            << " expected_depths=";
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
  for (std::size_t index = 0; index < config.play_depths.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << config.play_depths[index];
  }
#else
  for (std::size_t index = 0; index < config.expected_depths.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << config.expected_depths[index];
  }
#endif
  std::cout << " expected_seeds=";
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
  for (std::size_t index = 0; index < config.play_depths.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const auto expected = std::find(config.expected_depths.begin(),
                                    config.expected_depths.end(),
                                    config.play_depths[index]);
    if (expected == config.expected_depths.end()) {
      throw std::runtime_error("configuration play depth was not found");
    }
    const std::size_t expected_index = static_cast<std::size_t>(
        expected - config.expected_depths.begin());
    std::cout << config.expected_seeds[expected_index];
  }
#else
  for (std::size_t index = 0; index < config.expected_seeds.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << config.expected_seeds[index];
  }
#endif
  std::cout << " expected_sha256=";
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
  std::cout << "none";
#else
  for (std::size_t index = 0; index < config.expected_sha256.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << config.expected_sha256[index];
  }
#endif
  std::cout << " candidate_nodes=" << config.candidate_nodes
            << " reference_nodes=" << config.reference_nodes
            << " candidate_clock=" << config.candidate_first_ms << '/'
            << config.candidate_later_ms
            << " reference_clock=" << config.reference_first_ms << '/'
            << config.reference_later_ms
            << " operational_clock=" << config.operational_first_ms << '/'
            << config.operational_later_ms
            << " candidate_exact_proof_mask="
            << static_cast<unsigned int>(config.candidate_exact_proof_mask)
            << " reference_exact_proof_mask="
            << static_cast<unsigned int>(config.reference_exact_proof_mask)
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
            << " openings=preregistered-generated-public-rules"
#else
            << " openings=preregistered-public-rules"
#endif
               " replay_corrections=disabled transcripts=not-retained\n";
}

void run_self_test(const Config &config) {
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
  const std::vector<opening_bank::Bank> first = generate_all_banks(config);
  const std::vector<opening_bank::Bank> repeat = generate_all_banks(config);
  validate_banks(config, first);
  validate_banks(config, repeat);
  if (first.size() != 4U || repeat.size() != 4U || first != repeat ||
      first.front().records.empty()) {
    throw std::runtime_error(
        "four-bank generated corpus is not exactly deterministic");
  }
  const std::vector<opening_bank::Bank> selected =
      select_generated_banks(config, first);
  if (selected.size() != config.play_depths.size() || selected.empty()) {
    throw std::runtime_error("generated play-bank selection is invalid");
  }
  const ps::GameState opening =
      codingame_opening(selected.front().records.front());
#else
  const std::vector<SafeBankRead> contents = read_bank_contents(config);
  std::vector<opening_bank::Bank> first;
  std::vector<opening_bank::Bank> repeat;
  first.reserve(contents.size());
  repeat.reserve(contents.size());
  for (std::size_t index = 0; index < contents.size(); ++index) {
    first.push_back(contents[index].bank);
    repeat.push_back(opening_bank::parse_bank_text(
        contents[index].bytes, config.banks[index].generic_string()));
  }
  validate_banks(config, first);
  validate_banks(config, repeat);
  if (first != repeat || first.empty() || first.front().records.empty()) {
    throw std::runtime_error("preregistered bank load is not deterministic");
  }
  const ps::GameState opening = codingame_opening(first.front().records.front());
#endif
  ps::GameState paired_copy = opening;
  if (!same_state(opening, paired_copy)) {
    throw std::runtime_error("paired-color opening copy differs");
  }
#if defined(PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING)
  const auto resolved = [](int winner) {
    GameResult result;
    result.winner = winner;
    return result;
  };
  GameResult unresolved;
  unresolved.unfinished = true;
  Summary accounting;
  add_opening_pair(accounting, resolved(0), resolved(1));
  add_opening_pair(accounting, resolved(1), resolved(0));
  add_opening_pair(accounting, resolved(0), resolved(0));
  add_opening_pair(accounting, unresolved, unresolved);
  validate_pair_accounting(accounting);
  if (accounting.games != 8 || accounting.candidate_wins != 3 ||
      accounting.reference_wins != 3 || accounting.unfinished != 2 ||
      accounting.failed != 0 || accounting.candidate_sweeps != 1 ||
      accounting.reference_sweeps != 1 || accounting.split_pairs != 1 ||
      accounting.unresolved_pairs != 1) {
    throw std::runtime_error("held-out paired accounting self-test failed");
  }
  Summary partial_unresolved;
  add_opening_pair(partial_unresolved, resolved(0), unresolved);
  bool rejected_partial_pair = false;
  try {
    validate_pair_accounting(partial_unresolved);
  } catch (const std::runtime_error &) {
    rejected_partial_pair = true;
  }
  if (!rejected_partial_pair) {
    throw std::runtime_error("partial unresolved pair was not rejected");
  }
  std::cout << "heldout_pair_self_test candidate_sweeps=1"
               " reference_sweeps=1 split_pairs=1 unresolved_pairs=1"
               " exact_accounting=pass\n";
#elif defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
  enum class PairOutcome { Candidate, Reference, Unresolved };
  const std::array<std::array<PairOutcome, 2>, 4> pairs{{
      {PairOutcome::Candidate, PairOutcome::Candidate},
      {PairOutcome::Reference, PairOutcome::Reference},
      {PairOutcome::Candidate, PairOutcome::Reference},
      {PairOutcome::Unresolved, PairOutcome::Unresolved},
  }};
  std::array<int, 4> classifications{};
  for (const auto &pair : pairs) {
    if (pair[0] == PairOutcome::Candidate &&
        pair[1] == PairOutcome::Candidate) {
      ++classifications[0];
    } else if (pair[0] == PairOutcome::Reference &&
               pair[1] == PairOutcome::Reference) {
      ++classifications[1];
    } else if ((pair[0] == PairOutcome::Candidate &&
                pair[1] == PairOutcome::Reference) ||
               (pair[0] == PairOutcome::Reference &&
                pair[1] == PairOutcome::Candidate)) {
      ++classifications[2];
    } else if (pair[0] == PairOutcome::Unresolved &&
               pair[1] == PairOutcome::Unresolved) {
      ++classifications[3];
    } else {
      throw std::runtime_error(
          "partial unresolved generated pair was not rejected");
    }
  }
  if (classifications != std::array<int, 4>{1, 1, 1, 1}) {
    throw std::runtime_error(
        "generated paired accounting self-test failed");
  }
  std::cout << "heldout_pair_self_test candidate_sweeps=1"
               " reference_sweeps=1 split_pairs=1 unresolved_pairs=1"
               " exact_accounting=pass\n";
#endif
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
  std::cout << "self_test deterministic_bank_generation=pass"
               " exact_four_bank_count=pass render_parse_roundtrip=pass"
               " all_four_disjoint=pass paired_state=pass"
               " public_rules_only=pass transcripts=not-retained\n";
#else
  std::cout << "self_test deterministic_bank_load=pass"
               " strict_metadata_and_hashes=pass paired_state=pass"
               " public_rules_only=pass transcripts=not-retained\n";
#endif
}

#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
void sync_directory(int descriptor, std::string_view label) {
  if (::fsync(descriptor) != 0) {
    throw_system_error(label);
  }
}

void write_fixture_file(int parent, const std::string &name,
                        std::string_view bytes, mode_t final_mode) {
  const int opened = ::openat(parent, name.c_str(),
                              O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                  O_NOFOLLOW,
                              0600);
  if (opened < 0) {
    throw_system_error("could not create safe-reader fixture");
  }
  UniqueFd descriptor(opened);
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const std::size_t count = std::min(
        bytes.size() - offset,
        static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t written =
        ::write(descriptor.get(), bytes.data() + offset, count);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw_system_error("could not write safe-reader fixture");
    }
    if (written == 0) {
      throw std::runtime_error("safe-reader fixture write made no progress");
    }
    offset += static_cast<std::size_t>(written);
  }
  if (::fchmod(descriptor.get(), final_mode) != 0) {
    throw_system_error("could not set safe-reader fixture mode");
  }
  if (::fsync(descriptor.get()) != 0) {
    throw_system_error("could not sync safe-reader fixture");
  }
  sync_directory(parent, "could not sync safe-reader fixture directory");
}

UniqueFd make_fixture_directory(int parent, const std::string &name) {
  if (::mkdirat(parent, name.c_str(), 0700) != 0) {
    throw_system_error("could not create safe-reader fixture directory");
  }
  sync_directory(parent,
                 "could not sync safe-reader fixture parent directory");
  const int opened = ::openat(parent, name.c_str(),
                              O_DIRECTORY | O_RDONLY | O_CLOEXEC |
                                  O_NOFOLLOW);
  if (opened < 0) {
    throw_system_error("could not open safe-reader fixture directory");
  }
  UniqueFd descriptor(opened);
  const struct stat status = descriptor_status(
      descriptor.get(), "could not stat safe-reader fixture directory");
  if (!S_ISDIR(status.st_mode) || (status.st_mode & 07777) != 0700) {
    throw std::runtime_error("safe-reader fixture directory mode mismatch");
  }
  return descriptor;
}

template <typename Operation>
void require_safe_reader_rejection(Operation &&operation,
                                   std::string_view label) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::exception &) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error(std::string(label) + " was not rejected");
  }
}

void unlink_fixture(int parent, const std::string &name) {
  if (::unlinkat(parent, name.c_str(), 0) != 0) {
    throw_system_error("could not remove safe-reader fixture");
  }
}

void remove_fixture_directory(int parent, const std::string &name) {
  if (::unlinkat(parent, name.c_str(), AT_REMOVEDIR) != 0) {
    throw_system_error("could not remove safe-reader fixture directory");
  }
}

void run_safe_bank_reader_self_test(int argc, char **argv) {
  constexpr std::string_view kSelfTestOption =
      "--safe-bank-reader-self-test";
  constexpr std::string_view kRootOption = "--self-test-root";
  constexpr std::string_view kSelfTestRoot =
      "build/rank4-jacek-hybrid-tt-exact-collision/clang-release/tmp/"
      "safe-bank-reader-self-test";
  if (argc != 4 || std::string_view(argv[1]) != kSelfTestOption ||
      std::string_view(argv[2]) != kRootOption ||
      std::string_view(argv[3]) != kSelfTestRoot) {
    throw std::invalid_argument(
        "safe bank reader self-test requires the exact preregistered flags");
  }

  UniqueFd repository_root = open_repository_root();
  DirectoryChain parent_chain =
      traverse_parent_directory(repository_root.get(), kSelfTestRoot);
  const int temporary_parent = parent_chain.descriptors.back().get();
  const struct stat temporary_parent_status = descriptor_status(
      temporary_parent, "could not stat safe-reader TMPDIR");
  if (!S_ISDIR(temporary_parent_status.st_mode) ||
      (temporary_parent_status.st_mode & 07777) != 0700) {
    throw std::runtime_error("safe-reader TMPDIR mode mismatch");
  }
  struct stat preexisting {};
  if (::fstatat(temporary_parent, parent_chain.basename.c_str(),
                &preexisting, AT_SYMLINK_NOFOLLOW) == 0) {
    throw std::runtime_error("safe-reader self-test root already exists");
  }
  if (errno != ENOENT) {
    throw_system_error("could not verify safe-reader self-test root absence");
  }
  if (::mkdirat(temporary_parent, parent_chain.basename.c_str(), 0700) != 0) {
    throw_system_error("could not create safe-reader self-test root");
  }
  sync_directory(temporary_parent,
                 "could not sync safe-reader self-test parent");
  const int child_opened =
      ::openat(temporary_parent, parent_chain.basename.c_str(),
               O_DIRECTORY | O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (child_opened < 0) {
    throw_system_error("could not open safe-reader self-test root");
  }
  UniqueFd child(child_opened);
  const struct stat child_path = pathname_status(
      temporary_parent, parent_chain.basename,
      "could not stat safe-reader self-test root path");
  const struct stat child_descriptor = descriptor_status(
      child.get(), "could not stat safe-reader self-test root");
  if (!S_ISDIR(child_descriptor.st_mode) ||
      (child_descriptor.st_mode & 07777) != 0700 ||
      !same_identity(stat_identity(child_path),
                     stat_identity(child_descriptor))) {
    throw std::runtime_error("safe-reader self-test root identity mismatch");
  }

  UniqueFd ordinary = make_fixture_directory(child.get(), "ordinary");
  UniqueFd inner = make_fixture_directory(ordinary.get(), "inner");
  const opening_bank::Bank synthetic =
      opening_bank::generate_bank("development", 1U, 1U,
                                  UINT64_C(7640891576956012809));
  const std::string fixture = opening_bank::render_bank(synthetic);
  const std::string fixture_hash = sha256(fixture);
  const std::uint64_t fixture_size =
      static_cast<std::uint64_t>(fixture.size());
  write_fixture_file(inner.get(), "bank.tsv", fixture, 0444);
  write_fixture_file(child.get(), "hard-source.tsv", fixture, 0444);
  if (::linkat(child.get(), "hard-source.tsv", child.get(),
               "hard-link.tsv", 0) != 0) {
    throw_system_error("could not create safe-reader hardlink fixture");
  }
  if (::symlinkat("ordinary/inner/bank.tsv", child.get(),
                  "leaf-link.tsv") != 0 ||
      ::symlinkat("ordinary", child.get(), "ordinary-link") != 0) {
    throw_system_error("could not create safe-reader symlink fixture");
  }
  if (::mkfifoat(child.get(), "fifo", 0444) != 0) {
    throw_system_error("could not create safe-reader FIFO fixture");
  }
  write_fixture_file(child.get(), "wrong-mode.tsv", fixture, 0644);
  write_fixture_file(child.get(), "wrong-size.tsv", fixture, 0444);
  write_fixture_file(child.get(), "wrong-hash.tsv", fixture, 0444);
  write_fixture_file(child.get(), "swap.tsv", fixture, 0444);
  write_fixture_file(child.get(), "swap-replacement.tsv", fixture, 0444);
  sync_directory(child.get(), "could not sync safe-reader self-test root");

  {
    SafeBankReader reader(duplicate_fd(child.get()));
    const SafeBankRead accepted = reader.read_bank(
        "ordinary/inner/bank.tsv", fixture_size, fixture_hash);
    const opening_bank::Bank reparsed = opening_bank::parse_bank_text(
        accepted.bytes, "synthetic-safe-reader-repeat");
    if (accepted.bytes != fixture || accepted.bank != synthetic ||
        reparsed != synthetic) {
      throw std::runtime_error(
          "safe-reader did not hash and parse identical bytes");
    }
    require_safe_reader_rejection(
        [&] {
          reader.read_bank("ordinary-link/inner/bank.tsv", fixture_size,
                           fixture_hash);
        },
        "intermediate symlink");
    require_safe_reader_rejection(
        [&] {
          reader.read_bank("leaf-link.tsv", fixture_size, fixture_hash);
        },
        "leaf symlink");
    require_safe_reader_rejection(
        [&] {
          reader.read_bank("hard-link.tsv", fixture_size, fixture_hash);
        },
        "hardlink");
    require_safe_reader_rejection(
        [&] { reader.read_bank("fifo", fixture_size, fixture_hash); },
        "FIFO");
    require_safe_reader_rejection(
        [&] {
          reader.read_bank("wrong-mode.tsv", fixture_size, fixture_hash);
        },
        "wrong mode");
    require_safe_reader_rejection(
        [&] {
          reader.read_bank("wrong-size.tsv", fixture_size + 1U,
                           fixture_hash);
        },
        "wrong size");
    require_safe_reader_rejection(
        [&] {
          reader.read_bank("wrong-hash.tsv", fixture_size,
                           std::string(64U, '0'));
        },
        "wrong hash");
    require_safe_reader_rejection(
        [&] {
          reader.read_bank("ordinary/inner/bank.tsv", fixture_size,
                           fixture_hash);
        },
        "second open");
    bool swap_completed = false;
    const SafeReadHook swap_path =
        [&](int parent, const std::string &basename) {
          if (basename != "swap.tsv" ||
              ::renameat(parent, basename.c_str(), parent,
                         "swap-old.tsv") != 0 ||
              ::renameat(parent, "swap-replacement.tsv", parent,
                         basename.c_str()) != 0) {
            throw_system_error("could not perform safe-reader path swap");
          }
          swap_completed = true;
        };
    require_safe_reader_rejection(
        [&] {
          reader.read_bank("swap.tsv", fixture_size, fixture_hash,
                           swap_path);
        },
        "path swap");
    if (!swap_completed) {
      throw std::runtime_error("safe-reader path-swap fixture did not run");
    }
  }

  unlink_fixture(child.get(), "ordinary-link");
  unlink_fixture(child.get(), "leaf-link.tsv");
  unlink_fixture(child.get(), "hard-link.tsv");
  unlink_fixture(child.get(), "hard-source.tsv");
  unlink_fixture(child.get(), "fifo");
  unlink_fixture(child.get(), "wrong-mode.tsv");
  unlink_fixture(child.get(), "wrong-size.tsv");
  unlink_fixture(child.get(), "wrong-hash.tsv");
  unlink_fixture(child.get(), "swap.tsv");
  unlink_fixture(child.get(), "swap-old.tsv");
  unlink_fixture(inner.get(), "bank.tsv");
  sync_directory(inner.get(), "could not sync safe-reader inner directory");
  inner.reset();
  remove_fixture_directory(ordinary.get(), "inner");
  sync_directory(ordinary.get(),
                 "could not sync safe-reader ordinary directory");
  ordinary.reset();
  remove_fixture_directory(child.get(), "ordinary");
  sync_directory(child.get(), "could not sync safe-reader self-test root");
  child.reset();
  remove_fixture_directory(temporary_parent, parent_chain.basename);
  sync_directory(temporary_parent,
                 "could not sync safe-reader self-test parent");
  std::cout << "safe_bank_reader_self_test=pass\n";
}
#endif

}  // namespace

int main(int argc, char **argv) {
  try {
#if defined(PAPERSOCCER_TT_EXACT_COLLISION_CAMPAIGN_TARGET)
    if (argc >= 2 &&
        std::string_view(argv[1]) == "--safe-bank-reader-self-test") {
      run_safe_bank_reader_self_test(argc, argv);
      return 0;
    }
#endif
    const Config config = parse_options(argc, argv);
    if (config.self_test) {
      run_self_test(config);
      return 0;
    }
    const std::vector<opening_bank::Bank> banks = load_banks(config);
    Summary overall;
    for (std::size_t bank_index = 0; bank_index < banks.size(); ++bank_index) {
      const opening_bank::Bank &bank = banks[bank_index];
      Summary bank_summary;
      for (const opening_bank::OpeningRecord &record : bank.records) {
        const ps::GameState opening = codingame_opening(record);
#if defined(PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING)
        const GameResult candidate_player0 = play(opening, 0, config);
        const GameResult candidate_player1 = play(opening, 1, config);
        add_opening_pair(bank_summary, candidate_player0, candidate_player1);
#else
        add_game(bank_summary, play(opening, 0, config), 0);
        add_game(bank_summary, play(opening, 1, config), 1);
#endif
      }
#if defined(PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING)
      validate_pair_accounting(bank_summary);
#endif
      print_summary("bank_summary", bank_index, bank_summary);
      merge_summary(overall, bank_summary);
    }
#if defined(PAPERSOCCER_HELDOUT_SWEEP_ACCOUNTING)
    validate_pair_accounting(overall);
#endif
    print_summary("summary", 0, overall);
    print_configuration(config);
    return overall.unfinished == 0 && overall.failed == 0 &&
                   overall.candidate.illegal_actions == 0 &&
                   overall.reference.illegal_actions == 0 &&
                   overall.candidate.operational_failures == 0 &&
                   overall.reference.operational_failures == 0
               ? 0
               : 2;
  } catch (const std::invalid_argument &error) {
    std::cerr << "rank_4_jacek_hybrid comparison gate: " << error.what()
              << '\n';
    return 64;
  } catch (const std::exception &error) {
    std::cerr << "rank_4_jacek_hybrid comparison gate: " << error.what()
              << '\n';
    return 70;
  }
}
