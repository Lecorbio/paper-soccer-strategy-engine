#include "opening_bank_internal.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "papersoccer/rules.hpp"

namespace papersoccer::opening_bank {
namespace {

constexpr std::size_t kMaximumOpeningDepth = 316;
constexpr std::size_t kMaximumPairCount = 1'000'000;
constexpr std::size_t kMaximumGenerationAttempts = 10'000'000;
constexpr std::string_view kHeader =
    "opening_id\tphase\tdepth\tgeneration_seed\tstate_hash\tcanonical_key\t"
    "to_move\tmoves";

constexpr std::array<std::string_view, 10> kMetadataKeys{{
    "phase",
    "depth",
    "pairs",
    "rules",
    "generator",
    "generator_seed",
    "selection",
    "state_hash_algorithm",
    "canonicalization",
    "opening_ply_definition",
}};

struct SplitMix64 {
  std::uint64_t state{};

  std::uint64_t next() noexcept {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }

  std::size_t unbiased_index(std::size_t upper_bound) noexcept {
    const auto bound = static_cast<std::uint64_t>(upper_bound);
    const std::uint64_t threshold = (std::uint64_t{0} - bound) % bound;
    std::uint64_t value = 0;
    do {
      value = next();
    } while (value < threshold);
    return static_cast<std::size_t>(value % bound);
  }
};

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
                                     unsigned int shift) noexcept {
  return (value >> shift) | (value << (32U - shift));
}

class Sha256 {
 public:
  void update(std::string_view bytes) {
    for (const unsigned char byte : bytes) {
      block_[block_size_++] = byte;
      ++total_bytes_;
      if (block_size_ == block_.size()) {
        transform();
        block_size_ = 0;
      }
    }
  }

  std::string finish() {
    const std::uint64_t bit_length = total_bytes_ * 8U;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56U) {
      while (block_size_ < block_.size()) {
        block_[block_size_++] = 0;
      }
      transform();
      block_size_ = 0;
    }
    while (block_size_ < 56U) {
      block_[block_size_++] = 0;
    }
    for (unsigned int shift = 56U;; shift -= 8U) {
      block_[block_size_++] =
          static_cast<std::uint8_t>((bit_length >> shift) & 0xffU);
      if (shift == 0U) {
        break;
      }
    }
    transform();
    block_size_ = 0;

    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::setfill('0');
    for (const std::uint32_t word : state_) {
      out << std::setw(8) << word;
    }
    return out.str();
  }

 private:
  std::array<std::uint32_t, 8> state_{{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  }};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_{};
  std::uint64_t total_bytes_{};

  void transform() noexcept {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t offset = index * 4U;
      words[index] =
          (static_cast<std::uint32_t>(block_[offset]) << 24U) |
          (static_cast<std::uint32_t>(block_[offset + 1U]) << 16U) |
          (static_cast<std::uint32_t>(block_[offset + 2U]) << 8U) |
          static_cast<std::uint32_t>(block_[offset + 3U]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const std::uint32_t s0 = rotate_right(words[index - 15U], 7U) ^
                               rotate_right(words[index - 15U], 18U) ^
                               (words[index - 15U] >> 3U);
      const std::uint32_t s1 = rotate_right(words[index - 2U], 17U) ^
                               rotate_right(words[index - 2U], 19U) ^
                               (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum_one = rotate_right(e, 6U) ^
                                    rotate_right(e, 11U) ^
                                    rotate_right(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary_one =
          h + sum_one + choose + kSha256RoundConstants[index] + words[index];
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

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }
};

std::string sha256(std::string_view bytes) {
  Sha256 hash;
  hash.update(bytes);
  return hash.finish();
}

bool supported_rules(const RulesConfig &rules) noexcept {
  return rules.width == 8 && rules.height == 10 &&
         rules.goal_rule == GoalRule::OpponentGoalOnly &&
         rules.blocked_rule == BlockedRule::PlayerToMoveLoses;
}

Point transform_point(Point point, bool reflect) noexcept {
  return reflect ? Point{8 - point.x, point.y} : point;
}

std::string player_name(Player player) {
  return player == Player::One ? "one" : "two";
}

std::string status_name(Status status) {
  switch (status) {
    case Status::InProgress:
      return "in_progress";
    case Status::WonByOne:
      return "won_by_one";
    case Status::WonByTwo:
      return "won_by_two";
  }
  throw std::invalid_argument("unknown game status");
}

std::string logical_state_serialization(const GameState &state, bool reflect) {
  if (!supported_rules(state.config)) {
    throw std::invalid_argument(
        "opening-bank state requires the standard 8x10 demo rules");
  }

  std::vector<Segment> segments;
  segments.reserve(state.used_segments.size());
  for (const Segment &segment : state.used_segments) {
    segments.emplace_back(transform_point(segment.a, reflect),
                          transform_point(segment.b, reflect));
  }
  std::sort(segments.begin(), segments.end(),
            [](const Segment &left, const Segment &right) {
              return left.a < right.a ||
                     (!(right.a < left.a) && left.b < right.b);
            });

  std::vector<std::pair<Point, int>> visits;
  visits.reserve(state.visit_count.size());
  for (const auto &[point, count] : state.visit_count) {
    if (count <= 0) {
      throw std::invalid_argument(
          "opening-bank state contains a non-positive visit count");
    }
    visits.emplace_back(transform_point(point, reflect), count);
  }
  std::sort(visits.begin(), visits.end(),
            [](const auto &left, const auto &right) {
              return left.first < right.first;
            });

  const Point ball = transform_point(state.ball, reflect);
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "papersoccer.logical-game-state.v1\n"
      << "rules=" << kRules << '\n'
      << "ball=" << ball.x << ',' << ball.y << '\n'
      << "to_move=" << player_name(state.to_move) << '\n'
      << "status=" << status_name(state.status) << '\n'
      << "segments=" << segments.size() << '\n';
  for (const Segment &segment : segments) {
    out << segment.a.x << ',' << segment.a.y << '-' << segment.b.x << ','
        << segment.b.y << '\n';
  }
  out << "visits=" << visits.size() << '\n';
  for (const auto &[point, count] : visits) {
    out << point.x << ',' << point.y << ':' << count << '\n';
  }
  return out.str();
}

bool contains_move(const std::vector<Move> &moves, Move move) {
  return std::find(moves.begin(), moves.end(), move) != moves.end();
}

template <typename UInt>
UInt parse_unsigned(std::string_view value, std::string_view field) {
  static_assert(std::is_unsigned_v<UInt>);
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument(std::string(field) +
                                " must be an unsigned decimal integer");
  }
  UInt parsed{};
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
    throw std::invalid_argument(std::string(field) +
                                " must be an unsigned decimal integer");
  }
  return parsed;
}

int parse_integer(std::string_view value, std::string_view field) {
  if (value.empty()) {
    throw std::invalid_argument(std::string(field) + " must be an integer");
  }
  int parsed{};
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
    throw std::invalid_argument(std::string(field) + " must be an integer");
  }
  return parsed;
}

std::vector<std::string_view> split(std::string_view value, char separator) {
  std::vector<std::string_view> result;
  std::size_t start = 0;
  while (true) {
    const std::size_t end = value.find(separator, start);
    if (end == std::string_view::npos) {
      result.push_back(value.substr(start));
      return result;
    }
    result.push_back(value.substr(start, end - start));
    start = end + 1U;
  }
}

std::vector<std::string_view> lines(std::string_view text) {
  std::vector<std::string_view> result;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = text.find('\n', start);
    const std::size_t length =
        end == std::string_view::npos ? text.size() - start : end - start;
    const std::string_view line = text.substr(start, length);
    if (!line.empty() && line.back() == '\r') {
      throw std::invalid_argument("opening bank must use LF line endings");
    }
    result.push_back(line);
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1U;
  }
  return result;
}

bool lowercase_sha256(std::string_view value) noexcept {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

Point parse_point(std::string_view text, std::string_view source_name,
                  std::size_t line_number) {
  const std::vector<std::string_view> coordinates = split(text, ',');
  if (coordinates.size() != 2U) {
    throw std::invalid_argument(std::string(source_name) + ':' +
                                std::to_string(line_number) +
                                ": invalid move endpoint");
  }
  return Point{parse_integer(coordinates[0], "move x"),
               parse_integer(coordinates[1], "move y")};
}

void validate_record(const OpeningRecord &record, const Bank &bank,
                     std::string_view context) {
  if (record.phase != bank.phase || record.depth != bank.depth ||
      record.moves.size() != bank.depth) {
    throw std::invalid_argument(std::string(context) +
                                ": opening metadata/transcript mismatch");
  }
  if (!lowercase_sha256(record.state_hash) ||
      !lowercase_sha256(record.canonical_key)) {
    throw std::invalid_argument(std::string(context) +
                                ": opening hash is not lowercase SHA-256");
  }
  const GameState replayed = replay_transcript(record.moves);
  if (replayed.path.size() != record.depth + 1U ||
      replayed.used_segments.size() != record.depth) {
    throw std::invalid_argument(std::string(context) +
                                ": opening ply accounting is inconsistent");
  }
  if (record.to_move != replayed.to_move ||
      record.state_hash != state_hash(replayed) ||
      record.canonical_key != canonical_key(replayed)) {
    throw std::invalid_argument(std::string(context) +
                                ": opening replay/hash mismatch");
  }
  if (record.opening_id !=
      stable_opening_id(record.phase, record.depth, record.state_hash)) {
    throw std::invalid_argument(std::string(context) +
                                ": opening ID is not stable for its state");
  }
}

void validate_bank(const Bank &bank, std::string_view context) {
  if (!valid_phase(bank.phase)) {
    throw std::invalid_argument(std::string(context) +
                                ": unsupported opening phase");
  }
  if (bank.depth == 0 || bank.depth > kMaximumOpeningDepth) {
    throw std::invalid_argument(std::string(context) +
                                ": opening depth is outside 1..316");
  }
  if (bank.pairs == 0 || bank.pairs > kMaximumPairCount ||
      bank.records.size() != bank.pairs) {
    throw std::invalid_argument(std::string(context) +
                                ": opening pair count is inconsistent");
  }

  std::unordered_set<std::string> identifiers;
  std::unordered_set<std::string> state_hashes;
  std::unordered_set<std::string> canonical_keys;
  for (std::size_t index = 0; index < bank.records.size(); ++index) {
    const OpeningRecord &record = bank.records[index];
    validate_record(record, bank,
                    std::string(context) + ": record " +
                        std::to_string(index));
    if (!identifiers.insert(record.opening_id).second) {
      throw std::invalid_argument(std::string(context) +
                                  ": duplicate opening ID");
    }
    if (!state_hashes.insert(record.state_hash).second) {
      throw std::invalid_argument(std::string(context) +
                                  ": duplicate opening state");
    }
    if (!canonical_keys.insert(record.canonical_key).second) {
      throw std::invalid_argument(std::string(context) +
                                  ": horizontally equivalent opening state");
    }
  }
}

std::string metadata_value(const Bank &bank, std::string_view key) {
  if (key == "phase") {
    return bank.phase;
  }
  if (key == "depth") {
    return std::to_string(bank.depth);
  }
  if (key == "pairs") {
    return std::to_string(bank.pairs);
  }
  if (key == "rules") {
    return std::string(kRules);
  }
  if (key == "generator") {
    return std::string(kGenerator);
  }
  if (key == "generator_seed") {
    return std::to_string(bank.generator_seed);
  }
  if (key == "selection") {
    return std::string(kSelection);
  }
  if (key == "state_hash_algorithm") {
    return std::string(kStateHashAlgorithm);
  }
  if (key == "canonicalization") {
    return std::string(kCanonicalization);
  }
  if (key == "opening_ply_definition") {
    return std::string(kOpeningPlyDefinition);
  }
  throw std::logic_error("unknown opening-bank metadata key");
}

}  // namespace

bool valid_phase(std::string_view phase) noexcept {
  return phase == "development" || phase == "validation" || phase == "test";
}

std::string state_hash(const GameState &state) {
  return sha256(logical_state_serialization(state, false));
}

std::string canonical_key(const GameState &state) {
  const std::string original = logical_state_serialization(state, false);
  const std::string reflected = logical_state_serialization(state, true);
  return sha256(std::min(original, reflected));
}

std::string stable_opening_id(std::string_view phase, std::size_t depth,
                              std::string_view hash) {
  if (!valid_phase(phase) || !lowercase_sha256(hash)) {
    throw std::invalid_argument("cannot construct opening ID from invalid fields");
  }
  return std::string(phase) + "-d" + std::to_string(depth) + '-' +
         std::string(hash);
}

GameState replay_transcript(const std::vector<Move> &moves) {
  GameState state = make_initial_state();
  for (std::size_t index = 0; index < moves.size(); ++index) {
    if (is_terminal(state)) {
      throw std::invalid_argument("opening transcript continues after terminal ply " +
                                  std::to_string(index));
    }
    const std::vector<Move> legal = legal_moves(state);
    if (!contains_move(legal, moves[index])) {
      throw std::invalid_argument("opening transcript has illegal ply " +
                                  std::to_string(index + 1U));
    }
    state = apply_move(state, moves[index]);
  }
  if (is_terminal(state)) {
    throw std::invalid_argument("opening transcript ends in a terminal state");
  }
  return state;
}

void validate_disjoint(const std::vector<Bank> &banks) {
  std::unordered_set<std::string> identifiers;
  std::unordered_set<std::string> state_hashes;
  std::unordered_set<std::string> canonical_keys;
  for (std::size_t bank_index = 0; bank_index < banks.size(); ++bank_index) {
    const Bank &bank = banks[bank_index];
    validate_bank(bank, "opening bank " + std::to_string(bank_index));
    for (const OpeningRecord &record : bank.records) {
      if (!identifiers.insert(record.opening_id).second) {
        throw std::invalid_argument("opening ID overlap across banks: " +
                                    record.opening_id);
      }
      if (!state_hashes.insert(record.state_hash).second) {
        throw std::invalid_argument("opening state overlap across banks: " +
                                    record.opening_id);
      }
      if (!canonical_keys.insert(record.canonical_key).second) {
        throw std::invalid_argument(
            "horizontally equivalent opening overlap across banks: " +
            record.opening_id);
      }
    }
  }
}

Bank generate_bank(std::string phase, std::size_t depth, std::size_t pairs,
                   std::uint64_t seed,
                   const std::vector<Bank> &excluded_banks) {
  if (!valid_phase(phase)) {
    throw std::invalid_argument(
        "opening phase must be development, validation, or test");
  }
  if (depth == 0 || depth > kMaximumOpeningDepth) {
    throw std::invalid_argument("opening depth must be between 1 and 316");
  }
  if (pairs == 0 || pairs > kMaximumPairCount) {
    throw std::invalid_argument(
        "opening pair count must be between 1 and 1000000");
  }
  validate_disjoint(excluded_banks);

  std::unordered_set<std::string> used_state_hashes;
  std::unordered_set<std::string> used_canonical_keys;
  for (const Bank &bank : excluded_banks) {
    for (const OpeningRecord &record : bank.records) {
      used_state_hashes.insert(record.state_hash);
      used_canonical_keys.insert(record.canonical_key);
    }
  }

  Bank bank{std::move(phase), depth, pairs, seed, {}};
  bank.records.reserve(pairs);
  SplitMix64 seed_stream{seed};
  for (std::size_t attempt = 0;
       attempt < kMaximumGenerationAttempts && bank.records.size() < pairs;
       ++attempt) {
    const std::uint64_t generation_seed = seed_stream.next();
    SplitMix64 generator{generation_seed};
    GameState state = make_initial_state();
    std::vector<Move> transcript;
    transcript.reserve(depth);
    for (std::size_t ply = 0; ply < depth && !is_terminal(state); ++ply) {
      const std::vector<Move> moves = legal_moves(state);
      if (moves.empty()) {
        break;
      }
      const Move selected = moves[generator.unbiased_index(moves.size())];
      transcript.push_back(selected);
      state = apply_move(state, selected);
    }
    if (transcript.size() != depth || is_terminal(state)) {
      continue;
    }

    const GameState replayed = replay_transcript(transcript);
    const std::string hash = state_hash(replayed);
    const std::string canonical = canonical_key(replayed);
    if (used_state_hashes.contains(hash) ||
        used_canonical_keys.contains(canonical)) {
      continue;
    }

    OpeningRecord record;
    record.opening_id = stable_opening_id(bank.phase, depth, hash);
    record.phase = bank.phase;
    record.depth = depth;
    record.generation_seed = generation_seed;
    record.state_hash = hash;
    record.canonical_key = canonical;
    record.to_move = replayed.to_move;
    record.moves = std::move(transcript);
    used_state_hashes.insert(record.state_hash);
    used_canonical_keys.insert(record.canonical_key);
    bank.records.push_back(std::move(record));
  }
  if (bank.records.size() != pairs) {
    throw std::runtime_error(
        "could not generate enough unique non-terminal opening states");
  }
  validate_bank(bank, "generated opening bank");
  return bank;
}

std::string render_bank(const Bank &bank) {
  validate_bank(bank, "rendered opening bank");
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "schema\t" << kSchema << '\n';
  for (const std::string_view key : kMetadataKeys) {
    out << key << '\t' << metadata_value(bank, key) << '\n';
  }
  out << kHeader << '\n';
  for (const OpeningRecord &record : bank.records) {
    out << record.opening_id << '\t' << record.phase << '\t' << record.depth
        << '\t' << record.generation_seed << '\t' << record.state_hash << '\t'
        << record.canonical_key << '\t' << player_name(record.to_move) << '\t';
    for (std::size_t index = 0; index < record.moves.size(); ++index) {
      if (index != 0) {
        out << ';';
      }
      out << record.moves[index].to.x << ',' << record.moves[index].to.y;
    }
    out << '\n';
  }
  return out.str();
}

Bank parse_bank_text(std::string_view text, std::string_view source_name) {
  const std::vector<std::string_view> input_lines = lines(text);
  const std::size_t fixed_lines = 1U + kMetadataKeys.size() + 1U;
  if (input_lines.size() < fixed_lines ||
      input_lines.front() != std::string("schema\t") + std::string(kSchema)) {
    throw std::invalid_argument(std::string(source_name) +
                                ": unsupported opening-bank schema");
  }
  for (std::size_t index = 0; index < input_lines.size(); ++index) {
    if (input_lines[index].empty()) {
      throw std::invalid_argument(std::string(source_name) + ':' +
                                  std::to_string(index + 1U) +
                                  ": blank line in opening bank");
    }
  }

  std::array<std::string, kMetadataKeys.size()> metadata{};
  for (std::size_t index = 0; index < kMetadataKeys.size(); ++index) {
    const std::vector<std::string_view> fields = split(input_lines[index + 1U], '\t');
    if (fields.size() != 2U || fields[0] != kMetadataKeys[index]) {
      throw std::invalid_argument(std::string(source_name) + ':' +
                                  std::to_string(index + 2U) +
                                  ": unexpected opening-bank metadata");
    }
    metadata[index] = fields[1];
  }
  if (input_lines[1U + kMetadataKeys.size()] != kHeader) {
    throw std::invalid_argument(std::string(source_name) +
                                ": opening-bank header mismatch");
  }

  Bank bank;
  bank.phase = metadata[0];
  bank.depth = parse_unsigned<std::size_t>(metadata[1], "depth");
  bank.pairs = parse_unsigned<std::size_t>(metadata[2], "pairs");
  if (metadata[3] != kRules || metadata[4] != kGenerator ||
      metadata[6] != kSelection || metadata[7] != kStateHashAlgorithm ||
      metadata[8] != kCanonicalization ||
      metadata[9] != kOpeningPlyDefinition) {
    throw std::invalid_argument(std::string(source_name) +
                                ": opening-bank contract metadata mismatch");
  }
  bank.generator_seed =
      parse_unsigned<std::uint64_t>(metadata[5], "generator_seed");

  const std::size_t records_start = fixed_lines;
  bank.records.reserve(input_lines.size() - records_start);
  for (std::size_t line_index = records_start;
       line_index < input_lines.size(); ++line_index) {
    const std::vector<std::string_view> fields = split(input_lines[line_index], '\t');
    if (fields.size() != 8U) {
      throw std::invalid_argument(std::string(source_name) + ':' +
                                  std::to_string(line_index + 1U) +
                                  ": opening record must have eight fields");
    }
    OpeningRecord record;
    record.opening_id = fields[0];
    record.phase = fields[1];
    record.depth = parse_unsigned<std::size_t>(fields[2], "record depth");
    record.generation_seed =
        parse_unsigned<std::uint64_t>(fields[3], "generation_seed");
    record.state_hash = fields[4];
    record.canonical_key = fields[5];
    if (fields[6] == "one") {
      record.to_move = Player::One;
    } else if (fields[6] == "two") {
      record.to_move = Player::Two;
    } else {
      throw std::invalid_argument(std::string(source_name) + ':' +
                                  std::to_string(line_index + 1U) +
                                  ": side to move must be one or two");
    }
    if (!fields[7].empty()) {
      for (const std::string_view endpoint : split(fields[7], ';')) {
        record.moves.push_back(
            Move{parse_point(endpoint, source_name, line_index + 1U)});
      }
    }
    bank.records.push_back(std::move(record));
  }
  validate_bank(bank, source_name);
  return bank;
}

Bank load_bank(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open opening bank: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input.eof() && input.fail()) {
    throw std::runtime_error("could not read opening bank: " + path.string());
  }
  return parse_bank_text(buffer.str(), path.string());
}

}  // namespace papersoccer::opening_bank
