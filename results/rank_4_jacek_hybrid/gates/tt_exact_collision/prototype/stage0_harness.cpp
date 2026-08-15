#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mcts_internal.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"
#include "src/opening_bank/opening_bank_internal.hpp"
#include "submissions/codingame/bots/rank_4_jacek_hybrid/replay_book.hpp"
#include "submissions/codingame/bots/rank_4_jacek_hybrid/replay_value_model.hpp"
#include "submissions/codingame/bots/rank_4_jacek_hybrid/teacher_residual_model.hpp"

// Stage0 is a test-only translation unit.  Exposing the two table members here
// avoids adding instrumentation to either frozen engine source.
#define private public
#if PAPERSOCCER_TT_EXACT_COLLISION_CANDIDATE
#include "results/rank_4_jacek_hybrid/gates/tt_exact_collision/prototype/candidate_bot.cpp"
#else
#include "submissions/codingame/bots/rank_4_jacek_hybrid/bot.cpp"
#endif
#undef private

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

constexpr std::string_view kSchema =
    "rank4-jacek-hybrid-tt-exact-collision-stage0-v1";
constexpr std::string_view kSeedHex = "0x4f1bbcdc676f2b31";
constexpr std::uint64_t kSeed = UINT64_C(0x4f1bbcdc676f2b31);
constexpr std::size_t kProceduralStates = 1000;
constexpr std::size_t kActivationStates = 64;
constexpr std::uint64_t kActivationNodes = 50000;
constexpr std::size_t kDepthOneStates = 64;
constexpr std::size_t kDepthTwoStates = 24;
constexpr std::uint64_t kOracleMaxNodes = 2000000;
constexpr std::uint8_t kProofMask = 7;
constexpr std::size_t kTimingRepetitions = 8;
constexpr std::array<std::string_view, 4> kTacticalTranscripts{{
    "6/1",
    "4/3/6/4/3/0",
    "0/6/5/4/5/53/61/0633",
    "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35",
}};

#if PAPERSOCCER_TT_EXACT_COLLISION_CANDIDATE
constexpr std::string_view kEngine = "candidate";
constexpr std::string_view kPolicy = "depth-primary-exact-secondary";
constexpr bool kExactSecondary = true;
#else
constexpr std::string_view kEngine = "control";
constexpr std::string_view kPolicy = "legacy-depth-only";
constexpr bool kExactSecondary = false;
#endif

using JsonFields = std::vector<std::pair<std::string, std::string>>;

std::string json_quote(std::string_view value) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() + 2);
  result.push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\b': result += "\\b"; break;
      case '\f': result += "\\f"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (byte < 0x20U || byte >= 0x80U) {
          result += "\\u00";
          result.push_back(kHex[byte >> 4U]);
          result.push_back(kHex[byte & 0x0fU]);
        } else {
          result.push_back(static_cast<char>(byte));
        }
    }
  }
  result.push_back('"');
  return result;
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

template <typename Integer>
std::string json_integer(Integer value) {
  return std::to_string(value);
}

std::string json_object(JsonFields fields) {
  std::sort(fields.begin(), fields.end(),
            [](const auto &left, const auto &right) {
              return left.first < right.first;
            });
  std::string result{"{"};
  for (std::size_t index = 0; index < fields.size(); ++index) {
    if (index != 0) result.push_back(',');
    result += json_quote(fields[index].first);
    result.push_back(':');
    result += fields[index].second;
  }
  result.push_back('}');
  return result;
}

std::string json_array(const std::vector<std::string> &values) {
  std::string result{"["};
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) result.push_back(',');
    result += values[index];
  }
  result.push_back(']');
  return result;
}

class Sha256 {
 public:
  Sha256() { reset(); }

  void update(std::string_view bytes) {
    for (const unsigned char byte : bytes) {
      block_[block_size_++] = byte;
      bit_count_ += 8;
      if (block_size_ == block_.size()) {
        transform();
        block_size_ = 0;
      }
    }
  }

  std::string finish_hex() {
    const std::uint64_t original_bits = bit_count_;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56U) {
      while (block_size_ < block_.size()) block_[block_size_++] = 0;
      transform();
      block_size_ = 0;
    }
    while (block_size_ < 56U) block_[block_size_++] = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
      block_[block_size_++] =
          static_cast<std::uint8_t>(original_bits >> shift);
    }
    transform();

    static constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const std::uint32_t word : state_) {
      for (int shift = 28; shift >= 0; shift -= 4) {
        result.push_back(kHex[(word >> shift) & 0x0fU]);
      }
    }
    return result;
  }

 private:
  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_{};
  std::uint64_t bit_count_{};

  static constexpr std::uint32_t rotate_right(std::uint32_t value,
                                               unsigned bits) {
    return (value >> bits) | (value << (32U - bits));
  }

  void reset() {
    state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    block_.fill(0);
    block_size_ = 0;
    bit_count_ = 0;
  }

  void transform() {
    static constexpr std::array<std::uint32_t, 64> kRound{{
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
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t offset = index * 4U;
      words[index] = (static_cast<std::uint32_t>(block_[offset]) << 24U) |
                     (static_cast<std::uint32_t>(block_[offset + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(block_[offset + 2U]) << 8U) |
                     static_cast<std::uint32_t>(block_[offset + 3U]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const std::uint32_t s0 =
          rotate_right(words[index - 15U], 7U) ^
          rotate_right(words[index - 15U], 18U) ^
          (words[index - 15U] >> 3U);
      const std::uint32_t s1 =
          rotate_right(words[index - 2U], 17U) ^
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
      const std::uint32_t sigma1 = rotate_right(e, 6U) ^
                                   rotate_right(e, 11U) ^
                                   rotate_right(e, 25U);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t temp1 =
          h + sigma1 + choose + kRound[index] + words[index];
      const std::uint32_t sigma0 = rotate_right(a, 2U) ^
                                   rotate_right(a, 13U) ^
                                   rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
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

std::string sha256(std::string_view value) {
  Sha256 hash;
  hash.update(value);
  return hash.finish_hex();
}

std::string hex16(std::uint64_t value) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result(16, '0');
  for (int index = 15; index >= 0; --index) {
    result[static_cast<std::size_t>(index)] = kHex[value & 0x0fU];
    value >>= 4U;
  }
  return result;
}

std::string bound_name(cg::ScoreBound bound) {
  switch (bound) {
    case cg::ScoreBound::Exact: return "Exact";
    case cg::ScoreBound::Lower: return "Lower";
    case cg::ScoreBound::Upper: return "Upper";
  }
  throw std::logic_error("unreachable score bound");
}

std::string logical_entry_json(const cg::TranspositionEntry &entry,
                               bool include_occupied = true) {
  std::string result{"{"};
  result += "\"bound\":" + json_quote(bound_name(entry.bound));
  result += ",\"depth\":" + json_integer(entry.depth);
  result += ",\"key_high\":" + json_quote(hex16(entry.key.first));
  result += ",\"key_low\":" + json_quote(hex16(entry.key.second));
  result += ",\"move_x\":" + json_integer(entry.best_move.to.x);
  result += ",\"move_y\":" + json_integer(entry.best_move.to.y);
  if (include_occupied) {
    result += ",\"occupied\":" + json_bool(entry.occupied);
  }
  result += ",\"score\":" + json_integer(entry.score) + "}";
  return result;
}

bool same_entry(const cg::TranspositionEntry &left,
                const cg::TranspositionEntry &right) {
  return left.key == right.key && left.depth == right.depth &&
         left.score == right.score && left.best_move == right.best_move &&
         left.bound == right.bound && left.occupied == right.occupied;
}

std::string entry_pair_json(
    const std::array<cg::TranspositionEntry, 2> &entries) {
  return "[" + logical_entry_json(entries[0]) + "," +
         logical_entry_json(entries[1]) + "]";
}

ps::RulesConfig contest_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
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

ps::Point rotate_point(const ps::RulesConfig &rules, ps::Point point) {
  return {rules.width - point.x, rules.height + 2 - point.y};
}

ps::GameState rotate_state(const ps::GameState &source) {
  ps::GameState result;
  result.config = source.config;
  result.ball = rotate_point(source.config, source.ball);
  result.to_move = ps::opponent(source.to_move);
  switch (source.status) {
    case ps::Status::InProgress: result.status = ps::Status::InProgress; break;
    case ps::Status::WonByOne: result.status = ps::Status::WonByTwo; break;
    case ps::Status::WonByTwo: result.status = ps::Status::WonByOne; break;
  }
  result.path.reserve(source.path.size());
  for (const ps::Point point : source.path) {
    result.path.push_back(rotate_point(source.config, point));
  }
  result.used_segments.reserve(source.used_segments.size());
  for (const ps::Segment &segment : source.used_segments) {
    result.used_segments.insert(
        ps::Segment{rotate_point(source.config, segment.a),
                    rotate_point(source.config, segment.b)});
  }
  result.visit_count.reserve(source.visit_count.size());
  for (const auto &[point, visits] : source.visit_count) {
    result.visit_count.emplace(rotate_point(source.config, point), visits);
  }
  return result;
}

std::string contest_state_identity(const ps::GameState &state) {
  if (state.config.width != 8 || state.config.height != 10 ||
      state.config.goal_rule != ps::GoalRule::OwnGoalsAllowed ||
      state.config.blocked_rule != ps::BlockedRule::MoverLoses) {
    throw std::invalid_argument("Stage0 state does not use contest rules");
  }
  ps::GameState normalized = state;
  normalized.config = ps::RulesConfig{
      8, 10, ps::GoalRule::OpponentGoalOnly,
      ps::BlockedRule::PlayerToMoveLoses};
  return ps::opening_bank::state_hash(normalized);
}

std::uint64_t next_random(std::uint64_t &state) {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(contest_rules());
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    if (end == begin) throw std::invalid_argument("empty transcript turn");
    cg::apply_encoded_turn(state, transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) break;
    begin = separator + 1U;
  }
  return state;
}

std::vector<ps::GameState> make_procedural_states(std::size_t count) {
  std::vector<ps::GameState> states;
  states.reserve(count);
  std::uint64_t random_state = kSeed;
  ps::GameState state = ps::make_initial_state(contest_rules());
  while (states.size() < count) {
    if (ps::is_terminal(state)) {
      state = ps::make_initial_state(contest_rules());
    }
    states.push_back(state);
    const std::vector<ps::Move> legal = ps::legal_moves(state);
    if (legal.empty()) throw std::logic_error("live corpus state has no move");
    const std::size_t selected = static_cast<std::size_t>(
        next_random(random_state) % static_cast<std::uint64_t>(legal.size()));
    state = ps::apply_move(state, legal[selected]);
  }
  return states;
}

std::vector<ps::GameState> make_corpus() {
  std::vector<ps::GameState> corpus =
      make_procedural_states(kProceduralStates);
  corpus.reserve(1004);
  for (const std::string_view transcript : kTacticalTranscripts) {
    corpus.push_back(reconstruct(transcript));
  }
  return corpus;
}

std::string corpus_digest(const std::vector<ps::GameState> &corpus) {
  Sha256 hash;
  hash.update("[");
  for (std::size_t index = 0; index < corpus.size(); ++index) {
    if (index != 0) hash.update(",");
    const ps::GameState rotated = rotate_state(corpus[index]);
    hash.update(json_object({
        {"base_state_sha256",
         json_quote(contest_state_identity(corpus[index]))},
        {"rotated_state_sha256",
         json_quote(contest_state_identity(rotated))},
        {"state_index", json_integer(index)},
    }));
  }
  hash.update("]\n");
  return hash.finish_hex();
}

bool action_is_legal(const ps::GameState &initial,
                     const std::vector<ps::Move> &action,
                     std::string &encoded, ps::GameState &post) {
  encoded.clear();
  post = initial;
  if (action.empty()) return false;
  const ps::Player mover = post.to_move;
  for (const ps::Move move : action) {
    if (ps::is_terminal(post) || post.to_move != mover) return false;
    const std::vector<ps::Move> legal = ps::legal_moves(post);
    if (std::find(legal.begin(), legal.end(), move) == legal.end()) {
      return false;
    }
    encoded.push_back(cg::encode_direction(post.ball, move.to));
    post = ps::apply_move(post, move);
  }
  return ps::is_terminal(post) || post.to_move != mover;
}

std::string logical_table_digest(const cg::TranspositionTable &table) {
  static const std::string kEmpty =
      logical_entry_json(cg::TranspositionEntry{});
  Sha256 hash;
  hash.update("[");
  for (std::size_t index = 0; index < table.entries_.size(); ++index) {
    if (index != 0) hash.update(",");
    const cg::TranspositionEntry &entry = table.entries_[index];
    hash.update(entry.occupied ? logical_entry_json(entry) : kEmpty);
  }
  hash.update("]\n");
  return hash.finish_hex();
}

struct SearchObservation {
  std::string action_ascii{};
  bool budget_exhausted{};
  std::uint32_t completed_depth{};
  bool exception{};
  bool legal{};
  std::string logical_table_sha256{};
  std::uint64_t nodes{};
  std::string post_state_sha256{};
  std::uint64_t root_proof_hits{};
  std::uint64_t root_proof_probes{};
  bool root_proof_shortcut{};
  int root_score{};
  std::uint32_t attempted_depth{};
};

SearchObservation run_bounded_search(const ps::GameState &state,
                                     std::uint64_t max_nodes) {
  SearchObservation result;
  ps::GameState post = state;
  std::unique_ptr<cg::CompleteTurnSearch> search;
  try {
    cg::SearchConfig config;
    config.max_nodes = max_nodes;
    config.transposition_entries = cg::kTranspositionEntries;
    config.evaluation_entries = cg::kEvaluationEntries;
    config.exact_proof_mask = kProofMask;
    config.replay_value_blend_percent = 15;
    config.teacher_residual_weight_percent = 100;
    search = std::make_unique<cg::CompleteTurnSearch>(state, config);
    const std::vector<ps::Move> action = search->run();
    result.legal = action_is_legal(state, action, result.action_ascii, post);
  } catch (...) {
    result.exception = true;
  }
  if (search) {
    const cg::SearchStats &stats = search->stats();
    result.budget_exhausted = stats.budget_exhausted;
    result.completed_depth = stats.completed_turn_depth;
    result.nodes = stats.nodes;
    result.root_proof_hits =
        stats.root_rebound_win_hits + stats.root_rebound_loss_hits;
    result.root_proof_probes = stats.root_rebound_probes;
    result.root_proof_shortcut = stats.nodes == 0;
    result.root_score = stats.root_score;
    result.attempted_depth = stats.attempted_turn_depth;
    result.logical_table_sha256 = logical_table_digest(search->table_);
  } else {
    cg::TranspositionTable empty(0);
    result.logical_table_sha256 = logical_table_digest(empty);
  }
  result.post_state_sha256 = contest_state_identity(post);
  return result;
}

std::string activation_result_json(const SearchObservation &result) {
  return json_object({
      {"action_ascii", json_quote(result.action_ascii)},
      {"attempted_depth", json_integer(result.attempted_depth)},
      {"budget_exhausted", json_bool(result.budget_exhausted)},
      {"completed_depth", json_integer(result.completed_depth)},
      {"exception", json_bool(result.exception)},
      {"legal", json_bool(result.legal)},
      {"logical_table_sha256", json_quote(result.logical_table_sha256)},
      {"nodes", json_integer(result.nodes)},
      {"post_state_sha256", json_quote(result.post_state_sha256)},
      {"root_proof_hits", json_integer(result.root_proof_hits)},
      {"root_proof_probes", json_integer(result.root_proof_probes)},
      {"root_proof_shortcut", json_bool(result.root_proof_shortcut)},
      {"root_score", json_integer(result.root_score)},
  });
}

struct FixedDepthObservation {
  std::string action_ascii{};
  int action_value{};
  bool completed{};
  bool exception{};
  bool legal{};
  std::uint64_t nodes{};
  std::string post_state_sha256{};
  int root_score{};
  bool state_unchanged{};
};

FixedDepthObservation run_fixed_depth(const ps::GameState &state,
                                      std::uint32_t depth, bool use_tt) {
  FixedDepthObservation result;
  const ps::GameState before = state;
  ps::GameState post = state;
  try {
    cg::SearchConfig config;
    config.max_turn_depth = depth;
    config.max_nodes = kOracleMaxNodes;
    config.transposition_entries = use_tt ? cg::kTranspositionEntries : 0;
    config.evaluation_entries = cg::kEvaluationEntries;
    config.exact_proof_mask = kProofMask;
    config.replay_value_blend_percent = 15;
    config.teacher_residual_weight_percent = 100;
    cg::CompleteTurnSearch search(state, config);
    const cg::RootResult root = search.run_fixed_depth_for_test(depth);
    result.nodes = search.stats().nodes;
    result.root_score = root.score;
    result.action_value = root.score;
    result.legal =
        action_is_legal(state, root.action, result.action_ascii, post);
    result.completed = search.stats().completed_turn_depth == depth;
  } catch (...) {
    result.exception = true;
  }
  result.post_state_sha256 = contest_state_identity(post);
  result.state_unchanged = same_state(state, before);
  return result;
}

std::string fixed_depth_json(const FixedDepthObservation &result) {
  return json_object({
      {"action_ascii", json_quote(result.action_ascii)},
      {"action_value", json_integer(result.action_value)},
      {"completed", json_bool(result.completed)},
      {"exception", json_bool(result.exception)},
      {"legal", json_bool(result.legal)},
      {"nodes", json_integer(result.nodes)},
      {"post_state_sha256", json_quote(result.post_state_sha256)},
      {"root_score", json_integer(result.root_score)},
      {"state_unchanged", json_bool(result.state_unchanged)},
  });
}

cg::ScoreBound bound_from_index(std::size_t index) {
  switch (index) {
    case 0: return cg::ScoreBound::Exact;
    case 1: return cg::ScoreBound::Lower;
    case 2: return cg::ScoreBound::Upper;
    default: throw std::logic_error("invalid bound index");
  }
}

constexpr std::array<ps::detail::PositionKey, 3> kFixtureKeys{{
    {UINT64_C(0x1111111111111111), UINT64_C(0x2222222222222222)},
    {UINT64_C(0x3333333333333333), UINT64_C(0x4444444444444444)},
    {UINT64_C(0x5555555555555555), UINT64_C(0x6666666666666666)},
}};

cg::TranspositionEntry resident_entry(std::size_t way,
                                      std::size_t resident_index) {
  if (resident_index == 0) return {};
  const std::size_t ordinal = resident_index - 1U;
  const std::size_t key_index = ordinal / 12U;
  const std::size_t within_key = ordinal % 12U;
  const std::uint32_t depth = static_cast<std::uint32_t>(within_key / 3U);
  const std::size_t bound_index = within_key % 3U;
  return cg::TranspositionEntry{
      kFixtureKeys[key_index],
      depth,
      static_cast<int>(100000 + 10000 * way + 1000 * key_index +
                       100 * depth + 10 * bound_index),
      ps::Move{{static_cast<int>(10 * way + 3 * key_index + bound_index),
                static_cast<int>(10 * depth + bound_index)}},
      bound_from_index(bound_index),
      true,
  };
}

cg::TranspositionEntry incoming_entry(std::uint32_t depth,
                                      std::size_t bound_index) {
  return cg::TranspositionEntry{
      kFixtureKeys[0],
      depth,
      static_cast<int>(-200000 - 100 * depth - 10 * bound_index),
      ps::Move{{-10 - static_cast<int>(depth),
                10 + static_cast<int>(bound_index)}},
      bound_from_index(bound_index),
      true,
  };
}

struct ExpectedStore {
  bool returned{};
  std::optional<std::size_t> victim{};
  std::array<cg::TranspositionEntry, 2> after{};
};

ExpectedStore simulate_store(
    const std::array<cg::TranspositionEntry, 2> &before,
    const cg::TranspositionEntry &incoming) {
  ExpectedStore result;
  result.after = before;
  std::optional<std::size_t> victim;
  for (std::size_t way = 0; way < 2; ++way) {
    const cg::TranspositionEntry &entry = before[way];
    if (entry.occupied && entry.key == incoming.key) {
      if (entry.depth > incoming.depth) return result;
      victim = way;
      break;
    }
    if (!entry.occupied || !victim.has_value() ||
        entry.depth < before[*victim].depth ||
        (kExactSecondary && before[*victim].occupied &&
         entry.depth == before[*victim].depth &&
         entry.bound != cg::ScoreBound::Exact &&
         before[*victim].bound == cg::ScoreBound::Exact)) {
      victim = way;
    }
  }
  if (!victim.has_value()) return result;
  const cg::TranspositionEntry &selected = before[*victim];
  if (selected.occupied &&
      (selected.depth > incoming.depth ||
       (kExactSecondary && selected.key != incoming.key &&
        selected.depth == incoming.depth &&
        selected.bound == cg::ScoreBound::Exact &&
        incoming.bound != cg::ScoreBound::Exact))) {
    return result;
  }
  result.returned = true;
  result.victim = victim;
  result.after[*victim] = incoming;
  return result;
}

std::optional<std::size_t> changed_way(
    const std::array<cg::TranspositionEntry, 2> &before,
    const std::array<cg::TranspositionEntry, 2> &after, bool returned) {
  if (!returned) return std::nullopt;
  std::optional<std::size_t> changed;
  for (std::size_t way = 0; way < 2; ++way) {
    if (!same_entry(before[way], after[way])) {
      if (changed.has_value()) return std::nullopt;
      changed = way;
    }
  }
  return changed;
}

std::string optional_way_json(std::optional<std::size_t> way) {
  return way.has_value() ? json_integer(*way) : "null";
}

struct FindResult {
  std::optional<std::size_t> way{};
  cg::TranspositionEntry entry{};
};

FindResult find_in_entries(
    const std::array<cg::TranspositionEntry, 2> &entries,
    ps::detail::PositionKey key) {
  for (std::size_t way = 0; way < 2; ++way) {
    if (entries[way].occupied && entries[way].key == key) {
      return FindResult{way, entries[way]};
    }
  }
  return {};
}

FindResult find_in_table(const cg::TranspositionTable &table,
                         ps::detail::PositionKey key) {
  const cg::TranspositionEntry *entry = table.find(key);
  if (entry == nullptr) return {};
  const auto distance = entry - table.entries_.data();
  if (distance < 0 || distance > 1) {
    throw std::logic_error("capacity-two find escaped its bucket");
  }
  return FindResult{static_cast<std::size_t>(distance), *entry};
}

std::string find_json(const FindResult &found) {
  if (!found.way.has_value()) return "null";
  return json_object({
      {"entry", logical_entry_json(found.entry)},
      {"way", json_integer(*found.way)},
  });
}

bool same_find(const FindResult &left, const FindResult &right) {
  return left.way == right.way &&
         (!left.way.has_value() || same_entry(left.entry, right.entry));
}

std::size_t table_case_failures(std::string &digest) {
  Sha256 hash;
  hash.update("[");
  std::size_t failures = 0;
  std::size_t case_index = 0;
  for (std::size_t resident0 = 0; resident0 < 37; ++resident0) {
    for (std::size_t resident1 = 0; resident1 < 37; ++resident1) {
      for (std::uint32_t incoming_depth = 0; incoming_depth < 4;
           ++incoming_depth) {
        for (std::size_t incoming_bound = 0; incoming_bound < 3;
             ++incoming_bound, ++case_index) {
          const std::array<cg::TranspositionEntry, 2> before{{
              resident_entry(0, resident0), resident_entry(1, resident1)}};
          const cg::TranspositionEntry incoming =
              incoming_entry(incoming_depth, incoming_bound);
          const ExpectedStore expected = simulate_store(before, incoming);

          cg::TranspositionTable table(2);
          table.entries_[0] = before[0];
          table.entries_[1] = before[1];
          const bool actual_return =
              table.store(incoming.key, incoming.depth, incoming.score,
                          incoming.best_move, incoming.bound);
          const std::array<cg::TranspositionEntry, 2> actual_after{{
              table.entries_[0], table.entries_[1]}};
          const std::optional<std::size_t> actual_victim =
              changed_way(before, actual_after, actual_return);
          const bool nonvictim_unchanged = actual_victim.has_value()
              ? same_entry(before[1U - *actual_victim],
                           actual_after[1U - *actual_victim])
              : same_entry(before[0], actual_after[0]) &&
                    same_entry(before[1], actual_after[1]);

          const FindResult actual_a = find_in_table(table, kFixtureKeys[1]);
          const FindResult actual_b = find_in_table(table, kFixtureKeys[2]);
          const FindResult actual_incoming =
              find_in_table(table, kFixtureKeys[0]);
          const FindResult expected_a =
              find_in_entries(expected.after, kFixtureKeys[1]);
          const FindResult expected_b =
              find_in_entries(expected.after, kFixtureKeys[2]);
          const FindResult expected_incoming =
              find_in_entries(expected.after, kFixtureKeys[0]);

          bool case_ok = actual_return == expected.returned &&
                         actual_victim == expected.victim &&
                         nonvictim_unchanged;
          for (std::size_t way = 0; way < 2; ++way) {
            case_ok = case_ok &&
                      same_entry(actual_after[way], expected.after[way]);
          }
          case_ok = case_ok && same_find(actual_a, expected_a) &&
                    same_find(actual_b, expected_b) &&
                    same_find(actual_incoming, expected_incoming);
          if (!case_ok) ++failures;

          const std::string find_results = json_object({
              {"collision_a", find_json(actual_a)},
              {"collision_b", find_json(actual_b)},
              {"incoming", find_json(actual_incoming)},
          });
          const std::string record = json_object({
              {"actual_after", entry_pair_json(actual_after)},
              {"actual_return", json_bool(actual_return)},
              {"actual_victim", optional_way_json(actual_victim)},
              {"before", entry_pair_json(before)},
              {"case_index", json_integer(case_index)},
              {"expected_after", entry_pair_json(expected.after)},
              {"expected_return", json_bool(expected.returned)},
              {"expected_victim", optional_way_json(expected.victim)},
              {"find_results", find_results},
              {"incoming", logical_entry_json(incoming, false)},
              {"nonvictim_unchanged", json_bool(nonvictim_unchanged)},
          });
          if (case_index != 0) hash.update(",");
          hash.update(record);
        }
      }
    }
  }
  if (case_index != 16428) throw std::logic_error("case count drift");
  hash.update("]\n");
  digest = hash.finish_hex();
  return failures;
}

std::optional<std::size_t> expected_bucket(std::size_t capacity,
                                           ps::detail::PositionKey key) {
  if (capacity < 2) return std::nullopt;
  const std::uint64_t combined =
      key.first ^ ((key.second << 23U) | (key.second >> 41U));
  const std::size_t buckets = capacity / 2U;
  return 2U * static_cast<std::size_t>(combined % buckets);
}

std::size_t routing_failures(std::string &digest) {
  constexpr std::array<std::size_t, 7> kCapacities{{
      0, 1, 2, 4, 6, 4096, 262144,
  }};
  constexpr std::array<ps::detail::PositionKey, 3> kKeys{{
      {0, 0}, {1, 0}, {UINT64_MAX, UINT64_MAX},
  }};
  Sha256 hash;
  hash.update("[");
  std::size_t failures = 0;
  std::size_t count = 0;
  for (const std::size_t capacity : kCapacities) {
    for (const ps::detail::PositionKey key : kKeys) {
      for (std::uint32_t depth = 0; depth < 4; ++depth) {
        for (std::size_t bound_index = 0; bound_index < 3;
             ++bound_index, ++count) {
          cg::TranspositionTable table(capacity);
          const std::optional<std::size_t> expected =
              expected_bucket(capacity, key);
          std::optional<std::size_t> actual;
          if (capacity >= 2) actual = table.bucket_index(key);
          const cg::ScoreBound bound = bound_from_index(bound_index);
          const bool actual_store = table.store(
              key, depth, -200000 - static_cast<int>(100 * depth) -
                              static_cast<int>(10 * bound_index),
              ps::Move{{-10 - static_cast<int>(depth),
                        10 + static_cast<int>(bound_index)}},
              bound);
          const bool actual_find = table.find(key) != nullptr;
          const bool expected_store = capacity >= 2;
          const bool expected_find = expected_store;
          if (actual != expected || actual_store != expected_store ||
              actual_find != expected_find) {
            ++failures;
          }
          const std::string record = json_object({
              {"actual_bucket", optional_way_json(actual)},
              {"actual_find", json_bool(actual_find)},
              {"actual_store", json_bool(actual_store)},
              {"bound", json_quote(bound_name(bound))},
              {"capacity", json_integer(capacity)},
              {"depth", json_integer(depth)},
              {"expected_bucket", optional_way_json(expected)},
              {"expected_find", json_bool(expected_find)},
              {"expected_store", json_bool(expected_store)},
              {"key_high", json_quote(hex16(key.first))},
              {"key_low", json_quote(hex16(key.second))},
          });
          if (count != 0) hash.update(",");
          hash.update(record);
        }
      }
    }
  }
  if (count != 252) throw std::logic_error("routing count drift");
  hash.update("]\n");
  digest = hash.finish_hex();
  return failures;
}

int table_truth_mode(std::string_view expected_policy) {
  if (expected_policy != kPolicy) return 2;
  std::string case_digest;
  std::string routing_digest;
  const std::size_t failures =
      table_case_failures(case_digest) + routing_failures(routing_digest);
  const std::vector<std::string> capacities{
      "0", "1", "2", "4", "6", "4096", "262144"};
  const std::string output = json_object({
      {"capacities", json_array(capacities)},
      {"case_digest_sha256", json_quote(case_digest)},
      {"cases", "16428"},
      {"engine", json_quote(kEngine)},
      {"failure_count", json_integer(failures)},
      {"mode", json_quote("table-truth")},
      {"passed", json_bool(failures == 0)},
      {"policy", json_quote(kPolicy)},
      {"routing_cases", "252"},
      {"routing_digest_sha256", json_quote(routing_digest)},
      {"schema", json_quote(kSchema)},
  });
  std::cout << output << '\n';
  return 0;
}

int public_screen_mode() {
  const std::vector<ps::GameState> corpus = make_corpus();
  if (corpus.size() != 1004) return 2;
  std::vector<std::string> activation_records;
  activation_records.reserve(kActivationStates);
  for (std::size_t index = 0; index < kActivationStates; ++index) {
    const ps::GameState rotated = rotate_state(corpus[index]);
    const std::string base_hash = contest_state_identity(corpus[index]);
    const std::string rotated_hash = contest_state_identity(rotated);
    const SearchObservation base =
        run_bounded_search(corpus[index], kActivationNodes);
    const SearchObservation rotated_result =
        run_bounded_search(rotated, kActivationNodes);
    activation_records.push_back(json_object({
        {"base", activation_result_json(base)},
        {"base_state_sha256", json_quote(base_hash)},
        {"rotated", activation_result_json(rotated_result)},
        {"rotated_state_sha256", json_quote(rotated_hash)},
        {"state_index", json_integer(index)},
    }));
  }

  std::vector<std::string> oracle_records;
  oracle_records.reserve(kDepthOneStates + kDepthTwoStates);
  for (std::uint32_t depth = 1; depth <= 2; ++depth) {
    const std::size_t count =
        depth == 1 ? kDepthOneStates : kDepthTwoStates;
    for (std::size_t index = 0; index < count; ++index) {
      const ps::GameState rotated = rotate_state(corpus[index]);
      const FixedDepthObservation base_tt =
          run_fixed_depth(corpus[index], depth, true);
      const FixedDepthObservation base_no_tt =
          run_fixed_depth(corpus[index], depth, false);
      const FixedDepthObservation rotated_tt =
          run_fixed_depth(rotated, depth, true);
      const FixedDepthObservation rotated_no_tt =
          run_fixed_depth(rotated, depth, false);
      oracle_records.push_back(json_object({
          {"base", json_object({
               {"no_tt", fixed_depth_json(base_no_tt)},
               {"tt", fixed_depth_json(base_tt)},
           })},
          {"base_state_sha256",
           json_quote(contest_state_identity(corpus[index]))},
          {"depth", json_integer(depth)},
          {"rotated", json_object({
               {"no_tt", fixed_depth_json(rotated_no_tt)},
               {"tt", fixed_depth_json(rotated_tt)},
           })},
          {"rotated_state_sha256",
           json_quote(contest_state_identity(rotated))},
          {"state_index", json_integer(index)},
      }));
    }
  }

  const std::string output = json_object({
      {"activation_nodes", json_integer(kActivationNodes)},
      {"activation_records", json_array(activation_records)},
      {"activation_states", json_integer(kActivationStates)},
      {"corpus_sha256", json_quote(corpus_digest(corpus))},
      {"depth1_states", json_integer(kDepthOneStates)},
      {"depth2_states", json_integer(kDepthTwoStates)},
      {"engine", json_quote(kEngine)},
      {"exact_proof_mask", json_integer(kProofMask)},
      {"mode", json_quote("public-screen")},
      {"oracle_max_nodes", json_integer(kOracleMaxNodes)},
      {"oracle_records", json_array(oracle_records)},
      {"policy", json_quote(kPolicy)},
      {"procedural_states", json_integer(kProceduralStates)},
      {"schema", json_quote(kSchema)},
      {"seed_hex", json_quote(kSeedHex)},
  });
  std::cout << output << '\n';
  return 0;
}

std::optional<std::uint64_t> parse_decimal(std::string_view text) {
  if (text.empty() || (text.size() > 1 && text.front() == '0')) {
    return std::nullopt;
  }
  std::uint64_t value = 0;
  for (const char character : text) {
    if (character < '0' || character > '9') return std::nullopt;
    const std::uint64_t digit =
        static_cast<std::uint64_t>(character - '0');
    if (value > (UINT64_MAX - digit) / 10U) return std::nullopt;
    value = value * 10U + digit;
  }
  return value;
}

std::vector<ps::GameState> panel_states(std::string_view panel) {
  std::vector<ps::GameState> states;
  if (panel == "forced-prod") {
    states.reserve(8);
    for (const std::string_view transcript : kTacticalTranscripts) {
      ps::GameState state = reconstruct(transcript);
      states.push_back(state);
      states.push_back(rotate_state(state));
    }
  } else if (panel == "mixed-prod") {
    states.reserve(64);
    const std::vector<ps::GameState> procedural =
        make_procedural_states(32);
    for (const ps::GameState &state : procedural) {
      states.push_back(state);
      states.push_back(rotate_state(state));
    }
  }
  return states;
}

std::string timing_result_json(const SearchObservation &result) {
  return json_object({
      {"action_ascii", json_quote(result.action_ascii)},
      {"attempted_depth", json_integer(result.attempted_depth)},
      {"budget_exhausted", json_bool(result.budget_exhausted)},
      {"completed_depth", json_integer(result.completed_depth)},
      {"exception", json_bool(result.exception)},
      {"illegal", json_bool(!result.legal)},
      {"nodes", json_integer(result.nodes)},
      {"post_state_sha256", json_quote(result.post_state_sha256)},
      {"root_proof_hits", json_integer(result.root_proof_hits)},
      {"root_proof_probes", json_integer(result.root_proof_probes)},
      {"root_proof_shortcut", json_bool(result.root_proof_shortcut)},
      {"root_score", json_integer(result.root_score)},
  });
}

struct RawTimingObservation {
  std::vector<ps::Move> action{};
  cg::SearchStats stats{};
  bool exception{};
};

SearchObservation finish_timing_observation(
    const ps::GameState &state, RawTimingObservation raw) {
  SearchObservation result;
  result.exception = raw.exception;
  ps::GameState post = state;
  if (!raw.exception) {
    result.legal =
        action_is_legal(state, raw.action, result.action_ascii, post);
  }
  result.budget_exhausted = raw.stats.budget_exhausted;
  result.completed_depth = raw.stats.completed_turn_depth;
  result.nodes = raw.stats.nodes;
  result.root_proof_hits =
      raw.stats.root_rebound_win_hits + raw.stats.root_rebound_loss_hits;
  result.root_proof_probes = raw.stats.root_rebound_probes;
  result.root_proof_shortcut = raw.stats.nodes == 0 && !raw.exception;
  result.root_score = raw.stats.root_score;
  result.attempted_depth = raw.stats.attempted_turn_depth;
  result.post_state_sha256 = contest_state_identity(post);
  return result;
}

int timing_mode(std::string_view panel, std::string_view phase,
                std::uint64_t pair_index, std::uint64_t order_position,
                std::uint64_t state_index) {
  if (phase != "warmup" && phase != "measured") return 2;
  const std::uint64_t pair_limit = phase == "warmup" ? 30 : 300;
  if (pair_index >= pair_limit || order_position > 1) return 2;
  const std::vector<ps::GameState> states = panel_states(panel);
  if (states.empty() || state_index != pair_index % states.size()) return 2;
  const std::uint64_t expected_position =
      (pair_index % 2U == 0U)
          ? (kEngine == "control" ? 0U : 1U)
          : (kEngine == "candidate" ? 0U : 1U);
  if (order_position != expected_position) return 2;

  std::array<RawTimingObservation, kTimingRepetitions> raw_observations{};
  std::int64_t elapsed = 0;
  for (RawTimingObservation &raw : raw_observations) {
    cg::SearchConfig config;
    config.max_nodes = kActivationNodes;
    config.transposition_entries = cg::kTranspositionEntries;
    config.evaluation_entries = cg::kEvaluationEntries;
    config.exact_proof_mask = kProofMask;
    config.replay_value_blend_percent = 15;
    config.teacher_residual_weight_percent = 100;
    std::unique_ptr<cg::CompleteTurnSearch> search;
    const auto started = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point ended;
    try {
      search = std::make_unique<cg::CompleteTurnSearch>(
          states[state_index], config);
      raw.action = search->run();
      ended = std::chrono::steady_clock::now();
    } catch (...) {
      ended = std::chrono::steady_clock::now();
      raw.exception = true;
    }
    elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                   ended - started)
                   .count();
    if (search) raw.stats = search->stats();
  }
  if (elapsed <= 0) return 2;

  std::array<SearchObservation, kTimingRepetitions> observations{};
  std::vector<std::string> results;
  results.reserve(kTimingRepetitions);
  for (std::size_t index = 0; index < observations.size(); ++index) {
    observations[index] = finish_timing_observation(
        states[state_index], std::move(raw_observations[index]));
    const SearchObservation &observation = observations[index];
    results.push_back(timing_result_json(observation));
  }
  const std::string results_json = json_array(results);
  const std::string workload = json_object({
      {"evaluation_entries", json_integer(cg::kEvaluationEntries)},
      {"max_nodes", json_integer(kActivationNodes)},
      {"pair_index", json_integer(pair_index)},
      {"panel", json_quote(panel)},
      {"phase", json_quote(phase)},
      {"proof_mask", json_integer(kProofMask)},
      {"repetitions", json_integer(kTimingRepetitions)},
      {"state_index", json_integer(state_index)},
      {"state_sha256",
       json_quote(contest_state_identity(states[state_index]))},
      {"transposition_entries", json_integer(cg::kTranspositionEntries)},
  });
  const std::string output = json_object({
      {"elapsed_ns", json_integer(elapsed)},
      {"engine", json_quote(kEngine)},
      {"input_signature_sha256", json_quote(sha256(workload + "\n"))},
      {"mode", json_quote("timing")},
      {"order_position", json_integer(order_position)},
      {"pair_index", json_integer(pair_index)},
      {"panel", json_quote(panel)},
      {"phase", json_quote(phase)},
      {"policy", json_quote(kPolicy)},
      {"proof_mask", json_integer(kProofMask)},
      {"repetitions", json_integer(kTimingRepetitions)},
      {"results", results_json},
      {"result_signature_sha256", json_quote(sha256(results_json + "\n"))},
      {"schema", json_quote(kSchema)},
      {"state_index", json_integer(state_index)},
  });
  std::cout << output << '\n';
  return 0;
}

bool args_equal(int argc, char **argv,
                const std::vector<std::string_view> &expected) {
  if (argc != static_cast<int>(expected.size()) + 1) return false;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (std::string_view(argv[index + 1]) != expected[index]) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  try {
    if (argc == 5 && std::string_view(argv[1]) == "--mode" &&
        std::string_view(argv[2]) == "table-truth" &&
        std::string_view(argv[3]) == "--expected-policy") {
      return table_truth_mode(argv[4]);
    }

    const std::vector<std::string_view> public_args{
        "--mode", "public-screen", "--seed", kSeedHex,
        "--procedural-states", "1000", "--activation-states", "64",
        "--activation-nodes", "50000", "--depth1-states", "64",
        "--depth2-states", "24", "--oracle-max-nodes", "2000000",
        "--exact-proof-mask", "7",
    };
    if (args_equal(argc, argv, public_args)) return public_screen_mode();

    if (argc == 19 && std::string_view(argv[1]) == "--mode" &&
        std::string_view(argv[2]) == "timing" &&
        std::string_view(argv[3]) == "--panel" &&
        std::string_view(argv[5]) == "--phase" &&
        std::string_view(argv[7]) == "--pair-index" &&
        std::string_view(argv[9]) == "--order-position" &&
        std::string_view(argv[11]) == "--state-index" &&
        std::string_view(argv[13]) == "--nodes" &&
        std::string_view(argv[14]) == "50000" &&
        std::string_view(argv[15]) == "--repetitions" &&
        std::string_view(argv[16]) == "8" &&
        std::string_view(argv[17]) == "--exact-proof-mask" &&
        std::string_view(argv[18]) == "7") {
      const auto pair_index = parse_decimal(argv[8]);
      const auto order_position = parse_decimal(argv[10]);
      const auto state_index = parse_decimal(argv[12]);
      if (!pair_index || !order_position || !state_index) return 2;
      return timing_mode(argv[4], argv[6], *pair_index, *order_position,
                         *state_index);
    }
  } catch (...) {
    return 2;
  }
  return 2;
}
