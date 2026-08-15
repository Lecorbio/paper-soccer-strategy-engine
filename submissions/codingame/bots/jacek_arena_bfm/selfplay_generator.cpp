#define JACEK_ARENA_BFM_NO_MAIN
#include "submission.cpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace jab = jacek_arena_bfm;

namespace {

std::uint64_t next_random(std::uint64_t &state) {
  state += 0x9e3779b97f4a7c15ULL;
  std::uint64_t value = state;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

struct Turn {
  std::uint8_t player{0};
  std::uint64_t before_hash{0};
  jab::Action action{};
  bool procedural_opening{false};
  std::array<std::uint8_t, jab::kFeatureCount> features{};
};

class Sha256 {
 public:
  void update(const std::string &text) {
    for (const unsigned char byte : text) {
      buffer_[buffer_size_++] = byte;
      ++bytes_;
      if (buffer_size_ == 64) {
        transform();
        buffer_size_ = 0;
      }
    }
  }

  std::string finish() {
    const std::uint64_t bit_count = bytes_ * 8;
    buffer_[buffer_size_++] = 0x80;
    if (buffer_size_ > 56) {
      while (buffer_size_ < 64) buffer_[buffer_size_++] = 0;
      transform();
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56) buffer_[buffer_size_++] = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
      buffer_[buffer_size_++] = static_cast<std::uint8_t>(bit_count >> shift);
    }
    transform();
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto word : state_) output << std::setw(8) << word;
    return output.str();
  }

 private:
  static std::uint32_t rotate(std::uint32_t value, int amount) {
    return (value >> amount) | (value << (32 - amount));
  }

  void transform() {
    static constexpr std::array<std::uint32_t, 64> constants{
      0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
      0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
      0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
      0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
      0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
      0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
      0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
      0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
    std::array<std::uint32_t, 64> words{};
    for (int index = 0; index < 16; ++index) {
      words[index] = (static_cast<std::uint32_t>(buffer_[index * 4]) << 24U) |
                     (static_cast<std::uint32_t>(buffer_[index * 4 + 1]) << 16U) |
                     (static_cast<std::uint32_t>(buffer_[index * 4 + 2]) << 8U) |
                     buffer_[index * 4 + 3];
    }
    for (int index = 16; index < 64; ++index) {
      const auto s0 = rotate(words[index - 15], 7) ^
                      rotate(words[index - 15], 18) ^ (words[index - 15] >> 3U);
      const auto s1 = rotate(words[index - 2], 17) ^
                      rotate(words[index - 2], 19) ^ (words[index - 2] >> 10U);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    auto [a,b,c,d,e,f,g,h] = state_;
    for (int index = 0; index < 64; ++index) {
      const auto s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      const auto choice = (e & f) ^ (~e & g);
      const auto first = h + s1 + choice + constants[index] + words[index];
      const auto s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto second = s0 + majority;
      h=g; g=f; f=e; e=d+first; d=c; c=b; b=a; a=first+second;
    }
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
    state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }

  std::array<std::uint32_t, 8> state_{
      0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
      0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_{0};
  std::uint64_t bytes_{0};
};

std::string game_id(std::uint64_t seed, std::size_t index) {
  std::ostringstream output;
  output << "scratch-" << std::hex << std::setw(16) << std::setfill('0')
         << seed << '-' << std::dec << index;
  return output.str();
}

jab::Action choose_action(const jab::State &state, std::uint64_t &random,
                          bool opening) {
  const auto actions = jab::generate_actions(
      state, opening ? jab::GeneratorStrategy::Fixed250NineOne
                     : jab::GeneratorStrategy::TacticalProgressive,
      true);
  if (actions.empty()) throw std::runtime_error("generator returned no action");
  if (opening) return actions[next_random(random) % actions.size()];
  struct Candidate {
    double score;
    std::size_t index;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(actions.size());
  const std::uint8_t mover = state.to_move;
  for (std::size_t index = 0; index < actions.size(); ++index) {
    jab::State successor = state;
    if (!jab::apply_action(successor, actions[index])) continue;
    double value = 0.0;
    if (successor.terminal()) {
      value = successor.winner == mover ? 1.0 : -1.0;
    } else {
      value = successor.to_move == mover ? jab::evaluate(successor)
                                         : -jab::evaluate(successor);
    }
    // Seeded root jitter is deliberately small: the bootstrap network drives
    // play, while nearby alternatives still create a diverse fresh corpus.
    const double jitter = static_cast<double>(next_random(random) & 0xffffU) /
                              65535.0 * 0.04 -
                          0.02;
    candidates.push_back({value + jitter, index});
  }
  if (candidates.empty()) throw std::runtime_error("no legal evaluated action");
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &left, const Candidate &right) {
              if (left.score != right.score) return left.score > right.score;
              return left.index < right.index;
            });
  const std::size_t pool = std::min<std::size_t>(4, candidates.size());
  const std::uint64_t draw = next_random(random) % 10;
  const std::size_t choice = draw < 7 ? 0 : 1 +
      (next_random(random) % std::max<std::size_t>(1, pool - 1));
  return actions[candidates[std::min(choice, pool - 1)].index];
}

void write_game(std::ostream &output, std::size_t index, std::uint64_t seed,
                int opening_depth, const std::vector<Turn> &turns,
                const jab::State &terminal) {
  output << "{\"schema\":\"jacek_arena_bfm.scratch-game.v1\""
         << ",\"game_id\":\"" << game_id(seed, index) << "\""
         << ",\"seed\":" << seed
         << ",\"opening_depth\":" << opening_depth
         << ",\"winner\":" << static_cast<int>(terminal.winner)
         << ",\"physical_edges\":" << terminal.ply
         << ",\"model_identity\":\"" << jacek_arena_bfm::model::kIdentity
         << "\",\"turns\":[";
  for (std::size_t turn = 0; turn < turns.size(); ++turn) {
    if (turn != 0) output << ',';
    output << "{\"player\":" << static_cast<int>(turns[turn].player)
           << ",\"before_hash\":\"" << std::hex << std::setw(16)
           << std::setfill('0') << turns[turn].before_hash << std::dec << "\""
           << ",\"opening\":"
           << (turns[turn].procedural_opening ? "true" : "false")
           << ",\"action\":\"" << turns[turn].action.text() << "\"}";
  }
  output << "]}\n";
}

std::string evidence_digest(std::size_t index, std::uint64_t seed,
                            int opening_depth, const std::vector<Turn> &turns,
                            const jab::State &terminal) {
  std::ostringstream canonical;
  canonical << game_id(seed, index) << '|' << opening_depth << '|'
            << static_cast<int>(terminal.winner) << '|' << terminal.ply;
  for (const auto &turn : turns) {
    canonical << '|' << static_cast<int>(turn.player) << ':'
              << turn.before_hash << ':' << turn.action.text();
  }
  Sha256 digest;
  digest.update(canonical.str());
  return digest.finish();
}

void write_rows(std::ostream &output, std::size_t index, std::uint64_t seed,
                int opening_depth, const std::vector<Turn> &turns,
                const jab::State &terminal, const std::string &campaign_id,
                const std::string &timestamp, const std::string &producer_sha,
                double weight) {
  const std::string id = game_id(seed, index);
  const std::string evidence_sha = evidence_digest(
      index, seed, opening_depth, turns, terminal);
  for (std::size_t turn_index = 0; turn_index < turns.size(); ++turn_index) {
    const auto &turn = turns[turn_index];
    const int target = terminal.winner == turn.player ? 1 : -1;
    output << "{\"schema\":\"papersoccer.jacek-arena-bfm.corpus-row.v1\""
           << ",\"namespace\":\"jacek_arena_bfm\""
           << ",\"campaign_id\":\"" << campaign_id << "\""
           << ",\"sample_id\":\"" << id << ":" << turn_index << "\""
           << ",\"game_id\":\"" << id << "\""
           << ",\"generated_at_utc\":\"" << timestamp << "\""
           << ",\"evidence_at_utc\":\"" << timestamp << "\""
           << ",\"producer_source_sha256\":\"" << producer_sha << "\""
           << ",\"evidence_sha256\":\"" << evidence_sha << "\""
           << ",\"representation\":\"mover_relative_316_edges_plus_105x8_distance_v1\""
           << ",\"weight\":" << weight
           << ",\"kind\":\"value\",\"source_kind\":\"scratch_selfplay\""
           << ",\"target\":" << target
           << ",\"label_method\":\"terminal_outcome\""
           << ",\"opening_depth\":" << opening_depth
           << ",\"initialization\":\"random\",\"checkpoint_inputs\":[]"
           << ",\"window_id\":null,\"submission_id\":null"
           << ",\"features\":[";
    for (std::size_t feature = 0; feature < turn.features.size(); ++feature) {
      if (feature != 0) output << ',';
      output << static_cast<int>(turn.features[feature]);
    }
    output << "]}\n";
  }
}

std::uint64_t parse_u64(const char *text) {
  std::size_t consumed = 0;
  const std::string value(text);
  const auto parsed = std::stoull(value, &consumed, 0);
  if (consumed != value.size()) throw std::invalid_argument("invalid integer");
  return parsed;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    std::size_t games = 2000;
    std::size_t start_index = 0;
    std::uint64_t seed = 0x4a4143454b465245ULL;
    std::string output_path;
    bool row_format = false;
    std::string campaign_id;
    std::string timestamp;
    std::string producer_sha;
    double weight = 1.0;
    for (int argument = 1; argument < argc; ++argument) {
      const std::string option = argv[argument];
      if (option == "--games" && argument + 1 < argc) {
        games = static_cast<std::size_t>(parse_u64(argv[++argument]));
      } else if (option == "--start-index" && argument + 1 < argc) {
        start_index = static_cast<std::size_t>(parse_u64(argv[++argument]));
      } else if (option == "--seed" && argument + 1 < argc) {
        seed = parse_u64(argv[++argument]);
      } else if (option == "--output" && argument + 1 < argc) {
        output_path = argv[++argument];
      } else if (option == "--rows") {
        row_format = true;
      } else if (option == "--campaign-id" && argument + 1 < argc) {
        campaign_id = argv[++argument];
      } else if (option == "--timestamp" && argument + 1 < argc) {
        timestamp = argv[++argument];
      } else if (option == "--producer-source-sha256" && argument + 1 < argc) {
        producer_sha = argv[++argument];
      } else if (option == "--weight" && argument + 1 < argc) {
        weight = std::stod(argv[++argument]);
      } else {
        throw std::invalid_argument(
            "usage: selfplay_generator [--games N] [--start-index N] [--seed N] [--output FILE] [--rows --campaign-id ID --timestamp UTC --producer-source-sha256 SHA] [--weight W]");
      }
    }
    if (games == 0) throw std::invalid_argument("games must be positive");
    const auto valid_sha = [](const std::string &value) {
      return value.size() == 64 && std::all_of(value.begin(), value.end(),
          [](char character) { return (character >= '0' && character <= '9') ||
                                      (character >= 'a' && character <= 'f'); });
    };
    if (row_format && (campaign_id.empty() || timestamp.empty() ||
                       !valid_sha(producer_sha))) {
      throw std::invalid_argument(
          "--rows requires campaign ID, UTC timestamp, and lowercase producer SHA-256");
    }
    if (!(weight > 0.0 && weight <= 100.0)) {
      throw std::invalid_argument("weight must be in (0,100]");
    }
    std::ofstream file;
    std::ostream *output = &std::cout;
    if (!output_path.empty()) {
      file.open(output_path, std::ios::binary | std::ios::trunc);
      if (!file) throw std::runtime_error("cannot open output file");
      output = &file;
    }
    constexpr std::array<int, 4> kOpeningDepths{0, 4, 8, 12};
    for (std::size_t game = 0; game < games; ++game) {
      const std::size_t global_index = start_index + game;
      std::uint64_t random = seed ^ (global_index * 0xd1342543de82ef95ULL);
      const int opening_depth =
          kOpeningDepths[global_index % kOpeningDepths.size()];
      jab::State state = jab::initial_state();
      std::vector<Turn> turns;
      turns.reserve(96);
      while (!state.terminal()) {
        const bool opening = state.ply < opening_depth;
        const jab::Action action = choose_action(state, random, opening);
        Turn turn;
        turn.player = state.to_move;
        turn.before_hash = jab::state_hash(state);
        turn.action = action;
        turn.procedural_opening = opening;
        std::size_t feature_count = 0;
        const auto active = jab::active_features(state, feature_count);
        for (std::size_t item = 0; item < feature_count; ++item) {
          turn.features[active[item]] = 1;
        }
        if (!jab::apply_action(state, action)) {
          throw std::runtime_error("self-play selected illegal action");
        }
        turns.push_back(turn);
        if (turns.size() > jab::kEdgeCount) {
          throw std::runtime_error("self-play exceeded finite edge bound");
        }
      }
      if (row_format) {
        write_rows(*output, global_index, seed, opening_depth, turns, state,
                   campaign_id, timestamp, producer_sha, weight);
      } else {
        write_game(*output, global_index, seed, opening_depth, turns, state);
      }
    }
    std::cerr << "generated " << games << " fresh scratch games with seed "
              << seed << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "selfplay_generator: " << error.what() << '\n';
    return 1;
  }
}
