#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "jacek_features.hpp"
#include "mcts_internal.hpp"
#include "papersoccer/bot.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

constexpr std::size_t kSamplesPerGame = 24;
constexpr std::size_t kMaximumPlies = 192;
constexpr std::uint32_t kTeacherDepth = 5;
constexpr std::size_t kTeacherTableEntries = 16'384;
constexpr std::uint32_t kTeacherSearchPlies = 12;

struct Sample {
  std::size_t ply{};
  ps::Player player{ps::Player::One};
  int score{};
  std::uint32_t completed_depth{};
  std::uint64_t nodes{};
  ps::detail::JacekSparseFeatures features{};
};

std::uint64_t next_random(std::uint64_t &state) noexcept {
  state ^= state >> 12U;
  state ^= state << 25U;
  state ^= state >> 27U;
  return state * 2685821657736338717ULL;
}

std::uint64_t parse_unsigned(const char *text, const char *label) {
  try {
    std::size_t consumed = 0;
    const std::uint64_t value = std::stoull(text, &consumed);
    if (consumed != std::string(text).size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string(label) +
                                " must be an unsigned integer");
  }
}

std::vector<Sample> evenly_spaced(std::vector<Sample> samples) {
  if (samples.size() <= kSamplesPerGame) {
    return samples;
  }
  std::vector<Sample> result;
  result.reserve(kSamplesPerGame);
  for (std::size_t index = 0; index < kSamplesPerGame; ++index) {
    const std::size_t source =
        index * (samples.size() - 1) / (kSamplesPerGame - 1);
    result.push_back(std::move(samples[source]));
  }
  return result;
}

void write_sample(std::ostream &out, std::size_t game,
                  std::uint64_t seed, std::uint64_t teacher_nodes,
                  const Sample &sample) {
  out << "{\"schema\":\"papersoccer.jacek-training-sample.v2\","
      << "\"feature_schema\":"
         "\"canonical-edges316-onehot-true-turn-distance105x8-v1\","
      << "\"rules\":{\"width\":8,\"height\":10,"
         "\"goal_rule\":\"opponent-goal-only\","
         "\"blocked_rule\":\"player-to-move-loses\"},"
      << "\"teacher\":{\"kind\":\"alpha-beta\","
      << "\"max_turn_depth\":" << kTeacherDepth
      << ",\"max_nodes\":" << teacher_nodes
      << ",\"transposition_table_entries\":" << kTeacherTableEntries
      << ",\"max_search_plies\":" << kTeacherSearchPlies << "},"
      << "\"game\":" << game << ",\"seed\":\"" << seed
      << "\",\"ply\":" << sample.ply << ",\"player\":"
      << (sample.player == ps::Player::One ? 0 : 1)
      << ",\"teacher_score\":" << sample.score
      << ",\"completed_depth\":" << sample.completed_depth
      << ",\"nodes\":" << sample.nodes << ",\"active\":[";
  for (std::size_t index = 0; index < sample.features.count; ++index) {
    if (index != 0) {
      out << ',';
    }
    out << sample.features.indices[index];
  }
  out << "]}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 5) {
      throw std::invalid_argument(
          "usage: papersoccer_jacek_training_data OUTPUT.jsonl "
          "[GAMES=1024] [SEED=73194721] [TEACHER_NODES=4000]");
    }
    const std::string output_path = argv[1];
    const std::size_t games =
        argc >= 3 ? static_cast<std::size_t>(
                        parse_unsigned(argv[2], "game count"))
                  : 1024;
    const std::uint64_t base_seed =
        argc >= 4 ? parse_unsigned(argv[3], "seed") : 73194721ULL;
    const std::uint64_t teacher_nodes =
        argc >= 5 ? parse_unsigned(argv[4], "teacher nodes") : 4000ULL;
    if (games == 0 || teacher_nodes == 0) {
      throw std::invalid_argument(
          "game count and teacher nodes must be positive");
    }

    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("could not open training output");
    }

    ps::AlphaBetaConfig teacher_config;
    teacher_config.max_turn_depth = kTeacherDepth;
    teacher_config.max_nodes = teacher_nodes;
    teacher_config.transposition_table_entries = kTeacherTableEntries;
    teacher_config.max_search_plies = kTeacherSearchPlies;
    ps::AlphaBetaBot teacher(teacher_config);

    ps::RulesConfig rules;
    rules.width = 8;
    rules.height = 10;
    rules.goal_rule = ps::GoalRule::OpponentGoalOnly;
    rules.blocked_rule = ps::BlockedRule::PlayerToMoveLoses;
    auto topology = std::make_shared<ps::detail::SearchTopology>(rules);
    ps::detail::JacekFeatureEncoder encoder(topology);
    std::size_t written = 0;
    std::size_t completed_games = 0;
    for (std::size_t game = 0; game < games; ++game) {
      const std::uint64_t game_seed =
          base_seed + game * 0x9e3779b97f4a7c15ULL;
      std::uint64_t random_state =
          game_seed ^ 0xd1b54a32d192ed03ULL;
      ps::GameState state = ps::make_initial_state(rules);
      std::vector<Sample> samples;
      samples.reserve(96);

      for (std::size_t ply = 0;
           ply < kMaximumPlies && !ps::is_terminal(state); ++ply) {
        ps::detail::SearchPosition compact(topology, state);
        const ps::Move teacher_move = teacher.choose_move(state);
        const ps::AlphaBetaSearchStats &stats = teacher.last_search_stats();
        if (stats.completed_turn_depth > 0) {
          samples.push_back(Sample{
              ply,
              state.to_move,
              stats.root_score,
              stats.completed_turn_depth,
              stats.nodes,
              encoder.sparse_features(compact),
          });
        }

        const std::vector<ps::Move> legal = ps::legal_moves(state);
        const bool explore =
            ply < 6 || (next_random(random_state) % 100U) < 18U;
        const ps::Move played =
            explore ? legal[next_random(random_state) % legal.size()]
                    : teacher_move;
        state = ps::apply_move(state, played);
      }
      if (ps::is_terminal(state)) {
        ++completed_games;
      }
      for (const Sample &sample : evenly_spaced(std::move(samples))) {
        write_sample(output, game, game_seed, teacher_nodes, sample);
        ++written;
      }
      if ((game + 1) % 64 == 0 || game + 1 == games) {
        std::cerr << "generated " << (game + 1) << '/' << games
                  << " games, " << written << " samples\n";
      }
    }
    std::cerr << "complete games: " << completed_games << '/' << games
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Jacek training data: " << error.what() << '\n';
    return 1;
  }
}
