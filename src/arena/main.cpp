#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

#include "papersoccer/arena.hpp"

#ifndef __EMSCRIPTEN__
#include "opening_bank_internal.hpp"
#endif

namespace ps = papersoccer;
namespace arena = papersoccer::arena;

namespace {

enum class Mode { Matches, Positions, Provenance };

struct CliConfig {
  Mode mode{Mode::Matches};
  ps::RulesConfig rules{};
  arena::ArenaBotConfig candidate{arena::MatchesConfig{}.candidate};
  arena::ArenaBotConfig reference{arena::MatchesConfig{}.reference};
  std::uint64_t base_seed{ps::RandomBot::default_seed()};
  std::size_t pairs{200};
  std::size_t opening_plies{0};
  bool opening_plies_seen{false};
  std::string opening_bank_path{};
  std::size_t max_plies{512};
  std::size_t bootstrap_samples{10000};
  std::size_t warmup_decisions{0};
  std::size_t positions{16};
  std::size_t generation_plies{24};
};

void print_usage(std::ostream &out) {
  out <<
      "Usage:\n"
      "  papersoccer_arena matches [options]\n"
      "  papersoccer_arena positions [options]\n"
      "  papersoccer_arena provenance\n\n"
      "Common options:\n"
      "  --seed N                         Base seed (default: 828927513140)\n"
      "  --rules normal|codingame         Rule profile (default: normal)\n"
      "  --width N --height N             Board dimensions (default: 8x10)\n"
      "  --candidate-kind random|mcts|alpha-beta|jacek-inspired|rank5-derived|deep-turn-search|jacek-replay-bfm\n"
      "                                     Candidate bot (default: mcts)\n"
      "  --reference-kind random|mcts|alpha-beta|jacek-inspired|rank5-derived|deep-turn-search|jacek-replay-bfm\n"
      "                                     Reference bot (default: mcts)\n"
      "  --candidate-iterations N         MCTS iterations (default: 2000)\n"
      "  --reference-iterations N         MCTS iterations (default: 2000)\n"
      "  --candidate-policy uniform|tactical\n"
      "  --reference-policy uniform|tactical\n"
      "  --candidate-leaf-policy rollout-only|tactical-quiescence\n"
      "  --reference-leaf-policy rollout-only|tactical-quiescence\n"
      "  --candidate-quiescence-max-depth N\n"
      "  --reference-quiescence-max-depth N\n"
      "  --candidate-quiescence-max-nodes N\n"
      "  --reference-quiescence-max-nodes N\n"
      "  --candidate-reuse true|false     Reuse the candidate tree\n"
      "  --reference-reuse true|false     Reuse the reference tree\n"
      "  --candidate-max-nodes N          Candidate tree bound (minimum: 2)\n"
      "  --reference-max-nodes N          Reference tree bound (minimum: 2)\n"
      "  --candidate-exploration X        Candidate UCT exploration\n"
      "  --reference-exploration X        Reference UCT exploration\n"
      "  --candidate-alpha-beta-depth N   Candidate possession-handoff depth\n"
      "  --reference-alpha-beta-depth N   Reference possession-handoff depth\n"
      "  --candidate-alpha-beta-max-nodes N\n"
      "                                     Candidate alpha-beta node budget\n"
      "  --reference-alpha-beta-max-nodes N\n"
      "                                     Reference alpha-beta node budget\n"
      "  --candidate-alpha-beta-table-entries N\n"
      "                                     Candidate transposition entries\n"
      "  --reference-alpha-beta-table-entries N\n"
      "                                     Reference transposition entries\n"
      "  --candidate-alpha-beta-max-search-plies N\n"
      "                                     Candidate soft physical horizon\n"
      "  --reference-alpha-beta-max-search-plies N\n"
      "                                     Reference soft physical horizon\n"
      "  --candidate-complete-turn-max-nodes N\n"
      "                                     DeepTurnSearch work: 100000, 200000, or 400000\n"
      "  --reference-complete-turn-max-nodes N\n"
      "                                     DeepTurnSearch work: 100000, 200000, or 400000\n"
      "  --candidate-jacek-replay-model PATH\n"
      "  --reference-jacek-replay-model PATH\n"
      "                                     Required Jacek replay checkpoint\n"
      "  --candidate-jacek-replay-max-time-ms N\n"
      "  --reference-jacek-replay-max-time-ms N\n"
      "  --candidate-jacek-replay-max-tree-nodes N\n"
      "  --reference-jacek-replay-max-tree-nodes N\n"
      "  --candidate-jacek-replay-max-actions N\n"
      "  --reference-jacek-replay-max-actions N\n"
      "  --candidate-jacek-replay-max-partial-paths N\n"
      "  --reference-jacek-replay-max-partial-paths N\n"
      "  --candidate-jacek-replay-exploration X\n"
      "  --reference-jacek-replay-exploration X\n"
      "  --candidate-jacek-replay-fpu X\n"
      "  --reference-jacek-replay-fpu X\n\n"
      "Match options:\n"
      "  --pairs N                        Seed pairs / 2N games (default: 200)\n"
      "  --opening-plies N                Shared random opening length (default: 0)\n"
      "  --opening-bank PATH              Frozen uniform-legal-move transcript bank\n"
      "  --max-plies N                    Total per-game limit, opening included (default: 512)\n"
      "  --bootstrap-samples N            Paired resamples (default: 10000)\n"
      "  --warmup-decisions N             Untimed calls per entrant (default: 0)\n\n"
      "Position options:\n"
      "  --positions N                    Positions to measure (default: 16)\n"
      "  --generation-plies N             Random plies per position (default: 24)\n"
      "  -h, --help                       Show this help\n\n"
      "Defaults compare tactical quiescence MCTS (candidate) with tactical\n"
      "rollout-only MCTS (reference); both reuse their trees. JSON is written\n"
      "to standard output.\n";
}

template <typename UInt>
UInt parse_unsigned(std::string_view value, std::string_view option) {
  static_assert(std::is_unsigned_v<UInt>);
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument(std::string(option) + " requires an unsigned integer");
  }
  UInt parsed{};
  const char *first = value.data();
  const char *last = value.data() + value.size();
  const auto result = std::from_chars(first, last, parsed);
  if (result.ec != std::errc{} || result.ptr != last) {
    throw std::invalid_argument(std::string(option) + " requires an unsigned integer");
  }
  return parsed;
}

int parse_dimension(std::string_view value, std::string_view option) {
  const unsigned int parsed = parse_unsigned<unsigned int>(value, option);
  if (parsed > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string(option) + " is too large");
  }
  return static_cast<int>(parsed);
}

double parse_double(std::string_view value, std::string_view option) {
  std::string owned(value);
  std::size_t consumed = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(owned, &consumed);
  } catch (...) {
    throw std::invalid_argument(std::string(option) + " requires a number");
  }
  if (consumed != owned.size() || !std::isfinite(parsed)) {
    throw std::invalid_argument(std::string(option) + " requires a finite number");
  }
  return parsed;
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true" || value == "on" || value == "1") {
    return true;
  }
  if (value == "false" || value == "off" || value == "0") {
    return false;
  }
  throw std::invalid_argument(std::string(option) +
                              " requires true or false");
}

ps::BotKind parse_kind(std::string_view value, std::string_view option) {
  if (value == "random") {
    return ps::BotKind::Random;
  }
  if (value == "mcts") {
    return ps::BotKind::Mcts;
  }
  if (value == "alpha-beta") {
    return ps::BotKind::AlphaBeta;
  }
  if (value == "jacek-inspired") {
    return ps::BotKind::JacekInspired;
  }
  if (value == "rank5-derived") {
    return ps::BotKind::Rank5Derived;
  }
  if (value == "deep-turn-search") {
    return ps::BotKind::DeepTurnSearch;
  }
  if (value == "jacek-replay-bfm") {
    return ps::BotKind::JacekReplayBfm;
  }
  throw std::invalid_argument(std::string(option) +
                              " requires random, mcts, alpha-beta, "
                              "jacek-inspired, rank5-derived, or "
                              "deep-turn-search, or jacek-replay-bfm");
}

void apply_rules_profile(ps::RulesConfig &rules, std::string_view value,
                         std::string_view option) {
  if (value == "normal") {
    rules.goal_rule = ps::GoalRule::OpponentGoalOnly;
    rules.blocked_rule = ps::BlockedRule::PlayerToMoveLoses;
    return;
  }
  if (value == "codingame") {
    rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
    rules.blocked_rule = ps::BlockedRule::MoverLoses;
    return;
  }
  throw std::invalid_argument(std::string(option) +
                              " requires normal or codingame");
}

ps::MctsRolloutPolicy parse_policy(std::string_view value,
                                   std::string_view option) {
  if (value == "uniform") {
    return ps::MctsRolloutPolicy::Uniform;
  }
  if (value == "tactical") {
    return ps::MctsRolloutPolicy::Tactical;
  }
  throw std::invalid_argument(std::string(option) +
                              " requires uniform or tactical");
}

ps::MctsLeafPolicy parse_leaf_policy(std::string_view value,
                                     std::string_view option) {
  if (value == "rollout-only") {
    return ps::MctsLeafPolicy::RolloutOnly;
  }
  if (value == "tactical-quiescence") {
    return ps::MctsLeafPolicy::TacticalQuiescence;
  }
  throw std::invalid_argument(
      std::string(option) +
      " requires rollout-only or tactical-quiescence");
}

void require_mode(Mode actual, Mode expected, std::string_view option) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(option) + " is only valid in " +
                                (expected == Mode::Matches ? "matches" : "positions") +
                                " mode");
  }
}

CliConfig parse_cli(int argc, char **argv) {
  if (argc < 2) {
    throw std::invalid_argument("missing mode");
  }

  CliConfig config;
  const std::string_view mode(argv[1]);
  if (mode == "matches") {
    config.mode = Mode::Matches;
  } else if (mode == "positions") {
    config.mode = Mode::Positions;
  } else if (mode == "provenance") {
    if (argc != 2) {
      throw std::invalid_argument("provenance mode accepts no options");
    }
    config.mode = Mode::Provenance;
    return config;
  } else {
    throw std::invalid_argument("mode must be matches, positions, or provenance");
  }

  for (int index = 2; index < argc; ++index) {
    const std::string_view option(argv[index]);
    auto value = [&]() -> std::string_view {
      if (++index >= argc) {
        throw std::invalid_argument(std::string(option) + " requires a value");
      }
      return argv[index];
    };

    if (option == "--seed") {
      config.base_seed = parse_unsigned<std::uint64_t>(value(), option);
    } else if (option == "--rules") {
      apply_rules_profile(config.rules, value(), option);
    } else if (option == "--width") {
      config.rules.width = parse_dimension(value(), option);
    } else if (option == "--height") {
      config.rules.height = parse_dimension(value(), option);
    } else if (option == "--pairs") {
      require_mode(config.mode, Mode::Matches, option);
      config.pairs = parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--opening-plies") {
      require_mode(config.mode, Mode::Matches, option);
      config.opening_plies_seen = true;
      config.opening_plies = parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--opening-bank") {
      require_mode(config.mode, Mode::Matches, option);
      if (!config.opening_bank_path.empty()) {
        throw std::invalid_argument(
            "--opening-bank may be specified only once");
      }
      config.opening_bank_path = value();
      if (config.opening_bank_path.empty()) {
        throw std::invalid_argument("--opening-bank requires a nonempty path");
      }
    } else if (option == "--max-plies") {
      require_mode(config.mode, Mode::Matches, option);
      config.max_plies = parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--bootstrap-samples") {
      require_mode(config.mode, Mode::Matches, option);
      config.bootstrap_samples = parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--warmup-decisions") {
      require_mode(config.mode, Mode::Matches, option);
      config.warmup_decisions = parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--positions") {
      require_mode(config.mode, Mode::Positions, option);
      config.positions = parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--generation-plies") {
      require_mode(config.mode, Mode::Positions, option);
      config.generation_plies = parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--candidate-kind") {
      config.candidate.kind = parse_kind(value(), option);
    } else if (option == "--reference-kind") {
      config.reference.kind = parse_kind(value(), option);
    } else if (option == "--candidate-iterations") {
      config.candidate.iterations =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--reference-iterations") {
      config.reference.iterations =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--candidate-policy") {
      config.candidate.rollout_policy = parse_policy(value(), option);
    } else if (option == "--reference-policy") {
      config.reference.rollout_policy = parse_policy(value(), option);
    } else if (option == "--candidate-leaf-policy") {
      config.candidate.leaf_policy = parse_leaf_policy(value(), option);
    } else if (option == "--reference-leaf-policy") {
      config.reference.leaf_policy = parse_leaf_policy(value(), option);
    } else if (option == "--candidate-quiescence-max-depth") {
      config.candidate.quiescence_max_depth =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--reference-quiescence-max-depth") {
      config.reference.quiescence_max_depth =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--candidate-quiescence-max-nodes") {
      config.candidate.quiescence_max_nodes =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--reference-quiescence-max-nodes") {
      config.reference.quiescence_max_nodes =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--candidate-reuse") {
      config.candidate.reuse_tree = parse_bool(value(), option);
    } else if (option == "--reference-reuse") {
      config.reference.reuse_tree = parse_bool(value(), option);
    } else if (option == "--candidate-max-nodes") {
      config.candidate.max_nodes = parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--reference-max-nodes") {
      config.reference.max_nodes = parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--candidate-exploration") {
      config.candidate.exploration = parse_double(value(), option);
    } else if (option == "--reference-exploration") {
      config.reference.exploration = parse_double(value(), option);
    } else if (option == "--candidate-alpha-beta-depth") {
      config.candidate.alpha_beta_depth =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--reference-alpha-beta-depth") {
      config.reference.alpha_beta_depth =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--candidate-alpha-beta-max-nodes") {
      config.candidate.alpha_beta_max_nodes =
          parse_unsigned<std::uint64_t>(value(), option);
    } else if (option == "--reference-alpha-beta-max-nodes") {
      config.reference.alpha_beta_max_nodes =
          parse_unsigned<std::uint64_t>(value(), option);
    } else if (option == "--candidate-alpha-beta-table-entries") {
      config.candidate.alpha_beta_transposition_table_entries =
          parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--reference-alpha-beta-table-entries") {
      config.reference.alpha_beta_transposition_table_entries =
          parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--candidate-alpha-beta-max-search-plies") {
      config.candidate.alpha_beta_max_search_plies =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--reference-alpha-beta-max-search-plies") {
      config.reference.alpha_beta_max_search_plies =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--candidate-complete-turn-max-nodes") {
      config.candidate.complete_turn_max_nodes =
          parse_unsigned<std::uint64_t>(value(), option);
    } else if (option == "--reference-complete-turn-max-nodes") {
      config.reference.complete_turn_max_nodes =
          parse_unsigned<std::uint64_t>(value(), option);
    } else if (option == "--candidate-jacek-replay-model") {
      config.candidate.jacek_replay_bfm.model_path = value();
    } else if (option == "--reference-jacek-replay-model") {
      config.reference.jacek_replay_bfm.model_path = value();
    } else if (option == "--candidate-jacek-replay-max-time-ms") {
      config.candidate.jacek_replay_bfm.max_time_ms =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--reference-jacek-replay-max-time-ms") {
      config.reference.jacek_replay_bfm.max_time_ms =
          parse_unsigned<std::uint32_t>(value(), option);
    } else if (option == "--candidate-jacek-replay-max-tree-nodes") {
      config.candidate.jacek_replay_bfm.max_tree_nodes =
          parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--reference-jacek-replay-max-tree-nodes") {
      config.reference.jacek_replay_bfm.max_tree_nodes =
          parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--candidate-jacek-replay-max-actions") {
      config.candidate.jacek_replay_bfm.max_actions =
          parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--reference-jacek-replay-max-actions") {
      config.reference.jacek_replay_bfm.max_actions =
          parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--candidate-jacek-replay-max-partial-paths") {
      config.candidate.jacek_replay_bfm.max_partial_paths =
          parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--reference-jacek-replay-max-partial-paths") {
      config.reference.jacek_replay_bfm.max_partial_paths =
          parse_unsigned<std::size_t>(value(), option);
    } else if (option == "--candidate-jacek-replay-exploration") {
      config.candidate.jacek_replay_bfm.exploration =
          parse_double(value(), option);
    } else if (option == "--reference-jacek-replay-exploration") {
      config.reference.jacek_replay_bfm.exploration =
          parse_double(value(), option);
    } else if (option == "--candidate-jacek-replay-fpu") {
      config.candidate.jacek_replay_bfm.fpu = parse_double(value(), option);
    } else if (option == "--reference-jacek-replay-fpu") {
      config.reference.jacek_replay_bfm.fpu = parse_double(value(), option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (!config.opening_bank_path.empty() && config.opening_plies_seen) {
    throw std::invalid_argument(
        "--opening-bank is incompatible with --opening-plies");
  }
  return config;
}

#ifndef __EMSCRIPTEN__
std::vector<arena::FrozenOpening> load_frozen_openings(
    const std::string &path) {
  const papersoccer::opening_bank::Bank bank =
      papersoccer::opening_bank::load_bank(path);
  std::vector<arena::FrozenOpening> openings;
  openings.reserve(bank.records.size());
  for (const papersoccer::opening_bank::OpeningRecord &record : bank.records) {
    arena::FrozenOpening opening;
    opening.opening_id = record.opening_id;
    opening.phase = record.phase;
    opening.depth = record.depth;
    opening.generation_seed = record.generation_seed;
    opening.state_hash = record.state_hash;
    opening.canonical_key = record.canonical_key;
    opening.transcript = record.moves;
    opening.state =
        papersoccer::opening_bank::replay_transcript(record.moves);
    openings.push_back(std::move(opening));
  }
  return openings;
}
#endif

}  // namespace

int main(int argc, char **argv) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      print_usage(std::cout);
      return 0;
    }
  }

  try {
    const CliConfig cli = parse_cli(argc, argv);
    if (cli.mode == Mode::Provenance) {
      std::cout << arena::build_provenance_json() << '\n';
    } else if (cli.mode == Mode::Matches) {
      arena::MatchesConfig config;
      config.rules = cli.rules;
      config.candidate = cli.candidate;
      config.reference = cli.reference;
      config.base_seed = cli.base_seed;
      config.seed_pairs = cli.pairs;
      config.opening_plies = cli.opening_plies;
      config.max_plies = cli.max_plies;
      config.bootstrap_samples = cli.bootstrap_samples;
      config.warmup_decisions = cli.warmup_decisions;
      if (!cli.opening_bank_path.empty()) {
#ifndef __EMSCRIPTEN__
        config.frozen_openings =
            load_frozen_openings(cli.opening_bank_path);
#else
        throw std::invalid_argument(
            "--opening-bank is available only in the native arena");
#endif
      }
      std::cout << arena::run_matches_json(config) << '\n';
    } else {
      arena::PositionsConfig config;
      config.rules = cli.rules;
      config.candidate = cli.candidate;
      config.reference = cli.reference;
      config.base_seed = cli.base_seed;
      config.position_count = cli.positions;
      config.generation_plies = cli.generation_plies;
      std::cout << arena::run_positions_json(config) << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "papersoccer_arena: " << error.what() << "\n\n";
    print_usage(std::cerr);
    return 2;
  }
}
