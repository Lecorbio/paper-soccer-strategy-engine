#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#define PAPER_SOCCER_JACEK_NATIVE_RESTART_NO_MAIN
#include "jacek_native_restart_round2.cpp"

namespace {

constexpr std::string_view kLateGatePlanSchema =
    "papersoccer.jacek-native-late-pacing-game-gate-plan/v2";
constexpr std::string_view kLateGatePanelSha256 =
    "64283c5a4e7c5ac360969120a79e966a10cff9eb39d9cb0380cadccc14198246";
constexpr std::string_view kLateGateSelection =
    "all-root/source-game-heldout-panel-roots/v1";
constexpr std::string_view kLateGateHeader =
    "run_id\tpopulation\trole\tcollector_sha256\tarena_manifest_sha256\t"
    "game_id\tcandidate_player\tprefix_turn\tstate_id\tcanonical_key\t"
    "transcript";
constexpr std::size_t kLateGateRuns = 4U;
constexpr std::size_t kLateGateRoots = 64U;
constexpr std::size_t kLateGateGames = 128U;
constexpr std::size_t kLateGateRequiredWins = 70U;
constexpr std::size_t kLateGateRequiredPerOrientation = 31U;
constexpr std::uint64_t kLateGateWork = 4'096ULL;
constexpr std::uint64_t kLateGateSeed = 2'026'090'108ULL;
constexpr float kLateGateTemperature = 3.0F;
constexpr std::size_t kLateGateTemperatureTurns = 12U;
constexpr std::size_t kLateGateMaximumGeneratedTurns = 384U;

struct LateGateRoot {
  std::string run_id;
  std::string population;
  std::string role;
  std::string collector_sha256;
  std::string arena_manifest_sha256;
  std::string game_id;
  int candidate_player{};
  std::size_t prefix_turn{};
  std::string state_id;
  std::string canonical_key;
  std::string transcript;
  ps::GameState state;
};

struct LateGatePlan {
  std::string bytes;
  std::string sha256;
  std::string panel_sha256;
  std::string source_sha256;
  std::vector<LateGateRoot> roots;
};

struct LateGateArguments {
  std::string plan;
  std::string plan_sha256;
  std::string candidate_checkpoint;
  std::string candidate_artifact_sha256;
  std::string baseline_checkpoint;
  std::string baseline_artifact_sha256;
};

struct LateGateSummary {
  std::size_t candidate_wins{};
  std::size_t baseline_wins{};
  std::array<std::size_t, 2> orientation_wins{};
  std::array<std::size_t, 2> population_wins{};
  std::size_t unfinished{};
  std::size_t operational_failures{};
  std::uint64_t searches{};
  std::uint64_t expansions{};
  std::size_t maximum_tree{};
};

bool late_gate_state_id(std::string_view value) {
  return value.starts_with("fnv1a64:") && lower_hex(value.substr(8U), 16U);
}

std::string late_gate_file_sha256(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open gate input: " + path);
  const std::string raw{std::istreambuf_iterator<char>(input),
                        std::istreambuf_iterator<char>()};
  return native::sha256_hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(raw.data()), raw.size()));
}

std::vector<std::string> late_gate_actions(std::string_view transcript) {
  std::vector<std::string> result;
  if (transcript.empty()) return result;
  for (const std::string_view action : split_exact(transcript, '/')) {
    if (action.empty() || action.size() > 250U ||
        !std::all_of(action.begin(), action.end(), [](char direction) {
          return direction >= '0' && direction <= '7';
        })) {
      throw std::invalid_argument("gate plan contains an invalid action");
    }
    result.emplace_back(action);
  }
  return result;
}

LateGatePlan read_late_gate_plan(const std::string &path,
                                 std::string_view expected_sha256) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open late-pacing gate plan");
  LateGatePlan plan;
  plan.bytes.assign(std::istreambuf_iterator<char>(input),
                    std::istreambuf_iterator<char>());
  if (plan.bytes.empty() || plan.bytes.size() > 4U * 1024U * 1024U ||
      plan.bytes.back() != '\n' || plan.bytes.find('\0') != std::string::npos ||
      plan.bytes.find('\r') != std::string::npos) {
    throw std::invalid_argument("gate plan is not canonical LF text");
  }
  plan.sha256 = native::sha256_hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(plan.bytes.data()),
      plan.bytes.size()));
  if (plan.sha256 != expected_sha256) {
    throw std::invalid_argument("gate plan SHA-256 mismatch");
  }

  std::map<std::string, std::string> metadata;
  bool saw_header = false;
  std::set<std::tuple<std::string, std::string, std::size_t>> source_keys;
  std::set<std::pair<std::string, std::string>> source_games;
  std::set<std::string> canonical_keys;
  for (const std::string_view line : split_exact(plan.bytes, '\n')) {
    if (line.empty()) continue;
    if (!saw_header && line.starts_with("# ")) {
      const std::size_t equals = line.find('=');
      if (equals <= 2U || equals == std::string_view::npos ||
          line.find('=', equals + 1U) != std::string_view::npos) {
        throw std::invalid_argument("gate plan metadata syntax is invalid");
      }
      const std::string key(line.substr(2U, equals - 2U));
      const std::string value(line.substr(equals + 1U));
      if (!safe_identifier(key, 64U) || value.empty() || value.size() > 128U ||
          !metadata.emplace(key, value).second) {
        throw std::invalid_argument("gate plan metadata is unsafe");
      }
      continue;
    }
    if (!saw_header) {
      if (line != kLateGateHeader) {
        throw std::invalid_argument("gate plan TSV header is not exact");
      }
      saw_header = true;
      continue;
    }
    const auto fields = split_exact(line, '\t');
    if (fields.size() != 11U || plan.roots.size() >= kLateGateRoots) {
      throw std::invalid_argument("gate plan row schema is invalid");
    }
    LateGateRoot root;
    root.run_id = fields[0];
    root.population = fields[1];
    root.role = fields[2];
    root.collector_sha256 = fields[3];
    root.arena_manifest_sha256 = fields[4];
    root.game_id = fields[5];
    root.candidate_player =
        parse_decimal_exact<int>(fields[6], "candidate player");
    root.prefix_turn =
        parse_decimal_exact<std::size_t>(fields[7], "prefix turn");
    root.state_id = fields[8];
    root.canonical_key = fields[9];
    root.transcript = fields[10];
    const bool trap_role =
        root.role == "last-enemy-shell" ||
        root.role == "one-own-before-mate" ||
        root.role == "two-own-before-mate";
    if (!safe_identifier(root.run_id) ||
        (root.population != "trap" && root.population != "control") ||
        ((root.population == "trap") != trap_role) ||
        (root.population == "control" &&
         root.role != "matched-winning-control") ||
        !valid_sha256(root.collector_sha256) ||
        !valid_sha256(root.arena_manifest_sha256) || root.game_id.empty() ||
        !std::all_of(root.game_id.begin(), root.game_id.end(), [](char value) {
          return value >= '0' && value <= '9';
        }) ||
        (root.candidate_player != 0 && root.candidate_player != 1) ||
        root.prefix_turn == 0U || !late_gate_state_id(root.state_id) ||
        !late_gate_state_id(root.canonical_key) ||
        !source_keys.emplace(root.run_id, root.game_id, root.prefix_turn).second ||
        !source_games.emplace(root.run_id, root.game_id).second ||
        !canonical_keys.insert(root.canonical_key).second) {
      throw std::invalid_argument("gate plan row is unsafe or duplicated");
    }
    const auto actions = late_gate_actions(root.transcript);
    if (actions.size() != root.prefix_turn) {
      throw std::invalid_argument("gate plan transcript depth is stale");
    }
    root.state = ps::make_initial_state(restart_rules());
    for (const std::string &action : actions) {
      if (ps::is_terminal(root.state)) {
        throw std::invalid_argument("gate plan transcript continues after terminal");
      }
      native::apply_encoded_turn(root.state, action);
    }
    const auto active = active_features(root.state);
    const auto reflected = active_features(reflected_state(root.state));
    if (ps::is_terminal(root.state) ||
        (root.state.to_move == ps::Player::One ? 0 : 1) !=
            root.candidate_player ||
        feature_id(active) != root.state_id ||
        feature_id(reflected < active ? reflected : active) !=
            root.canonical_key) {
      throw std::invalid_argument("gate plan state replay identity is stale");
    }
    plan.roots.push_back(std::move(root));
  }

  const std::set<std::string> expected_metadata{
      "panel_sha256", "schema", "selection", "source_sha256"};
  std::set<std::string> actual_metadata;
  for (const auto &[key, value] : metadata) {
    (void)value;
    actual_metadata.insert(key);
  }
  if (!saw_header || actual_metadata != expected_metadata ||
      metadata["schema"] != kLateGatePlanSchema ||
      metadata["selection"] != kLateGateSelection ||
      metadata["panel_sha256"] != kLateGatePanelSha256 ||
      !valid_sha256(metadata["source_sha256"]) ||
      plan.roots.size() != kLateGateRoots) {
    throw std::invalid_argument("gate plan metadata/count contract is invalid");
  }
  plan.panel_sha256 = metadata["panel_sha256"];
  plan.source_sha256 = metadata["source_sha256"];

  std::set<std::string> runs;
  std::map<std::tuple<std::string, std::string, int>, std::size_t> groups;
  std::map<std::string, std::size_t> populations;
  std::array<std::size_t, 2> colors{};
  for (const LateGateRoot &root : plan.roots) {
    runs.insert(root.run_id);
    ++groups[{root.run_id, root.population, root.candidate_player}];
    ++populations[root.population];
    ++colors[static_cast<std::size_t>(root.candidate_player)];
  }
  if (runs.size() != kLateGateRuns || groups.size() != kLateGateRuns * 4U ||
      populations != std::map<std::string, std::size_t>{{"control", 32U},
                                                       {"trap", 32U}} ||
      colors != std::array<std::size_t, 2>{32U, 32U} ||
      source_games.size() != kLateGateRoots) {
    throw std::invalid_argument(
        "gate plan is not the exact 64-root held-out balance");
  }
  if (!std::is_sorted(plan.roots.begin(), plan.roots.end(),
                      [](const LateGateRoot &left, const LateGateRoot &right) {
    return std::tie(left.run_id, left.population, left.candidate_player,
                    left.canonical_key, left.game_id, left.prefix_turn) <
           std::tie(right.run_id, right.population, right.candidate_player,
                    right.canonical_key, right.game_id, right.prefix_turn);
  })) {
    throw std::invalid_argument("gate plan row ordering is not canonical");
  }
  return plan;
}

LateGateArguments parse_late_gate_arguments(int argc, char **argv) {
  LateGateArguments result;
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("every late-pacing gate option needs a value");
    }
    const std::string_view option = argv[index];
    const std::string value = argv[++index];
    if (option == "--plan") result.plan = value;
    else if (option == "--plan-sha256") result.plan_sha256 = value;
    else if (option == "--candidate-checkpoint") {
      result.candidate_checkpoint = value;
    } else if (option == "--candidate-artifact-sha256") {
      result.candidate_artifact_sha256 = value;
    } else if (option == "--baseline-checkpoint") {
      result.baseline_checkpoint = value;
    } else if (option == "--baseline-artifact-sha256") {
      result.baseline_artifact_sha256 = value;
    } else {
      throw std::invalid_argument("unknown late-pacing gate option " +
                                  std::string(option));
    }
  }
  if (result.plan.empty() || !valid_sha256(result.plan_sha256) ||
      result.candidate_checkpoint.empty() ||
      !valid_sha256(result.candidate_artifact_sha256) ||
      result.baseline_checkpoint.empty() ||
      !valid_sha256(result.baseline_artifact_sha256) ||
      result.candidate_artifact_sha256 == result.baseline_artifact_sha256) {
    throw std::invalid_argument("late-pacing gate inputs are incomplete");
  }
  return result;
}

int play_late_gate_game(const LateGateRoot &root, std::size_t root_index,
                        int candidate_orientation,
                        const LoadedModel &candidate,
                        const LoadedModel &baseline,
                        LateGateSummary &summary) {
  ps::GameState state = root.state;
  const LoadedModel &player_one =
      candidate_orientation == 0 ? candidate : baseline;
  const LoadedModel &player_two =
      candidate_orientation == 1 ? candidate : baseline;
  const std::size_t game = root_index * 2U +
                           static_cast<std::size_t>(candidate_orientation);
  const std::uint64_t game_seed =
      kLateGateSeed + game * 0x9e3779b97f4a7c15ULL;
  SplitMix64 random(game_seed ^ 0xd1b54a32d192ed03ULL);
  std::size_t generated = 0U;
  for (; generated < kLateGateMaximumGeneratedTurns && !ps::is_terminal(state);
       ++generated) {
    const LoadedModel &turn_model =
        state.to_move == ps::Player::One ? player_one : player_two;
    const native::SearchResult search = native::choose_complete_turn(
        state, search_config(game_seed ^ random.next(), turn_model.model.get(),
                             kLateGateWork));
    ++summary.searches;
    summary.expansions += search.stats.expansions;
    summary.maximum_tree = std::max(summary.maximum_tree,
                                    search.stats.tree_nodes);
    const std::string action =
        generated < kLateGateTemperatureTurns
            ? sampled_action(search, kLateGateTemperature, random)
            : search.encoded;
    if (action.empty()) {
      ++summary.operational_failures;
      return -1;
    }
    native::apply_encoded_turn(state, action);
  }
  if (!ps::is_terminal(state) || !ps::winner(state).has_value()) return -1;
  return *ps::winner(state) == ps::Player::One ? 0 : 1;
}

int run_late_gate(const LateGateArguments &arguments) {
  const LateGatePlan plan =
      read_late_gate_plan(arguments.plan, arguments.plan_sha256);
  if (late_gate_file_sha256(arguments.candidate_checkpoint) !=
          arguments.candidate_artifact_sha256 ||
      late_gate_file_sha256(arguments.baseline_checkpoint) !=
          arguments.baseline_artifact_sha256) {
    throw std::invalid_argument("runtime checkpoint artifact SHA-256 mismatch");
  }
  const LoadedModel candidate = load_checkpoint(
      arguments.candidate_checkpoint, arguments.candidate_artifact_sha256);
  const LoadedModel baseline = load_checkpoint(
      arguments.baseline_checkpoint, arguments.baseline_artifact_sha256);
  if (candidate.packed_sha256 == baseline.packed_sha256) {
    throw std::invalid_argument("late-pacing gate checkpoints are identical");
  }
  std::cout << "candidate_runtime_sha256=" << candidate.artifact_sha256
            << " candidate_model_sha256=" << candidate.model_sha256
            << " candidate_packed_sha256=" << candidate.packed_sha256 << '\n';
  std::cout << "baseline_runtime_sha256=" << baseline.artifact_sha256
            << " baseline_model_sha256=" << baseline.model_sha256
            << " baseline_packed_sha256=" << baseline.packed_sha256 << '\n';
  std::cout << "plan_sha256=" << plan.sha256
            << " panel_sha256=" << plan.panel_sha256
            << " source_sha256=" << plan.source_sha256 << '\n';

  LateGateSummary summary;
  for (std::size_t root_index = 0; root_index < plan.roots.size(); ++root_index) {
    const LateGateRoot &root = plan.roots[root_index];
    for (int orientation = 0; orientation < 2; ++orientation) {
      int winner = -1;
      try {
        winner = play_late_gate_game(root, root_index, orientation,
                                     candidate, baseline, summary);
      } catch (const std::exception &) {
        ++summary.operational_failures;
      }
      const bool candidate_won = winner == orientation;
      if (winner < 0) {
        ++summary.unfinished;
      } else if (candidate_won) {
        ++summary.candidate_wins;
        ++summary.orientation_wins[static_cast<std::size_t>(orientation)];
        ++summary.population_wins[root.population == "trap" ? 0U : 1U];
      } else {
        ++summary.baseline_wins;
      }
      std::cout << "game=" << (root_index * 2U + orientation)
                << " root=" << root_index
                << " run=" << root.run_id
                << " population=" << root.population
                << " source_color=" << root.candidate_player
                << " candidate_orientation=" << orientation
                << " winner=" << winner << '\n';
    }
  }
  const bool passed =
      summary.candidate_wins >= kLateGateRequiredWins &&
      summary.orientation_wins[0] >= kLateGateRequiredPerOrientation &&
      summary.orientation_wins[1] >= kLateGateRequiredPerOrientation &&
      summary.unfinished == 0U && summary.operational_failures == 0U &&
      summary.candidate_wins + summary.baseline_wins == kLateGateGames;
  std::cout << "summary candidate=" << summary.candidate_wins
            << " baseline=" << summary.baseline_wins
            << " unfinished=" << summary.unfinished
            << " operational_failures=" << summary.operational_failures
            << " candidate_player_one=" << summary.orientation_wins[0]
            << " candidate_player_two=" << summary.orientation_wins[1]
            << " trap_candidate=" << summary.population_wins[0]
            << " control_candidate=" << summary.population_wins[1]
            << " games=" << kLateGateGames
            << " searches=" << summary.searches
            << " expansions=" << summary.expansions
            << " maximum_tree=" << summary.maximum_tree
            << " work=" << kLateGateWork
            << " seed=" << kLateGateSeed
            << " temperature=3"
            << " temperature_turns=" << kLateGateTemperatureTurns
            << " maximum_generated_turns=" << kLateGateMaximumGeneratedTurns
            << " required_total=" << kLateGateRequiredWins
            << " required_per_orientation="
            << kLateGateRequiredPerOrientation
            << " passed=" << (passed ? "true" : "false") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

#ifndef PAPER_SOCCER_JACEK_NATIVE_LATE_PACING_GAME_GATE_NO_MAIN
int main(int argc, char **argv) {
  try {
    return run_late_gate(parse_late_gate_arguments(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << "jacek_native_late_pacing_game_gate failure: "
              << error.what() << '\n';
    return 1;
  }
}
#endif
