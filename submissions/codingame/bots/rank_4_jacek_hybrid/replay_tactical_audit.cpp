#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING
#include "bot.cpp"
#undef PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

constexpr std::size_t kExpectedReplayPaths = 24;
constexpr std::size_t kExpectedEligibleEntries = 512;
constexpr std::size_t kExpectedUniqueActivations = 511;
constexpr std::uint64_t kDefaultStateCap = 2'000'000;
constexpr std::uint64_t kDefaultGlobalStateCap = 100'000'000;

enum class TacticalCategory : int {
  ImmediateLoss = -2,
  OpponentComponentWin = -1,
  OpponentComponentUnknown = 0,
  OpponentComponentLoss = 1,
  ImmediateWin = 2,
};

struct Options {
  bool full{};
  std::uint64_t state_cap{kDefaultStateCap};
  std::uint64_t global_state_cap{kDefaultGlobalStateCap};
  std::size_t limit{kExpectedUniqueActivations};
};

struct Activation {
  int player_id{};
  std::string prefix;
  std::string correction;
};

struct Enumeration {
  std::uint64_t expanded_states{};
  std::uint64_t completed_actions{};
  int maximum{static_cast<int>(TacticalCategory::ImmediateLoss)};
  bool stopped_at_exact_upper_bound{};
};

class StateCapReached final : public std::exception {};

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

ps::RulesConfig codingame_rules() {
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

std::vector<std::string_view> split_transcript(std::string_view transcript) {
  std::vector<std::string_view> turns;
  if (transcript.empty()) {
    return turns;
  }
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    require(end > begin, "replay transcript contains an empty turn");
    turns.push_back(transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  return turns;
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  for (const std::string_view action : split_transcript(transcript)) {
    cg::apply_encoded_turn(state, action);
  }
  return state;
}

std::uint64_t stable_prefix_id(int player_id, std::string_view prefix) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash ^= static_cast<std::uint64_t>(player_id);
  hash *= 1099511628211ULL;
  for (const unsigned char character : prefix) {
    hash ^= character;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::vector<Activation> collect_activations() {
  require(cg::replay_book::kReplays.size() == kExpectedReplayPaths,
          "tracked replay path count changed");
  std::map<std::pair<int, std::string>, std::string> unique;
  std::size_t eligible = 0;
  std::size_t duplicates = 0;

  for (const cg::replay_book::Replay &replay : cg::replay_book::kReplays) {
    const std::vector<std::string_view> turns =
        split_transcript(replay.transcript);
    ps::GameState state = ps::make_initial_state(codingame_rules());
    std::string prefix;
    for (std::size_t turn = 0; turn < turns.size(); ++turn) {
      const std::string_view recorded_action = turns[turn];
      if (turn >= replay.first_turn &&
          turn % 2U == static_cast<std::size_t>(replay.player_id)) {
        ++eligible;
        require(!ps::is_terminal(state),
                "eligible replay activation is already terminal");
        require(state.to_move == cg::player_for_id(replay.player_id),
                "eligible replay activation has the wrong mover");

        std::string looked_up;
        require(cg::lookup_replay_correction(replay.player_id, prefix,
                                             looked_up),
                "eligible replay activation did not look up");
        require(looked_up == recorded_action,
                "replay lookup action differs from its tracked continuation");

        ps::GameState corrected = state;
        std::string applied = "sentinel";
        require(cg::try_replay_correction(corrected, replay.player_id, prefix,
                                          applied),
                "eligible replay correction did not apply legally");
        ps::GameState expected = state;
        cg::apply_encoded_turn(expected, recorded_action);
        require(applied == recorded_action && same_state(corrected, expected),
                "replay correction was not atomic and exact");

        const auto [iterator, inserted] = unique.emplace(
            std::make_pair(static_cast<int>(replay.player_id), prefix),
            std::string(recorded_action));
        if (!inserted) {
          ++duplicates;
          require(iterator->second == recorded_action,
                  "duplicate replay activation has conflicting corrections");
        }
      }

      cg::apply_encoded_turn(state, recorded_action);
      if (!prefix.empty()) {
        prefix.push_back('/');
      }
      prefix.append(recorded_action);
    }
  }

  require(eligible == kExpectedEligibleEntries,
          "eligible replay decision count changed");
  require(unique.size() == kExpectedUniqueActivations,
          "unique replay activation count changed");
  require(duplicates == eligible - unique.size() && duplicates == 1,
          "replay activation duplicate accounting changed");

  std::vector<Activation> activations;
  activations.reserve(unique.size());
  for (const auto &[key, action] : unique) {
    activations.push_back(Activation{key.first, key.second, action});
  }
  return activations;
}

cg::ReboundOutcome exact_root_component(const ps::GameState &state) {
  require(!ps::is_terminal(state),
          "cannot classify a terminal root component");
  cg::SearchConfig config;
  config.max_nodes = 1;
  config.transposition_entries = 1;
  config.evaluation_entries = 1;
  config.exact_proof_mask = cg::kExactProofRootGoal;
  cg::CompleteTurnSearch search(state, config);
  std::vector<ps::Move> unused;
  return search.analyze_rebound_component_for_test(false, unused);
}

int classify_completed_action(const ps::GameState &after, ps::Player mover) {
  if (ps::is_terminal(after)) {
    const std::optional<ps::Player> winning_player = ps::winner(after);
    require(winning_player.has_value(),
            "terminal replay successor has no winner");
    return *winning_player == mover
               ? static_cast<int>(TacticalCategory::ImmediateWin)
               : static_cast<int>(TacticalCategory::ImmediateLoss);
  }
  require(after.to_move == ps::opponent(mover),
          "complete action did not hand off to the opponent");
  switch (exact_root_component(after)) {
    case cg::ReboundOutcome::Loss:
      return static_cast<int>(TacticalCategory::OpponentComponentLoss);
    case cg::ReboundOutcome::Unknown:
      return static_cast<int>(TacticalCategory::OpponentComponentUnknown);
    case cg::ReboundOutcome::Win:
      return static_cast<int>(TacticalCategory::OpponentComponentWin);
  }
  throw std::logic_error("unreachable rebound component outcome");
}

int classify_correction(const ps::GameState &state,
                        std::string_view correction) {
  const ps::Player mover = state.to_move;
  ps::GameState after = state;
  cg::apply_encoded_turn(after, correction);
  return classify_completed_action(after, mover);
}

bool enumerate_complete_actions(const ps::GameState &state, ps::Player mover,
                                std::uint64_t state_cap,
                                std::uint64_t global_state_cap,
                                std::uint64_t &global_states,
                                Enumeration &enumeration) {
  const std::vector<ps::Move> moves = ps::legal_moves(state);
  require(!moves.empty(), "in-progress continuation has no legal move");
  for (const ps::Move move : moves) {
    if (enumeration.expanded_states >= state_cap ||
        global_states >= global_state_cap) {
      throw StateCapReached{};
    }
    ++enumeration.expanded_states;
    ++global_states;
    const ps::GameState child = ps::apply_move(state, move);
    if (ps::is_terminal(child) || child.to_move != mover) {
      ++enumeration.completed_actions;
      enumeration.maximum =
          std::max(enumeration.maximum, classify_completed_action(child, mover));
      // A root Unknown result has already excluded category +2. Category +1
      // is therefore the exact upper bound and safely ends enumeration.
      if (enumeration.maximum ==
          static_cast<int>(TacticalCategory::OpponentComponentLoss)) {
        enumeration.stopped_at_exact_upper_bound = true;
        return true;
      }
    } else if (enumerate_complete_actions(
                   child, mover, state_cap, global_state_cap, global_states,
                   enumeration)) {
      return true;
    }
  }
  return false;
}

int exact_maximum_category(const ps::GameState &state, int correction_category,
                           const Options &options,
                           std::uint64_t &global_states,
                           Enumeration &enumeration) {
  switch (exact_root_component(state)) {
    case cg::ReboundOutcome::Win:
      return static_cast<int>(TacticalCategory::ImmediateWin);
    case cg::ReboundOutcome::Loss:
      return static_cast<int>(TacticalCategory::ImmediateLoss);
    case cg::ReboundOutcome::Unknown:
      break;
  }

  // Root Unknown excludes both immediate categories and proves at least one
  // nonterminal handoff. +1 is consequently the exact maximum if the tracked
  // correction already has that category.
  if (correction_category ==
      static_cast<int>(TacticalCategory::OpponentComponentLoss)) {
    enumeration.maximum = correction_category;
    enumeration.stopped_at_exact_upper_bound = true;
    return correction_category;
  }
  enumeration.maximum = static_cast<int>(TacticalCategory::ImmediateLoss);
  (void)enumerate_complete_actions(state, state.to_move, options.state_cap,
                                   options.global_state_cap, global_states,
                                   enumeration);
  require(enumeration.completed_actions > 0,
          "full enumeration found no complete legal action");
  require(enumeration.maximum >=
              static_cast<int>(TacticalCategory::OpponentComponentWin) &&
              enumeration.maximum <=
                  static_cast<int>(TacticalCategory::OpponentComponentLoss),
          "root-Unknown enumeration produced an impossible maximum");
  return enumeration.maximum;
}

std::uint64_t parse_positive(std::string_view text, std::string_view option) {
  std::uint64_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
    throw std::invalid_argument(std::string(option) + " requires a positive integer");
  }
  return value;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--cheap") {
      options.full = false;
    } else if (argument == "--full") {
      options.full = true;
    } else if (argument == "--state-cap" ||
               argument == "--global-state-cap" || argument == "--limit") {
      if (++index >= argc) {
        throw std::invalid_argument(std::string(argument) + " requires a value");
      }
      const std::uint64_t value = parse_positive(argv[index], argument);
      if (argument == "--state-cap") {
        options.state_cap = value;
      } else if (argument == "--global-state-cap") {
        options.global_state_cap = value;
      } else {
        if (value > kExpectedUniqueActivations) {
          throw std::invalid_argument("--limit exceeds the frozen activation count");
        }
        options.limit = static_cast<std::size_t>(value);
      }
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(argument));
    }
  }
  return options;
}

int run(const Options &options) {
  const std::vector<Activation> activations = collect_activations();
  std::array<std::uint64_t, 5> correction_categories{};
  std::array<std::uint64_t, 5> maximum_categories{};
  std::uint64_t exact_maxima = 0;
  std::uint64_t unresolved = 0;
  std::uint64_t capped = 0;
  std::uint64_t displaced = 0;
  std::uint64_t global_states = 0;
  std::uint64_t maximum_prefix_states = 0;
  std::uint64_t completed_actions = 0;

  for (std::size_t index = 0; index < options.limit; ++index) {
    const Activation &activation = activations[index];
    const ps::GameState state = reconstruct(activation.prefix);
    require(state.to_move == cg::player_for_id(activation.player_id),
            "reconstructed activation has the wrong mover");
    std::string looked_up;
    require(cg::lookup_replay_correction(activation.player_id,
                                         activation.prefix, looked_up) &&
                looked_up == activation.correction,
            "unique activation no longer binds its correction");
    const int correction_category =
        classify_correction(state, activation.correction);
    ++correction_categories[static_cast<std::size_t>(correction_category + 2)];

    if (!options.full) {
      const cg::ReboundOutcome root = exact_root_component(state);
      if (root == cg::ReboundOutcome::Win ||
          root == cg::ReboundOutcome::Loss ||
          correction_category ==
              static_cast<int>(TacticalCategory::OpponentComponentLoss)) {
        ++exact_maxima;
      } else {
        ++unresolved;
      }
      continue;
    }

    Enumeration enumeration;
    try {
      const int maximum = exact_maximum_category(
          state, correction_category, options, global_states, enumeration);
      ++exact_maxima;
      ++maximum_categories[static_cast<std::size_t>(maximum + 2)];
      maximum_prefix_states =
          std::max(maximum_prefix_states, enumeration.expanded_states);
      completed_actions += enumeration.completed_actions;
      if (correction_category != maximum) {
        ++displaced;
        std::cerr << "displaced index=" << index
                  << " player=" << activation.player_id
                  << " prefix_id="
                  << stable_prefix_id(activation.player_id, activation.prefix)
                  << " prefix_turns=" << split_transcript(activation.prefix).size()
                  << " correction=" << activation.correction
                  << " category=" << correction_category
                  << " maximum=" << maximum
                  << " expanded_states=" << enumeration.expanded_states
                  << " completed_actions=" << enumeration.completed_actions
                  << '\n';
      }
    } catch (const StateCapReached &) {
      ++capped;
      std::cerr << "capped index=" << index
                << " player=" << activation.player_id
                << " prefix_id="
                << stable_prefix_id(activation.player_id, activation.prefix)
                << " prefix_turns=" << split_transcript(activation.prefix).size()
                << " expanded_states=" << enumeration.expanded_states
                << " completed_actions=" << enumeration.completed_actions
                << '\n';
    }
  }

  std::cout << "replay_tactical_audit mode="
            << (options.full ? "full" : "cheap")
            << " replay_paths=" << cg::replay_book::kReplays.size()
            << " eligible_entries=" << kExpectedEligibleEntries
            << " unique_activations=" << activations.size()
            << " audited=" << options.limit
            << " exact_maxima=" << exact_maxima
            << " unresolved=" << unresolved
            << " capped=" << capped
            << " displaced=" << displaced
            << " correction_categories=";
  for (std::size_t index = 0; index < correction_categories.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << static_cast<int>(index) - 2 << ':'
              << correction_categories[index];
  }
  std::cout << " maximum_categories=";
  for (std::size_t index = 0; index < maximum_categories.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << static_cast<int>(index) - 2 << ':' << maximum_categories[index];
  }
  std::cout << " expanded_states=" << global_states
            << " maximum_prefix_states=" << maximum_prefix_states
            << " completed_actions=" << completed_actions
            << " state_cap=" << options.state_cap
            << " global_state_cap=" << options.global_state_cap << '\n';

  if (capped != 0) {
    return 2;
  }
  if (displaced != 0) {
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << "replay_tactical_audit error=" << error.what() << '\n';
    return 3;
  }
}
