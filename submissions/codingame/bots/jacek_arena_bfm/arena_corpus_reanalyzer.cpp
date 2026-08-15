#include "engine.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace jab = jacek_arena_bfm;

namespace {

constexpr std::string_view kInputHeader =
    "game_id\tcandidate_player\twinner\tturns";
constexpr std::string_view kOutputSchema =
    "papersoccer.jacek-arena-bfm.reanalysis.v1";
constexpr std::size_t kWork30k = 30'000;
constexpr std::size_t kWork100k = 100'000;
constexpr std::size_t kMaximumGames = 4096;
constexpr std::size_t kMaximumTranscriptBytes = 65'536;

struct Config {
  std::string input_path;
  std::optional<std::string> ranking_game_ids_path;
  std::set<std::uint64_t> ranking_game_ids;
  std::size_t maximum_analyzed_decisions{2};
  std::size_t maximum_alternatives{6};
  std::size_t maximum_pairs_per_decision{4};
  std::size_t maximum_pairs_per_game{16};
  std::size_t search_width{8};
  jab::GeneratorStrategy generator{jab::GeneratorStrategy::TacticalProgressive};
};

struct Game {
  std::uint64_t game_id{};
  int candidate_player{};
  int winner{};
  std::vector<std::string> turns;
};

struct ParsedInput {
  std::map<std::string, std::string> metadata;
  std::vector<Game> games;
};

struct WorkLimit final {};

struct SearchContext {
  std::size_t limit{};
  std::size_t used{};
  std::size_t width{};
};

struct WorkValue {
  float value{};
  std::size_t work{};
  int completed_depth{};
};

struct Alternative {
  jab::Action action{};
  jab::State successor{};
  float initial_value{};
};

struct Decision {
  jab::State state{};
  jab::Action observed{};
  jab::State observed_successor{};
  std::vector<Alternative> alternatives;
  std::size_t turn_index{};
  int actor{};
  float difficulty{};
  bool has_exact_pair{};
};

struct AuditStats {
  std::size_t value_rows{};
  std::size_t pair_rows{};
  std::size_t decisions_analyzed{};
  std::size_t alternatives_analyzed{};
  std::size_t unstable_ordering_rejections{};
  std::size_t proved_losing_vs_winning_rejections{};
  std::size_t cap_rejections{};
  std::size_t exact_pairs{};
};

template <typename Integer>
Integer parse_integer(std::string_view raw, std::string_view field) {
  Integer value{};
  const char *begin = raw.data();
  const char *end = begin + raw.size();
  const auto [parsed, error] = std::from_chars(begin, end, value);
  if (error != std::errc{} || parsed != end) {
    throw std::runtime_error(std::string(field) + " is not an integer");
  }
  return value;
}

std::vector<std::string_view> split(std::string_view value, char separator) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = value.find(separator, begin);
    result.push_back(value.substr(
        begin, end == std::string_view::npos ? value.size() - begin
                                             : end - begin));
    if (end == std::string_view::npos) return result;
    begin = end + 1;
  }
}

std::string trim_cr(std::string value) {
  if (!value.empty() && value.back() == '\r') value.pop_back();
  return value;
}

ParsedInput read_input(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open auditor TSV: " + path);
  ParsedInput parsed;
  std::string line;
  bool header_seen = false;
  std::uint64_t previous_game_id = 0;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = trim_cr(std::move(line));
    if (!header_seen && line.starts_with("# ")) {
      const auto equal = line.find('=', 2);
      if (equal == std::string::npos || equal == 2 || equal + 1 == line.size()) {
        throw std::runtime_error("malformed auditor metadata at line " +
                                 std::to_string(line_number));
      }
      const std::string key = line.substr(2, equal - 2);
      const std::string value = line.substr(equal + 1);
      if (!parsed.metadata.emplace(key, value).second) {
        throw std::runtime_error("duplicate auditor metadata key: " + key);
      }
      continue;
    }
    if (!header_seen) {
      if (line != kInputHeader) {
        throw std::runtime_error("auditor TSV has a non-canonical header");
      }
      header_seen = true;
      continue;
    }
    if (line.empty() || line.starts_with('#')) {
      throw std::runtime_error("blank/comment line after auditor TSV header");
    }
    const auto fields = split(line, '\t');
    if (fields.size() != 4) {
      throw std::runtime_error("auditor TSV row must have exactly four fields");
    }
    Game game;
    game.game_id = parse_integer<std::uint64_t>(fields[0], "game_id");
    game.candidate_player = parse_integer<int>(fields[1], "candidate_player");
    game.winner = parse_integer<int>(fields[2], "winner");
    if (game.game_id == 0 || game.game_id <= previous_game_id) {
      throw std::runtime_error("auditor game IDs must be positive, unique, and sorted");
    }
    previous_game_id = game.game_id;
    if ((game.candidate_player != 0 && game.candidate_player != 1) ||
        (game.winner != 0 && game.winner != 1)) {
      throw std::runtime_error("candidate_player and winner must be zero or one");
    }
    if (fields[3].empty() || fields[3].size() > kMaximumTranscriptBytes) {
      throw std::runtime_error("auditor transcript is empty or too large");
    }
    for (const auto turn : split(fields[3], '/')) {
      if (turn.empty() || turn.size() > jab::kMaximumActionLength ||
          std::any_of(turn.begin(), turn.end(),
                      [](char value) { return value < '0' || value > '7'; })) {
        throw std::runtime_error("auditor transcript contains an invalid complete action");
      }
      game.turns.emplace_back(turn);
    }
    parsed.games.push_back(std::move(game));
    if (parsed.games.size() > kMaximumGames) {
      throw std::runtime_error("auditor TSV exceeds the game limit");
    }
  }
  if (!header_seen || parsed.games.empty()) {
    throw std::runtime_error("auditor TSV has no games");
  }
  return parsed;
}

jab::Action parse_action(std::string_view text) {
  jab::Action action;
  if (text.empty() || text.size() > action.directions.size()) {
    throw std::runtime_error("complete action has invalid length");
  }
  action.length = static_cast<std::uint16_t>(text.size());
  for (std::size_t index = 0; index < text.size(); ++index) {
    if (text[index] < '0' || text[index] > '7') {
      throw std::runtime_error("complete action contains an invalid direction");
    }
    action.directions[index] = static_cast<std::uint8_t>(text[index] - '0');
  }
  return action;
}

float value_for_player(const jab::State &state, int player) {
  if (state.terminal()) return state.winner == player ? 1.0F : -1.0F;
  const float mover_value = jab::evaluate(state);
  return state.to_move == player ? mover_value : -mover_value;
}

float counted_value(const jab::State &state, int player,
                    SearchContext &context) {
  if (context.used >= context.limit) throw WorkLimit{};
  ++context.used;
  return value_for_player(state, player);
}

struct OrderedChild {
  jab::State state{};
  float ordering_value{};
  std::string action;
};

float minimax(const jab::State &state, int root_player, int depth,
              float alpha, float beta, SearchContext &context) {
  if (state.terminal() || depth == 0) {
    return counted_value(state, root_player, context);
  }
  auto actions = jab::generate_actions(
      state, jab::GeneratorStrategy::PriorityBeam, false);
  if (actions.empty()) return counted_value(state, root_player, context);

  const std::size_t ordering_pool = std::min<std::size_t>(
      actions.size(), std::max<std::size_t>(context.width, context.width * 2));
  std::vector<OrderedChild> children;
  children.reserve(ordering_pool);
  for (std::size_t index = 0; index < ordering_pool; ++index) {
    jab::State successor = state;
    if (!jab::apply_action(successor, actions[index])) continue;
    children.push_back(OrderedChild{
        successor, counted_value(successor, root_player, context),
        actions[index].text()});
  }
  if (children.empty()) return counted_value(state, root_player, context);
  const bool maximize = state.to_move == root_player;
  std::sort(children.begin(), children.end(),
            [maximize](const OrderedChild &left, const OrderedChild &right) {
              if (left.ordering_value != right.ordering_value) {
                return maximize ? left.ordering_value > right.ordering_value
                                : left.ordering_value < right.ordering_value;
              }
              return left.action < right.action;
            });
  if (children.size() > context.width) children.resize(context.width);

  float best = maximize ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();
  for (const auto &child : children) {
    const float value = depth == 1
        ? child.ordering_value
        : minimax(child.state, root_player, depth - 1,
                  alpha, beta, context);
    if (maximize) {
      best = std::max(best, value);
      alpha = std::max(alpha, best);
    } else {
      best = std::min(best, value);
      beta = std::min(beta, best);
    }
    if (alpha >= beta) break;
  }
  return best;
}

WorkValue fixed_work_value(const jab::State &state, int root_player,
                           std::size_t limit, std::size_t width) {
  if (state.terminal()) return WorkValue{value_for_player(state, root_player), 0, 0};
  SearchContext context{limit, 0, width};
  float last = counted_value(state, root_player, context);
  int completed_depth = 0;
  for (int depth = 1; depth <= 64; ++depth) {
    try {
      const float value = minimax(
          state, root_player, depth, -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), context);
      last = value;
      completed_depth = depth;
    } catch (const WorkLimit &) {
      break;
    }
  }
  return WorkValue{std::clamp(last, -1.0F, 1.0F), context.used,
                   completed_depth};
}

std::vector<unsigned> active_feature_indices(const jab::State &state) {
  std::size_t count = 0;
  const auto sparse = jab::active_features(state, count);
  std::vector<unsigned> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    if (sparse[index] >= jab::kFeatureCount) {
      throw std::runtime_error("engine emitted an out-of-range feature");
    }
    result.push_back(sparse[index]);
  }
  std::sort(result.begin(), result.end());
  if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
    throw std::runtime_error("engine emitted duplicate active features");
  }
  return result;
}

void write_feature_array(std::ostream &output,
                         const std::vector<unsigned> &features) {
  output << '[';
  for (std::size_t index = 0; index < features.size(); ++index) {
    if (index) output << ',';
    output << features[index];
  }
  output << ']';
}

std::string position_id(const jab::State &state) {
  std::ostringstream output;
  output << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
         << jab::state_hash(state);
  return output.str();
}

std::vector<Alternative> select_alternatives(
    const jab::State &state, const jab::Action &observed,
    const jab::State &observed_successor, const Config &config) {
  auto actions = jab::generate_actions(state, config.generator, true);
  std::vector<Alternative> all;
  all.reserve(actions.size());
  const int actor = state.to_move;
  for (const auto &action : actions) {
    if (action == observed) continue;
    jab::State successor = state;
    if (!jab::apply_action(successor, action)) {
      throw std::runtime_error("generator emitted an illegal/incomplete action");
    }
    all.push_back(Alternative{action, successor,
                              value_for_player(successor, actor)});
  }
  std::sort(all.begin(), all.end(), [](const Alternative &left,
                                       const Alternative &right) {
    if (left.initial_value != right.initial_value) {
      return left.initial_value > right.initial_value;
    }
    return left.action.text() < right.action.text();
  });
  if (all.size() <= config.maximum_alternatives) return all;

  // Preserve the strongest counterfactuals (the hardest ordering test) and a
  // small tail sample (tactical/own-goal witnesses otherwise hidden by a
  // neural pre-order).  Every retained ordering still faces both work gates.
  std::vector<Alternative> selected;
  const std::size_t tail = std::min<std::size_t>(2, config.maximum_alternatives / 3);
  const std::size_t head = config.maximum_alternatives - tail;
  selected.insert(selected.end(), all.begin(), all.begin() + head);
  selected.insert(selected.end(), all.end() - tail, all.end());
  (void)observed_successor;
  return selected;
}

std::string value_record(const Game &game, std::size_t turn_index,
                         const jab::State &state) {
  std::ostringstream output;
  output << std::setprecision(9)
         << "{\"schema\":\"" << kOutputSchema << "\",\"kind\":\"value\""
         << ",\"game_id\":" << game.game_id
         << ",\"turn_index\":" << turn_index
         << ",\"actor\":" << static_cast<int>(state.to_move)
         << ",\"candidate_player\":" << game.candidate_player
         << ",\"winner\":" << game.winner
         << ",\"position_id\":\"" << position_id(state) << "\""
         << ",\"target\":" << (state.to_move == game.winner ? 1 : -1)
         << ",\"active_features\":";
  write_feature_array(output, active_feature_indices(state));
  output << "}\n";
  return output.str();
}

std::string pair_record(
    const Game &game, const Decision &decision, std::size_t pair_index,
    const Alternative &alternative, bool exact,
    const WorkValue &observed30, const WorkValue &alternative30,
    const WorkValue &observed100, const WorkValue &alternative100) {
  std::ostringstream output;
  output << std::setprecision(9)
         << "{\"schema\":\"" << kOutputSchema << "\",\"kind\":\"pairwise\""
         << ",\"game_id\":" << game.game_id
         << ",\"turn_index\":" << decision.turn_index
         << ",\"actor\":" << decision.actor
         << ",\"candidate_player\":" << game.candidate_player
         << ",\"winner\":" << game.winner
         << ",\"decision_id\":\"" << position_id(decision.state) << "\""
         << ",\"pair_index\":" << pair_index
         << ",\"observed_action\":\"" << decision.observed.text() << "\""
         << ",\"inferior_action\":\"" << alternative.action.text() << "\""
         << ",\"exact\":" << (exact ? "true" : "false")
         << ",\"preferred_terminal\":"
         << (decision.observed_successor.terminal() ? "true" : "false")
         << ",\"inferior_terminal\":"
         << (alternative.successor.terminal() ? "true" : "false")
         << ",\"preferred_value_30000\":" << observed30.value
         << ",\"inferior_value_30000\":" << alternative30.value
         << ",\"preferred_value_100000\":" << observed100.value
         << ",\"inferior_value_100000\":" << alternative100.value
         << ",\"preferred_work_30000\":" << observed30.work
         << ",\"inferior_work_30000\":" << alternative30.work
         << ",\"preferred_work_100000\":" << observed100.work
         << ",\"inferior_work_100000\":" << alternative100.work
         << ",\"preferred_depth_30000\":" << observed30.completed_depth
         << ",\"inferior_depth_30000\":" << alternative30.completed_depth
         << ",\"preferred_depth_100000\":" << observed100.completed_depth
         << ",\"inferior_depth_100000\":" << alternative100.completed_depth
         << ",\"preferred_active_features\":";
  write_feature_array(output, active_feature_indices(decision.observed_successor));
  output << ",\"inferior_active_features\":";
  write_feature_array(output, active_feature_indices(alternative.successor));
  output << "}\n";
  return output.str();
}

std::string reanalyze_game(const Game &game, const Config &config,
                           AuditStats &stats) {
  jab::State state = jab::initial_state();
  std::ostringstream values;
  std::vector<Decision> decisions;
  for (std::size_t turn_index = 0; turn_index < game.turns.size(); ++turn_index) {
    if (state.terminal()) {
      throw std::runtime_error("game " + std::to_string(game.game_id) +
                               " continues after a rule terminal state");
    }
    values << value_record(game, turn_index, state);
    ++stats.value_rows;
    const jab::Action observed = parse_action(game.turns[turn_index]);
    jab::State successor = state;
    if (!jab::apply_action(successor, observed, true)) {
      throw std::runtime_error("game " + std::to_string(game.game_id) +
                               " has an illegal or incomplete action at turn " +
                               std::to_string(turn_index));
    }
    const int actor = state.to_move;
    const bool ranking_authorized = config.ranking_game_ids_path
        ? config.ranking_game_ids.contains(game.game_id) : true;
    if (ranking_authorized && config.maximum_analyzed_decisions > 0 &&
        actor != game.candidate_player) {
      auto alternatives = select_alternatives(
          state, observed, successor, config);
      if (!alternatives.empty()) {
        const float observed_initial = value_for_player(successor, actor);
        float closest = std::numeric_limits<float>::infinity();
        bool exact_pair = false;
        for (const auto &alternative : alternatives) {
          closest = std::min(closest,
                             std::abs(observed_initial - alternative.initial_value));
          exact_pair = exact_pair ||
              (successor.terminal() && alternative.successor.terminal() &&
               observed_initial > alternative.initial_value);
        }
        decisions.push_back(Decision{state, observed, successor,
                                     std::move(alternatives), turn_index,
                                     actor, closest, exact_pair});
      }
    }
    state = successor;
  }
  if (!state.terminal() || state.winner != game.winner) {
    throw std::runtime_error("game " + std::to_string(game.game_id) +
                             " is not terminal with its asserted winner");
  }

  std::sort(decisions.begin(), decisions.end(),
            [](const Decision &left, const Decision &right) {
              if (left.has_exact_pair != right.has_exact_pair) {
                return left.has_exact_pair > right.has_exact_pair;
              }
              if (left.difficulty != right.difficulty) {
                return left.difficulty < right.difficulty;
              }
              return left.turn_index < right.turn_index;
            });
  if (decisions.size() > config.maximum_analyzed_decisions) {
    decisions.resize(config.maximum_analyzed_decisions);
  }
  stats.decisions_analyzed += decisions.size();

  std::ostringstream pairs;
  std::size_t game_pairs = 0;
  for (const auto &decision : decisions) {
    if (game_pairs >= config.maximum_pairs_per_game) break;
    const int actor = decision.actor;
    std::optional<WorkValue> observed30;
    std::optional<WorkValue> observed100;
    struct AcceptedPair {
      const Alternative *alternative{};
      bool exact{};
      WorkValue observed30{};
      WorkValue alternative30{};
      WorkValue observed100{};
      WorkValue alternative100{};
    };
    std::vector<AcceptedPair> accepted;
    for (const auto &alternative : decision.alternatives) {
      ++stats.alternatives_analyzed;
      const float observed_exact = value_for_player(
          decision.observed_successor, actor);
      const float alternative_exact = value_for_player(
          alternative.successor, actor);
      const bool exact = decision.observed_successor.terminal() &&
                         alternative.successor.terminal() &&
                         observed_exact > alternative_exact;
      if (!observed30) {
        observed30 = fixed_work_value(decision.observed_successor, actor,
                                      kWork30k, config.search_width);
        observed100 = fixed_work_value(decision.observed_successor, actor,
                                       kWork100k, config.search_width);
      }
      WorkValue alternative30 = fixed_work_value(
          alternative.successor, actor, kWork30k, config.search_width);
      WorkValue alternative100 = fixed_work_value(
          alternative.successor, actor, kWork100k, config.search_width);
      const bool stable = exact ||
          (observed30->value - alternative30.value >= 0.10F - 1e-6F &&
           observed100->value - alternative100.value >= 0.10F - 1e-6F);
      const bool proved_bad = observed100->value <= -1.0F + 1e-6F &&
                              alternative100.value >= 1.0F - 1e-6F;
      if (stable && !proved_bad) {
        accepted.push_back(AcceptedPair{
            &alternative, exact, *observed30, alternative30,
            *observed100, alternative100});
      } else if (proved_bad) {
        ++stats.proved_losing_vs_winning_rejections;
      } else {
        ++stats.unstable_ordering_rejections;
      }
    }
    std::sort(accepted.begin(), accepted.end(),
              [](const AcceptedPair &left, const AcceptedPair &right) {
                // Retain the strongest inferior alternatives: these are the
                // least trivial stable pairwise constraints.
                if (left.alternative100.value != right.alternative100.value) {
                  return left.alternative100.value > right.alternative100.value;
                }
                return left.alternative->action.text() <
                       right.alternative->action.text();
              });
    const std::size_t retained = std::min({
        accepted.size(), config.maximum_pairs_per_decision,
        config.maximum_pairs_per_game - game_pairs});
    stats.cap_rejections += accepted.size() - retained;
    for (std::size_t pair_index = 0; pair_index < retained; ++pair_index) {
      const auto &pair = accepted[pair_index];
      pairs << pair_record(game, decision, pair_index, *pair.alternative,
                           pair.exact, pair.observed30, pair.alternative30,
                           pair.observed100, pair.alternative100);
      ++game_pairs;
      ++stats.pair_rows;
      stats.exact_pairs += pair.exact;
    }
  }
  return values.str() + pairs.str();
}

std::size_t parse_size_argument(const char *text, std::string_view name,
                                std::size_t maximum) {
  const auto value = parse_integer<std::size_t>(text, name);
  if (value > maximum) {
    throw std::runtime_error(std::string(name) + " exceeds its safety limit");
  }
  return value;
}

Config parse_arguments(int argc, char **argv) {
  Config config;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&]() -> const char * {
      if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + argument);
      }
      return argv[++index];
    };
    if (argument == "--input") config.input_path = next();
    else if (argument == "--ranking-game-ids") {
      config.ranking_game_ids_path = next();
    } else if (argument == "--max-analyzed-decisions") {
      config.maximum_analyzed_decisions =
          parse_size_argument(next(), argument, 64);
    } else if (argument == "--max-alternatives") {
      config.maximum_alternatives = parse_size_argument(next(), argument, 32);
    } else if (argument == "--max-pairs-per-decision") {
      config.maximum_pairs_per_decision =
          parse_size_argument(next(), argument, 4);
    } else if (argument == "--max-pairs-per-game") {
      config.maximum_pairs_per_game =
          parse_size_argument(next(), argument, 32);
    } else if (argument == "--search-width") {
      config.search_width = parse_size_argument(next(), argument, 32);
    } else if (argument == "--generator") {
      const std::string value = next();
      if (value == "tactical-progressive") {
        config.generator = jab::GeneratorStrategy::TacticalProgressive;
      } else if (value == "priority-beam") {
        config.generator = jab::GeneratorStrategy::PriorityBeam;
      } else {
        throw std::runtime_error("--generator must be tactical-progressive or priority-beam");
      }
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (config.input_path.empty()) throw std::runtime_error("--input is required");
  if (config.maximum_alternatives == 0 &&
      config.maximum_analyzed_decisions != 0) {
    throw std::runtime_error("--max-alternatives must be positive when reanalysis is enabled");
  }
  if (config.search_width < 2 && config.maximum_analyzed_decisions != 0) {
    throw std::runtime_error("--search-width must be at least two");
  }
  if (config.ranking_game_ids_path) {
    std::ifstream input(*config.ranking_game_ids_path);
    if (!input) throw std::runtime_error("cannot open --ranking-game-ids");
    std::string line;
    std::uint64_t previous = 0;
    while (std::getline(input, line)) {
      line = trim_cr(std::move(line));
      const auto game_id = parse_integer<std::uint64_t>(line, "ranking game id");
      if (game_id == 0 || game_id <= previous) {
        throw std::runtime_error("ranking game IDs must be positive, unique, and sorted");
      }
      previous = game_id;
      config.ranking_game_ids.insert(game_id);
    }
  }
  return config;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Config config = parse_arguments(argc, argv);
    const ParsedInput input = read_input(config.input_path);
    std::vector<std::string> records;
    records.reserve(input.games.size());
    AuditStats stats;
    for (const auto &game : input.games) {
      records.push_back(reanalyze_game(game, config, stats));
    }
    std::cout << "{\"schema\":\"" << kOutputSchema
              << "\",\"kind\":\"meta\",\"games\":" << input.games.size()
              << ",\"value_rows\":" << stats.value_rows
              << ",\"pairwise_rows\":" << stats.pair_rows
              << ",\"decisions_analyzed\":" << stats.decisions_analyzed
              << ",\"alternatives_analyzed\":" << stats.alternatives_analyzed
              << ",\"unstable_ordering_rejections\":"
              << stats.unstable_ordering_rejections
              << ",\"proved_losing_vs_winning_rejections\":"
              << stats.proved_losing_vs_winning_rejections
              << ",\"cap_rejections\":" << stats.cap_rejections
              << ",\"exact_pairs\":" << stats.exact_pairs
              << ",\"work_checkpoints\":[30000,100000]"
              << ",\"maximum_analyzed_decisions_per_game\":"
              << config.maximum_analyzed_decisions
              << ",\"maximum_pairs_per_decision\":"
              << config.maximum_pairs_per_decision
              << ",\"maximum_pairs_per_game\":"
              << config.maximum_pairs_per_game
              << ",\"search_width\":" << config.search_width << "}\n";
    for (const auto &record : records) std::cout << record;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "arena corpus reanalysis failed: " << error.what() << '\n';
    return 2;
  }
}
