#ifdef JACEK_ARENA_BFM_EMBEDDED_RUNTIME
#define JACEK_ARENA_BFM_NO_MAIN
#include JACEK_ARENA_BFM_EMBEDDED_RUNTIME
#else
#include "engine.hpp"
#endif

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace jab_timeout_probe {

namespace jab = jacek_arena_bfm;

struct Case {
  std::uint64_t game_id{};
  int focus_player{};
  std::string transcript;
};

template <typename Integer>
Integer integer(std::string_view text, std::string_view field) {
  Integer value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::runtime_error(std::string(field) + " is not an integer");
  }
  return value;
}

std::vector<std::string_view> split(std::string_view value, char separator) {
  std::vector<std::string_view> fields;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = value.find(separator, begin);
    fields.push_back(value.substr(
        begin, end == std::string_view::npos ? value.size() - begin
                                             : end - begin));
    if (end == std::string_view::npos) return fields;
    begin = end + 1;
  }
}

std::vector<Case> load_cases(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open timeout case TSV");
  std::string line;
  bool header = false;
  std::vector<Case> cases;
  std::uint64_t previous = 0;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      throw std::runtime_error("timeout case TSV must use LF line endings");
    }
    if (!header && line.starts_with("# ")) continue;
    if (!header) {
      if (line != "game_id\tfocus_player\tvalid_prefix") {
        throw std::runtime_error("timeout case TSV has a non-canonical header");
      }
      header = true;
      continue;
    }
    const auto fields = split(line, '\t');
    if (fields.size() != 3) {
      throw std::runtime_error("timeout case TSV row must have three fields");
    }
    Case item;
    item.game_id = integer<std::uint64_t>(fields[0], "game_id");
    item.focus_player = integer<int>(fields[1], "focus_player");
    item.transcript = fields[2];
    if (item.game_id <= previous ||
        (item.focus_player != 0 && item.focus_player != 1) ||
        item.transcript.empty()) {
      throw std::runtime_error("timeout cases must be sorted and well formed");
    }
    previous = item.game_id;
    cases.push_back(std::move(item));
  }
  if (!header || cases.empty()) throw std::runtime_error("timeout case TSV is empty");
  return cases;
}

jab::Action action(std::string_view text) {
  jab::Action result;
  if (text.empty() || text.size() > result.directions.size()) {
    throw std::runtime_error("invalid complete action length");
  }
  result.length = static_cast<std::uint16_t>(text.size());
  for (std::size_t index = 0; index < text.size(); ++index) {
    if (text[index] < '0' || text[index] > '7') {
      throw std::runtime_error("invalid complete action direction");
    }
    result.directions[index] = static_cast<std::uint8_t>(text[index] - '0');
  }
  return result;
}

jab::State replay(const Case &item) {
  jab::State state = jab::initial_state();
  for (const auto encoded : split(item.transcript, '/')) {
    if (state.terminal() || !jab::apply_action(state, action(encoded), true)) {
      throw std::runtime_error("timeout prefix is not a legal complete-turn replay");
    }
  }
  if (state.terminal() || state.to_move != item.focus_player) {
    throw std::runtime_error("timeout prefix is not the asserted pre-timeout state");
  }
  return state;
}

std::uint64_t micros_since(std::chrono::steady_clock::time_point started) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - started).count());
}

void probe(const Case &item, int budget_ms, int repetitions) {
  const jab::State state = replay(item);
  std::uint64_t generator_max = 0;
  std::size_t root_actions = 0;
  std::size_t partials = 0;
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    jab::GeneratorStats stats;
    const auto started = std::chrono::steady_clock::now();
    const auto actions = jab::generate_actions(
        state, jab::GeneratorStrategy::TacticalProgressive, true, &stats);
    generator_max = std::max(generator_max, micros_since(started));
    root_actions = actions.size();
    partials = stats.partials;
    if (actions.empty()) throw std::runtime_error("generator returned no action");
  }

  std::uint64_t search_max = 0;
  std::size_t nodes = 0;
  std::string selected;
  std::size_t generator_deadline_stops = 0;
  std::uint64_t maximum_nested_generator_us = 0;
  bool deadline_reached = false;
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    const auto started = std::chrono::steady_clock::now();
    const auto result = jab::search(
        state, started + std::chrono::milliseconds(budget_ms));
    search_max = std::max(search_max, micros_since(started));
    jab::State successor = state;
    if (result.action.length == 0 ||
        !jab::apply_action(successor, result.action, true)) {
      throw std::runtime_error("search returned no legal complete action");
    }
    nodes = result.nodes;
    selected = result.action.text();
#ifndef JACEK_ARENA_BFM_LEGACY_RUNTIME
    generator_deadline_stops = std::max(
        generator_deadline_stops, result.generator_deadline_stops);
    maximum_nested_generator_us = std::max(
        maximum_nested_generator_us, result.maximum_generator_microseconds);
    deadline_reached = deadline_reached || result.deadline_reached;
#endif
  }
  std::cout << "{\"game_id\":" << item.game_id
            << ",\"focus_player\":" << item.focus_player
            << ",\"state_hash\":\"" << std::hex << std::setw(16)
            << std::setfill('0') << jab::state_hash(state) << std::dec << "\""
            << ",\"ply\":" << state.ply
            << ",\"budget_ms\":" << budget_ms
            << ",\"generator_max_us\":" << generator_max
            << ",\"generator_partials\":" << partials
            << ",\"root_actions\":" << root_actions
            << ",\"search_max_us\":" << search_max
            << ",\"search_nodes\":" << nodes
            << ",\"nested_generator_max_us\":"
            << maximum_nested_generator_us
            << ",\"generator_deadline_stops\":"
            << generator_deadline_stops
            << ",\"deadline_reached\":"
            << (deadline_reached ? "true" : "false")
            << ",\"selected\":\"" << selected << "\"}\n";
}

}  // namespace jab_timeout_probe

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      throw std::runtime_error(
          "usage: timeout_regression_probe CASES.tsv BUDGET_MS REPETITIONS");
    }
    const int budget = jab_timeout_probe::integer<int>(argv[2], "budget");
    const int repetitions =
        jab_timeout_probe::integer<int>(argv[3], "repetitions");
    if (budget < 1 || budget > 1000 || repetitions < 1 || repetitions > 100) {
      throw std::runtime_error("probe budget/repetitions outside safety limits");
    }
    for (const auto &item : jab_timeout_probe::load_cases(argv[1])) {
      jab_timeout_probe::probe(item, budget, repetitions);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "timeout_regression_probe: " << error.what() << '\n';
    return 2;
  }
}
