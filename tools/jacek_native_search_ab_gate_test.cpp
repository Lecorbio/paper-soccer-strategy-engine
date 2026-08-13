#define PAPER_SOCCER_JACEK_NATIVE_SEARCH_AB_GATE_NO_MAIN
#include "jacek_native_search_ab_gate.cpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gate = papersoccer::jacek_native_search_ab;
namespace native = papersoccer::jacek_native_bfm;

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Function>
void require_invalid(Function &&function, std::string_view message) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, message);
}

gate::Arguments parse(std::vector<std::string> values) {
  values.insert(values.begin(), "gate");
  std::vector<char *> argv;
  for (std::string &value : values) argv.push_back(value.data());
  return gate::parse_arguments(static_cast<int>(argv.size()), argv.data());
}

void profiles_are_independent_and_legacy_clocks_are_shared() {
  const gate::Arguments arguments = parse({
      "--checkpoint", "one.runtime", "--pairs", "8", "--first-ms", "800",
      "--later-ms", "155", "--candidate-tree-nodes", "120000",
      "--baseline-tree-nodes", "80000", "--candidate-c", "1.25",
      "--baseline-c", "0.95", "--candidate-fpu", "0.25",
      "--baseline-fpu", "0.5", "--candidate-supported-advance-penalty",
      "0.11", "--baseline-supported-advance-penalty", "0.15",
      "--candidate-final", "value-only",
      "--baseline-final", "value-log-visits"});
  require(arguments.checkpoint == "one.runtime" && arguments.pairs == 8 &&
              arguments.first_ms == 800 && arguments.later_ms == 155,
          "one runtime and one clock profile must drive both sides");
  require(arguments.candidate.first_ms == 800 &&
              arguments.candidate.later_ms == 155 &&
              arguments.baseline.first_ms == 800 &&
              arguments.baseline.later_ms == 155 &&
              !arguments.independent_clocks,
          "legacy shared clocks must remain byte-for-byte compatible");
  require(arguments.candidate.tree_nodes == 120000 &&
              arguments.baseline.tree_nodes == 80000 &&
              arguments.candidate.opening_exploration == 1.25 &&
              arguments.candidate.exploration == 1.25 &&
              arguments.candidate.exploration_opening_own_decisions == 0 &&
              arguments.baseline.opening_exploration == 0.95 &&
              arguments.baseline.exploration == 0.95 &&
              arguments.candidate.fpu == 0.25 &&
              arguments.baseline.fpu == 0.5 &&
              arguments.candidate.supported_advance_penalty == 0.11 &&
              arguments.baseline.supported_advance_penalty == 0.15 &&
              arguments.candidate.final_formula ==
                  gate::FinalFormula::ValueOnly &&
              arguments.baseline.final_formula ==
                  gate::FinalFormula::Production,
          "candidate and baseline search profiles must remain independent");
}

void phase_exploration_is_complete_and_uses_an_independent_cutoff() {
  const gate::Arguments arguments = parse({
      "--checkpoint", "one.runtime", "--candidate-opening-tree-nodes",
      "20000", "--candidate-later-tree-nodes", "80000",
      "--candidate-opening-own-decisions", "4", "--candidate-opening-c",
      "0.65", "--candidate-later-c", "0.80",
      "--candidate-exploration-opening-own-decisions", "12"});
  require(gate::tree_nodes_for_own_decision(arguments.candidate, 11) ==
                  80000 &&
              gate::exploration_for_own_decision(arguments.candidate, 11) ==
                  0.65 &&
              gate::exploration_for_own_decision(arguments.candidate, 12) ==
                  0.80,
          "exploration ordinals 11/12 must be independent of the tree cutoff");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime", "--candidate-c",
                        "0.95", "--candidate-opening-c", "0.65",
                        "--candidate-later-c", "0.80",
                        "--candidate-exploration-opening-own-decisions",
                        "12"}); },
      "legacy and phase exploration must not mix");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime",
                        "--candidate-opening-c", "0.65"}); },
      "partial phase exploration must fail closed");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime",
                        "--candidate-opening-c", "0.65",
                        "--candidate-later-c", "0.80",
                        "--candidate-exploration-opening-own-decisions",
                        "0"}); },
      "a zero exploration cutoff must fail closed");
}

void side_clocks_are_complete_independent_and_bounded() {
  const gate::Arguments arguments = parse({
      "--checkpoint", "one.runtime", "--candidate-first-ms", "950",
      "--candidate-later-ms", "175", "--baseline-first-ms", "950",
      "--baseline-later-ms", "170", "--candidate-root-reply-width", "4",
      "--baseline-root-reply-width", "0"});
  require(arguments.independent_clocks &&
              arguments.candidate.first_ms == 950 &&
              arguments.candidate.later_ms == 175 &&
              arguments.baseline.first_ms == 950 &&
              arguments.baseline.later_ms == 170 &&
              arguments.candidate.root_reply_width == 4 &&
              arguments.baseline.root_reply_width == 0,
          "all four clocks and both reply widths must remain side-specific");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime",
                        "--candidate-first-ms", "950"}); },
      "a partial independent clock profile must fail closed");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime",
                        "--first-ms", "800"}); },
      "a partial shared clock profile must fail closed");
  require_invalid(
      [] { (void)parse({
          "--checkpoint", "one.runtime", "--first-ms", "800",
          "--candidate-first-ms", "950", "--candidate-later-ms", "175",
          "--baseline-first-ms", "950", "--baseline-later-ms", "170"}); },
      "shared and independent clocks must not mix");
  require_invalid(
      [] { (void)parse({
          "--checkpoint", "one.runtime", "--candidate-first-ms", "950",
          "--candidate-first-ms", "900", "--candidate-later-ms", "175",
          "--baseline-first-ms", "950", "--baseline-later-ms", "170"}); },
      "conflicting duplicate clocks must fail closed");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime",
                        "--candidate-root-reply-width", "251"}); },
      "reply widths above the native root action cap must fail closed");
}

void phase_caps_are_complete_and_use_the_own_decision_ordinal() {
  const gate::Arguments arguments = parse({
      "--checkpoint", "one.runtime", "--candidate-opening-tree-nodes",
      "20000", "--candidate-later-tree-nodes", "80000",
      "--candidate-opening-own-decisions", "4",
      "--baseline-opening-tree-nodes", "80000",
      "--baseline-later-tree-nodes", "80000",
      "--baseline-opening-own-decisions", "4"});
  require(arguments.candidate.opening_tree_nodes == 20000 &&
              arguments.candidate.tree_nodes == 80000 &&
              arguments.candidate.opening_own_decisions == 4,
          "phase profile must preserve all three declared fields");
  require(gate::tree_nodes_for_own_decision(arguments.candidate, 3) == 20000 &&
              gate::tree_nodes_for_own_decision(arguments.candidate, 4) ==
                  80000,
          "ordinal N-1 must use the opening cap and N the later cap");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime",
                        "--candidate-tree-nodes", "20000",
                        "--candidate-opening-tree-nodes", "20000",
                        "--candidate-later-tree-nodes", "80000",
                        "--candidate-opening-own-decisions", "4"}); },
      "legacy and phase caps must not mix");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime",
                        "--candidate-opening-tree-nodes", "20000"}); },
      "partial phase profiles must fail closed");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime",
                        "--candidate-opening-tree-nodes", "20000",
                        "--candidate-later-tree-nodes", "80000",
                        "--candidate-opening-own-decisions", "0"}); },
      "a phase profile with no opening decisions must fail closed");
}

void final_formulas_use_the_declared_counters() {
  native::RootActionStat action{"1", 0.25F, 0.0F, 4, 1,
                                native::TacticalClass::SafeHandoff};
  require(std::abs(gate::final_score(action, gate::FinalFormula::Production) -
                       (0.25 + std::log(4.0))) < 1e-12,
          "production final score must use visits without an offset");
  require(std::abs(gate::final_score(
                       action, gate::FinalFormula::ValueLogVisitsPlus3) -
                       (0.25 + std::log(7.0))) < 1e-12,
          "visits-plus-three must use the backed-visit counter");
  require(std::abs(gate::final_score(
                       action,
                       gate::FinalFormula::ValueLogSelectionVisitsPlus3) -
                       (0.25 + std::log(4.0))) < 1e-12,
          "selection-visits-plus-three must use the selection counter");
  require(gate::final_score(action, gate::FinalFormula::ValueOnly) == 0.25,
          "value-only final selection must ignore visit counts");
  action.penalty_applied = 0.15;
  require(std::abs(gate::final_score(action, gate::FinalFormula::Production) -
                       (0.25 + std::log(4.0) - 0.15)) < 1e-12 &&
              std::abs(gate::final_score(
                           action, gate::FinalFormula::ValueOnly) -
                       0.10) < 1e-12,
          "Every gate final formula must preserve the native score overlay.");

  native::SearchResult search;
  search.root_actions = {
      {"7", 0.0F, 0.0F, 8, 7, native::TacticalClass::SafeHandoff},
      {"1", 1.0F, 0.0F, 1, 0, native::TacticalClass::SafeHandoff},
  };
  require(gate::select_action(search, gate::FinalFormula::Production) == "7" &&
              gate::select_action(search, gate::FinalFormula::ValueOnly) ==
                  "1",
          "the chosen root action must follow the requested final formula");

  search.root_actions.front().exact_reply_refuted = true;
  require(gate::select_action(search, gate::FinalFormula::Production) == "1",
          "alternate final formulas must never resurrect an exact refutation");
  search.root_actions.back().exact_reply_refuted = true;
  require(gate::select_action(search, gate::FinalFormula::Production) == "7",
          "all-refuted roots must retain a deterministic legal fallback");
}

void invalid_or_asymmetric_interfaces_fail_closed() {
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime",
                        "--candidate-tree-nodes", "120001"}); },
      "tree caps above the native maximum must fail");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime", "--candidate-c",
                        "-0.1"}); },
      "negative exploration must fail");
  require_invalid(
      [] {
        (void)parse({"--checkpoint", "one.runtime",
                     "--candidate-supported-advance-penalty", "-0.01"});
      },
      "negative supported-advance penalties must fail");
  require_invalid(
      [] {
        (void)parse({"--checkpoint", "one.runtime",
                     "--candidate-supported-advance-penalty", "0.201"});
      },
      "supported-advance penalties above the native bound must fail");
  require_invalid(
      [] { (void)parse({"--checkpoint", "one.runtime", "--candidate-final",
                        "mystery"}); },
      "unknown final formulas must fail");
  require_invalid(
      [] { (void)parse({"--candidate-tree-nodes", "120000"}); },
      "the shared runtime must be explicit");
}

void procedural_openings_are_reproducible() {
  const gate::Opening first = gate::make_opening(1234567, 8);
  const gate::Opening second = gate::make_opening(1234567, 8);
  require(first.seed == second.seed && first.complete_turns == 8 &&
              first.player_turns == second.player_turns &&
              first.state.ball == second.state.ball &&
              first.state.to_move == second.state.to_move &&
              first.state.status == second.state.status &&
              first.state.path == second.state.path &&
              first.state.used_segments == second.state.used_segments &&
              first.state.visit_count == second.state.visit_count,
          "paired opening construction must be deterministic");
  require(gate::opening_turns_text({0, 4, 8, 12}) == "0,4,8,12",
          "the transcript must render the exact opening-depth schedule");
  gate::SearchProfile profile;
  profile.opening_tree_nodes = 20'000;
  profile.tree_nodes = 80'000;
  profile.opening_own_decisions = 4;
  const gate::Opening early = gate::make_opening(7654321, 4);
  const gate::Opening late = gate::make_opening(7654321, 8);
  require(early.player_turns == std::array<std::uint32_t, 2>{2, 2} &&
              late.player_turns == std::array<std::uint32_t, 2>{4, 4} &&
              gate::tree_nodes_for_own_decision(
                  profile, early.player_turns[0]) == 20'000 &&
              gate::tree_nodes_for_own_decision(
                  profile, late.player_turns[0]) == 80'000,
          "procedural prefix turns must offset each player's phase ordinal");
}

void execution_order_balances_profile_warmup() {
  require(gate::candidate_player_order(0) == std::array<int, 2>{0, 1} &&
              gate::candidate_player_order(1) == std::array<int, 2>{1, 0} &&
              gate::candidate_player_order(2) == std::array<int, 2>{0, 1},
          "pair parity must balance which search profile runs first");
  require(gate::kTimingScope == "search-through-apply",
          "the A/B transcript must disclose that model loading is untimed");
  require(gate::kSearchConstructionTiming == "included",
          "per-side elapsed telemetry must include native search construction");
}

void construction_inclusive_clock_buckets_are_complete() {
  gate::SearchTotals totals;
  native::SearchResult search;
  search.encoded = "0";
  search.stats.tree_nodes = 17;
  search.root_actions = {
      {"0", 0.0F, 0.0F, 1, 0, native::TacticalClass::SafeHandoff}};
  search.root_actions.front().penalty_applied = 0.11;
  totals.observe(search, "0", 989.0, 0, 4, 12, true, 0);
  totals.observe(search, "0", 197.0, 12, 4, 12, true, 1);
  require(totals.decisions == 2 && totals.first_clock_decisions == 1 &&
              totals.later_clock_decisions == 1 &&
              totals.first_clock_milliseconds == 989.0 &&
              totals.later_clock_milliseconds == 197.0 &&
              totals.maximum_first_milliseconds == 989.0 &&
              totals.maximum_later_milliseconds == 197.0 &&
              totals.headroom_failures == 0 &&
              totals.operational_timeouts == 0 &&
              totals.supported_advance_penalized_actions == 2 &&
              totals.supported_advance_selected_actions == 2 &&
              totals.supported_advance_selection_overrides == 2 &&
              totals.supported_advance_selection_overrides_by_player ==
                  std::array<std::uint64_t, 2>{1, 1} &&
              totals.exploration_opening_decisions_by_player ==
                  std::array<std::uint64_t, 2>{1, 0} &&
              totals.exploration_later_decisions_by_player ==
                  std::array<std::uint64_t, 2>{0, 1},
          "per-side clock buckets must expose 950/170/175/180 selection data");
  totals.observe(search, "0", 990.0, 0, 4, 12, false, 0);
  totals.observe(search, "0", 198.0, 12, 4, 12, false, 1);
  require(totals.headroom_failures == 2 &&
              totals.operational_timeouts == 0,
          "construction-inclusive headroom must be strict below 990/198");
}

}  // namespace

int main() {
  try {
    profiles_are_independent_and_legacy_clocks_are_shared();
    side_clocks_are_complete_independent_and_bounded();
    phase_caps_are_complete_and_use_the_own_decision_ordinal();
    phase_exploration_is_complete_and_uses_an_independent_cutoff();
    final_formulas_use_the_declared_counters();
    invalid_or_asymmetric_interfaces_fail_closed();
    procedural_openings_are_reproducible();
    execution_order_balances_profile_warmup();
    construction_inclusive_clock_buckets_are_complete();
    std::cout << "Jacek-native search A/B gate tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
