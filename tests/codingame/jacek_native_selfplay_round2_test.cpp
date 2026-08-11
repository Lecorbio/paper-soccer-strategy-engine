#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#define PAPER_SOCCER_JACEK_NATIVE_ROUND2_NO_MAIN
#include "../../tools/jacek_native_selfplay_round2.cpp"

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

Round2Sample ranked_sample(std::size_t turn, float error, float uncertainty) {
  Round2Sample result;
  result.sample.turn = turn;
  result.outcome_error = error;
  result.uncertainty = uncertainty;
  return result;
}

void test_deterministic_hard_uncertain_selection() {
  const std::vector<Round2Sample> samples{
      ranked_sample(10, 0.2F, 0.01F),
      ranked_sample(11, 1.9F, 0.8F),
      ranked_sample(12, 1.2F, 0.02F),
      ranked_sample(13, 0.4F, 0.4F),
  };
  const auto first = select_reanalysis_indices(samples, 3);
  const auto second = select_reanalysis_indices(samples, 3);
  require(first == second, "reanalysis selection is not deterministic");
  require(first.size() == 3, "wrong reanalysis selection size");
  require(first[0] == std::pair<std::size_t, std::string>(1, "hard"),
          "highest-error sample was not selected first");
  require(first[1] == std::pair<std::size_t, std::string>(0, "uncertain"),
          "lowest-confidence sample was not selected second");
  require(first[2] == std::pair<std::size_t, std::string>(2, "hard"),
          "selection did not alternate back to hard states");
}

void test_depth_schedule_is_independent_of_color() {
  Round2Arguments arguments;
  arguments.seed = 19;
  arguments.opening_depths = {0, 4, 8, 12};
  for (std::size_t pair = 0; pair < arguments.opening_depths.size(); ++pair) {
    const Round2Schedule even = schedule_round2_game(arguments, pair * 2U);
    const Round2Schedule odd = schedule_round2_game(arguments, pair * 2U + 1U);
    require(even.opening_pair_index == odd.opening_pair_index,
            "colors do not share an opening pair");
    require(even.opening_depth == odd.opening_depth,
            "opening depth is coupled to color");
    require(even.opening_pair_seed == odd.opening_pair_seed,
            "paired colors do not share the same opening seed");
    require(!even.swap_models && odd.swap_models,
            "checkpoint colors were not alternated within a depth pair");
  }
}

void test_unsigned_contract_rejects_signed_values() {
  bool rejected = false;
  try {
    (void)parse_round2_unsigned("-1", "seed");
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "signed value was accepted as an unsigned seed");
}

void test_planned_caps_are_disclosed_not_operational_interruptions() {
  native::SearchResult primary;
  primary.encoded = "1";
  primary.value = 0.20F;
  primary.stats.tree_cap_reached = true;
  primary.stats.generator_truncations = 3;
  primary.stats.proof_truncations = 5;
  native::SearchResult verification;
  verification.encoded = "1";
  verification.value = 0.24F;
  verification.stats.expansion_cap_reached = true;
  verification.stats.generator_truncations = 7;
  verification.stats.proof_truncations = 11;
  const ReanalysisDecision decision = classify_reanalysis(
      primary, verification, ps::Player::One);
  require(!decision.operational_interruption,
          "planned capped BFM was treated as operational interruption");
  require(decision.primary_planned_work_exhaustion &&
              decision.verification_planned_work_exhaustion,
          "planned work exhaustion was not preserved");
  require(decision.primary_generator_sampling_truncations == 3 &&
              decision.verification_generator_sampling_truncations == 7 &&
              decision.primary_proof_sampling_truncations == 5 &&
              decision.verification_proof_sampling_truncations == 11,
          "planned generator/proof sampling caps were not disclosed");
  require(decision.stable && !decision.exact,
          "stable unsolved fixed-work label was rejected");
}

void test_unstable_and_operational_results_are_rejected() {
  native::SearchResult primary;
  primary.encoded = "1";
  primary.value = 0.1F;
  native::SearchResult verification;
  verification.encoded = "2";
  verification.value = 0.1F;
  require(!classify_reanalysis(primary, verification, ps::Player::One).stable,
          "action instability was accepted");
  verification.encoded = "1";
  verification.value = 0.3F;
  require(!classify_reanalysis(primary, verification, ps::Player::One).stable,
          "large value drift was accepted");
  verification.value = 0.1F;
  verification.stats.deadline_reached = true;
  require(!classify_reanalysis(primary, verification, ps::Player::One).stable,
          "deadline interruption was accepted");
}

void test_exact_result_overrides_operational_state() {
  native::SearchResult primary;
  primary.encoded = "1";
  primary.value = -0.8F;
  primary.stats.generator_truncations = 1;
  native::SearchResult verification;
  verification.encoded = "7";
  verification.value = 0.8F;
  verification.stats.deadline_reached = true;
  verification.solved = true;
  verification.solved_winner = ps::Player::Two;
  const ReanalysisDecision decision = classify_reanalysis(
      primary, verification, ps::Player::Two);
  require(decision.exact && decision.stable,
          "exact outcome did not override operational flags");
  require(decision.value == 1.0F,
          "exact result was not converted to mover-relative outcome");
}

}  // namespace

int main() {
  try {
    test_deterministic_hard_uncertain_selection();
    test_depth_schedule_is_independent_of_color();
    test_unsigned_contract_rejects_signed_values();
    test_planned_caps_are_disclosed_not_operational_interruptions();
    test_unstable_and_operational_results_are_rejected();
    test_exact_result_overrides_operational_state();
    std::cout << "Jacek-native round-two self-play tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Jacek-native round-two self-play test failure: "
              << error.what() << '\n';
    return 1;
  }
}
