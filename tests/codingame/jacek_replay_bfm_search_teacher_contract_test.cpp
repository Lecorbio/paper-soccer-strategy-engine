#include "jacek_replay_bfm_search_teacher_internal.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ps = papersoccer;
namespace teacher = papersoccer::jacek_replay_search_teacher;

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_rejected(Function &&function, const char *message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected, message);
}

ps::JacekReplayBfmSearchStats complete_unsolved(float value) {
  ps::JacekReplayBfmSearchStats stats;
  stats.root_value = value;
  stats.tree_cap_reached = true;
  stats.tree_nodes = 64'000U;
  stats.max_complete_turn_depth = 1U;
  stats.termination = ps::JacekReplayBfmSearchTermination::FixedWorkCap;
  return stats;
}

void direct_value_contract_is_mover_relative_and_proof_explicit() {
  const ps::JacekReplayBfmSearchStats unsolved = complete_unsolved(0.375F);
  require(teacher::direct_teacher_value(unsolved, ps::Player::One) ==
              0.375F &&
              teacher::direct_teacher_value(unsolved, ps::Player::Two) ==
                  0.375F,
          "An unsolved backed-up value must be emitted directly without "
          "normalization or an absolute-player sign change.");

  ps::JacekReplayBfmSearchStats won;
  won.root_value = 9'990.0F;
  won.root_solved = true;
  won.proven_winner = ps::Player::Two;
  won.max_complete_turn_depth = 1U;
  won.termination = ps::JacekReplayBfmSearchTermination::RootSolved;
  require(teacher::direct_teacher_value(won, ps::Player::Two) == 1.0F,
          "A mover proof must emit exactly positive one.");

  ps::JacekReplayBfmSearchStats lost = won;
  lost.root_value = -9'990.0F;
  require(teacher::direct_teacher_value(lost, ps::Player::One) == -1.0F,
          "An opponent proof must emit exactly negative one.");
}

void invalid_work_and_identity_fail_closed() {
  ps::JacekReplayBfmSearchStats invalid = complete_unsolved(0.0F);
  invalid.deadline_reached = true;
  require_rejected(
      [&] { teacher::direct_teacher_value(invalid, ps::Player::One); },
      "A deadline label was accepted.");

  invalid = complete_unsolved(0.0F);
  invalid.max_complete_turn_depth = 0U;
  require_rejected(
      [&] { teacher::direct_teacher_value(invalid, ps::Player::One); },
      "A zero-depth label was accepted.");

  invalid = complete_unsolved(0.0F);
  invalid.tree_cap_reached = false;
  require_rejected(
      [&] { teacher::direct_teacher_value(invalid, ps::Player::One); },
      "An incomplete unsolved label was accepted.");

  invalid = complete_unsolved(0.0F);
  invalid.termination =
      ps::JacekReplayBfmSearchTermination::ClosedUnsolvedRoot;
  require_rejected(
      [&] { teacher::direct_teacher_value(invalid, ps::Player::One); },
      "A premature termination reason was accepted as fixed work.");

  invalid = complete_unsolved(0.0F);
  invalid.closed_unsolved_nodes = 1U;
  invalid.closed_unsolved_nonexhaustive_nodes = 1U;
  require_rejected(
      [&] { teacher::direct_teacher_value(invalid, ps::Player::One); },
      "A closed unsolved frontier was accepted as fixed work.");

  invalid = complete_unsolved(1.0001F);
  require_rejected(
      [&] { teacher::direct_teacher_value(invalid, ps::Player::One); },
      "An out-of-range unsolved value was accepted.");

  invalid = complete_unsolved(-0.25F);
  invalid.proven_winner = ps::Player::One;
  require_rejected(
      [&] { teacher::direct_teacher_value(invalid, ps::Player::One); },
      "An unsolved root with a proof was accepted.");

  invalid = {};
  invalid.root_solved = true;
  invalid.root_value = 9'990.0F;
  invalid.max_complete_turn_depth = 1U;
  require_rejected(
      [&] { teacher::direct_teacher_value(invalid, ps::Player::One); },
      "A solved root without a winner was accepted.");

  const std::string one(64U, '1');
  const std::string two(64U, '2');
  teacher::validate_model_identity(one, one);
  require_rejected(
      [&] { teacher::validate_model_identity(one, two); },
      "A mismatched model SHA-256 was accepted.");
  require_rejected(
      [&] { teacher::validate_model_identity(one, "not-a-sha256"); },
      "A malformed asserted model SHA-256 was accepted.");
}

void position_seed_is_repeatable_and_budget_bound() {
  const std::uint64_t first = teacher::derive_search_seed(
      "selfsearch-pilot-20260825-v2", "position-17", 64'000U);
  require(first == teacher::derive_search_seed(
                       "selfsearch-pilot-20260825-v2", "position-17",
                       64'000U),
          "The same position search seed did not repeat.");
  require(first != teacher::derive_search_seed(
                       "selfsearch-pilot-20260825-v2", "position-18",
                       64'000U) &&
              first != teacher::derive_search_seed(
                           "selfsearch-pilot-20260825-v2", "position-17",
                           500'000U),
          "Position ID and fixed-work budget must both bind the search seed.");
}

}  // namespace

int main() {
  try {
    direct_value_contract_is_mover_relative_and_proof_explicit();
    invalid_work_and_identity_fail_closed();
    position_seed_is_repeatable_and_budget_bound();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek replay search teacher contract test: "
              << error.what() << '\n';
    return 1;
  }
}
