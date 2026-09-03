#include "jacek_replay_bfm_search_teacher_internal.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

float fixed_teacher_value(const ps::JacekReplayBfmSearchStats &stats,
                          ps::Player mover) {
  return teacher::direct_teacher_value(stats, mover, 64'000U);
}

ps::JacekReplayBfmSearchStats complete_unsolved(float value) {
  ps::JacekReplayBfmSearchStats stats;
  stats.root_value = value;
  stats.tree_cap_reached = true;
  stats.tree_nodes = 64'000U;
  stats.visits = 1U;
  stats.completed_actions = 1U;
  stats.max_complete_turn_depth = 1U;
  stats.termination = ps::JacekReplayBfmSearchTermination::FixedWorkCap;
  return stats;
}

void direct_value_contract_is_mover_relative_and_proof_explicit() {
  const ps::JacekReplayBfmSearchStats unsolved = complete_unsolved(0.375F);
  require(fixed_teacher_value(unsolved, ps::Player::One) ==
              0.375F &&
              fixed_teacher_value(unsolved, ps::Player::Two) ==
                  0.375F,
          "An unsolved backed-up value must be emitted directly without "
          "normalization or an absolute-player sign change.");

  ps::JacekReplayBfmSearchStats won;
  won.root_value = 9'990.0F;
  won.root_solved = true;
  won.proven_winner = ps::Player::Two;
  won.max_complete_turn_depth = 1U;
  won.completed_actions = 1U;
  won.termination = ps::JacekReplayBfmSearchTermination::RootSolved;
  require(fixed_teacher_value(won, ps::Player::Two) == 1.0F,
          "A mover proof must emit exactly positive one.");

  ps::JacekReplayBfmSearchStats lost = won;
  lost.root_value = -9'990.0F;
  require(fixed_teacher_value(lost, ps::Player::One) == -1.0F,
          "An opponent proof must emit exactly negative one.");
}

void invalid_work_and_identity_fail_closed() {
  ps::JacekReplayBfmSearchStats invalid = complete_unsolved(0.0F);
  invalid.deadline_reached = true;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "A deadline label was accepted.");

  invalid = complete_unsolved(0.0F);
  invalid.max_complete_turn_depth = 0U;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "A zero-depth label was accepted.");

  invalid = complete_unsolved(0.0F);
  invalid.tree_cap_reached = false;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "An incomplete unsolved label was accepted.");

  invalid = complete_unsolved(0.0F);
  invalid.termination =
      ps::JacekReplayBfmSearchTermination::ClosedUnsolvedRoot;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "A premature termination reason was accepted as fixed work.");

  invalid = complete_unsolved(0.0F);
  invalid.closed_unsolved_nodes = 1U;
  invalid.closed_unsolved_nonexhaustive_nodes = 1U;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "A closed unsolved frontier was accepted as fixed work.");

  invalid = complete_unsolved(0.0F);
  invalid.generation_queue_drops = 1U;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "A dropped partial frontier was accepted as fixed work.");

  invalid = complete_unsolved(1.0001F);
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "An out-of-range unsolved value was accepted.");

  invalid = complete_unsolved(-0.25F);
  invalid.proven_winner = ps::Player::One;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "An unsolved root with a proof was accepted.");

  invalid = {};
  invalid.root_solved = true;
  invalid.root_value = 9'990.0F;
  invalid.max_complete_turn_depth = 1U;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "A solved root without a winner was accepted.");

  invalid = complete_unsolved(0.0F);
  invalid.tree_nodes = 63'999U;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "A short fixed-work result was accepted as the exact tree cap.");

  invalid = complete_unsolved(0.0F);
  invalid.visits = 0U;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "A zero-visit unsolved result was accepted as a usable label.");

  invalid = complete_unsolved(0.0F);
  invalid.completed_actions = 0U;
  require_rejected(
      [&] { fixed_teacher_value(invalid, ps::Player::One); },
      "A zero-action result was accepted as a usable label.");

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
      "selfsearch-pilot-20260825-v5", "position-17", 64'000U);
  require(first == teacher::derive_search_seed(
                       "selfsearch-pilot-20260825-v5", "position-17",
                       64'000U),
          "The same position search seed did not repeat.");
  require(first != teacher::derive_search_seed(
                       "selfsearch-pilot-20260825-v5", "position-18",
                       64'000U) &&
              first != teacher::derive_search_seed(
                           "selfsearch-pilot-20260825-v5", "position-17",
                           500'000U),
          "Position ID and fixed-work budget must both bind the search seed.");
  require(
      teacher::derive_search_seed(
          "selfsearch-pilot-20260825-v2",
          "position:7ec5e5e233cbefe3cd0b1ee05eb64f1830030c357330353adade61bc8e2097ac",
          64'000U) == 5'574'964'850'576'697'603ULL,
      "The production partial-frontier regression seed changed.");
}

void complete_turn_action_values_are_explicit_and_fail_closed() {
  require(teacher::kCompleteTurnActionGroupSchema ==
              "papersoccer.jacek-replay-complete-turn-action-group.v1",
          "The complete-turn action-group schema changed.");

  ps::JacekReplayBfmRootActionAnalysis unresolved;
  unresolved.action = {ps::Move{{4, 5}}};
  unresolved.canonical_transcript = "0";
  unresolved.backed_up_value = 0.375F;
  unresolved.visits = 3U;
  require(teacher::direct_action_teacher_value(
              unresolved, ps::Player::One) == 0.375F,
          "An unresolved action value was not kept in the parent mover frame.");
  require(teacher::successor_frame_teacher_value(
              unresolved, ps::Player::One, ps::Player::Two) == -0.375F &&
              teacher::successor_frame_teacher_value(
                  unresolved, ps::Player::One, ps::Player::One) == 0.375F,
          "A Player-One handoff did not flip into the successor mover frame.");

  ps::JacekReplayBfmRootActionAnalysis solved = unresolved;
  solved.solved = true;
  solved.backed_up_value = 9'980.0F;
  solved.proven_winner = ps::Player::Two;
  require(teacher::direct_action_teacher_value(solved, ps::Player::Two) ==
              1.0F &&
              teacher::successor_frame_teacher_value(
                  solved, ps::Player::Two, ps::Player::Two) == 1.0F,
          "A solved action did not map to an exact mover-relative sign.");
  ps::JacekReplayBfmRootActionAnalysis lost = solved;
  lost.backed_up_value = -9'980.0F;
  require(teacher::direct_action_teacher_value(lost, ps::Player::One) ==
              -1.0F,
          "A solved loss did not map to an exact mover-relative sign.");

  solved.proven_winner.reset();
  require_rejected(
      [&] {
        teacher::direct_action_teacher_value(solved, ps::Player::One);
      },
      "A solved action without proof identity was accepted.");
  unresolved.backed_up_value = 1.01F;
  require_rejected(
      [&] {
        teacher::direct_action_teacher_value(unresolved, ps::Player::One);
      },
      "An unresolved action outside [-1,1] was accepted.");
  unresolved.backed_up_value = 0.0F;
  unresolved.visits = 0U;
  require_rejected(
      [&] {
        teacher::direct_action_teacher_value(unresolved, ps::Player::One);
      },
      "An unvisited action was accepted as backed-up evidence.");
}

std::string run_action_group_fixture() {
  const std::filesystem::path model =
      std::filesystem::path(PAPERSOCCER_SOURCE_DIR) /
      "models/jacek_replay_bfm_bootstrap.runtime";
  std::vector<std::string> arguments{
      "papersoccer_jacek_replay_search_teacher",
      "--model",
      model.string(),
      "--model-sha256",
      "02c7757443285cf6d10e971f25dee0869339d76e1d09bb9616ce5b83966cabf7",
      "--campaign-id",
      "action-group-contract-fixture",
      "--tree-nodes",
      "64",
      "--max-actions",
      "32",
      "--max-partial-paths",
      "256",
      "--emit-action-groups",
      "--source-bundle-body-sha256",
      std::string(64U, 'a'),
  };
  std::vector<char *> argv;
  argv.reserve(arguments.size());
  for (std::string &argument : arguments) argv.push_back(argument.data());
  std::istringstream input(
      "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n"
      "position:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\t"
      "root:fixture\tcontinuation:fixture\tfixture\ttrain\t0\t0\t\n");
  std::ostringstream output;
  require(teacher::run(static_cast<int>(argv.size()), argv.data(), input,
                       output) == 0,
          "The complete-turn action-group producer failed.");
  return output.str();
}

void complete_turn_action_group_output_is_deterministic_and_bound() {
  const std::string first = run_action_group_fixture();
  const std::string second = run_action_group_fixture();
  require(first == second,
          "Equal action-group searches did not produce identical bytes.");
  require(first.find(
              "\"schema\":\"papersoccer.jacek-replay-complete-turn-action-group.v1\"")
                  != std::string::npos &&
              first.find("\"source_bundle_body_sha256\":\"" +
                         std::string(64U, 'a') + "\"") != std::string::npos &&
              first.find("\"artifact_sha256\":\"02c7757443285cf6d10e971f25dee0869339d76e1d09bb9616ce5b83966cabf7\"")
                  != std::string::npos &&
              first.find("\"payload_sha256\":\"71856d2c3ae577ba83ae8efc6ebbbd1b574d0a7c207651148febe4e9257d08b8\"")
                  != std::string::npos &&
              first.find("\"parent_identity\":") != std::string::npos &&
              first.find("\"successors_exhaustive\":") != std::string::npos &&
              first.find("\"successors\":[{") != std::string::npos,
          "The action-group row omitted an immutable binding or successor set.");
}

}  // namespace

int main() {
  try {
    direct_value_contract_is_mover_relative_and_proof_explicit();
    invalid_work_and_identity_fail_closed();
    position_seed_is_repeatable_and_budget_bound();
    complete_turn_action_values_are_explicit_and_fail_closed();
    complete_turn_action_group_output_is_deterministic_and_bound();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek replay search teacher contract test: "
              << error.what() << '\n';
    return 1;
  }
}
