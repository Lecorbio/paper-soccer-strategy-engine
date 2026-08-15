#define JACEK_ARENA_BFM_NO_MAIN
#include "submission.cpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace jab = jacek_arena_bfm;

namespace {

struct Totals {
  std::uint64_t reference{0};
  std::uint64_t matched{0};
  std::uint64_t goal{0};
  std::uint64_t goal_matched{0};
  std::uint64_t own_goal{0};
  std::uint64_t own_goal_matched{0};
  std::uint64_t block{0};
  std::uint64_t block_matched{0};
  std::uint64_t forced{0};
  std::uint64_t forced_matched{0};
  std::uint64_t illegal{0};
  std::uint64_t rotation_failures{0};
  std::vector<long long> microseconds{};
};

bool same_boundary(const jab::State &left, const jab::State &right) {
  return left.used == right.used && left.ball == right.ball &&
         left.to_move == right.to_move && left.winner == right.winner;
}

enum class Witness { None, Goal, OwnGoal, Block };

Witness witness(const jab::State &before, const jab::State &after) {
  if (!after.terminal()) return Witness::None;
  const auto &topology = jab::Topology::get();
  if (topology.north_goal(after.ball) || topology.south_goal(after.ball)) {
    return after.winner == before.to_move ? Witness::Goal : Witness::OwnGoal;
  }
  return Witness::Block;
}

std::vector<jab::State> boundaries(const jab::State &state,
                                   const std::vector<jab::Action> &actions,
                                   Totals *totals) {
  std::vector<jab::State> result;
  result.reserve(actions.size());
  for (const auto &action : actions) {
    jab::State successor = state;
    if (!jab::apply_action(successor, action)) {
      if (totals) ++totals->illegal;
      continue;
    }
    result.push_back(successor);
  }
  return result;
}

jab::Action rotated_action(const jab::Action &action) {
  jab::Action result = action;
  for (std::size_t index = 0; index < result.length; ++index) {
    result.directions[index] =
        static_cast<std::uint8_t>((result.directions[index] + 4) & 7);
  }
  return result;
}

const char *name(jab::GeneratorStrategy strategy) {
  switch (strategy) {
    case jab::GeneratorStrategy::Fixed250NineOne: return "fixed250_9to1";
    case jab::GeneratorStrategy::TacticalProgressive: return "tactical_progressive";
    case jab::GeneratorStrategy::PriorityBeam: return "priority_beam";
    case jab::GeneratorStrategy::HighCapRecall: return "high_cap";
  }
  return "unknown";
}

std::uint64_t parse_count(const char *text) {
  std::size_t consumed = 0;
  const std::string value(text);
  const auto result = std::stoull(value, &consumed);
  if (consumed != value.size()) throw std::invalid_argument("invalid state count");
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const std::size_t state_count = argc > 1
        ? static_cast<std::size_t>(parse_count(argv[1])) : 128;
    if (state_count == 0 || argc > 2) {
      throw std::invalid_argument("usage: generator_benchmark [STATE_COUNT]");
    }
    constexpr std::array<jab::GeneratorStrategy, 3> kStrategies{
        jab::GeneratorStrategy::Fixed250NineOne,
        jab::GeneratorStrategy::TacticalProgressive,
        jab::GeneratorStrategy::PriorityBeam};
    std::array<Totals, kStrategies.size()> totals{};
    jab::State position = jab::initial_state();
    for (std::size_t sample = 0; sample < state_count; ++sample) {
      if (position.terminal() || sample % 17 == 0) position = jab::initial_state();
      const auto reference_actions = jab::generate_actions(
          position, jab::GeneratorStrategy::HighCapRecall, true);
      const auto reference = boundaries(position, reference_actions, nullptr);
      if (reference.empty()) throw std::runtime_error("empty reference generator");
      for (std::size_t strategy_index = 0;
           strategy_index < kStrategies.size(); ++strategy_index) {
        auto &total = totals[strategy_index];
        const auto started = std::chrono::steady_clock::now();
        const auto actions = jab::generate_actions(
            position, kStrategies[strategy_index], true);
        total.microseconds.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count());
        const auto candidate = boundaries(position, actions, &total);
        total.reference += reference.size();
        for (std::size_t index = 0; index < reference.size(); ++index) {
          const bool matched = std::any_of(
              candidate.begin(), candidate.end(), [&](const jab::State &item) {
                return same_boundary(item, reference[index]);
              });
          if (matched) ++total.matched;
          const Witness type = witness(position, reference[index]);
          if (type == Witness::Goal) {
            ++total.goal;
            if (matched) ++total.goal_matched;
          } else if (type == Witness::OwnGoal) {
            ++total.own_goal;
            if (matched) ++total.own_goal_matched;
          } else if (type == Witness::Block) {
            ++total.block;
            if (matched) ++total.block_matched;
          }
          if (reference.size() == 1) {
            ++total.forced;
            if (matched) ++total.forced_matched;
          }
        }
        const jab::State rotated_before = jab::rotate_and_swap(position);
        for (const auto &action : actions) {
          jab::State after = position;
          jab::State rotated_after = rotated_before;
          if (!jab::apply_action(after, action) ||
              !jab::apply_action(rotated_after, rotated_action(action)) ||
              !same_boundary(jab::rotate_and_swap(after), rotated_after)) {
            ++total.rotation_failures;
          }
        }
      }
      // The next procedural state is selected only from the recall generator;
      // no historical action, replay, or checkpoint can enter this benchmark.
      const std::size_t selected =
          static_cast<std::size_t>(jab::state_hash(position) + sample * 31U) %
          reference_actions.size();
      if (!jab::apply_action(position, reference_actions[selected])) {
        throw std::runtime_error("procedural state advance failed");
      }
    }
    std::cout << "strategy\tunique_recall\tgoal\town_goal\tblock\tforced"
                 "\tillegal\trotation_failures\tp99_us\n";
    for (std::size_t index = 0; index < kStrategies.size(); ++index) {
      auto &total = totals[index];
      std::sort(total.microseconds.begin(), total.microseconds.end());
      const std::size_t percentile = std::min(
          total.microseconds.size() - 1,
          (total.microseconds.size() * 99 + 99) / 100 - 1);
      const auto ratio = [](std::uint64_t matched, std::uint64_t all) {
        return all == 0 ? 1.0 : static_cast<double>(matched) / all;
      };
      std::cout << name(kStrategies[index]) << '\t'
                << ratio(total.matched, total.reference) << '\t'
                << ratio(total.goal_matched, total.goal) << '\t'
                << ratio(total.own_goal_matched, total.own_goal) << '\t'
                << ratio(total.block_matched, total.block) << '\t'
                << ratio(total.forced_matched, total.forced) << '\t'
                << total.illegal << '\t' << total.rotation_failures << '\t'
                << total.microseconds[percentile] << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "generator_benchmark: " << error.what() << '\n';
    return 1;
  }
}
