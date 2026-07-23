#include "jacek_neural_internal.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "jacek_neural_model.hpp"

namespace papersoccer::detail {

namespace {

constexpr int kMaximumEvaluation = 100'000;

}  // namespace

JacekNeuralEvaluator::JacekNeuralEvaluator(
    std::shared_ptr<const SearchTopology> topology)
    : encoder_(topology) {
  const RulesConfig &config = topology->config();
  if (config.goal_rule != GoalRule::OpponentGoalOnly ||
      config.blocked_rule != BlockedRule::PlayerToMoveLoses) {
    throw std::invalid_argument(
        "JacekInspiredBot model requires the demo goal and blocking rules");
  }
  static_assert(jacek_neural_model::kInputCount == kJacekInputCount);
  static_assert(jacek_neural_model::kHiddenOne == 32);
  static_assert(jacek_neural_model::kHiddenTwo == 32);
}

float JacekNeuralEvaluator::mover_logit(const SearchPosition &position) {
  const JacekSparseFeatures features = encoder_.sparse_features(position);
  hidden_one_ = jacek_neural_model::kB1;
  for (std::size_t active = 0; active < features.count; ++active) {
    const std::size_t row =
        static_cast<std::size_t>(features.indices[active]) *
        jacek_neural_model::kHiddenOne;
    for (std::size_t hidden = 0;
         hidden < jacek_neural_model::kHiddenOne; ++hidden) {
      hidden_one_[hidden] += jacek_neural_model::kW1[row + hidden];
    }
  }
  for (float &value : hidden_one_) {
    value = std::max(value, 0.0F);
  }

  hidden_two_ = jacek_neural_model::kB2;
  for (std::size_t first = 0;
       first < jacek_neural_model::kHiddenOne; ++first) {
    const std::size_t row = first * jacek_neural_model::kHiddenTwo;
    for (std::size_t second = 0;
         second < jacek_neural_model::kHiddenTwo; ++second) {
      hidden_two_[second] +=
          hidden_one_[first] * jacek_neural_model::kW2[row + second];
    }
  }
  for (float &value : hidden_two_) {
    value = std::max(value, 0.0F);
  }

  float result = jacek_neural_model::kB3;
  for (std::size_t hidden = 0;
       hidden < jacek_neural_model::kHiddenTwo; ++hidden) {
    result += hidden_two_[hidden] * jacek_neural_model::kW3[hidden];
  }
  return result;
}

int JacekNeuralEvaluator::evaluate(const SearchPosition &position) {
  const int mover_score = static_cast<int>(std::lround(
      mover_logit(position) * jacek_neural_model::kTargetTemperature));
  const int player_one_score =
      position.to_move() == Player::One ? mover_score : -mover_score;
  return std::clamp(player_one_score, -kMaximumEvaluation,
                    kMaximumEvaluation);
}

float jacek_article_mover_logit(const GameState &state) {
  auto topology = std::make_shared<SearchTopology>(state.config);
  SearchPosition position(topology, state);
  JacekNeuralEvaluator evaluator(topology);
  return evaluator.mover_logit(position);
}

}  // namespace papersoccer::detail
