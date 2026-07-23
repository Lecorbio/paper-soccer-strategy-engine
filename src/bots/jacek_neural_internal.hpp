#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "jacek_features.hpp"

namespace papersoccer::detail {

class JacekNeuralEvaluator {
 public:
  explicit JacekNeuralEvaluator(
      std::shared_ptr<const SearchTopology> topology);

  int evaluate(const SearchPosition &position);
  float mover_logit(const SearchPosition &position);

 private:
  JacekFeatureEncoder encoder_;
  std::array<float, 32> hidden_one_{};
  std::array<float, 32> hidden_two_{};
};

float jacek_article_mover_logit(const GameState &state);

}  // namespace papersoccer::detail
