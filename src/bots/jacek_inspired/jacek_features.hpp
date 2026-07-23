#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "mcts_internal.hpp"
#include "papersoccer/types.hpp"

namespace papersoccer::detail {

inline constexpr std::size_t kJacekEdgeInputs = 316;
inline constexpr std::size_t kJacekVertices = 105;
inline constexpr std::size_t kJacekDistanceBuckets = 8;
inline constexpr std::size_t kJacekInputCount =
    kJacekEdgeInputs + kJacekVertices * kJacekDistanceBuckets;
inline constexpr std::size_t kJacekMaximumActiveInputs =
    kJacekEdgeInputs + kJacekVertices;

struct JacekSparseFeatures {
  std::array<std::uint16_t, kJacekMaximumActiveInputs> indices{};
  std::size_t count{};

  bool operator==(const JacekSparseFeatures &) const noexcept = default;
};

class JacekFeatureEncoder {
 public:
  explicit JacekFeatureEncoder(
      std::shared_ptr<const SearchTopology> topology);

  JacekSparseFeatures sparse_features(
      const SearchPosition &position);
  std::array<std::uint8_t, kJacekInputCount> dense_features(
      const SearchPosition &position);

 private:
  std::shared_ptr<const SearchTopology> topology_;
  std::vector<SearchTopology::Adjacency> cooperative_adjacency_;
  std::array<SearchTopology::VertexIndex, kJacekVertices>
      rotated_vertices_{};
  std::array<SearchTopology::EdgeIndex, kJacekEdgeInputs> rotated_edges_{};
  std::array<int, kJacekVertices> distances_{};
  std::deque<SearchTopology::VertexIndex> queue_;

  void calculate_turn_distances(const SearchPosition &position);
};

JacekSparseFeatures jacek_article_sparse_features(const GameState &state);

std::array<std::uint8_t, kJacekInputCount> jacek_article_dense_features(
    const GameState &state);

}  // namespace papersoccer::detail
