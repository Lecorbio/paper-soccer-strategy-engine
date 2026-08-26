#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "papersoccer/types.hpp"

namespace papersoccer::detail {

inline constexpr std::size_t kReplayBfmEdgeInputs = 316;
inline constexpr std::size_t kReplayBfmVertices = 105;
inline constexpr std::size_t kReplayBfmVertexCategories = 57;
inline constexpr std::size_t kReplayBfmInputCount =
    kReplayBfmEdgeInputs +
    kReplayBfmVertices * kReplayBfmVertexCategories;
inline constexpr std::size_t kReplayBfmHiddenOne = 192;
inline constexpr std::size_t kReplayBfmHiddenTwo = 32;
inline constexpr std::size_t kReplayBfmResidualRank = 16;
inline constexpr std::size_t kReplayBfmWeightCount =
    kReplayBfmInputCount * kReplayBfmHiddenOne +
    kReplayBfmHiddenOne * kReplayBfmHiddenTwo +
    kReplayBfmHiddenTwo;
inline constexpr std::size_t kReplayBfmResidualParameterCount =
    2U + kReplayBfmHiddenOne * kReplayBfmResidualRank +
    kReplayBfmResidualRank;
inline constexpr std::size_t kReplayBfmRuntimeV2WeightCount =
    kReplayBfmWeightCount + kReplayBfmResidualParameterCount;
inline constexpr std::size_t kReplayBfmMaximumActiveInputs =
    kReplayBfmEdgeInputs + kReplayBfmVertices;
inline constexpr std::size_t kReplayBfmRuntimeHeaderBytes = 128;
inline constexpr std::string_view kReplayBfmFeatureSchema =
    "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:"
    "mover-relative-rotate180:true-turn-distance+free-degree";
inline constexpr std::string_view kReplayBfmFeatureSchemaSha256 =
    "9b8c1f03ce9c6ae91f90f31a9df7ed17e4815d4afe060def3d17e08af5326d66";

static_assert(kReplayBfmInputCount == 6301);
static_assert(kReplayBfmWeightCount == 1'215'968);
static_assert(kReplayBfmRuntimeV2WeightCount == 1'219'058);

struct ReplayBfmSparseFeatures {
  std::array<std::uint16_t, kReplayBfmMaximumActiveInputs> indices{};
  std::size_t count{};

  bool operator==(const ReplayBfmSparseFeatures &) const noexcept = default;
};

struct ReplayBfmPreparedEvaluation {
  ReplayBfmSparseFeatures base_features{};
  std::array<float, kReplayBfmHiddenOne> first_layer{};
};

class SearchTopology;
class SearchPosition;
struct PositionKey;

class ReplayBfmFeatureEncoder {
 public:
  explicit ReplayBfmFeatureEncoder(
      std::shared_ptr<const SearchTopology> topology);
  ~ReplayBfmFeatureEncoder();

  ReplayBfmFeatureEncoder(const ReplayBfmFeatureEncoder &) = delete;
  ReplayBfmFeatureEncoder &operator=(const ReplayBfmFeatureEncoder &) = delete;
  ReplayBfmFeatureEncoder(ReplayBfmFeatureEncoder &&) noexcept;
  ReplayBfmFeatureEncoder &operator=(ReplayBfmFeatureEncoder &&) noexcept;

  ReplayBfmSparseFeatures encode(const SearchPosition &position);
  ReplayBfmSparseFeatures encode(const SearchPosition &position,
                                 Player perspective);
  PositionKey canonical_position_key(
      const SearchPosition &position) const noexcept;
  std::uint8_t canonical_slot_rank(const SearchPosition &position,
                                   std::uint8_t slot) const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

ReplayBfmSparseFeatures replay_bfm_sparse_features(const GameState &state);

class ReplayBfmModel {
 public:
  explicit ReplayBfmModel(std::string_view path);

  float evaluate(const ReplayBfmSparseFeatures &features) const noexcept;
  ReplayBfmPreparedEvaluation prepare(
      const ReplayBfmSparseFeatures &features) const noexcept;
  float evaluate_delta(const ReplayBfmPreparedEvaluation &base,
                       const ReplayBfmSparseFeatures &features) const noexcept;
  std::string_view model_sha256() const noexcept { return model_sha256_; }
  std::string_view payload_sha256() const noexcept { return payload_sha256_; }
  std::span<const float> weights() const noexcept { return weights_; }
  std::uint32_t runtime_version() const noexcept { return runtime_version_; }

 private:
  std::vector<float> weights_{};
  std::string model_sha256_{};
  std::string payload_sha256_{};
  std::uint32_t runtime_version_{1U};
  bool adapter_output_is_zero_{true};
};

std::string replay_bfm_sha256_hex(std::span<const std::uint8_t> bytes);

}  // namespace papersoccer::detail
