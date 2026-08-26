#include "jacek_replay_bfm/jacek_replay_bfm_internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace papersoccer::detail {
namespace {

constexpr std::array<std::uint8_t, 8> kV1Magic{
    'J', 'R', 'B', 'F', 'M', 0, 0, 1};
constexpr std::array<std::uint8_t, 8> kV2Magic{
    'J', 'R', 'B', 'F', 'M', 0, 0, 2};
constexpr std::size_t kV1PayloadBytes =
    kReplayBfmWeightCount * sizeof(float);
constexpr std::size_t kV2PayloadBytes =
    kReplayBfmRuntimeV2WeightCount * sizeof(float);
constexpr std::size_t kV1RuntimeBytes =
    kReplayBfmRuntimeHeaderBytes + kV1PayloadBytes;
constexpr std::size_t kV2RuntimeBytes =
    kReplayBfmRuntimeHeaderBytes + kV2PayloadBytes;

std::uint32_t rotate_right(std::uint32_t value, unsigned shift) noexcept {
  return (value >> shift) | (value << (32U - shift));
}

std::uint32_t read_u32(std::span<const std::uint8_t> bytes,
                       std::size_t offset) {
  if (offset + 4U > bytes.size()) {
    throw std::invalid_argument("truncated Jacek replay BFM checkpoint");
  }
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
         (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
         (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
}

std::uint64_t read_u64(std::span<const std::uint8_t> bytes,
                       std::size_t offset) {
  const std::uint64_t low = read_u32(bytes, offset);
  const std::uint64_t high = read_u32(bytes, offset + 4U);
  return low | (high << 32U);
}

std::string bytes_to_hex(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result(bytes.size() * 2U, '0');
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    result[index * 2U] = digits[bytes[index] >> 4U];
    result[index * 2U + 1U] = digits[bytes[index] & 0x0fU];
  }
  return result;
}

std::vector<std::uint8_t> read_checkpoint(std::string_view path) {
  if (path.empty()) {
    throw std::invalid_argument(
        "JacekReplayBfmBot requires an external model path");
  }
  std::ifstream input(std::string(path), std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::invalid_argument(
        "could not open Jacek replay BFM checkpoint");
  }
  const std::streampos end = input.tellg();
  if (end < 0 ||
      (static_cast<std::uint64_t>(end) != kV1RuntimeBytes &&
       static_cast<std::uint64_t>(end) != kV2RuntimeBytes)) {
    throw std::invalid_argument(
        "Jacek replay BFM checkpoint has a wrong size or trailing bytes");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::invalid_argument(
        "could not read complete Jacek replay BFM checkpoint");
  }
  return bytes;
}

}  // namespace

std::string replay_bfm_sha256_hex(std::span<const std::uint8_t> input) {
  constexpr std::array<std::uint32_t, 64> constants{{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
  }};

  std::vector<std::uint8_t> bytes(input.begin(), input.end());
  const std::uint64_t bit_count =
      static_cast<std::uint64_t>(bytes.size()) * 8U;
  bytes.push_back(0x80U);
  while (bytes.size() % 64U != 56U) {
    bytes.push_back(0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    bytes.push_back(static_cast<std::uint8_t>(bit_count >> shift));
  }

  std::array<std::uint32_t, 8> hash{{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  }};
  for (std::size_t offset = 0; offset < bytes.size(); offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      const std::size_t byte = offset + index * 4U;
      words[index] =
          (static_cast<std::uint32_t>(bytes[byte]) << 24U) |
          (static_cast<std::uint32_t>(bytes[byte + 1U]) << 16U) |
          (static_cast<std::uint32_t>(bytes[byte + 2U]) << 8U) |
          static_cast<std::uint32_t>(bytes[byte + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t left = words[index - 15U];
      const std::uint32_t right = words[index - 2U];
      const std::uint32_t sigma_zero =
          rotate_right(left, 7U) ^ rotate_right(left, 18U) ^ (left >> 3U);
      const std::uint32_t sigma_one =
          rotate_right(right, 17U) ^ rotate_right(right, 19U) ^
          (right >> 10U);
      words[index] = words[index - 16U] + sigma_zero +
                     words[index - 7U] + sigma_one;
    }

    std::uint32_t a = hash[0];
    std::uint32_t b = hash[1];
    std::uint32_t c = hash[2];
    std::uint32_t d = hash[3];
    std::uint32_t e = hash[4];
    std::uint32_t f = hash[5];
    std::uint32_t g = hash[6];
    std::uint32_t h = hash[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum_one =
          rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temporary_one =
          h + sum_one + choice + constants[index] + words[index];
      const std::uint32_t sum_zero =
          rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary_two = sum_zero + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary_one;
      d = c;
      c = b;
      b = a;
      a = temporary_one + temporary_two;
    }
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
  }

  constexpr std::string_view digits = "0123456789abcdef";
  std::string result(64U, '0');
  std::size_t position = 0;
  for (const std::uint32_t word : hash) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      result[position++] = digits[(word >> shift) & 0x0fU];
    }
  }
  return result;
}

ReplayBfmModel::ReplayBfmModel(std::string_view path) {
  static_assert(sizeof(float) == 4U);
  static_assert(std::numeric_limits<float>::is_iec559);

  const std::vector<std::uint8_t> checkpoint = read_checkpoint(path);
  const std::span<const std::uint8_t> bytes(checkpoint);
  const std::uint32_t version = read_u32(bytes, 12U);
  const std::uint64_t weight_count = read_u64(bytes, 48U);
  const bool is_v1 =
      std::equal(kV1Magic.begin(), kV1Magic.end(), bytes.begin()) &&
      version == 1U && weight_count == kReplayBfmWeightCount &&
      bytes.size() == kV1RuntimeBytes;
  const bool is_v2 =
      std::equal(kV2Magic.begin(), kV2Magic.end(), bytes.begin()) &&
      version == 2U && weight_count == kReplayBfmRuntimeV2WeightCount &&
      bytes.size() == kV2RuntimeBytes;
  if ((!is_v1 && !is_v2) ||
      read_u32(bytes, 8U) != kReplayBfmRuntimeHeaderBytes ||
      read_u32(bytes, 16U) != kReplayBfmInputCount ||
      read_u32(bytes, 20U) != kReplayBfmHiddenOne ||
      read_u32(bytes, 24U) != kReplayBfmHiddenTwo ||
      read_u32(bytes, 28U) != 1U ||
      read_u32(bytes, 32U) != 1U ||
      read_u32(bytes, 36U) != 2U ||
      read_u32(bytes, 40U) != 3U ||
      std::bit_cast<float>(read_u32(bytes, 44U)) != 0.01F) {
    throw std::invalid_argument(
        "Jacek replay BFM checkpoint header is incompatible");
  }
  if (std::any_of(bytes.begin() + 120, bytes.begin() + 128,
                  [](std::uint8_t value) { return value != 0U; })) {
    throw std::invalid_argument(
        "Jacek replay BFM checkpoint reserved header bytes are not zero");
  }

  const std::string schema_hash =
      bytes_to_hex(bytes.subspan(56U, 32U));
  const auto schema_bytes = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(kReplayBfmFeatureSchema.data()),
      kReplayBfmFeatureSchema.size());
  if (replay_bfm_sha256_hex(schema_bytes) !=
          kReplayBfmFeatureSchemaSha256 ||
      schema_hash != kReplayBfmFeatureSchemaSha256) {
    throw std::invalid_argument(
        "Jacek replay BFM checkpoint feature schema does not match");
  }

  const std::size_t payload_bytes =
      is_v2 ? kV2PayloadBytes : kV1PayloadBytes;
  const std::span<const std::uint8_t> payload =
      bytes.subspan(kReplayBfmRuntimeHeaderBytes, payload_bytes);
  payload_sha256_ = replay_bfm_sha256_hex(payload);
  if (payload_sha256_ != bytes_to_hex(bytes.subspan(88U, 32U))) {
    throw std::invalid_argument(
        "Jacek replay BFM checkpoint payload hash does not match");
  }

  weights_.resize(static_cast<std::size_t>(weight_count));
  for (std::size_t index = 0; index < weights_.size(); ++index) {
    const float weight =
        std::bit_cast<float>(read_u32(payload, index * sizeof(float)));
    if (!std::isfinite(weight)) {
      throw std::invalid_argument(
          "Jacek replay BFM checkpoint contains a non-finite weight");
    }
    weights_[index] = weight;
  }
  runtime_version_ = version;
  if (is_v2) {
    const std::size_t adapter_b_offset =
        kReplayBfmWeightCount + 2U +
        kReplayBfmHiddenOne * kReplayBfmResidualRank;
    adapter_output_is_zero_ = std::all_of(
        weights_.begin() + static_cast<std::ptrdiff_t>(adapter_b_offset),
        weights_.end(), [](float value) { return value == 0.0F; });
  }
  model_sha256_ = replay_bfm_sha256_hex(bytes);
}

ReplayBfmPreparedEvaluation ReplayBfmModel::prepare(
    const ReplayBfmSparseFeatures &features) const noexcept {
  ReplayBfmPreparedEvaluation result;
  result.base_features = features;
  for (std::size_t active = 0; active < features.count; ++active) {
    const std::size_t input = features.indices[active];
    const std::size_t offset = input * kReplayBfmHiddenOne;
    for (std::size_t hidden = 0; hidden < result.first_layer.size();
         ++hidden) {
      result.first_layer[hidden] += weights_[offset + hidden];
    }
  }
  return result;
}

float ReplayBfmModel::evaluate_delta(
    const ReplayBfmPreparedEvaluation &base,
    const ReplayBfmSparseFeatures &features) const noexcept {
  std::array<float, kReplayBfmHiddenOne> first = base.first_layer;
  std::size_t before = 0;
  std::size_t after = 0;
  while (before < base.base_features.count || after < features.count) {
    const std::uint16_t old_input =
        before < base.base_features.count
            ? base.base_features.indices[before]
            : std::numeric_limits<std::uint16_t>::max();
    const std::uint16_t new_input =
        after < features.count
            ? features.indices[after]
            : std::numeric_limits<std::uint16_t>::max();
    if (old_input == new_input) {
      ++before;
      ++after;
      continue;
    }
    const bool added = new_input < old_input;
    const std::size_t input = added ? new_input : old_input;
    const float scale = added ? 1.0F : -1.0F;
    const std::size_t offset = input * kReplayBfmHiddenOne;
    for (std::size_t hidden = 0; hidden < first.size(); ++hidden) {
      first[hidden] += scale * weights_[offset + hidden];
    }
    if (added) {
      ++after;
    } else {
      ++before;
    }
  }

  for (float &value : first) {
    value = value < 0.0F ? 0.01F * value : value * value;
  }

  std::array<float, kReplayBfmHiddenTwo> second{};
  const std::size_t second_offset =
      kReplayBfmInputCount * kReplayBfmHiddenOne;
  for (std::size_t input = 0; input < first.size(); ++input) {
    for (std::size_t hidden = 0; hidden < second.size(); ++hidden) {
      second[hidden] +=
          first[input] *
          weights_[second_offset + input * kReplayBfmHiddenTwo + hidden];
    }
  }
  for (float &value : second) {
    value = value < 0.0F ? 0.01F * value : value;
  }

  const std::size_t output_offset =
      second_offset + kReplayBfmHiddenOne * kReplayBfmHiddenTwo;
  float output = 0.0F;
  for (std::size_t hidden = 0; hidden < second.size(); ++hidden) {
    output += second[hidden] * weights_[output_offset + hidden];
  }
  if (runtime_version_ == 1U) {
    return std::tanh(output);
  }

  const std::size_t gain_offset = kReplayBfmWeightCount;
  const float gain = weights_[gain_offset];
  const float bias = weights_[gain_offset + 1U];
  if (gain == 1.0F && bias == 0.0F && adapter_output_is_zero_) {
    return std::tanh(output);
  }

  const std::size_t adapter_a_offset = gain_offset + 2U;
  const std::size_t adapter_b_offset =
      adapter_a_offset + kReplayBfmHiddenOne * kReplayBfmResidualRank;
  float residual = 0.0F;
  for (std::size_t rank = 0; rank < kReplayBfmResidualRank; ++rank) {
    float adapter_pre = 0.0F;
    for (std::size_t hidden = 0; hidden < first.size(); ++hidden) {
      adapter_pre +=
          first[hidden] *
          weights_[adapter_a_offset +
                   hidden * kReplayBfmResidualRank + rank];
    }
    const float adapter_hidden =
        adapter_pre < 0.0F ? 0.01F * adapter_pre : adapter_pre;
    residual += adapter_hidden * weights_[adapter_b_offset + rank];
  }
  return std::tanh(gain * output + bias + residual);
}

float ReplayBfmModel::evaluate(
    const ReplayBfmSparseFeatures &features) const noexcept {
  return evaluate_delta(prepare(ReplayBfmSparseFeatures{}), features);
}

}  // namespace papersoccer::detail
