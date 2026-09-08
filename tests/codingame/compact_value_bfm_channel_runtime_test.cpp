#include "submissions/codingame/bots/compact_value_bfm/engine.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace cv = compact_value_bfm;

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void invalid(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("malformed channel descriptor accepted");
}

std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }

std::string base64(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 3) {
    const bool b = offset + 1 < bytes.size(), c = offset + 2 < bytes.size();
    const auto word = (static_cast<std::uint32_t>(bytes[offset]) << 16U) |
        (b ? static_cast<std::uint32_t>(bytes[offset + 1]) << 8U : 0U) |
        (c ? bytes[offset + 2] : 0U);
    result += alphabet[(word >> 18U) & 63U];
    result += alphabet[(word >> 12U) & 63U];
    result += b ? alphabet[(word >> 6U) & 63U] : '=';
    result += c ? alphabet[word & 63U] : '=';
  }
  return result;
}

struct Fixture {
  static constexpr std::size_t count = 6301 * 12 + 12 * 8 + 8;
  std::vector<std::int8_t> codes = std::vector<std::int8_t>(count);
  std::vector<std::uint8_t> packed = std::vector<std::uint8_t>((count * 3 + 7) / 8);
  std::array<float, 12> one{};
  std::array<float, 8> two{};
  std::array<float, 1> three{0.03125F};
  std::string encoded, hash;

  Fixture() {
    std::uint32_t state = 0x76543210U;
    for (std::size_t index = 0; index < count; ++index) {
      state = state * 1664525U + 1013904223U;
      codes[index] = static_cast<std::int8_t>(static_cast<int>((state >> 16U) % 7U) - 3);
      const auto code = static_cast<std::uint8_t>(codes[index]) & 7U;
      const auto bit = index * 3;
      packed[bit / 8] |= static_cast<std::uint8_t>(code << (bit % 8));
      if (bit % 8 > 5) packed[bit / 8 + 1] |= static_cast<std::uint8_t>(code >> (8 - bit % 8));
    }
    for (std::size_t i = 0; i < one.size(); ++i) one[i] = static_cast<float>(i + 1) / 2048.0F;
    for (std::size_t i = 0; i < two.size(); ++i) two[i] = static_cast<float>(i + 1) / 256.0F;
    refresh();
  }

  void refresh() { encoded = base64(packed); hash = cv::sha256_hex(packed); }
  void set_code(std::size_t index, std::int8_t code) {
    codes[index] = code;
    for (unsigned bit = 0; bit < 3; ++bit) {
      const auto position = index * 3 + bit;
      const auto mask = static_cast<std::uint8_t>(1U << (position % 8));
      packed[position / 8] &= static_cast<std::uint8_t>(~mask);
      if ((static_cast<std::uint8_t>(code) >> bit) & 1U) packed[position / 8] |= mask;
    }
    refresh();
  }
  cv::ChannelModelDescriptor descriptor() const {
    return {6301, 12, 8, one, two, three, encoded, hash};
  }
};

// Independent input-major scalar loop, including the legacy float operation
// order. This never invokes prepare/finish or the model's activation buffers.
float reference(const Fixture &fixture, const cv::SparseFeatures &features) {
  std::array<float, 12> first{};
  for (std::size_t j = 0; j < 12; ++j) {
    std::int32_t sum = 0;
    for (std::size_t row = 0; row < features.count; ++row)
      sum += fixture.codes[features.indices[row] * 12 + j];
    const float value = static_cast<float>(sum) * fixture.one[j];
    first[j] = value < 0.0F ? 0.01F * value : value * value;
  }
  std::array<float, 8> second{};
  for (std::size_t j = 0; j < 8; ++j) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < 12; ++i) {
      volatile float scaled = first[i] * fixture.two[j];
      volatile float term = scaled * static_cast<float>(fixture.codes[6301 * 12 + i * 8 + j]);
      sum = sum + term;
    }
    second[j] = sum < 0.0F ? 0.01F * sum : sum;
  }
  float sum = 0.0F;
  for (std::size_t i = 0; i < 8; ++i) {
    volatile float scaled = second[i] * fixture.three[0];
    volatile float term = scaled * static_cast<float>(fixture.codes[6301 * 12 + 12 * 8 + i]);
    sum = sum + term;
  }
  return cv::fast_tanh(sum);
}

void inference_and_owned_scales() {
  Fixture varying, equal;
  equal.one.fill(0.015625F); equal.two.fill(0.03125F);
  cv::QuantizedModel channel(varying.descriptor()), repeated(equal.descriptor());
  cv::QuantizedModel legacy(cv::ModelDescriptor{6301, 12, 8, equal.one[0], equal.two[0],
      equal.three[0], equal.encoded, equal.hash, false});
  require(std::equal(channel.weights().begin(), channel.weights().end(), varying.codes.begin()),
          "channel packed tensor layout changed");
  // Descriptors expose borrowed spans; construction must retain owned copies.
  auto copied_fixture = varying;
  cv::QuantizedModel owned(copied_fixture.descriptor());
  copied_fixture.one.fill(1.0F); copied_fixture.two.fill(1.0F); copied_fixture.three.fill(1.0F);
  const auto origin = cv::active_features(cv::initial_state());
  const auto base = channel.prepare(origin);
  const auto repeated_base = repeated.prepare(origin);
  std::size_t observed = 0, different = 0, nonzero = 0;
  for (unsigned path = 0; path < 8; ++path) {
    auto state = cv::initial_state();
    for (unsigned step = 0; step < 48; ++step) {
      const auto rotated = cv::rotate_and_swap(state);
      for (std::uint8_t perspective = 0; perspective < 2; ++perspective) {
        const auto features = cv::active_features(state, perspective);
        require(features == cv::active_features(rotated, 1 - perspective),
                "perspective rotation features changed");
        const auto actual = bits(channel.evaluate(features));
        require(actual == bits(reference(varying, features)), "channel scalar reference mismatch");
        require(actual == bits(channel.evaluate_delta(base, features)), "channel delta mismatch");
        require(actual == bits(owned.evaluate(features)), "channel scales retained borrowed memory");
        require(bits(repeated.evaluate(features)) == bits(legacy.evaluate(features)),
                "equal output scales changed legacy float behavior");
        require(bits(repeated.evaluate_delta(repeated_base, features)) == bits(legacy.evaluate(features)),
                "equal output scales changed legacy delta behavior");
        different += actual != bits(repeated.evaluate(features));
        nonzero += channel.evaluate(features) != 0.0F;
        ++observed;
      }
      if (state.terminal()) break;
      std::array<cv::Topology::Arc, 8> arcs{};
      const auto count = cv::legal_arcs(state, arcs);
      if (!count) break;
      require(cv::apply_edge(state, arcs[(step * 13 + path * 3) % count].direction),
              "bounded legal fixture edge failed");
    }
  }
  require(observed > 100 && different > 100 && nonzero > 100,
          "channel fixtures failed to exercise distinct nonsaturated evaluations");
}

void malformed_descriptors() {
  Fixture f;
  auto d = f.descriptor(); d.hidden_one = 8;
  invalid([&] { cv::QuantizedModel model(d); });
  d = f.descriptor(); d.hidden_two = 16;
  invalid([&] { cv::QuantizedModel model(d); });
  d = f.descriptor(); d.inputs = 6300;
  invalid([&] { cv::QuantizedModel model(d); });
  d = f.descriptor(); d.scales_one = std::span(f.one).first(11);
  invalid([&] { cv::QuantizedModel model(d); });
  d = f.descriptor(); d.scales_two = std::span(f.two).first(7);
  invalid([&] { cv::QuantizedModel model(d); });
  d = f.descriptor(); d.scales_three = {};
  invalid([&] { cv::QuantizedModel model(d); });
  for (float value : {0.0F, -0.0F, -0.125F, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    for (int layer = 0; layer < 3; ++layer) {
      Fixture bad = f;
      (layer == 0 ? bad.one[0] : layer == 1 ? bad.two[0] : bad.three[0]) = value;
      invalid([&] { cv::QuantizedModel model(bad.descriptor()); });
    }
  }
  f.one[0] = std::numeric_limits<float>::denorm_min();
  cv::QuantizedModel accepts_positive_subnormal(f.descriptor());
  d = f.descriptor(); d.packed_base64 = {};
  invalid([&] { cv::QuantizedModel model(d); });
  d = f.descriptor(); d.packed_sha256 = std::string_view("bad");
  invalid([&] { cv::QuantizedModel model(d); });
  f.packed[0] = static_cast<std::uint8_t>((f.packed[0] & ~7U) | 4U); f.refresh();
  invalid([&] { cv::QuantizedModel model(f.descriptor()); });
  f = Fixture(); f.packed.back() |= 16U; f.refresh();
  invalid([&] { cv::QuantizedModel model(f.descriptor()); });
  f = Fixture(); f.packed.pop_back(); f.refresh();
  invalid([&] { cv::QuantizedModel model(f.descriptor()); });
  f = Fixture(); f.hash[0] = f.hash[0] == '0' ? '1' : '0';
  invalid([&] { cv::QuantizedModel model(f.descriptor()); });
  f = Fixture(); f.encoded.push_back('=');
  invalid([&] { cv::QuantizedModel model(f.descriptor()); });
}

void effective_weight_overflow_is_rejected_only_for_channels() {
  for (int layer = 0; layer < 3; ++layer) {
    Fixture fixture;
    std::fill(fixture.codes.begin(), fixture.codes.end(), 0);
    std::fill(fixture.packed.begin(), fixture.packed.end(), 0);
    // Last input/output catches W2's nonzero packed offset (75612 % 8 == 4).
    const std::size_t index = layer == 0 ? 6301 * 12 - 1
                             : layer == 1 ? 6301 * 12 + 12 * 8 - 1
                                          : Fixture::count - 1;
    (layer == 0 ? fixture.one.back() : layer == 1 ? fixture.two.back() : fixture.three[0]) =
        std::numeric_limits<float>::max();
    for (std::int8_t code : {-3, -2, 2, 3}) {
      fixture.set_code(index, code);
      invalid([&] { cv::QuantizedModel model(fixture.descriptor()); });
    }
    for (std::int8_t code : {-1, 0, 1}) {
      fixture.set_code(index, code);
      cv::QuantizedModel finite_effective_weights(fixture.descriptor());
    }
  }
  Fixture legacy;
  cv::QuantizedModel unchanged_v1(cv::ModelDescriptor{6301, 12, 8,
      std::numeric_limits<float>::max(), 0.125F, 0.125F,
      legacy.encoded, legacy.hash, false});
}

}  // namespace

int main() {
  try {
    inference_and_owned_scales();
    malformed_descriptors();
    effective_weight_overflow_is_rejected_only_for_channels();
    std::cout << "channel runtime full/delta/perspective and descriptor checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
