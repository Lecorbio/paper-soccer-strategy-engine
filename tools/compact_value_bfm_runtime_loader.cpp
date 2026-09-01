#include "compact_value_bfm_runtime_loader.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace papersoccer::compact_value_bfm_runtime {
namespace {

namespace cv = ::compact_value_bfm;

struct Json {
  enum class Kind { Object, Array, String, Number, Boolean, Null };
  Kind kind{Kind::Null};
  // Keep recursive object values behind a complete, copyable handle.  A
  // vector<pair<string, Json>> recursively asks libstdc++ whether Json is
  // default-constructible while Json itself is still incomplete; Clang with
  // libstdc++ correctly rejects that instantiation.
  std::vector<std::pair<std::string, std::shared_ptr<Json>>> object;
  std::vector<Json> array;
  std::string text;
  bool boolean{};
};

class Parser {
 public:
  explicit Parser(std::string_view input) : input_(input) {}

  Json parse() {
    Json result = value();
    whitespace();
    if (position_ != input_.size()) fail("trailing JSON data");
    return result;
  }

 private:
  std::string_view input_;
  std::size_t position_{};

  [[noreturn]] void fail(const char *message) const {
    throw std::invalid_argument(std::string("compact runtime JSON: ") + message);
  }

  void whitespace() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\t' ||
            input_[position_] == '\r' || input_[position_] == '\n')) {
      ++position_;
    }
  }

  bool consume(char character) {
    whitespace();
    if (position_ < input_.size() && input_[position_] == character) {
      ++position_;
      return true;
    }
    return false;
  }

  Json value() {
    whitespace();
    if (position_ >= input_.size()) fail("missing value");
    switch (input_[position_]) {
      case '{': return object();
      case '[': return array();
      case '"': {
        Json result;
        result.kind = Json::Kind::String;
        result.text = string();
        return result;
      }
      case 't': return literal("true", Json::Kind::Boolean, true);
      case 'f': return literal("false", Json::Kind::Boolean, false);
      case 'n': return literal("null", Json::Kind::Null, false);
      default: return number();
    }
  }

  Json literal(std::string_view token, Json::Kind kind, bool boolean) {
    if (input_.substr(position_, token.size()) != token) fail("invalid literal");
    position_ += token.size();
    Json result;
    result.kind = kind;
    result.boolean = boolean;
    return result;
  }

  static int hex(char character) noexcept {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
  }

  std::string string() {
    if (input_[position_++] != '"') fail("missing string opener");
    std::string result;
    while (position_ < input_.size()) {
      const unsigned char character = input_[position_++];
      if (character == '"') return result;
      if (character < 0x20U || character >= 0x80U) {
        fail("runtime strings must be canonical ASCII");
      }
      if (character != '\\') {
        result.push_back(static_cast<char>(character));
        continue;
      }
      if (position_ >= input_.size()) fail("truncated string escape");
      const char escaped = input_[position_++];
      switch (escaped) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        case 'u': {
          if (position_ + 4U > input_.size()) fail("truncated unicode escape");
          int code = 0;
          for (int index = 0; index < 4; ++index) {
            const int digit = hex(input_[position_++]);
            if (digit < 0) fail("invalid unicode escape");
            code = code * 16 + digit;
          }
          if (code < 0x20 || code >= 0x80) {
            fail("runtime unicode escape is outside canonical ASCII");
          }
          result.push_back(static_cast<char>(code));
          break;
        }
        default: fail("invalid string escape");
      }
    }
    fail("unterminated string");
  }

  Json object() {
    if (!consume('{')) fail("missing object opener");
    Json result;
    result.kind = Json::Kind::Object;
    if (consume('}')) return result;
    std::string previous;
    for (;;) {
      whitespace();
      if (position_ >= input_.size() || input_[position_] != '"') {
        fail("object key is not a string");
      }
      std::string key = string();
      if (!previous.empty() && key <= previous) {
        fail("object keys are not unique canonical sort order");
      }
      previous = key;
      if (!consume(':')) fail("missing object colon");
      result.object.emplace_back(
          std::move(key), std::make_shared<Json>(value()));
      if (consume('}')) return result;
      if (!consume(',')) fail("missing object comma");
    }
  }

  Json array() {
    if (!consume('[')) fail("missing array opener");
    Json result;
    result.kind = Json::Kind::Array;
    if (consume(']')) return result;
    for (;;) {
      result.array.push_back(value());
      if (consume(']')) return result;
      if (!consume(',')) fail("missing array comma");
    }
  }

  Json number() {
    whitespace();
    const std::size_t begin = position_;
    if (position_ < input_.size() && input_[position_] == '-') ++position_;
    if (position_ >= input_.size()) fail("truncated number");
    if (input_[position_] == '0') {
      ++position_;
      if (position_ < input_.size() && input_[position_] >= '0' &&
          input_[position_] <= '9') fail("number has a leading zero");
    } else {
      if (input_[position_] < '1' || input_[position_] > '9') {
        fail("invalid number integer part");
      }
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') ++position_;
    }
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      const std::size_t fraction = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') ++position_;
      if (position_ == fraction) fail("number has an empty fraction");
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) ++position_;
      const std::size_t exponent = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') ++position_;
      if (position_ == exponent) fail("number has an empty exponent");
    }
    Json result;
    result.kind = Json::Kind::Number;
    result.text = std::string(input_.substr(begin, position_ - begin));
    return result;
  }
};

std::string quote(std::string_view value) {
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result{"\""};
  for (const unsigned char character : value) {
    switch (character) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\b': result += "\\b"; break;
      case '\f': result += "\\f"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (character < 0x20U) {
          result += "\\u00";
          result.push_back(digits[character >> 4U]);
          result.push_back(digits[character & 15U]);
        } else {
          result.push_back(static_cast<char>(character));
        }
    }
  }
  result.push_back('"');
  return result;
}

std::string serialize(const Json &value, std::string_view omitted = {}) {
  switch (value.kind) {
    case Json::Kind::String: return quote(value.text);
    case Json::Kind::Number: return value.text;
    case Json::Kind::Boolean: return value.boolean ? "true" : "false";
    case Json::Kind::Null: return "null";
    case Json::Kind::Array: {
      std::string result{"["};
      for (std::size_t index = 0; index < value.array.size(); ++index) {
        if (index != 0U) result.push_back(',');
        result += serialize(value.array[index]);
      }
      result.push_back(']');
      return result;
    }
    case Json::Kind::Object: {
      std::string result{"{"};
      bool first = true;
      for (const auto &[key, child] : value.object) {
        if (key == omitted) continue;
        if (!first) result.push_back(',');
        first = false;
        result += quote(key);
        result.push_back(':');
        if (!child) throw std::logic_error("null JSON object child");
        result += serialize(*child);
      }
      result.push_back('}');
      return result;
    }
  }
  throw std::logic_error("unknown JSON kind");
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::invalid_argument("could not open compact runtime");
  const std::streampos end = input.tellg();
  if (end <= 0 || end > static_cast<std::streamoff>(16U * 1024U * 1024U)) {
    throw std::invalid_argument("compact runtime size is invalid");
  }
  std::string bytes(static_cast<std::size_t>(end), '\0');
  input.seekg(0, std::ios::beg);
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::invalid_argument("could not read complete compact runtime");
  return bytes;
}

std::string digest(std::string_view bytes) {
  return cv::sha256_hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
}

bool sha256(std::string_view value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

const Json &field(const Json &object, std::string_view key) {
  if (object.kind != Json::Kind::Object) {
    throw std::invalid_argument("compact runtime field parent is not an object");
  }
  const auto found = std::lower_bound(
      object.object.begin(), object.object.end(), key,
      [](const auto &entry, std::string_view expected) {
        return entry.first < expected;
      });
  if (found == object.object.end() || found->first != key) {
    throw std::invalid_argument("compact runtime is missing field " +
                                std::string(key));
  }
  if (!found->second) {
    throw std::logic_error("compact runtime contains a null object child");
  }
  return *found->second;
}

void keys(const Json &object,
          std::initializer_list<std::string_view> expected) {
  if (object.kind != Json::Kind::Object || object.object.size() != expected.size()) {
    throw std::invalid_argument("compact runtime object fields changed");
  }
  std::vector<std::string_view> sorted(expected);
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t index = 0; index < sorted.size(); ++index) {
    if (object.object[index].first != sorted[index]) {
      throw std::invalid_argument("compact runtime object fields changed");
    }
  }
}

std::string_view string(const Json &value, std::string_view label) {
  if (value.kind != Json::Kind::String) {
    throw std::invalid_argument(std::string(label) + " is not a string");
  }
  return value.text;
}

bool boolean(const Json &value, std::string_view label) {
  if (value.kind != Json::Kind::Boolean) {
    throw std::invalid_argument(std::string(label) + " is not a boolean");
  }
  return value.boolean;
}

template <typename Integer>
Integer integer(const Json &value, std::string_view label) {
  if (value.kind != Json::Kind::Number ||
      value.text.find_first_of(".eE") != std::string::npos) {
    throw std::invalid_argument(std::string(label) + " is not an integer");
  }
  Integer result{};
  const auto [end, error] = std::from_chars(
      value.text.data(), value.text.data() + value.text.size(), result);
  if (error != std::errc{} || end != value.text.data() + value.text.size()) {
    throw std::invalid_argument(std::string(label) + " is outside its range");
  }
  std::array<char, 64> canonical{};
  const auto rendered =
      std::to_chars(canonical.data(), canonical.data() + canonical.size(), result);
  if (rendered.ec != std::errc{} ||
      std::string_view(canonical.data(), rendered.ptr - canonical.data()) !=
          value.text) {
    throw std::invalid_argument(std::string(label) +
                                " is not canonically serialized");
  }
  return result;
}

std::string python_float(double value) {
  std::array<char, 128> buffer{};
  const auto rendered = std::to_chars(
      buffer.data(), buffer.data() + buffer.size(), value,
      std::chars_format::general);
  if (rendered.ec != std::errc{}) {
    throw std::invalid_argument("could not render canonical runtime float");
  }
  std::string result(buffer.data(), rendered.ptr);
  const std::size_t exponent_at = result.find('e');
  if (exponent_at != std::string::npos) {
    int exponent{};
    const auto parsed = std::from_chars(
        result.data() + exponent_at + 1U, result.data() + result.size(), exponent);
    if (parsed.ec != std::errc{} || parsed.ptr != result.data() + result.size()) {
      throw std::logic_error("canonical runtime float exponent");
    }
    // Python's JSON encoder retains fixed notation through decimal exponent
    // 15, while std::to_chars switches earlier on some standard libraries.
    if (exponent >= 0 && exponent < 16) {
      std::string digits = result.substr(0, exponent_at);
      const std::size_t point = digits.find('.');
      if (point != std::string::npos) digits.erase(point, 1U);
      const std::size_t decimal = static_cast<std::size_t>(exponent) + 1U;
      if (decimal < digits.size()) digits.insert(decimal, 1U, '.');
      else digits.append(decimal - digits.size(), '0');
      result = std::move(digits);
    }
  }
  if (result.find_first_of(".e") == std::string::npos) result += ".0";
  return result;
}

float positive_float32(const Json &value, std::string_view label) {
  if (value.kind != Json::Kind::Number) {
    throw std::invalid_argument(std::string(label) + " is not numeric");
  }
  float result{};
  const auto [end, error] = std::from_chars(
      value.text.data(), value.text.data() + value.text.size(), result,
      std::chars_format::general);
  if (error != std::errc{} || end != value.text.data() + value.text.size() ||
      !std::isfinite(result) || result <= 0.0F) {
    throw std::invalid_argument(std::string(label) + " is not positive float32");
  }
  double original{};
  const auto [double_end, double_error] = std::from_chars(
      value.text.data(), value.text.data() + value.text.size(), original,
      std::chars_format::general);
  if (double_error != std::errc{} ||
      double_end != value.text.data() + value.text.size() ||
      static_cast<double>(result) != original) {
    throw std::invalid_argument(std::string(label) +
                                " is not an exact serialized float32");
  }
  const std::string canonical = python_float(static_cast<double>(result));
  const bool canonical_integer =
      value.text.find_first_of(".eE") == std::string::npos &&
      canonical == value.text + ".0";
  if (canonical != value.text && !canonical_integer) {
    throw std::invalid_argument(std::string(label) +
                                " is not canonically serialized");
  }
  return result;
}

void exact_string(const Json &value, std::string_view expected,
                  std::string_view label) {
  if (string(value, label) != expected) {
    throw std::invalid_argument(std::string(label) + " changed");
  }
}

}  // namespace

LoadedRuntime load(const std::filesystem::path &path) {
  const std::string bytes = read_file(path);
  if (bytes.back() != '\n' ||
      bytes.find('\n') != bytes.size() - 1U) {
    throw std::invalid_argument("compact runtime is not one canonical JSON line");
  }
  const std::string_view document_bytes(bytes.data(), bytes.size() - 1U);
  const Json document = Parser(document_bytes).parse();
  if (serialize(document) != document_bytes) {
    throw std::invalid_argument("compact runtime JSON is not canonical");
  }
  keys(document, {"schema", "feature_schema", "architecture", "quantization",
                  "selection", "body_sha256"});
  exact_string(field(document, "schema"), kRuntimeSchema, "runtime schema");
  exact_string(field(document, "feature_schema"), kFeatureSchema,
               "runtime feature schema");
  const std::string claimed_body(
      string(field(document, "body_sha256"), "runtime body SHA-256"));
  const std::string body_bytes = serialize(document, "body_sha256") + "\n";
  if (!sha256(claimed_body) || digest(body_bytes) != claimed_body) {
    throw std::invalid_argument("compact runtime body SHA-256 mismatch");
  }

  const Json &architecture = field(document, "architecture");
  keys(architecture, {"name", "dimensions", "biases", "activations",
                      "payload_layout"});
  const Json &dimensions = field(architecture, "dimensions");
  if (dimensions.kind != Json::Kind::Array || dimensions.array.size() != 4U) {
    throw std::invalid_argument("compact runtime dimensions changed");
  }
  const std::size_t inputs = integer<std::size_t>(dimensions.array[0], "inputs");
  const std::size_t hidden_one =
      integer<std::size_t>(dimensions.array[1], "hidden one");
  const std::size_t hidden_two =
      integer<std::size_t>(dimensions.array[2], "hidden two");
  const std::size_t outputs = integer<std::size_t>(dimensions.array[3], "outputs");
  const std::string architecture_name(
      string(field(architecture, "name"), "architecture name"));
  const bool eligible =
      (hidden_one == 8U && hidden_two == 8U &&
       architecture_name == "compact-8x8") ||
      (hidden_one == 8U && hidden_two == 16U &&
       architecture_name == "source-neutral-8x16") ||
      (hidden_one == 12U && hidden_two == 8U &&
       architecture_name == "capacity-12x8");
  if (inputs != cv::kFeatureCount || outputs != 1U || !eligible ||
      boolean(field(architecture, "biases"), "architecture biases")) {
    throw std::invalid_argument("compact runtime architecture changed");
  }
  exact_string(field(architecture, "payload_layout"),
               "w1-input-major,w2-input-major,w3", "payload layout");
  const Json &activations = field(architecture, "activations");
  constexpr std::array<std::string_view, 3> expected_activations{{
      "square-leaky-0.01", "leaky-relu-0.01", "fast-tanh-rational-v1"}};
  if (activations.kind != Json::Kind::Array ||
      activations.array.size() != expected_activations.size()) {
    throw std::invalid_argument("compact runtime activations changed");
  }
  for (std::size_t index = 0; index < expected_activations.size(); ++index) {
    exact_string(activations.array[index], expected_activations[index],
                 "runtime activation");
  }

  const Json &quantization = field(document, "quantization");
  keys(quantization, {"bits", "minimum", "maximum", "scheme", "packing",
                      "scales", "weight_counts", "packed_byte_count",
                      "payload_sha256", "payload_base64"});
  if (integer<int>(field(quantization, "bits"), "quantization bits") != 3 ||
      integer<int>(field(quantization, "minimum"), "quantization minimum") != -3 ||
      integer<int>(field(quantization, "maximum"), "quantization maximum") != 3) {
    throw std::invalid_argument("compact runtime quantization bounds changed");
  }
  exact_string(field(quantization, "scheme"),
               "symmetric-signed-three-bit-per-layer-fixed-scale",
               "quantization scheme");
  exact_string(field(quantization, "packing"),
               "signed-three-bit-twos-complement-lsb-first",
               "quantization packing");
  const Json &scales = field(quantization, "scales");
  keys(scales, {"w1", "w2", "w3"});
  const float scale_one = positive_float32(field(scales, "w1"), "w1 scale");
  const float scale_two = positive_float32(field(scales, "w2"), "w2 scale");
  const float scale_three = positive_float32(field(scales, "w3"), "w3 scale");
  const std::size_t w1 = cv::kFeatureCount * hidden_one;
  const std::size_t w2 = hidden_one * hidden_two;
  const std::size_t w3 = hidden_two;
  const std::size_t total = w1 + w2 + w3;
  const Json &counts = field(quantization, "weight_counts");
  keys(counts, {"w1", "w2", "w3", "total"});
  if (integer<std::size_t>(field(counts, "w1"), "w1 count") != w1 ||
      integer<std::size_t>(field(counts, "w2"), "w2 count") != w2 ||
      integer<std::size_t>(field(counts, "w3"), "w3 count") != w3 ||
      integer<std::size_t>(field(counts, "total"), "total count") != total ||
      integer<std::size_t>(field(quantization, "packed_byte_count"),
                           "packed byte count") != (total * 3U + 7U) / 8U) {
    throw std::invalid_argument("compact runtime tensor sizes changed");
  }
  const std::string payload_sha(
      string(field(quantization, "payload_sha256"), "payload SHA-256"));
  const std::string payload(
      string(field(quantization, "payload_base64"), "payload base64"));
  if (!sha256(payload_sha)) {
    throw std::invalid_argument("compact runtime payload SHA-256 is invalid");
  }

  const Json &selection = field(document, "selection");
  keys(selection, {"arm", "seed", "float_epoch", "qat_epoch",
                   "source_bundle_body_sha256"});
  const std::string arm(string(field(selection, "arm"), "selection arm"));
  if (arm != "search-target" && arm != "teacher-assisted") {
    throw std::invalid_argument("compact runtime selection arm changed");
  }
  const std::uint64_t seed =
      integer<std::uint64_t>(field(selection, "seed"), "selection seed");
  const std::uint64_t float_epoch = integer<std::uint64_t>(
      field(selection, "float_epoch"), "selection float epoch");
  const std::uint64_t qat_epoch = integer<std::uint64_t>(
      field(selection, "qat_epoch"), "selection QAT epoch");
  const std::string source_bundle(string(
      field(selection, "source_bundle_body_sha256"), "source bundle SHA-256"));
  if ((seed != 20260907U && seed != 20260908U && seed != 20260909U) ||
      float_epoch == 0U || float_epoch > 50U || qat_epoch > 4U ||
      !sha256(source_bundle)) {
    throw std::invalid_argument("compact runtime selection binding changed");
  }

  auto model = std::make_unique<cv::QuantizedModel>(cv::ModelDescriptor{
      inputs, hidden_one, hidden_two, scale_one, scale_two, scale_three,
      payload, payload_sha, false});
  if (model->payload_sha256() != payload_sha) {
    throw std::logic_error("compact runtime/model payload identity mismatch");
  }
  return LoadedRuntime{
      Identity{digest(bytes), claimed_body, payload_sha, source_bundle,
               digest(serialize(selection) + "\n"), architecture_name, arm,
               seed},
      std::move(model)};
}

}  // namespace papersoccer::compact_value_bfm_runtime
