import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const inputPath = path.join(directory, "neural_puct_model.json");
const outputPath = path.join(directory, "neural_puct_model.hpp");
const checkOnly = process.argv[2] === "--check";

if (process.argv.length > 3 ||
    (process.argv.length === 3 && !checkOnly)) {
  throw new Error(
    "Usage: node generate_neural_puct_model_header.mjs [--check]",
  );
}

const source = fs.readFileSync(inputPath);
const report = JSON.parse(source.toString("utf8"));
const policyHead = report.policy_head ?? "legacy-directional";
const actionConditioned = policyHead === "shared-action-conditioned-v1";
if (report.schema !== "papersoccer.neural-puct-model.v1" ||
    report.feature_schema !== "neural-puct-features-v1" ||
    report.input_count !== 1286 || report.edge_count !== 316 ||
    report.vertex_count !== 105 || report.distance_buckets !== 8 ||
    report.global_count !== 25 || report.hidden_one !== 32 ||
    report.hidden_two !== 32 || report.policy_count !== 8 ||
    !["legacy-directional", "shared-action-conditioned-v1"].includes(policyHead) ||
    (actionConditioned &&
      (report.action_feature_count !== 16 || report.action_policy_hidden !== 8 ||
       !Array.isArray(report.action_feature_names) ||
       report.action_feature_names.length !== 16)) ||
    report.quantization !==
      "signed-int4-symmetric-per-output-channel") {
  throw new Error("Invalid neural PUCT model schema or dimensions");
}
const expectedGoldenInput = actionConditioned
  ? "stride17-mod11-action-mod13-v1"
  : "stride17-mod11-v1";
if (report.golden?.input !== expectedGoldenInput ||
    typeof report.golden.value !== "number" ||
    !Number.isFinite(report.golden.value) ||
    !Array.isArray(report.golden.policy) ||
    report.golden.policy.length !== 8 ||
    report.golden.policy.some((value) =>
      typeof value !== "number" || !Number.isFinite(value))) {
  throw new Error("Invalid neural PUCT quantized golden output");
}

const shapes = actionConditioned ? {
  w1: [1286, 32],
  w2: [32, 32],
  wv: [32, 1],
  wps: [32, 8],
  wpa: [16, 8],
  wpo: [8, 1],
} : {
  w1: [1286, 32],
  w2: [32, 32],
  wv: [32, 1],
  wp: [32, 8],
};

function signedNibble(value) {
  if (!Number.isInteger(value) || value < -7 || value > 7) {
    throw new Error(`Invalid symmetric int4 weight: ${value}`);
  }
  return value < 0 ? value + 16 : value;
}

function validatePacked(bytes, count, name) {
  if (bytes.length !== Math.ceil(count / 2)) {
    throw new Error(
      `${name} has ${bytes.length} packed bytes; expected ${Math.ceil(count / 2)}`,
    );
  }
  for (let index = 0; index < count; ++index) {
    const byte = bytes[Math.floor(index / 2)];
    const nibble = index % 2 === 0 ? byte & 15 : byte >> 4;
    if (nibble === 8) {
      throw new Error(`${name} contains forbidden asymmetric int4 value -8`);
    }
  }
  if (count % 2 !== 0 && (bytes.at(-1) >> 4) !== 0) {
    throw new Error(`${name} has a nonzero padding nibble`);
  }
}

const packedTensors = [];
const tensorScales = {};
for (const [name, shape] of Object.entries(shapes)) {
  const tensor = report.model?.[name];
  const count = shape.reduce((product, value) => product * value, 1);
  const outputs = shape.at(-1);
  if (!tensor || JSON.stringify(tensor.shape) !== JSON.stringify(shape) ||
      !Array.isArray(tensor.scales) || tensor.scales.length !== outputs ||
      tensor.scales.some((scale) => typeof scale !== "number" ||
        !Number.isFinite(scale) || scale <= 0)) {
    throw new Error(`Invalid neural PUCT tensor ${name}`);
  }
  const hasData = typeof tensor.data === "string";
  const hasFill = Object.hasOwn(tensor, "fill");
  if (hasData === hasFill) {
    throw new Error(`${name} must provide exactly one of data or fill`);
  }
  let bytes;
  if (hasData) {
    bytes = Buffer.from(tensor.data, "base64");
  } else {
    const nibble = signedNibble(tensor.fill);
    bytes = Buffer.alloc(Math.ceil(count / 2), nibble | (nibble << 4));
    if (count % 2 !== 0) {
      bytes[bytes.length - 1] &= 15;
    }
  }
  validatePacked(bytes, count, name);
  packedTensors.push(bytes);
  tensorScales[name] = tensor.scales;
}

const biases = actionConditioned ? {
  b1: 32,
  b2: 32,
  bv: 1,
  bpa: 8,
} : {
  b1: 32,
  b2: 32,
  bv: 1,
  bp: 8,
};
for (const [name, length] of Object.entries(biases)) {
  const values = report.model?.[name];
  if (!Array.isArray(values) || values.length !== length ||
      values.some((value) => typeof value !== "number" ||
        !Number.isFinite(value))) {
    throw new Error(`Invalid neural PUCT bias ${name}`);
  }
}

const floats = (values) => values
  .map((value) => `${Number(value).toPrecision(9)}F`)
  .join(", ");
const encodedWeights = Buffer.concat(packedTensors).toString("base64");
const artifactSha256 = crypto.createHash("sha256").update(source).digest("hex");
const unpackedWeightCount = Object.values(shapes)
  .map((shape) => shape.reduce((product, value) => product * value, 1))
  .reduce((sum, value) => sum + value, 0);

const lines = [
  "#pragma once",
  "",
  "#include <array>",
  "#include <cstddef>",
  "#include <string_view>",
  "",
  "namespace papersoccer::neural_puct::model_data {",
  "",
  `inline constexpr std::string_view kFeatureSchema = "${report.feature_schema}";`,
  `inline constexpr std::string_view kModelKind = "${report.model_kind}";`,
  `inline constexpr std::string_view kArtifactSha256 = "${artifactSha256}";`,
  `inline constexpr std::size_t kInputCount = ${report.input_count};`,
  `inline constexpr std::size_t kEdgeCount = ${report.edge_count};`,
  `inline constexpr std::size_t kVertexCount = ${report.vertex_count};`,
  `inline constexpr std::size_t kDistanceBuckets = ${report.distance_buckets};`,
  `inline constexpr std::size_t kGlobalCount = ${report.global_count};`,
  `inline constexpr std::size_t kHiddenOne = ${report.hidden_one};`,
  `inline constexpr std::size_t kHiddenTwo = ${report.hidden_two};`,
  `inline constexpr std::size_t kPolicyCount = ${report.policy_count};`,
  `inline constexpr bool kActionConditionedPolicy = ${actionConditioned};`,
  "inline constexpr std::size_t kActionFeatureCount = 16;",
  "inline constexpr std::size_t kActionPolicyHidden = 8;",
  `inline constexpr std::size_t kUnpackedWeightCount = ${unpackedWeightCount};`,
  `inline constexpr std::string_view kGoldenInput = "${report.golden.input}";`,
  `inline constexpr float kGoldenValue = ${floats([report.golden.value])};`,
  `inline constexpr std::array<float, 8> kGoldenPolicy{{${floats(report.golden.policy)}}};`,
  `inline constexpr std::array<float, 32> kW1Scales{{${floats(tensorScales.w1)}}};`,
  `inline constexpr std::array<float, 32> kW2Scales{{${floats(tensorScales.w2)}}};`,
  `inline constexpr std::array<float, 1> kValueScales{{${floats(tensorScales.wv)}}};`,
  `inline constexpr std::array<float, 8> kPolicyScales{{${floats(actionConditioned ? Array(8).fill(1) : tensorScales.wp)}}};`,
  `inline constexpr std::array<float, 8> kPolicyStateScales{{${floats(actionConditioned ? tensorScales.wps : Array(8).fill(1))}}};`,
  `inline constexpr std::array<float, 8> kPolicyActionScales{{${floats(actionConditioned ? tensorScales.wpa : Array(8).fill(1))}}};`,
  `inline constexpr std::array<float, 1> kPolicyOutputScales{{${floats(actionConditioned ? tensorScales.wpo : [1])}}};`,
  `inline constexpr std::array<float, 32> kB1{{${floats(report.model.b1)}}};`,
  `inline constexpr std::array<float, 32> kB2{{${floats(report.model.b2)}}};`,
  `inline constexpr float kValueBias = ${floats(report.model.bv)};`,
  `inline constexpr std::array<float, 8> kPolicyBias{{${floats(actionConditioned ? Array(8).fill(0) : report.model.bp)}}};`,
  `inline constexpr std::array<float, 8> kActionPolicyBias{{${floats(actionConditioned ? report.model.bpa : Array(8).fill(0))}}};`,
  `inline constexpr std::string_view kEncodedPackedWeights = "${encodedWeights}";`,
  "",
  "}  // namespace papersoccer::neural_puct::model_data",
  "",
];
const output = lines.join("\n");

if (checkOnly) {
  if (!fs.existsSync(outputPath) ||
      fs.readFileSync(outputPath, "utf8") !== output) {
    throw new Error("neural_puct_model.hpp is stale; regenerate it");
  }
  process.stdout.write(
    `Neural PUCT model is current (${artifactSha256}).\n`,
  );
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(
    `Wrote neural_puct_model.hpp (${output.length} characters).\n`,
  );
}
