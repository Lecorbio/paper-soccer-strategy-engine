import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const inputPath = path.join(directory, "neural_model.json");
const outputPath = path.join(directory, "neural_model.hpp");
const checkOnly = process.argv[2] === "--check";

if (process.argv.length > 3 ||
    (process.argv.length === 3 && !checkOnly)) {
  throw new Error("Usage: node generate_neural_model_header.mjs [--check]");
}

const report = JSON.parse(fs.readFileSync(inputPath, "utf8"));
if (report.schema !== "papersoccer.policy-value-model.v1" ||
    report.input_count !== 423 || report.edge_count !== 316 ||
    report.vertex_count !== 105 || report.hidden_one !== 16 ||
    report.hidden_two !== 16 || report.policy_count !== 8) {
  throw new Error("Invalid self-play policy/value model schema or dimensions");
}

const shapes = {
  w1: [423, 16],
  w2: [16, 16],
  wv: [16, 1],
  wp: [16, 8],
};
const tensors = [];
for (const [name, shape] of Object.entries(shapes)) {
  const tensor = report.model?.[name];
  if (!tensor || JSON.stringify(tensor.shape) !== JSON.stringify(shape) ||
      typeof tensor.scale !== "number" || !Number.isFinite(tensor.scale) ||
      tensor.scale <= 0 || typeof tensor.data !== "string") {
    throw new Error(`Invalid policy/value tensor ${name}`);
  }
  const bytes = Buffer.from(tensor.data, "base64");
  const expected = shape.reduce((product, value) => product * value, 1);
  if (bytes.length !== expected) {
    throw new Error(`Policy/value tensor ${name} has ${bytes.length} bytes; expected ${expected}`);
  }
  tensors.push(bytes);
}
for (const [name, length] of Object.entries({b1: 16, b2: 16, bv: 1, bp: 8})) {
  const values = report.model?.[name];
  if (!Array.isArray(values) || values.length !== length ||
      values.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new Error(`Invalid policy/value bias ${name}`);
  }
}

const encodedWeights = Buffer.concat(tensors).toString("base64");
const floats = (values) => values
  .map((value) => Number(value).toPrecision(9) + "F")
  .join(", ");
const scalar = (value) => Number(value).toPrecision(9) + "F";

const lines = [
  "#pragma once",
  "",
  "#include <array>",
  "#include <cstddef>",
  "#include <string_view>",
  "",
  "namespace papersoccer::selfplay_nn::neural_model {",
  "",
  `inline constexpr std::size_t kInputCount = ${report.input_count};`,
  `inline constexpr std::size_t kEdgeCount = ${report.edge_count};`,
  `inline constexpr std::size_t kVertexCount = ${report.vertex_count};`,
  `inline constexpr std::size_t kHiddenOne = ${report.hidden_one};`,
  `inline constexpr std::size_t kHiddenTwo = ${report.hidden_two};`,
  `inline constexpr std::size_t kPolicyCount = ${report.policy_count};`,
  `inline constexpr float kW1Scale = ${scalar(report.model.w1.scale)};`,
  `inline constexpr float kW2Scale = ${scalar(report.model.w2.scale)};`,
  `inline constexpr float kValueScale = ${scalar(report.model.wv.scale)};`,
  `inline constexpr float kPolicyScale = ${scalar(report.model.wp.scale)};`,
  `inline constexpr std::array<float, 16> kB1{{${floats(report.model.b1)}}};`,
  `inline constexpr std::array<float, 16> kB2{{${floats(report.model.b2)}}};`,
  `inline constexpr float kValueBias = ${scalar(report.model.bv[0])};`,
  `inline constexpr std::array<float, 8> kPolicyBias{{${floats(report.model.bp)}}};`,
  `inline constexpr std::string_view kEncodedWeights = "${encodedWeights}";`,
  "",
  "}  // namespace papersoccer::selfplay_nn::neural_model",
  "",
];
const output = lines.join("\n");

if (checkOnly) {
  if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== output) {
    throw new Error("neural_model.hpp is stale; regenerate it");
  }
  process.stdout.write(
    `Self-play neural model is current (${report.games.train + report.games.validation + report.games.test} games).\n`,
  );
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(
    `Wrote neural_model.hpp (${output.length} characters).\n`,
  );
}
