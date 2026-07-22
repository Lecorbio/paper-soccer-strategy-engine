import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const inputPath = path.join(directory, "jacek_value_model.json");
const outputPath = path.join(directory, "jacek_value_model.hpp");
const trainerPath = path.join(directory, "train_jacek_value.py");
const holdoutPath = path.join(directory, "known_arena_loss_regressions.json");
const observedPath = path.join(directory, "observed_arena_loss_regressions.hpp");
const checkOnly = process.argv[2] === "--check";

if (process.argv.length > 3 ||
    (process.argv.length === 3 && !checkOnly)) {
  throw new Error("Usage: node generate_jacek_value_header.mjs [--check]");
}

const sha256 = (value) => crypto.createHash("sha256").update(value).digest("hex");
const report = JSON.parse(fs.readFileSync(inputPath, "utf8"));
if (report.schema !== "papersoccer.jacek-value-model.v1" ||
    report.input_count !== 1156 || report.edge_count !== 316 ||
    report.vertex_count !== 105 || report.hidden_one !== 32 ||
    report.hidden_two !== 32 || report.quantization_bits !== 3 ||
    report.feature_schema !==
        "canonical-edges316-onehot-true-turn-distance105x8-v1" ||
    report.target !== "mover-relative-soft-teacher-root-score" ||
    report.target_temperature !== 12000) {
  throw new Error("Invalid Jacek value model schema or topology");
}
if (report.trainer_sha256 !== sha256(fs.readFileSync(trainerPath))) {
  throw new Error("Jacek value model is stale for its trainer");
}
const holdoutBytes = Buffer.concat([
  fs.readFileSync(holdoutPath), fs.readFileSync(observedPath),
]);
if (report.arena_holdout_sha256 !== sha256(holdoutBytes)) {
  throw new Error("Jacek value model is stale for its arena holdout");
}

const dimensions = {
  w1: [1156, 32],
  w2: [32, 32],
  w3: [32, 1],
};
for (const [name, shape] of Object.entries(dimensions)) {
  const tensor = report.model?.[name];
  if (!tensor || tensor.bits !== 3 ||
      JSON.stringify(tensor.shape) !== JSON.stringify(shape) ||
      !Array.isArray(tensor.scales) || tensor.scales.length !== shape[1] ||
      typeof tensor.data !== "string") {
    throw new Error(`Invalid Jacek value tensor ${name}`);
  }
  const count = shape[0] * shape[1];
  if (Buffer.from(tensor.data, "base64").length !== Math.ceil(count * 3 / 8)) {
    throw new Error(`Invalid packed size for Jacek value tensor ${name}`);
  }
}
if (report.model.b1?.length !== 32 || report.model.b2?.length !== 32 ||
    report.model.b3?.length !== 1) {
  throw new Error("Invalid Jacek value model biases");
}

const floats = (values) => values
  .map((value) => Number(value).toPrecision(9) + "F")
  .join(", ");
const lines = [
  "#pragma once",
  "",
  "#include <array>",
  "#include <cstddef>",
  "#include <string_view>",
  "",
  "namespace papersoccer::turn_action_v2::replay_value_model {",
  "",
  `inline constexpr std::size_t kInputCount = ${report.input_count};`,
  `inline constexpr std::size_t kEdgeCount = ${report.edge_count};`,
  `inline constexpr std::size_t kVertexCount = ${report.vertex_count};`,
  `inline constexpr std::size_t kHiddenOne = ${report.hidden_one};`,
  `inline constexpr std::size_t kHiddenTwo = ${report.hidden_two};`,
  "inline constexpr std::size_t kQuantizationBits = 3;",
  `inline constexpr float kTargetTemperature = ${Number(report.target_temperature).toPrecision(9)}F;`,
  `inline constexpr std::array<float, 32> kW1Scales{{${floats(report.model.w1.scales)}}};`,
  `inline constexpr std::array<float, 32> kW2Scales{{${floats(report.model.w2.scales)}}};`,
  `inline constexpr float kW3Scale = ${Number(report.model.w3.scales[0]).toPrecision(9)}F;`,
  `inline constexpr std::array<float, 32> kB1{{${floats(report.model.b1)}}};`,
  `inline constexpr std::array<float, 32> kB2{{${floats(report.model.b2)}}};`,
  `inline constexpr float kB3 = ${Number(report.model.b3[0]).toPrecision(9)}F;`,
  `inline constexpr std::string_view kEncodedW1 = "${report.model.w1.data}";`,
  `inline constexpr std::string_view kEncodedW2 = "${report.model.w2.data}";`,
  `inline constexpr std::string_view kEncodedW3 = "${report.model.w3.data}";`,
  "",
  "}  // namespace papersoccer::turn_action_v2::replay_value_model",
  "",
];
const output = lines.join("\n");

if (checkOnly) {
  if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== output) {
    throw new Error("jacek_value_model.hpp is stale; regenerate it");
  }
  process.stdout.write(
    `Jacek value model is current (${Object.values(report.games).reduce((a, b) => a + b, 0)} games).\n`,
  );
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(
    `Wrote jacek_value_model.hpp (${output.length} characters).\n`,
  );
}
