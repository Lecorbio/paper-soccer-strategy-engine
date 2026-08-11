import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const inputPath = path.join(directory, "replay_value_model.json");
const outputPath = path.join(directory, "replay_value_model.hpp");
const checkOnly = process.argv[2] === "--check";

if (process.argv.length > 3 ||
    (process.argv.length === 3 && !checkOnly)) {
  throw new Error("Usage: node generate_replay_value_header.mjs [--check]");
}

const report = JSON.parse(fs.readFileSync(inputPath, "utf8"));
if (report.schema !== "papersoccer.replay-value-model.v1" ||
    report.input_count !== 1156 || report.edge_count !== 316 ||
    report.vertex_count !== 105) {
  throw new Error("Invalid replay value model schema or topology");
}

for (const name of ["w1", "w2", "w3"]) {
  const value = report.model?.[name];
  if (!value || !Array.isArray(value.shape) ||
      typeof value.scale !== "number" || typeof value.data !== "string") {
    throw new Error(`Invalid replay value tensor ${name}`);
  }
}
if (report.model.b1?.length !== 8 || report.model.b2?.length !== 8 ||
    report.model.b3?.length !== 1) {
  throw new Error("Invalid replay value model biases");
}

const weights = ["w1", "w2", "w3"].map((name) =>
  Buffer.from(report.model[name].data, "base64")
);
const encodedWeights = Buffer.concat(weights).toString("base64");
const floats = (values) => values
  .map((value) => Number(value).toPrecision(9) + "F")
  .join(", ");

const lines = [
  "#pragma once",
  "//f",
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
  "inline constexpr std::size_t kHiddenOne = 8;",
  "inline constexpr std::size_t kHiddenTwo = 8;",
  `inline constexpr float kW1Scale = ${Number(report.model.w1.scale).toPrecision(9)}F;`,
  `inline constexpr float kW2Scale = ${Number(report.model.w2.scale).toPrecision(9)}F;`,
  `inline constexpr float kW3Scale = ${Number(report.model.w3.scale).toPrecision(9)}F;`,
  `inline constexpr std::array<float, 8> kB1{{${floats(report.model.b1)}}};`,
  `inline constexpr std::array<float, 8> kB2{{${floats(report.model.b2)}}};`,
  `inline constexpr float kB3 = ${Number(report.model.b3[0]).toPrecision(9)}F;`,
  `inline constexpr std::string_view kEncodedWeights = "${encodedWeights}";`,
  "",
  "}  // namespace papersoccer::turn_action_v2::replay_value_model",
  "",
];
const output = lines.join("\n");

if (checkOnly) {
  if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== output) {
    throw new Error("replay_value_model.hpp is stale; regenerate it");
  }
  process.stdout.write(
    `Replay value model is current (${report.games} games).\n`,
  );
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(
    `Wrote replay_value_model.hpp (${output.length} characters).\n`,
  );
}
