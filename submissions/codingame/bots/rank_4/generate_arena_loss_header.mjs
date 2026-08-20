import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const inputPath = path.join(directory, "known_arena_loss_regressions.json");
const outputPath = path.join(directory, "arena_loss_regressions.hpp");
const checkOnly = process.argv[2] === "--check";
if (process.argv.length > 3 ||
    (process.argv.length === 3 && !checkOnly)) {
  throw new Error("Usage: node generate_arena_loss_header.mjs [--check]");
}

const report = JSON.parse(fs.readFileSync(inputPath, "utf8"));
if (report.schema !== "papersoccer.known-arena-loss-regressions.v1" ||
    !Array.isArray(report.cases) || report.cases.length === 0) {
  throw new Error("Invalid arena-loss regression schema");
}
for (const item of report.cases) {
  if (typeof item.label !== "string" ||
      !Number.isInteger(item.player_id) || item.player_id < 0 ||
      item.player_id > 1 || typeof item.prefix !== "string") {
    throw new Error("Invalid arena-loss regression case");
  }
}

const lines = [
  "#pragma once",
  "",
  "#include <array>",
  "#include <string_view>",
  "",
  "namespace papersoccer::rank_4_regressions {",
  "struct Case { std::string_view label; int player_id; std::string_view prefix; };",
  `inline constexpr std::array<Case, ${report.cases.length}> kCases{{`,
  ...report.cases.map((item) =>
    `  {${JSON.stringify(item.label)}, ${item.player_id}, ${JSON.stringify(item.prefix)}},`),
  "}};",
  "}  // namespace papersoccer::rank_4_regressions",
  "",
];
const output = lines.join("\n");
if (checkOnly) {
  if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== output) {
    throw new Error("arena_loss_regressions.hpp is stale; regenerate it");
  }
  process.stdout.write(`Arena-loss regressions are current (${report.cases.length} cases).\n`);
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(`Wrote arena_loss_regressions.hpp (${report.cases.length} cases).\n`);
}
