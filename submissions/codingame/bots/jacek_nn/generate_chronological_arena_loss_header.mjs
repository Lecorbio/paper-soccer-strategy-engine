import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const outputPath = path.join(
  directory, "chronological_arena_loss_regressions.hpp");
const checkOnly = process.argv[2] === "--check";
if (process.argv.length > 3 ||
    (process.argv.length === 3 && !checkOnly)) {
  throw new Error(
    "Usage: node generate_chronological_arena_loss_header.mjs [--check]");
}

const files = fs.readdirSync(directory)
  .filter((name) => /^arena_batch_[0-9]+\.json$/.test(name)).sort();
const unique = new Map();
for (const file of files) {
  const report = JSON.parse(fs.readFileSync(path.join(directory, file), "utf8"));
  if (report.schema !== "papersoccer.completed-arena-loss-regressions.v1" ||
      !Array.isArray(report.regression_cases)) {
    throw new Error(`Invalid completed arena batch: ${file}`);
  }
  for (const item of report.regression_cases) {
    const key = `${item.player_id}:${item.prefix}`;
    if (!unique.has(key)) {
      unique.set(key, {
        label: `agent-${report.agent_id}-game-${item.game_id}-turn-${item.turn_index}`,
        player_id: item.player_id,
        prefix: item.prefix,
      });
    }
  }
}

const cases = [...unique.values()];
const lines = [
  "#pragma once",
  "",
  "#include <array>",
  "#include <string_view>",
  "",
  "namespace papersoccer::jacek_nn_chronological_regressions {",
  "struct Case { std::string_view label; int player_id; std::string_view prefix; };",
  `inline constexpr std::array<Case, ${cases.length}> kCases{{`,
  ...cases.map((item) =>
    `  {${JSON.stringify(item.label)}, ${item.player_id}, ${JSON.stringify(item.prefix)}},`),
  "}};",
  "}  // namespace papersoccer::jacek_nn_chronological_regressions",
  "",
];
const output = lines.join("\n");

if (checkOnly) {
  if (!fs.existsSync(outputPath) ||
      fs.readFileSync(outputPath, "utf8") !== output) {
    throw new Error(
      "chronological_arena_loss_regressions.hpp is stale; regenerate it");
  }
  process.stdout.write(
    `Chronological arena-loss regressions are current (${cases.length} cases).\n`);
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(
    `Wrote chronological_arena_loss_regressions.hpp (${cases.length} cases).\n`);
}
