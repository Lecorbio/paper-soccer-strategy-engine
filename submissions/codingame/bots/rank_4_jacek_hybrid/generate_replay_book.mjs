import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const inputPath = path.join(directory, "replay_book.json");
const outputPath = path.join(directory, "replay_book.hpp");
const checkOnly = process.argv[2] === "--check";

if (process.argv.length > 3 ||
    (process.argv.length === 3 && !checkOnly)) {
  throw new Error("Usage: node generate_replay_book.mjs [--check]");
}

const book = JSON.parse(fs.readFileSync(inputPath, "utf8"));
if (book.schema !== "papersoccer.replay-book.v1" ||
    !Array.isArray(book.replays) || book.replays.length === 0) {
  throw new Error("Invalid replay book schema");
}

const decisions = new Map();
for (const [index, replay] of book.replays.entries()) {
  const actions = replay.transcript.split("/");
  if ((replay.player_id !== 0 && replay.player_id !== 1) ||
      !Number.isInteger(replay.first_turn) || replay.first_turn < 0 ||
      replay.first_turn >= actions.length ||
      replay.first_turn % 2 !== replay.player_id ||
      actions.some((action) => !/^[0-7]+$/.test(action))) {
    throw new Error(`Invalid replay ${index}: ${replay.label}`);
  }
  for (let turn = replay.first_turn; turn < actions.length; turn += 2) {
    const prefix = actions.slice(0, turn).join("/");
    const key = `${replay.player_id}:${prefix}`;
    const previous = decisions.get(key);
    if (previous !== undefined && previous !== actions[turn]) {
      throw new Error(`Conflicting replay decision at ${key}`);
    }
    decisions.set(key, actions[turn]);
  }
}

const lines = [
  "#pragma once",
  "",
  "#include <array>",
  "#include <cstdint>",
  "#include <string_view>",
  "",
  "namespace papersoccer::turn_action_v2::replay_book {",
  "",
  "struct Replay {",
  "  std::uint8_t player_id;",
  "  std::uint8_t first_turn;",
  "  std::string_view transcript;",
  "};",
  "",
  `inline constexpr std::array<Replay, ${book.replays.length}> kReplays{{`,
];
for (const replay of book.replays) {
  lines.push(
    `    {${replay.player_id}, ${replay.first_turn}, "${replay.transcript}"},`,
  );
}
lines.push(
  "}};",
  "",
  "}  // namespace papersoccer::turn_action_v2::replay_book",
  "",
);
const output = lines.join("\n");

if (checkOnly) {
  if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== output) {
    throw new Error("replay_book.hpp is stale; regenerate it");
  }
  process.stdout.write(
    `Replay book is current (${book.replays.length} paths, ${decisions.size} decisions).\n`,
  );
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(
    `Wrote replay_book.hpp (${book.replays.length} paths, ${decisions.size} decisions).\n`,
  );
}
