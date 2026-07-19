import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, "../..");
const manifestPath = path.join(scriptDirectory, "alpha_beta.sources");
const outputPath = path.join(scriptDirectory, "paper_soccer_alpha_beta.cpp");

const allowedLocalIncludes = new Set([
  "mcts_internal.hpp",
  "papersoccer/geometry.hpp",
  "papersoccer/rules.hpp",
  "papersoccer/types.hpp",
  "replay_value_model.hpp",
]);

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

const argumentsList = process.argv.slice(2);
if (argumentsList.length > 1 ||
    (argumentsList.length === 1 && argumentsList[0] !== "--check")) {
  fail("Usage: node submissions/codingame/generate_submission.mjs [--check]");
}
const checkOnly = argumentsList[0] === "--check";

const sources = fs.readFileSync(manifestPath, "utf8")
  .replaceAll("\r\n", "\n")
  .split("\n")
  .map((line) => line.trim())
  .filter((line) => line.length > 0 && !line.startsWith("#"));

const systemIncludes = new Set();
const bodies = [];
for (const relativePath of sources) {
  const absolutePath = path.join(repositoryRoot, relativePath);
  if (!fs.existsSync(absolutePath)) {
    fail(`Missing submission source: ${relativePath}`);
  }

  const keptLines = [];
  const lines = fs.readFileSync(absolutePath, "utf8")
    .replaceAll("\r\n", "\n")
    .split("\n");
  for (const line of lines) {
    if (line.trim() === "#pragma once") {
      continue;
    }

    const systemMatch = line.match(/^\s*#include\s*<([^>]+)>\s*$/);
    if (systemMatch) {
      systemIncludes.add(systemMatch[1]);
      continue;
    }

    const localMatch = line.match(/^\s*#include\s*"([^"]+)"\s*$/);
    if (localMatch) {
      if (!allowedLocalIncludes.has(localMatch[1])) {
        fail(`Unexpected local include in ${relativePath}: ${localMatch[1]}`);
      }
      continue;
    }

    keptLines.push(line);
  }

  while (keptLines.length > 0 && keptLines[0].trim() === "") {
    keptLines.shift();
  }
  while (keptLines.length > 0 && keptLines.at(-1).trim() === "") {
    keptLines.pop();
  }
  bodies.push(`// --- ${relativePath} ---\n${keptLines.join("\n")}`);
}

const banner = [
  "// Paper Soccer complete-turn AlphaBeta V2/replay correction.",
  "// Paste this complete file into the C++ editor.",
  "// Rebuild with: node submissions/codingame/generate_submission.mjs",
  "",
  "#if defined(__GNUG__) && !defined(__clang__)",
  "#pragma GCC optimize(\"O3\")",
  "#endif",
  "",
].join("\n");
const includes = [...systemIncludes]
  .sort()
  .map((header) => `#include <${header}>`)
  .join("\n");
const output = `${banner}${includes}\n\n${bodies.join("\n\n")}\n`;

if ([...output].some((character) => character.codePointAt(0) > 0x7f)) {
  fail("Generated submission must contain ASCII characters only.");
}
if (output.length > 100_000) {
  fail(`Generated submission is ${output.length} characters; limit is 100000.`);
}
if (/^\s*#include\s*"/m.test(output)) {
  fail("Generated submission still contains a local include.");
}

if (checkOnly) {
  if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== output) {
    fail("paper_soccer_alpha_beta.cpp is stale; regenerate it.");
  }
  process.stdout.write(
    `Submission is current (${output.length} characters).\n`,
  );
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(
    `Wrote ${path.relative(repositoryRoot, outputPath)} (${output.length} characters).\n`,
  );
}
