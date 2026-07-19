import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const toolsDirectory = path.dirname(fileURLToPath(import.meta.url));
const codingameDirectory = path.resolve(toolsDirectory, "..");
const botsDirectory = path.join(codingameDirectory, "bots");
const repositoryRoot = path.resolve(codingameDirectory, "../..");

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function containedPath(root, relativePath, label) {
  if (typeof relativePath !== "string" || relativePath.length === 0 ||
      path.isAbsolute(relativePath)) {
    fail(`${label} must be a non-empty relative path.`);
  }
  const resolved = path.resolve(root, relativePath);
  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) {
    fail(`${label} escapes its allowed directory: ${relativePath}`);
  }
  return resolved;
}

const argumentsList = process.argv.slice(2);
if (argumentsList.length < 1 || argumentsList.length > 2 ||
    (argumentsList.length === 2 && argumentsList[1] !== "--check")) {
  fail(
    "Usage: node submissions/codingame/tools/generate_submission.mjs " +
      "<bot-name> [--check]",
  );
}

const botName = argumentsList[0];
if (!/^[a-z0-9][a-z0-9_-]*$/.test(botName)) {
  fail(`Invalid bot name: ${botName}`);
}
const checkOnly = argumentsList[1] === "--check";
const botDirectory = path.join(botsDirectory, botName);
const configPath = path.join(botDirectory, "submission.json");
if (!fs.existsSync(configPath)) {
  fail(`Missing bot configuration: ${path.relative(repositoryRoot, configPath)}`);
}

const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
if (config.schema !== "papersoccer.codingame-submission.v1" ||
    !Array.isArray(config.allowed_local_includes) ||
    !config.allowed_local_includes.every((value) => typeof value === "string")) {
  fail(`Invalid bot configuration: ${path.relative(repositoryRoot, configPath)}`);
}

const manifestPath = containedPath(
  botDirectory,
  config.sources ?? "sources.txt",
  "sources",
);
const outputPath = containedPath(
  botDirectory,
  config.output ?? "submission.cpp",
  "output",
);
const sourceLimit = config.source_limit ?? 100_000;
if (!Number.isInteger(sourceLimit) || sourceLimit <= 0) {
  fail("source_limit must be a positive integer.");
}

const generators = config.generators ?? [];
if (!Array.isArray(generators) ||
    !generators.every((value) => typeof value === "string")) {
  fail("generators must be an array of relative script paths.");
}
for (const generator of generators) {
  const generatorPath = containedPath(botDirectory, generator, "generator");
  const result = spawnSync(
    process.execPath,
    [generatorPath, ...(checkOnly ? ["--check"] : [])],
    { encoding: "utf8" },
  );
  if (result.status !== 0) {
    fail(result.stderr || `Generator failed: ${generator}`);
  }
  process.stdout.write(result.stdout);
}

if (!fs.existsSync(manifestPath)) {
  fail(`Missing source manifest: ${path.relative(repositoryRoot, manifestPath)}`);
}
const sources = fs.readFileSync(manifestPath, "utf8")
  .replaceAll("\r\n", "\n")
  .split("\n")
  .map((line) => line.trim())
  .filter((line) => line.length > 0 && !line.startsWith("#"));
if (sources.length === 0) {
  fail("The source manifest is empty.");
}

const allowedLocalIncludes = new Set(config.allowed_local_includes);
const systemIncludes = new Set();
const bodies = [];
for (const relativePath of sources) {
  const absolutePath = containedPath(repositoryRoot, relativePath, "source");
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
  `// Paper Soccer CodinGame submission: ${botName}.`,
  "// Paste this complete file into the C++ editor.",
  "// Rebuild with the shared generator in submissions/codingame/tools.",
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
const readableOutput = `${banner}${includes}\n\n${bodies.join("\n\n")}\n`;
const compactLines = [];
for (const line of readableOutput.split("\n")) {
  if (line.trimStart().startsWith("//")) {
    continue;
  }
  if (line.trim() === "" && compactLines.at(-1)?.trim() === "") {
    continue;
  }
  compactLines.push(line);
}
const output = compactLines.join("\n");

if ([...output].some((character) => character.codePointAt(0) > 0x7f)) {
  fail("Generated submission must contain ASCII characters only.");
}
if (output.length > sourceLimit) {
  fail(`Generated submission is ${output.length} characters; limit is ${sourceLimit}.`);
}
if (/^\s*#include\s*"/m.test(output)) {
  fail("Generated submission still contains a local include.");
}

if (checkOnly) {
  if (!fs.existsSync(outputPath) ||
      fs.readFileSync(outputPath, "utf8") !== output) {
    fail(`${path.relative(repositoryRoot, outputPath)} is stale; regenerate it.`);
  }
  process.stdout.write(
    `${botName} submission is current (${output.length} characters).\n`,
  );
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(
    `Wrote ${path.relative(repositoryRoot, outputPath)} ` +
      `(${output.length} characters).\n`,
  );
}
