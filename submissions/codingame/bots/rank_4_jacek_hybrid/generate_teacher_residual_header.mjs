import fs from "node:fs";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const provenanceDirectory = path.resolve(directory, "../selfplay_nn_v2");
const inputPath = path.join(directory, "teacher_residual_model.json");
const outputPath = path.join(directory, "teacher_residual_model.hpp");
const checkOnly = process.argv[2] === "--check";

if (process.argv.length > 3 ||
    (process.argv.length === 3 && !checkOnly)) {
  throw new Error("Usage: node generate_teacher_residual_header.mjs [--check]");
}

const report = JSON.parse(fs.readFileSync(inputPath, "utf8"));
if (report.schema !== "papersoccer.teacher-residual-model.v1" ||
    report.input_count !== 24 || report.target_scale !== 20000 ||
    report.hard_cap !== 6000 ||
    report.feature_schema !==
        "rank5-hidden2-hand-scalars-used-edge-phase-v2") {
  throw new Error("Invalid teacher residual model schema");
}
const sha256 = (file) => crypto.createHash("sha256")
  .update(fs.readFileSync(path.join(provenanceDirectory, file))).digest("hex");
if (report.corpus_sha256 !== sha256("teacher_samples.jsonl") ||
    report.trainer_sha256 !== sha256("train_teacher_residual.py") ||
    report.arena_loss_holdout_sha256 !==
        sha256("known_arena_loss_regressions.json")) {
  throw new Error("Teacher residual model provenance is stale; retrain it");
}
const weights = report.model?.weights;
const bias = report.model?.bias;
if (!Array.isArray(weights) || weights.length !== report.input_count ||
    weights.some((value) => typeof value !== "number" || !Number.isFinite(value)) ||
    typeof bias !== "number" || !Number.isFinite(bias)) {
  throw new Error("Invalid teacher residual model parameters");
}

const scalar = (value) => Number(value).toPrecision(9) + "F";
const lines = [
  "#pragma once",
  "",
  "#include <array>",
  "#include <cstddef>",
  "",
  "namespace papersoccer::turn_action_v2::teacher_residual_model {",
  "",
  `inline constexpr std::size_t kInputCount = ${report.input_count};`,
  `inline constexpr std::array<float, kInputCount> kWeights{{${weights.map(scalar).join(", ")}}};`,
  `inline constexpr float kBias = ${scalar(bias)};`,
  "",
  "}  // namespace papersoccer::turn_action_v2::teacher_residual_model",
  "",
];
const output = lines.join("\n");

if (checkOnly) {
  if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== output) {
    throw new Error("teacher_residual_model.hpp is stale; regenerate it");
  }
  process.stdout.write(
    `Teacher residual model is current (${report.samples.train + report.samples.validation + report.samples.test} samples).\n`,
  );
} else {
  fs.writeFileSync(outputPath, output, "utf8");
  process.stdout.write(`Wrote teacher_residual_model.hpp (${output.length} characters).\n`);
}
