import path from "node:path";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(directory, "../../../..");
const submission = JSON.parse(readFileSync(
  path.join(directory, "submission.json"), "utf8",
));
const declaredExporter = submission?.purity_dependencies?.[3];
const allowedExporters = new Set([
  "submissions/codingame/tools/generate_jacek_native_model.py",
  "submissions/codingame/tools/generate_jacek_native_model_round2.py",
]);
if (!allowedExporters.has(declaredExporter)) {
  process.stderr.write("native model exporter is not a frozen purity dependency\n");
  process.exit(1);
}
const exporter = path.join(root, declaredExporter);
const model = path.join(root, "models/jacek_native_bootstrap_model.json");
const output = path.join(directory, "jacek_native_model.hpp");
const argumentsList = [exporter, "--model", model, "--output", output];
if (process.argv.slice(2).includes("--check")) argumentsList.push("--check");
const result = spawnSync("python3", argumentsList, { encoding: "utf8" });
process.stdout.write(result.stdout ?? "");
process.stderr.write(result.stderr ?? "");
process.exit(result.status ?? 1);
