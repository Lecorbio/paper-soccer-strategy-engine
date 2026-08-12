import path from "node:path";
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
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
const roundOneExporter =
  "submissions/codingame/tools/generate_jacek_native_model.py";
if (!allowedExporters.has(declaredExporter)) {
  process.stderr.write("native model exporter is not a frozen purity dependency\n");
  process.exit(1);
}
const output = path.join(directory, "jacek_native_model.hpp");
const deployment = path.join(
  root,
  "models/jacek_native_round2_deployment.json",
);
let program;
let argumentsList;
if (existsSync(deployment)) {
  program = path.join(
    root,
    "submissions/codingame/tools/jacek_native_round2_activation.py",
  );
  argumentsList = [
    program,
    "generate",
    "--deployment",
    deployment,
    "--output",
    output,
  ];
} else {
  // Descriptor absence is an explicit rollback to the frozen round-one path,
  // independent of any later submission-metadata installation order.
  program = path.join(root, roundOneExporter);
  const model = path.join(root, "models/jacek_native_bootstrap_model.json");
  argumentsList = [program, "--model", model, "--output", output];
}
if (process.argv.slice(2).includes("--check")) argumentsList.push("--check");
const result = spawnSync("python3", argumentsList, { encoding: "utf8" });
process.stdout.write(result.stdout ?? "");
process.stderr.write(result.stderr ?? "");
process.exit(result.status ?? 1);
