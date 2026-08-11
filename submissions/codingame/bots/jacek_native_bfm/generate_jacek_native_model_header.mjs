import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(directory, "../../../..");
const exporter = path.join(
  root,
  "submissions/codingame/tools/generate_jacek_native_model.py",
);
const model = path.join(root, "models/jacek_native_bootstrap_model.json");
const output = path.join(directory, "jacek_native_model.hpp");
const argumentsList = [exporter, "--model", model, "--output", output];
if (process.argv.slice(2).includes("--check")) argumentsList.push("--check");
const result = spawnSync("python3", argumentsList, { encoding: "utf8" });
process.stdout.write(result.stdout ?? "");
process.stderr.write(result.stderr ?? "");
process.exit(result.status ?? 1);
