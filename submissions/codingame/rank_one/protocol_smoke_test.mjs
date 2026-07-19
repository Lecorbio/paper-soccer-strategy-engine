import { spawnSync } from "node:child_process";

const executable = process.argv[2];
if (!executable) {
  process.stderr.write("Usage: node protocol_smoke_test.mjs <submission-executable>\n");
  process.exit(1);
}

function runCase(label, input) {
  const result = spawnSync(executable, [], {
    encoding: "utf8",
    input,
    timeout: 3000,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${label} exited with status ${result.status}: ${result.stderr}`);
  }

  const lines = result.stdout.trim().split(/\r?\n/);
  if (lines.length !== 1 || !/^[0-7]+$/.test(lines[0])) {
    throw new Error(`${label} produced invalid stdout: ${JSON.stringify(result.stdout)}`);
  }
}

runCase("Player 0", "0\n1\n-\n");
runCase("Player 1", "1\n1\n0\n");
process.stdout.write("Player 0 and Player 1 protocol smoke tests passed.\n");
