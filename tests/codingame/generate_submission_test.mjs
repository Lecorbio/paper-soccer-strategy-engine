import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const generator = path.join(root, "submissions/codingame/tools/generate_submission.mjs");

function fixture(run) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "papersoccer-submission-"));
  const tools = path.join(directory, "submissions/codingame/tools");
  const bot = path.join(directory, "submissions/codingame/bots/fixture");
  fs.mkdirSync(tools, { recursive: true });
  fs.mkdirSync(bot, { recursive: true });
  const script = path.join(tools, "generate_submission.mjs");
  fs.copyFileSync(generator, script);
  const invoke = (check = false) => spawnSync(process.execPath,
    [script, "fixture", ...(check ? ["--check"] : [])], { encoding: "utf8" });
  try {
    run({ directory, bot, invoke });
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

function standalone(bot, source, changes = {}) {
  const bytes = Buffer.from(source);
  fs.writeFileSync(path.join(bot, "submission.cpp"), bytes);
  const config = {
    schema: "papersoccer.codingame-submission.v1",
    mode: "standalone",
    source_limit: 99_999,
    allowed_local_includes: [],
    source_sha256: createHash("sha256").update(bytes).digest("hex"),
    ...changes,
  };
  fs.writeFileSync(path.join(bot, "submission.json"), `${JSON.stringify(config)}\n`);
}

function snapshot(directory) {
  const files = {};
  function visit(current) {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const file = path.join(current, entry.name);
      if (entry.isDirectory()) {
        visit(file);
      } else {
        const stat = fs.statSync(file, { bigint: true });
        files[path.relative(directory, file)] = {
          sha256: createHash("sha256").update(fs.readFileSync(file)).digest("hex"),
          mtimeNs: stat.mtimeNs.toString(),
        };
      }
    }
  }
  visit(directory);
  return files;
}

test("standalone normal invocation and --check preserve exact bytes and file timestamps", () => {
  fixture(({ directory, bot, invoke }) => {
    standalone(bot, "// Keep this comment and CRLF.\r\n#include <iostream>\r\nint main() { return 0; }\r\n");
    const before = snapshot(directory);
    for (const check of [false, true]) {
      const result = invoke(check);
      assert.equal(result.status, 0, result.stderr);
      assert.match(result.stdout, /standalone submission verified/);
      assert.deepEqual(snapshot(directory), before);
    }
  });
});

test("standalone rejects a corrupted source without rewriting it", () => {
  fixture(({ directory, bot, invoke }) => {
    standalone(bot, "int main() { return 0; }\n");
    fs.appendFileSync(path.join(bot, "submission.cpp"), "// changed\n");
    const before = snapshot(directory);
    for (const check of [false, true]) {
      const result = invoke(check);
      assert.equal(result.status, 1);
      assert.match(result.stderr, /source_sha256 digest/);
      assert.deepEqual(snapshot(directory), before);
    }
  });
});

test("standalone rejects non-ASCII and empty sources even when their hashes match", () => {
  for (const source of ["// café\nint main() {}\n", ""]) {
    fixture(({ bot, invoke }) => {
      standalone(bot, source);
      for (const check of [false, true]) {
        const result = invoke(check);
        assert.equal(result.status, 1);
        assert.match(result.stderr, /non-empty ASCII/);
      }
    });
  }
});

test("standalone enforces declared and platform size limits", () => {
  for (const [source, sourceLimit] of [["x".repeat(33), 32], ["x".repeat(100_000), 200_000]]) {
    fixture(({ bot, invoke }) => {
      standalone(bot, source, { source_limit: sourceLimit });
      const result = invoke();
      assert.equal(result.status, 1);
      assert.match(result.stderr, /characters; limit is/);
    });
  }
});

test("standalone refuses transformations before running a generator", () => {
  fixture(({ directory, bot, invoke }) => {
    fs.writeFileSync(path.join(bot, "mutate.mjs"),
      'import fs from "node:fs"; fs.writeFileSync(new URL("marker", import.meta.url), "changed");\n');
    standalone(bot, "int main() {}\n", { generators: ["mutate.mjs"] });
    const before = snapshot(directory);
    const result = invoke();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /cannot use generators or source transformations/);
    assert.deepEqual(snapshot(directory), before);
  });
});

test("standalone rejects missing digests, local includes, and escaped output paths", () => {
  const cases = [
    ["int main() {}\n", { source_sha256: undefined }, /require a source_sha256/],
    ['#include "engine.hpp"\n', {}, /cannot contain a local include/],
    ['# include "engine.hpp"\n', {}, /cannot contain a local include/],
    ['\t#\tinclude\t"engine.hpp"\n', {}, /cannot contain a local include/],
    ["int main() {}\n", { output: "../../outside.cpp" }, /escapes its allowed directory/],
  ];
  for (const [source, changes, message] of cases) {
    fixture(({ bot, invoke }) => {
      standalone(bot, source, changes);
      const result = invoke();
      assert.equal(result.status, 1);
      assert.match(result.stderr, message);
    });
  }
});

test("generated submissions still assemble sources and reject a stale --check", () => {
  fixture(({ bot, invoke }) => {
    fs.writeFileSync(path.join(bot, "submission.json"), JSON.stringify({
      schema: "papersoccer.codingame-submission.v1",
      allowed_local_includes: [],
      source_limit: 100_000,
    }));
    fs.writeFileSync(path.join(bot, "sources.txt"), "submissions/codingame/bots/fixture/bot.cpp\n");
    fs.writeFileSync(path.join(bot, "bot.cpp"),
      "// maintained comment\n#include <iostream>\n#include <iostream>\nint main() { return 0; }\n");
    let result = invoke();
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /Wrote .*submission.cpp/);
    const output = fs.readFileSync(path.join(bot, "submission.cpp"), "utf8");
    assert.equal((output.match(/#include <iostream>/g) ?? []).length, 1);
    assert.doesNotMatch(output, /maintained comment/);
    assert.match(output, /int main\(\)/);
    result = invoke(true);
    assert.equal(result.status, 0, result.stderr);
    fs.appendFileSync(path.join(bot, "bot.cpp"), "int extra = 1;\n");
    result = invoke(true);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /is stale/);
    assert.equal(fs.readFileSync(path.join(bot, "submission.cpp"), "utf8"), output);
  });
});
