import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../../", import.meta.url);
const urls = {
  gameHtml: new URL("web/index.html", root),
  html: new URL("web/jacek-native/index.html", root),
  css: new URL("web/jacek-native/status.css", root),
  script: new URL("web/jacek-native/status.js", root),
  ledger: new URL("web/jacek-native/status.json", root),
};
const [gameHtml, html, css, script, ledgerText] = await Promise.all(
  Object.values(urls).map((url) => readFile(url, "utf8")),
);
const ledger = JSON.parse(ledgerText);

await import("../../web/jacek-native/status.js");
const statusPage = globalThis.PaperSoccerJacekNativeStatus;

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

test("the game header links to the public native research ledger", () => {
  assert.match(
    gameHtml,
    /<a\b(?=[^>]*\bclass="[^"]*\bresearch-link\b[^"]*")(?=[^>]*\bhref="jacek-native\/index\.html")[^>]*>[\s\S]*?Native bot status\s*<\/a>/,
  );
});

test("the native status page is semantic, responsive, and independent of Wasm", () => {
  assert.match(html, /<html\b[^>]*\blang="en"/i);
  assert.match(html, /<main\b[^>]*\bclass="status-shell"/i);
  assert.match(html, /<caption\b/i);
  assert.match(html, /<th\b[^>]*\bscope="col"/i);
  assert.match(html, /<nav\b[^>]*\baria-label="Site"/i);
  assert.match(html, /id="auditorLink"/);
  assert.match(html, /<div\b[^>]*\brole="alert"/i);
  assert.doesNotMatch(html, /papersoccer-wasm|<canvas\b|game-engine|board-view/i);
  assert.match(css, /@media\s*\([^)]*max-width/i);
  assert.match(css, /overflow-x:\s*auto/i);
  assert.match(css, /:focus-visible/i);
  assert.match(script, /textContent/);
  assert.doesNotMatch(script, /innerHTML/);
});

test("the checked-in ledger is conservative and exposes no live claim", () => {
  assert.equal(statusPage.validate(ledger), ledger);
  assert.equal(ledger.schema, "papersoccer.jacek-native-status.v1");
  assert.equal(ledger.candidate.stage, "stopped-external-gate");
  assert.equal(ledger.candidate.status, "failed");
  assert.match(ledger.candidate.claim, /external Rank 4 screen failed 35–71/i);
  assert.match(ledger.candidate.claim, /no parity run or upload/i);
  assert.equal(ledger.live.status, "not-run");
  assert.equal(ledger.live.submissionId, null);
  assert.equal(ledger.live.agentId, null);
  assert.equal(ledger.live.rank, null);
  assert.equal(ledger.live.score, null);
  assert.equal(ledger.live.sourceBound, false);
  assert.match(ledger.live.label, /No upload/i);
  assert.equal(ledger.training.status, "hardened-bootstrap");
  assert.equal(ledger.training.games, 10_000);
  assert.equal(ledger.training.shards, 14);
  assert.equal(ledger.training.augmentedSamples, 1_106_236);
  assert.deepEqual(ledger.training.splitGames, {
    train: 8000, validation: 999, test: 1001,
  });
  assert.deepEqual(
    [ledger.training.trainSamples, ledger.training.validationSamples,
      ledger.training.testSamples],
    [884764, 100260, 100931],
  );
  assert.deepEqual(ledger.training.openingDepths, [0]);
  assert.deepEqual(ledger.training.leagueOpeningDepths, [0, 4, 8, 12]);
  assert.match(ledger.architecture.tacticalWitness, /64-state FIFO/i);
  assert.match(ledger.architecture.tacticalWitness, /exact found witnesses/i);
  assert.match(ledger.architecture.tacticalWitness, /truncation/i);
  assert.match(ledger.architecture.trainingProvenance, /source, compiler, build/i);
  assert.match(ledger.architecture.dependencyBoundary, /No incumbent source/i);
  assert.match(ledger.architecture.dependencyBoundary, /replay data/i);
  assert.match(ledger.architecture.dependencyBoundary, /action labels/i);
  assert.match(ledger.training.provenance, /Per-game build SHA/i);
  assert.match(ledger.training.provenance, /source\/compiler\/binary contract/i);
  assert.match(html, /replay auditor is read-only diagnostic tooling/i);
  assert.match(html, /never become training labels or promotion evidence/i);
  assert.throws(
    () => statusPage.validate({
      ...ledger,
      architecture: { ...ledger.architecture, tacticalWitness: undefined },
    }),
    /missing tacticalWitness/,
  );
  assert.throws(
    () => statusPage.validate({
      ...ledger,
      training: { ...ledger.training, provenance: "" },
    }),
    /training provenance is incomplete/,
  );
  assert.throws(
    () => statusPage.validate({
      ...ledger,
      provenance: { ...ledger.provenance, producerSha256: "not-a-hash" },
    }),
    /producerSha256 is incomplete/,
  );
  assert.equal(ledger.diagnostics.length, 1);
  assert.match(ledger.diagnostics[0].label, /sibling seed/i);
  assert.equal(ledger.diagnostics[0].status, "failed");
  assert.match(ledger.diagnostics[0].result, /484–516/);
  assert.match(ledger.diagnostics[0].requirement, /not a previous-champion/i);
  assert.doesNotMatch(ledgerText, /matches\.json|protected[_ -]?bank|sealed[_ -]?bank/i);
  assert.match(ledger.links.auditor, /REPLAY_DECISION_AUDITOR\.md$/);
});

test("every file-backed artifact hash and size matches the repository", async () => {
  const artifact = (id) => ledger.artifacts.find((item) => item.id === id);
  for (const item of ledger.artifacts.filter((entry) => entry.path)) {
    const bytes = await readFile(new URL(item.path, root));
    assert.equal(sha256(bytes), item.sha256, item.label);
    const observed = item.sizeUnit === "characters"
      ? bytes.toString("utf8").length
      : bytes.length;
    assert.equal(observed, item.size, item.label);
  }
  assert.equal(artifact("workflow").sha256, ledger.provenance.workflowSha256);
  assert.equal(artifact("build-provenance").sha256,
               ledger.provenance.buildProvenanceSha256);
  assert.equal(artifact("producer-contract").sha256,
               ledger.provenance.producerSha256);
  assert.equal(artifact("selfplay-binary").sha256,
               ledger.provenance.selfplayBinarySha256);
  assert.equal(artifact("selfplay-manifest").sha256,
               ledger.provenance.manifestSha256);
  assert.equal(artifact("corpus-report").sha256,
               ledger.provenance.corpusReportSha256);
  assert.equal(artifact("corpus-identity").sha256,
               ledger.provenance.corpusSha256);
  assert.equal(artifact("untrained-runtime").sha256,
               ledger.provenance.untrainedRuntimeSha256);

  const submission = artifact("submission");
  assert.ok(submission.size < ledger.verification.sourceLimit);
  assert.equal(ledger.verification.sourceLimit - submission.size, 471);

  const header = await readFile(
    new URL("submissions/codingame/bots/jacek_native_bfm/jacek_native_model.hpp", root),
    "utf8",
  );
  const model = artifact("model-json");
  const packed = artifact("packed-weights");
  assert.match(header, new RegExp(`kModelSha256 = "${model.sha256}"`));
  assert.match(header, new RegExp(`kPackedSha256 = "${packed.sha256}"`));
  assert.match(header, new RegExp(`kPackedByteCount = ${packed.size};`));
  assert.match(header, new RegExp(`kTrainingSeed = ${ledger.training.chosenSeed}ULL;`));

  const modelDocument = JSON.parse(await readFile(
    new URL("models/jacek_native_bootstrap_model.json", root), "utf8",
  ));
  assert.equal(modelDocument.provenance.games, ledger.training.games);
  assert.equal(
    modelDocument.provenance.corpus_sha256,
    ledger.provenance.corpusSha256,
  );
  assert.equal(
    modelDocument.provenance.corpus_validator_sha256,
    artifact("corpus-validator").sha256,
  );
  assert.equal(
    modelDocument.provenance.trainer_sha256,
    artifact("trainer").sha256,
  );
  assert.deepEqual(
    modelDocument.provenance.generation.build_provenance_sha256,
    [ledger.provenance.buildProvenanceSha256],
  );
  assert.deepEqual(
    modelDocument.provenance.generation.producer_sha256,
    [ledger.provenance.producerSha256],
  );
  assert.deepEqual(
    modelDocument.provenance.generation.model_artifact_sha256,
    [ledger.provenance.untrainedRuntimeSha256],
  );
  assert.equal(modelDocument.provenance.incumbent_labels, false);
  assert.equal(modelDocument.provenance.protected_data, false);
});

test("clock and promotion fields encode the frozen time-based gates", () => {
  assert.deepEqual(ledger.clock.search, { firstMs: 800, laterMs: 165 });
  assert.deepEqual(ledger.clock.headroom, { firstMs: 900, laterMs: 180 });
  assert.deepEqual(ledger.clock.official, { firstMs: 1000, laterMs: 200 });
  assert.match(html, /P0 and P1 run in separate fresh processes/i);
  assert.deepEqual(ledger.clock.timingProbe.playerZero, {
    firstMs: 400.968,
    laterMs: 167.306,
    maxRssBytes: 81739776,
    peakFootprintBytes: 76382616,
  });
  assert.deepEqual(ledger.clock.timingProbe.playerOne, {
    firstMs: 405.716,
    laterMs: 168.314,
    maxRssBytes: 91881472,
    peakFootprintBytes: 78610840,
  });
  assert.ok(ledger.clock.timingProbe.playerZero.firstMs < ledger.clock.headroom.firstMs);
  assert.ok(ledger.clock.timingProbe.playerZero.laterMs < ledger.clock.headroom.laterMs);
  assert.ok(ledger.clock.timingProbe.playerOne.firstMs < ledger.clock.headroom.firstMs);
  assert.ok(ledger.clock.timingProbe.playerOne.laterMs < ledger.clock.headroom.laterMs);
  assert.deepEqual(ledger.clock.decisiveBootstrap, {
    candidateFirstMaxMs: 451.303,
    candidateLaterMaxMs: 179.918,
    headroomFailures: 0,
  });
  assert.deepEqual(ledger.clock.externalDevelopment, {
    candidateFirstMaxMs: 0,
    candidateLaterMaxMs: 180.513,
    headroomFailures: 1,
  });
  assert.ok(
    ledger.clock.externalDevelopment.candidateLaterMaxMs >=
      ledger.clock.headroom.laterMs,
  );
  assert.deepEqual(
    ledger.gates.map((gate) => gate.status),
    ["passed", "passed", "failed", "not-run"],
  );
  assert.match(ledger.gates[0].requirement, /530 wins in 1,000/);
  assert.match(ledger.gates[1].requirement, /112 wins in 212/);
  assert.match(ledger.gates[2].requirement, /106 games \(53 paired openings\)/);
  assert.match(ledger.gates[3].requirement, /424 games \(212 paired openings\)/);
  assert.match(ledger.gates[3].requirement, /Wilson 95% lower/);
  assert.match(ledger.gates[3].requirement, /≥ 102 wins in each color/);
  assert.match(ledger.gates[0].result, /689–311/);
  assert.match(ledger.gates[1].result, /144–68/);
  assert.match(ledger.gates[1].result, /179\.918 ms/);
  assert.match(ledger.gates[2].result, /35–71/);
  assert.match(ledger.gates[2].result, /Wilson lower 0\.24798/);
  assert.match(ledger.gates[2].result, /one 180\.513 ms headroom failure/);
  assert.match(ledger.gates[3].result, /no second seed/i);
  assert.equal(ledger.verification.status, "passed");
  assert.deepEqual(
    ledger.verification.checks.map((check) => check.status),
    ledger.verification.checks.map(() => "passed"),
  );
});
