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
  assert.match(html, /id="liveDetails"/);
  assert.match(html, /<div\b[^>]*\brole="alert"/i);
  assert.doesNotMatch(html, /papersoccer-wasm|<canvas\b|game-engine|board-view/i);
  assert.match(css, /@media\s*\([^)]*max-width/i);
  assert.match(css, /overflow-x:\s*auto/i);
  assert.match(css, /:focus-visible/i);
  assert.match(script, /textContent/);
  assert.doesNotMatch(script, /innerHTML/);
});

test("the checked-in ledger binds the completed exploratory live diagnostic", () => {
  assert.equal(statusPage.validate(ledger), ledger);
  assert.equal(ledger.schema, "papersoccer.jacek-native-status.v1");
  assert.equal(ledger.candidate.stage, "live-diagnostic-analyzed");
  assert.equal(ledger.candidate.status, "in-progress");
  assert.match(ledger.candidate.claim, /52–38 raw and 41–38 clean/i);
  assert.match(ledger.candidate.claim, /rank 9/i);
  assert.match(ledger.candidate.claim, /not a promotion or parity claim/i);
  assert.equal(ledger.live.status, "passed");
  assert.equal(ledger.live.batchStatus, "complete");
  assert.equal(ledger.live.submissionId, 41123817);
  assert.equal(ledger.live.agentId, 6609056);
  assert.equal(ledger.live.historyVersion, 61);
  assert.equal(ledger.live.rank, 9);
  assert.equal(ledger.live.score, 39.54);
  assert.equal(ledger.live.percentage, 100);
  assert.equal(ledger.live.sourceBound, false);
  assert.equal(ledger.live.sourceBinding, "asserted-not-api-verified");
  assert.equal(
    ledger.live.sourceCommit,
    "8cf6005aace930016b86ac05de2ac8743447612c",
  );
  assert.equal(
    ledger.live.sourceSha256,
    "3bda271b35695292324c4e1943062211d102d66b0bb69f43615ba7a0b89e6e20",
  );
  assert.equal(
    ledger.live.modelSha256,
    "19f954092bea404ab18ccc7aaec8b7f6627f0b459017a7f83b6d666b6bb03acc",
  );
  assert.equal(
    ledger.live.packedSha256,
    "7125339d76ade22b0d8e3de249876927b99611372ff81396994c074522394218",
  );
  assert.equal(ledger.live.developmentContaminated, true);
  assert.equal(ledger.live.candidateOperationalFailures, 0);
  assert.deepEqual(ledger.live.raw, {
    games: 90,
    wins: 52,
    losses: 38,
    colors: {
      playerZero: { wins: 29, losses: 15 },
      playerOne: { wins: 23, losses: 23 },
    },
  });
  assert.deepEqual(ledger.live.clean, {
    games: 79,
    wins: 41,
    losses: 38,
    colors: {
      playerZero: { wins: 21, losses: 15 },
      playerOne: { wins: 20, losses: 23 },
    },
  });
  assert.deepEqual(ledger.live.opponentForfeits, {
    total: 11, illegalAction: 7, timeout: 4,
  });
  assert.deepEqual(ledger.live.cleanCohorts, {
    top5: { wins: 4, losses: 25 },
    top10: { wins: 6, losses: 32 },
    top20: { wins: 22, losses: 37 },
  });
  assert.match(ledger.live.label, /90-game diagnostic archived/i);
  assert.match(ledger.live.label, /no promotion claim/i);
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
  assert.match(ledger.training.nextRound, /round-two native league/i);
  assert.match(ledger.training.nextRound, /pilot execution/i);
  assert.match(ledger.training.nextRound, /no checkpoint or result is retained/i);
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
  assert.deepEqual(ledger.live.evidence, {
    manifestSha256: "0328bded1916af5bd34554bbd315577cc346b7ba2e32b83f34ef3ef0e30351cf",
    cleanTsvSha256: "d5cea44b03a340f220fcb5d2f4864c59151bfd25ad659302bf4c0ead1768b79b",
    fixed30kAuditSha256: "7f06835b8cfc0e4a8a51ff02195aed12d06a729af70a66cf0ddced0cafd86fee",
    fixed30kSummarySha256: "4d9d56bc1c66c8cac6366c64b4b2c2683bdd5a9f0302c45591c76a57a672972b",
  });
  assert.deepEqual(ledger.live.audit.classifications, {
    bfmOverride: 1070,
    match: 702,
    initialEvaluatorOrdering: 136,
    generatorOmission: 6,
    operationalFailure: 4,
  });
  assert.equal(ledger.live.audit.games, 79);
  assert.equal(ledger.live.audit.decisions, 1918);
  assert.equal(ledger.live.audit.workPerDecision, 30000);
  assert.match(ledger.live.audit.interpretation, /diagnostic/i);
  assert.match(ledger.live.audit.interpretation, /not correct-move or promotion labels/i);
  assert.match(ledger.live.audit.interpretation, /evaluator, reanalysis, and BFM allocation/i);
  assert.match(html, /800 \/ 155 ms/);
  assert.match(html, /no promotion or parity claim is made/i);
  assert.match(html, /development-contaminated/i);
  assert.match(html, /not correct-move labels/i);
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
  assert.equal(
    submission.sha256,
    "3bda271b35695292324c4e1943062211d102d66b0bb69f43615ba7a0b89e6e20",
  );
  assert.ok(submission.size < ledger.verification.sourceLimit);
  assert.equal(ledger.verification.sourceLimit - submission.size, 471);
  assert.equal(submission.sha256, ledger.live.sourceSha256);
  assert.equal(artifact("model-json").sha256, ledger.live.modelSha256);
  assert.equal(artifact("packed-weights").sha256, ledger.live.packedSha256);
  assert.equal(
    artifact("live-manifest").sha256,
    ledger.live.evidence.manifestSha256,
  );
  assert.equal(
    artifact("clean-auditor-input").sha256,
    ledger.live.evidence.cleanTsvSha256,
  );
  assert.equal(
    artifact("fixed30k-audit").sha256,
    ledger.live.evidence.fixed30kAuditSha256,
  );
  assert.equal(
    artifact("fixed30k-summary").sha256,
    ledger.live.evidence.fixed30kSummarySha256,
  );

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
  assert.deepEqual(ledger.clock.search, { firstMs: 800, laterMs: 155 });
  assert.deepEqual(ledger.clock.headroom, { firstMs: 900, laterMs: 180 });
  assert.deepEqual(ledger.clock.official, { firstMs: 1000, laterMs: 200 });
  assert.match(html, /P0 and P1 run in separate fresh processes/i);
  assert.deepEqual(ledger.clock.timingProbe.playerZero, {
    firstMs: 425.910,
    laterMs: 157.571,
    maxRssBytes: 81461248,
    peakFootprintBytes: 76497304,
  });
  assert.deepEqual(ledger.clock.timingProbe.playerOne, {
    firstMs: 397.385,
    laterMs: 157.921,
    maxRssBytes: 91504640,
    peakFootprintBytes: 78791088,
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
  assert.deepEqual(ledger.clock.diagnosticSafety, {
    games: 24,
    candidateWins: 8,
    referenceWins: 16,
    candidateColorWins: [5, 3],
    unfinishedGames: 0,
    headroomFailures: 0,
    operationalFailures: 0,
    candidateLaterMaxMs: 166.726,
  });
  assert.deepEqual(ledger.clock.shuffleSeedDiagnostic, {
    gamesPerVariant: 24,
    constantSeed: { candidateWins: 4, referenceWins: 20 },
    variedSeed: { candidateWins: 5, referenceWins: 19 },
    conclusion: "The one-game change does not support shuffle-seed mismatch as the main strength bottleneck",
  });
  assert.deepEqual(
    ledger.gates.map((gate) => gate.status),
    ["passed", "passed", "failed", "passed", "not-run"],
  );
  assert.match(ledger.gates[0].requirement, /530 wins in 1,000/);
  assert.match(ledger.gates[1].requirement, /112 wins in 212/);
  assert.match(ledger.gates[2].requirement, /106 games \(53 paired openings\)/);
  assert.match(ledger.gates[3].label, /800\/155 ms exploratory operational screen/);
  assert.match(ledger.gates[3].requirement, /strength is not assessed/i);
  assert.match(ledger.gates[4].requirement, /424 games \(212 paired openings\)/);
  assert.match(ledger.gates[4].requirement, /Wilson 95% lower/);
  assert.match(ledger.gates[4].requirement, /≥ 102 wins in each color/);
  assert.match(ledger.gates[0].result, /689–311/);
  assert.match(ledger.gates[1].result, /144–68/);
  assert.match(ledger.gates[1].result, /179\.918 ms/);
  assert.match(ledger.gates[2].result, /35–71/);
  assert.match(ledger.gates[2].result, /Wilson lower 0\.24798/);
  assert.match(ledger.gates[2].result, /one 180\.513 ms headroom failure/);
  assert.match(ledger.gates[3].result, /8–16/);
  assert.match(ledger.gates[3].result, /colors 5\/3/);
  assert.match(ledger.gates[3].result, /candidate later max 166\.726 ms/);
  assert.match(ledger.gates[4].result, /does not establish parity/i);
  assert.equal(ledger.verification.status, "passed");
  assert.deepEqual(
    ledger.verification.checks.map((check) => check.status),
    ledger.verification.checks.map(() => "passed"),
  );
});
