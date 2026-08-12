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
  assert.match(html, /id="historicalLiveDetails"/);
  assert.match(html, /<div\b[^>]*\brole="alert"/i);
  assert.doesNotMatch(html, /papersoccer-wasm|<canvas\b|game-engine|board-view/i);
  assert.match(css, /@media\s*\([^)]*max-width/i);
  assert.match(css, /overflow-x:\s*auto/i);
  assert.match(css, /:focus-visible/i);
  assert.match(script, /textContent/);
  assert.doesNotMatch(script, /innerHTML/);
});

test("the checked-in v2 ledger separates current round two from historical live evidence", () => {
  assert.equal(statusPage.validate(ledger), ledger);
  assert.equal(ledger.schema, "papersoccer.jacek-native-status.v2");
  assert.equal(ledger.candidate.stage, "round2-live-diagnostic-archived");
  assert.equal(ledger.candidate.status, "in-progress");
  assert.match(ledger.candidate.claim, /seed 20260822/i);
  assert.match(ledger.candidate.claim, /post-activation timing/i);
  assert.match(ledger.candidate.claim, /90-game CodinGame diagnostic/i);
  assert.match(ledger.candidate.claim, /no Rank 4 parity claim/i);
  assert.equal(ledger.live.status, "passed");
  assert.equal(ledger.live.batchStatus, "complete");
  assert.equal(ledger.live.submissionId, 41124914);
  assert.equal(ledger.live.agentId, 6609905);
  assert.equal(ledger.live.historyVersion, 62);
  assert.equal(ledger.live.rank, 5);
  assert.equal(ledger.live.score, 42.68);
  assert.equal(ledger.live.percentage, 100);
  assert.equal(ledger.live.sourceBound, false);
  assert.equal(ledger.live.sourceBinding, "asserted-not-api-verified");
  assert.equal(
    ledger.live.sourceCommit,
    "e1ae4c7c66a03d9a2c3b82ddf79adafcb7e0c661",
  );
  assert.equal(
    ledger.live.sourceSha256,
    "653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90",
  );
  assert.equal(
    ledger.live.modelSha256,
    "b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14",
  );
  assert.equal(
    ledger.live.packedSha256,
    "e2304195d491d7b2d5ae1334a8341b38d67d315073accc37915885ede3c6a2cb",
  );
  assert.equal(ledger.live.developmentContaminated, true);
  assert.equal(ledger.live.candidateOperationalFailures, 0);
  assert.deepEqual(ledger.live.coverage, {
    expectedGames: 90,
    acceptedGames: 90,
    fullWindowAccounted: true,
    cleanRuleTerminalGames: 71,
  });
  assert.deepEqual(ledger.live.raw, {
    games: 90,
    wins: 63,
    losses: 27,
    colors: {
      playerZero: { wins: 30, losses: 13 },
      playerOne: { wins: 33, losses: 14 },
    },
  });
  assert.deepEqual(ledger.live.clean, {
    games: 71,
    wins: 44,
    losses: 27,
    colors: {
      playerZero: { wins: 18, losses: 13 },
      playerOne: { wins: 26, losses: 14 },
    },
  });
  assert.deepEqual(ledger.live.opponentForfeits, {
    total: 19, illegalAction: 10, timeout: 9,
  });
  assert.deepEqual(ledger.live.cleanCohorts, {
    top5: { wins: 11, losses: 14 },
    top10: { wins: 26, losses: 22 },
    top20: { wins: 34, losses: 24 },
  });
  assert.deepEqual(ledger.live.namedCleanOpponents, [
    { name: "jacek", frozenRank: 1, games: 9, wins: 0, losses: 9 },
    { name: "Deltaspace", frozenRank: 3, games: 7, wins: 6, losses: 1 },
    { name: "Laars", frozenRank: 4, games: 7, wins: 5, losses: 2 },
    { name: "Waffle3z", frozenRank: 7, games: 4, wins: 4, losses: 0 },
    { name: "derjack", frozenRank: 8, games: 4, wins: 1, losses: 3 },
    { name: "EricSMSO", frozenRank: 9, games: 6, wins: 3, losses: 3 },
    { name: "YurkovAS", frozenRank: 10, games: 7, wins: 7, losses: 0 },
  ]);
  assert.match(ledger.live.label, /90-game diagnostic archived/i);
  assert.match(ledger.live.label, /no Rank 4 parity claim/i);
  assert.equal(ledger.historicalLive.submissionId, 41123817);
  assert.equal(ledger.historicalLive.agentId, 6609056);
  assert.equal(ledger.historicalLive.historyVersion, 61);
  assert.equal(ledger.historicalLive.rank, 9);
  assert.equal(ledger.historicalLive.score, 39.54);
  assert.deepEqual(ledger.historicalLive.raw, {
    games: 90,
    wins: 52,
    losses: 38,
    colors: {
      playerZero: { wins: 29, losses: 15 },
      playerOne: { wins: 23, losses: 23 },
    },
  });
  assert.deepEqual(ledger.historicalLive.clean, {
    games: 79,
    wins: 41,
    losses: 38,
    colors: {
      playerZero: { wins: 21, losses: 15 },
      playerOne: { wins: 20, losses: 23 },
    },
  });
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
  assert.match(ledger.training.nextRound, /historical round-one checkpoint/i);
  assert.match(ledger.training.nextRound, /seed 20260822/i);
  assert.match(ledger.training.nextRound, /activated, uploaded, and archived/i);
  assert.match(html, /selected clean states may seed fresh native continuations/i);
  assert.match(html, /never become policy or value labels/i);
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
    manifestSha256: "bb5aaf7340ddee174ddf916aee2bfbcd4b3afb01f5d53bbc5c7d728bf610b4cf",
    cleanTsvSha256: "4a25768aed11e7c4bc368e63bce8c335e420f02fee7e131d2b56dfa97ab048e2",
    fixed30kAuditSha256: "9eb8d1741bbd11390c3d0b1c15ea0cac35f73931e6695635c2f5778d7c7ff8f1",
    fixed30kSummarySha256: "79d88d6f608d2a78e86dd1f6abca8a5406a9866a417489dc7b2d99beb17cdd46",
  });
  assert.deepEqual(ledger.live.audit.classifications, {
    bfmOverride: 1015,
    match: 715,
    initialEvaluatorOrdering: 76,
    generatorOmission: 4,
    operationalFailure: 1,
    boundaryEquivalent: 0,
    tacticalMiss: 0,
  });
  assert.equal(ledger.live.audit.games, 71);
  assert.equal(ledger.live.audit.decisions, 1811);
  assert.equal(ledger.live.audit.workPerDecision, 30000);
  assert.match(ledger.live.audit.interpretation, /diagnostic/i);
  assert.match(ledger.live.audit.interpretation, /not correct-move or promotion labels/i);
  assert.match(ledger.live.audit.interpretation, /0–9 clean result against jacek/i);
  assert.match(html, /800 \/ 155 ms/);
  assert.match(html, /no Rank 4 parity claim is made/i);
  assert.match(html, /development-contaminated/i);
  assert.match(html, /not correct-move labels/i);
  assert.match(html, /Round-two live evidence/i);
  assert.match(html, /Historical history-61 evidence/i);
  assert.equal(ledger.round2.scope, "live-diagnostic");
  assert.equal(ledger.round2.uploaded, true);
  assert.equal(ledger.round2.status, "passed");
  assert.equal(ledger.round2.corpus.games, 22_238);
  assert.equal(ledger.round2.corpus.samples, 2_151_889);
  assert.equal(ledger.round2.corpus.observedMovePolicyLabels, 0);
  assert.deepEqual(ledger.round2.corpus.splitGames, {
    train: 17_779, validation: 2_230, test: 2_229,
  });
  assert.deepEqual(ledger.round2.corpus.splitSamples, {
    train: 1_755_307, validation: 198_858, test: 197_724,
  });
  assert.deepEqual(ledger.round2.corpus.overlapsRemoved, {
    train: 0, validation: 21_961, test: 23_469,
  });
  assert.equal(ledger.round2.corpus.lineage.strictCurrent.games, 12_000);
  assert.equal(ledger.round2.corpus.lineage.archivedRound1.games, 10_000);
  assert.equal(ledger.round2.corpus.lineage.liveRestartRound2.games, 238);
  assert.equal(
    ledger.round2.corpus.lineage.liveRestartRound2.selectedPrefixes, 119,
  );
  assert.equal(
    ledger.round2.corpus.lineage.liveRestartRound2.sourceBinding,
    "asserted-not-api-verified",
  );
  assert.deepEqual(ledger.round2.model.seeds, [20260821, 20260822, 20260823]);
  assert.equal(ledger.round2.model.provisionalSeed, 20260823);
  assert.equal(ledger.round2.model.chosenSeedInModel, null);
  assert.equal(ledger.round2.model.selectedSeed, 20260822);
  assert.equal(ledger.round2.selection.kind, "promotion");
  assert.equal(ledger.round2.selection.promotionEligible, true);
  assert.deepEqual(ledger.round2.selection.thresholdShortfalls, []);
  assert.equal(ledger.round2.deployment.activated, true);
  assert.equal(ledger.round2.deployment.sourceCharacters, 94_771);
  assert.equal(ledger.round2.deployment.sourceHeadroom, 228);
  assert.equal(ledger.round2.gates.length, 6);
  assert.equal(ledger.round2.verification.status, "passed");
  assert.deepEqual(
    ledger.round2.verification.checks.map(({ id, status, tests }) => ({
      id, status, tests,
    })),
    [
      { id: "activation", status: "passed", tests: undefined },
      { id: "purity", status: "passed", tests: undefined },
      { id: "source", status: "passed", tests: undefined },
      { id: "gcc", status: "passed", tests: 3 },
      { id: "appleclang", status: "passed", tests: 7 },
      { id: "sanitizers", status: "passed", tests: 4 },
    ],
  );
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
  for (const gate of ledger.round2.gates) {
    const [reportBytes, stdoutBytes] = await Promise.all([
      readFile(new URL(gate.reportPath, root)),
      readFile(new URL(gate.stdoutPath, root)),
    ]);
    assert.equal(sha256(reportBytes), gate.reportSha256);
    assert.equal(sha256(stdoutBytes), gate.stdoutSha256);
    assert.match(gate.reportPath, new RegExp(`${gate.reportSha256}\\.json$`));
    assert.match(
      gate.stdoutPath, new RegExp(`${gate.stdoutSha256}\\.stdout\\.txt$`),
    );
    const report = JSON.parse(reportBytes.toString("utf8"));
    assert.equal(report.candidate.seed, gate.seed);
    assert.equal(report.profile.name, gate.profile);
    assert.equal(report.result.candidate, gate.candidateWins);
    assert.equal(report.result.baseline, gate.baselineWins);
    assert.deepEqual(
      [report.result.candidate_player_one,
        report.result.candidate_player_two],
      gate.candidateColorWins,
    );
    assert.equal(report.result.passed, gate.status === "passed");
    assert.equal(report.result.unfinished, gate.unfinishedGames);
    assert.equal(
      report.result.candidate_headroom_failures,
      gate.candidateHeadroomFailures,
    );
    assert.equal(
      report.result.candidate_operational_timeouts,
      gate.candidateOperationalTimeouts,
    );
    assert.equal(
      report.result.baseline_headroom_failures,
      gate.baselineHeadroomFailures,
    );
    assert.equal(
      report.result.baseline_operational_timeouts,
      gate.baselineOperationalTimeouts,
    );
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

  const submission = artifact("history61-source");
  assert.equal(
    submission.sha256,
    "3bda271b35695292324c4e1943062211d102d66b0bb69f43615ba7a0b89e6e20",
  );
  assert.ok(submission.size < ledger.verification.sourceLimit);
  assert.equal(ledger.verification.sourceLimit - submission.size, 471);
  assert.equal(submission.sha256, ledger.historicalLive.sourceSha256);
  assert.equal(artifact("model-json").sha256, ledger.historicalLive.modelSha256);
  assert.equal(artifact("packed-weights").sha256, ledger.historicalLive.packedSha256);
  assert.equal(
    artifact("history61-live-manifest").sha256,
    ledger.historicalLive.evidence.manifestSha256,
  );
  assert.equal(
    artifact("history61-clean-auditor-input").sha256,
    ledger.historicalLive.evidence.cleanTsvSha256,
  );
  assert.equal(
    artifact("history61-fixed30k-audit").sha256,
    ledger.historicalLive.evidence.fixed30kAuditSha256,
  );
  assert.equal(
    artifact("history61-fixed30k-summary").sha256,
    ledger.historicalLive.evidence.fixed30kSummarySha256,
  );

  const header = await readFile(
    new URL("submissions/codingame/bots/jacek_native_bfm/jacek_native_model.hpp", root),
    "utf8",
  );
  const round2Model = artifact("round2-model-json");
  const round2Runtime = artifact("round2-runtime");
  const round2Selection = artifact("round2-selection");
  const round2Deployment = artifact("round2-deployment");
  const round2Header = artifact("round2-model-header");
  const round2Source = artifact("round2-source");
  assert.equal(round2Model.sha256, ledger.round2.model.sha256);
  assert.equal(round2Runtime.sha256, ledger.round2.model.runtimeSha256);
  assert.equal(round2Selection.sha256, ledger.round2.selection.sha256);
  assert.equal(round2Deployment.sha256, ledger.round2.deployment.sha256);
  assert.equal(round2Header.sha256, ledger.round2.deployment.headerSha256);
  assert.equal(round2Source.sha256, ledger.round2.deployment.sourceSha256);
  assert.equal(round2Source.sha256, ledger.live.sourceSha256);
  assert.equal(round2Model.sha256, ledger.live.modelSha256);
  assert.equal(ledger.round2.model.packedSha256, ledger.live.packedSha256);
  assert.equal(
    artifact("round2-live-manifest").sha256,
    ledger.live.evidence.manifestSha256,
  );
  assert.equal(
    artifact("round2-clean-auditor-input").sha256,
    ledger.live.evidence.cleanTsvSha256,
  );
  assert.equal(
    artifact("round2-fixed30k-audit").sha256,
    ledger.live.evidence.fixed30kAuditSha256,
  );
  assert.equal(
    artifact("round2-fixed30k-summary").sha256,
    ledger.live.evidence.fixed30kSummarySha256,
  );
  assert.ok(round2Source.size < ledger.round2.deployment.sourceLimit);
  assert.equal(
    ledger.round2.deployment.sourceLimit - round2Source.size,
    ledger.round2.deployment.sourceHeadroom,
  );
  assert.match(header, new RegExp(`kModelSha256 = "${round2Model.sha256}"`));
  assert.match(
    header, new RegExp(`kPackedSha256 = "${ledger.round2.model.packedSha256}"`),
  );
  assert.match(header, /kPackedByteCount = 14268;/);
  assert.match(
    header, new RegExp(`kTrainingSeed = ${ledger.round2.model.selectedSeed}ULL;`),
  );

  const round2ModelDocument = JSON.parse(await readFile(
    new URL("models/jacek_native_round2_candidate.json", root), "utf8",
  ));
  const selectionDocument = JSON.parse(await readFile(
    new URL("models/jacek_native_round2_selection.json", root), "utf8",
  ));
  const deploymentDocument = JSON.parse(await readFile(
    new URL("models/jacek_native_round2_deployment.json", root), "utf8",
  ));
  assert.equal(round2ModelDocument.training.chosen_seed, null);
  assert.equal(
    round2ModelDocument.training.provisional_seed,
    ledger.round2.model.provisionalSeed,
  );
  assert.equal(
    round2ModelDocument.provenance.corpus_sha256,
    ledger.round2.corpus.corpusSha256,
  );
  assert.equal(
    round2ModelDocument.provenance.observed_move_policy_labels,
    ledger.round2.corpus.observedMovePolicyLabels,
  );
  assert.equal(selectionDocument.decision.kind, ledger.round2.selection.kind);
  assert.equal(
    selectionDocument.decision.promotion_eligible,
    ledger.round2.selection.promotionEligible,
  );
  assert.equal(
    selectionDocument.selection_payload_sha256,
    ledger.round2.selection.payloadSha256,
  );
  assert.equal(selectionDocument.selected.seed, ledger.round2.model.selectedSeed);
  assert.equal(
    selectionDocument.selected.runtime_sha256,
    ledger.round2.model.runtimeSha256,
  );
  assert.equal(
    selectionDocument.selected.packed_sha256,
    ledger.round2.model.packedSha256,
  );
  assert.equal(deploymentDocument.selected_seed, ledger.round2.model.selectedSeed);
  assert.equal(deploymentDocument.decision.kind, ledger.round2.selection.kind);
  assert.equal(
    deploymentDocument.runtime.sha256,
    ledger.round2.model.runtimeSha256,
  );

  const model = artifact("model-json");
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
  assert.equal(ledger.round2.timing.status, "passed");
  assert.deepEqual(ledger.round2.timing.playerZero, {
    firstMs: 424.894, laterMs: 157.944,
  });
  assert.deepEqual(ledger.round2.timing.playerOne, {
    firstMs: 419.405, laterMs: 157.479,
  });
  for (const player of [
    ledger.round2.timing.playerZero, ledger.round2.timing.playerOne,
  ]) {
    assert.ok(player.firstMs < ledger.clock.headroom.firstMs);
    assert.ok(player.laterMs < ledger.clock.headroom.laterMs);
  }
  assert.deepEqual(
    ledger.round2.gates.map(({ seed, profile, status }) => ({
      seed, profile, status,
    })),
    [
      { seed: 20260821, profile: "screen", status: "passed" },
      { seed: 20260821, profile: "decisive", status: "failed" },
      { seed: 20260822, profile: "screen", status: "passed" },
      { seed: 20260822, profile: "decisive", status: "passed" },
      { seed: 20260823, profile: "screen", status: "passed" },
      { seed: 20260823, profile: "decisive", status: "failed" },
    ],
  );
});
