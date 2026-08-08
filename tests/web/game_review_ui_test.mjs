import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

await import("../../web/app-support.js");

const support = globalThis.PaperSoccerWebSupport;
const [html, appSource, boardSource, workerSource] = await Promise.all([
  readFile(new URL("../../web/index.html", import.meta.url), "utf8"),
  readFile(new URL("../../web/app.js", import.meta.url), "utf8"),
  readFile(new URL("../../web/board-view.js", import.meta.url), "utf8"),
  readFile(new URL("../../web/game-review-worker.js", import.meta.url), "utf8"),
]);

function reviewSnapshot() {
  return {
    complete: true,
    possessions: [
      {
        startPly: 1,
        endPly: 3,
        grade: "Best",
        analyzed: true,
        matchedRecommendation: true,
        estimatedLossPercentagePoints: 0,
      },
      {
        firstPly: 4,
        lastPly: 4,
        grade: "Inaccuracy",
        analyzed: true,
        estimatedLossPercentagePoints: 7.5,
      },
      {
        playedAction: [{ ply: 5 }, { ply: 6 }],
        grade: "Blunder",
        analyzed: true,
        estimatedLossPercentagePoints: 22,
      },
      {
        startPly: 7,
        endPly: 7,
        grade: "Unclear",
        analyzed: true,
      },
    ],
  };
}

test("review support maps possessions without replacing edge navigation", () => {
  const snapshot = reviewSnapshot();

  assert.deepEqual(support.possessionRange(snapshot.possessions[0]), {
    startPly: 1,
    endPly: 3,
  });
  assert.equal(support.possessionForPly(snapshot, 2), snapshot.possessions[0]);
  assert.equal(support.possessionForPly(snapshot, 4), snapshot.possessions[1]);
  assert.equal(support.possessionForPly(snapshot, 99), null);
  assert.deepEqual(support.possessionRange(snapshot.possessions[2]), {
    startPly: 5,
    endPly: 6,
  });
});

test("review summary exposes only requested aggregate measures", () => {
  const summary = support.reviewSummary(reviewSnapshot());

  assert.equal(summary.counts.Best, 1);
  assert.equal(summary.counts.Inaccuracy, 1);
  assert.equal(summary.counts.Blunder, 1);
  assert.equal(summary.counts.Unclear, 1);
  assert.equal(summary.bestActions, 1);
  assert.equal(summary.totalActions, 3);
  assert.equal(summary.largestLoss, 22);
  assert.equal("accuracy" in summary, false);
  assert.equal("winProbability" in summary, false);
});

test("game-review exports unwrap through the established replay entry point", () => {
  const replay = { schema: "papersoccer.replay.v2", moves: [] };
  const document = {
    schema: "papersoccer.game-review.v1",
    sourceReplay: replay,
    possessions: [],
  };

  assert.equal(support.importedReview(document), document);
  assert.equal(support.unwrapReplayDocument(document), replay);
  assert.throws(
    () => support.unwrapReplayDocument({ schema: document.schema }),
    /does not contain its source replay/u,
  );
});

test("the review panel offers refinement, cancellation, diagnostics, and sandboxing", () => {
  assert.match(html, /<script defer src="game-review-gate\.js"><\/script>/u);
  for (const id of [
    "reviewGameButton",
    "gameReviewPanel",
    "fastReviewButton",
    "deepReviewButton",
    "cancelReviewButton",
    "reviewProgress",
    "possessionTimeline",
    "possessionGrade",
    "possessionLoss",
    "possessionProof",
    "possessionDiagnostics",
    "tryLineButton",
    "exportReviewButton",
  ]) {
    assert.match(html, new RegExp(`id="${id}"`, "u"));
  }
  assert.match(html, /complete possessions, including every rebound edge/u);
  assert.match(html, /deterministic engine estimates rather than objective truth/u);
  assert.doesNotMatch(html, /win.probability graph/iu);
  assert.doesNotMatch(html, /overall accuracy/iu);
  assert.match(appSource, /event\.mode === gameReview\.ReviewMode\.Fast && pendingDeepRefinement/u);
  assert.match(appSource, /gameReview\.canonicalJson\(sourceReplay\) !== originalFingerprint/u);
  assert.match(appSource, /gameReviewPanel\.dataset\.sourceReplaySha256/u);
  assert.match(appSource, /await analysisRunner\.startSandbox/u);
  assert.match(appSource, /await analysisRunner\.playSandbox/u);
  assert.match(appSource, /interactiveLegalMoves\(\)/u);
  assert.match(workerSource, /request\.type === "sandbox-play"/u);
});

test("the board renders played and recommended paths with first divergence", () => {
  assert.match(boardSource, /getReviewOverlay/u);
  assert.match(boardSource, /overlay\.playedPath/u);
  assert.match(boardSource, /overlay\.recommendedPath/u);
  assert.match(boardSource, /drawDivergenceMarker/u);
  assert.match(boardSource, /255, 132, 82/u);
  assert.match(boardSource, /103, 224, 255/u);
});

test("review is post-game only and imports validate before loading", () => {
  assert.match(
    appSource,
    /return Boolean\(liveSnapshot\) && !gameError && isLiveTerminal\(\)/u,
  );
  assert.match(
    appSource,
    /await analysisRunner\.validate\(raw\)[\s\S]*?validatedImportedDocument = raw;[\s\S]*?loadReplay\(raw\)/u,
  );
  assert.match(appSource, /loadReplay\(generated\.replay, true, \{ validated: true \}\)/u);
  assert.match(appSource, /Imported diagnostics are read-only/u);
  assert.match(appSource, /!reviewSandboxReady/u);
});

test("the hosted worker is classic and keeps session-tagged results", () => {
  assert.match(workerSource, /importScripts/u);
  assert.doesNotMatch(workerSource, /type:\s*["']module["']/u);
  assert.match(workerSource, /sessionId !== activeSessionId/u);
  assert.match(workerSource, /setTimeout\(function \(\) \{ step\(sessionId\); \}, 0\)/u);
  assert.match(workerSource, /papersoccer-analysis-wasm\.js/u);
});

test("Expert is not exposed without a positive locked strength gate", () => {
  assert.doesNotMatch(html, /Expert\s*[—-]\s*DeepTurnSearch/u);
  assert.equal(html.match(/<option value="random" selected>/gu)?.length, 3);
  assert.equal(support.expertOpponentEnabled({
    schema: support.GAME_REVIEW_GATE_WEB_SCHEMA,
    expertOpponentEnabled: false,
    strengthStatus: "strength unresolved",
    selectorLabel: null,
    selectedProfileId: "deep-turn-search-200k",
    testGames: 1600,
    compactResultSha256: "a".repeat(64),
  }), false);
});

test("a positive frozen gate alone installs Expert in all opponent selectors", () => {
  const appended = [];
  const selects = Array.from({ length: 3 }, () => ({
    options: [{ value: "random" }],
    append(option) {
      this.options.push(option);
      appended.push(option);
    },
  }));
  const fakeDocument = {
    createElement(tag) {
      assert.equal(tag, "option");
      return { value: "", textContent: "" };
    },
  };
  const positive = {
    schema: support.GAME_REVIEW_GATE_WEB_SCHEMA,
    expertOpponentEnabled: true,
    strengthStatus: "validated",
    selectorLabel: "Expert — DeepTurnSearch",
    selectedProfileId: "deep-turn-search-200k",
    testGames: 1600,
    compactResultSha256: "b".repeat(64),
  };

  assert.equal(
    support.installExpertOpponentOptions(selects, positive, fakeDocument),
    true,
  );
  assert.equal(appended.length, 3);
  assert.ok(appended.every((option) =>
    option.value === "deepTurnSearch" &&
    option.textContent === "Expert — DeepTurnSearch"));
  assert.equal(
    support.installExpertOpponentOptions(selects, positive, fakeDocument),
    true,
  );
  assert.equal(appended.length, 3, "reinstalling must not duplicate the option");
});
