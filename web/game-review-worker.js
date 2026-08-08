(function (worker) {
  "use strict";

  worker.importScripts(
    "game-review-client.js",
    "papersoccer-analysis-wasm.js",
  );

  const review = worker.PaperSoccerGameReview;
  if (!review || typeof worker.createPaperSoccerAnalysisModule !== "function") {
    throw new Error("The Game Review analysis module did not load.");
  }

  let cabi = null;
  let activeSessionId = 0;
  let cancelled = false;

  function post(type, sessionId, fields) {
    worker.postMessage(Object.assign({ type, sessionId }, fields));
  }

  function fail(sessionId, error) {
    post("error", sessionId, {
      error: error instanceof Error ? error.message : String(error),
    });
  }

  function step(sessionId) {
    if (!cabi || cancelled || sessionId !== activeSessionId) {
      return;
    }
    try {
      const stepResult = cabi.step();
      const snapshot = cabi.snapshot();
      if (cancelled || sessionId !== activeSessionId) {
        return;
      }
      if (review.reviewDone(snapshot) || stepResult > 1) {
        post("complete", sessionId, { snapshot });
        return;
      }
      post("progress", sessionId, { snapshot });
      worker.setTimeout(function () { step(sessionId); }, 0);
    } catch (error) {
      fail(sessionId, error);
    }
  }

  worker.onmessage = function (event) {
    const request = event.data;
    if (!request || !cabi) {
      return;
    }
    if (request.type === "cancel") {
      if (request.sessionId === activeSessionId) {
        cancelled = true;
        cabi.cancel();
        post("cancelled", request.sessionId);
      }
      return;
    }
    if (request.type === "sandbox-start" || request.type === "sandbox-play") {
      try {
        const snapshot = request.type === "sandbox-start"
          ? cabi.sandboxStart(request.possession)
          : cabi.sandboxPlay(request.point);
        post("sandbox", request.sessionId, {
          requestId: request.requestId,
          snapshot,
        });
      } catch (error) {
        post("sandbox-error", request.sessionId, {
          requestId: request.requestId,
          error: error instanceof Error ? error.message : String(error),
        });
      }
      return;
    }
    if (request.type === "sandbox-close") {
      cabi.sandboxClose();
      return;
    }
    if (request.type !== "start" && request.type !== "validate") {
      return;
    }

    activeSessionId = request.sessionId;
    cancelled = false;
    try {
      const snapshot = review.populateSession(
        cabi,
        request.prepared,
        request.mode,
      );
      post("validated", request.sessionId, { snapshot });
      if (request.type === "start") {
        worker.setTimeout(function () { step(request.sessionId); }, 0);
      }
    } catch (error) {
      fail(request.sessionId, error);
    }
  };

  worker.createPaperSoccerAnalysisModule().then(function (module) {
    cabi = review.createCabiClient(module);
    worker.postMessage({ type: "ready" });
  }).catch(function (error) {
    throw error;
  });
}(self));
