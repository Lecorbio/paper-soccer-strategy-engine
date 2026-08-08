(function (root) {
  "use strict";

  const HUMAN_MATCH_SCHEMA = "papersoccer.human-match.v1";
  const GAME_REVIEW_SCHEMA = "papersoccer.game-review.v1";
  const GAME_REVIEW_GATE_WEB_SCHEMA = "papersoccer.game-review-gate-web.v1";
  const EXPERT_LABEL = "Expert — DeepTurnSearch";

  function unwrapReplayDocument(document) {
    if (document?.schema === GAME_REVIEW_SCHEMA) {
      if (!document.sourceReplay || typeof document.sourceReplay !== "object") {
        throw new Error("The game-review export does not contain its source replay.");
      }
      return document.sourceReplay;
    }
    if (document?.schema === HUMAN_MATCH_SCHEMA) {
      if (!document.replay || typeof document.replay !== "object") {
        throw new Error("The human-match export does not contain a replay.");
      }
      return document.replay;
    }
    return document;
  }

  function importedReview(document) {
    return document?.schema === GAME_REVIEW_SCHEMA ? document : null;
  }

  function reviewPossessions(snapshot) {
    if (Array.isArray(snapshot?.possessions)) {
      return snapshot.possessions;
    }
    if (Array.isArray(snapshot?.reviews)) {
      return snapshot.reviews;
    }
    return [];
  }

  function possessionRange(possession) {
    const played = Array.isArray(possession?.playedAction)
      ? possession.playedAction
      : (Array.isArray(possession?.played) ? possession.played : []);
    const firstMove = played[0];
    const lastMove = played.at(-1);
    const startPly = Number.isInteger(possession?.startPly)
      ? possession.startPly
      : (Number.isInteger(possession?.firstPly)
          ? possession.firstPly
          : (Number.isInteger(firstMove?.ply) ? firstMove.ply : 1));
    const endPly = Number.isInteger(possession?.endPly)
      ? possession.endPly
      : (Number.isInteger(possession?.lastPly)
          ? possession.lastPly
          : (Number.isInteger(lastMove?.ply) ? lastMove.ply : startPly));
    return { startPly, endPly };
  }

  function possessionForPly(snapshot, ply) {
    return reviewPossessions(snapshot).find(function (possession) {
      const range = possessionRange(possession);
      return ply >= range.startPly && ply <= range.endPly;
    }) ?? null;
  }

  function normalizedGrade(grade) {
    const value = String(grade ?? "Unclear").toLowerCase();
    const labels = {
      forced: "Forced",
      best: "Best",
      good: "Good",
      inaccuracy: "Inaccuracy",
      mistake: "Mistake",
      blunder: "Blunder",
      unclear: "Unclear",
    };
    return labels[value] ?? "Unclear";
  }

  function estimatedLoss(possession) {
    const candidates = [
      possession?.estimatedLossPercentagePoints,
      possession?.estimatedLoss,
      possession?.lossPercentagePoints,
      possession?.loss,
    ];
    const value = candidates
      .filter(function (candidate) {
        return candidate !== null && candidate !== undefined;
      })
      .map(Number)
      .find(Number.isFinite);
    return value === undefined ? null : Math.max(0, value);
  }

  function reviewSummary(snapshot) {
    const possessions = reviewPossessions(snapshot);
    const counts = {
      Forced: 0,
      Best: 0,
      Good: 0,
      Inaccuracy: 0,
      Mistake: 0,
      Blunder: 0,
      Unclear: 0,
    };
    let bestActions = 0;
    let totalActions = 0;
    let largestLoss = null;
    for (const possession of possessions) {
      if (possession.analyzed !== true) {
        continue;
      }
      const grade = normalizedGrade(possession.grade);
      counts[grade] += 1;
      const eligible = possession.analyzed === true &&
        possession.truncated !== true && grade !== "Unclear";
      if (eligible) {
        totalActions += 1;
        if (possession.recommendedActionMatched === true ||
            possession.matchedRecommendation === true) {
          bestActions += 1;
        }
      }
      const loss = estimatedLoss(possession);
      if (loss !== null && (largestLoss === null || loss > largestLoss)) {
        largestLoss = loss;
      }
    }
    return {
      counts,
      bestActions,
      totalActions,
      gradedActions: totalActions,
      largestLoss,
    };
  }

  function gameReviewFilename(replaySha256, mode) {
    const hash = safeFilenamePart(String(replaySha256 ?? "unknown").slice(0, 12));
    const profile = String(mode).toLowerCase() === "deep" ? "deep" : "fast";
    return "papersoccer-game-review-" + hash + "-" + profile + ".json";
  }

  function expertOpponentEnabled(gateStatus) {
    return gateStatus?.schema === GAME_REVIEW_GATE_WEB_SCHEMA &&
      gateStatus.expertOpponentEnabled === true &&
      gateStatus.strengthStatus === "validated" &&
      gateStatus.selectorLabel === EXPERT_LABEL &&
      gateStatus.testGames === 1600 &&
      /^[0-9a-f]{64}$/u.test(String(gateStatus.compactResultSha256 ?? ""));
  }

  function installExpertOpponentOptions(selects, gateStatus, documentHost) {
    if (!expertOpponentEnabled(gateStatus)) {
      return false;
    }
    const host = documentHost ?? root.document;
    if (!host || typeof host.createElement !== "function") {
      throw new Error("A document is required to install the Expert opponent.");
    }
    for (const select of selects) {
      const existing = Array.from(select?.options ?? []).some(function (option) {
        return option.value === "deepTurnSearch";
      });
      if (existing) {
        continue;
      }
      const option = host.createElement("option");
      option.value = "deepTurnSearch";
      option.textContent = EXPERT_LABEL;
      select.append(option);
    }
    return true;
  }

  function liveGameControlState(options) {
    const hasGame = Boolean(options.hasGame);
    const locked = hasGame && Boolean(options.movesEnabled) &&
      !Boolean(options.isTerminal);
    return {
      hasGame: hasGame,
      locked: locked,
      usesMcts: Boolean(options.usesMcts),
      usesDepth: Boolean(options.usesDepth),
      primaryLabel: hasGame ? "Start new game" : "Start",
    };
  }

  function syncConfigurationControls(elements, state) {
    elements.botSelect.disabled = state.locked;
    elements.sideSelect.disabled = state.locked;
    elements.seedInput.disabled = state.locked;
    elements.budgetInput.disabled = state.locked || !state.usesMcts;
    if (elements.depthInput) {
      elements.depthInput.disabled = state.locked || !state.usesDepth;
    }
    elements.startButton.hidden = state.locked;
    elements.startButton.textContent = state.primaryLabel;
    elements.changeSettingsButton.hidden = !state.locked;
    elements.exportButton.hidden = !state.hasGame;
  }

  function humanPlayerForTurnOrder(turnOrder, players) {
    return turnOrder === players.Two ? players.Two : players.One;
  }

  function scheduleOpeningBotTurn(options) {
    const snapshot = options.snapshot;
    if (!options.movesEnabled || options.gameError || !snapshot?.state ||
        snapshot.state.toMove === snapshot.humanPlayer) {
      return false;
    }
    options.schedule();
    return true;
  }

  function createStaleSafeTimer(timerHost) {
    let handle = null;
    let generation = 0;
    let pending = false;

    function cancel() {
      generation += 1;
      if (handle !== null) {
        timerHost.clearTimeout(handle);
      }
      handle = null;
      pending = false;
    }

    function schedule(callback, delay) {
      cancel();
      const scheduledGeneration = generation;
      pending = true;
      handle = timerHost.setTimeout(function () {
        handle = null;
        pending = false;
        if (scheduledGeneration === generation) {
          callback();
        }
      }, delay);
    }

    return Object.freeze({
      cancel: cancel,
      schedule: schedule,
      get pending() {
        return pending;
      },
    });
  }

  function createLiveGameLifecycle(timerHost) {
    const timer = createStaleSafeTimer(timerHost);
    let movesEnabled = false;

    function start() {
      timer.cancel();
      movesEnabled = true;
    }

    function restore(enabled) {
      timer.cancel();
      movesEnabled = Boolean(enabled);
    }

    function stop() {
      timer.cancel();
      movesEnabled = false;
    }

    function scheduleBot(callback, delay) {
      if (!movesEnabled) {
        return false;
      }
      timer.schedule(callback, delay);
      return true;
    }

    return Object.freeze({
      cancelPendingBot: timer.cancel,
      changeSettings: stop,
      finish: stop,
      restore,
      scheduleBot,
      start,
      get botThinking() {
        return timer.pending;
      },
      get movesEnabled() {
        return movesEnabled;
      },
    });
  }

  function humanMatchFilename(configuration) {
    const kind = safeFilenamePart(configuration?.kind ?? "bot");
    const seed = safeFilenamePart(configuration?.seed ?? "unknown");
    let suffix = "no-search";
    if (configuration?.kind === "MctsBot" &&
        Number.isInteger(configuration.iterations)) {
      suffix = String(configuration.iterations) + "-new-simulations-per-move";
    } else if ((configuration?.kind === "AlphaBetaBot" ||
                configuration?.kind === "JacekInspiredBot") &&
               Number.isInteger(configuration.depth)) {
      suffix = "handoff-depth-" + String(configuration.depth);
    }
    return "papersoccer-human-match-" + kind + "-seed-" + seed +
      "-" + suffix + ".json";
  }

  function alphaBetaRootScoreText(search) {
    if (search?.rootScoreValid !== true ||
        Number(search.completedDepth) <= 0 ||
        !Number.isFinite(Number(search.rootScore))) {
      return "No completed Player 1 score";
    }
    return "Player 1 score " +
      Number(search.rootScore).toLocaleString("en-US");
  }

  function safeFilenamePart(value) {
    const part = String(value).replace(/[^A-Za-z0-9_-]+/g, "-");
    return part.length > 0 ? part : "unknown";
  }

  root.PaperSoccerWebSupport = Object.freeze({
    GAME_REVIEW_SCHEMA,
    GAME_REVIEW_GATE_WEB_SCHEMA,
    HUMAN_MATCH_SCHEMA,
    alphaBetaRootScoreText,
    createLiveGameLifecycle,
    createStaleSafeTimer,
    estimatedLoss,
    expertOpponentEnabled,
    gameReviewFilename,
    humanPlayerForTurnOrder,
    humanMatchFilename,
    importedReview,
    installExpertOpponentOptions,
    liveGameControlState,
    normalizedGrade,
    possessionForPly,
    possessionRange,
    reviewPossessions,
    reviewSummary,
    scheduleOpeningBotTurn,
    syncConfigurationControls,
    unwrapReplayDocument,
  });
}(globalThis));
