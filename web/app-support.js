(function (root) {
  "use strict";

  const HUMAN_MATCH_SCHEMA = "papersoccer.human-match.v1";

  function unwrapReplayDocument(document) {
    if (document?.schema !== HUMAN_MATCH_SCHEMA) {
      return document;
    }
    if (!document.replay || typeof document.replay !== "object") {
      throw new Error("The human-match export does not contain a replay.");
    }
    return document.replay;
  }

  function liveGameControlState(options) {
    const hasGame = Boolean(options.hasGame);
    const locked = hasGame && Boolean(options.movesEnabled) &&
      !Boolean(options.isTerminal);
    return {
      hasGame: hasGame,
      locked: locked,
      usesMcts: Boolean(options.usesMcts),
      usesAlphaBeta: Boolean(options.usesAlphaBeta),
      primaryLabel: hasGame ? "Start new game" : "Start",
    };
  }

  function syncConfigurationControls(elements, state) {
    elements.botSelect.disabled = state.locked;
    elements.sideSelect.disabled = state.locked;
    elements.seedInput.disabled = state.locked;
    elements.budgetInput.disabled = state.locked || !state.usesMcts;
    if (elements.depthInput) {
      elements.depthInput.disabled = state.locked || !state.usesAlphaBeta;
    }
    elements.startButton.hidden = state.locked;
    elements.startButton.textContent = state.primaryLabel;
    elements.changeSettingsButton.hidden = !state.locked;
    elements.exportButton.hidden = !state.hasGame;
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
    } else if (configuration?.kind === "AlphaBetaBot" &&
               Number.isInteger(configuration.depth)) {
      suffix = "handoff-depth-" + String(configuration.depth);
    }
    return "papersoccer-human-match-" + kind + "-seed-" + seed +
      "-" + suffix + ".json";
  }

  function safeFilenamePart(value) {
    const part = String(value).replace(/[^A-Za-z0-9_-]+/g, "-");
    return part.length > 0 ? part : "unknown";
  }

  root.PaperSoccerWebSupport = Object.freeze({
    HUMAN_MATCH_SCHEMA,
    createLiveGameLifecycle,
    createStaleSafeTimer,
    humanMatchFilename,
    liveGameControlState,
    syncConfigurationControls,
    unwrapReplayDocument,
  });
}(globalThis));
