(async function () {
  "use strict";

  const engine = globalThis.PaperSoccer;
  if (!engine) {
    throw new Error("Paper soccer game engine did not load");
  }
  const support = globalThis.PaperSoccerWebSupport;
  if (!support) {
    throw new Error("Paper soccer web support did not load");
  }
  const board = globalThis.PaperSoccerBoard;
  if (!board) {
    throw new Error("Paper soccer board view did not load");
  }

  const { BotKind, Player, Status, samePoint } = engine;
  const gameSetup = document.querySelector("#gameSetup");
  const preventSetupSubmit = function (event) { event.preventDefault(); };
  gameSetup.addEventListener("submit", preventSetupSubmit);
  const gameEngine = await engine.ready;

  const canvas = document.querySelector("#boardCanvas");
  const boardDescription = document.querySelector("#boardDescription");
  const boardHeading = document.querySelector("#boardHeading");
  const boardTargets = document.querySelector("#boardTargets");
  const boardTurnBadge = document.querySelector("#boardTurnBadge");
  const replayTitle = document.querySelector("#replayTitle");
  const controlPanel = document.querySelector(".control-panel");
  const plyLabel = document.querySelector("#plyLabel");
  const statusLabel = document.querySelector("#statusLabel");
  const moveLabel = document.querySelector("#moveLabel");
  const moveListHeading = document.querySelector("#moveListHeading");
  const prevButton = document.querySelector("#prevButton");
  const playButton = document.querySelector("#playButton");
  const nextButton = document.querySelector("#nextButton");
  const plyRange = document.querySelector("#plyRange");
  const speedSelect = document.querySelector("#speedSelect");
  const moveList = document.querySelector("#moveList");
  const fileInput = document.querySelector("#fileInput");
  const playModeButton = document.querySelector("#playModeButton");
  const replayModeButton = document.querySelector("#replayModeButton");
  const playSetupEyebrow = document.querySelector("#playSetupEyebrow");
  const botSelect = document.querySelector("#botSelect");
  const sideSelect = document.querySelector("#sideSelect");
  const seedInput = document.querySelector("#seedInput");
  const botSeedField = document.querySelector("#botSeedField");
  const botIterationsField = document.querySelector("#botIterationsField");
  const botIterationsInput = document.querySelector("#botIterationsInput");
  const botDepthField = document.querySelector("#botDepthField");
  const botDepthInput = document.querySelector("#botDepthInput");
  const botRank5Profile = document.querySelector("#botRank5Profile");
  const newGameButton = document.querySelector("#newGameButton");
  const changeSettingsButton = document.querySelector("#changeSettingsButton");
  const exportGameButton = document.querySelector("#exportGameButton");
  const liveDiagnostics = document.querySelector("#liveDiagnostics");
  const activeBotConfiguration = document.querySelector(
    "#activeBotConfiguration",
  );
  const latestSearch = document.querySelector("#latestSearch");
  const searchVisits = document.querySelector("#searchVisits");
  const searchShape = document.querySelector("#searchShape");
  const searchIndicators = document.querySelector("#searchIndicators");
  const searchDetails = document.querySelector("#searchDetails");
  const searchPending = document.querySelector("#searchPending");
  const setupError = document.querySelector("#setupError");
  const replaySetup = document.querySelector("#replaySetup");
  const generateReplayButton = document.querySelector("#generateReplayButton");
  const replayOneBotSelect = document.querySelector("#replayOneBotSelect");
  const replayOneSeedInput = document.querySelector("#replayOneSeedInput");
  const replayOneSeedField = document.querySelector("#replayOneSeedField");
  const replayOneIterationsField = document.querySelector(
    "#replayOneIterationsField",
  );
  const replayOneIterationsInput = document.querySelector(
    "#replayOneIterationsInput",
  );
  const replayOneDepthField = document.querySelector("#replayOneDepthField");
  const replayOneDepthInput = document.querySelector("#replayOneDepthInput");
  const replayOneRank5Profile = document.querySelector(
    "#replayOneRank5Profile",
  );
  const replayTwoBotSelect = document.querySelector("#replayTwoBotSelect");
  const replayTwoSeedInput = document.querySelector("#replayTwoSeedInput");
  const replayTwoSeedField = document.querySelector("#replayTwoSeedField");
  const replayTwoIterationsField = document.querySelector(
    "#replayTwoIterationsField",
  );
  const replayTwoIterationsInput = document.querySelector(
    "#replayTwoIterationsInput",
  );
  const replayTwoDepthField = document.querySelector("#replayTwoDepthField");
  const replayTwoDepthInput = document.querySelector("#replayTwoDepthInput");
  const replayTwoRank5Profile = document.querySelector(
    "#replayTwoRank5Profile",
  );
  const replaySetupError = document.querySelector("#replaySetupError");
  const replayGenerationStatus = document.querySelector(
    "#replayGenerationStatus",
  );
  const moveChoicesSection = document.querySelector("#moveChoicesSection");
  const moveChoices = document.querySelector("#moveChoices");
  const replayControls = document.querySelector("#replayControls");
  const liveHistoryControls = document.querySelector("#liveHistoryControls");
  const undoMoveButton = document.querySelector("#undoMoveButton");
  const resumeBotButton = document.querySelector("#resumeBotButton");

  const MODE_PLAY = "play";
  const MODE_REPLAY = "replay";
  const BOT_DELAY_MS = 520;
  const DEFAULT_MCTS_ITERATIONS = 2000;
  const MAX_MCTS_ITERATIONS = 10000;
  const DEFAULT_ALPHA_BETA_DEPTH = 6;
  const MAX_ALPHA_BETA_DEPTH = 64;
  const RANK5_DERIVED_MAX_NODES = 50000;
  const MAX_REPLAY_PLIES = 512;
  const MAX_UINT64 = (1n << 64n) - 1n;

  const defaultReplay = normalizeReplay(readDefaultReplay());
  let mode = MODE_PLAY;
  let replay = newGamePreviewReplay(defaultReplay);
  let replaySession = { replay: defaultReplay, currentPly: 0 };
  let liveSession = null;
  let currentPly = 0;
  let isPlaying = false;
  let speed = 1;
  let lastStepTime = 0;
  let liveSnapshot = null;
  let humanPlayer = Player.One;
  const liveGame = support.createLiveGameLifecycle(window);
  let gameError = "";
  let lastLiveMove = null;
  let activeBotConfig = {
    kind: BotKind.Random,
    seed: "12345",
    iterations: DEFAULT_MCTS_ITERATIONS,
    depth: DEFAULT_ALPHA_BETA_DEPTH,
  };
  let replayGenerationId = 0;
  let isGeneratingReplay = false;
  let botPausedAfterUndo = false;

  const boardView = board.createBoardView({
    boardTargets: boardTargets,
    canHumanMove: canHumanMove,
    canvas: canvas,
    currentBall: currentBall,
    currentMoveStartIndex: currentMoveStartIndex,
    getCurrentPly: function () { return currentPly; },
    getReplay: function () { return replay; },
    liveLegalMoves: liveLegalMoves,
    liveState: liveState,
    onHumanMove: playHumanMove,
    playerTwo: Player.Two,
    samePoint: samePoint,
  });

  function readDefaultReplay() {
    const raw = document.querySelector("#default-replay").textContent;
    return JSON.parse(raw);
  }

  function newGamePreviewReplay(source) {
    return {
      schema: "papersoccer.replay.v2",
      rules: { width: source.rules.width, height: source.rules.height },
      players: {},
      start: { x: source.start.x, y: source.start.y },
      status: Status.InProgress,
      winner: null,
      truncated: false,
      moves: [],
    };
  }

  function normalizeReplay(raw) {
    raw = support.unwrapReplayDocument(raw);
    const sourceSchema = raw.schema ?? "papersoccer.replay.v1";
    if (sourceSchema !== "papersoccer.replay.v1" &&
        sourceSchema !== "papersoccer.replay.v2") {
      throw new Error("Unsupported replay schema: " + sourceSchema);
    }

    const rules = raw.rules ?? {};
    const width = positiveInteger(rules.width, 8);
    const height = positiveInteger(rules.height, 10);
    const yOffset = sourceSchema === "papersoccer.replay.v1" ? 1 : 0;
    const start = normalizePoint(
      raw.start,
      { x: Math.floor(width / 2), y: Math.floor(height / 2) + 1 },
      yOffset,
    );

    let previousPoint = start;
    const moves = Array.isArray(raw.moves)
      ? raw.moves.map(function (move, index) {
        const normalized = normalizeMove(move, index, previousPoint, yOffset);
        previousPoint = normalized.to;
        return normalized;
      })
      : [];

    return {
      schema: "papersoccer.replay.v2",
      rules: { width: width, height: height },
      players: raw.players ?? {},
      start: start,
      status: raw.status ?? Status.InProgress,
      winner: raw.winner ?? null,
      truncated: Boolean(raw.truncated),
      moves: moves,
    };
  }

  function positiveInteger(value, fallback) {
    return Number.isInteger(value) && value > 0 ? value : fallback;
  }

  function normalizeMove(move, index, previousPoint, yOffset) {
    const from = normalizePoint(move.from, previousPoint, yOffset);
    return {
      ply: index + 1,
      player: move.player === Player.Two ? Player.Two : Player.One,
      from: from,
      to: normalizePoint(move.to, from, yOffset),
      extraTurn: Boolean(move.extraTurn),
      statusAfter: move.statusAfter ?? Status.InProgress,
    };
  }

  function normalizePoint(point, fallback, yOffset) {
    const x = Number.isFinite(point?.x) ? point.x : fallback.x;
    const y = Number.isFinite(point?.y) ? point.y + yOffset : fallback.y;
    return { x: x, y: y };
  }

  function playerName(player) {
    return player === Player.Two ? "Player 2" : "Player 1";
  }

  function relativePlayerName(player) {
    if (mode !== MODE_PLAY) {
      return playerName(player);
    }
    return player === humanPlayer ? "You" : activeBotName();
  }

  function statusName(status) {
    if (status === Status.WonByOne) {
      return "Player 1 won";
    }
    if (status === Status.WonByTwo) {
      return "Player 2 won";
    }
    if (status === Status.InProgress) {
      return "In progress";
    }
    return String(status);
  }

  function formatPoint(point) {
    return "r" + point.y + " c" + point.x;
  }

  function describePlayers() {
    return describePlayer(replay.players.one, "Player 1") + " · " +
      describePlayer(replay.players.two, "Player 2");
  }

  function describePlayer(player, label) {
    if (!player) {
      return label;
    }
    const hasSeed = player.kind !== "Rank5DerivedBot" &&
      player.seed !== undefined && player.seed !== null &&
      String(player.seed).length > 0;
    const details = [];
    if (hasSeed) {
      details.push("seed " + String(player.seed));
    }
    if (Number.isInteger(player.iterations) && player.iterations > 0) {
      details.push(player.iterations + " new simulations per bot move");
    }
    if (Number.isInteger(player.depth) && player.depth > 0) {
      details.push("possession-handoff depth " + player.depth);
    }
    if (player.kind === "Rank5DerivedBot" &&
        Number.isInteger(player.maxNodes)) {
      details.push(formatCount(player.maxNodes) + " deterministic nodes");
    }
    const suffix = details.length > 0 ? " (" + details.join(", ") + ")" : "";
    return label + ": " + (player.kind ?? "Unknown") + suffix;
  }

  function botDisplayName(kind) {
    if (kind === BotKind.Random) {
      return "RandomBot";
    }
    if (kind === BotKind.Mcts) {
      return "MctsBot";
    }
    if (kind === BotKind.AlphaBeta) {
      return "AlphaBetaBot";
    }
    if (kind === BotKind.JacekInspired) {
      return "JacekInspiredBot";
    }
    if (kind === BotKind.Rank5Derived) {
      return "Rank5DerivedBot";
    }
    return "UnknownBot";
  }

  function isDepthBot(kind) {
    return kind === BotKind.AlphaBeta || kind === BotKind.JacekInspired;
  }

  function activeBotName() {
    return botDisplayName(activeBotConfig.kind);
  }

  function parseSeed(input) {
    const value = input.trim();
    if (!/^(0|[1-9][0-9]*)$/.test(value)) {
      throw new Error("Enter a non-negative whole number for the seed.");
    }
    const seed = BigInt(value);
    if (seed > MAX_UINT64) {
      throw new Error("The seed must be at most 18446744073709551615.");
    }
    return seed;
  }

  function parseIterations(input) {
    const value = input.trim();
    if (!/^[1-9][0-9]*$/.test(value)) {
      throw new Error("Enter a positive whole number for new simulations per bot move.");
    }
    const iterations = Number(value);
    if (!Number.isSafeInteger(iterations) || iterations > MAX_MCTS_ITERATIONS) {
      throw new Error("New simulations per bot move must be at most " +
        MAX_MCTS_ITERATIONS + ".");
    }
    return iterations;
  }

  function parseDepth(input, kind) {
    const name = botDisplayName(kind);
    const value = input.trim();
    if (!/^[1-9][0-9]*$/.test(value)) {
      throw new Error("Enter a positive whole number for " + name + " depth.");
    }
    const depth = Number(value);
    if (!Number.isSafeInteger(depth) || depth > MAX_ALPHA_BETA_DEPTH) {
      throw new Error(name + " depth must be at most " +
        MAX_ALPHA_BETA_DEPTH + ".");
    }
    return depth;
  }

  function selectedBotKind(value) {
    if (value === BotKind.Random || value === BotKind.Mcts ||
        value === BotKind.AlphaBeta || value === BotKind.JacekInspired ||
        value === BotKind.Rank5Derived) {
      return value;
    }
    throw new Error("Unsupported bot kind: " + String(value));
  }

  function readBotConfig(kindSelect, botSeedInput, iterationsInput, depthInput) {
    const kind = selectedBotKind(kindSelect.value);
    if (kind === BotKind.Rank5Derived) {
      return {
        kind: kind,
        seed: "0",
        iterations: DEFAULT_MCTS_ITERATIONS,
        depth: DEFAULT_ALPHA_BETA_DEPTH,
        maxNodes: RANK5_DERIVED_MAX_NODES,
      };
    }
    let seed;
    try {
      seed = parseSeed(botSeedInput.value);
    } catch (error) {
      error.input = botSeedInput;
      throw error;
    }

    let iterations = DEFAULT_MCTS_ITERATIONS;
    if (kind === BotKind.Mcts) {
      try {
        iterations = parseIterations(iterationsInput.value);
      } catch (error) {
        error.input = iterationsInput;
        throw error;
      }
    }

    let depth = DEFAULT_ALPHA_BETA_DEPTH;
    if (isDepthBot(kind)) {
      try {
        depth = parseDepth(depthInput.value, kind);
      } catch (error) {
        error.input = depthInput;
        throw error;
      }
    }

    return {
      kind: kind,
      seed: seed.toString(),
      iterations: iterations,
      depth: depth,
    };
  }

  function botSearchSetting(config) {
    if (config.kind === BotKind.Rank5Derived) {
      return RANK5_DERIVED_MAX_NODES;
    }
    return isDepthBot(config.kind) ? config.depth : config.iterations;
  }

  function clearBotTimer() {
    liveGame.cancelPendingBot();
  }

  function liveState() {
    return liveSnapshot?.state ?? null;
  }

  function liveLegalMoves() {
    return liveSnapshot?.legalMoves ?? [];
  }

  function isLiveTerminal() {
    return !liveSnapshot || liveState().status !== Status.InProgress;
  }

  function adoptLiveSnapshot(snapshot) {
    liveSnapshot = snapshot;
    humanPlayer = snapshot.humanPlayer;
    replay = normalizeReplay(snapshot.replay);
    currentPly = replay.moves.length;
    lastLiveMove = replay.moves.at(-1) ?? null;
    boardView.resetHover();
    plyRange.max = String(currentPly);
    plyRange.value = String(currentPly);
    if (snapshot.state.status !== Status.InProgress) {
      liveGame.finish();
    }
  }

  function saveLiveSession() {
    if (mode !== MODE_PLAY || !liveSnapshot) {
      return;
    }
    clearBotTimer();
    liveSession = {
      replay: replay,
      currentPly: currentPly,
      snapshot: liveSnapshot,
      humanPlayer: humanPlayer,
      botConfig: activeBotConfig,
      movesEnabled: liveGame.movesEnabled,
      gameError: gameError,
      lastLiveMove: lastLiveMove,
      botPausedAfterUndo: botPausedAfterUndo,
    };
  }

  function showReplayMode() {
    if (mode === MODE_REPLAY) {
      return;
    }

    saveLiveSession();
    mode = MODE_REPLAY;
    replay = replaySession.replay;
    currentPly = replaySession.currentPly;
    liveSnapshot = null;
    gameError = "";
    boardView.resetHover();
    lastLiveMove = null;
    isPlaying = false;
    plyRange.max = String(replay.moves.length);
    plyRange.value = String(currentPly);
    renderInterface();
  }

  function showPlayMode() {
    if (mode === MODE_PLAY) {
      return;
    }

    cancelReplayGeneration();
    replaySession = { replay: replay, currentPly: currentPly };
    stopPlayback();
    if (!liveSession) {
      mode = MODE_PLAY;
      replay = newGamePreviewReplay(defaultReplay);
      currentPly = 0;
      liveSnapshot = null;
      gameError = "";
      lastLiveMove = null;
      botPausedAfterUndo = false;
      boardView.resetHover();
      plyRange.max = "0";
      plyRange.value = "0";
      renderInterface();
      return;
    }

    const session = liveSession;
    liveSession = null;
    mode = MODE_PLAY;
    replay = session.replay;
    currentPly = session.currentPly;
    liveSnapshot = session.snapshot;
    humanPlayer = session.humanPlayer;
    activeBotConfig = session.botConfig;
    liveGame.restore(session.movesEnabled);
    gameError = session.gameError;
    lastLiveMove = session.lastLiveMove;
    botPausedAfterUndo = session.botPausedAfterUndo;
    boardView.resetHover();
    plyRange.max = String(replay.moves.length);
    plyRange.value = String(currentPly);
    renderInterface();

    if (liveGame.movesEnabled && !botPausedAfterUndo && !gameError &&
        !isLiveTerminal() &&
        liveState().toMove !== humanPlayer) {
      scheduleBotTurn();
    }
  }

  function startNewGame(event) {
    event?.preventDefault();

    let botConfig;
    try {
      botConfig = readBotConfig(
        botSelect,
        seedInput,
        botIterationsInput,
        botDepthInput,
      );
    } catch (error) {
      setupError.textContent = error.message;
      error.input?.focus();
      return;
    }

    liveGame.changeSettings();
    stopPlayback();
    setupError.textContent = "";
    gameError = "";
    botPausedAfterUndo = false;
    mode = MODE_PLAY;
    liveSession = null;
    const selectedHuman = support.humanPlayerForTurnOrder(
      sideSelect.value,
      Player,
    );

    try {
      adoptLiveSnapshot(gameEngine.startGame(
        botConfig.seed,
        selectedHuman,
        botConfig.kind,
        botSearchSetting(botConfig),
      ));
      activeBotConfig = botConfig;
      liveGame.start();
    } catch (error) {
      gameError = "The C++ game engine could not start: " + error.message;
    }
    renderInterface();

    support.scheduleOpeningBotTurn({
      snapshot: liveSnapshot,
      movesEnabled: liveGame.movesEnabled,
      gameError: gameError,
      schedule: scheduleBotTurn,
    });
  }

  function changeSettings() {
    if (!liveSnapshot || isLiveTerminal() || !liveGame.movesEnabled) {
      return;
    }
    liveGame.changeSettings();
    gameError = "";
    botPausedAfterUndo = false;
    boardView.resetHover();
    renderInterface();
    botSelect.focus();
  }

  function exportHumanMatch() {
    if (!liveSnapshot) {
      return;
    }

    try {
      const exported = gameEngine.humanMatch();
      const contents = JSON.stringify(exported, null, 2) + "\n";
      const blob = new Blob([contents], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = support.humanMatchFilename(exported.botConfiguration);
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(function () {
        URL.revokeObjectURL(url);
      }, 0);
    } catch (error) {
      window.alert("Could not export game: " + error.message);
    }
  }

  function canHumanMove() {
    return mode === MODE_PLAY && liveSnapshot !== null && !gameError &&
      liveGame.movesEnabled && !liveGame.botThinking && !isLiveTerminal() &&
      liveState().toMove === humanPlayer;
  }

  function canUndoMove() {
    if (mode !== MODE_PLAY || !liveSnapshot || replay.moves.length === 0) {
      return false;
    }
    return liveGame.movesEnabled || isLiveTerminal();
  }

  function restoreActiveGameSettings() {
    botSelect.value = activeBotConfig.kind;
    sideSelect.value = humanPlayer;
    seedInput.value = activeBotConfig.seed;
    botIterationsInput.value = String(activeBotConfig.iterations);
    botDepthInput.value = String(activeBotConfig.depth);
    syncBotSetupFields();
  }

  function undoLastMove() {
    if (!canUndoMove()) {
      return;
    }

    clearBotTimer();
    try {
      const snapshot = gameEngine.undo(
        liveSnapshot.sessionId,
        liveSnapshot.revision,
      );
      liveGame.start();
      adoptLiveSnapshot(snapshot);
      restoreActiveGameSettings();
      gameError = "";
      botPausedAfterUndo = !isLiveTerminal() &&
        liveState().toMove !== humanPlayer;
      renderInterface();
      if (undoMoveButton.disabled && !resumeBotButton.hidden) {
        resumeBotButton.focus();
      }
    } catch (error) {
      gameError = "Could not undo the last move: " + error.message;
      renderInterface();
    }
  }

  function resumeBotTurn() {
    if (!botPausedAfterUndo || !liveSnapshot || isLiveTerminal() ||
        liveState().toMove === humanPlayer) {
      return;
    }
    botPausedAfterUndo = false;
    scheduleBotTurn();
  }

  function playHumanMove(move, keyboardActivated) {
    if (!canHumanMove()) {
      return;
    }

    const legal = liveLegalMoves().find(function (candidate) {
      return candidate.id === move.id && samePoint(candidate.to, move.to);
    });
    if (!legal) {
      return;
    }

    try {
      adoptLiveSnapshot(
        gameEngine.playHuman(
          liveSnapshot.sessionId,
          liveSnapshot.revision,
          legal.id,
        ),
      );
      botPausedAfterUndo = false;
      continueLiveGame();
    } catch (error) {
      gameError = "Your move was not accepted: " + error.message;
      renderInterface();
    }
    if (keyboardActivated) {
      canvas.focus();
    }
  }

  function continueLiveGame() {
    if (isLiveTerminal()) {
      renderInterface();
      return;
    }

    if (liveState().toMove !== humanPlayer) {
      scheduleBotTurn();
      return;
    }

    renderInterface();
  }

  function scheduleBotTurn() {
    clearBotTimer();
    if (!liveGame.movesEnabled || botPausedAfterUndo || !liveSnapshot ||
        isLiveTerminal() ||
        liveState().toMove === humanPlayer) {
      renderInterface();
      return;
    }

    const expectedSessionId = liveSnapshot.sessionId;
    const expectedRevision = liveSnapshot.revision;
    liveGame.scheduleBot(function () {
      if (!liveGame.movesEnabled || mode !== MODE_PLAY || !liveSnapshot ||
          liveSnapshot.sessionId !== expectedSessionId ||
          liveSnapshot.revision !== expectedRevision) {
        return;
      }
      try {
        adoptLiveSnapshot(
          gameEngine.playBot(expectedSessionId, expectedRevision),
        );
        continueLiveGame();
      } catch (error) {
        gameError = activeBotName() + " could not move: " + error.message;
        renderInterface();
      }
    }, BOT_DELAY_MS);
    renderInterface();
  }

  function syncReplayGenerationControls() {
    if (!replaySetup) {
      return;
    }
    for (const control of replaySetup.elements) {
      if (control !== replayOneIterationsInput &&
          control !== replayTwoIterationsInput &&
          control !== replayOneDepthInput &&
          control !== replayTwoDepthInput &&
          control !== fileInput) {
        control.disabled = isGeneratingReplay;
      }
    }
    replayOneIterationsInput.disabled = isGeneratingReplay ||
      replayOneBotSelect.value !== BotKind.Mcts;
    replayTwoIterationsInput.disabled = isGeneratingReplay ||
      replayTwoBotSelect.value !== BotKind.Mcts;
    replayOneDepthInput.disabled = isGeneratingReplay ||
      !isDepthBot(replayOneBotSelect.value);
    replayTwoDepthInput.disabled = isGeneratingReplay ||
      !isDepthBot(replayTwoBotSelect.value);
    replayOneSeedInput.disabled = isGeneratingReplay ||
      replayOneBotSelect.value === BotKind.Rank5Derived;
    replayTwoSeedInput.disabled = isGeneratingReplay ||
      replayTwoBotSelect.value === BotKind.Rank5Derived;
    generateReplayButton.textContent = isGeneratingReplay
      ? "Generating…"
      : "Generate replay";
    replaySetup.setAttribute("aria-busy", String(isGeneratingReplay));
  }

  function cancelReplayGeneration(clearStatus = true) {
    if (isGeneratingReplay) {
      replayGenerationId += 1;
      isGeneratingReplay = false;
    }
    if (clearStatus && replayGenerationStatus) {
      replayGenerationStatus.textContent = "";
    }
    if (clearStatus && replaySetupError) {
      replaySetupError.textContent = "";
    }
    syncReplayGenerationControls();
  }

  function nextAnimationFrame() {
    return new Promise(function (resolve) {
      window.requestAnimationFrame(resolve);
    });
  }

  async function generateBotReplay(event) {
    event.preventDefault();

    let playerOne;
    let playerTwo;
    try {
      playerOne = readBotConfig(
        replayOneBotSelect,
        replayOneSeedInput,
        replayOneIterationsInput,
        replayOneDepthInput,
      );
      playerTwo = readBotConfig(
        replayTwoBotSelect,
        replayTwoSeedInput,
        replayTwoIterationsInput,
        replayTwoDepthInput,
      );
    } catch (error) {
      replaySetupError.textContent = error.message;
      error.input?.focus();
      return;
    }

    stopPlayback();
    replaySetupError.textContent = "";
    const generationId = replayGenerationId + 1;
    replayGenerationId = generationId;
    isGeneratingReplay = true;
    replayGenerationStatus.textContent = "Preparing bot match…";
    syncReplayGenerationControls();

    try {
      await nextAnimationFrame();
      if (generationId !== replayGenerationId) {
        return;
      }

      let generated = gameEngine.startBotReplay(
        playerOne,
        playerTwo,
        MAX_REPLAY_PLIES,
      );
      replayGenerationStatus.textContent = "Generating bot match…";
      let nextProgressUpdate = 25;
      while (!generated.done) {
        if (generated.replay.moves.length >= nextProgressUpdate) {
          replayGenerationStatus.textContent = "Generating… " +
            generated.replay.moves.length + " moves";
          nextProgressUpdate += 25;
        }
        await nextAnimationFrame();
        if (generationId !== replayGenerationId) {
          return;
        }
        generated = gameEngine.playBotReplay(
          generated.sessionId,
          generated.revision,
        );
      }

      if (generationId !== replayGenerationId) {
        return;
      }
      isGeneratingReplay = false;
      syncReplayGenerationControls();
      loadReplay(generated.replay, true);
      const moveCount = generated.replay.moves.length;
      replayGenerationStatus.textContent = "Generated " + moveCount +
        (moveCount === 1 ? " move." : " moves.") +
        (generated.replay.truncated ? " Replay reached the 512-move limit." : "");
    } catch (error) {
      if (generationId !== replayGenerationId) {
        return;
      }
      isGeneratingReplay = false;
      replaySetupError.textContent = "Could not generate replay: " + error.message;
      replayGenerationStatus.textContent = "";
      syncReplayGenerationControls();
    }
  }

  function loadReplay(raw, preserveGeneration = false) {
    if (!preserveGeneration) {
      cancelReplayGeneration();
    }
    const normalized = normalizeReplay(raw);
    if (mode === MODE_PLAY) {
      saveLiveSession();
    } else {
      stopPlayback();
    }
    replay = normalized;
    replaySession = { replay: replay, currentPly: 0 };
    mode = MODE_REPLAY;
    liveSnapshot = null;
    gameError = "";
    boardView.resetHover();
    lastLiveMove = null;
    currentPly = 0;
    stopPlayback();
    plyRange.max = String(replay.moves.length);
    setReplayPly(0);
  }

  function clampPly(ply) {
    return Math.max(0, Math.min(replay.moves.length, ply));
  }

  function setReplayPly(ply) {
    if (mode !== MODE_REPLAY) {
      return;
    }
    currentPly = clampPly(ply);
    plyRange.value = String(currentPly);
    if (currentPly >= replay.moves.length) {
      isPlaying = false;
    }
    renderInterface();
  }

  function currentBall() {
    if (currentPly === 0) {
      return replay.start;
    }
    return replay.moves[currentPly - 1].to;
  }

  function currentMoveStartIndex() {
    if (currentPly === 0) {
      return 0;
    }

    const player = replay.moves[currentPly - 1].player;
    let startIndex = currentPly - 1;
    while (startIndex > 0 &&
           replay.moves[startIndex - 1].extraTurn &&
           replay.moves[startIndex - 1].player === player) {
      startIndex -= 1;
    }
    return startIndex;
  }

  function formatCount(value) {
    return Number(value).toLocaleString("en-US");
  }

  function formatDecisionTime(nanoseconds) {
    const milliseconds = Number(nanoseconds) / 1_000_000;
    if (milliseconds < 1) {
      return "<1 ms";
    }
    if (milliseconds < 10) {
      return milliseconds.toFixed(1) + " ms";
    }
    return Math.round(milliseconds).toLocaleString("en-US") + " ms";
  }

  function updateLiveDiagnostics() {
    const diagnostics = liveSnapshot?.diagnostics;
    const configuration = diagnostics?.botConfiguration;
    liveDiagnostics.hidden = !liveSnapshot || !configuration;
    if (liveDiagnostics.hidden) {
      return;
    }

    const usesMcts = configuration.kind === "MctsBot";
    const usesDepth = configuration.kind === "AlphaBetaBot" ||
      configuration.kind === "JacekInspiredBot";
    const usesRank5Profile = configuration.kind === "Rank5DerivedBot";
    activeBotConfiguration.textContent = configuration.kind +
      (usesRank5Profile
        ? " — 50k demo profile · deterministic " +
          formatCount(configuration.maxNodes) + "-node work"
        : " · seed " + configuration.seed) + (usesMcts
        ? " · " + formatCount(configuration.iterations) +
          " new simulations per bot move"
        : usesDepth
          ? " · depth " + formatCount(configuration.depth) +
            " possession handoffs"
          : "") + " · You: " + playerName(liveSnapshot.humanPlayer);

    const search = diagnostics.lastBotSearch;
    latestSearch.hidden = !search;
    searchPending.hidden = (!usesMcts &&
      configuration.kind !== "JacekInspiredBot" && !usesRank5Profile) ||
      Boolean(search);
    if (!search) {
      return;
    }

    if (search.searchType === "alphaBeta") {
      searchVisits.textContent = "Latest completed · depth " +
        formatCount(search.completedDepth) + " of " +
        formatCount(search.requestedDepth) + " · " +
        support.alphaBetaRootScoreText(search);
      searchShape.textContent = formatCount(search.nodes) + " nodes · " +
        formatCount(search.leafEvaluations) + " neural evaluations · " +
        formatDecisionTime(search.decisionTimeNs);
      const indicators = [];
      if (search.budgetExhausted) {
        indicators.push("Safety budget reached at attempted depth " +
          formatCount(search.attemptedDepth));
      }
      if (search.transpositionProbes > 0) {
        indicators.push("TT hits " + formatCount(search.transpositionHits) +
          "/" + formatCount(search.transpositionProbes));
      }
      searchIndicators.textContent = indicators.join(" · ");
      searchDetails.textContent = "Move " + formatCount(search.ply) + " · " +
        formatCount(search.terminalNodes) + " terminal nodes · " +
        formatCount(search.cutoffs) + " cutoffs · max physical ply " +
        formatCount(search.maxPhysicalPly);
      return;
    }

    if (search.searchType === "rank5Derived") {
      searchVisits.textContent = "Latest completed · depth " +
        formatCount(search.completedDepth) + " · Player 1 score " +
        formatCount(search.rootScore);
      searchShape.textContent = formatCount(search.visitedNodes) + " of " +
        formatCount(search.requestedNodes) + " deterministic nodes · " +
        formatDecisionTime(search.decisionTimeNs);
      searchIndicators.textContent = search.budgetExhausted
        ? "50,000-node budget reached"
        : "Search completed within the 50,000-node budget";
      searchDetails.textContent = "Move " + formatCount(search.ply) +
        " · attempted depth " + formatCount(search.attemptedDepth) +
        " · planned complete turn: " +
        formatCount(search.plannedActionLength) + " edges · playing edge " +
        formatCount(search.currentEdgeIndex + 1) +
        (search.cachedContinuation ? " · cached continuation" :
          " · fresh complete-turn search");
      return;
    }

    searchVisits.textContent = "Latest completed · " +
      formatCount(search.completedIterations) + " simulations · root visits " +
      formatCount(search.totalRootVisits) + " · reused " +
      formatCount(search.reusedVisits);
    searchShape.textContent = formatCount(search.nodes) + " nodes · depth " +
      formatCount(search.maxDepth) + " · " +
      formatDecisionTime(search.decisionTimeNs);

    const indicators = [];
    if (search.provenWinner) {
      indicators.push("Proven winner: " + playerName(search.provenWinner));
    }
    if (search.rebuildCount > 0) {
      indicators.push("Tree rebuilds: " + formatCount(search.rebuildCount));
    }
    if (search.expansionSaturated) {
      indicators.push("Expansion saturated");
    }
    searchIndicators.textContent = indicators.join(" · ");
    let details = "Move " + formatCount(search.ply) + " · " +
      formatCount(search.simulatedPlies) + " rollout plies · " +
      formatCount(search.provenNodes) + " proven nodes · root value " +
      Number(search.rootValue).toFixed(4);
    if (search.tacticalProbes > 0 || search.tacticalDepthCutoffs > 0 ||
        search.tacticalNodeCutoffs > 0) {
      details += " · tactical: " + formatCount(search.tacticalProbes) +
        " probes, " + formatCount(search.tacticalSolvedPositions) +
        " solved, " + formatCount(search.tacticalNodes) + " nodes, depth " +
        formatCount(search.maxTacticalDepth) + ", cutoffs " +
        formatCount(search.tacticalDepthCutoffs) + " depth / " +
        formatCount(search.tacticalNodeCutoffs) + " node";
    }
    searchDetails.textContent = details;
  }

  function updateText() {
    if (mode === MODE_PLAY) {
      updatePlayText();
    } else {
      updateReplayText();
    }
  }

  function updatePlayText() {
    if (!liveSnapshot) {
      replayTitle.textContent = "Choose who takes the opening move";
      plyLabel.textContent = "New game";
      statusLabel.textContent = gameError ? "Game could not start" : "Ready to start";
      moveLabel.textContent = gameError ||
        "Choose whether to move first or second, then select Start.";
      boardTurnBadge.textContent = gameError ? "Game unavailable" : "Ready to start";
      boardDescription.textContent = statusLabel.textContent + ". Ball at row " +
        currentBall().y + ", column " + currentBall().x +
        ". Choose the game settings before starting.";
      return;
    }

    const attackDirection = humanPlayer === Player.One ? "top" : "bottom";
    replayTitle.textContent = "You are " + playerName(humanPlayer) +
      " · Attack the " + attackDirection + " goal";
    plyLabel.textContent = replay.moves.length === 0
      ? "New game"
      : "Move " + replay.moves.length;

    if (gameError) {
      statusLabel.textContent = "Game stopped";
      moveLabel.textContent = gameError;
      boardTurnBadge.textContent = "Game stopped";
    } else if (isLiveTerminal()) {
      const humanWon = liveState().winner === humanPlayer;
      statusLabel.textContent = humanWon ? "You won!" : activeBotName() + " won";
      moveLabel.textContent = "Game over after " + replay.moves.length +
        (replay.moves.length === 1 ? " move." : " moves.") +
        " Start a new game to play again.";
      boardTurnBadge.textContent = humanWon ? "You won!" : activeBotName() + " won";
    } else if (!liveGame.movesEnabled) {
      statusLabel.textContent = "Settings unlocked";
      moveLabel.textContent = "This game is paused and remains available to export. " +
        "Choose settings, then start a new game.";
      boardTurnBadge.textContent = "Game paused";
    } else if (botPausedAfterUndo && liveState().toMove !== humanPlayer) {
      statusLabel.textContent = "Bot move undone";
      moveLabel.textContent = "The bot is paused at this position. Undo another " +
        "move or continue when you are ready.";
      boardTurnBadge.textContent = "Bot paused";
    } else if (liveGame.botThinking || liveState().toMove !== humanPlayer) {
      const movesAgain = lastLiveMove?.player !== humanPlayer &&
        lastLiveMove?.extraTurn;
      statusLabel.textContent = movesAgain
        ? activeBotName() + " moves again"
        : activeBotName() + "’s turn";
      moveLabel.textContent = movesAgain
        ? "It bounced off a boundary or visited point. Choosing another move…"
        : "Choosing a move…";
      boardTurnBadge.textContent = movesAgain ? "Bot moves again" : "Bot is thinking";
    } else {
      const legalCount = liveLegalMoves().length;
      const movesAgain = lastLiveMove?.player === humanPlayer &&
        lastLiveMove?.extraTurn;
      statusLabel.textContent = movesAgain ? "Your turn again" : "Your turn";
      const choiceText = legalCount === 1
        ? "Choose the highlighted point."
        : "Choose one of " + legalCount + " highlighted points.";
      moveLabel.textContent = (movesAgain
        ? "You bounced off a boundary or visited point. "
        : "") + choiceText + " You attack the " + attackDirection + " goal.";
      boardTurnBadge.textContent = movesAgain ? "Your turn again" : "Your turn";
    }

    const legalCount = canHumanMove() ? liveLegalMoves().length : 0;
    boardDescription.textContent = statusLabel.textContent + ". Ball at row " +
      currentBall().y + ", column " + currentBall().x + ". " +
      (legalCount > 0
        ? legalCount + " legal destinations are available as buttons on the board."
        : "No move buttons are currently available.");
  }

  function updateReplayText() {
    replayTitle.textContent = describePlayers();

    if (currentPly === 0) {
      plyLabel.textContent = replay.moves.length + " moves";
      statusLabel.textContent = replay.moves.length === 0 &&
        replay.status !== Status.InProgress
        ? statusName(replay.status)
        : "Ready to replay";
      moveLabel.textContent = "Ball starts at " + formatPoint(replay.start);
    } else {
      const move = replay.moves[currentPly - 1];
      const atEnd = currentPly === replay.moves.length;
      const status = atEnd && !replay.truncated ? replay.status : move.statusAfter;
      const finished = status !== Status.InProgress;
      plyLabel.textContent = "Move " + currentPly + " of " + replay.moves.length;
      statusLabel.textContent = finished
        ? statusName(status)
        : playerName(move.player) + " moved";
      moveLabel.textContent = formatPoint(move.from) + " → " + formatPoint(move.to) +
        (move.extraTurn ? " · Extra turn" : "");

      if (atEnd && replay.truncated) {
        statusLabel.textContent = "Replay ended early";
      }
    }

    boardDescription.textContent = statusLabel.textContent + ". Ball at row " +
      currentBall().y + ", column " + currentBall().x +
      ". Use the replay controls or move list to inspect the game.";
  }

  function updateModeControls() {
    const playingGame = mode === MODE_PLAY;
    gameSetup.hidden = !playingGame;
    replaySetup.hidden = playingGame;
    moveChoicesSection.hidden = !playingGame;
    liveHistoryControls.hidden = !playingGame;
    replayControls.hidden = playingGame;
    boardTargets.hidden = !playingGame;
    boardTurnBadge.hidden = !playingGame;
    playModeButton.setAttribute("aria-pressed", String(playingGame));
    replayModeButton.setAttribute("aria-pressed", String(!playingGame));
    playModeButton.classList.toggle("primary-control", playingGame);
    replayModeButton.classList.toggle("primary-control", !playingGame);
    moveListHeading.textContent = playingGame ? "Move history" : "Replay moves";
    boardHeading.textContent = playingGame ? "Playable paper soccer board" : "Replay board";
    controlPanel.setAttribute(
      "aria-label",
      playingGame ? "Game controls" : "Replay controls",
    );
    controlPanel.classList.toggle("is-replay-mode", !playingGame);
    canvas.classList.toggle("is-playable", canHumanMove());
    syncBotSetupFields();
    syncLiveGameControls();
    syncLiveHistoryControls();
    syncReplayGenerationControls();
    updateLiveDiagnostics();
  }

  function syncLiveGameControls() {
    const state = support.liveGameControlState({
      hasGame: Boolean(liveSnapshot),
      movesEnabled: liveGame.movesEnabled,
      isTerminal: isLiveTerminal(),
      usesMcts: botSelect.value === BotKind.Mcts,
      usesDepth: isDepthBot(botSelect.value),
    });
    support.syncConfigurationControls({
      botSelect: botSelect,
      sideSelect: sideSelect,
      seedInput: seedInput,
      budgetInput: botIterationsInput,
      depthInput: botDepthInput,
      startButton: newGameButton,
      changeSettingsButton: changeSettingsButton,
      exportButton: exportGameButton,
    }, state);
    if (botSelect.value === BotKind.Rank5Derived) {
      seedInput.disabled = true;
    }
    gameSetup.classList.toggle("is-configuration-locked", state.locked);
  }

  function syncLiveHistoryControls() {
    undoMoveButton.disabled = !canUndoMove();
    resumeBotButton.hidden = !botPausedAfterUndo || !liveSnapshot ||
      isLiveTerminal() || liveState().toMove === humanPlayer;
  }

  function syncBotSetupFields() {
    const playUsesMcts = botSelect.value === BotKind.Mcts;
    const playUsesDepth = isDepthBot(botSelect.value);
    const playUsesRank5Profile = botSelect.value === BotKind.Rank5Derived;
    botSeedField.hidden = playUsesRank5Profile;
    seedInput.disabled = playUsesRank5Profile;
    botIterationsField.hidden = !playUsesMcts;
    botIterationsInput.disabled = !playUsesMcts;
    botDepthField.hidden = !playUsesDepth;
    botDepthInput.disabled = !playUsesDepth;
    botRank5Profile.hidden = !playUsesRank5Profile;
    playSetupEyebrow.textContent = "Play vs " + botDisplayName(botSelect.value);

    const playerOneUsesMcts = replayOneBotSelect.value === BotKind.Mcts;
    const playerTwoUsesMcts = replayTwoBotSelect.value === BotKind.Mcts;
    const playerOneUsesDepth = isDepthBot(replayOneBotSelect.value);
    const playerTwoUsesDepth = isDepthBot(replayTwoBotSelect.value);
    const playerOneUsesRank5Profile =
      replayOneBotSelect.value === BotKind.Rank5Derived;
    const playerTwoUsesRank5Profile =
      replayTwoBotSelect.value === BotKind.Rank5Derived;
    replayOneSeedField.hidden = playerOneUsesRank5Profile;
    replayTwoSeedField.hidden = playerTwoUsesRank5Profile;
    replayOneSeedInput.disabled = isGeneratingReplay || playerOneUsesRank5Profile;
    replayTwoSeedInput.disabled = isGeneratingReplay || playerTwoUsesRank5Profile;
    replayOneIterationsField.hidden = !playerOneUsesMcts;
    replayTwoIterationsField.hidden = !playerTwoUsesMcts;
    replayOneIterationsInput.disabled = isGeneratingReplay || !playerOneUsesMcts;
    replayTwoIterationsInput.disabled = isGeneratingReplay || !playerTwoUsesMcts;
    replayOneDepthField.hidden = !playerOneUsesDepth;
    replayTwoDepthField.hidden = !playerTwoUsesDepth;
    replayOneDepthInput.disabled = isGeneratingReplay || !playerOneUsesDepth;
    replayTwoDepthInput.disabled = isGeneratingReplay || !playerTwoUsesDepth;
    replayOneRank5Profile.hidden = !playerOneUsesRank5Profile;
    replayTwoRank5Profile.hidden = !playerTwoUsesRank5Profile;
  }

  function updateReplayControls() {
    if (mode !== MODE_REPLAY) {
      return;
    }

    const atStart = currentPly === 0;
    const atEnd = currentPly === replay.moves.length;
    const hasMoves = replay.moves.length > 0;
    prevButton.disabled = atStart;
    nextButton.disabled = atEnd;
    playButton.disabled = !hasMoves;
    playButton.textContent = isPlaying ? "Pause" : "Play";
    playButton.setAttribute("aria-pressed", String(isPlaying));
    plyRange.disabled = !hasMoves;
    plyRange.setAttribute(
      "aria-valuetext",
      currentPly + " of " + replay.moves.length + " moves",
    );
  }

  function updateMoveChoices() {
    moveChoices.replaceChildren();
    if (mode !== MODE_PLAY) {
      return;
    }

    if (!canHumanMove()) {
      const message = document.createElement("p");
      message.className = "empty-moves";
      if (gameError) {
        message.textContent = "Start a new game to continue.";
      } else if (!liveSnapshot) {
        message.textContent = "Choose your settings, then start the game.";
      } else if (isLiveTerminal()) {
        message.textContent = "The game is finished.";
      } else if (!liveGame.movesEnabled) {
        message.textContent = "Settings are unlocked. Start a new game to continue.";
      } else if (botPausedAfterUndo) {
        message.textContent = "The bot is paused. Undo again or continue the bot.";
      } else {
        message.textContent = "Waiting for " + activeBotName() + "…";
      }
      moveChoices.append(message);
      return;
    }

    for (const move of liveLegalMoves()) {
      const button = document.createElement("button");
      const extraTurn = move.extraTurn;
      button.type = "button";
      button.className = "move-choice" + (extraTurn ? " is-extra-turn" : "");
      button.textContent = formatPoint(move.to) + (extraTurn ? " ↻" : "");
      button.setAttribute(
        "aria-label",
        boardView.destinationLabel(liveState().ball, move.to, extraTurn),
      );
      button.dataset.point = boardView.pointId(move.to);
      button.addEventListener("mouseenter", function () {
        boardView.setHoveredDestination(move.to);
      });
      button.addEventListener("mouseleave", boardView.clearHoveredDestination);
      button.addEventListener("focus", function () {
        boardView.setHoveredDestination(move.to);
      });
      button.addEventListener("blur", boardView.clearHoveredDestination);
      button.addEventListener("click", function (event) {
        playHumanMove(move, event.detail === 0);
      });
      moveChoices.append(button);
    }
  }

  function updateMoveList() {
    moveList.replaceChildren();
    if (replay.moves.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-history";
      empty.textContent = mode === MODE_PLAY
        ? "Your moves and the bot’s moves will appear here."
        : "This replay contains no moves.";
      moveList.append(empty);
      return;
    }

    for (const move of replay.moves) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "move-row";
      if (move.ply === currentPly) {
        row.classList.add("is-active");
        row.setAttribute("aria-current", "step");
      }

      if (mode === MODE_REPLAY) {
        row.addEventListener("click", function () {
          setReplayPly(move.ply);
        });
      } else {
        row.disabled = true;
        row.classList.add("is-static");
      }

      const ply = document.createElement("span");
      ply.textContent = "#" + move.ply;

      const detail = document.createElement("span");
      const player = document.createElement("span");
      player.className = move.player === Player.Two ? "player-two" : "player-one";
      player.textContent = relativePlayerName(move.player);
      detail.append(
        player,
        " " + formatPoint(move.from) + " → " + formatPoint(move.to),
      );

      row.append(ply, detail);
      if (move.extraTurn) {
        const badge = document.createElement("span");
        badge.className = "extra-turn";
        badge.textContent = "Extra turn";
        row.append(badge);
      } else {
        row.append(document.createElement("span"));
      }
      moveList.append(row);
    }

    const active = moveList.querySelector(".is-active");
    if (active) {
      keepMoveRowVisible(active);
    }
  }

  function keepMoveRowVisible(row) {
    const listBounds = moveList.getBoundingClientRect();
    const rowBounds = row.getBoundingClientRect();
    if (rowBounds.top < listBounds.top) {
      moveList.scrollTop -= listBounds.top - rowBounds.top;
    } else if (rowBounds.bottom > listBounds.bottom) {
      moveList.scrollTop += rowBounds.bottom - listBounds.bottom;
    }
  }

  function startPlayback() {
    if (mode !== MODE_REPLAY || replay.moves.length === 0) {
      return;
    }
    if (currentPly >= replay.moves.length) {
      setReplayPly(0);
    }
    isPlaying = true;
    lastStepTime = 0;
    updateReplayControls();
  }

  function stopPlayback() {
    isPlaying = false;
    updateReplayControls();
  }

  function tick(timestamp) {
    if (mode === MODE_REPLAY && isPlaying) {
      const delay = 850 / speed;
      if (lastStepTime === 0) {
        lastStepTime = timestamp;
      }
      if (timestamp - lastStepTime >= delay) {
        setReplayPly(currentPly + 1);
        lastStepTime = timestamp;
      }
    }
    window.requestAnimationFrame(tick);
  }

  function renderInterface() {
    updateModeControls();
    updateText();
    updateReplayControls();
    updateMoveChoices();
    updateMoveList();
    boardView.draw();
    boardView.renderTargets();
  }

  prevButton.addEventListener("click", function () {
    setReplayPly(currentPly - 1);
  });
  playButton.addEventListener("click", function () {
    if (isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });
  nextButton.addEventListener("click", function () {
    setReplayPly(currentPly + 1);
  });
  plyRange.addEventListener("input", function () {
    setReplayPly(Number(plyRange.value));
  });
  speedSelect.addEventListener("change", function () {
    speed = Number(speedSelect.value);
  });
  gameSetup.removeEventListener("submit", preventSetupSubmit);
  gameSetup.addEventListener("submit", startNewGame);
  changeSettingsButton.addEventListener("click", changeSettings);
  exportGameButton.addEventListener("click", exportHumanMatch);
  undoMoveButton.addEventListener("click", undoLastMove);
  resumeBotButton.addEventListener("click", resumeBotTurn);
  replaySetup.addEventListener("submit", generateBotReplay);
  botSelect.addEventListener("change", syncBotSetupFields);
  replayOneBotSelect.addEventListener("change", syncBotSetupFields);
  replayTwoBotSelect.addEventListener("change", syncBotSetupFields);
  playModeButton.addEventListener("click", showPlayMode);
  replayModeButton.addEventListener("click", showReplayMode);

  fileInput.addEventListener("change", async function () {
    const file = fileInput.files?.[0];
    if (!file) {
      return;
    }
    cancelReplayGeneration();
    try {
      const raw = JSON.parse(await file.text());
      loadReplay(raw);
    } catch (error) {
      window.alert("Could not load replay: " + error.message);
    } finally {
      fileInput.value = "";
    }
  });

  canvas.addEventListener("click", function (event) {
    const move = boardView.nearestLegalMove(event);
    if (move) {
      playHumanMove(move, false);
    }
  });
  canvas.addEventListener("pointermove", function (event) {
    const move = boardView.nearestLegalMove(event);
    if (move) {
      boardView.setHoveredDestination(move.to);
    } else {
      boardView.clearHoveredDestination();
    }
  });
  canvas.addEventListener("pointerleave", boardView.clearHoveredDestination);

  window.addEventListener("resize", function () {
    boardView.draw();
    boardView.renderTargets();
  });
  if ("ResizeObserver" in window) {
    new ResizeObserver(function () {
      boardView.draw();
      boardView.renderTargets();
    }).observe(canvas);
  }

  document.addEventListener("keydown", function (event) {
    if (mode !== MODE_REPLAY) {
      return;
    }
    const target = event.target;
    if (target instanceof Element &&
        target.closest("button, input, select, textarea, a, [contenteditable]")) {
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setReplayPly(currentPly - 1);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      setReplayPly(currentPly + 1);
    }
    if (event.key === " ") {
      event.preventDefault();
      isPlaying ? stopPlayback() : startPlayback();
    }
  });

  newGameButton.disabled = false;
  renderInterface();
  window.requestAnimationFrame(tick);
}()).catch(function (error) {
  const status = document.querySelector("#statusLabel");
  const detail = document.querySelector("#moveLabel");
  const setupError = document.querySelector("#setupError");
  const turnBadge = document.querySelector("#boardTurnBadge");
  if (status) {
    status.textContent = "Game engine unavailable";
  }
  if (detail) {
    detail.textContent = "The compiled C++ engine could not load.";
  }
  if (setupError) {
    setupError.textContent = error.message;
  }
  if (turnBadge) {
    turnBadge.textContent = "Engine unavailable";
  }
  for (const control of document.querySelectorAll("button, input, select")) {
    control.disabled = true;
  }
});
