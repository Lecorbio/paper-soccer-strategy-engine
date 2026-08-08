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
  const gameReview = globalThis.PaperSoccerGameReview;
  if (!gameReview) {
    throw new Error("Paper soccer Game Review client did not load");
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
  support.installExpertOpponentOptions(
    [botSelect, replayOneBotSelect, replayTwoBotSelect],
    globalThis.PaperSoccerGameReviewGate,
    document,
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
  const reviewGameButton = document.querySelector("#reviewGameButton");
  const gameReviewPanel = document.querySelector("#gameReviewPanel");
  const closeReviewButton = document.querySelector("#closeReviewButton");
  const fastReviewButton = document.querySelector("#fastReviewButton");
  const deepReviewButton = document.querySelector("#deepReviewButton");
  const cancelReviewButton = document.querySelector("#cancelReviewButton");
  const reviewTransportNotice = document.querySelector("#reviewTransportNotice");
  const reviewProgress = document.querySelector("#reviewProgress");
  const reviewProgressText = document.querySelector("#reviewProgressText");
  const reviewSummary = document.querySelector("#reviewSummary");
  const reviewGradeCounts = document.querySelector("#reviewGradeCounts");
  const reviewBestActionRate = document.querySelector("#reviewBestActionRate");
  const reviewLargestLoss = document.querySelector("#reviewLargestLoss");
  const possessionTimelineSection = document.querySelector(
    "#possessionTimelineSection",
  );
  const possessionTimeline = document.querySelector("#possessionTimeline");
  const possessionReviewDetail = document.querySelector(
    "#possessionReviewDetail",
  );
  const possessionReviewTitle = document.querySelector("#possessionReviewTitle");
  const possessionGrade = document.querySelector("#possessionGrade");
  const possessionLoss = document.querySelector("#possessionLoss");
  const possessionProof = document.querySelector("#possessionProof");
  const possessionBorderline = document.querySelector("#possessionBorderline");
  const possessionDiagnostics = document.querySelector("#possessionDiagnostics");
  const tryLineButton = document.querySelector("#tryLineButton");
  const leaveTryLineButton = document.querySelector("#leaveTryLineButton");
  const tryLineStatus = document.querySelector("#tryLineStatus");
  const exportReviewButton = document.querySelector("#exportReviewButton");

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

  const defaultSourceReplay = gameReview.cloneJson(readDefaultReplay());
  const defaultReplay = normalizeReplay(defaultSourceReplay);
  let mode = MODE_PLAY;
  let replay = newGamePreviewReplay(defaultReplay);
  let sourceReplay = gameReview.cloneJson(replay);
  let sourceReplayValidated = true;
  let importedReviewDocument = null;
  let replaySession = {
    replay: defaultReplay,
    sourceReplay: defaultSourceReplay,
    sourceReplayValidated: true,
    importedReviewDocument: null,
    currentPly: 0,
  };
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
  let reviewPanelOpen = false;
  let reviewRunning = false;
  let reviewMode = gameReview.ReviewMode.Fast;
  let reviewSnapshot = null;
  let fastReviewSnapshot = null;
  let deepReviewSnapshot = null;
  let reviewPrepared = null;
  let reviewSandboxReady = false;
  let selectedPossessionIndex = 0;
  let pendingDeepRefinement = false;
  let tryLineReplay = null;
  let tryLinePly = 0;
  let tryLineSnapshot = null;
  let tryLinePending = false;
  let replayImportId = 0;
  let validatedImportedDocument = null;

  const analysisRunner = gameReview.createAnalysisRunner({
    globalObject: globalThis,
    documentObject: document,
    timerHost: window,
    protocol: window.location.protocol,
    onEvent: handleReviewEvent,
  });

  const boardView = board.createBoardView({
    boardTargets: boardTargets,
    canHumanMove: canHumanMove,
    canvas: canvas,
    currentBall: currentBall,
    currentMoveStartIndex: currentMoveStartIndex,
    getCurrentPly: displayCurrentPly,
    getReplay: displayReplay,
    getReviewOverlay: currentReviewOverlay,
    liveLegalMoves: interactiveLegalMoves,
    liveState: interactiveState,
    onHumanMove: playHumanMove,
    playerTwo: Player.Two,
    samePoint: samePoint,
  });

  function readDefaultReplay() {
    const raw = document.querySelector("#default-replay").textContent;
    return JSON.parse(raw);
  }

  function displayReplay() {
    return tryLineReplay ?? replay;
  }

  function displayCurrentPly() {
    return tryLineReplay ? tryLinePly : currentPly;
  }

  function invalidateReview(closePanel = true) {
    analysisRunner.cancel();
    reviewRunning = false;
    reviewSnapshot = null;
    fastReviewSnapshot = null;
    deepReviewSnapshot = null;
    reviewPrepared = null;
    reviewSandboxReady = false;
    selectedPossessionIndex = 0;
    pendingDeepRefinement = false;
    analysisRunner.closeSandbox();
    tryLineReplay = null;
    tryLinePly = 0;
    tryLineSnapshot = null;
    tryLinePending = false;
    if (closePanel) {
      reviewPanelOpen = false;
    }
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
      player.kind !== "DeepTurnSearchBot" &&
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
    if ((player.kind === "Rank5DerivedBot" ||
         player.kind === "DeepTurnSearchBot") &&
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
    if (kind === BotKind.DeepTurnSearch) {
      return "DeepTurnSearchBot";
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
        value === BotKind.Rank5Derived || value === BotKind.DeepTurnSearch) {
      return value;
    }
    throw new Error("Unsupported bot kind: " + String(value));
  }

  function readBotConfig(kindSelect, botSeedInput, iterationsInput, depthInput) {
    const kind = selectedBotKind(kindSelect.value);
    if (kind === BotKind.Rank5Derived || kind === BotKind.DeepTurnSearch) {
      return {
        kind: kind,
        seed: "0",
        iterations: DEFAULT_MCTS_ITERATIONS,
        depth: DEFAULT_ALPHA_BETA_DEPTH,
        maxNodes: kind === BotKind.Rank5Derived
          ? RANK5_DERIVED_MAX_NODES
          : undefined,
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
    if (config.kind === BotKind.DeepTurnSearch) {
      return 0;
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

  function interactiveState() {
    return tryLineSnapshot?.state ?? liveState();
  }

  function interactiveLegalMoves() {
    return tryLineSnapshot?.legalMoves ?? liveLegalMoves();
  }

  function isLiveTerminal() {
    return !liveSnapshot || liveState().status !== Status.InProgress;
  }

  function adoptLiveSnapshot(snapshot) {
    invalidateReview();
    liveSnapshot = snapshot;
    humanPlayer = snapshot.humanPlayer;
    sourceReplay = gameReview.cloneJson(snapshot.replay);
    sourceReplayValidated = true;
    importedReviewDocument = null;
    replay = normalizeReplay(sourceReplay);
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
      sourceReplay: sourceReplay,
      sourceReplayValidated: sourceReplayValidated,
      importedReviewDocument: importedReviewDocument,
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
    sourceReplay = replaySession.sourceReplay;
    sourceReplayValidated = replaySession.sourceReplayValidated;
    importedReviewDocument = replaySession.importedReviewDocument;
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
    replaySession = {
      replay: replay,
      sourceReplay: sourceReplay,
      sourceReplayValidated: sourceReplayValidated,
      importedReviewDocument: importedReviewDocument,
      currentPly: currentPly,
    };
    invalidateReview();
    stopPlayback();
    if (!liveSession) {
      mode = MODE_PLAY;
      replay = newGamePreviewReplay(defaultReplay);
      sourceReplay = gameReview.cloneJson(replay);
      sourceReplayValidated = true;
      importedReviewDocument = null;
      invalidateReview();
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
    sourceReplay = session.sourceReplay;
    sourceReplayValidated = session.sourceReplayValidated;
    importedReviewDocument = session.importedReviewDocument;
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
    if (tryLineSnapshot) {
      return !tryLinePending &&
        tryLineSnapshot.state?.status === Status.InProgress &&
        interactiveLegalMoves().length > 0;
    }
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

    if (tryLineSnapshot) {
      playSandboxMove(move, keyboardActivated);
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
      replayOneBotSelect.value === BotKind.Rank5Derived ||
      replayOneBotSelect.value === BotKind.DeepTurnSearch;
    replayTwoSeedInput.disabled = isGeneratingReplay ||
      replayTwoBotSelect.value === BotKind.Rank5Derived ||
      replayTwoBotSelect.value === BotKind.DeepTurnSearch;
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
      loadReplay(generated.replay, true, { validated: true });
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

  function loadReplay(raw, preserveGeneration = false, options = {}) {
    if (!preserveGeneration) {
      cancelReplayGeneration();
    }
    const exactSource = gameReview.cloneJson(support.unwrapReplayDocument(raw));
    const normalized = normalizeReplay(exactSource);
    const imported = support.importedReview(raw);
    if (mode === MODE_PLAY) {
      saveLiveSession();
    } else {
      stopPlayback();
    }
    invalidateReview();
    replay = normalized;
    sourceReplay = exactSource;
    sourceReplayValidated = options.validated === true ||
      raw === validatedImportedDocument;
    importedReviewDocument = imported ? gameReview.cloneJson(imported) : null;
    replaySession = {
      replay: replay,
      sourceReplay: sourceReplay,
      sourceReplayValidated: sourceReplayValidated,
      importedReviewDocument: importedReviewDocument,
      currentPly: 0,
    };
    mode = MODE_REPLAY;
    liveSnapshot = null;
    gameError = "";
    boardView.resetHover();
    lastLiveMove = null;
    currentPly = 0;
    stopPlayback();
    plyRange.max = String(replay.moves.length);
    setReplayPly(0);

    if (importedReviewDocument) {
      try {
        reviewPrepared = gameReview.prepareReplay(sourceReplay);
        reviewSnapshot = importedReviewDocument;
        reviewSandboxReady = false;
        if (String(importedReviewDocument.mode).toLowerCase() === "deep") {
          deepReviewSnapshot = importedReviewDocument;
          reviewMode = gameReview.ReviewMode.Deep;
        } else {
          fastReviewSnapshot = importedReviewDocument;
          reviewMode = gameReview.ReviewMode.Fast;
        }
        reviewPanelOpen = true;
        selectedPossessionIndex = 0;
        renderInterface();
      } catch (error) {
        importedReviewDocument = null;
      }
    }
  }

  function clampPly(ply) {
    return Math.max(0, Math.min(replay.moves.length, ply));
  }

  function setReplayPly(ply) {
    if (mode !== MODE_REPLAY) {
      return;
    }
    leaveTryLine();
    currentPly = clampPly(ply);
    if (reviewSnapshot && currentPly > 0) {
      const selected = support.possessionForPly(reviewSnapshot, currentPly);
      const possessions = support.reviewPossessions(reviewSnapshot);
      const index = possessions.indexOf(selected);
      if (index >= 0) {
        selectedPossessionIndex = index;
      }
    }
    plyRange.value = String(currentPly);
    if (currentPly >= replay.moves.length) {
      isPlaying = false;
    }
    renderInterface();
  }

  function currentBall() {
    const shownReplay = displayReplay();
    const shownPly = displayCurrentPly();
    if (shownPly === 0) {
      return shownReplay.start;
    }
    return shownReplay.moves[shownPly - 1].to;
  }

  function currentMoveStartIndex() {
    const shownReplay = displayReplay();
    const shownPly = displayCurrentPly();
    if (shownPly === 0) {
      return 0;
    }

    const player = shownReplay.moves[shownPly - 1].player;
    let startIndex = shownPly - 1;
    while (startIndex > 0 &&
           shownReplay.moves[startIndex - 1].extraTurn &&
           shownReplay.moves[startIndex - 1].player === player) {
      startIndex -= 1;
    }
    return startIndex;
  }

  function possessionAction(possession, name) {
    const value = possession?.[name] ??
      (name === "playedAction" ? possession?.played : possession?.recommended);
    if (Array.isArray(value)) {
      return value;
    }
    if (Array.isArray(value?.edges)) {
      return value.edges;
    }
    if (Array.isArray(value?.moves)) {
      return value.moves;
    }
    return [];
  }

  function actionPath(action, fallbackStart) {
    if (!Array.isArray(action) || action.length === 0) {
      return [];
    }
    if (action[0]?.from && action[0]?.to) {
      return [action[0].from].concat(action.map(function (edge) { return edge.to; }));
    }
    if (action.every(function (point) {
      return Number.isInteger(point?.x) && Number.isInteger(point?.y);
    })) {
      const beginsAtStart = fallbackStart && samePoint(action[0], fallbackStart);
      return beginsAtStart ? action : [fallbackStart].concat(action);
    }
    return [];
  }

  function selectedPossession() {
    const possessions = support.reviewPossessions(reviewSnapshot);
    if (possessions.length === 0) {
      return null;
    }
    selectedPossessionIndex = Math.max(
      0,
      Math.min(possessions.length - 1, selectedPossessionIndex),
    );
    return possessions[selectedPossessionIndex];
  }

  function possessionBoundaryPoint(possession) {
    const played = possessionAction(possession, "playedAction");
    if (played[0]?.from) {
      return played[0].from;
    }
    const range = support.possessionRange(possession);
    return range.startPly <= 1
      ? replay.start
      : replay.moves[range.startPly - 2]?.to ?? replay.start;
  }

  function divergenceIndex(possession) {
    const value = possession?.firstDivergenceEdge ??
      possession?.firstDivergenceIndex ?? possession?.divergenceIndex ??
      possession?.divergence?.edgeIndex ?? possession?.divergence;
    return Number.isInteger(value) && value >= 0 ? value : null;
  }

  function currentReviewOverlay() {
    if (!reviewPanelOpen || !reviewSnapshot) {
      return null;
    }
    const possession = selectedPossession();
    if (!possession) {
      return null;
    }
    const boundary = possessionBoundaryPoint(possession);
    const playedPath = actionPath(
      possessionAction(possession, "playedAction"),
      boundary,
    );
    const recommendedPath = actionPath(
      possessionAction(possession, "recommendedAction"),
      boundary,
    );
    const index = divergenceIndex(possession);
    const divergence = index !== null && playedPath[index] && playedPath[index + 1]
      ? { from: playedPath[index], to: playedPath[index + 1] }
      : null;
    return {
      playedPath: tryLineReplay ? [] : playedPath,
      recommendedPath,
      divergence: tryLineReplay ? null : divergence,
      sandbox: Boolean(tryLineReplay),
    };
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
    const usesDeepProfile = configuration.kind === "DeepTurnSearchBot";
    activeBotConfiguration.textContent = configuration.kind +
      (usesRank5Profile || usesDeepProfile
        ? " — " + (usesRank5Profile ? "50k demo profile" : configuration.profile) +
          " · deterministic " +
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
      configuration.kind !== "JacekInspiredBot" && !usesRank5Profile &&
      !usesDeepProfile) ||
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

    if (search.searchType === "rank5Derived" ||
        search.searchType === "deepTurnSearch") {
      searchVisits.textContent = "Latest completed · depth " +
        formatCount(search.completedDepth) + " · Player 1 score " +
        formatCount(search.rootScore);
      searchShape.textContent = formatCount(search.visitedNodes) + " of " +
        formatCount(search.requestedNodes) + " deterministic nodes · " +
        formatDecisionTime(search.decisionTimeNs);
      searchIndicators.textContent = search.budgetExhausted
        ? formatCount(search.requestedNodes) + "-node budget reached"
        : "Search completed within the " + formatCount(search.requestedNodes) +
          "-node budget";
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

  function reviewProgressCounts(snapshot) {
    const completed = Number(
      snapshot?.completedPossessions ?? snapshot?.completedSteps ??
      snapshot?.progress?.completed ?? 0,
    );
    const total = Number(
      snapshot?.totalPossessions ?? snapshot?.totalSteps ??
      snapshot?.progress?.total ?? 0,
    );
    return {
      completed: Number.isFinite(completed) ? Math.max(0, completed) : 0,
      total: Number.isFinite(total) ? Math.max(0, total) : 0,
    };
  }

  function handleReviewEvent(event) {
    if (event.type === "transport") {
      reviewTransportNotice.textContent = event.transport === "worker"
        ? "Analysis runs in a separate browser worker."
        : "Direct-file fallback: analysis runs on the main thread, one " +
          "position per event-loop task.";
      renderInterface();
      return;
    }
    if (event.type === "loading") {
      if (event.kind === "validate") {
        return;
      }
      reviewRunning = true;
      reviewSandboxReady = false;
      reviewMode = event.mode ?? reviewMode;
      reviewPanelOpen = true;
      reviewProgressText.textContent = "Loading the dedicated analysis engine…";
      renderInterface();
      return;
    }
    if (event.type === "validated") {
      if (reviewRunning) {
        reviewSnapshot = event.snapshot;
        reviewProgressText.textContent = "Replay validated. Preparing possession analysis…";
        renderInterface();
      }
      return;
    }
    if (event.type === "progress") {
      reviewSnapshot = event.snapshot;
      reviewRunning = true;
      renderInterface();
      return;
    }
    if (event.type === "complete") {
      reviewSnapshot = event.snapshot;
      reviewPrepared = event.prepared;
      reviewRunning = false;
      reviewSandboxReady = true;
      reviewMode = event.mode;
      if (event.mode === gameReview.ReviewMode.Deep) {
        deepReviewSnapshot = event.snapshot;
      } else {
        fastReviewSnapshot = event.snapshot;
      }
      selectedPossessionIndex = Math.min(
        selectedPossessionIndex,
        Math.max(0, support.reviewPossessions(reviewSnapshot).length - 1),
      );
      renderInterface();
      if (event.mode === gameReview.ReviewMode.Fast && pendingDeepRefinement) {
        pendingDeepRefinement = false;
        window.setTimeout(function () {
          runReview(gameReview.ReviewMode.Deep);
        }, 0);
      }
      return;
    }
    if (event.type === "cancelled") {
      reviewRunning = false;
      pendingDeepRefinement = false;
      reviewProgressText.textContent =
        "Review cancelled. A search already in progress may finish internally; " +
        "its stale result will be ignored.";
      renderInterface();
      return;
    }
    if (event.type === "error") {
      reviewRunning = false;
      pendingDeepRefinement = false;
      reviewProgressText.textContent = "Game Review could not continue: " +
        event.error.message;
      renderInterface();
    }
  }

  function canReviewCurrentGame() {
    if (mode === MODE_PLAY) {
      return Boolean(liveSnapshot) && !gameError && isLiveTerminal();
    }
    return sourceReplayValidated && replay.moves.length > 0;
  }

  function openGameReview() {
    if (!canReviewCurrentGame()) {
      return;
    }
    if (mode === MODE_PLAY) {
      loadReplay(sourceReplay, false, { validated: true });
    }
    reviewPanelOpen = true;
    selectedPossessionIndex = 0;
    if (!reviewSnapshot) {
      runReview(gameReview.ReviewMode.Fast);
    } else {
      renderInterface();
    }
  }

  function runReview(requestedMode) {
    if (!sourceReplayValidated || reviewRunning) {
      return;
    }
    leaveTryLine();
    reviewSandboxReady = false;
    reviewMode = requestedMode;
    reviewPanelOpen = true;
    reviewProgressText.textContent = requestedMode === gameReview.ReviewMode.Deep
      ? "Starting Deep refinement after the Fast preview…"
      : "Starting deterministic Fast analysis…";
    try {
      const request = analysisRunner.start(sourceReplay, requestedMode);
      reviewPrepared = request.prepared;
    } catch (error) {
      reviewRunning = false;
      reviewProgressText.textContent = "Game Review could not start: " + error.message;
    }
    renderInterface();
  }

  function startFastReview() {
    pendingDeepRefinement = false;
    runReview(gameReview.ReviewMode.Fast);
  }

  function startDeepReview() {
    if (reviewRunning) {
      return;
    }
    if (!fastReviewSnapshot) {
      pendingDeepRefinement = true;
      runReview(gameReview.ReviewMode.Fast);
      return;
    }
    runReview(gameReview.ReviewMode.Deep);
  }

  function cancelReview() {
    if (!reviewRunning) {
      return;
    }
    analysisRunner.cancel();
  }

  function closeGameReview() {
    analysisRunner.cancel();
    reviewRunning = false;
    pendingDeepRefinement = false;
    reviewPanelOpen = false;
    leaveTryLine();
    renderInterface();
  }

  function formatLoss(loss) {
    return loss === null ? "Unavailable" : Number(loss).toFixed(1) + " pp";
  }

  function proofText(possession) {
    const proof = String(
      possession?.proofStatus ?? possession?.proof ?? "unknown",
    ).toLowerCase();
    if (proof === "provenwin" || proof === "proven_win" || proof === "win") {
      return "Proven win";
    }
    if (proof === "provenloss" || proof === "proven_loss" || proof === "loss") {
      return "Proven loss";
    }
    return "Unknown — heuristic estimate";
  }

  function diagnosticText(possession) {
    const diagnostic = possession?.before ?? possession?.diagnostics ??
      possession?.search ?? {};
    const search = diagnostic.search ?? diagnostic;
    const parts = [];
    const depth = search.completedDepth ?? search.completedTurnDepth ??
      search.completed_turn_depth;
    const nodes = search.nodes ?? search.visitedNodes;
    if (Number.isInteger(depth)) {
      parts.push("depth " + formatCount(depth));
    }
    if (Number.isFinite(Number(nodes))) {
      parts.push(formatCount(nodes) + " nodes");
    }
    if (diagnostic.exact === true) {
      const reachable = diagnostic.reachableEdgeCount ??
        diagnostic.reachable_edge_count;
      parts.push("exact oracle" + (Number.isInteger(reachable)
        ? " · " + reachable + " reachable edges"
        : ""));
    }
    return parts.length > 0 ? parts.join(" · ") :
      "No completed search diagnostics are available.";
  }

  function updateReviewPanel() {
    gameReviewPanel.hidden = !reviewPanelOpen;
    reviewGameButton.hidden = reviewPanelOpen || !canReviewCurrentGame();
    if (!reviewPanelOpen) {
      return;
    }
    gameReviewPanel.dataset.sourceReplaySha256 = gameReview.sha256(
      gameReview.canonicalJson(sourceReplay),
    );

    fastReviewButton.disabled = reviewRunning;
    deepReviewButton.disabled = reviewRunning;
    cancelReviewButton.hidden = !reviewRunning;
    exportReviewButton.disabled = reviewRunning || !reviewSnapshot ||
      !gameReview.reviewDone(reviewSnapshot);
    reviewProgress.hidden = !reviewRunning;

    const progress = reviewProgressCounts(reviewSnapshot);
    reviewProgress.max = Math.max(1, progress.total);
    reviewProgress.value = Math.min(progress.completed, reviewProgress.max);
    if (reviewRunning && progress.total > 0) {
      reviewProgressText.textContent = (reviewMode === gameReview.ReviewMode.Deep
        ? "Deep refinement"
        : "Fast preview") + ": " + progress.completed + " of " +
        progress.total + " possession steps complete. Cancellation takes " +
        "effect after the current synchronous search.";
    } else if (!reviewRunning && reviewSnapshot && gameReview.reviewDone(reviewSnapshot)) {
      reviewProgressText.textContent = (reviewMode === gameReview.ReviewMode.Deep
        ? "Deep refinement complete."
        : "Fast preview complete. You can refine it with Deep analysis.");
    }

    const possessions = support.reviewPossessions(reviewSnapshot);
    const hasResults = possessions.some(function (possession) {
      return possession.analyzed === true;
    });
    reviewSummary.hidden = !hasResults;
    possessionTimelineSection.hidden = !hasResults;
    possessionReviewDetail.hidden = !hasResults;
    possessionTimeline.replaceChildren();
    if (!hasResults) {
      return;
    }

    const summary = support.reviewSummary(reviewSnapshot);
    reviewGradeCounts.textContent = Object.entries(summary.counts)
      .filter(function (entry) { return entry[1] > 0; })
      .map(function (entry) { return entry[0] + " " + entry[1]; })
      .join(" · ") || "No graded possessions";
    reviewBestActionRate.textContent = summary.totalActions === 0
      ? "Unavailable"
      : summary.bestActions + " of " + summary.totalActions + " (" +
        Math.round(100 * summary.bestActions / summary.totalActions) + "%)";
    reviewLargestLoss.textContent = formatLoss(summary.largestLoss);

    possessions.forEach(function (possession, index) {
      const grade = support.normalizedGrade(possession.grade);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "possession-badge grade-" + grade.toLowerCase();
      button.textContent = String(index + 1);
      button.title = "Possession " + (index + 1) + ": " + grade;
      button.setAttribute("role", "listitem");
      if (index === selectedPossessionIndex) {
        button.classList.add("is-active");
        button.setAttribute("aria-current", "step");
      }
      button.addEventListener("click", function () {
        selectedPossessionIndex = index;
        leaveTryLine();
        const range = support.possessionRange(possession);
        setReplayPly(range.endPly);
      });
      possessionTimeline.append(button);
    });

    const possession = selectedPossession();
    const range = support.possessionRange(possession);
    const grade = support.normalizedGrade(possession.grade);
    possessionReviewTitle.textContent = "Possession " +
      (selectedPossessionIndex + 1) + " · edges " + range.startPly +
      (range.endPly === range.startPly ? "" : "–" + range.endPly);
    possessionGrade.textContent = grade;
    possessionGrade.className = "possession-grade grade-" + grade.toLowerCase();
    possessionLoss.textContent = "Estimated win-chance loss: " +
      formatLoss(support.estimatedLoss(possession));
    possessionProof.textContent = "Proof status: " + proofText(possession);
    possessionBorderline.textContent = possession.borderline === true
      ? "Borderline: within one percentage point of a grade threshold."
      : "";
    possessionDiagnostics.textContent = diagnosticText(possession);
    tryLineButton.disabled = reviewRunning || tryLinePending ||
      !reviewSandboxReady ||
      possessionAction(possession, "recommendedAction").length === 0;
    tryLineButton.hidden = Boolean(tryLineReplay);
    leaveTryLineButton.hidden = !tryLineReplay;
    if (!reviewSandboxReady && importedReviewDocument) {
      tryLineStatus.textContent = "Imported diagnostics are read-only. Run Fast " +
        "preview or Deep refinement locally to enable an authoritative fork.";
    } else if (!tryLineReplay && !tryLinePending &&
               !tryLineStatus.textContent.startsWith("Could not")) {
      tryLineStatus.textContent = "";
    }
  }

  async function tryRecommendedLine() {
    const possession = selectedPossession();
    if (!possession || tryLineReplay || tryLinePending) {
      return;
    }
    const recommended = possessionAction(possession, "recommendedAction");
    if (recommended.length === 0) {
      return;
    }
    const originalFingerprint = gameReview.canonicalJson(sourceReplay);
    tryLinePending = true;
    tryLineStatus.textContent = "Creating an authoritative temporary fork…";
    renderInterface();
    try {
      const snapshot = await analysisRunner.startSandbox(
        selectedPossessionIndex,
      );
      if (gameReview.canonicalJson(sourceReplay) !== originalFingerprint) {
        throw new Error("The try-line sandbox changed the source replay.");
      }
      tryLineSnapshot = snapshot;
      const forkReplay = gameReview.cloneJson(snapshot.replay);
      forkReplay.players = gameReview.cloneJson(replay.players);
      tryLineReplay = normalizeReplay(forkReplay);
      tryLinePly = tryLineReplay.moves.length;
      tryLineStatus.textContent = "Temporary fork: start at the possession boundary " +
        "and follow the ghosted recommendation or choose another highlighted legal " +
        "edge. You control either side; the original replay remains unchanged.";
    } catch (error) {
      analysisRunner.closeSandbox();
      tryLineSnapshot = null;
      tryLineReplay = null;
      tryLinePly = 0;
      tryLineStatus.textContent = "Could not create the temporary fork: " +
        error.message;
    } finally {
      tryLinePending = false;
      boardView.resetHover();
      renderInterface();
    }
  }

  async function playSandboxMove(move, keyboardActivated) {
    const legal = interactiveLegalMoves().find(function (candidate) {
      return candidate.id === move.id && samePoint(candidate.to, move.to);
    });
    if (!legal || tryLinePending) {
      return;
    }
    const originalFingerprint = gameReview.canonicalJson(sourceReplay);
    tryLinePending = true;
    tryLineStatus.textContent = "Applying the sandbox edge with the C++ rules engine…";
    renderInterface();
    try {
      const snapshot = await analysisRunner.playSandbox(legal.to);
      if (gameReview.canonicalJson(sourceReplay) !== originalFingerprint) {
        throw new Error("The try-line sandbox changed the source replay.");
      }
      tryLineSnapshot = snapshot;
      const forkReplay = gameReview.cloneJson(snapshot.replay);
      forkReplay.players = gameReview.cloneJson(replay.players);
      tryLineReplay = normalizeReplay(forkReplay);
      tryLinePly = tryLineReplay.moves.length;
      if (snapshot.state.status === Status.InProgress) {
        tryLineStatus.textContent = "Temporary fork: choose another highlighted " +
          "legal edge. You control both sides; the original replay is unchanged.";
      } else {
        tryLineStatus.textContent = "The temporary fork reached " +
          statusName(snapshot.state.status).toLowerCase() +
          ". The original replay is unchanged.";
      }
      if (keyboardActivated) {
        canvas.focus();
      }
    } catch (error) {
      tryLineStatus.textContent = "The sandbox move was rejected: " + error.message;
    } finally {
      tryLinePending = false;
      boardView.resetHover();
      renderInterface();
    }
  }

  function leaveTryLine() {
    if (!tryLineReplay) {
      analysisRunner.closeSandbox();
      tryLineSnapshot = null;
      tryLinePending = false;
      return;
    }
    analysisRunner.closeSandbox();
    tryLineReplay = null;
    tryLinePly = 0;
    tryLineSnapshot = null;
    tryLinePending = false;
    tryLineStatus.textContent = "";
    boardView.resetHover();
  }

  function exportGameReview() {
    if (!reviewSnapshot || !reviewPrepared || reviewRunning ||
        !gameReview.reviewDone(reviewSnapshot)) {
      return;
    }
    const exported = gameReview.buildReviewExport(reviewPrepared, reviewSnapshot);
    const contents = JSON.stringify(exported, null, 2) + "\n";
    const blob = new Blob([contents], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = support.gameReviewFilename(
      exported.replaySha256,
      exported.mode ?? reviewMode,
    );
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 0);
  }

  function updateText() {
    if (tryLineReplay) {
      updateReplayText();
    } else if (mode === MODE_PLAY) {
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

    if (tryLineReplay) {
      plyLabel.textContent = "Temporary recommended line";
      statusLabel.textContent = "Try this line sandbox";
      const legalCount = interactiveLegalMoves().length;
      moveLabel.textContent = tryLineSnapshot?.state?.status === Status.InProgress
        ? "Continue from the reviewed boundary with " + legalCount +
          (legalCount === 1 ? " legal edge." : " legal edges.")
        : "The temporary fork has reached a terminal position.";
      boardDescription.textContent = statusLabel.textContent + ". Ball at row " +
        currentBall().y + ", column " + currentBall().x +
        ". The source replay has not been changed. " + moveLabel.textContent;
      return;
    }

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
    const playingSandbox = Boolean(tryLineSnapshot);
    gameSetup.hidden = !playingGame;
    replaySetup.hidden = playingGame;
    if (reviewPanelOpen) {
      replaySetup.hidden = true;
    }
    moveChoicesSection.hidden = !playingGame && !playingSandbox;
    liveHistoryControls.hidden = !playingGame;
    replayControls.hidden = playingGame || playingSandbox;
    boardTargets.hidden = !playingGame && !playingSandbox;
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
    controlPanel.classList.toggle("is-review-mode", reviewPanelOpen);
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
    if (botSelect.value === BotKind.Rank5Derived ||
        botSelect.value === BotKind.DeepTurnSearch) {
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
    const playUsesFixedProfile = playUsesRank5Profile ||
      botSelect.value === BotKind.DeepTurnSearch;
    botSeedField.hidden = playUsesFixedProfile;
    seedInput.disabled = playUsesFixedProfile;
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
    const playerOneUsesFixedProfile = playerOneUsesRank5Profile ||
      replayOneBotSelect.value === BotKind.DeepTurnSearch;
    const playerTwoUsesFixedProfile = playerTwoUsesRank5Profile ||
      replayTwoBotSelect.value === BotKind.DeepTurnSearch;
    replayOneSeedField.hidden = playerOneUsesFixedProfile;
    replayTwoSeedField.hidden = playerTwoUsesFixedProfile;
    replayOneSeedInput.disabled = isGeneratingReplay || playerOneUsesFixedProfile;
    replayTwoSeedInput.disabled = isGeneratingReplay || playerTwoUsesFixedProfile;
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
    const playingSandbox = Boolean(tryLineSnapshot);
    if (mode !== MODE_PLAY && !playingSandbox) {
      return;
    }

    if (!canHumanMove()) {
      const message = document.createElement("p");
      message.className = "empty-moves";
      if (playingSandbox && tryLinePending) {
        message.textContent = "Applying the sandbox move…";
      } else if (playingSandbox) {
        message.textContent = "The temporary fork is finished.";
      } else if (gameError) {
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

    for (const move of interactiveLegalMoves()) {
      const button = document.createElement("button");
      const extraTurn = move.extraTurn;
      button.type = "button";
      button.className = "move-choice" + (extraTurn ? " is-extra-turn" : "");
      button.textContent = formatPoint(move.to) + (extraTurn ? " ↻" : "");
      button.setAttribute(
        "aria-label",
        boardView.destinationLabel(interactiveState().ball, move.to, extraTurn),
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
    updateReviewPanel();
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
  reviewGameButton.addEventListener("click", openGameReview);
  closeReviewButton.addEventListener("click", closeGameReview);
  fastReviewButton.addEventListener("click", startFastReview);
  deepReviewButton.addEventListener("click", startDeepReview);
  cancelReviewButton.addEventListener("click", cancelReview);
  tryLineButton.addEventListener("click", tryRecommendedLine);
  leaveTryLineButton.addEventListener("click", function () {
    leaveTryLine();
    renderInterface();
  });
  exportReviewButton.addEventListener("click", exportGameReview);
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
    const importId = replayImportId + 1;
    replayImportId = importId;
    try {
      const raw = JSON.parse(await file.text());
      replayGenerationStatus.textContent = "Validating replay with the C++ rules engine…";
      await analysisRunner.validate(raw);
      if (importId !== replayImportId) {
        return;
      }
      validatedImportedDocument = raw;
      loadReplay(raw);
      validatedImportedDocument = null;
      replayGenerationStatus.textContent = "Replay validated and ready for review.";
    } catch (error) {
      if (importId === replayImportId) {
        replayGenerationStatus.textContent = "";
        window.alert("Could not load replay: " + error.message);
      }
    } finally {
      validatedImportedDocument = null;
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
