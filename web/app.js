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

  const { BotKind, Player, Status, samePoint } = engine;
  const gameEngine = await engine.ready;

  const canvas = document.querySelector("#boardCanvas");
  const ctx = canvas.getContext("2d");
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
  const gameSetup = document.querySelector("#gameSetup");
  const playSetupEyebrow = document.querySelector("#playSetupEyebrow");
  const botSelect = document.querySelector("#botSelect");
  const sideSelect = document.querySelector("#sideSelect");
  const seedInput = document.querySelector("#seedInput");
  const botIterationsField = document.querySelector("#botIterationsField");
  const botIterationsInput = document.querySelector("#botIterationsInput");
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
  const replayOneIterationsField = document.querySelector(
    "#replayOneIterationsField",
  );
  const replayOneIterationsInput = document.querySelector(
    "#replayOneIterationsInput",
  );
  const replayTwoBotSelect = document.querySelector("#replayTwoBotSelect");
  const replayTwoSeedInput = document.querySelector("#replayTwoSeedInput");
  const replayTwoIterationsField = document.querySelector(
    "#replayTwoIterationsField",
  );
  const replayTwoIterationsInput = document.querySelector(
    "#replayTwoIterationsInput",
  );
  const replaySetupError = document.querySelector("#replaySetupError");
  const replayGenerationStatus = document.querySelector(
    "#replayGenerationStatus",
  );
  const moveChoicesSection = document.querySelector("#moveChoicesSection");
  const moveChoices = document.querySelector("#moveChoices");
  const replayControls = document.querySelector("#replayControls");

  const MODE_PLAY = "play";
  const MODE_REPLAY = "replay";
  const BOT_DELAY_MS = 520;
  const DEFAULT_MCTS_ITERATIONS = 2000;
  const MAX_MCTS_ITERATIONS = 10000;
  const MAX_REPLAY_PLIES = 512;
  const MAX_UINT64 = (1n << 64n) - 1n;

  let mode = MODE_PLAY;
  let replay = normalizeReplay(readDefaultReplay());
  let replaySession = { replay: replay, currentPly: 0 };
  let liveSession = null;
  let currentPly = 0;
  let isPlaying = false;
  let speed = 1;
  let lastStepTime = 0;
  let liveSnapshot = null;
  let humanPlayer = Player.One;
  const liveGame = support.createLiveGameLifecycle(window);
  let gameError = "";
  let hoveredDestination = null;
  let lastLiveMove = null;
  let activeBotConfig = {
    kind: BotKind.Random,
    seed: "12345",
    iterations: DEFAULT_MCTS_ITERATIONS,
  };
  let replayGenerationId = 0;
  let isGeneratingReplay = false;

  function readDefaultReplay() {
    const raw = document.querySelector("#default-replay").textContent;
    return JSON.parse(raw);
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
    const hasSeed = player.seed !== undefined && player.seed !== null &&
      String(player.seed).length > 0;
    const details = [];
    if (hasSeed) {
      details.push("seed " + String(player.seed));
    }
    if (Number.isInteger(player.iterations) && player.iterations > 0) {
      details.push(player.iterations + " new simulations per bot move");
    }
    const suffix = details.length > 0 ? " (" + details.join(", ") + ")" : "";
    return label + ": " + (player.kind ?? "Unknown") + suffix;
  }

  function botDisplayName(kind) {
    return kind === BotKind.Mcts ? "MctsBot" : "RandomBot";
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

  function readBotConfig(kindSelect, botSeedInput, iterationsInput) {
    const kind = kindSelect.value === BotKind.Mcts
      ? BotKind.Mcts
      : BotKind.Random;
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

    return {
      kind: kind,
      seed: seed.toString(),
      iterations: iterations,
    };
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
    hoveredDestination = null;
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
    hoveredDestination = null;
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
      startNewGame();
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
    hoveredDestination = null;
    plyRange.max = String(replay.moves.length);
    plyRange.value = String(currentPly);
    renderInterface();

    if (liveGame.movesEnabled && !gameError && !isLiveTerminal() &&
        liveState().toMove !== humanPlayer) {
      scheduleBotTurn();
    }
  }

  function startNewGame(event) {
    event?.preventDefault();

    let botConfig;
    try {
      botConfig = readBotConfig(botSelect, seedInput, botIterationsInput);
    } catch (error) {
      setupError.textContent = error.message;
      error.input?.focus();
      return;
    }

    liveGame.changeSettings();
    stopPlayback();
    setupError.textContent = "";
    gameError = "";
    mode = MODE_PLAY;
    liveSession = null;
    const selectedHuman = sideSelect.value === Player.Two
      ? Player.Two
      : Player.One;

    try {
      adoptLiveSnapshot(gameEngine.startGame(
        botConfig.seed,
        selectedHuman,
        botConfig.kind,
        botConfig.iterations,
      ));
      activeBotConfig = botConfig;
      liveGame.start();
    } catch (error) {
      gameError = "The C++ game engine could not start: " + error.message;
    }
    renderInterface();

    if (liveGame.movesEnabled && !gameError &&
        liveState().toMove !== humanPlayer) {
      scheduleBotTurn();
    }
  }

  function changeSettings() {
    if (!liveSnapshot || isLiveTerminal() || !liveGame.movesEnabled) {
      return;
    }
    liveGame.changeSettings();
    gameError = "";
    hoveredDestination = null;
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
    if (!liveGame.movesEnabled || !liveSnapshot || isLiveTerminal() ||
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
          control !== replayTwoIterationsInput) {
        control.disabled = isGeneratingReplay;
      }
    }
    replayOneIterationsInput.disabled = isGeneratingReplay ||
      replayOneBotSelect.value !== BotKind.Mcts;
    replayTwoIterationsInput.disabled = isGeneratingReplay ||
      replayTwoBotSelect.value !== BotKind.Mcts;
    generateReplayButton.textContent = isGeneratingReplay
      ? "Generating…"
      : "Generate";
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
      );
      playerTwo = readBotConfig(
        replayTwoBotSelect,
        replayTwoSeedInput,
        replayTwoIterationsInput,
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
    hoveredDestination = null;
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
    activeBotConfiguration.textContent = configuration.kind + " · seed " +
      configuration.seed + (usesMcts
        ? " · " + formatCount(configuration.iterations) +
          " new simulations per bot move"
        : "") + " · You: " + playerName(liveSnapshot.humanPlayer);

    const search = diagnostics.lastBotSearch;
    latestSearch.hidden = !search;
    searchPending.hidden = !usesMcts || Boolean(search);
    if (!search) {
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
    searchDetails.textContent = "Move " + formatCount(search.ply) + " · " +
      formatCount(search.simulatedPlies) + " rollout plies · " +
      formatCount(search.provenNodes) + " proven nodes · root value " +
      Number(search.rootValue).toFixed(4);
  }

  function updateText() {
    if (mode === MODE_PLAY) {
      updatePlayText();
    } else {
      updateReplayText();
    }
  }

  function updatePlayText() {
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
    syncReplayGenerationControls();
    updateLiveDiagnostics();
  }

  function syncLiveGameControls() {
    const state = support.liveGameControlState({
      hasGame: Boolean(liveSnapshot),
      movesEnabled: liveGame.movesEnabled,
      isTerminal: isLiveTerminal(),
      usesMcts: botSelect.value === BotKind.Mcts,
    });
    support.syncConfigurationControls({
      botSelect: botSelect,
      sideSelect: sideSelect,
      seedInput: seedInput,
      budgetInput: botIterationsInput,
      startButton: newGameButton,
      changeSettingsButton: changeSettingsButton,
      exportButton: exportGameButton,
    }, state);
    gameSetup.classList.toggle("is-configuration-locked", state.locked);
  }

  function syncBotSetupFields() {
    const playUsesMcts = botSelect.value === BotKind.Mcts;
    botIterationsField.hidden = !playUsesMcts;
    botIterationsInput.disabled = !playUsesMcts;
    playSetupEyebrow.textContent = "Play vs " + botDisplayName(botSelect.value);

    const playerOneUsesMcts = replayOneBotSelect.value === BotKind.Mcts;
    const playerTwoUsesMcts = replayTwoBotSelect.value === BotKind.Mcts;
    replayOneIterationsField.hidden = !playerOneUsesMcts;
    replayTwoIterationsField.hidden = !playerTwoUsesMcts;
    replayOneIterationsInput.disabled = isGeneratingReplay || !playerOneUsesMcts;
    replayTwoIterationsInput.disabled = isGeneratingReplay || !playerTwoUsesMcts;
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
      } else if (isLiveTerminal()) {
        message.textContent = "The game is finished.";
      } else if (!liveGame.movesEnabled) {
        message.textContent = "Settings are unlocked. Start a new game to continue.";
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
        destinationLabel(liveState().ball, move.to, extraTurn),
      );
      button.dataset.point = pointId(move.to);
      button.addEventListener("mouseenter", function () {
        setHoveredDestination(move.to);
      });
      button.addEventListener("mouseleave", clearHoveredDestination);
      button.addEventListener("focus", function () {
        setHoveredDestination(move.to);
      });
      button.addEventListener("blur", clearHoveredDestination);
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

  function makeMapper() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const spanX = replay.rules.width;
    const spanY = replay.rules.height + 2;
    const shortSide = Math.min(width, height);
    const safeMargin = Math.max(24, shortSide * 0.045);
    const coordinateFontSize = Math.max(12, Math.min(14, shortSide * 0.024));
    const directionFontSize = Math.max(10, Math.min(12, shortSide * 0.02));
    const axisGap = Math.max(24, coordinateFontSize + 10);
    const directionOffset = Math.max(16, directionFontSize / 2 + 8);
    const coordinateOffset = directionOffset + directionFontSize / 2 + 8 +
      coordinateFontSize / 2;
    const leftGutter = safeMargin + axisGap + coordinateFontSize;
    const rightGutter = safeMargin + 8;
    const topGutter = safeMargin + directionOffset + directionFontSize / 2;
    const bottomGutter = safeMargin + coordinateOffset + coordinateFontSize / 2;
    const availableWidth = Math.max(1, width - leftGutter - rightGutter);
    const availableHeight = Math.max(1, height - topGutter - bottomGutter);
    const cell = Math.min(availableWidth / spanX, availableHeight / spanY);
    const boardWidth = cell * spanX;
    const boardHeight = cell * spanY;
    const originX = leftGutter + (availableWidth - boardWidth) / 2;
    const originY = topGutter + (availableHeight - boardHeight) / 2;

    return {
      axisGap: axisGap,
      cell: cell,
      coordinateFontSize: coordinateFontSize,
      coordinateOffset: coordinateOffset,
      directionFontSize: directionFontSize,
      directionOffset: directionOffset,
      point: function (point) {
        return {
          x: originX + point.x * cell,
          y: originY + point.y * cell,
        };
      },
    };
  }

  function fieldBottomY() {
    return replay.rules.height + 1;
  }

  function southGoalY() {
    return replay.rules.height + 2;
  }

  function drawBoard() {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * ratio) ||
        canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#386f3d";
    ctx.fillRect(0, 0, width, height);

    const mapper = makeMapper();
    drawField(mapper);
    drawWalls(mapper);
    drawPath(mapper);
    drawPoints(mapper);
    drawLegalMoveGuides(mapper);
    drawBall(mapper);
    drawCoordinates(mapper);
    drawDirectionLabels(mapper);
  }

  function drawField(mapper) {
    const topLeft = mapper.point({ x: 0, y: 1 });
    const bottomRight = mapper.point({ x: replay.rules.width, y: fieldBottomY() });
    const pad = mapper.cell * 0.34;

    ctx.fillStyle = "#4f8d4d";
    roundRect(
      topLeft.x - pad,
      topLeft.y - pad,
      bottomRight.x - topLeft.x + pad * 2,
      bottomRight.y - topLeft.y + pad * 2,
      8,
    );
    ctx.fill();

    ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= replay.rules.width; x += 1) {
      drawLine(
        mapper.point({ x: x, y: 1 }),
        mapper.point({ x: x, y: fieldBottomY() }),
      );
    }
    for (let y = 1; y <= fieldBottomY(); y += 1) {
      drawLine(
        mapper.point({ x: 0, y: y }),
        mapper.point({ x: replay.rules.width, y: y }),
      );
    }
  }

  function drawWalls(mapper) {
    const left = Math.floor(replay.rules.width / 2) - 1;
    const right = Math.floor(replay.rules.width / 2) + 1;
    ctx.strokeStyle = "#f1ecd6";
    ctx.lineWidth = Math.max(5, mapper.cell * 0.08);
    ctx.lineCap = "round";

    for (let x = 0; x < replay.rules.width; x += 1) {
      if (!(x >= left && x < right)) {
        drawSegment(mapper, { x: x, y: 1 }, { x: x + 1, y: 1 });
        drawSegment(
          mapper,
          { x: x, y: fieldBottomY() },
          { x: x + 1, y: fieldBottomY() },
        );
      }
    }

    for (let y = 1; y < fieldBottomY(); y += 1) {
      drawSegment(mapper, { x: 0, y: y }, { x: 0, y: y + 1 });
      drawSegment(
        mapper,
        { x: replay.rules.width, y: y },
        { x: replay.rules.width, y: y + 1 },
      );
    }

    for (let x = left; x < right; x += 1) {
      drawSegment(mapper, { x: x, y: 0 }, { x: x + 1, y: 0 });
      drawSegment(
        mapper,
        { x: x, y: southGoalY() },
        { x: x + 1, y: southGoalY() },
      );
    }
    drawSegment(mapper, { x: left, y: 0 }, { x: left, y: 1 });
    drawSegment(mapper, { x: right, y: 0 }, { x: right, y: 1 });
    drawSegment(
      mapper,
      { x: left, y: fieldBottomY() },
      { x: left, y: southGoalY() },
    );
    drawSegment(
      mapper,
      { x: right, y: fieldBottomY() },
      { x: right, y: southGoalY() },
    );
  }

  function drawPoints(mapper) {
    const radius = Math.max(3, mapper.cell * 0.055);
    const left = Math.floor(replay.rules.width / 2) - 1;
    const right = Math.floor(replay.rules.width / 2) + 1;
    for (let y = 1; y <= fieldBottomY(); y += 1) {
      for (let x = 0; x <= replay.rules.width; x += 1) {
        const onGoalLine = y === 1 || y === fieldBottomY();
        const insideGoalOpening = x > left && x < right;
        const boundary = x === 0 || x === replay.rules.width ||
          (onGoalLine && !insideGoalOpening);
        drawCircle(
          mapper.point({ x: x, y: y }),
          radius,
          boundary ? "#fff8dc" : "#e4ead3",
          "#355438",
          1,
        );
      }
    }

    for (let x = left; x <= right; x += 1) {
      drawGoalNode(mapper.point({ x: x, y: 0 }), "north", mapper.cell);
      drawGoalNode(
        mapper.point({ x: x, y: southGoalY() }),
        "south",
        mapper.cell,
      );
    }
  }

  function drawGoalNode(point, direction, cell) {
    const size = Math.max(8, cell * 0.16);
    ctx.beginPath();
    if (direction === "north") {
      ctx.moveTo(point.x, point.y - size);
      ctx.lineTo(point.x - size * 0.75, point.y + size * 0.55);
      ctx.lineTo(point.x + size * 0.75, point.y + size * 0.55);
    } else {
      ctx.moveTo(point.x, point.y + size);
      ctx.lineTo(point.x - size * 0.75, point.y - size * 0.55);
      ctx.lineTo(point.x + size * 0.75, point.y - size * 0.55);
    }
    ctx.closePath();
    ctx.fillStyle = "#f7f0d4";
    ctx.strokeStyle = "#2c4b32";
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
  }

  function drawPath(mapper) {
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const highlightedFrom = currentMoveStartIndex();
    for (let i = 0; i < currentPly; i += 1) {
      const move = replay.moves[i];
      const from = mapper.point(move.from);
      const to = mapper.point(move.to);
      const isCurrentMove = i >= highlightedFrom;

      ctx.strokeStyle = isCurrentMove
        ? "#ffd166"
        : (move.player === Player.Two ? "#153f66" : "#782525");
      ctx.lineWidth = isCurrentMove
        ? Math.max(6, mapper.cell * 0.095)
        : Math.max(4, mapper.cell * 0.07);
      drawLine(from, to);

      ctx.strokeStyle = move.player === Player.Two ? "#2067a6" : "#c93b3b";
      ctx.lineWidth = Math.max(2, mapper.cell * 0.035);
      drawLine(from, to);
    }
  }

  function drawLegalMoveGuides(mapper) {
    if (!canHumanMove()) {
      return;
    }

    const from = mapper.point(liveState().ball);
    ctx.save();
    ctx.lineCap = "round";
    ctx.setLineDash([5, 5]);
    for (const move of liveLegalMoves()) {
      const hovered = hoveredDestination &&
        samePoint(hoveredDestination, move.to);
      ctx.strokeStyle = hovered
        ? "rgba(255, 243, 201, 0.98)"
        : "rgba(255, 243, 201, 0.55)";
      ctx.lineWidth = hovered
        ? Math.max(4, mapper.cell * 0.07)
        : Math.max(2, mapper.cell * 0.04);
      drawLine(from, mapper.point(move.to));
    }
    ctx.restore();
  }

  function drawBall(mapper) {
    const point = mapper.point(currentBall());
    const radius = Math.max(8, mapper.cell * 0.14);
    drawCircle(point, radius, "#ffd166", "#1f2328", 2);
    drawCircle(point, radius * 0.42, "#20252b", "transparent", 0);
  }

  function drawCoordinates(mapper) {
    const xAxisY = mapper.point({ x: 0, y: southGoalY() }).y +
      mapper.coordinateOffset;
    const yAxisX = mapper.point({ x: 0, y: 0 }).x - mapper.axisGap;

    ctx.fillStyle = "rgba(255, 250, 230, 0.8)";
    ctx.font = "600 " + mapper.coordinateFontSize +
      "px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let x = 0; x <= replay.rules.width; x += 1) {
      const point = mapper.point({ x: x, y: southGoalY() });
      ctx.fillText(String(x), point.x, xAxisY);
    }

    ctx.textAlign = "right";
    for (let y = 0; y <= southGoalY(); y += 1) {
      const point = mapper.point({ x: 0, y: y });
      ctx.fillText(String(y), yAxisX, point.y);
    }
  }

  function drawDirectionLabels(mapper) {
    const center = Math.floor(replay.rules.width / 2);
    const top = mapper.point({ x: center, y: 0 });
    const bottom = mapper.point({ x: center, y: southGoalY() });

    ctx.fillStyle = "rgba(255, 248, 220, 0.88)";
    ctx.font = "750 " + mapper.directionFontSize +
      "px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("P1 ATTACKS ↑", top.x, top.y - mapper.directionOffset);
    ctx.fillText("P2 ATTACKS ↓", bottom.x, bottom.y + mapper.directionOffset);
  }

  function renderBoardTargets() {
    const focusedPoint = document.activeElement instanceof HTMLElement &&
      boardTargets.contains(document.activeElement)
      ? document.activeElement.dataset.point
      : null;
    boardTargets.replaceChildren();
    if (!canHumanMove()) {
      return;
    }

    const mapper = makeMapper();
    const targetSize = Math.max(32, Math.min(44, mapper.cell * 0.78));
    const moves = liveLegalMoves().slice().sort(function (left, right) {
      return left.to.y - right.to.y || left.to.x - right.to.x;
    });

    for (const move of moves) {
      const point = mapper.point(move.to);
      const extraTurn = move.extraTurn;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "board-target" + (extraTurn ? " is-extra-turn" : "");
      button.style.left = point.x + "px";
      button.style.top = point.y + "px";
      button.style.setProperty("--target-size", targetSize + "px");
      button.setAttribute(
        "aria-label",
        destinationLabel(liveState().ball, move.to, extraTurn),
      );
      button.dataset.point = pointId(move.to);
      button.addEventListener("mouseenter", function () {
        setHoveredDestination(move.to);
      });
      button.addEventListener("mouseleave", clearHoveredDestination);
      button.addEventListener("focus", function () {
        setHoveredDestination(move.to);
      });
      button.addEventListener("blur", clearHoveredDestination);
      button.addEventListener("click", function (event) {
        playHumanMove(move, event.detail === 0);
      });
      boardTargets.append(button);
    }

    if (focusedPoint) {
      const replacement = Array.from(boardTargets.children).find(function (element) {
        return element.dataset.point === focusedPoint;
      });
      replacement?.focus({ preventScroll: true });
    }
  }

  function destinationLabel(from, to, extraTurn) {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const vertical = dy < 0 ? "north" : (dy > 0 ? "south" : "");
    const horizontal = dx < 0 ? "west" : (dx > 0 ? "east" : "");
    const direction = vertical + horizontal || "destination";
    return "Move " + direction + " to row " + to.y + ", column " + to.x +
      (extraTurn ? "; grants another move" : "");
  }

  function pointId(point) {
    return point.x + "," + point.y;
  }

  function setHoveredDestination(point) {
    if (hoveredDestination && samePoint(hoveredDestination, point)) {
      return;
    }
    hoveredDestination = { x: point.x, y: point.y };
    syncHoveredControls();
    drawBoard();
  }

  function clearHoveredDestination() {
    if (!hoveredDestination) {
      return;
    }
    hoveredDestination = null;
    syncHoveredControls();
    drawBoard();
  }

  function syncHoveredControls() {
    const hoveredId = hoveredDestination ? pointId(hoveredDestination) : "";
    for (const element of document.querySelectorAll("[data-point]")) {
      element.classList.toggle("is-hovered", element.dataset.point === hoveredId);
    }
  }

  function nearestLegalMove(event) {
    if (!canHumanMove()) {
      return null;
    }
    const rect = canvas.getBoundingClientRect();
    const pointer = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
    const mapper = makeMapper();
    let nearest = null;
    let nearestDistance = Infinity;

    for (const move of liveLegalMoves()) {
      const target = mapper.point(move.to);
      const distance = Math.hypot(pointer.x - target.x, pointer.y - target.y);
      if (distance < nearestDistance) {
        nearest = move;
        nearestDistance = distance;
      }
    }

    return nearestDistance <= Math.max(18, mapper.cell * 0.38) ? nearest : null;
  }

  function drawSegment(mapper, from, to) {
    drawLine(mapper.point(from), mapper.point(to));
  }

  function drawLine(from, to) {
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }

  function drawCircle(point, radius, fill, stroke, lineWidth) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (lineWidth > 0) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  }

  function roundRect(x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
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
    drawBoard();
    renderBoardTargets();
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
  gameSetup.addEventListener("submit", startNewGame);
  changeSettingsButton.addEventListener("click", changeSettings);
  exportGameButton.addEventListener("click", exportHumanMatch);
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
    const move = nearestLegalMove(event);
    if (move) {
      playHumanMove(move, false);
    }
  });
  canvas.addEventListener("pointermove", function (event) {
    const move = nearestLegalMove(event);
    if (move) {
      setHoveredDestination(move.to);
    } else {
      clearHoveredDestination();
    }
  });
  canvas.addEventListener("pointerleave", clearHoveredDestination);

  window.addEventListener("resize", function () {
    drawBoard();
    renderBoardTargets();
  });
  if ("ResizeObserver" in window) {
    new ResizeObserver(function () {
      drawBoard();
      renderBoardTargets();
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

  startNewGame();
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
