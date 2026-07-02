(() => {
  "use strict";

  const canvas = document.querySelector("#boardCanvas");
  const ctx = canvas.getContext("2d");
  const replayTitle = document.querySelector("#replayTitle");
  const plyLabel = document.querySelector("#plyLabel");
  const statusLabel = document.querySelector("#statusLabel");
  const moveLabel = document.querySelector("#moveLabel");
  const startButton = document.querySelector("#startButton");
  const prevButton = document.querySelector("#prevButton");
  const playButton = document.querySelector("#playButton");
  const nextButton = document.querySelector("#nextButton");
  const endButton = document.querySelector("#endButton");
  const plyRange = document.querySelector("#plyRange");
  const speedRange = document.querySelector("#speedRange");
  const speedLabel = document.querySelector("#speedLabel");
  const moveList = document.querySelector("#moveList");
  const fileInput = document.querySelector("#fileInput");

  let replay = normalizeReplay(readDefaultReplay());
  let currentPly = 0;
  let isPlaying = false;
  let speed = 1;
  let lastStepTime = 0;

  function readDefaultReplay() {
    const raw = document.querySelector("#default-replay").textContent;
    return JSON.parse(raw);
  }

  function normalizeReplay(raw) {
    const rules = raw.rules ?? {};
    const width = Number.isFinite(rules.width) ? rules.width : 8;
    const height = Number.isFinite(rules.height) ? rules.height : 10;
    const start = normalizePoint(raw.start, { x: Math.floor(width / 2), y: Math.floor(height / 2) });
    const moves = Array.isArray(raw.moves)
      ? raw.moves.map((move, index) => normalizeMove(move, index, start))
      : [];

    return {
      schema: raw.schema ?? "papersoccer.replay.v1",
      rules: { width, height },
      players: raw.players ?? {},
      start,
      status: raw.status ?? "inProgress",
      winner: raw.winner ?? null,
      truncated: Boolean(raw.truncated),
      moves,
    };
  }

  function normalizeMove(move, index, start) {
    const fallbackFrom = index === 0 ? start : normalizePoint({}, start);
    return {
      ply: Number.isFinite(move.ply) ? move.ply : index + 1,
      player: move.player === "two" ? "two" : "one",
      from: normalizePoint(move.from, fallbackFrom),
      to: normalizePoint(move.to, fallbackFrom),
      extraTurn: Boolean(move.extraTurn),
      statusAfter: move.statusAfter ?? "inProgress",
    };
  }

  function normalizePoint(point, fallback) {
    const x = Number.isFinite(point?.x) ? point.x : fallback.x;
    const y = Number.isFinite(point?.y) ? point.y : fallback.y;
    return { x, y };
  }

  function playerName(player) {
    return player === "two" ? "Player 2" : "Player 1";
  }

  function statusName(status) {
    if (status === "wonByOne") {
      return "Won by Player 1";
    }
    if (status === "wonByTwo") {
      return "Won by Player 2";
    }
    if (status === "inProgress") {
      return "In progress";
    }
    return String(status);
  }

  function formatPoint(point) {
    return `(${point.y}, ${point.x})`;
  }

  function describePlayers() {
    const one = replay.players.one;
    const two = replay.players.two;
    return `${describePlayer(one, "Player 1")} vs ${describePlayer(two, "Player 2")}`;
  }

  function describePlayer(player, fallback) {
    if (!player) {
      return fallback;
    }
    const seed = Number.isFinite(player.seed) ? ` ${player.seed}` : "";
    return `${player.kind ?? fallback}${seed}`;
  }

  function clampPly(ply) {
    return Math.max(0, Math.min(replay.moves.length, ply));
  }

  function setPly(ply) {
    currentPly = clampPly(ply);
    plyRange.value = String(currentPly);
    if (currentPly >= replay.moves.length) {
      stopPlayback();
    }
    updateText();
    updateMoveList();
    drawBoard();
  }

  function currentBall() {
    if (currentPly === 0) {
      return replay.start;
    }
    return replay.moves[currentPly - 1].to;
  }

  function currentStatus() {
    if (currentPly === replay.moves.length) {
      return replay.truncated ? "Truncated" : statusName(replay.status);
    }
    if (currentPly === 0) {
      return "Ready";
    }
    return statusName(replay.moves[currentPly - 1].statusAfter);
  }

  function updateText() {
    replayTitle.textContent = describePlayers();
    plyLabel.textContent = `${currentPly} / ${replay.moves.length}`;
    statusLabel.textContent = currentStatus();

    if (currentPly === 0) {
      moveLabel.textContent = `Start ${formatPoint(replay.start)}`;
    } else {
      const move = replay.moves[currentPly - 1];
      const suffix = move.extraTurn ? " extra" : "";
      moveLabel.textContent = `${playerName(move.player)} ${formatPoint(move.from)} -> ${formatPoint(move.to)}${suffix}`;
    }
  }

  function updateMoveList() {
    moveList.replaceChildren();
    for (const move of replay.moves) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "move-row";
      if (move.ply === currentPly) {
        row.classList.add("is-active");
      }
      row.addEventListener("click", () => setPly(move.ply));

      const ply = document.createElement("span");
      ply.textContent = `#${move.ply}`;

      const detail = document.createElement("span");
      const player = document.createElement("span");
      player.className = move.player === "two" ? "player-two" : "player-one";
      player.textContent = playerName(move.player);
      detail.append(player, ` ${formatPoint(move.from)} -> ${formatPoint(move.to)}`);

      row.append(ply, detail);
      if (move.extraTurn) {
        const badge = document.createElement("span");
        badge.className = "extra-turn";
        badge.textContent = "extra";
        row.append(badge);
      } else {
        row.append(document.createElement("span"));
      }
      moveList.append(row);
    }

    const active = moveList.querySelector(".is-active");
    active?.scrollIntoView({ block: "nearest" });
  }

  function makeMapper() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const spanX = replay.rules.width;
    const spanY = replay.rules.height + 2;
    const gutter = Math.max(34, Math.min(width, height) * 0.07);
    const cell = Math.min((width - gutter * 2) / spanX, (height - gutter * 2) / spanY);
    const boardWidth = cell * spanX;
    const boardHeight = cell * spanY;
    const originX = (width - boardWidth) / 2;
    const originY = (height - boardHeight) / 2;

    return {
      cell,
      point(point) {
        return {
          x: originX + point.x * cell,
          y: originY + (point.y + 1) * cell,
        };
      },
    };
  }

  function drawBoard() {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
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
    drawBall(mapper);
    drawCoordinates(mapper);
  }

  function drawField(mapper) {
    const topLeft = mapper.point({ x: 0, y: 0 });
    const bottomRight = mapper.point({ x: replay.rules.width, y: replay.rules.height });
    const pad = mapper.cell * 0.34;

    ctx.fillStyle = "#4f8d4d";
    roundRect(topLeft.x - pad, topLeft.y - pad, bottomRight.x - topLeft.x + pad * 2,
      bottomRight.y - topLeft.y + pad * 2, 8);
    ctx.fill();

    ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= replay.rules.width; x += 1) {
      const a = mapper.point({ x, y: 0 });
      const b = mapper.point({ x, y: replay.rules.height });
      drawLine(a, b);
    }
    for (let y = 0; y <= replay.rules.height; y += 1) {
      const a = mapper.point({ x: 0, y });
      const b = mapper.point({ x: replay.rules.width, y });
      drawLine(a, b);
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
        drawSegment(mapper, { x, y: 0 }, { x: x + 1, y: 0 });
        drawSegment(mapper, { x, y: replay.rules.height }, { x: x + 1, y: replay.rules.height });
      }
    }

    for (let y = 0; y < replay.rules.height; y += 1) {
      drawSegment(mapper, { x: 0, y }, { x: 0, y: y + 1 });
      drawSegment(mapper, { x: replay.rules.width, y }, { x: replay.rules.width, y: y + 1 });
    }

    for (let x = left; x < right; x += 1) {
      drawSegment(mapper, { x, y: -1 }, { x: x + 1, y: -1 });
      drawSegment(mapper, { x, y: replay.rules.height + 1 }, { x: x + 1, y: replay.rules.height + 1 });
    }
    drawSegment(mapper, { x: left, y: -1 }, { x: left, y: 0 });
    drawSegment(mapper, { x: right, y: -1 }, { x: right, y: 0 });
    drawSegment(mapper, { x: left, y: replay.rules.height }, { x: left, y: replay.rules.height + 1 });
    drawSegment(mapper, { x: right, y: replay.rules.height }, { x: right, y: replay.rules.height + 1 });
  }

  function drawPoints(mapper) {
    const radius = Math.max(3, mapper.cell * 0.055);
    for (let y = 0; y <= replay.rules.height; y += 1) {
      for (let x = 0; x <= replay.rules.width; x += 1) {
        const boundary = x === 0 || x === replay.rules.width || y === 0 || y === replay.rules.height;
        drawCircle(mapper.point({ x, y }), radius, boundary ? "#fff8dc" : "#e4ead3", "#355438", 1);
      }
    }

    const left = Math.floor(replay.rules.width / 2) - 1;
    const right = Math.floor(replay.rules.width / 2) + 1;
    for (let x = left; x <= right; x += 1) {
      drawGoalNode(mapper.point({ x, y: -1 }), "north", mapper.cell);
      drawGoalNode(mapper.point({ x, y: replay.rules.height + 1 }), "south", mapper.cell);
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
    for (let i = 0; i < currentPly; i += 1) {
      const move = replay.moves[i];
      const from = mapper.point(move.from);
      const to = mapper.point(move.to);
      const isCurrent = i === currentPly - 1;

      ctx.strokeStyle = isCurrent ? "#ffd166" : (move.player === "two" ? "#153f66" : "#782525");
      ctx.lineWidth = isCurrent ? Math.max(6, mapper.cell * 0.095) : Math.max(4, mapper.cell * 0.07);
      drawLine(from, to);

      ctx.strokeStyle = move.player === "two" ? "#2067a6" : "#c93b3b";
      ctx.lineWidth = Math.max(2, mapper.cell * 0.035);
      drawLine(from, to);
    }
  }

  function drawBall(mapper) {
    const point = mapper.point(currentBall());
    const radius = Math.max(8, mapper.cell * 0.14);
    drawCircle(point, radius, "#ffd166", "#1f2328", 2);
    drawCircle(point, radius * 0.42, "#20252b", "transparent", 0);
  }

  function drawCoordinates(mapper) {
    ctx.fillStyle = "rgba(255, 250, 230, 0.75)";
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let x = 0; x <= replay.rules.width; x += 1) {
      const point = mapper.point({ x, y: replay.rules.height });
      ctx.fillText(String(x), point.x, point.y + mapper.cell * 0.48);
    }

    ctx.textAlign = "right";
    for (let y = -1; y <= replay.rules.height + 1; y += 1) {
      const point = mapper.point({ x: 0, y });
      ctx.fillText(String(y), point.x - mapper.cell * 0.34, point.y);
    }
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
    if (currentPly >= replay.moves.length) {
      setPly(0);
    }
    isPlaying = true;
    playButton.textContent = "Pause";
    lastStepTime = 0;
  }

  function stopPlayback() {
    isPlaying = false;
    playButton.textContent = "Play";
  }

  function tick(timestamp) {
    if (isPlaying) {
      const delay = 850 / speed;
      if (lastStepTime === 0) {
        lastStepTime = timestamp;
      }
      if (timestamp - lastStepTime >= delay) {
        setPly(currentPly + 1);
        lastStepTime = timestamp;
      }
    }
    window.requestAnimationFrame(tick);
  }

  function loadReplay(raw) {
    replay = normalizeReplay(raw);
    currentPly = 0;
    stopPlayback();
    plyRange.max = String(replay.moves.length);
    setPly(0);
  }

  startButton.addEventListener("click", () => setPly(0));
  prevButton.addEventListener("click", () => setPly(currentPly - 1));
  playButton.addEventListener("click", () => {
    if (isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });
  nextButton.addEventListener("click", () => setPly(currentPly + 1));
  endButton.addEventListener("click", () => setPly(replay.moves.length));
  plyRange.addEventListener("input", () => setPly(Number(plyRange.value)));
  speedRange.addEventListener("input", () => {
    speed = Number(speedRange.value);
    speedLabel.textContent = `${speed}x`;
  });
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) {
      return;
    }
    try {
      const raw = JSON.parse(await file.text());
      loadReplay(raw);
    } catch (error) {
      window.alert(`Could not load replay: ${error.message}`);
    } finally {
      fileInput.value = "";
    }
  });

  window.addEventListener("resize", drawBoard);
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) {
      return;
    }
    if (event.key === "ArrowLeft") {
      setPly(currentPly - 1);
    }
    if (event.key === "ArrowRight") {
      setPly(currentPly + 1);
    }
    if (event.key === " ") {
      event.preventDefault();
      isPlaying ? stopPlayback() : startPlayback();
    }
  });

  loadReplay(replay);
  window.requestAnimationFrame(tick);
})();
