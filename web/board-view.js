(function (global) {
  "use strict";

  function createBoardView(options) {
    const {
      boardTargets,
      canHumanMove,
      canvas,
      currentBall,
      currentMoveStartIndex,
      getCurrentPly,
      getReplay,
      liveLegalMoves,
      liveState,
      onHumanMove,
      playerTwo,
      samePoint,
    } = options;
    const ctx = canvas.getContext("2d");
    let hoveredDestination = null;

    function makeMapper() {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const spanX = getReplay().rules.width;
      const spanY = getReplay().rules.height + 2;
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
      return getReplay().rules.height + 1;
    }

    function southGoalY() {
      return getReplay().rules.height + 2;
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
      const bottomRight = mapper.point({ x: getReplay().rules.width, y: fieldBottomY() });
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
      for (let x = 0; x <= getReplay().rules.width; x += 1) {
        drawLine(
          mapper.point({ x: x, y: 1 }),
          mapper.point({ x: x, y: fieldBottomY() }),
        );
      }
      for (let y = 1; y <= fieldBottomY(); y += 1) {
        drawLine(
          mapper.point({ x: 0, y: y }),
          mapper.point({ x: getReplay().rules.width, y: y }),
        );
      }
    }

    function drawWalls(mapper) {
      const left = Math.floor(getReplay().rules.width / 2) - 1;
      const right = Math.floor(getReplay().rules.width / 2) + 1;
      ctx.strokeStyle = "#f1ecd6";
      ctx.lineWidth = Math.max(5, mapper.cell * 0.08);
      ctx.lineCap = "round";

      for (let x = 0; x < getReplay().rules.width; x += 1) {
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
          { x: getReplay().rules.width, y: y },
          { x: getReplay().rules.width, y: y + 1 },
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
      const left = Math.floor(getReplay().rules.width / 2) - 1;
      const right = Math.floor(getReplay().rules.width / 2) + 1;
      for (let y = 1; y <= fieldBottomY(); y += 1) {
        for (let x = 0; x <= getReplay().rules.width; x += 1) {
          const onGoalLine = y === 1 || y === fieldBottomY();
          const insideGoalOpening = x > left && x < right;
          const boundary = x === 0 || x === getReplay().rules.width ||
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
      for (let i = 0; i < getCurrentPly(); i += 1) {
        const move = getReplay().moves[i];
        const from = mapper.point(move.from);
        const to = mapper.point(move.to);
        const isCurrentMove = i >= highlightedFrom;

        ctx.strokeStyle = isCurrentMove
          ? "#ffd166"
          : (move.player === playerTwo ? "#153f66" : "#782525");
        ctx.lineWidth = isCurrentMove
          ? Math.max(6, mapper.cell * 0.095)
          : Math.max(4, mapper.cell * 0.07);
        drawLine(from, to);

        ctx.strokeStyle = move.player === playerTwo ? "#2067a6" : "#c93b3b";
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
      ctx.setLineDash([3, 6]);
      for (const move of liveLegalMoves()) {
        const hovered = hoveredDestination &&
          samePoint(hoveredDestination, move.to);
        ctx.strokeStyle = hovered
          ? "rgba(255, 243, 201, 0.9)"
          : "rgba(255, 243, 201, 0.24)";
        ctx.lineWidth = hovered
          ? Math.max(2.5, mapper.cell * 0.04)
          : Math.max(1, mapper.cell * 0.018);
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

      for (let x = 0; x <= getReplay().rules.width; x += 1) {
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
      const center = Math.floor(getReplay().rules.width / 2);
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
      const markerSize = Math.max(16, Math.min(20, mapper.cell * 0.3));
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
        button.style.setProperty("--marker-size", markerSize + "px");
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
          onHumanMove(move, event.detail === 0);
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

    function resetHover() {
      hoveredDestination = null;
      syncHoveredControls();
    }

    return {
      clearHoveredDestination: clearHoveredDestination,
      destinationLabel: destinationLabel,
      draw: drawBoard,
      nearestLegalMove: nearestLegalMove,
      pointId: pointId,
      renderTargets: renderBoardTargets,
      resetHover: resetHover,
      setHoveredDestination: setHoveredDestination,
    };
  }

  global.PaperSoccerBoard = Object.freeze({
    createBoardView: createBoardView,
  });
}(globalThis));
