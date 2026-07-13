(function (root) {
  "use strict";

  const Player = Object.freeze({
    One: "one",
    Two: "two",
  });

  const Status = Object.freeze({
    InProgress: "inProgress",
    WonByOne: "wonByOne",
    WonByTwo: "wonByTwo",
  });

  const DEFAULT_CONFIG = Object.freeze({ width: 8, height: 10 });
  const DEFAULT_RANDOM_BOT_SEED = 0xc0ffee1234n;

  const UINT64_BITS = 64;
  const SPLITMIX_INCREMENT = 0x9e3779b97f4a7c15n;
  const SPLITMIX_MULTIPLIER_ONE = 0xbf58476d1ce4e5b9n;
  const SPLITMIX_MULTIPLIER_TWO = 0x94d049bb133111ebn;

  function opponent(player) {
    return player === Player.One ? Player.Two : Player.One;
  }

  function clonePoint(point) {
    return { x: point.x, y: point.y };
  }

  function samePoint(left, right) {
    return left.x === right.x && left.y === right.y;
  }

  function pointComesBefore(left, right) {
    return left.y < right.y || (left.y === right.y && left.x < right.x);
  }

  function pointKey(point) {
    return `${point.x},${point.y}`;
  }

  function makeSegment(first, second) {
    if (pointComesBefore(second, first)) {
      return { a: clonePoint(second), b: clonePoint(first) };
    }
    return { a: clonePoint(first), b: clonePoint(second) };
  }

  function segmentKey(first, second) {
    const segment = makeSegment(first, second);
    return `${pointKey(segment.a)}>${pointKey(segment.b)}`;
  }

  function fieldBottomY(config) {
    return config.height + 1;
  }

  function southGoalY(config) {
    return config.height + 2;
  }

  function centerX(config) {
    return Math.trunc(config.width / 2);
  }

  function mouthLeftX(config) {
    return centerX(config) - 1;
  }

  function mouthRightX(config) {
    return centerX(config) + 1;
  }

  function isMouthX(config, x) {
    return x >= mouthLeftX(config) && x <= mouthRightX(config);
  }

  function isNorthGoal(config, point) {
    return isMouthX(config, point.x) && point.y === 0;
  }

  function isSouthGoal(config, point) {
    return isMouthX(config, point.x) && point.y === southGoalY(config);
  }

  function isGoalMouthPoint(config, point) {
    return isMouthX(config, point.x) &&
      (point.y === 1 || point.y === fieldBottomY(config));
  }

  function isRegularPoint(config, point) {
    return point.x >= 0 && point.x <= config.width &&
      point.y >= 1 && point.y <= fieldBottomY(config);
  }

  function isGoalPoint(config, point) {
    return isNorthGoal(config, point) || isSouthGoal(config, point);
  }

  function isBoundaryPoint(config, point) {
    if (!isRegularPoint(config, point)) {
      return false;
    }
    return point.x === 0 || point.x === config.width ||
      point.y === 1 || point.y === fieldBottomY(config);
  }

  function isNeighbor(from, to) {
    const dx = Math.abs(from.x - to.x);
    const dy = Math.abs(from.y - to.y);
    return dx <= 1 && dy <= 1 && dx + dy > 0;
  }

  function isNorthGoalPostSegment(config, segment) {
    const touchesNorthGoal = isNorthGoal(config, segment.a) ||
      isNorthGoal(config, segment.b);
    if (!touchesNorthGoal) {
      return false;
    }

    return segment.a.x === segment.b.x &&
      (segment.a.x === mouthLeftX(config) || segment.a.x === mouthRightX(config)) &&
      ((segment.a.y === 0 && segment.b.y === 1) ||
        (segment.a.y === 1 && segment.b.y === 0));
  }

  function isSouthGoalPostSegment(config, segment) {
    const touchesSouthGoal = isSouthGoal(config, segment.a) ||
      isSouthGoal(config, segment.b);
    if (!touchesSouthGoal) {
      return false;
    }

    return segment.a.x === segment.b.x &&
      (segment.a.x === mouthLeftX(config) || segment.a.x === mouthRightX(config)) &&
      ((segment.a.y === fieldBottomY(config) && segment.b.y === southGoalY(config)) ||
        (segment.a.y === southGoalY(config) && segment.b.y === fieldBottomY(config)));
  }

  function isForbiddenBoundarySegment(config, segment) {
    if (isNorthGoalPostSegment(config, segment) ||
        isSouthGoalPostSegment(config, segment)) {
      return true;
    }

    if (!isRegularPoint(config, segment.a) || !isRegularPoint(config, segment.b)) {
      return false;
    }

    if (!(isBoundaryPoint(config, segment.a) && isBoundaryPoint(config, segment.b))) {
      return false;
    }

    const dx = Math.abs(segment.a.x - segment.b.x);
    const dy = Math.abs(segment.a.y - segment.b.y);

    if (segment.a.y === segment.b.y &&
        (segment.a.y === 1 || segment.a.y === fieldBottomY(config)) &&
        dx === 1 && dy === 0) {
      return true;
    }

    return segment.a.x === segment.b.x &&
      (segment.a.x === 0 || segment.a.x === config.width) &&
      dx === 0 && dy === 1;
  }

  function neighbors(config, from, player) {
    const result = [];
    if (!isRegularPoint(config, from)) {
      return result;
    }

    // Keep the C++ iteration order: x delta first, then y delta. RandomBot
    // depends on this order because it selects a move by index.
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        if (dx === 0 && dy === 0) {
          continue;
        }
        const candidate = { x: from.x + dx, y: from.y + dy };
        if (isRegularPoint(config, candidate)) {
          result.push(candidate);
        }
      }
    }

    if (isGoalMouthPoint(config, from)) {
      if (player === Player.One && from.y === 1) {
        for (let goalX = mouthLeftX(config); goalX <= mouthRightX(config); goalX += 1) {
          const goalPoint = { x: goalX, y: 0 };
          if (isNeighbor(from, goalPoint)) {
            result.push(goalPoint);
          }
        }
      }
      if (player === Player.Two && from.y === fieldBottomY(config)) {
        for (let goalX = mouthLeftX(config); goalX <= mouthRightX(config); goalX += 1) {
          const goalPoint = { x: goalX, y: southGoalY(config) };
          if (isNeighbor(from, goalPoint)) {
            result.push(goalPoint);
          }
        }
      }
    }

    return result;
  }

  function makeInitialState(config = DEFAULT_CONFIG) {
    const stateConfig = { width: config.width, height: config.height };
    const ball = {
      x: Math.trunc(stateConfig.width / 2),
      y: Math.trunc(stateConfig.height / 2) + 1,
    };

    return {
      config: stateConfig,
      ball,
      toMove: Player.One,
      status: Status.InProgress,
      path: [clonePoint(ball)],
      usedSegments: new Set(),
      visitCount: new Map([[pointKey(ball), 1]]),
    };
  }

  function cloneState(state) {
    return {
      config: { width: state.config.width, height: state.config.height },
      ball: clonePoint(state.ball),
      toMove: state.toMove,
      status: state.status,
      path: state.path.map(clonePoint),
      usedSegments: new Set(state.usedSegments),
      visitCount: new Map(state.visitCount),
    };
  }

  function legalMoves(state) {
    const result = [];
    if (state.status !== Status.InProgress) {
      return result;
    }

    for (const destination of neighbors(state.config, state.ball, state.toMove)) {
      const segment = makeSegment(state.ball, destination);
      if (state.usedSegments.has(segmentKey(segment.a, segment.b))) {
        continue;
      }
      if (isForbiddenBoundarySegment(state.config, segment)) {
        continue;
      }
      result.push({ to: clonePoint(destination) });
    }

    return result;
  }

  function grantsExtraTurn(before, destination) {
    if (isBoundaryPoint(before.config, destination)) {
      return true;
    }
    return (before.visitCount.get(pointKey(destination)) ?? 0) > 0;
  }

  function applyMove(state, move) {
    if (state.status !== Status.InProgress) {
      throw new Error("cannot apply move to terminal game state");
    }

    const matchingMove = legalMoves(state).find((candidate) => samePoint(candidate.to, move.to));
    if (!matchingMove) {
      throw new Error("illegal move");
    }

    const destination = matchingMove.to;
    const next = cloneState(state);
    next.usedSegments.add(segmentKey(state.ball, destination));
    next.ball = clonePoint(destination);
    next.path.push(clonePoint(destination));
    const destinationKey = pointKey(destination);
    next.visitCount.set(destinationKey, (next.visitCount.get(destinationKey) ?? 0) + 1);

    if (isGoalPoint(next.config, destination)) {
      next.status = state.toMove === Player.One ? Status.WonByOne : Status.WonByTwo;
      return next;
    }

    next.toMove = grantsExtraTurn(state, destination) ? state.toMove : opponent(state.toMove);
    next.status = Status.InProgress;

    if (legalMoves(next).length === 0) {
      next.status = next.toMove === Player.One ? Status.WonByTwo : Status.WonByOne;
    }

    return next;
  }

  function isTerminal(state) {
    return state.status !== Status.InProgress;
  }

  function winner(state) {
    if (state.status === Status.WonByOne) {
      return Player.One;
    }
    if (state.status === Status.WonByTwo) {
      return Player.Two;
    }
    return null;
  }

  function normalizeSeed(seed) {
    if (typeof seed === "number") {
      if (!Number.isSafeInteger(seed)) {
        throw new RangeError("RandomBot number seeds must be safe integers");
      }
      return BigInt.asUintN(UINT64_BITS, BigInt(seed));
    }

    try {
      return BigInt.asUintN(UINT64_BITS, BigInt(seed));
    } catch (_error) {
      throw new TypeError("RandomBot seed must be an integer, bigint, or integer string");
    }
  }

  class RandomBot {
    constructor(seed = DEFAULT_RANDOM_BOT_SEED) {
      this.state = normalizeSeed(seed);
    }

    name() {
      return "RandomBot";
    }

    chooseMove(state) {
      const moves = legalMoves(state);
      if (moves.length === 0) {
        throw new Error("bot cannot choose move without legal moves");
      }

      const index = Number(this.nextRandom() % BigInt(moves.length));
      return moves[index];
    }

    nextRandom() {
      this.state = BigInt.asUintN(UINT64_BITS, this.state + SPLITMIX_INCREMENT);
      let value = this.state;
      value = BigInt.asUintN(
        UINT64_BITS,
        (value ^ (value >> 30n)) * SPLITMIX_MULTIPLIER_ONE,
      );
      value = BigInt.asUintN(
        UINT64_BITS,
        (value ^ (value >> 27n)) * SPLITMIX_MULTIPLIER_TWO,
      );
      return BigInt.asUintN(UINT64_BITS, value ^ (value >> 31n));
    }
  }

  root.PaperSoccer = Object.freeze({
    Player,
    Status,
    DEFAULT_CONFIG,
    DEFAULT_RANDOM_BOT_SEED,
    opponent,
    samePoint,
    pointKey,
    makeSegment,
    segmentKey,
    fieldBottomY,
    southGoalY,
    isRegularPoint,
    isGoalPoint,
    isBoundaryPoint,
    isNeighbor,
    isForbiddenBoundarySegment,
    neighbors,
    makeInitialState,
    cloneState,
    legalMoves,
    grantsExtraTurn,
    applyMove,
    isTerminal,
    winner,
    RandomBot,
  });
})(globalThis);
