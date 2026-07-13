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

  function samePoint(left, right) {
    return left.x === right.x && left.y === right.y;
  }

  function createClient(module) {
    function call(name, returnType, argumentTypes, argumentsList) {
      return module.ccall(
        name,
        returnType,
        argumentTypes ?? [],
        argumentsList ?? [],
      );
    }

    function lastError() {
      return call("ps_last_error", "string") || "The C++ game engine rejected the command.";
    }

    function snapshot() {
      const raw = call("ps_snapshot_json", "string");
      if (!raw) {
        throw new Error(lastError());
      }

      const value = JSON.parse(raw);
      if (value.schema !== "papersoccer.web-session.v1") {
        throw new Error("Unsupported C++ web-session schema: " + String(value.schema));
      }
      return value;
    }

    function runCommand(name, argumentTypes, argumentsList) {
      const succeeded = call(name, "number", argumentTypes, argumentsList);
      if (!succeeded) {
        throw new Error(lastError());
      }
      return snapshot();
    }

    return Object.freeze({
      startGame: function (seed, humanPlayer) {
        const humanValue = humanPlayer === Player.Two ? 2 : 1;
        return runCommand(
          "ps_start_game",
          ["string", "number"],
          [String(seed), humanValue],
        );
      },
      playHuman: function (expectedSessionId, expectedRevision, moveId) {
        return runCommand(
          "ps_play_human",
          ["number", "number", "number"],
          [expectedSessionId, expectedRevision, moveId],
        );
      },
      playBot: function (expectedSessionId, expectedRevision) {
        return runCommand(
          "ps_play_bot",
          ["number", "number"],
          [expectedSessionId, expectedRevision],
        );
      },
      snapshot: snapshot,
    });
  }

  const ready = Promise.resolve().then(function () {
    if (typeof root.createPaperSoccerModule !== "function") {
      throw new Error("The compiled C++ game engine did not load.");
    }
    return root.createPaperSoccerModule();
  }).then(createClient);

  root.PaperSoccer = Object.freeze({
    Player,
    Status,
    samePoint,
    ready,
  });
}(globalThis));
