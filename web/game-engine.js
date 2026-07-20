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

  const BotKind = Object.freeze({
    Random: "random",
    Mcts: "mcts",
    AlphaBeta: "alphaBeta",
  });

  const DEFAULT_BOT_ITERATIONS = 2000;
  const DEFAULT_ALPHA_BETA_DEPTH = 6;
  const MAX_ALPHA_BETA_DEPTH = 64;
  const MAX_UINT32 = 0xffffffff;

  function samePoint(left, right) {
    return left.x === right.x && left.y === right.y;
  }

  function botKindValue(kind) {
    if (kind === BotKind.Random) {
      return 0;
    }
    if (kind === BotKind.Mcts) {
      return 1;
    }
    if (kind === BotKind.AlphaBeta) {
      return 2;
    }
    throw new Error("Unsupported bot kind: " + String(kind));
  }

  function unsignedInteger(value, label, allowZero, maximum = MAX_UINT32) {
    const number = Number(value);
    const minimum = allowZero ? 0 : 1;
    if (!Number.isInteger(number) || number < minimum || number > maximum) {
      throw new Error(
        label + " must be an integer between " + minimum + " and " + maximum + ".",
      );
    }
    return number;
  }

  function botConfigArguments(config, label) {
    if (!config || typeof config !== "object") {
      throw new Error(label + " bot configuration is required.");
    }

    const kindValue = botKindValue(config.kind);
    const searchSetting = config.kind === BotKind.AlphaBeta
      ? unsignedInteger(
        config.depth ?? DEFAULT_ALPHA_BETA_DEPTH,
        label + " AlphaBetaBot depth",
        false,
        MAX_ALPHA_BETA_DEPTH,
      )
      : unsignedInteger(
        config.iterations ?? DEFAULT_BOT_ITERATIONS,
        label + " new simulations per bot move",
        false,
      );

    return [String(config.seed), kindValue, searchSetting];
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

    function readSnapshot(name, expectedSchema, sessionName) {
      const raw = call(name, "string");
      if (!raw) {
        throw new Error(lastError());
      }

      const value = JSON.parse(raw);
      if (value.schema !== expectedSchema) {
        throw new Error(
          "Unsupported C++ " + sessionName + " schema: " + String(value.schema),
        );
      }
      return value;
    }

    function snapshot() {
      return readSnapshot(
        "ps_snapshot_json",
        "papersoccer.web-session.v1",
        "web-session",
      );
    }

    function botReplaySnapshot() {
      return readSnapshot(
        "ps_bot_replay_snapshot_json",
        "papersoccer.bot-replay-session.v1",
        "bot-replay-session",
      );
    }

    function humanMatch() {
      return readSnapshot(
        "ps_human_match_json",
        "papersoccer.human-match.v1",
        "human-match",
      );
    }

    function runCommand(name, argumentTypes, argumentsList, snapshotReader) {
      const succeeded = call(name, "number", argumentTypes, argumentsList);
      if (!succeeded) {
        throw new Error(lastError());
      }
      return snapshotReader();
    }

    return Object.freeze({
      startGame: function (
        seed,
        humanPlayer,
        botKind = BotKind.Random,
        searchSetting,
      ) {
        const humanValue = humanPlayer === Player.Two ? 2 : 1;
        const kindValue = botKindValue(botKind);
        const settingValue = botKind === BotKind.AlphaBeta
          ? unsignedInteger(
            searchSetting ?? DEFAULT_ALPHA_BETA_DEPTH,
            "AlphaBetaBot depth",
            false,
            MAX_ALPHA_BETA_DEPTH,
          )
          : unsignedInteger(
            searchSetting ?? DEFAULT_BOT_ITERATIONS,
            "New simulations per bot move",
            false,
          );
        return runCommand(
          "ps_start_game",
          ["string", "number", "number", "number"],
          [String(seed), humanValue, kindValue, settingValue],
          snapshot,
        );
      },
      playHuman: function (expectedSessionId, expectedRevision, moveId) {
        return runCommand(
          "ps_play_human",
          ["number", "number", "number"],
          [expectedSessionId, expectedRevision, moveId],
          snapshot,
        );
      },
      playBot: function (expectedSessionId, expectedRevision) {
        return runCommand(
          "ps_play_bot",
          ["number", "number"],
          [expectedSessionId, expectedRevision],
          snapshot,
        );
      },
      undo: function (expectedSessionId, expectedRevision) {
        return runCommand(
          "ps_undo",
          ["number", "number"],
          [expectedSessionId, expectedRevision],
          snapshot,
        );
      },
      humanMatch: humanMatch,
      startBotReplay: function (oneConfig, twoConfig, maxPlies = 512) {
        const one = botConfigArguments(oneConfig, "Player 1");
        const two = botConfigArguments(twoConfig, "Player 2");
        const maximum = unsignedInteger(
          maxPlies,
          "Maximum replay moves",
          true,
        );
        return runCommand(
          "ps_start_bot_replay",
          [
            "string",
            "number",
            "number",
            "string",
            "number",
            "number",
            "number",
          ],
          [one[0], one[1], one[2], two[0], two[1], two[2], maximum],
          botReplaySnapshot,
        );
      },
      playBotReplay: function (expectedSessionId, expectedRevision) {
        return runCommand(
          "ps_play_bot_replay",
          ["number", "number"],
          [expectedSessionId, expectedRevision],
          botReplaySnapshot,
        );
      },
      snapshot: snapshot,
      botReplaySnapshot: botReplaySnapshot,
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
    BotKind,
    samePoint,
    ready,
  });
}(globalThis));
