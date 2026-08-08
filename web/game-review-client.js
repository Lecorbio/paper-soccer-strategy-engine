(function (root) {
  "use strict";

  const GAME_REVIEW_SCHEMA = "papersoccer.game-review.v1";
  const HUMAN_MATCH_SCHEMA = "papersoccer.human-match.v1";
  const REPLAY_V1_SCHEMA = "papersoccer.replay.v1";
  const REPLAY_V2_SCHEMA = "papersoccer.replay.v2";
  const REVIEW_SESSION_SCHEMA = "papersoccer.review-session.v1";

  const ReviewMode = Object.freeze({
    Fast: "fast",
    Deep: "deep",
  });

  const STATUS_VALUES = Object.freeze({
    inProgress: 0,
    wonByOne: 1,
    wonByTwo: 2,
  });

  function cloneJson(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function sourceReplayFrom(document) {
    if (!document || typeof document !== "object") {
      throw new Error("A replay document is required.");
    }
    if (document.schema === HUMAN_MATCH_SCHEMA) {
      if (!document.replay || typeof document.replay !== "object") {
        throw new Error("The human-match export does not contain a replay.");
      }
      return document.replay;
    }
    if (document.schema === GAME_REVIEW_SCHEMA) {
      if (!document.sourceReplay || typeof document.sourceReplay !== "object") {
        throw new Error("The game-review export does not contain its source replay.");
      }
      return document.sourceReplay;
    }
    return document;
  }

  function canonicalJson(value) {
    if (value === null || typeof value !== "object") {
      return JSON.stringify(value);
    }
    if (Array.isArray(value)) {
      return "[" + value.map(canonicalJson).join(",") + "]";
    }
    const keys = Object.keys(value).sort();
    return "{" + keys.map(function (key) {
      return JSON.stringify(key) + ":" + canonicalJson(value[key]);
    }).join(",") + "}";
  }

  function utf8Bytes(text) {
    if (typeof TextEncoder === "function") {
      return new TextEncoder().encode(text);
    }
    const encoded = unescape(encodeURIComponent(text));
    return Uint8Array.from(encoded, function (character) {
      return character.charCodeAt(0);
    });
  }

  function rotateRight(value, count) {
    return (value >>> count) | (value << (32 - count));
  }

  // A small synchronous implementation keeps replay identities available for
  // direct-file pages, where SubtleCrypto is not uniformly exposed.
  function sha256(text) {
    const input = utf8Bytes(text);
    const bitLength = input.length * 8;
    const paddedLength = Math.ceil((input.length + 9) / 64) * 64;
    const bytes = new Uint8Array(paddedLength);
    bytes.set(input);
    bytes[input.length] = 0x80;
    const view = new DataView(bytes.buffer);
    const high = Math.floor(bitLength / 0x100000000);
    const low = bitLength >>> 0;
    view.setUint32(paddedLength - 8, high, false);
    view.setUint32(paddedLength - 4, low, false);

    const constants = [
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
      0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
      0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
      0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    const hash = [
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    const words = new Uint32Array(64);

    for (let offset = 0; offset < bytes.length; offset += 64) {
      for (let i = 0; i < 16; i += 1) {
        words[i] = view.getUint32(offset + i * 4, false);
      }
      for (let i = 16; i < 64; i += 1) {
        const before = words[i - 15];
        const last = words[i - 2];
        const smallZero = rotateRight(before, 7) ^ rotateRight(before, 18) ^
          (before >>> 3);
        const smallOne = rotateRight(last, 17) ^ rotateRight(last, 19) ^
          (last >>> 10);
        words[i] = (words[i - 16] + smallZero + words[i - 7] + smallOne) >>> 0;
      }

      let a = hash[0];
      let b = hash[1];
      let c = hash[2];
      let d = hash[3];
      let e = hash[4];
      let f = hash[5];
      let g = hash[6];
      let h = hash[7];
      for (let i = 0; i < 64; i += 1) {
        const bigOne = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
        const choose = (e & f) ^ (~e & g);
        const temporaryOne = (h + bigOne + choose + constants[i] + words[i]) >>> 0;
        const bigZero = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
        const majority = (a & b) ^ (a & c) ^ (b & c);
        const temporaryTwo = (bigZero + majority) >>> 0;
        h = g;
        g = f;
        f = e;
        e = (d + temporaryOne) >>> 0;
        d = c;
        c = b;
        b = a;
        a = (temporaryOne + temporaryTwo) >>> 0;
      }
      hash[0] = (hash[0] + a) >>> 0;
      hash[1] = (hash[1] + b) >>> 0;
      hash[2] = (hash[2] + c) >>> 0;
      hash[3] = (hash[3] + d) >>> 0;
      hash[4] = (hash[4] + e) >>> 0;
      hash[5] = (hash[5] + f) >>> 0;
      hash[6] = (hash[6] + g) >>> 0;
      hash[7] = (hash[7] + h) >>> 0;
    }
    return hash.map(function (value) {
      return value.toString(16).padStart(8, "0");
    }).join("");
  }

  function requireInteger(value, label) {
    if (!Number.isInteger(value)) {
      throw new Error(label + " must be an integer.");
    }
    return value;
  }

  function requirePoint(point, label, yOffset) {
    if (!point || typeof point !== "object") {
      throw new Error(label + " is required.");
    }
    return {
      x: requireInteger(point.x, label + " x"),
      y: requireInteger(point.y, label + " y") + yOffset,
    };
  }

  function statusValue(status, label) {
    if (!Object.prototype.hasOwnProperty.call(STATUS_VALUES, status)) {
      throw new Error(label + " has an unsupported status: " + String(status));
    }
    return STATUS_VALUES[status];
  }

  function playerValue(player, label, allowNull) {
    if (allowNull && (player === null || player === undefined)) {
      return 0;
    }
    if (player === "one") {
      return 1;
    }
    if (player === "two") {
      return 2;
    }
    throw new Error(label + " must be Player 1 or Player 2.");
  }

  function prepareReplay(document) {
    const sourceReplay = cloneJson(sourceReplayFrom(document));
    const sourceSchema = sourceReplay.schema ?? REPLAY_V1_SCHEMA;
    if (sourceSchema !== REPLAY_V1_SCHEMA && sourceSchema !== REPLAY_V2_SCHEMA) {
      throw new Error("Unsupported replay schema: " + String(sourceSchema));
    }
    if (!sourceReplay.rules || sourceReplay.rules.width !== 8 ||
        sourceReplay.rules.height !== 10) {
      throw new Error("Game Review supports the standard 8×10 demo rules only.");
    }
    if (!Array.isArray(sourceReplay.moves)) {
      throw new Error("The replay moves must be an array.");
    }

    const yOffset = sourceSchema === REPLAY_V1_SCHEMA ? 1 : 0;
    const start = requirePoint(sourceReplay.start, "Replay start", yOffset);
    if (start.x !== 4 || start.y !== 6) {
      throw new Error(
        "Game Review requires the standard initial ball position at (4, 6).",
      );
    }
    const moves = sourceReplay.moves.map(function (move, index) {
      if (!move || typeof move !== "object") {
        throw new Error("Replay move " + (index + 1) + " is invalid.");
      }
      return {
        ply: requireInteger(move.ply, "Replay move " + (index + 1) + " ply"),
        player: playerValue(move.player, "Replay move " + (index + 1) + " player", false),
        from: requirePoint(move.from, "Replay move " + (index + 1) + " from", yOffset),
        to: requirePoint(move.to, "Replay move " + (index + 1) + " to", yOffset),
        extraTurn: move.extraTurn === true ? 1 : (move.extraTurn === false ? 0 : -1),
        statusAfter: statusValue(
          move.statusAfter,
          "Replay move " + (index + 1) + " statusAfter",
        ),
      };
    });
    const status = statusValue(sourceReplay.status, "Replay outcome");
    const winner = playerValue(sourceReplay.winner, "Replay winner", true);
    if (typeof sourceReplay.truncated !== "boolean") {
      throw new Error("Replay truncated must be true or false.");
    }
    const replaySha256 = sha256(canonicalJson(sourceReplay));
    if (document?.schema === GAME_REVIEW_SCHEMA &&
        document.replaySha256 !== replaySha256) {
      throw new Error(
        "The game-review export replay SHA-256 does not match its source replay.",
      );
    }
    return Object.freeze({
      sourceReplay,
      sourceSchema,
      start,
      moves,
      status,
      winner,
      truncated: sourceReplay.truncated ? 1 : 0,
      replaySha256,
    });
  }

  function createCabiClient(module) {
    function call(name, returnType, argumentTypes, argumentsList) {
      return module.ccall(
        name,
        returnType,
        argumentTypes ?? [],
        argumentsList ?? [],
      );
    }

    function lastError() {
      return call("ps_review_last_error", "string") ||
        "The C++ analysis engine rejected the command.";
    }

    function command(name, argumentTypes, argumentsList) {
      const result = call(name, "number", argumentTypes, argumentsList);
      if (!result) {
        throw new Error(lastError());
      }
      return result;
    }

    function snapshot() {
      const raw = call("ps_review_snapshot_json", "string");
      if (!raw) {
        throw new Error(lastError());
      }
      const value = JSON.parse(raw);
      if (!value || typeof value !== "object") {
        throw new Error("The C++ analysis engine returned an invalid snapshot.");
      }
      return value;
    }

    function sandboxSnapshot() {
      const raw = call("ps_review_sandbox_snapshot_json", "string");
      if (!raw) {
        throw new Error(lastError());
      }
      const value = JSON.parse(raw);
      if (!value || value.schema !== "papersoccer.review-sandbox.v1") {
        throw new Error("The C++ analysis engine returned an invalid sandbox snapshot.");
      }
      return value;
    }

    return Object.freeze({
      start(mode) {
        return command(
          "ps_review_start",
          ["number"],
          [mode === ReviewMode.Deep ? 1 : 0],
        );
      },
      appendMove(move) {
        return command(
          "ps_review_append_move",
          ["number", "number", "number", "number", "number", "number", "number", "number"],
          [move.ply, move.player, move.from.x, move.from.y, move.to.x, move.to.y,
            move.extraTurn, move.statusAfter],
        );
      },
      finalize(replay) {
        return command(
          "ps_review_finalize",
          ["number", "number", "number"],
          [replay.status, replay.winner, replay.truncated],
        );
      },
      step() {
        return command("ps_review_step");
      },
      snapshot,
      cancel() {
        return call("ps_review_cancel", "number") || 0;
      },
      sandboxStart(possession) {
        command("ps_review_sandbox_start", ["number"], [possession]);
        return sandboxSnapshot();
      },
      sandboxPlay(point) {
        command(
          "ps_review_sandbox_play",
          ["number", "number"],
          [point.x, point.y],
        );
        return sandboxSnapshot();
      },
      sandboxClose() {
        return call("ps_review_sandbox_close", "number") || 0;
      },
      sandboxSnapshot,
      lastError,
      heapBytes() {
        return call("ps_review_heap_bytes", "number");
      },
    });
  }

  function populateSession(cabi, prepared, mode) {
    cabi.start(mode);
    for (const move of prepared.moves) {
      cabi.appendMove(move);
    }
    cabi.finalize(prepared);
    return cabi.snapshot();
  }

  function reviewDone(snapshot) {
    if (snapshot?.done === true || snapshot?.complete === true) {
      return true;
    }
    if (snapshot?.status === "complete" || snapshot?.status === "cancelled") {
      return true;
    }
    const completed = snapshot?.completedPossessions ??
      snapshot?.progress?.completed;
    const total = snapshot?.totalPossessions ?? snapshot?.progress?.total;
    return Number.isInteger(completed) && Number.isInteger(total) &&
      total >= 0 && completed >= total;
  }

  function stripWallClockFields(value) {
    if (Array.isArray(value)) {
      return value.map(stripWallClockFields);
    }
    if (!value || typeof value !== "object") {
      return value;
    }
    const result = {};
    for (const [key, child] of Object.entries(value)) {
      if (/(?:wall.?clock|elapsed|duration|latency|decision.?time|time.?ns)/iu.test(key)) {
        continue;
      }
      result[key] = stripWallClockFields(child);
    }
    return result;
  }

  function buildReviewExport(prepared, snapshot) {
    const deterministic = stripWallClockFields(snapshot);
    const exported = {
      schema: GAME_REVIEW_SCHEMA,
      sourceReplay: cloneJson(prepared.sourceReplay),
      replaySha256: prepared.replaySha256,
    };
    for (const key of [
      "analyzer", "profile", "calibration", "oracle", "rankedSource",
      "mode", "complete", "completedPossessions", "totalPossessions",
      "replayStatus", "truncated", "summary", "possessions",
    ]) {
      if (deterministic[key] !== undefined) {
        exported[key] = deterministic[key];
      }
    }
    if (!exported.possessions && Array.isArray(deterministic.reviews)) {
      exported.possessions = deterministic.reviews;
    }
    return exported;
  }

  function createDefaultScriptLoader(documentObject, globalObject) {
    return function loadScript(url, factoryName) {
      if (typeof globalObject[factoryName] === "function") {
        return Promise.resolve();
      }
      return new Promise(function (resolve, reject) {
        const script = documentObject.createElement("script");
        script.src = url;
        script.async = true;
        script.addEventListener("load", function () {
          if (typeof globalObject[factoryName] === "function") {
            resolve();
          } else {
            reject(new Error("The analysis module loaded without its factory."));
          }
        }, { once: true });
        script.addEventListener("error", function () {
          reject(new Error("The analysis module could not be loaded."));
        }, { once: true });
        documentObject.head.append(script);
      });
    };
  }

  function createAnalysisRunner(options) {
    const settings = options ?? {};
    const globalObject = settings.globalObject ?? root;
    const documentObject = settings.documentObject ?? globalObject.document;
    const timerHost = settings.timerHost ?? globalObject;
    const WorkerConstructor = settings.WorkerConstructor ?? globalObject.Worker;
    const protocol = settings.protocol ?? globalObject.location?.protocol ?? "";
    const workerUrl = settings.workerUrl ?? "game-review-worker.js";
    const analysisScriptUrl = settings.analysisScriptUrl ??
      "papersoccer-analysis-wasm.js";
    const factoryName = settings.factoryName ?? "createPaperSoccerAnalysisModule";
    const loadScript = settings.loadScript ??
      createDefaultScriptLoader(documentObject, globalObject);
    const onEvent = typeof settings.onEvent === "function"
      ? settings.onEvent
      : function () {};

    let issuedSessionId = 0;
    let active = null;
    let backendPromise = null;
    let backend = null;
    let transport = null;
    let sandboxSessionId = 0;
    let sandboxRequestId = 0;
    const sandboxPending = new Map();

    function emit(event) {
      onEvent(Object.freeze(event));
    }

    function nextSessionId() {
      issuedSessionId += 1;
      return issuedSessionId;
    }

    function dispatch(message) {
      if (message?.type === "sandbox" || message?.type === "sandbox-error") {
        const pending = sandboxPending.get(message.requestId);
        if (!pending || message.sessionId !== pending.sessionId ||
            message.sessionId !== sandboxSessionId) {
          return;
        }
        sandboxPending.delete(message.requestId);
        if (message.type === "sandbox-error") {
          pending.reject(new Error(message.error || "The try-line sandbox failed."));
          return;
        }
        pending.resolve(message.snapshot);
        emit({
          type: "sandbox",
          sessionId: message.sessionId,
          snapshot: message.snapshot,
          transport,
        });
        return;
      }
      if (!message || !active || message.sessionId !== active.sessionId) {
        return;
      }
      if (message.type === "error") {
        const error = new Error(message.error || "Game Review failed.");
        const request = active;
        active = null;
        request.reject?.(error);
        emit({ type: "error", sessionId: request.sessionId, error });
        return;
      }
      if (message.type === "validated") {
        const request = active;
        if (request.kind === "validate") {
          active = null;
          request.resolve(message.snapshot);
        }
        emit({
          type: "validated",
          sessionId: request.sessionId,
          snapshot: message.snapshot,
          transport,
        });
        return;
      }
      if (message.type === "progress") {
        emit({
          type: "progress",
          sessionId: active.sessionId,
          mode: active.mode,
          snapshot: message.snapshot,
          transport,
        });
        return;
      }
      if (message.type === "complete") {
        const request = active;
        active = null;
        emit({
          type: "complete",
          sessionId: request.sessionId,
          mode: request.mode,
          snapshot: message.snapshot,
          prepared: request.prepared,
          transport,
        });
      }
    }

    function createWorkerBackend() {
      return new Promise(function (resolve, reject) {
        if (typeof WorkerConstructor !== "function") {
          reject(new Error("Classic Web Workers are unavailable."));
          return;
        }
        let worker;
        try {
          worker = new WorkerConstructor(workerUrl);
        } catch (error) {
          reject(error);
          return;
        }
        let settled = false;
        const timeout = timerHost.setTimeout(function () {
          if (settled) {
            return;
          }
          settled = true;
          worker.terminate?.();
          reject(new Error("The analysis worker did not finish loading."));
        }, settings.workerReadyTimeoutMs ?? 4000);
        worker.onmessage = function (event) {
          const message = event.data;
          if (!settled && message?.type === "ready") {
            settled = true;
            timerHost.clearTimeout(timeout);
            resolve(Object.freeze({
              post(request) { worker.postMessage(request); },
              dispose() { worker.terminate?.(); },
            }));
            return;
          }
          if (settled) {
            dispatch(message);
          }
        };
        worker.onerror = function (event) {
          const error = new Error(event?.message || "The analysis worker failed.");
          if (!settled) {
            settled = true;
            timerHost.clearTimeout(timeout);
            worker.terminate?.();
            reject(error);
          } else if (active) {
            dispatch({ type: "error", sessionId: active.sessionId, error: error.message });
          }
        };
      });
    }

    async function createFallbackBackend() {
      if (!documentObject) {
        throw new Error("The direct-file analysis fallback needs a document.");
      }
      await loadScript(analysisScriptUrl, factoryName);
      const factory = globalObject[factoryName];
      if (typeof factory !== "function") {
        throw new Error("The analysis module factory is unavailable.");
      }
      const module = await factory();
      const cabi = createCabiClient(module);
      let localSessionId = 0;
      let cancelled = false;

      function fail(sessionId, error) {
        dispatch({ type: "error", sessionId, error: error.message });
      }

      function step(sessionId) {
        if (sessionId !== localSessionId || cancelled) {
          return;
        }
        try {
          const stepResult = cabi.step();
          const snapshot = cabi.snapshot();
          if (sessionId !== localSessionId || cancelled) {
            return;
          }
          if (reviewDone(snapshot) || stepResult > 1) {
            dispatch({ type: "complete", sessionId, snapshot });
            return;
          }
          dispatch({ type: "progress", sessionId, snapshot });
          timerHost.setTimeout(function () { step(sessionId); }, 0);
        } catch (error) {
          fail(sessionId, error);
        }
      }

      return Object.freeze({
        post(request) {
          if (request.type === "sandbox-start" ||
              request.type === "sandbox-play") {
            try {
              const snapshot = request.type === "sandbox-start"
                ? cabi.sandboxStart(request.possession)
                : cabi.sandboxPlay(request.point);
              dispatch({
                type: "sandbox",
                sessionId: request.sessionId,
                requestId: request.requestId,
                snapshot,
              });
            } catch (error) {
              dispatch({
                type: "sandbox-error",
                sessionId: request.sessionId,
                requestId: request.requestId,
                error: error.message,
              });
            }
            return;
          }
          if (request.type === "sandbox-close") {
            cabi.sandboxClose();
            return;
          }
          if (request.type === "cancel") {
            if (request.sessionId === localSessionId) {
              cancelled = true;
              cabi.cancel();
            }
            return;
          }
          if (request.type !== "start" && request.type !== "validate") {
            return;
          }
          localSessionId = request.sessionId;
          cancelled = false;
          try {
            const snapshot = populateSession(cabi, request.prepared, request.mode);
            dispatch({ type: "validated", sessionId: request.sessionId, snapshot });
            if (request.type === "start") {
              timerHost.setTimeout(function () { step(request.sessionId); }, 0);
            }
          } catch (error) {
            fail(request.sessionId, error);
          }
        },
        dispose() {
          cancelled = true;
          cabi.cancel();
        },
      });
    }

    function ensureBackend() {
      if (backendPromise) {
        return backendPromise;
      }
      backendPromise = createWorkerBackend().then(function (created) {
        backend = created;
        transport = "worker";
        emit({ type: "transport", transport });
        return created;
      }).catch(async function (workerError) {
        if (protocol !== "file:") {
          throw workerError;
        }
        const created = await createFallbackBackend();
        backend = created;
        transport = "main-thread-fallback";
        emit({ type: "transport", transport, workerError });
        return created;
      });
      return backendPromise;
    }

    function begin(kind, document, mode) {
      closeSandbox();
      cancel();
      const prepared = prepareReplay(document);
      const sessionId = nextSessionId();
      let resolve;
      let reject;
      const promise = kind === "validate" ? new Promise(function (accept, decline) {
        resolve = accept;
        reject = decline;
      }) : null;
      active = { kind, sessionId, mode, prepared, resolve, reject };
      emit({ type: "loading", kind, sessionId, mode });
      ensureBackend().then(function (created) {
        if (!active || active.sessionId !== sessionId) {
          return;
        }
        created.post({ type: kind, sessionId, mode, prepared });
      }).catch(function (error) {
        dispatch({ type: "error", sessionId, error: error.message });
      });
      return kind === "validate"
        ? promise
        : Object.freeze({ sessionId, prepared });
    }

    function cancel() {
      if (!active) {
        return false;
      }
      const cancelled = active;
      active = null;
      cancelled.reject?.(new Error("Game Review was cancelled."));
      backend?.post({ type: "cancel", sessionId: cancelled.sessionId });
      emit({ type: "cancelled", sessionId: cancelled.sessionId });
      return true;
    }

    function sandboxCommand(type, fields, createSession) {
      if (active) {
        return Promise.reject(new Error(
          "Wait for Game Review analysis to finish before opening the sandbox.",
        ));
      }
      if (createSession) {
        closeSandbox();
        sandboxSessionId = nextSessionId();
      } else if (sandboxSessionId === 0) {
        return Promise.reject(new Error("No try-line sandbox is active."));
      }
      const sessionId = sandboxSessionId;
      sandboxRequestId += 1;
      const requestId = sandboxRequestId;
      return new Promise(function (resolve, reject) {
        sandboxPending.set(requestId, { sessionId, resolve, reject });
        ensureBackend().then(function (created) {
          const pending = sandboxPending.get(requestId);
          if (!pending || sandboxSessionId !== sessionId) {
            return;
          }
          created.post(Object.assign({
            type,
            sessionId,
            requestId,
          }, fields));
        }).catch(function (error) {
          const pending = sandboxPending.get(requestId);
          sandboxPending.delete(requestId);
          pending?.reject(error);
        });
      });
    }

    function closeSandbox() {
      if (sandboxSessionId === 0) {
        return false;
      }
      const closingSession = sandboxSessionId;
      sandboxSessionId = 0;
      for (const [requestId, pending] of sandboxPending) {
        if (pending.sessionId === closingSession) {
          pending.reject(new Error("The try-line sandbox was closed."));
          sandboxPending.delete(requestId);
        }
      }
      backend?.post({ type: "sandbox-close", sessionId: closingSession });
      emit({ type: "sandbox-closed", sessionId: closingSession });
      return true;
    }

    return Object.freeze({
      start(document, mode = ReviewMode.Fast) {
        return begin("start", document, mode);
      },
      validate(document) {
        return begin("validate", document, ReviewMode.Fast);
      },
      startSandbox(possession) {
        return sandboxCommand(
          "sandbox-start",
          { possession },
          true,
        );
      },
      playSandbox(point) {
        return sandboxCommand("sandbox-play", { point }, false);
      },
      closeSandbox,
      cancel,
      dispose() {
        closeSandbox();
        cancel();
        backend?.dispose();
        backend = null;
        backendPromise = null;
      },
      get transport() { return transport; },
    });
  }

  root.PaperSoccerGameReview = Object.freeze({
    GAME_REVIEW_SCHEMA,
    REVIEW_SESSION_SCHEMA,
    ReviewMode,
    buildReviewExport,
    canonicalJson,
    cloneJson,
    createAnalysisRunner,
    createCabiClient,
    populateSession,
    prepareReplay,
    reviewDone,
    sha256,
    sourceReplayFrom,
    stripWallClockFields,
  });
}(globalThis));
