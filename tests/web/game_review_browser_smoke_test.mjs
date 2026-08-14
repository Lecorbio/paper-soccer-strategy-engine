import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import {
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

const REPOSITORY = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const WEB_ROOT = path.join(REPOSITORY, "web");
const CALIBRATION_LOCK = path.join(
  REPOSITORY,
  "benchmarks/game_review_gate/game_review_calibration_lock.hpp",
);
// The checked-in pre-selection artifact intentionally refuses review sessions.
// Presence of the validation-derived lock turns this into a mandatory smoke.
const LOCKED = existsSync(CALIBRATION_LOCK);
const EXPECTED_POSSESSIONS = 9;
const CHROME_STARTUP_ATTEMPTS = 2;
const CHROME_STARTUP_TIMEOUT_MS = 30000;

const TERMINAL_REPLAY = Object.freeze({
  schema: "papersoccer.replay.v2",
  rules: { width: 8, height: 10 },
  players: {
    one: { kind: "RandomBot", seed: "17" },
    two: { kind: "RandomBot", seed: "18" },
  },
  start: { x: 4, y: 6 },
  status: "wonByOne",
  winner: "one",
  truncated: false,
  moves: [
    { ply: 1, player: "one", from: { x: 4, y: 6 }, to: { x: 4, y: 5 }, extraTurn: false, statusAfter: "inProgress" },
    { ply: 2, player: "two", from: { x: 4, y: 5 }, to: { x: 4, y: 4 }, extraTurn: false, statusAfter: "inProgress" },
    { ply: 3, player: "one", from: { x: 4, y: 4 }, to: { x: 4, y: 3 }, extraTurn: false, statusAfter: "inProgress" },
    { ply: 4, player: "two", from: { x: 4, y: 3 }, to: { x: 4, y: 2 }, extraTurn: false, statusAfter: "inProgress" },
    { ply: 5, player: "one", from: { x: 4, y: 2 }, to: { x: 4, y: 1 }, extraTurn: false, statusAfter: "inProgress" },
    { ply: 6, player: "two", from: { x: 4, y: 1 }, to: { x: 3, y: 2 }, extraTurn: false, statusAfter: "inProgress" },
    { ply: 7, player: "one", from: { x: 3, y: 2 }, to: { x: 2, y: 3 }, extraTurn: false, statusAfter: "inProgress" },
    { ply: 8, player: "two", from: { x: 2, y: 3 }, to: { x: 3, y: 3 }, extraTurn: false, statusAfter: "inProgress" },
    { ply: 9, player: "one", from: { x: 3, y: 3 }, to: { x: 4, y: 2 }, extraTurn: true, statusAfter: "inProgress" },
    { ply: 10, player: "one", from: { x: 4, y: 2 }, to: { x: 3, y: 1 }, extraTurn: true, statusAfter: "inProgress" },
    { ply: 11, player: "one", from: { x: 3, y: 1 }, to: { x: 4, y: 0 }, extraTurn: false, statusAfter: "wonByOne" },
  ],
});

function chromeExecutable() {
  const configured = process.env.PAPERSOCCER_CHROME || process.env.CHROME_PATH;
  const candidates = [
    configured,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  return candidates.find((candidate) => existsSync(candidate)) ?? null;
}

class ChromeStartupTimeoutError extends Error {}

function timeoutAfter(milliseconds, message, ErrorType = Error) {
  return new Promise((_, reject) => {
    const timer = setTimeout(() => {
      const detail = typeof message === "function" ? message() : message;
      reject(new ErrorType(detail));
    }, milliseconds);
    timer.unref?.();
  });
}

async function launchChromeAttempt(executable, profileDirectory) {
  const args = [
    "--headless=new",
    "--remote-debugging-pipe",
    `--user-data-dir=${profileDirectory}`,
    "--allow-file-access-from-files",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--disable-gpu",
    "--disable-sync",
    "--metrics-recording-only",
    "--no-default-browser-check",
    "--no-first-run",
    "--password-store=basic",
    "about:blank",
  ];
  if (typeof process.getuid === "function" && process.getuid() === 0) {
    args.push("--no-sandbox");
  }

  const child = spawn(executable, args, {
    stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"],
  });
  let output = "";
  child.stderr.on("data", (chunk) => {
    output = (output + chunk.toString()).slice(-65536);
  });
  const connection = new CdpConnection(child.stdio[3], child.stdio[4]);
  child.once("error", (error) => connection.fail(error));
  child.once("exit", (code, signal) => connection.fail(new Error(
    `Chrome exited (${code ?? signal}).\n${output}`,
  )));

  try {
    await Promise.race([
      connection.send("Browser.getVersion"),
      timeoutAfter(CHROME_STARTUP_TIMEOUT_MS, () =>
        `Chrome (${executable}, pid ${child.pid ?? "unknown"}) did not expose ` +
        `its DevTools pipe after ${CHROME_STARTUP_TIMEOUT_MS} ms.\n` +
        (output || "(no stderr output)"), ChromeStartupTimeoutError),
    ]);
    return { child, connection, output: () => output };
  } catch (error) {
    try {
      await stopChrome(child);
    } catch (cleanupError) {
      const launchDetail = error instanceof Error ? error.message : String(error);
      const cleanupDetail = cleanupError instanceof Error
        ? cleanupError.message
        : String(cleanupError);
      throw new Error(
        `${launchDetail}\nChrome cleanup also failed: ${cleanupDetail}`,
      );
    }
    throw error;
  }
}

async function launchChrome(executable, profileDirectory) {
  const failures = [];
  for (let attempt = 1; attempt <= CHROME_STARTUP_ATTEMPTS; attempt += 1) {
    try {
      return await launchChromeAttempt(
        executable,
        `${profileDirectory}-attempt-${attempt}`,
      );
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      failures.push(`Attempt ${attempt}: ${detail}`);
      if (!(error instanceof ChromeStartupTimeoutError)) {
        throw error;
      }
    }
  }
  throw new Error(
    `Chrome failed to expose its DevTools pipe after ` +
    `${CHROME_STARTUP_ATTEMPTS} attempts.\n${failures.join("\n")}`,
  );
}

async function stopChrome(child) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return;
  }
  child.kill("SIGTERM");
  if (await waitForProcessExit(child, 3000)) {
    return;
  }
  child.kill("SIGKILL");
  if (!await waitForProcessExit(child, 5000)) {
    throw new Error("Chrome did not exit after forced browser-smoke cleanup.");
  }
}

function waitForProcessExit(child, timeoutMs) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve(true);
  }
  return new Promise((resolve) => {
    let settled = false;
    const finish = (exited) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      child.off("exit", onExit);
      resolve(exited);
    };
    const onExit = () => finish(true);
    const timer = setTimeout(() => finish(false), timeoutMs);
    timer.unref?.();
    child.once("exit", onExit);
    if (child.exitCode !== null || child.signalCode !== null) {
      finish(true);
    }
  });
}

async function stopStaticServer(server) {
  server.closeAllConnections?.();
  await Promise.race([
    new Promise((resolve, reject) => {
      server.close((error) => {
        if (error && error.code !== "ERR_SERVER_NOT_RUNNING") {
          reject(error);
        } else {
          resolve();
        }
      });
    }),
    timeoutAfter(5000, "The browser-smoke HTTP server did not close."),
  ]);
}

class CdpConnection {
  constructor(writer, reader) {
    this.writer = writer;
    this.reader = reader;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    this.buffer = "";
    reader.setEncoding("utf8");
    reader.on("data", (chunk) => {
      this.buffer += chunk;
      for (;;) {
        const terminator = this.buffer.indexOf("\0");
        if (terminator < 0) {
          break;
        }
        const text = this.buffer.slice(0, terminator);
        this.buffer = this.buffer.slice(terminator + 1);
        if (text) {
          this.receive(text);
        }
      }
    });
    reader.once("error", (error) => this.fail(error));
    reader.once("end", () => this.fail(new Error(
      "Chrome closed its DevTools pipe.",
    )));
    writer.once("error", (error) => this.fail(error));
  }

  receive(text) {
    try {
      const message = JSON.parse(text);
      if (message.id !== undefined) {
        const pending = this.pending.get(message.id);
        if (!pending) {
          return;
        }
        this.pending.delete(message.id);
        if (message.error) {
          pending.reject(new Error(
            `${pending.method}: ${message.error.message}`,
          ));
        } else {
          pending.resolve(message.result ?? {});
        }
        return;
      }
      const key = this.eventKey(message.sessionId, message.method);
      for (const listener of this.listeners.get(key) ?? []) {
        listener(message.params ?? {});
      }
    } catch (error) {
      this.fail(error);
    }
  }

  eventKey(sessionId, method) {
    return `${sessionId ?? ""}\0${method}`;
  }

  on(method, listener, sessionId) {
    const key = this.eventKey(sessionId, method);
    const listeners = this.listeners.get(key) ?? [];
    listeners.push(listener);
    this.listeners.set(key, listeners);
  }

  send(method, params = {}, sessionId) {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { method, resolve, reject });
      const message = { id, method, params };
      if (sessionId) {
        message.sessionId = sessionId;
      }
      this.writer.write(JSON.stringify(message) + "\0");
    });
  }

  fail(error) {
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  }
}

class CdpPage {
  constructor(connection, sessionId, targetId) {
    this.connection = connection;
    this.sessionId = sessionId;
    this.targetId = targetId;
  }

  on(method, listener) {
    this.connection.on(method, listener, this.sessionId);
  }

  send(method, params = {}) {
    return this.connection.send(method, params, this.sessionId);
  }

  async evaluate(expression, returnByValue = true) {
    const response = await this.send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue,
      userGesture: true,
    });
    if (response.exceptionDetails) {
      const description = response.exceptionDetails.exception?.description ??
        response.exceptionDetails.text;
      throw new Error(`Browser evaluation failed: ${description}`);
    }
    return returnByValue ? response.result.value : response.result;
  }

  async waitFor(expression, message, timeoutMs = 120000) {
    const deadline = Date.now() + timeoutMs;
    let lastError;
    while (Date.now() < deadline) {
      try {
        const value = await this.evaluate(expression);
        if (value) {
          return value;
        }
      } catch (error) {
        lastError = error;
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    const detail = lastError ? ` Last browser error: ${lastError.message}` : "";
    throw new Error(`${message}${detail}`);
  }

  async click(selector) {
    const quoted = JSON.stringify(selector);
    return this.evaluate(`(() => {
      const element = document.querySelector(${quoted});
      if (!element) throw new Error("Missing element: " + ${quoted});
      if (element.disabled) throw new Error("Disabled element: " + ${quoted});
      element.click();
      return true;
    })()`);
  }

  close() {
    return this.connection.send("Target.closeTarget", {
      targetId: this.targetId,
    });
  }
}

async function openPage(connection) {
  const { targetId } = await connection.send("Target.createTarget", {
    url: "about:blank",
  });
  const { sessionId } = await connection.send("Target.attachToTarget", {
    targetId,
    flatten: true,
  });
  return new CdpPage(connection, sessionId, targetId);
}

function mimeType(filename) {
  switch (path.extname(filename)) {
    case ".html": return "text/html; charset=utf-8";
    case ".js":
    case ".mjs": return "text/javascript; charset=utf-8";
    case ".css": return "text/css; charset=utf-8";
    case ".json": return "application/json; charset=utf-8";
    default: return "application/octet-stream";
  }
}

async function startStaticServer() {
  const server = http.createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url, "http://127.0.0.1");
      const relative = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, "") ||
        "index.html";
      const filename = path.resolve(WEB_ROOT, relative);
      if (filename !== WEB_ROOT && !filename.startsWith(WEB_ROOT + path.sep)) {
        response.writeHead(403).end();
        return;
      }
      const metadata = await stat(filename);
      if (!metadata.isFile()) {
        response.writeHead(404).end();
        return;
      }
      const body = await readFile(filename);
      response.writeHead(200, {
        "Cache-Control": "no-store",
        "Content-Length": body.length,
        "Content-Type": mimeType(filename),
      });
      response.end(request.method === "HEAD" ? undefined : body);
    } catch {
      response.writeHead(404).end();
    }
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert.ok(address && typeof address === "object");
  return {
    server,
    url: `http://127.0.0.1:${address.port}/index.html`,
  };
}

async function setReplayFile(page, replayPath) {
  const remoteObject = await page.evaluate(
    "document.querySelector('#fileInput')",
    false,
  );
  assert.ok(remoteObject.objectId, "the replay file input must exist");
  await page.send("DOM.setFileInputFiles", {
    files: [replayPath],
    objectId: remoteObject.objectId,
  });
}

async function captureReviewExport(page) {
  const count = await page.evaluate(
    "globalThis.__browserSmokeReviewExports.length",
  );
  await page.click("#exportReviewButton");
  await page.waitFor(
    `globalThis.__browserSmokeReviewExports.length > ${count}`,
    "Game Review did not produce an export.",
  );
  return page.evaluate(
    `JSON.parse(globalThis.__browserSmokeReviewExports[${count}])`,
  );
}

async function exerciseScenario(page, url, replayPath, expectedTransport, forceWorkerFailure) {
  const dialogs = [];
  page.on("Page.javascriptDialogOpening", (dialog) => {
    dialogs.push(dialog.message);
    void page.send("Page.handleJavaScriptDialog", { accept: false });
  });
  await Promise.all([
    page.send("Page.enable"),
    page.send("Runtime.enable"),
    page.send("DOM.enable"),
  ]);

  const forceWorker = forceWorkerFailure ? `
    globalThis.__browserSmokeForcedWorkerAttempts = 0;
    globalThis.Worker = class ForcedGameReviewWorkerFailure {
      constructor() {
        globalThis.__browserSmokeForcedWorkerAttempts += 1;
        throw new Error("forced file fallback for browser smoke test");
      }
    };
  ` : "";
  await page.send("Page.addScriptToEvaluateOnNewDocument", {
    source: `
      globalThis.__browserSmokeErrors = [];
      addEventListener("error", (event) => {
        globalThis.__browserSmokeErrors.push(String(event.error?.stack || event.message));
      });
      addEventListener("unhandledrejection", (event) => {
        globalThis.__browserSmokeErrors.push(String(event.reason?.stack || event.reason));
      });
      ${forceWorker}
    `,
  });

  await page.send("Page.navigate", { url });
  await page.waitFor(
    "document.readyState === 'complete' && document.querySelector('#newGameButton')?.disabled === false",
    "The browser demo did not finish loading.",
    30000,
  );
  assert.equal(
    await page.evaluate("globalThis.PaperSoccerGameReviewGate?.schema"),
    "papersoccer.game-review-gate-web.v1",
    "the frozen gate identity script must load in both browser transports",
  );
  await page.click("#replayModeButton");
  await setReplayFile(page, replayPath);
  await page.waitFor(
    "document.querySelector('#replayGenerationStatus')?.textContent.includes('ready for review')",
    "The imported terminal replay was not validated.",
    30000,
  );
  assert.deepEqual(dialogs, [], "the replay import must not open an error dialog");

  await page.waitFor(
    `document.querySelector('#reviewTransportNotice')?.textContent.includes(${JSON.stringify(expectedTransport)})`,
    `Game Review did not use the expected ${expectedTransport} transport.`,
    30000,
  );
  if (forceWorkerFailure) {
    assert.ok(
      await page.evaluate("globalThis.__browserSmokeForcedWorkerAttempts"),
      "the file:// scenario must actually reject Worker construction",
    );
  }

  await page.click("#reviewGameButton");
  await page.waitFor(
    "document.querySelector('#reviewProgressText')?.textContent.includes('Fast preview complete')",
    "Fast Game Review did not complete.",
  );
  assert.equal(
    await page.evaluate("document.querySelectorAll('#possessionTimeline .possession-badge').length"),
    EXPECTED_POSSESSIONS,
    "the terminal rebound action must remain one possession badge",
  );
  assert.equal(
    await page.evaluate("document.querySelector('#exportReviewButton').disabled"),
    false,
  );

  await page.evaluate(`(() => {
    globalThis.__browserSmokeReviewExports = [];
    const createObjectURL = URL.createObjectURL.bind(URL);
    URL.createObjectURL = function (blob) {
      blob.text().then((text) => globalThis.__browserSmokeReviewExports.push(text));
      return createObjectURL(blob);
    };
    const anchorClick = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () {
      if (this.hasAttribute("download")) return undefined;
      return anchorClick.call(this);
    };
  })()`);
  const beforeSandbox = await captureReviewExport(page);
  const liveSourceShaBefore = await page.evaluate(
    "document.querySelector('#gameReviewPanel').dataset.sourceReplaySha256",
  );
  assert.equal(liveSourceShaBefore, beforeSandbox.replaySha256);

  await page.click("#possessionTimeline .possession-badge:last-child");
  assert.equal(
    await page.evaluate("document.querySelector('#plyLabel').textContent"),
    "Move 11 of 11",
  );
  await page.click("#prevButton");
  assert.equal(
    await page.evaluate("document.querySelector('#plyLabel').textContent"),
    "Move 10 of 11",
    "each rebound edge must remain selectable in review mode",
  );
  assert.equal(
    await page.evaluate("document.querySelector('#possessionTimeline .possession-badge[aria-current=\"step\"]').textContent"),
    String(EXPECTED_POSSESSIONS),
    "edge navigation inside a rebound action must retain its possession grade",
  );
  await page.click("#prevButton");
  assert.equal(
    await page.evaluate("document.querySelector('#plyLabel').textContent"),
    "Move 9 of 11",
  );
  await page.click("#nextButton");
  assert.equal(
    await page.evaluate("document.querySelector('#plyLabel').textContent"),
    "Move 10 of 11",
    "edge-by-edge Next navigation must remain available in review mode",
  );
  await page.click("#possessionTimeline .possession-badge:first-child");
  assert.equal(
    await page.evaluate("document.querySelector('#plyLabel').textContent"),
    "Move 1 of 11",
  );

  await page.click("#tryLineButton");
  await page.waitFor(
    "!document.querySelector('#leaveTryLineButton').hidden && document.querySelector('#tryLineStatus').textContent.includes('start at the possession boundary')",
    "The try-line sandbox did not open at the reviewed boundary.",
    30000,
  );
  const boundaryDescription = await page.evaluate(
    "document.querySelector('#boardDescription').textContent",
  );
  assert.match(boundaryDescription, /Ball at row 6, column 4/u);
  assert.match(boundaryDescription, /source replay has not been changed/iu);
  assert.ok(
    await page.evaluate("document.querySelectorAll('#moveChoices .move-choice').length"),
    "the boundary-start sandbox must expose at least one legal edge",
  );
  await page.click("#moveChoices .move-choice:first-child");
  await page.waitFor(
    "!document.querySelector('#tryLineStatus').textContent.includes('start at the possession boundary') && !document.querySelector('#tryLineStatus').textContent.includes('Applying')",
    "The authoritative sandbox did not apply its legal edge.",
    30000,
  );
  assert.doesNotMatch(
    await page.evaluate("document.querySelector('#boardDescription').textContent"),
    /Ball at row 6, column 4/u,
    "the sandbox ball must advance after its legal edge",
  );
  assert.equal(
    await page.evaluate("document.querySelectorAll('#moveList .move-row').length"),
    TERMINAL_REPLAY.moves.length,
    "the sandbox must not replace the source replay's move list",
  );

  const afterSandbox = await captureReviewExport(page);
  const liveSourceShaAfter = await page.evaluate(
    "document.querySelector('#gameReviewPanel').dataset.sourceReplaySha256",
  );
  assert.equal(beforeSandbox.schema, "papersoccer.game-review.v1");
  assert.deepEqual(beforeSandbox.sourceReplay, TERMINAL_REPLAY);
  assert.deepEqual(afterSandbox.sourceReplay, TERMINAL_REPLAY);
  assert.equal(afterSandbox.replaySha256, beforeSandbox.replaySha256);
  assert.equal(
    liveSourceShaAfter,
    liveSourceShaBefore,
    "the live viewer's exact source replay identity must remain unchanged",
  );
  assert.equal(
    JSON.stringify(afterSandbox.sourceReplay),
    JSON.stringify(beforeSandbox.sourceReplay),
    "the sandbox must leave the exact source replay JSON unchanged",
  );

  await page.click("#leaveTryLineButton");
  assert.equal(
    await page.evaluate("document.querySelector('#plyLabel').textContent"),
    "Move 1 of 11",
    "leaving the sandbox must restore the original edge selection",
  );
  assert.deepEqual(
    await page.evaluate("globalThis.__browserSmokeErrors"),
    [],
    "the browser scenario must not raise an unhandled page error",
  );
  assert.deepEqual(dialogs, [], "the browser scenario must not open an error dialog");
}

test("Game Review works in hosted Worker and direct-file fallback browsers", {
  skip: LOCKED ? false :
    "validation-fitted calibration is not locked yet; the browser smoke becomes mandatory with the lock",
}, async () => {
  const executable = chromeExecutable();
  assert.ok(
    executable,
    "A locked Game Review requires Chrome/Chromium for browser smoke coverage. " +
      "Set PAPERSOCCER_CHROME to its executable if it is not in a standard location.",
  );

  const temporary = await mkdtemp(path.join(os.tmpdir(), "papersoccer-review-browser-"));
  let hosted = null;
  let chrome = null;
  let cleanupFailure = null;
  try {
    const replayPath = path.join(temporary, "short-terminal-replay.json");
    await writeFile(replayPath, JSON.stringify(TERMINAL_REPLAY, null, 2) + "\n");
    hosted = await startStaticServer();
    chrome = await launchChrome(executable, path.join(temporary, "chrome-profile"));

    const hostedPage = await openPage(chrome.connection);
    try {
      await exerciseScenario(
        hostedPage,
        hosted.url,
        replayPath,
        "separate browser worker",
        false,
      );
    } finally {
      await hostedPage.close().catch(() => {});
    }

    const filePage = await openPage(chrome.connection);
    try {
      await exerciseScenario(
        filePage,
        pathToFileURL(path.join(WEB_ROOT, "index.html")).href,
        replayPath,
        "Direct-file fallback",
        true,
      );
    } finally {
      await filePage.close().catch(() => {});
    }
  } finally {
    if (chrome) {
      try {
        await stopChrome(chrome.child);
      } catch (error) {
        cleanupFailure ??= error;
      }
    }
    if (hosted) {
      try {
        await stopStaticServer(hosted.server);
      } catch (error) {
        cleanupFailure ??= error;
      }
    }
    try {
      await rm(temporary, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 100,
      });
    } catch (error) {
      cleanupFailure ??= error;
    }
    if (cleanupFailure) {
      throw cleanupFailure;
    }
  }
});
