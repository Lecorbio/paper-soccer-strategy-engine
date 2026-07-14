import { performance } from "node:perf_hooks";

import createPaperSoccerModule from "../web/papersoccer-wasm.js";

globalThis.createPaperSoccerModule = createPaperSoccerModule;
await import("../web/game-engine.js");

const { BotKind, Player, ready } = globalThis.PaperSoccer;
const engine = await ready;

function positiveInteger(raw, fallback, label) {
  if (raw === undefined) {
    return fallback;
  }
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return value;
}

const iterations = positiveInteger(process.argv[2], 2000, "iterations");
const samples = positiveInteger(process.argv[3], 9, "samples");

function measure(seed) {
  const snapshot = engine.startGame(
    String(seed),
    Player.Two,
    BotKind.Mcts,
    iterations,
  );
  const started = performance.now();
  engine.playBot(snapshot.sessionId, snapshot.revision);
  return performance.now() - started;
}

// Exclude module/JIT warm-up from the reported samples.
measure(0x5eed);

const milliseconds = [];
for (let sample = 0; sample < samples; sample += 1) {
  milliseconds.push(measure(0x5eed + sample + 1));
}
milliseconds.sort((left, right) => left - right);

function percentile(values, fraction) {
  const index = Math.ceil(fraction * values.length) - 1;
  return values[Math.max(0, Math.min(index, values.length - 1))];
}

const medianMilliseconds = percentile(milliseconds, 0.5);
const report = {
  schema: "papersoccer.wasm-benchmark.v1",
  position: "initial",
  iterations,
  samples,
  medianMilliseconds,
  p95Milliseconds: percentile(milliseconds, 0.95),
  iterationsPerSecond: iterations / (medianMilliseconds / 1000),
  milliseconds,
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
