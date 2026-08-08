#!/usr/bin/env node

import crypto from "node:crypto";
import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

const DEPTHS = [4, 8, 12, 20];
const EXPECTED_EMSCRIPTEN = "6.0.2";
const INITIAL_MEMORY_BYTES = 67_108_864;
const REFERENCE_ID = "rank5-derived-fixed-50k";

function fail(message) {
  throw new Error(message);
}

function parseArguments(argv) {
  const result = {
    manifest: "benchmarks/game_review_gate/manifest.json",
    module: "web/papersoccer-analysis-wasm.js",
    nativeRoot: null,
    output: null,
    emscriptenVersion: EXPECTED_EMSCRIPTEN,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const option = argv[index];
    const value = argv[index + 1];
    if (option === "--manifest") {
      result.manifest = value;
    } else if (option === "--module") {
      result.module = value;
    } else if (option === "--native-root") {
      result.nativeRoot = value;
    } else if (option === "--output") {
      result.output = value;
    } else if (option === "--emscripten-version") {
      result.emscriptenVersion = value;
    } else if (option === "--help" || option === "-h") {
      process.stdout.write(
        "Usage: node benchmarks/game_review_gate/measure_wasm_latency.mjs " +
          "[--manifest PATH] [--module PATH] [--native-root PATH] " +
          "[--output PATH] --emscripten-version 6.0.2\n",
      );
      process.exit(0);
    } else {
      fail(`unknown or incomplete option: ${option}`);
    }
    index += 1;
  }
  return result;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function sha256(data) {
  return crypto.createHash("sha256").update(data).digest("hex");
}

function sha256File(file) {
  return sha256(fs.readFileSync(file));
}

function canonicalize(value) {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

function canonicalJson(value) {
  return `${JSON.stringify(canonicalize(value))}\n`;
}

function repositoryRelative(repository, target) {
  const relative = path.relative(repository, target).split(path.sep).join("/");
  if (!relative || relative === ".." || relative.startsWith("../")) {
    fail(`path is outside the repository: ${target}`);
  }
  return relative;
}

function nearestRank(values, probability) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.max(0, Math.ceil(probability * ordered.length) - 1)];
}

function parseOpeningBank(file, declaration) {
  const bytes = fs.readFileSync(file);
  if (sha256(bytes) !== declaration.sha256) {
    fail(`opening-bank SHA-256 mismatch: ${file}`);
  }
  const text = bytes.toString("utf8");
  if (text.includes("\r")) {
    fail(`opening bank does not use LF endings: ${file}`);
  }
  const lines = text.trimEnd().split("\n");
  const expectedHeader =
    "opening_id\tphase\tdepth\tgeneration_seed\tstate_hash\t" +
    "canonical_key\tto_move\tmoves";
  if (lines[0] !== "schema\tpapersoccer.opening-bank.v1" ||
      lines[11] !== expectedHeader) {
    fail(`opening-bank schema/header mismatch: ${file}`);
  }
  const records = lines.slice(12).map((line) => {
    const fields = line.split("\t");
    if (fields.length !== 8) {
      fail(`opening-bank row is malformed: ${file}`);
    }
    const moves = fields[7].split(";").map((encoded) => {
      const coordinates = encoded.split(",").map(Number);
      if (coordinates.length !== 2 || coordinates.some(Number.isNaN)) {
        fail(`opening-bank move is malformed: ${file}`);
      }
      return { x: coordinates[0], y: coordinates[1] };
    });
    return {
      id: fields[0],
      phase: fields[1],
      depth: Number(fields[2]),
      stateHash: fields[4],
      toMove: fields[6],
      moves,
    };
  });
  if (records.length !== declaration.pairs ||
      records.some((record) =>
        record.phase !== "validation" ||
        record.depth !== declaration.depth ||
        record.moves.length !== declaration.depth)) {
    fail(`opening-bank content differs from manifest: ${file}`);
  }
  return records;
}

function ccall(module, name, returnType, argumentTypes = [], values = []) {
  return module.ccall(name, returnType, argumentTypes, values);
}

function probeError(module) {
  return ccall(module, "ps_analysis_probe_last_error", "string") ||
    "analysis probe failed without an error message";
}

function startAndAppend(module, budget, opening, counts) {
  if (ccall(module, "ps_analysis_probe_start", "number", ["number"], [budget]) !== 1) {
    counts.incomplete_actions += 1;
    fail(`analysis probe start failed: ${probeError(module)}`);
  }
  for (const move of opening.moves) {
    if (ccall(
      module,
      "ps_analysis_probe_append_move",
      "number",
      ["number", "number"],
      [move.x, move.y],
    ) !== 1) {
      counts.illegal_moves += 1;
      fail(`authoritative opening replay failed for ${opening.id}: ${probeError(module)}`);
    }
  }
}

function parseProbeResult(module, opening, profile, counts) {
  const encoded = ccall(module, "ps_analysis_probe_result_json", "string");
  if (!encoded) {
    counts.incomplete_actions += 1;
    fail(`analysis probe result failed for ${opening.id}: ${probeError(module)}`);
  }
  const result = JSON.parse(encoded);
  const expectedProfileName = `deep-${profile.max_nodes / 1000}k`;
  const expectedBall = opening.moves.at(-1);
  if (result.schema !== "papersoccer.analysis-probe.v1" ||
      result.analyzerIdentity !== "complete-turn-analysis-fresh-search-v1" ||
      result.candidateIdentity !== profile.id ||
      result.profile?.identity !== expectedProfileName ||
      result.profile?.maxTurnDepth !== 32 ||
      result.profile?.maxNodes !== profile.max_nodes ||
      result.profile?.transpositionEntries !== 65_536 ||
      result.profile?.evaluationCacheEntries !== 32_768 ||
      result.profile?.wallClock !== false ||
      result.profile?.replayCorrections !== false ||
      result.profile?.learnedValueBlendPercent !== 0 ||
      result.openingPlies !== opening.depth ||
      result.position?.possessionBoundary !== true ||
      result.position?.status !== "inProgress" ||
      result.position?.toMove !== opening.toMove ||
      result.position?.ball?.x !== expectedBall.x ||
      result.position?.ball?.y !== expectedBall.y ||
      result.completeAction !== true ||
      !Array.isArray(result.action) || result.action.length === 0 ||
      !Number.isInteger(result.rootScore) ||
      result.diagnostics?.rootScore !== result.rootScore ||
      !Number.isInteger(result.diagnostics?.nodes) ||
      result.diagnostics.nodes <= 0 ||
      result.diagnostics.nodes > profile.max_nodes) {
    counts.incomplete_actions += 1;
    fail(`analysis probe contract mismatch for ${profile.id}/${opening.id}`);
  }
  return result;
}

function nativePosition(report, opening, pairIndex, profile) {
  if (report.schema !== "papersoccer.arena.v1" ||
      report.runtime !== "native" ||
      report.mode !== "matches" ||
      report.configuration?.candidate?.kind !== "deep-turn-search" ||
      report.configuration?.candidate?.max_nodes !== profile.max_nodes ||
      report.configuration?.reference?.kind !== "rank5-derived") {
    fail(`native parity report configuration mismatch for ${profile.id}`);
  }
  const reportedOpening = report.openings.find(
    (candidate) => candidate.pair_index === pairIndex,
  );
  if (!reportedOpening || reportedOpening.opening_id !== opening.id ||
      reportedOpening.state_hash !== opening.stateHash ||
      reportedOpening.actual_plies !== opening.depth) {
    fail(`native parity opening mismatch for ${profile.id}/${opening.id}`);
  }
  const playerField = reportedOpening.state.to_move === "one"
    ? "player_one"
    : "player_two";
  const game = report.games.find(
    (candidate) =>
      candidate.pair_index === pairIndex &&
      candidate[playerField]?.bot === "candidate",
  );
  if (!game || !Array.isArray(game.decisions) || game.decisions.length === 0) {
    fail(`native candidate-to-move game is missing for ${profile.id}/${opening.id}`);
  }
  const decision = game.decisions[0];
  const diagnostics = decision.deep_turn_search;
  if (decision.bot !== "candidate" || decision.ply !== opening.depth + 1 ||
      diagnostics?.cached_continuation !== false ||
      diagnostics?.current_edge_index !== 0 ||
      diagnostics?.profile_node_budget !== profile.max_nodes) {
    fail(`native fresh-search diagnostic mismatch for ${profile.id}/${opening.id}`);
  }
  const length = diagnostics.planned_action_length;
  const actionDecisions = game.decisions.slice(0, length);
  if (!Number.isInteger(length) || length < 1 || actionDecisions.length !== length ||
      actionDecisions.some((edge, edgeIndex) =>
        edge.bot !== "candidate" || edge.legal !== true ||
        edge.deep_turn_search?.current_edge_index !== edgeIndex ||
        edge.deep_turn_search?.planned_action_length !== length)) {
    fail(`native complete action is missing for ${profile.id}/${opening.id}`);
  }
  return {
    action: actionDecisions.map(({ to }) => to),
    rootScore: diagnostics.root_score,
    diagnostics,
  };
}

const DIAGNOSTIC_KEYS = [
  ["completedDepth", "completed_turn_depth"],
  ["attemptedDepth", "attempted_turn_depth"],
  ["nodes", "visited_nodes"],
  ["leafEvaluations", "leaf_evaluations"],
  ["terminalNodes", "terminal_nodes"],
  ["completedActions", "completed_actions"],
  ["cutoffs", "cutoffs"],
  ["transpositionProbes", "transposition_probes"],
  ["transpositionHits", "transposition_hits"],
  ["transpositionCutoffs", "transposition_cutoffs"],
  ["transpositionStores", "transposition_stores"],
  ["continuationTranspositionHits", "continuation_transposition_hits"],
  ["evaluationCacheProbes", "evaluation_cache_probes"],
  ["evaluationCacheHits", "evaluation_cache_hits"],
  ["terminalBoundCutoffs", "terminal_bound_cutoffs"],
  ["forcedEdges", "forced_edges"],
  ["rootSeedActions", "root_seed_actions"],
  ["rootTranspositionReuses", "root_transposition_reuses"],
  ["maxActionEdges", "max_action_edges"],
  ["rootScore", "root_score"],
  ["budgetExhausted", "budget_exhausted"],
];

function checkNativeParity(probe, native, profile, opening, counts) {
  const wasmTranscript = {
    action: probe.action,
    root_score: probe.rootScore,
    diagnostics: {},
  };
  const nativeTranscript = {
    action: native.action,
    root_score: native.rootScore,
    diagnostics: {},
  };
  for (const [wasmKey, nativeKey] of DIAGNOSTIC_KEYS) {
    wasmTranscript.diagnostics[nativeKey === "visited_nodes" ? "nodes" : nativeKey] =
      probe.diagnostics[wasmKey];
    nativeTranscript.diagnostics[nativeKey === "visited_nodes" ? "nodes" : nativeKey] =
      native.diagnostics[nativeKey];
  }
  if (canonicalJson(wasmTranscript) !== canonicalJson(nativeTranscript)) {
    counts.parity_failures += 1;
    fail(`native/Wasm transcript parity failed for ${profile.id}/${opening.id}`);
  }
  return wasmTranscript;
}

function runProbe(module, profile, opening, native, counts, timed) {
  startAndAppend(module, profile.max_nodes, opening, counts);
  let status;
  let elapsed = null;
  if (timed) {
    const started = performance.now();
    status = ccall(module, "ps_analysis_probe_run", "number");
    elapsed = performance.now() - started;
  } else {
    status = ccall(module, "ps_analysis_probe_run", "number");
  }
  if (status !== 1) {
    counts.incomplete_actions += 1;
    fail(`analysis probe search failed for ${opening.id}: ${probeError(module)}`);
  }
  const result = parseProbeResult(module, opening, profile, counts);
  const parityTranscript = native
    ? checkNativeParity(result, native, profile, opening, counts)
    : null;
  return { result, elapsed, parityTranscript };
}

async function loadFactory(modulePath) {
  const namespace = await import(pathToFileURL(modulePath).href);
  const factory = namespace.default?.default ?? namespace.default;
  if (typeof factory !== "function") {
    fail(`analysis artifact does not export an Emscripten module factory: ${modulePath}`);
  }
  return factory;
}

function emptyCounts() {
  return {
    illegal_moves: 0,
    incomplete_actions: 0,
    unexplained_truncations: 0,
    parity_failures: 0,
  };
}

function requireFixedHeap(module) {
  const heapBytes = ccall(module, "ps_review_heap_bytes", "number");
  if (heapBytes !== INITIAL_MEMORY_BYTES) {
    fail(
      `analysis module heap is ${heapBytes}; ` +
        `expected fixed ${INITIAL_MEMORY_BYTES}`,
    );
  }
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  const manifestPath = path.resolve(args.manifest);
  const repository = path.resolve(path.dirname(manifestPath), "../..");
  const manifestBytes = fs.readFileSync(manifestPath);
  const manifest = JSON.parse(manifestBytes.toString("utf8"));
  const manifestSha256 = sha256(manifestBytes);
  let emscriptenVersionOutput;
  try {
    emscriptenVersionOutput = childProcess.execFileSync(
      process.env.EMCC || "emcc",
      ["--version"],
      { encoding: "utf8" },
    );
  } catch (error) {
    fail(`could not verify the pinned Emscripten compiler: ${error.message}`);
  }
  if (manifest.schema_version !== "papersoccer.game-review-gate-manifest.v1" ||
      args.emscriptenVersion !== EXPECTED_EMSCRIPTEN ||
      !new RegExp(`\\b${EXPECTED_EMSCRIPTEN.replaceAll(".", "\\.")}(?:-git)?\\b`, "u")
        .test(emscriptenVersionOutput) ||
      manifest.latency_protocol?.emscripten_version !== EXPECTED_EMSCRIPTEN ||
      manifest.latency_protocol?.initial_memory_bytes !== INITIAL_MEMORY_BYTES ||
      manifest.latency_protocol?.memory_growth !== false ||
      manifest.latency_protocol?.samples_per_opening_depth_per_candidate !== 20 ||
      manifest.latency_protocol?.warmup_searches_per_candidate !== 8) {
    fail("manifest or Emscripten latency protocol differs from the frozen contract");
  }
  const profiles = manifest.profiles.candidates;
  if (JSON.stringify(profiles.map(({ max_nodes: nodes }) => nodes)) !==
      JSON.stringify([100_000, 200_000, 400_000])) {
    fail("candidate node budgets differ from the frozen contract");
  }
  const modulePath = path.resolve(args.module);
  if (repositoryRelative(repository, modulePath) !==
      "web/papersoccer-analysis-wasm.js") {
    fail("latency must use the dedicated checked-in analysis artifact");
  }
  const rawRoot = args.nativeRoot
    ? path.resolve(args.nativeRoot)
    : path.join(
      repository,
      manifest.outputs.raw_results_root,
      manifestSha256,
      "validation",
      "shards",
    );
  const output = args.output
    ? path.resolve(args.output)
    : path.join(
      repository,
      manifest.outputs.raw_results_root,
      manifestSha256,
      "wasm_latency_input.json",
    );
  const validationResult = readJson(path.join(
    repository,
    manifest.outputs.phase_results.validation,
  ));
  const competitionSource = validationResult.source?.competition_source;
  if (validationResult.schema !== "papersoccer.game-review-gate-phase.v1" ||
      validationResult.phase !== "validation" ||
      validationResult.manifest_sha256 !== manifestSha256 ||
      competitionSource?.schema !==
        "papersoccer.game-review-competition-source.v1" ||
      !/^[0-9a-f]{64}$/u.test(competitionSource.sha256 ?? "")) {
    fail("validation result is missing its frozen competition-source identity");
  }

  const openingsByDepth = new Map();
  for (const depth of DEPTHS) {
    const declaration = manifest.openings.banks.find(
      (bank) => bank.phase === "validation" && bank.depth === depth,
    );
    if (!declaration) {
      fail(`validation opening bank is missing for depth ${depth}`);
    }
    openingsByDepth.set(
      depth,
      parseOpeningBank(path.join(repository, declaration.path), declaration),
    );
  }

  const nativeReports = new Map();
  const nativeHashes = new Map();
  for (const profile of profiles) {
    for (const depth of DEPTHS) {
      const unit = `validation--${profile.id}-vs-${REFERENCE_ID}--d${
        String(depth).padStart(2, "0")}`;
      const file = path.join(rawRoot, `${unit}.json`);
      const report = readJson(file);
      if (report.game_review_gate?.manifest_sha256 !== manifestSha256 ||
          report.game_review_gate?.unit_id !== unit ||
          canonicalJson(report.game_review_gate?.competition_source) !==
            canonicalJson(competitionSource)) {
        fail(`native parity shard identity mismatch: ${file}`);
      }
      nativeReports.set(`${profile.id}:${depth}`, report);
      nativeHashes.set(`${profile.id}:${depth}`, sha256File(file));
    }
  }

  const factory = await loadFactory(modulePath);
  const discoveryModule = await factory();
  requireFixedHeap(discoveryModule);
  const discoveryProfile = profiles[0];
  const selectedOpenings = [];
  for (const depth of DEPTHS) {
    const records = openingsByDepth.get(depth);
    let accepted = 0;
    for (let pairIndex = 0; pairIndex < records.length && accepted < 20; pairIndex += 1) {
      const opening = { ...records[pairIndex], pairIndex };
      const counts = emptyCounts();
      startAndAppend(discoveryModule, discoveryProfile.max_nodes, opening, counts);
      const status = ccall(discoveryModule, "ps_analysis_probe_run", "number");
      if (status !== 1) {
        const error = probeError(discoveryModule);
        if (error.includes("possession-boundary")) {
          continue;
        }
        fail(`opening discovery failed for ${opening.id}: ${error}`);
      }
      parseProbeResult(discoveryModule, opening, discoveryProfile, counts);
      selectedOpenings.push(opening);
      accepted += 1;
    }
    if (accepted !== 20) {
      fail(`depth ${depth} has only ${accepted} usable possession-boundary openings`);
    }
  }

  const profileResults = {};
  for (const profile of profiles) {
    const module = await factory();
    requireFixedHeap(module);
    const counts = emptyCounts();
    const warmups = DEPTHS.flatMap((depth) =>
      selectedOpenings.filter((opening) => opening.depth === depth).slice(0, 2));
    for (const opening of warmups) {
      const report = nativeReports.get(`${profile.id}:${opening.depth}`);
      const native = nativePosition(report, opening, opening.pairIndex, profile);
      runProbe(module, profile, opening, native, counts, false);
    }

    const samples = [];
    const parity = [];
    for (const opening of selectedOpenings) {
      const report = nativeReports.get(`${profile.id}:${opening.depth}`);
      const native = nativePosition(report, opening, opening.pairIndex, profile);
      const { elapsed, parityTranscript } = runProbe(
        module,
        profile,
        opening,
        native,
        counts,
        true,
      );
      samples.push(elapsed);
      parity.push({
        sample_id: opening.id,
        action_sha256: sha256(canonicalJson(parityTranscript.action)),
        transcript_sha256: sha256(canonicalJson(parityTranscript)),
        wasm_transcript: parityTranscript,
      });
    }
    if (Object.values(counts).some((count) => count !== 0)) {
      fail(`operational failures occurred while measuring ${profile.id}`);
    }
    profileResults[profile.id] = {
      profile_sha256: sha256(canonicalJson(profile)),
      node_budget: profile.max_nodes,
      sample_ids: selectedOpenings.map(({ id }) => id),
      opening_depths: selectedOpenings.map(({ depth }) => depth),
      samples_ms: samples,
      timing_ms: {
        samples: samples.length,
        median: nearestRank(samples, 0.5),
        p95: nearestRank(samples, 0.95),
        maximum: Math.max(...samples),
      },
      native_shard_sha256: Object.fromEntries(
        DEPTHS.map((depth) => [
          String(depth),
          nativeHashes.get(`${profile.id}:${depth}`),
        ]),
      ),
      parity,
      operational_counts: counts,
    };
  }

  const cpu = os.cpus();
  const artifact = {
    schema: "papersoccer.game-review-gate-wasm-latency.v1",
    manifest_sha256: manifestSha256,
    competition_source: competitionSource,
    module: {
      path: repositoryRelative(repository, modulePath),
      sha256: sha256File(modulePath),
      emscripten_version: args.emscriptenVersion,
      initial_memory_bytes: INITIAL_MEMORY_BYTES,
      memory_growth: false,
    },
    environment: {
      runtime: "node-webassembly",
      node_version: process.version,
      v8_version: process.versions.v8,
      platform: process.platform,
      architecture: process.arch,
      cpu_model: cpu[0]?.model || "unknown",
      logical_cpus: cpu.length,
      total_memory_bytes: os.totalmem(),
      timer: "performance.now",
      warmup_searches_per_candidate: 8,
    },
    sample_source: {
      phase: "validation",
      opening_depths: DEPTHS,
      fresh_possession_boundaries_only: true,
      samples_per_opening_depth_per_candidate: 20,
      native_parity_reference: REFERENCE_ID,
      native_raw_root: repositoryRelative(repository, rawRoot),
    },
    profiles: profileResults,
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(artifact, null, 2)}\n`, { flag: "wx" });
  process.stdout.write(`${JSON.stringify({
    output: repositoryRelative(repository, output),
    sha256: sha256File(output),
    samples_per_candidate: selectedOpenings.length,
  }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`error: ${error.message}\n`);
  process.exitCode = 2;
});
