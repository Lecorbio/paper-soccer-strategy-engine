(function nativeStatusPage(root) {
  "use strict";

  const EXPECTED_SCHEMA = "papersoccer.jacek-native-status.v1";

  function validate(ledger) {
    if (!ledger || ledger.schema !== EXPECTED_SCHEMA) {
      throw new Error("Unsupported or missing native status ledger.");
    }
    for (const key of [
      "candidate", "architecture", "provenance", "training", "clock",
      "verification", "live", "links",
    ]) {
      if (!ledger[key] || typeof ledger[key] !== "object") {
        throw new Error(`Native status ledger is missing ${key}.`);
      }
    }
    if (!Array.isArray(ledger.artifacts) || !Array.isArray(ledger.diagnostics) ||
        !Array.isArray(ledger.gates) ||
        !Array.isArray(ledger.verification.checks)) {
      throw new Error("Native status ledger collections are incomplete.");
    }
    for (const key of [
      "searchAction", "actionCap", "dequeSchedule", "network", "search",
      "productionLimits", "tacticalWitness", "trainingProvenance",
      "dependencyBoundary",
    ]) {
      if (ledger.architecture[key] === undefined || ledger.architecture[key] === null) {
        throw new Error(`Native architecture ledger is missing ${key}.`);
      }
    }
    if (typeof ledger.training.provenance !== "string" ||
        ledger.training.provenance.length === 0) {
      throw new Error("Native training provenance is incomplete.");
    }
    for (const [key, value] of Object.entries(ledger.provenance)) {
      if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
        throw new Error(`Native provenance identity ${key} is incomplete.`);
      }
    }
    for (const artifact of ledger.artifacts) {
      if (typeof artifact.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(artifact.sha256)) {
        throw new Error("Native artifact identity is incomplete.");
      }
    }
    for (const diagnostic of ledger.diagnostics) {
      for (const key of ["candidateRuntimeSha256", "referenceRuntimeSha256"]) {
        if (typeof diagnostic[key] !== "string" ||
            !/^[0-9a-f]{64}$/.test(diagnostic[key])) {
          throw new Error(`Native diagnostic identity ${key} is incomplete.`);
        }
      }
    }
    return ledger;
  }

  function statusClass(status) {
    if (status === "passed") return "status-passed";
    if (status === "failed") return "status-failed";
    if (status === "in-progress") return "status-in-progress";
    return "status-pending";
  }

  function statusLabel(status) {
    return String(status || "pending").replaceAll("-", " ");
  }

  function formatInteger(value) {
    return Number(value).toLocaleString("en-US");
  }

  function createElement(doc, tag, className, text) {
    const element = doc.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function definition(doc, term, value) {
    const wrapper = createElement(doc, "div");
    wrapper.append(createElement(doc, "dt", "", term));
    wrapper.append(createElement(doc, "dd", "", value));
    return wrapper;
  }

  function setFields(doc, values) {
    for (const [name, value] of Object.entries(values)) {
      for (const element of doc.querySelectorAll(`[data-field="${name}"]`)) {
        element.textContent = value;
      }
    }
  }

  function render(ledgerValue, doc) {
    const ledger = validate(ledgerValue);
    setFields(doc, {
      claim: ledger.candidate.claim,
      stage: ledger.candidate.stageLabel,
      updatedAt: ledger.updatedAt,
      candidateName: ledger.candidate.name,
      trainingGames: formatInteger(ledger.training.games),
      clockProfile: `${ledger.clock.search.firstMs} / ${ledger.clock.search.laterMs} ms`,
      liveLabel: ledger.live.label,
    });
    for (const element of doc.querySelectorAll('[data-field="stage"]')) {
      element.className = `status-pill ${statusClass(ledger.candidate.status)}`;
    }

    const architecture = doc.getElementById("architectureGrid");
    if (architecture) {
      const labels = {
        searchAction: "Search action",
        actionCap: "Maximum actions",
        dequeSchedule: "Deque schedule",
        network: "Value network",
        search: "Tree search",
        productionLimits: "Production limits",
        tacticalWitness: "Tactical discovery",
        trainingProvenance: "Training provenance",
        dependencyBoundary: "Dependency boundary",
      };
      architecture.replaceChildren(...Object.entries(labels).map(([key, label]) =>
        definition(doc, label, String(ledger.architecture[key]))));
    }

    const artifactRows = doc.getElementById("artifactRows");
    if (artifactRows) {
      artifactRows.replaceChildren(...ledger.artifacts.map((artifact) => {
        const row = createElement(doc, "tr");
        row.append(createElement(doc, "th", "", artifact.label));
        row.lastChild.scope = "row";
        row.append(createElement(
          doc, "td", "", `${formatInteger(artifact.size)} ${artifact.sizeUnit}`));
        row.append(createElement(doc, "td", "", artifact.sha256));
        return row;
      }));
    }

    const training = doc.getElementById("trainingDetails");
    if (training) {
      training.replaceChildren(
        definition(doc, "Self-play run", `${formatInteger(ledger.training.games)} games / ${ledger.training.shards} shards · ${ledger.training.runSeconds.toFixed(3)} s`),
        definition(doc, "Openings", `${ledger.training.openingDepths.join(", ")} complete turns`),
        definition(doc, "Later league rounds", `${ledger.training.leagueOpeningDepths.join(", ")} complete turns`),
        definition(doc, "Sampling", `Temperature through turn ${ledger.training.temperatureTurns}`),
        definition(doc, "Provenance", ledger.training.provenance),
        definition(doc, "Whole-game split", `${formatInteger(ledger.training.splitGames.train)} / ${formatInteger(ledger.training.splitGames.validation)} / ${formatInteger(ledger.training.splitGames.test)} games`),
        definition(doc, "Retained rows", `${formatInteger(ledger.training.trainSamples)} / ${formatInteger(ledger.training.validationSamples)} / ${formatInteger(ledger.training.testSamples)}`),
        definition(doc, "Overlap removals", `${formatInteger(ledger.training.overlapsRemoved.train)} / ${formatInteger(ledger.training.overlapsRemoved.validation)} / ${formatInteger(ledger.training.overlapsRemoved.test)}`),
        definition(doc, "Quantized train", `MSE ${ledger.training.trainMse.toFixed(4)} · ${(ledger.training.trainSignAccuracy * 100).toFixed(1)}% sign`),
        definition(doc, "Quantized validation", `MSE ${ledger.training.validationMse.toFixed(4)} · ${(ledger.training.validationSignAccuracy * 100).toFixed(1)}% sign`),
        definition(doc, "Quantized test", `MSE ${ledger.training.testMse.toFixed(4)} · ${(ledger.training.testSignAccuracy * 100).toFixed(1)}% sign`),
        definition(doc, "Training resources", `${ledger.training.trainingSeconds.toFixed(2)} s · ${formatInteger(ledger.training.peakRssBytes)} bytes peak RSS`),
      );
    }
    const warning = doc.getElementById("trainingWarning");
    if (warning) warning.textContent = ledger.training.warning;

    const clock = doc.getElementById("clockDetails");
    if (clock) {
      const probe = ledger.clock.timingProbe;
      clock.replaceChildren(
        definition(doc, "Search budgets", `${ledger.clock.search.firstMs} / ${ledger.clock.search.laterMs} ms`),
        definition(doc, "Pre-upload ceilings", `${ledger.clock.headroom.firstMs} / ${ledger.clock.headroom.laterMs} ms`),
        definition(doc, "Official limits", `${ledger.clock.official.firstMs} / ${ledger.clock.official.laterMs} ms`),
        definition(doc, "Fresh-process probe", `P0 ${probe.playerZero.firstMs.toFixed(1)} / ${probe.playerZero.laterMs.toFixed(1)} · P1 ${probe.playerOne.firstMs.toFixed(1)} / ${probe.playerOne.laterMs.toFixed(1)} ms`),
        definition(doc, "P0 memory", `${formatInteger(probe.playerZero.maxRssBytes)} bytes RSS · ${formatInteger(probe.playerZero.peakFootprintBytes)} footprint`),
        definition(doc, "P1 memory", `${formatInteger(probe.playerOne.maxRssBytes)} bytes RSS · ${formatInteger(probe.playerOne.peakFootprintBytes)} footprint`),
        definition(doc, "Decisive bootstrap max", `${ledger.clock.decisiveBootstrap.candidateFirstMaxMs.toFixed(3)} / ${ledger.clock.decisiveBootstrap.candidateLaterMaxMs.toFixed(3)} ms · ${ledger.clock.decisiveBootstrap.headroomFailures} failures`),
        definition(doc, "External gate max", `${ledger.clock.externalDevelopment.candidateFirstMaxMs.toFixed(3)} / ${ledger.clock.externalDevelopment.candidateLaterMaxMs.toFixed(3)} ms · ${ledger.clock.externalDevelopment.headroomFailures} failure`),
      );
    }

    const verificationStatus = doc.getElementById("verificationStatus");
    if (verificationStatus) {
      verificationStatus.className = `status-pill ${statusClass(ledger.verification.status)}`;
      verificationStatus.textContent = statusLabel(ledger.verification.status);
    }
    const verificationList = doc.getElementById("verificationList");
    if (verificationList) {
      verificationList.replaceChildren(...ledger.verification.checks.map((check) => {
        const item = createElement(doc, "li");
        item.append(createElement(doc, "span", "", check.label));
        item.append(createElement(
          doc, "span", `status-pill ${statusClass(check.status)}`, statusLabel(check.status)));
        return item;
      }));
    }

    const gateGrid = doc.getElementById("gateGrid");
    if (gateGrid) {
      gateGrid.replaceChildren(...[...ledger.diagnostics, ...ledger.gates].map((gate) => {
        const card = createElement(doc, "article", "gate-card");
        card.append(createElement(doc, "span", `status-pill ${statusClass(gate.status)}`, statusLabel(gate.status)));
        card.append(createElement(doc, "h3", "", gate.label));
        card.append(createElement(doc, "p", "", gate.requirement));
        card.append(createElement(doc, "p", "gate-result", gate.result));
        return card;
      }));
    }

    const liveStatus = doc.getElementById("liveStatus");
    if (liveStatus) {
      liveStatus.className = `status-pill ${statusClass(ledger.live.status)}`;
      liveStatus.textContent = statusLabel(ledger.live.status);
    }
    const readme = doc.getElementById("readmeLink");
    const experiments = doc.getElementById("experimentsLink");
    const auditor = doc.getElementById("auditorLink");
    if (readme) readme.href = ledger.links.readme;
    if (experiments) experiments.href = ledger.links.experiments;
    if (auditor) auditor.href = ledger.links.auditor;
    return true;
  }

  async function load(doc, fetchFunction) {
    try {
      const response = await fetchFunction("status.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`Status request failed: ${response.status}`);
      render(await response.json(), doc);
    } catch (error) {
      const message = doc.getElementById("loadError");
      if (message) message.hidden = false;
      if (root.console && typeof root.console.error === "function") {
        root.console.error(error);
      }
    }
  }

  root.PaperSoccerJacekNativeStatus = { validate, render, statusClass };
  if (root.document && typeof root.fetch === "function") {
    root.addEventListener("DOMContentLoaded", () => load(root.document, root.fetch));
  }
})(globalThis);
