(function benchmarkOverview(root) {
  "use strict";

  const EXPECTED_SCHEMA = "papersoccer.benchmark-summary.v1";
  const FAMILY_COLORS = ["#137c6a", "#2b70a1", "#cc6a18", "#9a4e77"];

  function formatPercent(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0 || numeric > 1) {
      throw new TypeError("Percentage must be a finite number in [0, 1].");
    }

    // Convert the proportion to percentage tenths, then round an exact halfway
    // case toward the even neighbor. The tolerance only absorbs IEEE-754 noise
    // introduced by decimal parsing and reverse-matchup subtraction.
    const scaled = numeric * 1000;
    const lower = Math.floor(scaled);
    const fraction = scaled - lower;
    const tieTolerance = Number.EPSILON * Math.max(1, Math.abs(scaled)) * 4;
    const roundedTenths = Math.abs(fraction - 0.5) <= tieTolerance
      ? (Math.abs(lower % 2) === 0 ? lower : lower + 1)
      : Math.round(scaled);
    return `${(roundedTenths / 10).toFixed(1)}%`;
  }

  function formatInteger(value) {
    return Number(value).toLocaleString("en-US");
  }

  function formatAbility(value) {
    const numeric = Number(value);
    return `${numeric > 0 ? "+" : ""}${numeric.toFixed(3)}`;
  }

  function formatLatency(value) {
    return `${Number(value).toFixed(1)} ms`;
  }

  function matchupDisplay(matchup) {
    return {
      score: formatPercent(matchup.leftScore),
      intervalLower: formatPercent(matchup.leftScoreLower),
      intervalUpper: formatPercent(matchup.leftScoreUpper)
    };
  }

  function validationStrengthLabel(candidate) {
    const score = formatPercent(candidate.strength);
    return candidate.strengthIsReference ? `${score} defined reference` : score;
  }

  function validationIntervalLabel(candidate) {
    return candidate.strengthLower === null
      ? "Not observed"
      : `${formatPercent(candidate.strengthLower)}–${formatPercent(candidate.strengthUpper)}`;
  }

  function invertMatchup(matchup) {
    if (!matchup || typeof matchup !== "object") {
      return null;
    }
    return Object.assign({}, matchup, {
      leftId: matchup.rightId,
      rightId: matchup.leftId,
      leftScore: 1 - Number(matchup.leftScore),
      leftScoreLower: 1 - Number(matchup.leftScoreUpper),
      leftScoreUpper: 1 - Number(matchup.leftScoreLower)
    });
  }

  function pairwiseLookup(matchups, leftId, rightId) {
    const records = Array.isArray(matchups) ? matchups : Object.values(matchups || {});
    const direct = records.find((matchup) =>
      matchup.leftId === leftId && matchup.rightId === rightId);
    if (direct) {
      return direct;
    }
    const reverse = records.find((matchup) =>
      matchup.leftId === rightId && matchup.rightId === leftId);
    return reverse ? invertMatchup(reverse) : null;
  }

  function matchupStatus(matchup) {
    if (!matchup) {
      return "Not available";
    }
    if (matchup.classification === "statistically_unresolved") {
      return "Statistically unresolved";
    }
    return matchup.strongerId === matchup.leftId ? "Resolved lead" : "Resolved deficit";
  }

  function matchupAccessibleLabel(leftLabel, rightLabel, matchup) {
    const display = matchupDisplay(matchup);
    return `${leftLabel} against ${rightLabel}: ${display.score}, 95% interval ` +
      `${display.intervalLower} to ${display.intervalUpper}. ${matchupStatus(matchup)}.`;
  }

  function statusClass(matchup) {
    if (!matchup || matchup.classification === "statistically_unresolved") {
      return "status-unresolved";
    }
    return matchup.strongerId === matchup.leftId ? "status-lead" : "status-deficit";
  }

  function createElement(doc, tag, className, text) {
    const element = doc.createElement(tag);
    if (className) {
      element.className = className;
    }
    if (text !== undefined) {
      element.textContent = text;
    }
    return element;
  }

  function setPercent(element, property, value) {
    const bounded = Math.max(0, Math.min(100, value));
    element.style.setProperty(property, `${bounded.toFixed(2)}%`);
  }

  function position(value, minimum, maximum) {
    if (maximum === minimum) {
      return 50;
    }
    return ((value - minimum) / (maximum - minimum)) * 100;
  }

  function uniqueNumbers(values) {
    return values.filter((value, index) =>
      values.findIndex((candidate) => Math.abs(candidate - value) < 0.000001) === index);
  }

  function niceStep(value) {
    if (!Number.isFinite(value) || value <= 0) {
      return 1;
    }
    const magnitude = 10 ** Math.floor(Math.log10(value));
    const fraction = value / magnitude;
    const niceFraction = fraction <= 1
      ? 1
      : fraction <= 2
        ? 2
        : fraction <= 2.5
          ? 2.5
          : fraction <= 5 ? 5 : 10;
    return niceFraction * magnitude;
  }

  function axisNumber(value) {
    return String(Number(Number(value).toPrecision(12)));
  }

  function axisPercent(value) {
    const percentage = Number(Number(value * 100).toPrecision(12));
    return `${percentage}%`;
  }

  function niceAxis(maximum, targetIntervals, formatter) {
    const intervalTarget = Math.max(1, Math.round(targetIntervals));
    const step = niceStep(maximum / intervalTarget);
    const intervalCount = Math.max(1, Math.ceil((maximum / step) - 0.000000001));
    const niceMaximum = Number((step * intervalCount).toPrecision(12));
    const ticks = Array.from({length: intervalCount + 1}, (_, index) => {
      const value = Number((step * index).toPrecision(12));
      return {
        value,
        label: formatter(value),
        position: (index / intervalCount) * 100,
        anchor: index === 0 ? "start" : index === intervalCount ? "end" : "middle"
      };
    });
    return {maximum: niceMaximum, step: Number(step.toPrecision(12)), ticks};
  }

  function paretoAxisModel(results) {
    const candidates = Array.isArray(results.validationCandidates)
      ? results.validationCandidates
      : [];
    const maximumLatency = candidates.length > 0
      ? Math.max(...candidates.map((candidate) => Number(candidate.p95LatencyMs)))
      : Number(results.study.latencyGateMs);
    const maximumStrength = candidates.length > 0
      ? Math.max(...candidates.map((candidate) => Number(candidate.strength)))
      : 0.6;
    const xAxis = niceAxis(
      Math.max(Number(results.study.latencyGateMs) * 1.2, maximumLatency),
      6,
      axisNumber
    );
    const yAxis = niceAxis(Math.max(0.6, maximumStrength), 6, axisPercent);
    return {
      xMaximum: xAxis.maximum,
      yMaximum: yAxis.maximum,
      xTicks: xAxis.ticks,
      yTicks: yAxis.ticks
    };
  }

  function validateResults(results) {
    if (!results || results.schema !== EXPECTED_SCHEMA) {
      throw new Error("Unsupported or missing benchmark summary.");
    }
    for (const key of ["entrants", "matchups", "validationCandidates", "calibration"]) {
      if (!Array.isArray(results[key])) {
        throw new Error(`Benchmark summary is missing ${key}.`);
      }
    }
    if (!results.study || !results.links || !results.caveats) {
      throw new Error("Benchmark summary is incomplete.");
    }
  }

  function renderLinks(doc, links) {
    for (const id of ["heroReportLink", "reportLink"]) {
      const link = doc.getElementById(id);
      if (link && typeof links.report === "string") {
        link.href = links.report;
      }
    }
  }

  function renderStudySummary(doc, results) {
    const headline = doc.getElementById("benchmarkHeadline");
    if (headline) {
      headline.textContent = results.study.headline;
    }

    const stats = doc.getElementById("benchmarkStats");
    if (!stats) {
      return;
    }
    const definitions = [
      ["Bots compared", formatInteger(results.study.entrantCount)],
      ["Decisive games", formatInteger(results.study.games)],
      ["Color-swapped pairs", formatInteger(results.study.pairs)],
      ["Latency gate", `${results.study.latencyGateMs} ms p95`]
    ];
    const cards = definitions.map(([term, value], index) => {
      const card = createElement(doc, "div", "stat-card");
      const dt = createElement(doc, "dt", "", term);
      const dd = createElement(doc, "dd");
      if (index === 3) {
        dd.append(doc.createTextNode(`${results.study.latencyGateMs} `));
        dd.append(createElement(doc, "span", "", "ms p95"));
      } else {
        dd.textContent = value;
      }
      card.append(dt, dd);
      return card;
    });
    stats.replaceChildren(...cards);
  }

  function renderAbility(doc, entrants) {
    const chart = doc.getElementById("abilityChart");
    if (!chart) {
      return;
    }
    const ordered = [...entrants].sort((left, right) =>
      right.bradleyTerry.estimate - left.bradleyTerry.estimate);
    const lower = Math.min(0, ...ordered.map((entrant) => entrant.bradleyTerry.lower));
    const upper = Math.max(0, ...ordered.map((entrant) => entrant.bradleyTerry.upper));
    const padding = Math.max(0.08, (upper - lower) * 0.08);
    const domainMinimum = lower - padding;
    const domainMaximum = upper + padding;
    const zeroPosition = position(0, domainMinimum, domainMaximum);

    const axisRow = createElement(doc, "div", "ability-axis-row");
    axisRow.setAttribute("aria-hidden", "true");
    axisRow.append(createElement(doc, "span", "", "Ranked by estimate"));
    const axis = createElement(doc, "div", "ability-axis");
    for (const tickValue of uniqueNumbers([domainMinimum, 0, domainMaximum]).sort((a, b) => a - b)) {
      const tick = createElement(doc, "span", "ability-tick", formatAbility(tickValue));
      setPercent(tick, "--position", position(tickValue, domainMinimum, domainMaximum));
      axis.append(tick);
    }
    axisRow.append(axis, createElement(doc, "span", "", "Estimate · 95% interval"));

    const list = createElement(doc, "ol", "ability-list");
    for (const [index, entrant] of ordered.entries()) {
      const item = createElement(doc, "li", "ability-row");
      const identity = createElement(doc, "div", "ability-identity");
      identity.append(createElement(doc, "span", "ability-rank", String(index + 1)));
      const identityText = createElement(doc, "span");
      identityText.append(createElement(doc, "strong", "", entrant.shortLabel));
      identityText.append(createElement(
        doc,
        "small",
        "",
        `Selected profile · ${formatLatency(entrant.validation.p95LatencyMs)} p95`
      ));
      identity.append(identityText);

      const measure = createElement(doc, "div", "ability-measure");
      measure.setAttribute("aria-hidden", "true");
      const zero = createElement(doc, "span", "ability-zero");
      setPercent(zero, "--zero", zeroPosition);
      const interval = createElement(doc, "span", "ability-interval");
      setPercent(interval, "--lower", position(
        entrant.bradleyTerry.lower, domainMinimum, domainMaximum));
      setPercent(interval, "--upper", position(
        entrant.bradleyTerry.upper, domainMinimum, domainMaximum));
      const estimate = createElement(doc, "span", "ability-estimate");
      setPercent(estimate, "--estimate", position(
        entrant.bradleyTerry.estimate, domainMinimum, domainMaximum));
      measure.append(zero, interval, estimate);

      const value = createElement(doc, "div", "ability-value");
      value.append(createElement(doc, "strong", "", formatAbility(entrant.bradleyTerry.estimate)));
      value.append(createElement(
        doc,
        "small",
        "",
        `${formatAbility(entrant.bradleyTerry.lower)} to ${formatAbility(entrant.bradleyTerry.upper)}`
      ));
      item.append(identity, measure, value);
      list.append(item);
    }

    const accessibleSummary = ordered.map((entrant) =>
      `${entrant.shortLabel}: ${formatAbility(entrant.bradleyTerry.estimate)}, ` +
      `95% interval ${formatAbility(entrant.bradleyTerry.lower)} to ` +
      `${formatAbility(entrant.bradleyTerry.upper)}`).join("; ");
    chart.setAttribute("role", "img");
    chart.setAttribute("aria-label", `Relative playing strength. ${accessibleSummary}.`);
    chart.replaceChildren(axisRow, list);
  }

  function renderMatchups(doc, entrants, matchups) {
    const table = doc.getElementById("headToHeadTable");
    if (!table) {
      return;
    }
    const ordered = [...entrants].sort((left, right) =>
      right.bradleyTerry.estimate - left.bradleyTerry.estimate);
    const firstMatchup = matchups[0];
    const caption = createElement(
      doc,
      "caption",
      "",
      `Paired score for the row bot; ${formatInteger(firstMatchup.pairs)} pairs ` +
        `(${formatInteger(firstMatchup.games)} games) per matchup`
    );
    const thead = createElement(doc, "thead");
    const headerRow = createElement(doc, "tr");
    const corner = createElement(doc, "th", "", "Row bot score");
    corner.scope = "col";
    headerRow.append(corner);
    for (const entrant of ordered) {
      const th = createElement(doc, "th", "", entrant.shortLabel);
      th.scope = "col";
      headerRow.append(th);
    }
    thead.append(headerRow);

    const tbody = createElement(doc, "tbody");
    for (const rowEntrant of ordered) {
      const row = createElement(doc, "tr");
      const rowHeader = createElement(doc, "th", "", rowEntrant.shortLabel);
      rowHeader.scope = "row";
      row.append(rowHeader);
      for (const columnEntrant of ordered) {
        if (rowEntrant.id === columnEntrant.id) {
          const diagonal = createElement(doc, "td", "matchup-diagonal", "—");
          diagonal.setAttribute("aria-label", "Same bot; no matchup");
          row.append(diagonal);
          continue;
        }
        const matchup = pairwiseLookup(matchups, rowEntrant.id, columnEntrant.id);
        if (!matchup) {
          row.append(createElement(doc, "td", "matchup-cell", "Not available"));
          continue;
        }
        const state = matchupStatus(matchup);
        const display = matchupDisplay(matchup);
        const cell = createElement(doc, "td", `matchup-cell ${statusClass(matchup)}`);
        cell.append(createElement(doc, "strong", "", display.score));
        cell.append(createElement(doc, "span", "", state));
        cell.append(createElement(
          doc,
          "small",
          "matchup-ci",
          `${display.intervalLower}–${display.intervalUpper}`
        ));
        cell.setAttribute(
          "aria-label",
          matchupAccessibleLabel(rowEntrant.shortLabel, columnEntrant.shortLabel, matchup)
        );
        row.append(cell);
      }
      tbody.append(row);
    }
    table.replaceChildren(caption, thead, tbody);
  }

  function familyLabel(family) {
    const labels = {
      alpha_beta: "Hand alpha-beta",
      jacek_inspired: "Neural alpha-beta",
      mcts: "Tactical MCTS",
      rank5_derived: "Rank5Derived"
    };
    return labels[family] || family.replaceAll("_", " ");
  }

  function formatBudget(candidate) {
    const budget = Number(candidate.budget);
    const compact = budget >= 1000 && budget % 1000 === 0
      ? `${budget / 1000}k`
      : formatInteger(budget);
    if (candidate.fixed) {
      return `fixed ${compact} demo profile`;
    }
    return candidate.family === "mcts" ? `${compact} iterations` : `${compact} nodes`;
  }

  function profileLabel(candidate) {
    return `${familyLabel(candidate.family)} · ${formatBudget(candidate)}`;
  }

  function renderPareto(doc, results) {
    const container = doc.getElementById("paretoPlot");
    if (!container) {
      return;
    }
    const candidates = [...results.validationCandidates].sort((left, right) =>
      left.p95LatencyMs - right.p95LatencyMs);
    const axisModel = paretoAxisModel(results);
    const {xMaximum, yMaximum} = axisModel;
    const families = [...new Set(candidates.map((candidate) => candidate.family))];
    const familyIndex = new Map(families.map((family, index) => [family, index]));

    const chart = createElement(doc, "div", "pareto-chart");
    chart.setAttribute("aria-hidden", "true");
    chart.append(createElement(doc, "div", "pareto-y-title", "Validation paired score"));
    const area = createElement(doc, "div", "pareto-area");

    for (const tick of axisModel.xTicks.slice(1, -1)) {
      const line = createElement(doc, "span", "pareto-grid-line grid-x");
      setPercent(line, "--position", tick.position);
      area.append(line);
    }
    for (const tick of axisModel.yTicks.slice(1, -1)) {
      const line = createElement(doc, "span", "pareto-grid-line grid-y");
      setPercent(line, "--position", tick.position);
      area.append(line);
    }

    const gate = createElement(doc, "div", "pareto-gate");
    setPercent(gate, "--gate", position(results.study.latencyGateMs, 0, xMaximum));
    gate.append(createElement(doc, "span", "", `${results.study.latencyGateMs} ms gate`));
    area.append(gate);

    for (const candidate of candidates) {
      const index = familyIndex.get(candidate.family);
      const marker = createElement(doc, "span", `pareto-marker marker-${index % 4}`);
      marker.classList.toggle("is-ineligible", !candidate.gateEligible);
      marker.classList.toggle("is-selected", candidate.selected);
      marker.style.setProperty("--marker-color", FAMILY_COLORS[index % FAMILY_COLORS.length]);
      setPercent(marker, "--x", position(candidate.p95LatencyMs, 0, xMaximum));
      setPercent(marker, "--y", position(candidate.strength, 0, yMaximum));
      marker.title = `${profileLabel(candidate)}: ${validationStrengthLabel(candidate)} at ` +
        `${formatLatency(candidate.p95LatencyMs)} p95`;
      area.append(marker);
    }

    for (const tick of axisModel.xTicks) {
      const label = createElement(
        doc,
        "span",
        `pareto-axis-label axis-x is-${tick.anchor}`,
        tick.label
      );
      setPercent(label, "--position", tick.position);
      area.append(label);
    }
    for (const tick of axisModel.yTicks) {
      const label = createElement(
        doc,
        "span",
        `pareto-axis-label axis-y is-${tick.anchor}`,
        tick.label
      );
      setPercent(label, "--position", tick.position);
      area.append(label);
    }
    chart.append(area, createElement(doc, "div", "pareto-x-title", "Validation p95 decision latency (ms)"));

    const legend = createElement(doc, "div", "plot-legend");
    for (const candidate of candidates) {
      const index = familyIndex.get(candidate.family);
      const item = createElement(doc, "div", "plot-legend-item");
      const symbol = createElement(doc, "span", `legend-marker marker-${index % 4}`);
      symbol.style.setProperty("--marker-color", FAMILY_COLORS[index % FAMILY_COLORS.length]);
      const state = candidate.selected
        ? "selected"
        : candidate.gateEligible ? "evaluated" : "missed gate";
      item.append(
        symbol,
        createElement(doc, "strong", "", `${profileLabel(candidate)} · ${state}`),
        createElement(
          doc,
          "span",
          "",
          `${validationStrengthLabel(candidate)} · ${formatLatency(candidate.p95LatencyMs)}`
        )
      );
      legend.append(item);
    }

    const accessibleSummary = candidates.map(validationAccessibleSummary).join("; ");
    container.setAttribute("role", "img");
    container.setAttribute(
      "aria-label",
      `Validation strength versus latency; the gate is ${results.study.latencyGateMs} milliseconds. ` +
        `${accessibleSummary}.`
    );
    container.replaceChildren(chart, legend);
  }

  function renderParetoDetails(doc, candidates) {
    const content = doc.getElementById("paretoDetailsContent");
    if (!content) {
      return;
    }
    const intro = createElement(
      doc,
      "p",
      "",
      "Observed validation scores include pair-bootstrap 95% intervals. " +
        "Rank5Derived is shown as a defined reference and therefore has no observed interval."
    );
    const wrapper = createElement(doc, "div", "table-scroll");
    wrapper.tabIndex = 0;
    wrapper.setAttribute("aria-label", "Scrollable validation profile results");
    const table = createElement(doc, "table", "data-table");
    const caption = createElement(doc, "caption", "", "All frozen validation candidates");
    const thead = createElement(doc, "thead");
    const header = createElement(doc, "tr");
    for (const title of ["Profile", "Strength", "95% interval", "p95 latency", "Gate", "Status"]) {
      const th = createElement(doc, "th", "", title);
      th.scope = "col";
      header.append(th);
    }
    thead.append(header);
    const tbody = createElement(doc, "tbody");
    const ordered = [...candidates].sort((left, right) => {
      const familyOrder = familyLabel(left.family).localeCompare(familyLabel(right.family));
      return familyOrder || left.budget - right.budget;
    });
    for (const candidate of ordered) {
      const row = createElement(doc, "tr");
      const profile = createElement(doc, "th", "", profileLabel(candidate));
      profile.scope = "row";
      const strength = createElement(
        doc,
        "td",
        `numeric-cell${candidate.strengthIsReference ? " reference-value" : ""}`,
        validationStrengthLabel(candidate)
      );
      const interval = createElement(
        doc,
        "td",
        "numeric-cell",
        validationIntervalLabel(candidate)
      );
      const latency = createElement(doc, "td", "numeric-cell", formatLatency(candidate.p95LatencyMs));
      const gate = createElement(doc, "td");
      gate.append(createElement(
        doc,
        "span",
        `profile-status ${candidate.gateEligible ? "selected" : "missed-gate"}`,
        candidate.gateEligible ? "Pass" : "Miss"
      ));
      const state = createElement(doc, "td");
      const stateText = candidate.fixed
        ? "Fixed comparator"
        : candidate.selected ? "Selected" : candidate.paretoOptimal ? "Frontier" : "Evaluated";
      state.append(createElement(
        doc,
        "span",
        `profile-status${candidate.selected ? " selected" : ""}`,
        stateText
      ));
      row.append(profile, strength, interval, latency, gate, state);
      tbody.append(row);
    }
    table.append(caption, thead, tbody);
    wrapper.append(table);
    content.replaceChildren(intro, wrapper);
  }

  function createTakeaway(doc, number, title, body) {
    const article = createElement(doc, "article", "takeaway-card");
    article.append(
      createElement(doc, "span", "takeaway-number", String(number)),
      createElement(doc, "h3", "", title),
      createElement(doc, "p", "", body)
    );
    return article;
  }

  function validationAccessibleSummary(candidate) {
    return `${profileLabel(candidate)}: ${validationStrengthLabel(candidate)}, ` +
      `${formatLatency(candidate.p95LatencyMs)} p95, ` +
      `${candidate.gateEligible ? "within" : "outside"} the gate` +
      `${candidate.selected ? ", selected" : ""}`;
  }

  function takeawayModels(results) {
    const leader = [...results.entrants].sort((left, right) =>
      right.bradleyTerry.estimate - left.bradleyTerry.estimate)[0];
    const unresolved = results.matchups.find((matchup) =>
      matchup.classification === "statistically_unresolved");
    const entrantById = new Map(results.entrants.map((entrant) => [entrant.id, entrant]));
    const bestObservedCandidate = results.validationCandidates
      .filter((candidate) => candidate.selected && candidate.gateEligible && !candidate.strengthIsReference)
      .sort((left, right) => right.strength - left.strength)[0];

    const unresolvedLeft = unresolved ? entrantById.get(unresolved.leftId) : null;
    const unresolvedRight = unresolved ? entrantById.get(unresolved.rightId) : null;
    const unresolvedDisplay = unresolved ? matchupDisplay(unresolved) : null;
    return [
      {
        number: 1,
        title: `${leader.shortLabel} leads the model`,
        body: `Its ${formatAbility(leader.bradleyTerry.estimate)} relative-strength estimate is ` +
          "the highest in this four-bot field."
      },
      {
        number: 2,
        title: "The top direct matchup is unresolved",
        body: unresolved
          ? `${unresolvedLeft.shortLabel} scored ${unresolvedDisplay.score} against ` +
            `${unresolvedRight.shortLabel}; the ${unresolvedDisplay.intervalLower}–` +
            `${unresolvedDisplay.intervalUpper} interval crosses 50%.`
          : "No statistically unresolved matchup was reported."
      },
      {
        number: 3,
        title: "The latency gate changes profile choice",
        body: bestObservedCandidate
          ? `${profileLabel(bestObservedCandidate)} had the strongest observed selected validation score ` +
            `(${formatPercent(bestObservedCandidate.strength)}) at ` +
            `${formatLatency(bestObservedCandidate.p95LatencyMs)} p95.`
          : `Selected profiles had to stay within the ${results.study.latencyGateMs} ms p95 gate.`
      }
    ];
  }

  function renderTakeaways(doc, results) {
    const container = doc.getElementById("takeaways");
    if (!container) {
      return;
    }
    const cards = takeawayModels(results).map((card) =>
      createTakeaway(doc, card.number, card.title, card.body));
    container.replaceChildren(...cards);
  }

  function renderCalibration(doc, calibration) {
    const content = doc.getElementById("calibrationContent");
    if (!content) {
      return;
    }
    const intro = createElement(
      doc,
      "p",
      "",
      "Calibration measures how well each validation-fitted bot score predicts the eventual " +
        "game winner on the frozen test set. Lower Brier score and log loss are better; " +
        "they do not measure playing strength, and uncertainty is clustered by game pair."
    );
    const wrapper = createElement(doc, "div", "table-scroll");
    wrapper.tabIndex = 0;
    wrapper.setAttribute("aria-label", "Scrollable calibration results");
    const table = createElement(doc, "table", "data-table");
    const caption = createElement(doc, "caption", "", "Frozen test calibration metrics");
    const thead = createElement(doc, "thead");
    const header = createElement(doc, "tr");
    for (const title of ["Bot", "Brier score", "Log loss", "Scored samples"]) {
      const th = createElement(doc, "th", "", title);
      th.scope = "col";
      header.append(th);
    }
    thead.append(header);
    const tbody = createElement(doc, "tbody");
    for (const result of calibration) {
      const row = createElement(doc, "tr");
      const bot = createElement(doc, "th", "", result.label);
      bot.scope = "row";
      row.append(
        bot,
        createElement(doc, "td", "numeric-cell", Number(result.brierScore).toFixed(3)),
        createElement(doc, "td", "numeric-cell", Number(result.logLoss).toFixed(3)),
        createElement(doc, "td", "numeric-cell", formatInteger(result.samples))
      );
      tbody.append(row);
    }
    table.append(caption, thead, tbody);
    wrapper.append(table);
    content.replaceChildren(intro, wrapper);
  }

  function renderMethodology(doc, study, matchups) {
    const content = doc.getElementById("methodologyContent");
    if (!content) {
      return;
    }
    const pairsPerMatchup = matchups.length ? matchups[0].pairs : 0;
    const gamesPerMatchup = matchups.length ? matchups[0].games : 0;
    const list = createElement(doc, "ul", "method-list");
    for (const text of [
      `${study.entrantCount} frozen entrants played ${formatInteger(study.games)} decisive test games.`,
      `${formatInteger(pairsPerMatchup)} color-swapped pairs (${formatInteger(gamesPerMatchup)} games) were played for each direct matchup.`,
      `Disjoint openings were sampled at ${study.openingDepths.join(", ")} physical plies.`,
      `Profile selection used validation results and a ${study.latencyGateMs} ms single-thread p95 gate before the test tournament.`,
      "Pair-bootstrap 95% intervals determine whether a direct matchup is resolved."
    ]) {
      list.append(createElement(doc, "li", "", text));
    }
    content.replaceChildren(list);
  }

  function renderCaveats(doc, caveats) {
    const list = doc.getElementById("caveatList");
    if (!list) {
      return;
    }
    const values = [
      caveats.rank5,
      caveats.validationReference,
      caveats.relativeStrength,
      caveats.latency
    ].filter((value) => typeof value === "string" && value.length > 0);
    list.replaceChildren(...values.map((value) => createElement(doc, "li", "", value)));
  }

  function showError(doc) {
    const error = doc.getElementById("benchmarkError");
    if (error) {
      error.hidden = false;
    }
    const loading = doc.getElementById("benchmarkLoadState");
    if (loading) {
      loading.hidden = true;
    }
  }

  function render(results, doc) {
    const targetDocument = doc || (typeof document === "undefined" ? null : document);
    if (!targetDocument) {
      return false;
    }
    try {
      validateResults(results);
      renderStudySummary(targetDocument, results);
      renderLinks(targetDocument, results.links);
      renderAbility(targetDocument, results.entrants);
      renderMatchups(targetDocument, results.entrants, results.matchups);
      renderPareto(targetDocument, results);
      renderParetoDetails(targetDocument, results.validationCandidates);
      renderTakeaways(targetDocument, results);
      renderCalibration(targetDocument, results.calibration);
      renderMethodology(targetDocument, results.study, results.matchups);
      renderCaveats(targetDocument, results.caveats);

      const loading = targetDocument.getElementById("benchmarkLoadState");
      if (loading) {
        loading.hidden = true;
      }
      const error = targetDocument.getElementById("benchmarkError");
      if (error) {
        error.hidden = true;
      }
      return true;
    } catch (error) {
      showError(targetDocument);
      return false;
    }
  }

  const api = Object.freeze({
    EXPECTED_SCHEMA,
    formatPercent,
    invertMatchup,
    matchupAccessibleLabel,
    matchupDisplay,
    matchupStatus,
    pairwiseLookup,
    paretoAxisModel,
    takeawayModels,
    validationAccessibleSummary,
    validationIntervalLabel,
    validationStrengthLabel,
    render
  });
  root.PaperSoccerBenchmarks = api;

  if (typeof document !== "undefined") {
    const renderPage = () => render(root.PaperSoccerBenchmarkResults, document);
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", renderPage, {once: true});
    } else {
      renderPage();
    }
  }
})(typeof globalThis === "undefined" ? this : globalThis);
