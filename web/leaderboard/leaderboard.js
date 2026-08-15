(function codingameLeaderboard(root) {
  "use strict";

  const EXPECTED_SCHEMA = "papersoccer.codingame-leaderboard-summary.v1";

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

  function formatInteger(value) {
    return Number(value).toLocaleString("en-US");
  }

  function formatPercent(value) {
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function formatRating(value) {
    return Number(value).toFixed(2);
  }

  function formatDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.valueOf())) {
      return String(value);
    }
    return new Intl.DateTimeFormat("en", {
      year: "numeric",
      month: "short",
      day: "numeric",
      timeZone: "UTC"
    }).format(date);
  }

  function shortHash(value) {
    const hash = String(value || "");
    return hash.length > 16 ? `${hash.slice(0, 12)}…` : hash;
  }

  function requireFiniteNumber(value, label) {
    if (!Number.isFinite(Number(value))) {
      throw new Error(`${label} must be numeric.`);
    }
  }

  function validateResults(results) {
    if (!results || results.schema !== EXPECTED_SCHEMA) {
      throw new Error("Unsupported or missing leaderboard summary.");
    }
    if (!results.tournament || !Array.isArray(results.standings) ||
        !Array.isArray(results.headToHead)) {
      throw new Error("Leaderboard summary is incomplete.");
    }

    const requiredTournamentFields = [
      "id",
      "generatedAtUtc",
      "entrantCount",
      "gameCount",
      "gamesPerEntrant",
      "playerOneGamesPerEntrant",
      "playerTwoGamesPerEntrant",
      "rulesLabel",
      "scoringLabel",
      "scheduleSeed",
      "sourceCommit",
      "rawResultsUrl"
    ];
    for (const key of requiredTournamentFields) {
      if (results.tournament[key] === undefined || results.tournament[key] === null) {
        throw new Error(`Tournament metadata is missing ${key}.`);
      }
    }
    if (!results.tournament.environment ||
        typeof results.tournament.environment !== "object") {
      throw new Error("Tournament metadata is missing environment.");
    }
    for (const key of ["os", "cpu", "compiler"]) {
      if (typeof results.tournament.environment[key] !== "string") {
        throw new Error(`Runtime environment is missing ${key}.`);
      }
    }
    for (const key of [
      "entrantCount",
      "gameCount",
      "gamesPerEntrant",
      "playerOneGamesPerEntrant",
      "playerTwoGamesPerEntrant"
    ]) {
      requireFiniteNumber(results.tournament[key], `tournament.${key}`);
    }
    if (Number(results.tournament.entrantCount) !== results.standings.length) {
      throw new Error("Entrant count does not match the standings.");
    }

    const ids = new Set();
    for (const entrant of results.standings) {
      for (const key of ["rank", "score", "mu", "sigma", "games", "wins", "losses",
        "winRate", "forfeits"]) {
        requireFiniteNumber(entrant[key], `standings.${key}`);
      }
      if (!entrant.id || !entrant.displayName || ids.has(entrant.id) ||
          !Array.isArray(entrant.aliases) || !entrant.playerOne || !entrant.playerTwo) {
        throw new Error("A standing is malformed or has a duplicate id.");
      }
      for (const side of [entrant.playerOne, entrant.playerTwo]) {
        requireFiniteNumber(side.games, "standings.side.games");
        requireFiniteNumber(side.wins, "standings.side.wins");
      }
      ids.add(entrant.id);
    }

    for (const matchup of results.headToHead) {
      if (!ids.has(matchup.rowId) || !ids.has(matchup.columnId) ||
          matchup.rowId === matchup.columnId) {
        throw new Error("A head-to-head entry references an unknown bot.");
      }
      for (const key of ["games", "wins", "losses", "score"]) {
        requireFiniteNumber(matchup[key], `headToHead.${key}`);
      }
    }
    return results;
  }

  function pairwiseLookup(matchups, rowId, columnId) {
    const direct = matchups.find((matchup) =>
      matchup.rowId === rowId && matchup.columnId === columnId);
    if (direct) {
      return direct;
    }
    const reverse = matchups.find((matchup) =>
      matchup.rowId === columnId && matchup.columnId === rowId);
    if (!reverse) {
      return null;
    }
    return {
      rowId,
      columnId,
      games: Number(reverse.games),
      wins: Number(reverse.losses),
      losses: Number(reverse.wins),
      score: 1 - Number(reverse.score)
    };
  }

  function renderStats(doc, tournament) {
    const stats = doc.getElementById("leaderboardStats");
    if (!stats) {
      return;
    }
    const definitions = [
      ["Unique entrants", formatInteger(tournament.entrantCount)],
      ["Decisive games", formatInteger(tournament.gameCount)],
      ["Rules", tournament.rulesLabel],
      ["Tournament date", formatDate(tournament.generatedAtUtc)]
    ];
    const cards = definitions.map(([label, value], index) => {
      const card = createElement(doc, "div", "stat-card");
      const term = createElement(doc, "dt", "", label);
      const definition = createElement(doc, "dd", index === 2 ? "stat-card-text" : "", value);
      if (index === 3) {
        const time = createElement(doc, "time", "", value);
        time.dateTime = tournament.generatedAtUtc;
        definition.replaceChildren(time);
      }
      card.append(term, definition);
      return card;
    });
    stats.replaceChildren(...cards);

    const label = doc.getElementById("tournamentLabel");
    if (label) {
      label.textContent = `${tournament.id} · seed ${tournament.scheduleSeed}`;
    }
  }

  function createBotIdentity(doc, entrant, withRank) {
    const wrapper = createElement(doc, "div", "bot-identity");
    const name = typeof entrant.documentationUrl === "string" && entrant.documentationUrl
      ? createElement(doc, "a", "bot-name", entrant.displayName)
      : createElement(doc, "span", "bot-name", entrant.displayName);
    if (name.tagName === "A") {
      name.href = entrant.documentationUrl;
    }
    const metadata = createElement(
      doc,
      "small",
      "bot-id",
      `${withRank ? `#${entrant.rank} · ` : ""}${entrant.id}`
    );
    wrapper.append(name, metadata);
    if (entrant.aliases.length > 0) {
      wrapper.append(createElement(doc, "small", "bot-alias", `Alias: ${entrant.aliases.join(", ")}`));
    }
    return wrapper;
  }

  function renderStandingsHead(doc, table, includeForfeits) {
    const head = createElement(doc, "thead");
    const row = createElement(doc, "tr");
    const labels = [
      "Rank",
      "Bot",
      "Local CodinGame-style score",
      ["TrueSkill mean", "μ"],
      ["TrueSkill standard deviation", "σ"],
      "W–L",
      "Win rate",
      "Side split"
    ];
    if (includeForfeits) {
      labels.push("Forfeits");
    }
    for (const label of labels) {
      const heading = createElement(doc, "th");
      heading.scope = "col";
      if (Array.isArray(label)) {
        const abbreviation = createElement(doc, "abbr", "", label[1]);
        abbreviation.title = label[0];
        heading.append(abbreviation);
      } else {
        heading.textContent = label;
      }
      row.append(heading);
    }
    head.append(row);
    if (table.tHead) {
      table.tHead.replaceWith(head);
    } else {
      table.append(head);
    }
  }

  function renderStandings(doc, standings) {
    const table = doc.getElementById("standingsTable");
    if (!table) {
      return;
    }
    const ordered = [...standings].sort((left, right) =>
      Number(left.rank) - Number(right.rank));
    const includeForfeits = ordered.some((entrant) => Number(entrant.forfeits) > 0);
    renderStandingsHead(doc, table, includeForfeits);
    const note = doc.getElementById("standingsNote");
    if (note) {
      note.textContent = "P1 attacks the top goal; P2 attacks the bottom goal." +
        (includeForfeits ? " Forfeits count as ordinary losses." : "");
    }
    const body = createElement(doc, "tbody");
    for (const entrant of ordered) {
      const row = createElement(doc, "tr", Number(entrant.rank) <= 3 ? "podium-row" : "");
      const rank = createElement(doc, "th", "rank-cell", String(entrant.rank));
      rank.scope = "row";
      rank.setAttribute("aria-label", `Rank ${entrant.rank}`);
      const identity = createElement(doc, "td", "bot-cell");
      identity.append(createBotIdentity(doc, entrant, false));
      const score = createElement(doc, "td", "numeric-cell score-cell");
      score.append(createElement(doc, "strong", "", formatRating(entrant.score)));
      const mu = createElement(doc, "td", "numeric-cell", formatRating(entrant.mu));
      const sigma = createElement(doc, "td", "numeric-cell", formatRating(entrant.sigma));
      const record = createElement(
        doc, "td", "numeric-cell", `${formatInteger(entrant.wins)}–${formatInteger(entrant.losses)}`);
      const winRate = createElement(doc, "td", "numeric-cell", formatPercent(entrant.winRate));
      const sideSplit = createElement(doc, "td", "side-cell");
      for (const [label, side] of [["P1", entrant.playerOne], ["P2", entrant.playerTwo]]) {
        const lossCount = Number(side.games) - Number(side.wins);
        sideSplit.append(createElement(
          doc,
          "span",
          "side-record",
          `${label} ${formatInteger(side.wins)}–${formatInteger(lossCount)}`
        ));
      }
      row.append(rank, identity, score, mu, sigma, record, winRate, sideSplit);
      if (includeForfeits) {
        row.append(createElement(
          doc,
          "td",
          `numeric-cell${Number(entrant.forfeits) > 0 ? " has-forfeit" : ""}`,
          formatInteger(entrant.forfeits)
        ));
      }
      body.append(row);
    }
    table.tBodies[0]?.replaceWith(body);
  }

  function renderHeadToHead(doc, standings, matchups) {
    const table = doc.getElementById("headToHeadTable");
    if (!table) {
      return;
    }
    const ordered = [...standings].sort((left, right) =>
      Number(left.rank) - Number(right.rank));
    const head = createElement(doc, "thead");
    const headerRow = createElement(doc, "tr");
    headerRow.append(createElement(doc, "th", "matrix-corner", "Bot"));
    headerRow.firstChild.scope = "col";
    for (const entrant of ordered) {
      const heading = createElement(doc, "th", "matrix-column");
      heading.scope = "col";
      const abbreviation = createElement(doc, "abbr", "", `#${entrant.rank}`);
      abbreviation.title = entrant.displayName;
      heading.append(abbreviation);
      headerRow.append(heading);
    }
    head.append(headerRow);

    const body = createElement(doc, "tbody");
    for (const rowEntrant of ordered) {
      const row = createElement(doc, "tr");
      const heading = createElement(doc, "th", "matrix-bot-cell");
      heading.scope = "row";
      heading.append(createBotIdentity(doc, rowEntrant, true));
      row.append(heading);
      for (const columnEntrant of ordered) {
        if (rowEntrant.id === columnEntrant.id) {
          const diagonal = createElement(doc, "td", "matrix-diagonal", "—");
          diagonal.setAttribute("aria-label", `${rowEntrant.displayName}, same bot`);
          row.append(diagonal);
          continue;
        }
        const matchup = pairwiseLookup(matchups, rowEntrant.id, columnEntrant.id);
        if (!matchup) {
          const missing = createElement(doc, "td", "matrix-missing", "—");
          missing.setAttribute(
            "aria-label",
            `${rowEntrant.displayName} versus ${columnEntrant.displayName}: result unavailable`
          );
          row.append(missing);
          continue;
        }
        const score = Number(matchup.score);
        const cellClass = score > 0.5 ? "matrix-win" : score < 0.5 ? "matrix-loss" : "matrix-even";
        const cell = createElement(doc, "td", cellClass);
        cell.setAttribute(
          "aria-label",
          `${rowEntrant.displayName} versus ${columnEntrant.displayName}: ` +
            `${formatPercent(score)}, ${matchup.wins} wins and ${matchup.losses} losses`
        );
        cell.append(
          createElement(doc, "strong", "", formatPercent(score)),
          createElement(doc, "small", "", `${matchup.wins}–${matchup.losses}`)
        );
        row.append(cell);
      }
      body.append(row);
    }
    table.tHead?.replaceWith(head);
    table.tBodies[0]?.replaceWith(body);
  }

  function definitionRow(doc, term, value) {
    const wrapper = createElement(doc, "div", "provenance-row");
    wrapper.append(createElement(doc, "dt", "", term), createElement(doc, "dd", "", value));
    return wrapper;
  }

  function renderAliases(doc, standings) {
    const content = doc.getElementById("aliasContent");
    if (!content) {
      return;
    }
    const list = createElement(doc, "ul", "alias-list");
    for (const entrant of [...standings].sort((left, right) => left.id.localeCompare(right.id))) {
      const item = createElement(doc, "li", "alias-item");
      const identity = createBotIdentity(doc, entrant, false);
      const details = createElement(doc, "span", "artifact-hash");
      details.append(doc.createTextNode("Submission "));
      const hash = createElement(doc, "code", "", shortHash(entrant.submissionSha256));
      hash.title = entrant.submissionSha256;
      details.append(hash);
      item.append(identity, details);
      list.append(item);
    }
    content.replaceChildren(list);
  }

  function renderProvenance(doc, tournament) {
    const content = doc.getElementById("provenanceContent");
    if (!content) {
      return;
    }
    const values = createElement(doc, "dl", "provenance-list");
    values.append(
      definitionRow(doc, "Tournament", tournament.id),
      definitionRow(doc, "Generated", tournament.generatedAtUtc),
      definitionRow(doc, "Source commit", tournament.sourceCommit),
      definitionRow(doc, "Schedule seed", String(tournament.scheduleSeed)),
      definitionRow(doc, "Operating system", tournament.environment.os),
      definitionRow(doc, "CPU", tournament.environment.cpu),
      definitionRow(doc, "Compiler", tournament.environment.compiler)
    );
    const rawLink = createElement(doc, "a", "artifact-link", "Open canonical raw tournament artifact");
    rawLink.href = tournament.rawResultsUrl;
    content.replaceChildren(values, rawLink);
  }

  function renderMethodology(doc, tournament) {
    const content = doc.getElementById("methodologyContent");
    if (!content) {
      return;
    }
    const list = createElement(doc, "ul", "method-list");
    for (const text of [
      `${formatInteger(tournament.entrantCount)} unique reviewed submissions played ` +
        `${formatInteger(tournament.gameCount)} decisive games; every bot played ` +
        `${formatInteger(tournament.gamesPerEntrant)} games.`,
      `Each bot played ${formatInteger(tournament.playerOneGamesPerEntrant)} games as P1 and ` +
        `${formatInteger(tournament.playerTwoGamesPerEntrant)} as P2.`,
      `${tournament.rulesLabel}; bot processes received 1,000 ms for their first response ` +
        "and 200 ms thereafter.",
      "Timeouts, crashes, malformed or illegal actions, and incomplete rebound sequences " +
        "were scored as forfeits.",
      `${tournament.scoringLabel}: decisive 1v1 TrueSkill with μ=25, σ=25/3, β=25/6, ` +
        "τ=25/300, no draws, ranked by μ − 3σ."
    ]) {
      list.append(createElement(doc, "li", "", text));
    }
    const sources = createElement(doc, "p", "method-sources");
    sources.append(doc.createTextNode("Sources: "));
    const rulesLink = createElement(doc, "a", "", "CodinGame Paper Soccer rules");
    rulesLink.href = "https://www.codingame.com/multiplayer/bot-programming/paper-soccer";
    const ratingLink = createElement(doc, "a", "", "Microsoft TrueSkill overview");
    ratingLink.href = "https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/";
    sources.append(rulesLink, doc.createTextNode(" · "), ratingLink);
    content.replaceChildren(list, sources);
  }

  function showError(doc) {
    const loading = doc.getElementById("leaderboardLoadState");
    const error = doc.getElementById("leaderboardError");
    const overview = doc.getElementById("leaderboardOverview");
    if (loading) {
      loading.hidden = true;
    }
    if (error) {
      error.hidden = false;
    }
    if (overview) {
      overview.setAttribute("aria-busy", "false");
    }
  }

  function render(results, doc) {
    const targetDocument = doc || (typeof document === "undefined" ? null : document);
    if (!targetDocument) {
      return false;
    }
    try {
      validateResults(results);
      renderStats(targetDocument, results.tournament);
      renderStandings(targetDocument, results.standings);
      renderHeadToHead(targetDocument, results.standings, results.headToHead);
      renderAliases(targetDocument, results.standings);
      renderProvenance(targetDocument, results.tournament);
      renderMethodology(targetDocument, results.tournament);

      const loading = targetDocument.getElementById("leaderboardLoadState");
      const error = targetDocument.getElementById("leaderboardError");
      const overview = targetDocument.getElementById("leaderboardOverview");
      if (loading) {
        loading.hidden = true;
      }
      if (error) {
        error.hidden = true;
      }
      if (overview) {
        overview.setAttribute("aria-busy", "false");
      }
      return true;
    } catch (error) {
      showError(targetDocument);
      return false;
    }
  }

  const api = Object.freeze({
    EXPECTED_SCHEMA,
    formatDate,
    formatPercent,
    pairwiseLookup,
    validateResults,
    render
  });
  root.PaperSoccerCodingameLeaderboard = api;

  if (typeof document !== "undefined") {
    const renderPage = () => render(root.PAPERSOCCER_CODINGAME_LEADERBOARD_RESULTS, document);
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", renderPage, {once: true});
    } else {
      renderPage();
    }
  }
})(typeof globalThis === "undefined" ? this : globalThis);
