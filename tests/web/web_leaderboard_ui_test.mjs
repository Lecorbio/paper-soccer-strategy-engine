import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const urls = {
  gameHtml: new URL("../../web/index.html", import.meta.url),
  benchmarkHtml: new URL("../../web/benchmarks/index.html", import.meta.url),
  leaderboardHtml: new URL("../../web/leaderboard/index.html", import.meta.url),
  leaderboardCss: new URL("../../web/leaderboard/leaderboard.css", import.meta.url),
  leaderboardSource: new URL("../../web/leaderboard/leaderboard.js", import.meta.url),
  leaderboardResults: new URL("../../web/leaderboard/leaderboard-results.js", import.meta.url),
  siteCss: new URL("../../web/site.css", import.meta.url),
};

const [
  gameHtml,
  benchmarkHtml,
  leaderboardHtml,
  leaderboardCss,
  leaderboardSource,
  leaderboardResults,
  siteCss,
] =
  await Promise.all(Object.values(urls).map((url) => readFile(url, "utf8")));

await import("../../web/leaderboard/leaderboard.js");
const renderer = globalThis.PaperSoccerCodingameLeaderboard;

function standing({
  rank,
  id,
  displayName,
  score,
  wins,
  aliases = [],
  forfeits = 0,
}) {
  return {
    rank,
    id,
    displayName,
    aliases,
    score,
    mu: 25 + score,
    sigma: 2,
    games: 4,
    wins,
    losses: 4 - wins,
    winRate: wins / 4,
    forfeits,
    playerOne: {games: 2, wins: Math.min(wins, 2)},
    playerTwo: {games: 2, wins: Math.max(0, wins - 2)},
    submissionSha256: id.repeat(64).slice(0, 64),
    documentationUrl: `https://example.test/${id}`,
  };
}

const fixture = {
  schema: "papersoccer.codingame-leaderboard-summary.v1",
  tournament: {
    id: "test-tournament",
    generatedAtUtc: "2026-08-13T12:00:00Z",
    entrantCount: 3,
    gameCount: 6,
    gamesPerEntrant: 4,
    playerOneGamesPerEntrant: 2,
    playerTwoGamesPerEntrant: 2,
    rulesLabel: "CodinGame Paper Soccer",
    scoringLabel: "Local CodinGame-style score",
    scheduleSeed: "20260813",
    sourceCommit: "0123456789abcdef",
    environment: {os: "Test OS", cpu: "Test CPU", compiler: "Test compiler"},
    rawResultsUrl: "../../benchmarks/codingame_leaderboard/tournament.json",
  },
  standings: [
    standing({rank: 1, id: "alpha", displayName: "Alpha", score: 20, wins: 4}),
    standing({rank: 2, id: "beta", displayName: "Beta", score: 15, wins: 2}),
    standing({
      rank: 3,
      id: "gamma",
      displayName: "Gamma",
      score: 10,
      wins: 0,
      aliases: ["gamma-v2"],
    }),
  ],
  headToHead: [
    {rowId: "alpha", columnId: "beta", games: 2, wins: 2, losses: 0, score: 1},
    {rowId: "alpha", columnId: "gamma", games: 2, wins: 2, losses: 0, score: 1},
    {rowId: "beta", columnId: "gamma", games: 2, wins: 2, losses: 0, score: 1},
  ],
};

class MiniElement {
  constructor(tagName, ownerDocument) {
    this.tagName = tagName.toUpperCase();
    this.ownerDocument = ownerDocument;
    this.parentNode = null;
    this.children = [];
    this.attributes = new Map();
    this.className = "";
    this.hidden = false;
    this._id = "";
    this._text = "";
  }

  set id(value) {
    if (this._id) {
      this.ownerDocument.ids.delete(this._id);
    }
    this._id = String(value);
    if (this._id) {
      this.ownerDocument.ids.set(this._id, this);
    }
  }

  get id() {
    return this._id;
  }

  set textContent(value) {
    for (const child of this.children) {
      child.parentNode = null;
    }
    this.children = [];
    this._text = String(value);
  }

  get textContent() {
    return this._text + this.children.map((child) => child.textContent).join("");
  }

  get firstChild() {
    return this.children[0] || null;
  }

  get tBodies() {
    return this.children.filter((child) => child.tagName === "TBODY");
  }

  get tHead() {
    return this.children.find((child) => child.tagName === "THEAD") || null;
  }

  append(...values) {
    this._text = "";
    for (const value of values) {
      const child = value instanceof MiniElement
        ? value
        : this.ownerDocument.createTextNode(String(value));
      if (child.parentNode) {
        const index = child.parentNode.children.indexOf(child);
        if (index >= 0) {
          child.parentNode.children.splice(index, 1);
        }
      }
      child.parentNode = this;
      this.children.push(child);
    }
  }

  replaceChildren(...values) {
    for (const child of this.children) {
      child.parentNode = null;
    }
    this.children = [];
    this._text = "";
    this.append(...values);
  }

  replaceWith(value) {
    assert.ok(this.parentNode, `${this.tagName} must have a parent before replaceWith`);
    const replacement = value instanceof MiniElement
      ? value
      : this.ownerDocument.createTextNode(String(value));
    const index = this.parentNode.children.indexOf(this);
    assert.notEqual(index, -1);
    replacement.parentNode = this.parentNode;
    this.parentNode.children[index] = replacement;
    this.parentNode = null;
  }

  setAttribute(name, value) {
    this.attributes.set(String(name), String(value));
  }

  getAttribute(name) {
    return this.attributes.get(String(name)) ?? null;
  }
}

class MiniDocument {
  constructor() {
    this.ids = new Map();
    this.listeners = new Map();
    this.readyState = "loading";
    this.body = this.createElement("body");
  }

  createElement(tagName) {
    return new MiniElement(tagName, this);
  }

  createTextNode(value) {
    const node = new MiniElement("#text", this);
    node.textContent = value;
    return node;
  }

  getElementById(id) {
    return this.ids.get(String(id)) || null;
  }

  addEventListener(type, callback) {
    const callbacks = this.listeners.get(type) || [];
    callbacks.push(callback);
    this.listeners.set(type, callbacks);
  }

  dispatchDOMContentLoaded() {
    this.readyState = "interactive";
    const callbacks = this.listeners.get("DOMContentLoaded") || [];
    this.listeners.delete("DOMContentLoaded");
    for (const callback of callbacks) {
      callback();
    }
    this.readyState = "complete";
  }
}

function createDirectFileDocument() {
  const document = new MiniDocument();
  const elements = {};
  const add = (tagName, id, parent = document.body) => {
    const element = document.createElement(tagName);
    element.id = id;
    parent.append(element);
    elements[id] = element;
    return element;
  };

  const overview = add("main", "leaderboardOverview");
  overview.setAttribute("aria-busy", "true");
  add("span", "tournamentLabel", overview);
  add("dl", "leaderboardStats", overview);
  add("div", "leaderboardLoadState", overview);
  const error = add("div", "leaderboardError", overview);
  error.hidden = true;

  const standings = add("table", "standingsTable", overview);
  standings.append(document.createElement("tbody"));
  add("p", "standingsNote", overview);
  const headToHead = add("table", "headToHeadTable", overview);
  headToHead.append(document.createElement("thead"), document.createElement("tbody"));
  add("div", "aliasContent", overview);
  add("div", "provenanceContent", overview);
  add("div", "methodologyContent", overview);
  return {document, elements};
}

function descendants(element, tagName) {
  const matches = [];
  for (const child of element.children) {
    if (child.tagName === tagName) {
      matches.push(child);
    }
    matches.push(...descendants(child, tagName));
  }
  return matches;
}

function navigation(html) {
  const nav = html.match(/<nav class="site-nav"[^>]*>[\s\S]*?<\/nav>/)?.[0];
  assert.ok(nav, "page should contain the shared primary navigation");
  return Array.from(nav.matchAll(/<a\b([^>]*)>([^<]+)<\/a>/g), (match) => ({
    attributes: match[1],
    label: match[2].trim(),
  }));
}

function colorToken(css, name) {
  const value = css.match(new RegExp(`--${name}:\\s*(#[0-9a-f]{6})`, "i"))?.[1];
  assert.ok(value, `CSS should define --${name}`);
  return value;
}

function relativeLuminance(hex) {
  const channels = [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255);
  const [red, green, blue] = channels.map((value) =>
    value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(left, right) {
  const [lighter, darker] = [relativeLuminance(left), relativeLuminance(right)]
    .sort((a, b) => b - a);
  return (lighter + 0.05) / (darker + 0.05);
}

test("all three pages expose consistent Game, Leaderboard, and Benchmarks navigation", () => {
  const menus = [gameHtml, leaderboardHtml, benchmarkHtml].map(navigation);
  for (const menu of menus) {
    assert.deepEqual(menu.map((entry) => entry.label), ["Game", "Leaderboard", "Benchmarks"]);
    assert.equal(menu.filter((entry) => /aria-current="page"/.test(entry.attributes)).length, 1);
  }
  assert.match(menus[0][1].attributes, /href="leaderboard\/index\.html"/);
  assert.match(menus[1][1].attributes, /aria-current="page"/);
  assert.match(menus[2][1].attributes, /href="\.\.\/leaderboard\/index\.html"/);
});

test("all pages use one stable research-site header", () => {
  for (const [html, stylesheet] of [
    [gameHtml, "site.css"],
    [leaderboardHtml, "../site.css"],
    [benchmarkHtml, "../site.css"],
  ]) {
    const header = html.match(/<header class="site-header">[\s\S]*?<\/header>/)?.[0];
    assert.ok(header, "page should use the canonical site header");
    assert.match(header, /<div class="site-header-inner">/);
    assert.match(
      header,
      /class="site-identity"[\s\S]*?<span>Paper Soccer<\/span>[\s\S]*?<small>Strategy research engine<\/small>/,
    );
    assert.doesNotMatch(header, /<button\b|mode-switch|playModeButton|replayModeButton/);
    assert.ok(
      html.includes(`<link rel="stylesheet" href="${stylesheet}">`),
      `page should load ${stylesheet}`,
    );
  }
  assert.match(siteCss, /\.site-header-inner\s*\{[^}]*width:\s*min\(100%, 1180px\)/s);
  assert.match(siteCss, /\.site-header-inner\s*\{[^}]*min-height:\s*70px/s);
  assert.match(siteCss, /\.site-header \.site-nav a\[aria-current="page"\]::after/);
  assert.match(
    siteCss,
    /\.site-header \.site-nav a\[aria-current="page"\]\s*\{[^}]*font-weight:\s*650/s,
  );
  assert.ok(
    contrastRatio(colorToken(siteCss, "study-muted"), colorToken(siteCss, "study-page")) >= 4.5,
    "small muted research copy should meet WCAG AA contrast against the page",
  );
  assert.doesNotMatch(siteCss, /#fff0b8|#806215/);
});

test("leaderboard page is a local classic-script view with accessible fallback states", () => {
  const scripts = Array.from(
    leaderboardHtml.matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*><\/script>/g),
    (match) => match[1],
  );
  assert.deepEqual(scripts, ["leaderboard-results.js", "leaderboard.js"]);
  assert.ok(scripts.every((source) => !/^(?:https?:)?\/\//.test(source)));
  assert.doesNotMatch(
    leaderboardHtml,
    /papersoccer-wasm|game-engine|board-view|app-support|<canvas\b|type="module"/i,
  );
  assert.match(leaderboardHtml, /<main[^>]*aria-busy="true"/);
  assert.match(leaderboardHtml, /id="leaderboardLoadState"[^>]*role="status"/);
  assert.match(leaderboardHtml, /id="leaderboardError"[^>]*role="alert"[^>]*hidden/);
  assert.match(leaderboardHtml, /<noscript>[\s\S]*frozen tables/i);
});

test("leaderboard tables are semantic and horizontally keyboard-scrollable", () => {
  for (const id of ["standingsTable", "headToHeadTable"]) {
    assert.match(leaderboardHtml, new RegExp(`<table id="${id}"`));
  }
  assert.match(leaderboardHtml, /aria-label="Scrollable bot standings"/);
  assert.match(leaderboardHtml, /aria-label="Scrollable head-to-head matrix"/);
  assert.ok((leaderboardHtml.match(/class="table-scroll[^"]*" tabindex="0"/g) || []).length >= 2);
  assert.ok((leaderboardHtml.match(/<caption>/g) || []).length >= 2);
  assert.match(leaderboardHtml, /<th scope="col">Local CodinGame-style score<\/th>/);
  assert.match(leaderboardSource, /heading\.scope = "row"/);
  assert.match(leaderboardSource, /heading\.scope = "col"/);
  assert.match(leaderboardCss, /\.matrix-scroll\s*\{[^}]*overflow:\s*auto/s);
  assert.match(leaderboardCss, /\.head-to-head-table thead th\s*\{[^}]*position:\s*sticky/s);
  assert.match(leaderboardCss, /@media \(max-width:\s*680px\)/);
});

test("renderer accepts the published contract and rejects inconsistent snapshots", () => {
  assert.equal(renderer.EXPECTED_SCHEMA, fixture.schema);
  assert.equal(renderer.validateResults(fixture), fixture);

  assert.throws(
    () => renderer.validateResults({...fixture, schema: "unknown"}),
    /unsupported or missing/i,
  );
  assert.throws(
    () => renderer.validateResults({
      ...fixture,
      tournament: {...fixture.tournament, entrantCount: 20},
    }),
    /entrant count/i,
  );
  assert.throws(
    () => renderer.validateResults({
      ...fixture,
      standings: [fixture.standings[0], fixture.standings[0], fixture.standings[2]],
    }),
    /duplicate id/i,
  );
});

test("head-to-head lookup derives the reverse row view without mutating the snapshot", () => {
  const direct = renderer.pairwiseLookup(fixture.headToHead, "alpha", "beta");
  const reverse = renderer.pairwiseLookup(fixture.headToHead, "beta", "alpha");
  assert.deepEqual(direct, fixture.headToHead[0]);
  assert.deepEqual(reverse, {
    rowId: "beta",
    columnId: "alpha",
    games: 2,
    wins: 0,
    losses: 2,
    score: 0,
  });
  assert.equal(renderer.pairwiseLookup(fixture.headToHead, "beta", "missing"), null);
  assert.equal(renderer.formatPercent(0.625), "62.5%");
});

test("standings omit an all-zero forfeits column", () => {
  const {document, elements} = createDirectFileDocument();
  assert.equal(renderer.render(fixture, document), true);

  const header = elements.standingsTable.tHead.children[0];
  assert.equal(header.children.length, 8);
  assert.ok(header.children.every((heading) => heading.scope === "col"));
  assert.doesNotMatch(header.textContent, /forfeits/i);
  assert.doesNotMatch(elements.standingsNote.textContent, /forfeits/i);
  assert.ok(elements.standingsTable.tBodies[0].children.every(
    (row) => row.children.length === 8,
  ));
});

test("standings restore the forfeits column when any entrant forfeits", () => {
  const standings = fixture.standings.map((entrant, index) => ({
    ...entrant,
    forfeits: index === 1 ? 1 : 0,
  }));
  const results = {...fixture, standings};
  const {document, elements} = createDirectFileDocument();
  assert.equal(renderer.render(results, document), true);

  const header = elements.standingsTable.tHead.children[0];
  assert.equal(header.children.length, 9);
  assert.ok(header.children.every((heading) => heading.scope === "col"));
  assert.match(header.children[8].textContent, /^Forfeits$/);
  assert.match(elements.standingsNote.textContent, /Forfeits count as ordinary losses/);
  assert.ok(elements.standingsTable.tBodies[0].children.every(
    (row) => row.children.length === 9,
  ));
  const forfeitCells = elements.standingsTable.tBodies[0].children.map(
    (row) => row.children[8],
  );
  assert.equal(forfeitCells[1].textContent, "1");
  assert.match(forfeitCells[1].className, /\bhas-forfeit\b/);
  assert.equal(forfeitCells[0].textContent, "0");
  assert.doesNotMatch(forfeitCells[0].className, /\bhas-forfeit\b/);
});

test("method and caveat copy distinguishes local ratings from live CodinGame scores", () => {
  const copy = `${leaderboardHtml}\n${leaderboardSource}`;
  assert.match(copy, /Local CodinGame-style score/);
  assert.match(copy, /μ − 3σ/);
  assert.match(copy, /1,000 ms[^]*200 ms/);
  assert.match(copy, /does not publish its complete matchmaking and rating configuration/i);
  assert.match(copy, /not[^.]*interchangeable with live CodinGame league scores/i);
  assert.match(copy, /Historical CodinGame results[^.]*never[^.]*local-score column/i);
});

test("checked-in classic scripts render the full direct-file leaderboard without network or Wasm", () => {
  const {document, elements} = createDirectFileDocument();
  const blocked = {network: 0, wasm: 0};
  const forbidNetwork = () => {
    blocked.network += 1;
    throw new Error("network access is forbidden in the frozen leaderboard");
  };
  const context = vm.createContext({
    console,
    document,
    fetch: forbidNetwork,
    XMLHttpRequest: class ForbiddenXMLHttpRequest {
      constructor() {
        forbidNetwork();
      }
    },
    WebSocket: class ForbiddenWebSocket {
      constructor() {
        forbidNetwork();
      }
    },
  });
  Object.defineProperty(context, "WebAssembly", {
    configurable: false,
    get() {
      blocked.wasm += 1;
      throw new Error("WebAssembly is forbidden in the frozen leaderboard");
    },
  });

  vm.runInContext(leaderboardResults, context, {
    filename: urls.leaderboardResults.href,
  });
  vm.runInContext(leaderboardSource, context, {
    filename: urls.leaderboardSource.href,
  });
  document.dispatchDOMContentLoaded();

  const results = context.PAPERSOCCER_CODINGAME_LEADERBOARD_RESULTS;
  assert.equal(results.schema, "papersoccer.codingame-leaderboard-summary.v1");
  assert.equal(results.standings.length, 22);
  assert.equal(context.PaperSoccerCodingameLeaderboard.EXPECTED_SCHEMA, results.schema);
  assert.equal(elements.standingsTable.tBodies[0].children.length, 22);
  const hasForfeits = results.standings.some((entry) => entry.forfeits > 0);
  const standingsColumns = hasForfeits ? 9 : 8;
  assert.equal(
    elements.standingsTable.tHead.children[0].children.length,
    standingsColumns,
  );
  if (hasForfeits) {
    assert.match(elements.standingsTable.tHead.textContent, /forfeits/i);
  } else {
    assert.doesNotMatch(elements.standingsTable.tHead.textContent, /forfeits/i);
  }
  assert.ok(elements.standingsTable.tBodies[0].children.every(
    (row) => row.children.length === standingsColumns,
  ));

  const matrixRows = elements.headToHeadTable.tBodies[0].children;
  assert.equal(matrixRows.length, 22);
  assert.ok(matrixRows.every((row) => row.children.length === 23));
  assert.equal(elements.headToHeadTable.tHead.children[0].children.length, 23);

  assert.equal(elements.aliasContent.children[0].children.length, 22);
  assert.doesNotMatch(elements.aliasContent.textContent, /Alias:/);
  assert.match(elements.aliasContent.textContent, /Submission [0-9a-f]{12}…/);
  assert.match(elements.provenanceContent.textContent, /Runtime|Tournament/);
  assert.match(elements.provenanceContent.textContent, new RegExp(results.tournament.id));
  assert.match(elements.provenanceContent.textContent, /Operating system/);
  const artifactLink = descendants(elements.provenanceContent, "A").find(
    (element) => /canonical raw tournament artifact/i.test(element.textContent),
  );
  assert.ok(artifactLink);
  assert.equal(artifactLink.href, results.tournament.rawResultsUrl);

  assert.equal(elements.leaderboardLoadState.hidden, true);
  assert.equal(elements.leaderboardError.hidden, true);
  assert.equal(elements.leaderboardOverview.getAttribute("aria-busy"), "false");
  assert.equal(blocked.network, 0);
  assert.equal(blocked.wasm, 0);
});
