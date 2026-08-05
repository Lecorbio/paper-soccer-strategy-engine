import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const gameHtmlUrl = new URL("../../web/index.html", import.meta.url);
const gameCssUrl = new URL("../../web/styles.css", import.meta.url);
const benchmarkHtmlUrl = new URL(
  "../../web/benchmarks/index.html",
  import.meta.url,
);
const benchmarkCssUrl = new URL(
  "../../web/benchmarks/benchmarks.css",
  import.meta.url,
);
const benchmarkSourceUrl = new URL(
  "../../web/benchmarks/benchmarks.js",
  import.meta.url,
);

const [gameHtml, gameCss, benchmarkHtml, benchmarkCss, benchmarkSource] =
  await Promise.all([
    readFile(gameHtmlUrl, "utf8"),
    readFile(gameCssUrl, "utf8"),
    readFile(benchmarkHtmlUrl, "utf8"),
    readFile(benchmarkCssUrl, "utf8"),
    readFile(benchmarkSourceUrl, "utf8"),
  ]);

await import("../../web/benchmarks/benchmark-results.js");
await import("../../web/benchmarks/benchmarks.js");

const results = globalThis.PaperSoccerBenchmarkResults;
const renderer = globalThis.PaperSoccerBenchmarks;

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function elementWithId(source, tag, id) {
  const escapedId = escapeRegExp(id);
  return source.match(
    new RegExp(`<${tag}\\b(?=[^>]*\\bid=["']${escapedId}["'])[^>]*>`, "i"),
  )?.[0];
}

function assertApproximately(actual, expected) {
  assert.ok(
    Math.abs(actual - expected) < 1e-12,
    `${actual} should be approximately ${expected}`,
  );
}

test("benchmark results is the first game-header action", () => {
  const modeGroup = gameHtml.match(
    /<div class="mode-switch"[^>]*>[\s\S]*?<\/div>/,
  )?.[0];
  assert.ok(modeGroup, "the game header should contain its mode group");
  assert.doesNotMatch(modeGroup, /Benchmark results/);

  const benchmarkLink = gameHtml.match(
    /<a\b(?=[^>]*\bclass="[^"]*\bbenchmark-link\b[^"]*")(?=[^>]*\bhref="benchmarks\/index\.html")[^>]*>[\s\S]*?Benchmark results\s*<\/a>/,
  )?.[0];
  assert.ok(benchmarkLink, "the game header should link to the benchmark overview");
  assert.ok(
    gameHtml.indexOf(benchmarkLink) < gameHtml.indexOf(modeGroup),
    "the benchmark link should appear before the mode group",
  );
  const header = gameHtml.match(/<header\b[^>]*>[\s\S]*?<\/header>/)?.[0];
  assert.ok(header, "the game page should have a header");
  assert.doesNotMatch(header, /Open (?:existing|replay)|id="fileInput"/);
  assert.match(benchmarkLink, /aria-hidden="true"/);
  assert.match(
    gameCss,
    /\.benchmark-link\s*\{[^}]*background:\s*#fff0b8[^}]*font-weight:\s*750/s,
  );
});

test("the benchmark overview returns to the game and stays independent of Wasm", () => {
  assert.match(
    benchmarkHtml,
    /<a\b(?=[^>]*\bclass="[^"]*\bback-link\b[^"]*")(?=[^>]*\bhref="\.\.\/index\.html")[^>]*>[\s\S]*?Back to game\s*<\/a>/,
  );
  const localBackTargets = Array.from(
    benchmarkHtml.matchAll(/<a\b[^>]*\bhref="(\.\.[^"]*)"[^>]*>/g),
    (match) => match[1],
  );
  assert.ok(localBackTargets.length >= 2);
  assert.ok(localBackTargets.every((target) => target === "../index.html"));

  const scriptSources = Array.from(
    benchmarkHtml.matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*><\/script>/g),
    (match) => match[1],
  );
  assert.deepEqual(scriptSources, ["benchmark-results.js", "benchmarks.js"]);
  assert.doesNotMatch(
    benchmarkHtml,
    /papersoccer-wasm|game-engine|board-view|app-support|(?:^|["/])app\.js/i,
  );
  assert.doesNotMatch(benchmarkHtml, /<canvas\b/i);
  assert.doesNotMatch(benchmarkHtml, /<script\b[^>]*\btype="module"/i);
});

test("pairwise lookup preserves direct results and correctly inverts the reverse view", () => {
  assert.equal(results?.schema, "papersoccer.benchmark-summary.v1");
  assert.equal(typeof renderer?.pairwiseLookup, "function");

  const matchup = results.matchups[0];
  const direct = renderer.pairwiseLookup(
    results.matchups,
    matchup.leftId,
    matchup.rightId,
  );
  const reverse = renderer.pairwiseLookup(
    results.matchups,
    matchup.rightId,
    matchup.leftId,
  );

  assert.deepEqual(direct, matchup);
  assert.equal(reverse.leftId, matchup.rightId);
  assert.equal(reverse.rightId, matchup.leftId);
  assertApproximately(reverse.leftScore, 1 - matchup.leftScore);
  assertApproximately(reverse.leftScoreLower, 1 - matchup.leftScoreUpper);
  assertApproximately(reverse.leftScoreUpper, 1 - matchup.leftScoreLower);
  assert.equal(reverse.pairs, matchup.pairs);
  assert.equal(reverse.games, matchup.games);
  assert.equal(reverse.classification, matchup.classification);
  assert.equal(reverse.strongerId, matchup.strongerId);
});

test("the neural and Rank5 result is presented as statistically unresolved", () => {
  const unresolved = renderer.pairwiseLookup(
    results.matchups,
    "jacek-20k",
    "rank5-fixed-50k",
  );
  assert.ok(unresolved, "the snapshot should preserve the unresolved neural/Rank5 result");
  assert.equal(unresolved.classification, "statistically_unresolved");
  assert.equal(unresolved.strongerId, null);
  assertApproximately(unresolved.leftScore, 0.51375);
  assertApproximately(unresolved.leftScoreLower, 0.4825);
  assertApproximately(unresolved.leftScoreUpper, 0.545);
  assert.equal(renderer.matchupStatus(unresolved), "Statistically unresolved");
  assert.match(results.study.headline, /statistically unresolved/i);
  assert.match(
    `${benchmarkHtml}\n${benchmarkSource}`,
    /statistically unresolved/i,
  );
  assert.doesNotMatch(
    `${benchmarkHtml}\n${benchmarkSource}`,
    /(?:neural[^.]{0,80}defeats?[^.]{0,40}rank5|rank5[^.]{0,80}defeats?[^.]{0,40}neural)/i,
  );
});

test("Rank5 validation strength is labeled as a defined reference", () => {
  const rank5 = results.entrants.find((entrant) =>
    entrant.id === "rank5-fixed-50k");
  assert.ok(rank5);
  assert.equal(rank5.validation.strength, 0.5);
  assert.equal(rank5.validation.strengthIsReference, true);
  assert.equal(rank5.validation.pairs, null);
  assert.equal(
    renderer.validationStrengthLabel(rank5.validation),
    "50.0% defined reference",
  );
  assert.equal(
    renderer.validationStrengthLabel(
      results.entrants.find((entrant) => entrant.id === "jacek-20k").validation,
    ),
    "55.5%",
  );
  assert.match(
    `${benchmarkHtml}\n${benchmarkSource}\n${results.caveats.validationReference}`,
    /defined(?: common-opponent)? reference[^.]*not (?:an )?(?:independently )?observed/i,
  );
});

test("the Pareto plot uses aligned, readable axis scales", () => {
  assert.equal(typeof renderer.paretoAxisModel, "function");
  const axes = renderer.paretoAxisModel(results);

  assert.equal(axes.xMaximum, 150);
  assert.deepEqual(
    axes.xTicks.map((tick) => tick.label),
    ["0", "25", "50", "75", "100", "125", "150"],
  );
  assert.equal(axes.yMaximum, 0.6);
  assert.deepEqual(
    axes.yTicks.map((tick) => tick.label),
    ["0%", "10%", "20%", "30%", "40%", "50%", "60%"],
  );

  for (const ticks of [axes.xTicks, axes.yTicks]) {
    assert.equal(ticks[0].position, 0);
    assert.equal(ticks[0].anchor, "start");
    assert.equal(ticks.at(-1).position, 100);
    assert.equal(ticks.at(-1).anchor, "end");
    assert.ok(ticks.slice(1).every((tick, index) =>
      tick.position > ticks[index].position));
  }

  assert.match(benchmarkSource, /pareto-grid-line grid-x/);
  assert.match(benchmarkSource, /pareto-grid-line grid-y/);
  const paretoAreaCss = benchmarkCss.match(/\.pareto-area\s*\{[^}]*\}/s)?.[0];
  assert.ok(paretoAreaCss);
  assert.doesNotMatch(paretoAreaCss, /background-(?:image|size)/);
  assert.match(
    benchmarkCss,
    /\.pareto-chart\s*\{[^}]*grid-template-columns:\s*18px 30px minmax\(0,\s*1fr\)/s,
  );
  assert.match(
    benchmarkCss,
    /\.pareto-area\s*\{[^}]*grid-column:\s*3/s,
  );
  assert.match(
    benchmarkCss,
    /\.pareto-axis-label\.axis-y\.is-start\s*\{[^}]*translate\(-100%,\s*0\)/s,
  );
});

test("a failed render replaces the loading state with an accessible error", () => {
  const elements = {
    benchmarkError: { hidden: true },
    benchmarkLoadState: { hidden: false },
  };
  const documentStub = {
    getElementById(id) {
      return elements[id] ?? null;
    },
  };

  assert.equal(renderer.render({ schema: "unsupported" }, documentStub), false);
  assert.equal(elements.benchmarkLoadState.hidden, true);
  assert.equal(elements.benchmarkError.hidden, false);
});

test("calibration is described as test-tournament evidence, not validation data", () => {
  assert.match(
    benchmarkSource,
    /Calibration measures[^.]*frozen test (?:set|tournament)/i,
  );
  assert.match(benchmarkSource, /Frozen test calibration metrics/);
  assert.doesNotMatch(benchmarkSource, /validation calibration metrics/i);
});

test("the overview provides semantic, accessible, responsive result structures", () => {
  assert.match(benchmarkHtml, /<html\b[^>]*\blang="en"/i);
  assert.match(
    benchmarkHtml,
    /<meta\b(?=[^>]*\bname="viewport")(?=[^>]*\bcontent="width=device-width, initial-scale=1")[^>]*>/i,
  );
  assert.ok(elementWithId(benchmarkHtml, "main", "benchmarkOverview"));
  assert.ok(elementWithId(benchmarkHtml, "div", "benchmarkError"));
  assert.match(elementWithId(benchmarkHtml, "div", "benchmarkError"), /role="alert"/i);
  assert.ok(elementWithId(benchmarkHtml, "dl", "benchmarkStats"));
  assert.ok(elementWithId(benchmarkHtml, "div", "abilityChart"));
  assert.ok(elementWithId(benchmarkHtml, "table", "headToHeadTable"));
  assert.ok(elementWithId(benchmarkHtml, "div", "paretoPlot"));
  assert.ok(elementWithId(benchmarkHtml, "details", "paretoDetails"));
  assert.ok(elementWithId(benchmarkHtml, "details", "calibrationDetails"));
  assert.ok(elementWithId(benchmarkHtml, "details", "methodologyDetails"));

  assert.match(benchmarkSource, /setAttribute\("role",\s*"img"\)/);
  assert.match(benchmarkSource, /setAttribute\("aria-label",\s*[^)]+\)/);
  assert.match(benchmarkHtml, /<caption\b/i);
  assert.match(benchmarkHtml, /<th\b[^>]*\bscope="(?:col|row)"/i);
  assert.match(benchmarkHtml, /<summary\b/i);
  assert.match(benchmarkCss, /@media\s*\([^)]*max-width/i);
  assert.match(benchmarkCss, /overflow-x\s*:\s*auto/i);
  assert.match(
    benchmarkCss,
    /\.benchmark-shell\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/s,
  );
  assert.match(
    benchmarkCss,
    /\.benchmark-shell\s*>\s*\*\s*\{[^}]*min-width:\s*0/s,
  );
  assert.match(
    benchmarkCss,
    /\.ability-tick:last-child\s*\{[^}]*translateX\(-100%\)/s,
  );
  assert.match(benchmarkCss, /:focus-visible/i);
});
