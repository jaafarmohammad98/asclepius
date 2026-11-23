// Opens the app's page in the locally installed Google Chrome after a run has added its findings to
// the knowledge graph, and opens the solid finding line (test_web_bottleneck_browser.py runs it).
//   node finding_link.mjs <app url> <expected JSON>
// Exits 77 when Chrome is not installed, 1 with a message at the first check that fails.
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(fileURLToPath(new URL("../../../extractor/package.json", import.meta.url)));
const { chromium } = require("playwright-core");
const [url, expectedJson] = process.argv.slice(2);
const expected = JSON.parse(expectedJson);

function check(ok, message) {
  if (!ok) { console.error(`FAILED: ${message}`); process.exit(1); }
}

let browser;
try {
  browser = await chromium.launch({ channel: "chrome" });
} catch {
  process.exit(77);
}
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(url + "/");
  await page.waitForSelector("#map-body:not([hidden])", { timeout: 30_000 });

  const status = await page.textContent("#map-status");
  check(status.includes(expected.fingerprint), `map status names the run's dataset: ${status}`);
  const id = `${expected.from}→${expected.term}`;
  const line = await page.evaluate((id) => {
    const edge = MAP.cy.getElementById(id);
    return edge.empty() ? null : { kind: edge.data("kind"), style: edge.style("line-style") };
  }, id);
  check(line && line.kind === "finding" && line.style === "solid", `finding line ${id}: ${JSON.stringify(line)}`);
  const legend = await page.textContent("#map-legend");
  check(legend.includes("a solid one"), `legend: ${legend}`);

  await page.evaluate((id) => MAP.cy.getElementById(id).emit("tap"), id);
  await page.waitForSelector("#edge-heading");
  const heading = await page.textContent("#edge-heading");
  check(heading.startsWith(`PSSD – ${expected.term}`) && heading.includes(expected.label), `edge heading: ${heading}`);
  const findings = await page.$$eval("#edge-panel ul li", (items) => items.map((i) => i.textContent));
  check(findings.some((f) => f.startsWith("Androsterone: strong, PSSD against healthy control")), `findings ${findings}`);

  check(errors.length === 0, `page errors: ${errors}`);
  console.log("PASSED");
} finally {
  await browser.close();
}
