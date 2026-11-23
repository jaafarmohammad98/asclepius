// Opens the app's page in the locally installed Google Chrome and uses the bottleneck map as a
// researcher would (test_web_bottleneck_browser.py runs it). Uses the extractor's playwright-core.
//   node bottleneck_map.mjs <app url> <expected JSON>
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
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

let browser;
try {
  browser = await chromium.launch({ channel: "chrome" });
} catch {
  process.exit(77);
}
try {
  const page = await browser.newPage();
  const elsewhere = [];
  page.on("request", (r) => { if (!r.url().startsWith(url)) elsewhere.push(r.url()); });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(url + "/");
  await page.waitForSelector("#map-body:not([hidden])", { timeout: 30_000 });

  const nodes = await page.evaluate(() => MAP.cy.nodes().map((n) => n.id()));
  check(nodes.includes("BDNF") && nodes.includes("post-SSRI sexual dysfunction"), `network nodes ${nodes}`);
  const colour = await page.evaluate(() => MAP.cy.getElementById("BDNF").style("background-color"));
  const two = await page.evaluate(() => MAP.cy.getElementById("BDNF").data("overlap"));
  check(two === 2 && colour, `BDNF drawn with overlap ${two}, colour ${colour}`);

  const order = () => page.$$eval(".bottleneck-table tbody tr", (rows) => rows.map((r) => r.dataset.term));
  check(same(await order(), expected.default), `default order ${await order()} not ${expected.default}`);
  const firstRow = await page.textContent(".bottleneck-table tbody tr");
  check(firstRow.includes("PFS, PSSD") && firstRow.includes(expected.top_score), `first row: ${firstRow}`);

  for (const [id, value] of [["weight-overlap", "0"], ["weight-betweenness", "0"], ["weight-cascade", "100"]]) {
    await page.fill(`#${id}`, value); // a range input: fill sets it and fires input
  }
  check(same(await order(), expected.cascade_only), `cascade-only order ${await order()} not ${expected.cascade_only}`);
  check((await page.textContent("#weight-cascade-value")) === "100%", "cascade weight shown as 100%");
  await page.click("#weights-reset");
  check(same(await order(), expected.default), "default weights restore the order");

  check(await page.evaluate(() => MAP.cy.getElementById("BDNF|serotonin").empty()), "2-paper line hidden by default");
  await page.fill("#map-min-papers", "1");
  await page.dispatchEvent("#map-min-papers", "change");
  check((await page.textContent("#map-count")).includes("2 co-mentions"), await page.textContent("#map-count"));
  await page.evaluate(() => MAP.cy.getElementById("BDNF|serotonin").emit("tap"));
  await page.waitForSelector("#edge-heading");
  const heading = await page.textContent("#edge-heading");
  check(heading.includes("co-mentioned in 2 papers"), `edge heading: ${heading}`);
  const passages = await page.$$eval(".edge-passages li", (items) => items.map((i) => i.textContent));
  check(passages.length === 2 && passages.some((p) => p.includes("BDNF and serotonin fell in PSSD.")), `passages ${passages}`);
  check(passages.some((p) => p.includes("PubMed 31000001")), "a passage links to its PubMed record");

  await page.evaluate(() => MAP.cy.getElementById("post-finasteride syndrome→COMT").emit("tap"));
  await page.waitForFunction(() => document.querySelector("#edge-heading")?.textContent.startsWith("PFS – COMT"));

  check(errors.length === 0, `page errors: ${errors}`);
  check(elsewhere.length === 0, `requests outside the app: ${elsewhere}`);
  console.log("PASSED");
} finally {
  await browser.close();
}
