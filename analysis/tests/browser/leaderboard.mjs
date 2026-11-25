// Opens the app's page in the locally installed Google Chrome and uses the drug repurposing
// leaderboard as a researcher would (test_web_leaderboard_browser.py runs it). Uses the extractor's
// playwright-core.
//   node leaderboard.mjs <app url> <expected JSON>
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
  const elsewhere = [], asked = [];
  page.on("request", (r) => {
    if (!r.url().startsWith(url)) elsewhere.push(r.url());
    else asked.push(new URL(r.url()).pathname);
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(url + "/");
  await page.waitForSelector("#leaderboard-body:not([hidden])", { timeout: 30_000 });

  await page.waitForSelector(".leaderboard-table tbody tr[data-drug]", { timeout: 5_000 }).catch(() => {});
  check(errors.length === 0, `page errors: ${errors}`); // say why, should the table not be drawn
  const order = () => page.$$eval(".leaderboard-table tbody tr[data-drug]", (rows) => rows.map((r) => r.dataset.drug));
  check(same(await order(), expected.default), `default order ${await order()} not ${expected.default}`);
  const rows = await page.$$eval(".leaderboard-table tbody tr[data-drug]", (rs) => rs.map((r) => ({
    text: r.textContent, chembl: r.querySelector("a.chembl")?.href })));
  check(rows.every((r) => r.text.includes("candidate for study")), "every row is a candidate for study");
  check(rows.every((r) => r.chembl.startsWith("https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL_")), "ChEMBL links");
  const beta = rows[expected.default.indexOf("betadrug")].text;
  check(beta.includes("libido") && beta.includes("raises blood pressure"), `betadrug's risks: ${beta}`);
  const gamma = rows[expected.default.indexOf("gammadrug")].text;
  check(gamma.includes("assumed"), `gammadrug's penalty is marked assumed: ${gamma}`);

  const before = asked.length;
  for (const [key, value] of [["impact", "0"], ["coverage", "0"], ["bbb", "0"], ["readiness", "0"], ["safety", "100"]]) {
    await page.fill(`#topsis-${key}`, value); // a range input: fill sets it and fires input
  }
  check(same(await order(), expected.safety_only), `safety-only order ${await order()} not ${expected.safety_only}`);
  check((await page.textContent("#topsis-safety-value")) === "100%", "safety weight shown as 100%");
  check(asked.length === before, `a slider asked the server: ${asked.slice(before)}`);
  await page.click("#topsis-reset");
  check(same(await order(), expected.default), "default weights restore the order");

  const known = await page.textContent("#known-causes");
  check(known.includes("SERTRALINE") && known.includes("SSRI"), `known causes: ${known}`);
  check(!(await order()).includes("SERTRALINE"), "a known cause is not ranked");
  check((await page.textContent("#leaderboard-disclaimer")).includes("not medical advice"), "the disclaimer");

  await page.fill("#leaderboard-filter", "comt");
  check(same(await order(), expected.default.filter((n) => n !== "alphadrug")), `filtered by COMT: ${await order()}`);
  await page.fill("#leaderboard-filter", "");
  await page.selectOption("#leaderboard-syndrome", "PSSD");
  check(same(await order(), expected.default.filter((n) => n !== "betadrug")), `filtered by PSSD: ${await order()}`);
  await page.selectOption("#leaderboard-syndrome", "");
  await page.click("#leaderboard-sort button[data-sort=name]");
  check(same(await order(), [...expected.default].sort()), `sorted by name: ${await order()}`);

  await page.click(".leaderboard-table tr[data-drug=gammadrug] button.details");
  await page.click(".leaderboard-table tr.detail button[data-link='post-finasteride syndrome→COMT']");
  await page.waitForSelector("#leaderboard-panel .edge-passages li");
  const passages = await page.textContent("#leaderboard-panel");
  check(passages.includes("BDNF and COMT changed in PFS."), `the passages behind COMT: ${passages}`);
  check((await page.textContent("#leaderboard-panel h3")).startsWith("PFS – COMT"), "the passage panel's heading");

  check(errors.length === 0, `page errors: ${errors}`);
  check(elsewhere.length === 0, `requests outside the app: ${elsewhere}`);
  console.log("PASSED");
} finally {
  await browser.close();
}
