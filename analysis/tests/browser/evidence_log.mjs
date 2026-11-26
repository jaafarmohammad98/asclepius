// Opens the app's page in the locally installed Google Chrome and uses the evidence log as a researcher
// would (test_web_evidence_log_browser.py runs it): logs a refuting RCT, sees the leaderboard and the map
// re-rank, downloads the log and imports another. Uses the extractor's playwright-core.
//   node evidence_log.mjs <app url> <expected JSON>
// Exits 77 when Chrome is not installed, 1 with a message at the first check that fails.
import { readFile } from "node:fs/promises";
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
  const page = await browser.newPage({ acceptDownloads: true });
  const elsewhere = [];
  page.on("request", (r) => { if (!r.url().startsWith(url)) elsewhere.push(r.url()); });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(url + "/");
  await page.waitForSelector(".leaderboard-table tbody tr[data-drug]", { timeout: 30_000 });
  await page.waitForSelector("#evidence-body:not([hidden])", { timeout: 30_000 });

  const drugs = () => page.$$eval(".leaderboard-table tbody tr[data-drug]", (rs) => rs.map((r) => r.dataset.drug));
  const terms = () => page.$$eval(".bottleneck-table tbody tr[data-term]", (rs) => rs.map((r) => r.dataset.term));
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  check(same(await drugs(), expected.before), `leaderboard before: ${await drugs()}`);
  check((await page.textContent("#evidence-table")).includes("No result logged yet."), "an empty log says so");
  check((await page.textContent("#evidence-provisional")) === "provisional, needs expert review", "the study types are provisional");

  await page.selectOption("#evidence-node", "BDNF");
  await page.selectOption("#evidence-direction", "refutes");
  await page.selectOption("#evidence-study-type", "rct");
  await page.fill("#evidence-source", "PMID 31000009");
  await page.click("#evidence-add");
  await page.waitForSelector("#evidence-table tr[data-term=BDNF]");
  const row = await page.textContent("#evidence-table tr[data-term=BDNF]");
  check(row.includes("Beta(1.75, 1), mean 0.64") && row.includes("Beta(1.75, 5), mean 0.26"), `BDNF row: ${row}`);
  check(row.includes("refutes · randomised controlled trial (adds 4) · PMID:31000009"), `BDNF's entry: ${row}`);
  check(await page.getAttribute("#evidence-table tr[data-term=BDNF] a", "href") === "https://pubmed.ncbi.nlm.nih.gov/31000009/", "the PMID links to PubMed");

  // the done-when of M6-11: the leaderboard order changes, and the map with it
  await page.waitForFunction((after) => JSON.stringify([...document.querySelectorAll(".leaderboard-table tbody tr[data-drug]")]
    .map((r) => r.dataset.drug)) === JSON.stringify(after), expected.after, { timeout: 10_000 })
    .catch(async () => check(false, `leaderboard after: ${await drugs()}`));
  await page.waitForFunction((after) => JSON.stringify([...document.querySelectorAll(".bottleneck-table tbody tr[data-term]")]
    .map((r) => r.dataset.term)) === JSON.stringify(after), expected.map_after, { timeout: 10_000 })
    .catch(async () => check(false, `map after: ${await terms()}`));

  // the same result again is refused
  await page.fill("#evidence-source", "31000009");
  await page.click("#evidence-add");
  await page.waitForSelector("#evidence-form-error:not([hidden])");
  check((await page.textContent("#evidence-form-error")).includes("in the log already"), "a duplicate is refused");

  const [download] = await Promise.all([page.waitForEvent("download"), page.click("a[href='/api/export/evidence-log.json']")]);
  check(download.suggestedFilename() === "asclepius-evidence-log.json", `file name ${download.suggestedFilename()}`);
  const log = JSON.parse(await readFile(await download.path(), "utf-8"));
  check(log.entries.length === 1 && log.entries[0].source === "PMID:31000009", `the downloaded log: ${JSON.stringify(log)}`);

  await page.setInputFiles("#evidence-import", expected.import);
  await page.waitForFunction(() => document.querySelector("#evidence-message").textContent.startsWith("Imported 1 new entry of 1"));
  check((await page.textContent("#evidence-count")).startsWith("2 results logged, on 2 of"), await page.textContent("#evidence-count"));
  const comt = await page.textContent("#evidence-table tr[data-term=COMT]");
  check(comt.includes("supports · cohort study (adds 2) · doi:10.1234/xyz"), `COMT's imported entry: ${comt}`);

  check(errors.length === 0, `page errors: ${errors}`);
  check(elsewhere.length === 0, `requests outside the app: ${elsewhere}`);
  console.log("PASSED");
} finally {
  await browser.close();
}
