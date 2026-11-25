// Opens the app's page in the locally installed Google Chrome and builds a lab experiment as a
// researcher would (test_web_experiment_browser.py runs it): ticks drugs in the leaderboard, builds the
// design on the server, and downloads its CSV. Uses the extractor's playwright-core.
//   node experiment.mjs <app url> <expected JSON>
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
  page.on("request", (r) => { if (!r.url().startsWith(url) && !r.url().startsWith("blob:")) elsewhere.push(r.url()); });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(url + "/");
  await page.waitForSelector(".leaderboard-table tbody tr[data-drug]", { timeout: 30_000 });
  await page.waitForSelector(".bottleneck-table tbody tr[data-term]", { timeout: 30_000 });
  check(await page.isDisabled("#experiment-build"), "no design without two factors");

  // a term ticked in the bottleneck table, and removed again from the factor list
  await page.check(".bottleneck-table tr[data-term=COMT] input.factor");
  check((await page.textContent("#experiment-factors")).includes("COMT"), "COMT is a factor");
  await page.click("#experiment-factors button[aria-label='Remove COMT']");
  check(!(await page.isChecked(".bottleneck-table tr[data-term=COMT] input.factor")), "removing unticks it");

  for (const name of expected.drugs) await page.check(`.leaderboard-table tr[data-drug="${name}"] input.factor`);
  await page.fill("#experiment-seed", String(expected.seed));
  await page.click("#experiment-build");
  await page.waitForSelector("#experiment-design:not([hidden])", { timeout: 10_000 });
  check((await page.textContent("#experiment-label")) === expected.label, `label: ${await page.textContent("#experiment-label")}`);
  check((await page.textContent("#experiment-note")) === expected.note, "the note as the server gave it");
  const rows = await page.$$eval(".experiment-table tbody tr", (rs) => rs.length);
  check(rows === 16, `${rows} runs shown`);
  const risks = await page.textContent("#experiment-risks");
  check(risks.includes("libido; raises blood pressure"), `betadrug's risks: ${risks}`);
  check(risks.includes("no US label: label risks unknown"), `gammadrug's unknown risks: ${risks}`);
  check((risks.match(/candidate for study/g) || []).length === 5, "each factor is a candidate for study");

  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#experiment-download")]);
  check(download.suggestedFilename() === expected.filename, `file name ${download.suggestedFilename()}`);
  const text = await readFile(await download.path(), "utf-8");
  check(text === expected.csv, `the CSV is the server's:\n${text}`);
  const runs = text.split("\n").filter((l) => l && !l.startsWith("#"));
  check(runs.length === 17, `${runs.length - 1} runs in the CSV`);

  // ticking another factor hides the design built for the others
  await page.check(".bottleneck-table tr[data-term=COMT] input.factor");
  check(await page.isHidden("#experiment-design"), "a changed factor list hides the old design");

  check(errors.length === 0, `page errors: ${errors}`);
  check(elsewhere.length === 0, `requests outside the app: ${elsewhere}`);
  console.log("PASSED");
} finally {
  await browser.close();
}
