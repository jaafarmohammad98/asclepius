// Usage: tsx scripts/accuracy.ts [folder]   (default: ../samples)
// For every folder below it that holds both report.pdf and expected.json,
// compares the extractor's output with the human reading and prints the figures.
// The expected.json format is described in src/accuracy.ts.
import { existsSync } from "node:fs";
import { readdir, readFile } from "node:fs/promises";
import { join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { extractDutch } from "../src/index.ts";
import { compareMarkers, parseExpected, percent, totalAccuracy } from "../src/accuracy.ts";
import type { ReportAccuracy } from "../src/accuracy.ts";

const root = resolve(
  process.argv[2] ?? fileURLToPath(new URL("../../samples/", import.meta.url)),
);

/** Folders under `dir` (at any depth) that contain a report and its expected values. */
async function findCases(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const here =
    existsSync(join(dir, "report.pdf")) && existsSync(join(dir, "expected.json")) ? [dir] : [];
  const below = await Promise.all(
    entries.filter((e) => e.isDirectory()).map((e) => findCases(join(dir, e.name))),
  );
  return [...here, ...below.flat()].sort();
}

if (!existsSync(root)) {
  console.error(`Folder not found: ${root}\nRun "npm run samples" to download the public sample reports.`);
  process.exit(1);
}
const cases = await findCases(root);
if (cases.length === 0) {
  console.error(`No folder under ${root} has both report.pdf and expected.json.`);
  process.exit(1);
}

const results: ReportAccuracy[] = [];
let broken = 0;
for (const dir of cases) {
  const label = relative(root, dir) || ".";
  let r: ReportAccuracy;
  try {
    const expected = parseExpected(JSON.parse(await readFile(join(dir, "expected.json"), "utf8")));
    const record = await extractDutch(new Uint8Array(await readFile(join(dir, "report.pdf"))));
    r = compareMarkers(expected, record.markers);
  } catch (err) {
    broken++;
    console.log(`${label}\n  error   ${(err as Error).message}`);
    continue;
  }
  results.push(r);
  console.log(
    `${label.padEnd(20)} ${percent(r.matched, r.expected).padStart(6)}  ` +
      `${r.matched}/${r.expected} matched, ${r.missed.length} missed, ${r.wrong.length} wrong, ${r.extra.length} extra`,
  );
  for (const id of r.missed) console.log(`  missed  ${id}`);
  for (const w of r.wrong) console.log(`  wrong   ${w.id}: ${w.differences.join("; ")}`);
  for (const id of r.extra) console.log(`  extra   ${id}`);
}

const t = totalAccuracy(results);
console.log(
  `\nOverall: ${percent(t.matched, t.expected)} ` +
    `(${t.matched} of ${t.expected} values in ${results.length} report${results.length === 1 ? "" : "s"}; ` +
    `${t.missed} missed, ${t.wrong} wrong, ${t.extra} extra)`,
);
console.log(
  "A value counts as matched when its number, unit and reference range all equal the human reading.",
);
if (broken > 0) {
  console.error(`${broken} folder${broken === 1 ? "" : "s"} could not be measured (see "error" above).`);
  process.exit(1);
}
