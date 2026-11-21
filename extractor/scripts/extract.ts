// Usage: tsx scripts/extract.ts [--csv] <report.pdf> [more.pdf ...]
// Prints the anonymised record(s) as JSON, or as one long CSV with --csv.
import { readFile } from "node:fs/promises";
import { batchToCsv, batchToJson, extractDutch, summariseBatch, type DutchRecord } from "../src/index.ts";

const args = process.argv.slice(2);
const csv = args.includes("--csv");
const files = args.filter((a) => !a.startsWith("--"));
if (files.length === 0) {
  console.error("usage: tsx scripts/extract.ts [--csv] <report.pdf> [more.pdf ...]");
  process.exit(1);
}

const results: (DutchRecord | Error)[] = [];
for (const file of files) {
  try {
    results.push(await extractDutch(new Uint8Array(await readFile(file))));
  } catch (err) {
    results.push(err instanceof Error ? err : new Error(String(err)));
  }
}
// ids are running numbers, never file names (file names often contain patient names)
const batch = summariseBatch(results);
for (const [i, r] of batch.reports.entries()) {
  if (r.error) console.error(`[${files[i]}] error: could not open the file: ${r.error}`);
  for (const w of r.record?.warnings ?? []) console.error(`[${files[i]}] warning: ${w}`);
}
const fileOf = (id: string) => files[batch.reports.findIndex((r) => r.id === id)];
for (const group of batch.duplicates) {
  const copies = group.map((id) => `${id} (${fileOf(id)})`).join(", ");
  console.error(`warning: duplicate report, the same values appear in ${copies}`);
}
if (csv) console.log(batchToCsv(batch));
else if (results.length === 1 && batch.reports[0].record) console.log(JSON.stringify(batch.reports[0].record, null, 2));
else console.log(batchToJson(batch));
if (batch.reports.some((r) => r.state === "failed")) process.exitCode = 1;
