// Usage: tsx scripts/extract.ts [--csv] <report.pdf> [more.pdf ...]
// Prints the anonymised record(s) as JSON, or as one long CSV with --csv.
import { readFile } from "node:fs/promises";
import { extractDutch, findDuplicates, toCsv, type DutchRecord } from "../src/index.ts";

const args = process.argv.slice(2);
const csv = args.includes("--csv");
const files = args.filter((a) => !a.startsWith("--"));
if (files.length === 0) {
  console.error("usage: tsx scripts/extract.ts [--csv] <report.pdf> [more.pdf ...]");
  process.exit(1);
}

const records: { id: string; record: DutchRecord }[] = [];
for (const [i, file] of files.entries()) {
  const record = await extractDutch(new Uint8Array(await readFile(file)));
  // the id is a running number, never the file name (file names often contain patient names)
  records.push({ id: `r${String(i + 1).padStart(3, "0")}`, record });
  for (const w of record.warnings) console.error(`[${file}] warning: ${w}`);
}
for (const group of findDuplicates(records.map((r) => r.record))) {
  const copies = group.map((i) => `${records[i].id} (${files[i]})`).join(", ");
  console.error(`warning: duplicate report, the same values appear in ${copies}`);
}
console.log(csv ? toCsv(records) : JSON.stringify(records.length === 1 ? records[0].record : records, null, 2));
