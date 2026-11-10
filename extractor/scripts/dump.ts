// Usage: tsx scripts/dump.ts <report.pdf>  — prints the text lines the parser sees.
import { readFile } from "node:fs/promises";
import { readLines } from "../src/pdfText.ts";

const path = process.argv[2];
if (!path) throw new Error("usage: tsx scripts/dump.ts <report.pdf>");
const lines = await readLines(new Uint8Array(await readFile(path)));
for (const l of lines) console.log(`p${l.page}\t${l.text}`);
