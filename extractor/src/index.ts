import { readLines } from "./pdfText.ts";
import { parseDutch } from "./parse.ts";
import type { DutchRecord } from "./types.ts";

export type { DutchRecord, Marker, Check } from "./types.ts";
export { parseDutch } from "./parse.ts";
export { toCsv } from "./csv.ts";

/** Read a DUTCH report PDF and return its anonymised record. */
export async function extractDutch(pdf: Uint8Array): Promise<DutchRecord> {
  return parseDutch(await readLines(pdf));
}
