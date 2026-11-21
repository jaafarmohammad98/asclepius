import { toCsv } from "./csv.ts";
import { findDuplicates } from "./duplicates.ts";
import type { DutchRecord } from "./types.ts";

/**
 * How one file in a batch fared:
 * - read: result rows were found
 * - unreadable: it opened, but no result tables could be read (warnings say why)
 * - failed: it could not be opened at all (not a PDF, damaged, password-protected)
 */
export type ReportState = "read" | "unreadable" | "failed";

export interface BatchReport {
  /** Running number in the batch, e.g. "r002". Never derived from the file name. */
  id: string;
  state: ReportState;
  record: DutchRecord | null;
  error: string | null;
}

export interface Batch {
  reports: BatchReport[];
  /** One group of record ids per report that appears more than once. */
  duplicates: string[][];
}

/** Turn the per-file results of one batch, in the order given, into what the user reviews. */
export function summariseBatch(results: (DutchRecord | Error)[]): Batch {
  const reports = results.map((result, i): BatchReport => {
    const id = `r${String(i + 1).padStart(3, "0")}`;
    if (result instanceof Error) return { id, state: "failed", record: null, error: result.message };
    return { id, state: result.markers.length > 0 ? "read" : "unreadable", record: result, error: null };
  });
  const opened = reports.filter((r) => r.record !== null);
  const duplicates = findDuplicates(opened.map((r) => r.record!)).map((group) => group.map((i) => opened[i].id));
  return { reports, duplicates };
}

/** The reports that opened, as `{ id, record }` pairs ready to download. */
function downloadable(batch: Batch): { id: string; record: DutchRecord }[] {
  return batch.reports.flatMap((r) => (r.record ? [{ id: r.id, record: r.record }] : []));
}

/** JSON download: an array of `{ id, record }`, unreadable reports included with their warnings. */
export function batchToJson(batch: Batch): string {
  return JSON.stringify(downloadable(batch), null, 2);
}

/** CSV download: one row per marker per report. */
export function batchToCsv(batch: Batch): string {
  return toCsv(downloadable(batch));
}
