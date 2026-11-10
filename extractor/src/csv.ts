import type { DutchRecord } from "./types.ts";

const COLUMNS = [
  "record_id",
  "sex",
  "age_years",
  "collection_month",
  "report_month",
  "marker_id",
  "marker_name",
  "category",
  "comparator",
  "value",
  "unit",
  "ref_low",
  "ref_high",
  "status",
] as const;

function cell(v: string | number | null | undefined): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** One row per marker per record ("long" format), ready for pandas or Excel. */
export function toCsv(records: { id: string; record: DutchRecord }[]): string {
  const rows = [COLUMNS.join(",")];
  for (const { id, record } of records) {
    for (const m of record.markers) {
      rows.push(
        [
          id,
          record.patient.sex,
          record.patient.ageYears,
          record.collectionMonth,
          record.source.reportMonth,
          m.id,
          m.name,
          m.category,
          m.comparator,
          m.value,
          m.unit,
          m.refLow,
          m.refHigh,
          m.status,
        ]
          .map(cell)
          .join(","),
      );
    }
  }
  return rows.join("\n") + "\n";
}
