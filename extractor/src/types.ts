/**
 * The anonymised record produced from one DUTCH report.
 *
 * By design there is no field for name, date of birth, address, accession
 * number or ordering physician: the parser never reads them into the record.
 */
export interface DutchRecord {
  schemaVersion: 1;
  source: {
    lab: "Precision Analytical";
    test: "DUTCH";
    /** Month the report was issued (YYYY-MM); reference ranges change over time. */
    reportMonth: string | null;
    /** Layout version printed on newer reports, e.g. "1.6.0". */
    reportVersion: string | null;
    pages: number;
  };
  patient: {
    sex: "male" | "female" | null;
    ageYears: number | null;
  };
  /** Month of the first urine collection (YYYY-MM). Day and time are dropped. */
  collectionMonth: string | null;
  markers: Marker[];
  checks: Check[];
  /** Problems a human should look at before trusting this record. */
  warnings: string[];
}

export interface Marker {
  /** Stable machine name derived from the printed name, e.g. "5a_dht". */
  id: string;
  /** Name as printed on the report (font-spacing artefacts repaired). */
  name: string;
  /** Section heading the row sits under, e.g. "Androgens and Metabolites". */
  category: string | null;
  value: number;
  /** Present when the lab printed "<" or ">" in front of the value. */
  comparator?: "<" | ">";
  unit: string;
  refLow: number | null;
  refHigh: number | null;
  /** The lab's own verdict, e.g. "Within range", "Below luteal range". */
  status: string | null;
  /** Extra range column on female reports (postmenopausal), kept verbatim. */
  altRange?: string;
  page: number;
}

/** Result of one internal-consistency test on the extracted numbers. */
export interface Check {
  name: string;
  outcome: "pass" | "fail" | "skipped";
  detail: string;
}
