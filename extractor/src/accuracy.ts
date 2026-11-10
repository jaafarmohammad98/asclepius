import { slug } from "./parse.ts";
import type { Marker } from "./types.ts";

/**
 * Accuracy measurement: compare what the extractor read with what a human
 * read from the same pages (`expected.json`, stored next to the report).
 *
 * expected.json looks like this, one entry per result row on the report:
 *
 *   { "markers": [
 *       { "id": "testosterone", "value": 46.8, "unit": "ng/mg", "range": "25 - 115" },
 *       { "name": "5a-DHT", "value": 7.9, "unit": "ng/mg", "range": "5 - 25" }
 *   ] }
 *
 * - `id` is the extractor's marker id; `name` (as printed) may be given instead.
 * - `range` is the reference range as printed, or null when none is printed.
 * - `comparator` ("<" or ">") is added when the lab printed one before the value.
 * - A marker printed in two tables (DHEA-S) is listed once.
 */
export interface ExpectedMarker {
  id: string;
  value: number;
  comparator?: "<" | ">";
  unit: string;
  refLow: number | null;
  refHigh: number | null;
}

export interface Mismatch {
  id: string;
  /** One entry per field that differs, e.g. `value: expected 46.9, got 46.8`. */
  differences: string[];
}

export interface ReportAccuracy {
  /** Number of values the human read. */
  expected: number;
  /** Read by the extractor with the same value, unit and range. */
  matched: number;
  /** On the page, absent from the extractor output. */
  missed: string[];
  /** Read by the extractor, but something differs from the page. */
  wrong: Mismatch[];
  /** In the extractor output, not on the page according to the human. */
  extra: string[];
}

const RANGE = /^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$/;

/** Validate the contents of an expected.json file. Throws with the reason when it is malformed. */
export function parseExpected(json: unknown): ExpectedMarker[] {
  const list = (json as { markers?: unknown } | null)?.markers;
  if (!Array.isArray(list)) throw new Error('expected.json must be { "markers": [...] }');

  const seen = new Set<string>();
  return list.map((raw, i) => {
    const e = (raw ?? {}) as Record<string, unknown>;
    const id =
      typeof e.id === "string" ? e.id : typeof e.name === "string" ? slug(e.name) : "";
    const where = `markers[${i}]${id ? ` (${id})` : ""}`;
    if (!id) throw new Error(`${where}: needs an "id" or a "name"`);
    if (seen.has(id)) throw new Error(`${where}: listed twice`);
    seen.add(id);
    if (typeof e.value !== "number" || !Number.isFinite(e.value)) {
      throw new Error(`${where}: "value" must be a number`);
    }
    if (typeof e.unit !== "string" || !e.unit) throw new Error(`${where}: "unit" is missing`);
    if (e.comparator !== undefined && e.comparator !== "<" && e.comparator !== ">") {
      throw new Error(`${where}: "comparator" must be "<" or ">"`);
    }
    let refLow: number | null = null;
    let refHigh: number | null = null;
    if (e.range !== null) {
      const m = typeof e.range === "string" ? RANGE.exec(e.range.trim()) : null;
      if (!m) throw new Error(`${where}: "range" must look like "25 - 115", or be null`);
      refLow = Number(m[1]);
      refHigh = Number(m[2]);
    }
    const marker: ExpectedMarker = { id, value: e.value, unit: e.unit, refLow, refHigh };
    if (e.comparator) marker.comparator = e.comparator;
    return marker;
  });
}

type Comparable = Pick<Marker, "value" | "comparator" | "unit" | "refLow" | "refHigh">;

const showValue = (m: Comparable) => `${m.comparator ?? ""}${m.value}`;
const showRange = (m: Comparable) =>
  m.refLow === null || m.refHigh === null ? "none" : `${m.refLow} - ${m.refHigh}`;

/** Compare one report's extracted markers with the human reading. */
export function compareMarkers(expected: ExpectedMarker[], extracted: Marker[]): ReportAccuracy {
  const byId = new Map(extracted.map((m) => [m.id, m]));
  const result: ReportAccuracy = {
    expected: expected.length,
    matched: 0,
    missed: [],
    wrong: [],
    extra: [],
  };

  for (const want of expected) {
    const got = byId.get(want.id);
    if (!got) {
      result.missed.push(want.id);
      continue;
    }
    const differences: string[] = [];
    const diff = (field: string, show: (m: Comparable) => string) => {
      if (show(want) !== show(got)) {
        differences.push(`${field}: expected ${show(want)}, got ${show(got)}`);
      }
    };
    diff("value", showValue);
    diff("unit", (m) => m.unit);
    diff("range", showRange);
    if (differences.length) result.wrong.push({ id: want.id, differences });
    else result.matched++;
  }

  const wanted = new Set(expected.map((e) => e.id));
  result.extra = extracted.filter((m) => !wanted.has(m.id)).map((m) => m.id);
  return result;
}

/** Add up the figures of several reports. */
export function totalAccuracy(reports: ReportAccuracy[]) {
  const sum = (pick: (r: ReportAccuracy) => number) => reports.reduce((s, r) => s + pick(r), 0);
  return {
    expected: sum((r) => r.expected),
    matched: sum((r) => r.matched),
    missed: sum((r) => r.missed.length),
    wrong: sum((r) => r.wrong.length),
    extra: sum((r) => r.extra.length),
  };
}

/** Share of the human-read values that the extractor read correctly, e.g. "97.9%". */
export function percent(matched: number, expected: number): string {
  return expected === 0 ? "n/a" : `${((100 * matched) / expected).toFixed(1)}%`;
}
