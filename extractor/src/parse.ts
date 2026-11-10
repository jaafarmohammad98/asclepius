import type { TextLine } from "./pdfText.ts";
import type { DutchRecord, Marker } from "./types.ts";
import { runChecks } from "./checks.ts";

const VALUE = /^([<>])?\s*(\d+(?:\.\d+)?)$/;
const UNIT = /^[a-zµμ]+\/[a-zµμ]+$/i;
const RANGE = /^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$/;
const STATUS = /\b((?:Within|Above|Below|Low end of|High end of)\b.*\brange)$/i;

/** Repair font-kerning gaps such as "T estosterone" and "(W aking)". */
export function cleanName(raw: string): string {
  return raw
    .replace(/\b([TVWY]) (?=[a-z]{2})/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

export function slug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function toMonth(iso: string): string {
  return iso.slice(0, 7);
}

/** Try to read one line as a result-table row. */
export function parseRow(line: TextLine, category: string | null): Marker | null {
  const cells = line.text.split("\t").map((c) => c.trim());
  // find "<value> <unit>" side by side
  let vi = -1;
  for (let i = 1; i < cells.length - 1; i++) {
    if (VALUE.test(cells[i]) && UNIT.test(cells[i + 1])) {
      vi = i;
      break;
    }
  }
  if (vi < 1) return null;

  let nameCell = cells[0];
  let status: string | null = cells.slice(1, vi).join(" ") || null;
  if (status === null) {
    // name and status ended up in one cell
    const m = STATUS.exec(nameCell);
    if (m) {
      status = m[1];
      nameCell = nameCell.slice(0, m.index).trim();
    }
  }
  const name = cleanName(nameCell);
  if (!name) return null;

  const [, comparator, num] = VALUE.exec(cells[vi])!;
  const range = RANGE.exec(cells[vi + 2] ?? "");
  const marker: Marker = {
    id: slug(name),
    name,
    category,
    value: Number(num),
    unit: cells[vi + 1],
    refLow: range ? Number(range[1]) : null,
    refHigh: range ? Number(range[2]) : null,
    status: status ? cleanName(status) : null,
    page: line.page,
  };
  if (comparator) marker.comparator = comparator as "<" | ">";
  const alt = cells[vi + 3];
  if (alt) marker.altRange = alt;
  return marker;
}

function parseCategory(line: TextLine): string | null {
  const first = line.text.split("\t")[0].trim();
  const m = /^(.*?)\s*(?:-\s*)?\(Urine\)$/.exec(first);
  return m ? cleanName(m[1]) : null;
}

/** Turn the text lines of a DUTCH report into an anonymised record. */
export function parseDutch(lines: TextLine[]): DutchRecord {
  const warnings: string[] = [];
  const all = lines.map((l) => l.text).join("\n");

  if (!/Precision Analytical|DutchTest\.com|DUTCH/i.test(all)) {
    warnings.push("This does not look like a DUTCH report from Precision Analytical.");
  }

  // --- header facts (only the non-identifying ones) ---
  const sexMatch = /(?:Gender|Sex):\s*(Male|Female)\b/i.exec(all);
  const ageMatch = /\bAge:\s*(\d{1,3})\b/.exec(all);
  const collected = [...all.matchAll(/\b(\d{4}-\d{2}-\d{2}) \d{1,2}:\d{2}\s*[AP]M\b/g)]
    .map((m) => m[1])
    .sort();
  const reportDate =
    /Final Report\s*-\s*(\d{1,2})\/(\d{1,2})\/(\d{4})/i.exec(all) ??
    /\b(\d{2})\/(\d{2})\/(\d{4})\tDutchTest\.com/.exec(all);
  const version = /Report Version\s+([\d.]+)/i.exec(all);

  if (!sexMatch) warnings.push("Sex not found on the report.");
  if (!ageMatch) warnings.push("Age not found on the report.");
  if (collected.length === 0) warnings.push("Collection date not found on the report.");

  // --- result tables ---
  const markers: Marker[] = [];
  const seen = new Map<string, Marker>();
  let category: string | null = null;
  let categoryPage = -1;
  for (const line of lines) {
    const cat = parseCategory(line);
    if (cat !== null) {
      category = cat;
      categoryPage = line.page;
      continue;
    }
    const row = parseRow(line, line.page === categoryPage ? category : null);
    if (!row) continue;
    if (row.refLow === null) {
      warnings.push(`${row.name}: reference range could not be read.`);
    }
    const prev = seen.get(row.id);
    if (prev) {
      // some markers (e.g. DHEA-S) are printed in two tables
      if (prev.value !== row.value || prev.unit !== row.unit) {
        warnings.push(
          `${row.name} appears twice with different values (${prev.value} on page ${prev.page}, ${row.value} on page ${row.page}).`,
        );
      }
      continue;
    }
    seen.set(row.id, row);
    markers.push(row);
  }

  if (markers.length === 0) {
    warnings.push(
      "No result tables could be read. The report may be a scan, or use a layout this version does not support.",
    );
  }

  const pages = lines.reduce((max, l) => Math.max(max, l.page), 0);
  const checks = runChecks(markers);
  for (const c of checks) {
    if (c.outcome === "fail") warnings.push(`Consistency check failed: ${c.name} (${c.detail}).`);
  }

  return {
    schemaVersion: 1,
    source: {
      lab: "Precision Analytical",
      test: "DUTCH",
      reportMonth: reportDate
        ? `${reportDate[3]}-${reportDate[1].padStart(2, "0")}`
        : null,
      reportVersion: version ? version[1].replace(/\.$/, "") : null,
      pages,
    },
    patient: {
      sex: sexMatch ? (sexMatch[1].toLowerCase() as "male" | "female") : null,
      ageYears: ageMatch ? Number(ageMatch[1]) : null,
    },
    collectionMonth: collected.length ? toMonth(collected[0]) : null,
    markers,
    checks,
    warnings,
  };
}
