import { describe, expect, it } from "vitest";
import { batchToCsv, batchToJson, NO_TABLES_WARNING, parseDutch, summariseBatch } from "../src/index.ts";
import type { TextLine } from "../src/pdfText.ts";

/** Build a record from invented report text; "|" marks a column gap. */
function report({ testosterone = "1.5" } = {}) {
  const text = (page: number, s: string): TextLine[] =>
    s
      .trim()
      .split("\n")
      .map((t, i) => ({ page, y: 800 - i * 10, items: [], text: t.trim().replace(/\s*\|\s*/g, "\t") }));
  return parseDutch([
    ...text(
      1,
      `
Ordering Physician: | DOB: 1980-02-03 | Collection Times:
2024-03-09 06:01AM
Precision Analytical | Age: 44
Gender: Female | 2024-03-09 10:01PM
`,
    ),
    ...text(
      2,
      `
Category | Test | Result | Units | Normal Range
Androgens and Metabolites (Urine)
Testosterone | Below range | ${testosterone} | ng/mg | 2.3 - 14
`,
    ),
  ]);
}

const unreadable = parseDutch([]);

describe("summariseBatch", () => {
  it("numbers reports by their position and sorts them into read, unreadable and failed", () => {
    const batch = summariseBatch([report(), unreadable, new Error("Invalid PDF structure.")]);
    expect(batch.reports.map((r) => [r.id, r.state])).toEqual([
      ["r001", "read"],
      ["r002", "unreadable"],
      ["r003", "failed"],
    ]);
    expect(batch.reports[2]).toMatchObject({ error: "Invalid PDF structure.", record: null });
  });

  it("lists each repeated report as a group of record ids", () => {
    const batch = summariseBatch([report(), report({ testosterone: "9.9" }), new Error("x"), report()]);
    expect(batch.duplicates).toEqual([["r001", "r004"]]);
  });

  it("downloads every report that opened, by record id, and leaves out files that failed", () => {
    const batch = summariseBatch([new Error("x"), report(), unreadable]);
    const json = JSON.parse(batchToJson(batch));
    expect(json.map((r: { id: string }) => r.id)).toEqual(["r002", "r003"]);
    expect(json[1].record.warnings).toContain(NO_TABLES_WARNING);
    expect(batchToCsv(batch).trim().split("\n").slice(1)).toEqual([
      "r002,female,44,2024-03,,testosterone,Testosterone,Androgens and Metabolites,,1.5,ng/mg,2.3,14,Below range",
    ]);
  });
});
