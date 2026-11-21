import { describe, expect, it } from "vitest";
import { findDuplicates, fingerprint, parseDutch } from "../src/index.ts";
import type { TextLine } from "../src/pdfText.ts";

/** Build a record from invented report text; "|" marks a column gap. */
function report({ age = 44, gender = "Female", testosterone = "1.5" } = {}) {
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
Precision Analytical | Age: ${age}
Gender: ${gender} | 2024-03-09 10:01PM
`,
    ),
    ...text(
      2,
      `
Category | Test | Result | Units | Normal Range
Androgens and Metabolites (Urine)
Testosterone | Below range | ${testosterone} | ng/mg | 2.3 - 14
5a-DHT | Above range | 7.2 | ng/mg | 0 - 6.6
`,
    ),
  ]);
}

describe("findDuplicates", () => {
  it("groups identical reports by their position in the batch", () => {
    expect(findDuplicates([report(), report({ testosterone: "9.9" }), report()])).toEqual([[0, 2]]);
  });

  it("does not match reports that differ in one value", () => {
    expect(findDuplicates([report(), report({ testosterone: "1.6" })])).toEqual([]);
  });

  it("does not match reports with the same values but a different age or sex", () => {
    expect(findDuplicates([report(), report({ age: 45 }), report({ gender: "Male" })])).toEqual([]);
  });

  it("matches the same report whatever order its rows were read in", () => {
    const a = report();
    const b = { ...a, markers: [...a.markers].reverse() };
    expect(findDuplicates([a, b])).toEqual([[0, 1]]);
  });

  it("never groups reports that had no readable markers", () => {
    const empty = parseDutch([]);
    expect(findDuplicates([empty, empty])).toEqual([]);
  });
});

describe("fingerprint", () => {
  it("is a short hash that exposes none of the values it was built from", () => {
    expect(fingerprint(report())).toMatch(/^[0-9a-f]{16}$/);
  });

  it("is the same for identical reports and changes when a value changes", () => {
    expect(fingerprint(report())).toBe(fingerprint(report()));
    expect(fingerprint(report())).not.toBe(fingerprint(report({ testosterone: "1.6" })));
  });
});
