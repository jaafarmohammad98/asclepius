import { describe, expect, it } from "vitest";
import { cleanName, parseDutch, slug } from "../src/parse.ts";
import { toCsv } from "../src/csv.ts";
import type { TextLine } from "../src/pdfText.ts";

/** Build parser input from plain text; "|" marks a column gap. Invented data. */
function lines(page: number, text: string): TextLine[] {
  return text
    .trim()
    .split("\n")
    .map((t, i) => ({ page, y: 800 - i * 10, items: [], text: t.trim().replace(/\s*\|\s*/g, "\t") }));
}

const HEADER = `
Accession # 99999999
Jane Fictional
1 Nowhere Road
Ordering Physician: | DOB: 1980-02-03 | Collection Times:
2024-03-09 06:01AM
Precision Analytical | Age: 44
Gender: Female | 2024-03-09 10:01PM
`;

const TABLE = `
Category | T est | Result | Units | Normal Range
Androgens and Metabolites (Urine)
T estosterone | Below range | 1.5 | ng/mg | 2.3 - 14
5a-DHT | Above range | 7.2 | ng/mg | 0 - 6.6
Daily Free Cortisol and Cortisone (Urine)
Cortisol A (W aking) | Within range | 20.0 | ng/mg | 10 - 50
Cortisol B (Morning) | Within range | 40.0 | ng/mg | 30 - 130
Cortisol C (Afternoon) | Within range | 10.0 | ng/mg | 7 - 30
Cortisol D (Night) | Within range | 5.0 | ng/mg | 0 - 14
24hr Free Cortisol | Within range | 75.0 | ng/mg | 65 - 200
McMinnville, OR 97128 | 03/20/2024 | DutchTest.com
`;

describe("parseDutch", () => {
  const record = parseDutch([...lines(1, HEADER), ...lines(2, TABLE)]);

  it("reads sex, age and months", () => {
    expect(record.patient).toEqual({ sex: "female", ageYears: 44 });
    expect(record.collectionMonth).toBe("2024-03");
    expect(record.source.reportMonth).toBe("2024-03");
  });

  it("reads every table row with value, unit, range and category", () => {
    expect(record.markers).toHaveLength(7);
    expect(record.markers[0]).toMatchObject({
      id: "testosterone",
      name: "Testosterone",
      category: "Androgens and Metabolites",
      value: 1.5,
      unit: "ng/mg",
      refLow: 2.3,
      refHigh: 14,
      status: "Below range",
    });
    expect(record.markers[2].id).toBe("cortisol_a_waking");
  });

  it("passes its consistency checks and raises no warnings", () => {
    expect(record.checks.filter((c) => c.outcome === "fail")).toEqual([]);
    expect(record.checks.find((c) => c.name.startsWith("24hr free cortisol"))?.outcome).toBe("pass");
    expect(record.warnings).toEqual([]);
  });

  it("never carries identifying details into the record or the CSV", () => {
    const out = JSON.stringify(record) + toCsv([{ id: "r001", record }]);
    for (const secret of ["Jane", "Fictional", "1980", "99999999", "Nowhere", "2024-03-09"]) {
      expect(out).not.toContain(secret);
    }
  });

  it("flags a total that does not match its components", () => {
    const edited = TABLE.replace("| 20.0 |", "| 35.0 |");
    const r = parseDutch([...lines(1, HEADER), ...lines(2, edited)]);
    expect(r.checks.find((c) => c.name.startsWith("24hr free cortisol"))?.outcome).toBe("fail");
    expect(r.warnings.some((w) => w.includes("Consistency check failed"))).toBe(true);
  });

  it("flags a value that contradicts the lab's verdict", () => {
    const edited = TABLE.replace("Below range | 1.5", "Below range | 9.5");
    const r = parseDutch([...lines(1, HEADER), ...lines(2, edited)]);
    expect(r.checks.at(-1)?.outcome).toBe("fail");
  });

  it("flags the same marker printed with two different values", () => {
    const extra = TABLE + "\n5a-DHT | Above range | 9.9 | ng/mg | 0 - 6.6";
    const r = parseDutch([...lines(1, HEADER), ...lines(2, extra)]);
    expect(r.warnings.some((w) => w.includes("appears twice"))).toBe(true);
    expect(r.markers.filter((m) => m.id === "5a_dht")).toHaveLength(1);
  });

  it("warns instead of returning an empty success when nothing is readable", () => {
    const r = parseDutch(lines(1, "Some unrelated document\nwith no tables"));
    expect(r.markers).toEqual([]);
    expect(r.warnings.join(" ")).toMatch(/No result tables/);
    expect(r.warnings.join(" ")).toMatch(/does not look like a DUTCH report/);
  });
});

describe("names", () => {
  it("repairs kerning gaps without touching real words", () => {
    expect(cleanName("Epi-T estosterone")).toBe("Epi-Testosterone");
    expect(cleanName("Creatinine A (W aking)")).toBe("Creatinine A (Waking)");
    expect(cleanName("Cortisol B (Morning)")).toBe("Cortisol B (Morning)");
  });
  it("makes stable ids", () => {
    expect(slug("Estrone(E1)")).toBe("estrone_e1");
    expect(slug("b-Tetrahydrocortisol (b-THF)")).toBe("b_tetrahydrocortisol_b_thf");
  });
});
