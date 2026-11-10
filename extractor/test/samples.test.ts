// Runs against the public DUTCH sample reports when they have been downloaded
// (npm run samples). They are not committed: the lab owns the copyright.
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { extractDutch } from "../src/index.ts";

const dir = new URL("../../samples/", import.meta.url);
const sample = (name: string) => new URL(`${name}/report.pdf`, dir);

const CASES = [
  {
    name: "male-2019",
    sex: "male",
    age: 50,
    markers: 47,
    spot: { testosterone: 46.8, "5a_dht": 7.9, "24hr_free_cortisol": 171.3, androsterone: 1461 },
  },
  {
    name: "female-a",
    sex: "female",
    age: 26,
    markers: 47,
    spot: { testosterone: 1.5, "5a_dht": 7.2, "24hr_free_cortisol": 52, total_estrogen: 67.59 },
  },
  {
    name: "female-2018",
    sex: "female",
    age: 37,
    markers: 46,
    spot: { testosterone: 1.5, "5_hydroxyindoleacetate_5hiaa": 16, etiocholanolone: 680 },
  },
] as const;

describe("public sample reports", () => {
  for (const c of CASES) {
    it.skipIf(!existsSync(sample(c.name)))(`${c.name}: all rows read, checks pass`, async () => {
      const r = await extractDutch(new Uint8Array(await readFile(sample(c.name))));
      expect(r.patient).toEqual({ sex: c.sex, ageYears: c.age });
      expect(r.markers).toHaveLength(c.markers);
      for (const [id, value] of Object.entries(c.spot)) {
        expect(r.markers.find((m) => m.id === id)?.value, id).toBe(value);
      }
      expect(r.checks.filter((k) => k.outcome === "fail")).toEqual([]);
      expect(r.warnings).toEqual([]);
      expect(JSON.stringify(r)).not.toMatch(/Sample Report|Accession|A Street|\d{4}-\d{2}-\d{2}/);
    });
  }

  it.skipIf(!existsSync(sample("female-2025")))(
    "female-2025 (v1.6.0 layout): reports that it cannot read the tables",
    async () => {
      const r = await extractDutch(new Uint8Array(await readFile(sample("female-2025"))));
      expect(r.source.reportVersion).toBe("1.6.0");
      expect(r.markers).toEqual([]);
      expect(r.warnings.join(" ")).toMatch(/No result tables/);
    },
  );
});
