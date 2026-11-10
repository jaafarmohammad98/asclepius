import { describe, expect, it } from "vitest";
import { compareMarkers, parseExpected, percent, totalAccuracy } from "../src/accuracy.ts";
import type { Marker } from "../src/types.ts";

function marker(id: string, value: number, extra: Partial<Marker> = {}): Marker {
  return {
    id,
    name: id,
    category: null,
    value,
    unit: "ng/mg",
    refLow: 0,
    refHigh: 10,
    status: null,
    page: 2,
    ...extra,
  };
}

const row = (id: string, value: number, extra: object = {}) => ({
  id,
  value,
  unit: "ng/mg",
  range: "0 - 10",
  ...extra,
});

describe("parseExpected", () => {
  it("reads id, value, unit and the printed range", () => {
    expect(parseExpected({ markers: [row("testosterone", 46.8, { range: "25 - 115" })] })).toEqual([
      { id: "testosterone", value: 46.8, unit: "ng/mg", refLow: 25, refHigh: 115 },
    ]);
  });

  it("accepts the printed name instead of an id, a null range and a comparator", () => {
    expect(
      parseExpected({
        markers: [{ name: "5a-DHT", value: 0.2, comparator: "<", unit: "ng/mg", range: null }],
      }),
    ).toEqual([
      { id: "5a_dht", value: 0.2, comparator: "<", unit: "ng/mg", refLow: null, refHigh: null },
    ]);
  });

  it.each([
    ["no markers list", {}, /markers/],
    ["value typed as text", { markers: [row("a", "1.5" as unknown as number)] }, /a.*"value"/],
    ["missing unit", { markers: [{ id: "a", value: 1, range: null }] }, /"unit"/],
    ["unreadable range", { markers: [row("a", 1, { range: "under 5" })] }, /"range"/],
    ["range left out", { markers: [{ id: "a", value: 1, unit: "ng/mg" }] }, /"range"/],
    ["same marker twice", { markers: [row("a", 1), row("a", 2)] }, /twice/],
    ["no id or name", { markers: [{ value: 1, unit: "ng/mg", range: null }] }, /"id" or a "name"/],
  ])("rejects %s", (_, json, message) => {
    expect(() => parseExpected(json)).toThrow(message);
  });
});

describe("compareMarkers", () => {
  const expected = parseExpected({ markers: [row("a", 1), row("b", 2), row("c", 3)] });

  it("counts every value as matched when the extractor agrees with the page", () => {
    const r = compareMarkers(expected, [marker("a", 1), marker("b", 2), marker("c", 3)]);
    expect(r).toEqual({ expected: 3, matched: 3, missed: [], wrong: [], extra: [] });
  });

  it("reports a differing value as wrong, with both numbers", () => {
    const r = compareMarkers(expected, [marker("a", 1), marker("b", 2.1), marker("c", 3)]);
    expect(r.matched).toBe(2);
    expect(r.wrong).toEqual([{ id: "b", differences: ["value: expected 2, got 2.1"] }]);
  });

  it("reports a differing unit, range or comparator as wrong", () => {
    const r = compareMarkers(expected, [
      marker("a", 1, { unit: "ug/mg" }),
      marker("b", 2, { refHigh: 100 }),
      marker("c", 3, { comparator: "<" }),
    ]);
    expect(r.matched).toBe(0);
    expect(r.wrong).toEqual([
      { id: "a", differences: ["unit: expected ng/mg, got ug/mg"] },
      { id: "b", differences: ["range: expected 0 - 10, got 0 - 100"] },
      { id: "c", differences: ["value: expected 3, got <3"] },
    ]);
  });

  it("reports a range the extractor could not read", () => {
    const r = compareMarkers(expected.slice(0, 1), [marker("a", 1, { refLow: null, refHigh: null })]);
    expect(r.wrong).toEqual([{ id: "a", differences: ["range: expected 0 - 10, got none"] }]);
  });

  it("separates values the extractor missed from values it added", () => {
    const r = compareMarkers(expected, [marker("a", 1), marker("c", 3), marker("z", 9)]);
    expect(r).toMatchObject({ matched: 2, missed: ["b"], wrong: [], extra: ["z"] });
  });

  it("counts everything as missed when no table was read", () => {
    const r = compareMarkers(expected, []);
    expect(r).toMatchObject({ matched: 0, missed: ["a", "b", "c"] });
  });
});

describe("totals", () => {
  it("adds up reports and gives the share of human-read values that matched", () => {
    const expected = parseExpected({ markers: [row("a", 1), row("b", 2)] });
    const t = totalAccuracy([
      compareMarkers(expected, [marker("a", 1), marker("b", 2)]),
      compareMarkers(expected, [marker("a", 5), marker("x", 2)]),
    ]);
    expect(t).toEqual({ expected: 4, matched: 2, missed: 1, wrong: 1, extra: 1 });
    expect(percent(t.matched, t.expected)).toBe("50.0%");
    expect(percent(0, 0)).toBe("n/a");
  });
});
