import type { Check, Marker } from "./types.ts";

/**
 * Internal-consistency tests. A DUTCH report prints totals that are sums of
 * other printed values, and a verdict ("Above range") next to every number.
 * A misread or hand-edited value breaks one of these.
 */

interface SumRule {
  name: string;
  total: RegExp;
  parts: RegExp;
  /** Exact number of parts expected, when it is fixed. */
  count?: number;
  minParts: number;
}

const SUM_RULES: SumRule[] = [
  {
    name: "24hr free cortisol = sum of the daily cortisol samples",
    total: /^24hr_free_cortisol$/,
    parts: /^cortisol_[a-e]_/,
    minParts: 4,
  },
  {
    name: "24hr free cortisone = sum of the daily cortisone samples",
    total: /^24hr_free_cortisone$/,
    parts: /^cortisone_[a-e]_/,
    minParts: 4,
  },
  {
    name: "Metabolized cortisol = a-THF + b-THF + b-THE",
    total: /^metabolized_cortisol/,
    parts: /^[ab]_tetrahydrocortiso(l|ne)_/,
    count: 3,
    minParts: 3,
  },
  {
    name: "Total estrogen = sum of the ten estrogen metabolites",
    total: /^total_estrogen$/,
    parts: /^(estrone_e1|estradiol_e2|estriol_e3|2_oh_e1|4_oh_e1|16_oh_e1|2_methoxy_e1|2_oh_e2|4_oh_e2|2_methoxy_e2)$/,
    count: 10,
    minParts: 10,
  },
];

function sumChecks(markers: Marker[]): Check[] {
  return SUM_RULES.map((rule) => {
    const total = markers.find((m) => rule.total.test(m.id));
    const parts = markers.filter((m) => rule.parts.test(m.id));
    if (!total || parts.length < rule.minParts || (rule.count && parts.length !== rule.count)) {
      return {
        name: rule.name,
        outcome: "skipped" as const,
        detail: !total ? "total not on this report" : `found ${parts.length} component values`,
      };
    }
    const sum = parts.reduce((s, m) => s + m.value, 0);
    // each printed value is rounded, so allow a little slack per component
    const tolerance = Math.max(0.06 * parts.length, Math.abs(total.value) * 0.005);
    const ok = Math.abs(sum - total.value) <= tolerance;
    return {
      name: rule.name,
      outcome: ok ? ("pass" as const) : ("fail" as const),
      detail: `components add to ${+sum.toFixed(2)}, report prints ${total.value}`,
    };
  });
}

function statusCheck(markers: Marker[]): Check {
  const name = "Each value agrees with the lab's own verdict next to it";
  const bad: string[] = [];
  let tested = 0;
  for (const m of markers) {
    if (m.status === null || m.refLow === null || m.refHigh === null || m.comparator) continue;
    tested++;
    const s = m.status.toLowerCase();
    // the lab compares unrounded values, so a printed value may equal the limit
    const ok = s.startsWith("above")
      ? m.value >= m.refHigh
      : s.startsWith("below")
        ? m.value <= m.refLow
        : m.value >= m.refLow && m.value <= m.refHigh;
    if (!ok) bad.push(`${m.name} ${m.value} is marked "${m.status}" against ${m.refLow}-${m.refHigh}`);
  }
  if (tested === 0) return { name, outcome: "skipped", detail: "no rows with a verdict and range" };
  return bad.length
    ? { name, outcome: "fail", detail: bad.join("; ") }
    : { name, outcome: "pass", detail: `${tested} rows agree` };
}

export function runChecks(markers: Marker[]): Check[] {
  if (markers.length === 0) return [];
  return [...sumChecks(markers), statusCheck(markers)];
}
