import type { DutchRecord } from "./types.ts";

/** What makes two records "the same report": every marker value, plus sex and age. */
function identity(record: DutchRecord): string {
  const pairs = record.markers.map((m) => `${m.id}:${m.value}`).sort();
  return [`sex:${record.patient.sex}`, `age:${record.patient.ageYears}`, ...pairs].join("|");
}

/** 64-bit FNV-1a. Synchronous and dependency-free so it runs the same in Node and the browser. */
function fnv1a64(text: string): string {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(text)) {
    hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, "0");
}

/**
 * A short hash of a record's sorted `id:value` pairs plus sex and age.
 * Two submissions of the same report get the same fingerprint, whatever
 * order the rows were read in. It is not derived from any identifying field.
 */
export function fingerprint(record: DutchRecord): string {
  return fnv1a64(identity(record));
}

/**
 * Find reports that appear more than once in a batch. Returns one group per
 * repeated report, each listing the positions (indexes into `records`) of its
 * copies in ascending order. Records with no markers are never grouped:
 * unreadable reports all look alike, which says nothing about the patients.
 */
export function findDuplicates(records: DutchRecord[]): number[][] {
  const byFingerprint = new Map<string, number[]>();
  records.forEach((record, i) => {
    if (record.markers.length === 0) return;
    const key = fingerprint(record);
    const group = byFingerprint.get(key);
    if (group) group.push(i);
    else byFingerprint.set(key, [i]);
  });
  return [...byFingerprint.values()].filter((group) => group.length > 1);
}
