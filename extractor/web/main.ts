// Browser entry point: reads the dropped PDFs on this device and shows, per
// report, what was read so a human can review it before downloading.
import { GlobalWorkerOptions } from "pdfjs-dist/legacy/build/pdf.mjs";
import PdfWorker from "pdfjs-dist/legacy/build/pdf.worker.mjs?worker";
import {
  batchToCsv,
  batchToJson,
  extractDutch,
  summariseBatch,
  type Batch,
  type BatchReport,
  type DutchRecord,
} from "../src/index.ts";

// The pdf.js worker is bundled with the page (never a CDN) and started now, so
// that opening a report later needs no request at all.
GlobalWorkerOptions.workerPort = new PdfWorker();

// The service worker (built by web/vite.config.ts) keeps a copy of the page so
// it loads with no connection. Registered unconditionally rather than behind
// import.meta.env.PROD, which is false when Vite builds under NODE_ENV=test;
// the dev server has no sw.js, so there the registration simply fails.
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js").catch(() => {
    // offline use is a convenience; the page works online without it
  });
}

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;
const input = $<HTMLInputElement>("#files");
const drop = $<HTMLLabelElement>("#drop");
const section = $<HTMLElement>("#batch");
const status = $<HTMLParagraphElement>("#status");
const duplicatesBox = $<HTMLDivElement>("#duplicates");
const reportsBox = $<HTMLDivElement>("#reports");
const csvButton = $<HTMLButtonElement>("#download-csv");
const jsonButton = $<HTMLButtonElement>("#download-json");

/** Results so far, in the order the files were added. File names stay on screen only. */
let results: (DutchRecord | Error)[] = [];
let fileNames: string[] = [];
let batch: Batch = summariseBatch([]);

/** Build an element. Text always goes in as text, never as HTML: it comes from the user's files. */
function h(tag: string, attrs: Record<string, string> = {}, ...children: (Node | string | null)[]): HTMLElement {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of children) if (c !== null) el.append(c);
  return el;
}

const show = (v: string | number | null | undefined) => (v === null || v === undefined ? "not found" : String(v));

async function addFiles(files: File[]) {
  const pdfs = files.filter((f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
  if (pdfs.length === 0) return;
  setBusy(true, `Reading ${pdfs.length} report${pdfs.length === 1 ? "" : "s"}…`);
  for (const file of pdfs) {
    try {
      results.push(await extractDutch(new Uint8Array(await file.arrayBuffer())));
    } catch (err) {
      results.push(err instanceof Error ? err : new Error(String(err)));
    }
    fileNames.push(file.name);
  }
  batch = summariseBatch(results);
  render();
  setBusy(false);
}

function setBusy(busy: boolean, message = "") {
  section.hidden = false;
  input.disabled = busy;
  csvButton.disabled = jsonButton.disabled = busy || !batch.reports.some((r) => r.record);
  if (busy) {
    status.dataset.state = "working";
    status.textContent = message;
  }
}

function render() {
  const count = (state: BatchReport["state"]) => batch.reports.filter((r) => r.state === state).length;
  const problems = count("unreadable") + count("failed");
  status.dataset.state = "done";
  status.textContent =
    `${batch.reports.length} report${batch.reports.length === 1 ? "" : "s"}: ${count("read")} read` +
    (problems ? `, ${problems} could not be read` : "");

  duplicatesBox.hidden = batch.duplicates.length === 0;
  duplicatesBox.replaceChildren(
    h("p", {}, h("strong", {}, "Possible duplicates. "), "These reports have exactly the same values, sex and age. If they are the same report added twice, both copies are in the downloads: delete one before analysis, or one patient will be counted twice."),
    h("ul", {}, ...batch.duplicates.map((group) => h("li", {}, group.join(", ")))),
  );

  reportsBox.replaceChildren(...batch.reports.map((r, i) => renderReport(r, fileNames[i])));
}

function renderReport(r: BatchReport, fileName: string): HTMLElement {
  const card = h("article", { class: "report", "data-state": r.state, "data-id": r.id });
  card.append(h("h2", {}, `Report ${r.id}`), h("p", { class: "file" }, `File: ${fileName} (the file name is not included in downloads)`));

  if (r.state === "failed") {
    card.append(
      h("div", { class: "banner bad" }, h("p", {}, h("strong", {}, "This file could not be opened. "), "It may not be a PDF, or it may be damaged or password-protected."), h("p", {}, `Details: ${r.error}`)),
    );
    return card;
  }

  const rec = r.record!;
  if (r.state === "unreadable") {
    card.append(
      h(
        "div",
        { class: "banner bad" },
        h("p", {}, h("strong", {}, "No results could be read from this report. "), "It may be a scan or photo, or use a layout this version does not support. Nothing from it will be in the CSV."),
      ),
    );
  }

  const facts: [string, string][] = [
    ["Sex", show(rec.patient.sex)],
    ["Age", show(rec.patient.ageYears)],
    ["Collected", show(rec.collectionMonth)],
    ["Report issued", show(rec.source.reportMonth)],
    ["Layout version", rec.source.reportVersion ?? "classic"],
    ["Pages", show(rec.source.pages)],
    ["Values read", String(rec.markers.length)],
  ];
  card.append(h("dl", { class: "facts" }, ...facts.map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, v)))));

  const warnings = r.state === "unreadable" ? rec.warnings.filter((w) => !w.startsWith("No result tables")) : rec.warnings;
  if (warnings.length) {
    card.append(
      h("div", { class: "banner warn warnings" }, h("p", {}, h("strong", {}, `Check before trusting this report (${warnings.length})`)), h("ul", {}, ...warnings.map((w) => h("li", {}, w)))),
    );
  }
  if (r.state === "unreadable") return card;

  if (rec.checks.length) {
    const passed = rec.checks.filter((c) => c.outcome === "pass").length;
    card.append(
      h("h3", {}, `Consistency checks: ${passed} of ${rec.checks.length} passed`),
      h("ul", { class: "checks" }, ...rec.checks.map((c) => h("li", { class: c.outcome, title: c.detail }, `${c.name}: ${c.detail}`))),
    );
  }

  const rows: HTMLElement[] = [];
  let category: string | null | undefined;
  for (const m of rec.markers) {
    if (m.category !== category) {
      category = m.category;
      rows.push(h("tr", { class: "category" }, h("td", { colspan: "5" }, category ?? "Uncategorised")));
    }
    const range = m.refLow === null && m.refHigh === null ? "" : `${show(m.refLow)} – ${show(m.refHigh)}`;
    rows.push(
      h(
        "tr",
        {},
        h("td", {}, m.name),
        h("td", { class: "num" }, `${m.comparator ?? ""}${m.value}`),
        h("td", {}, m.unit),
        h("td", { class: "num" }, range),
        h("td", {}, m.status ?? ""),
      ),
    );
  }
  card.append(
    h("h3", {}, "Values"),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { class: "markers" },
        h("thead", {}, h("tr", {}, ...["Test", "Result", "Units", "Normal range", "Lab verdict"].map((t) => h("th", { scope: "col" }, t)))),
        h("tbody", {}, ...rows),
      ),
    ),
  );
  return card;
}

function download(name: string, type: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: name }) as HTMLAnchorElement;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

input.addEventListener("change", async () => {
  await addFiles([...(input.files ?? [])]);
  input.value = "";
});
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  void addFiles([...(e.dataTransfer?.files ?? [])]);
});
// a file dropped beside the drop zone must not open in the tab and lose the batch
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

csvButton.addEventListener("click", () => download("dutch-records.csv", "text/csv", batchToCsv(batch)));
jsonButton.addEventListener("click", () => download("dutch-records.json", "application/json", batchToJson(batch)));
$<HTMLButtonElement>("#clear").addEventListener("click", () => {
  results = [];
  fileNames = [];
  batch = summariseBatch([]);
  reportsBox.replaceChildren();
  duplicatesBox.hidden = true;
  section.hidden = true;
});
