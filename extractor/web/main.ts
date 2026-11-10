// Browser entry point: reads the chosen PDFs on this device and prints their
// anonymised records. The review interface will be built on top of this.
import { GlobalWorkerOptions } from "pdfjs-dist/legacy/build/pdf.mjs";
import PdfWorker from "pdfjs-dist/legacy/build/pdf.worker.mjs?worker";
import { extractDutch, type DutchRecord } from "../src/index.ts";

// The pdf.js worker is bundled with the page (never a CDN) and started now, so
// that opening a report later needs no request at all.
GlobalWorkerOptions.workerPort = new PdfWorker();

const input = document.querySelector<HTMLInputElement>("#files")!;
const out = document.querySelector<HTMLPreElement>("#out")!;

input.addEventListener("change", async () => {
  out.dataset.state = "working";
  out.textContent = "";
  try {
    const records: DutchRecord[] = [];
    for (const file of input.files ?? []) {
      records.push(await extractDutch(new Uint8Array(await file.arrayBuffer())));
    }
    out.textContent = JSON.stringify(records, null, 2);
    out.dataset.state = "done";
  } catch (err) {
    out.textContent = `Could not read the report: ${err instanceof Error ? err.message : String(err)}`;
    out.dataset.state = "error";
  }
});
