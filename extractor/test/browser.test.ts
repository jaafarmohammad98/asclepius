// Builds the static page, opens it in the locally installed Google Chrome and
// uses it the way a person would. Skipped when the public samples have not
// been downloaded (npm run samples) or Chrome is absent.
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser, type BrowserContext, type Page } from "playwright-core";
import { build, preview, type PreviewServer } from "vite";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { batchToCsv, extractDutch, summariseBatch } from "../src/index.ts";

const config = fileURLToPath(new URL("../web/vite.config.ts", import.meta.url));
const sample = (name: string) => fileURLToPath(new URL(`../../samples/${name}/report.pdf`, import.meta.url));
const male = sample("male-2019");
const female = sample("female-a");
const layout2025 = sample("female-2025");
const haveSamples = [male, female, layout2025].every((f) => existsSync(f));
const notAPdf = { name: "notes.pdf", mimeType: "application/pdf", buffer: Buffer.from("not really a pdf") };

let server: PreviewServer | undefined;
let browser: Browser | undefined;
let outDir: string | undefined;
let url = "";

beforeAll(async () => {
  if (!haveSamples) return;
  try {
    browser = await chromium.launch({ channel: "chrome" });
  } catch {
    return; // Chrome is not installed; the tests skip themselves
  }
  // built into a temporary folder so the test never replaces extractor/dist
  outDir = await mkdtemp(join(tmpdir(), "extractor-web-"));
  const shared = { configFile: config, logLevel: "silent", build: { outDir, emptyOutDir: true } } as const;
  await build(shared);
  server = await preview({ ...shared, preview: { port: 0 } });
  url = server.resolvedUrls!.local[0];
}, 60_000);

afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (outDir) await rm(outDir, { recursive: true, force: true });
});

/** Open the page in a fresh profile, recording every network request made after it has loaded. */
async function openPage(): Promise<{ context: BrowserContext; page: Page; requests: string[] }> {
  const context = await browser!.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  await page.goto(url, { waitUntil: "networkidle" });
  const requests: string[] = [];
  // blob: links are the downloads being saved, not network traffic
  page.on("request", (r) => (/^https?:/.test(r.url()) ? requests.push(r.url()) : undefined));
  return { context, page, requests };
}

async function addFiles(page: Page, files: Parameters<Page["setInputFiles"]>[1]) {
  await page.setInputFiles("#files", files);
  await page.waitForSelector("#status[data-state=done]");
}

async function downloadText(page: Page, button: string): Promise<string> {
  const [download] = await Promise.all([page.waitForEvent("download"), page.click(button)]);
  return readFile((await download.path())!, "utf8");
}

describe.skipIf(!haveSamples)("built static page", () => {
  it("shows two dropped reports with their values and checks, and downloads them like the command line", async (ctx) => {
    if (!browser) return ctx.skip("Google Chrome is not installed");
    const { context, page, requests } = await openPage();

    await expect(page.textContent(".privacy")).resolves.toContain("never leave this device");
    await addFiles(page, [male, female]);

    const cards = page.locator("article.report");
    await expect(cards.count()).resolves.toBe(2);
    for (const i of [0, 1]) {
      await expect(cards.nth(i).getAttribute("data-state")).resolves.toBe("read");
      await expect(cards.nth(i).locator("table.markers td.num").count()).resolves.toBeGreaterThan(40);
      await expect(cards.nth(i).locator("ul.checks li.pass").count()).resolves.toBeGreaterThan(0);
    }
    await expect(cards.nth(0).textContent()).resolves.toContain("46.8"); // male testosterone

    const records = await Promise.all([male, female].map(async (f) => extractDutch(new Uint8Array(await readFile(f)))));
    const expected = summariseBatch(records);
    const json = JSON.parse(await downloadText(page, "#download-json"));
    expect(json).toEqual(JSON.parse(JSON.stringify(expected.reports.map((r) => ({ id: r.id, record: r.record })))));
    const csv = await downloadText(page, "#download-csv");
    expect(csv).toBe(batchToCsv(expected));
    expect(csv).not.toMatch(/report\.pdf/);

    await expect(page.isHidden("#duplicates")).resolves.toBe(true);
    expect(requests).toEqual([]);
    await context.close();
  }, 120_000);

  it("shows an error state, not an empty table, for files it cannot read", async (ctx) => {
    if (!browser) return ctx.skip("Google Chrome is not installed");
    const { context, page } = await openPage();
    // Playwright cannot mix paths and buffers in one call, so the sample goes in as a buffer too
    const report2025 = { name: "report.pdf", mimeType: "application/pdf", buffer: await readFile(layout2025) };
    await addFiles(page, [report2025, notAPdf]);

    const [unreadable, failed] = [page.locator("article.report").nth(0), page.locator("article.report").nth(1)];
    await expect(unreadable.getAttribute("data-state")).resolves.toBe("unreadable");
    await expect(unreadable.locator(".banner.bad").textContent()).resolves.toContain("No results could be read");
    await expect(unreadable.locator("table").count()).resolves.toBe(0);
    await expect(failed.getAttribute("data-state")).resolves.toBe("failed");
    await expect(failed.locator(".banner.bad").textContent()).resolves.toContain("could not be opened");
    await expect(page.textContent("#status")).resolves.toContain("2 could not be read");
    // nothing was read, so a CSV would be only a header; the JSON still carries the warnings
    await expect(page.isDisabled("#download-csv")).resolves.toBe(true);
    await expect(page.isDisabled("#download-json")).resolves.toBe(false);
    await context.close();
  }, 120_000);

  it("warns when the same report is added twice", async (ctx) => {
    if (!browser) return ctx.skip("Google Chrome is not installed");
    const { context, page } = await openPage();
    await addFiles(page, [male, female]);
    await addFiles(page, [male]); // added again later, in a second drop
    await expect(page.isVisible("#duplicates")).resolves.toBe(true);
    await expect(page.textContent("#duplicates li")).resolves.toBe("r001, r003");

    await page.getByRole("button", { name: "Remove report r003" }).click();
    await expect(page.locator("article.report").count()).resolves.toBe(2);
    await expect(page.isHidden("#duplicates")).resolves.toBe(true);
    const csv = await downloadText(page, "#download-csv");
    expect(new Set(csv.trim().split("\n").slice(1).map((l) => l.split(",")[0]))).toEqual(new Set(["r001", "r002"]));
    await context.close();
  }, 120_000);

  it("opens and reads a report with the network disconnected, once it has been loaded", async (ctx) => {
    if (!browser) return ctx.skip("Google Chrome is not installed");
    const { context, page } = await openPage();
    await page.waitForSelector("#offline[data-state=ready]");
    await expect(page.textContent("#offline")).resolves.toContain("Ready to use offline");
    await context.setOffline(true);
    await page.reload({ waitUntil: "load" });
    await addFiles(page, [male]);
    await expect(page.locator("article.report").getAttribute("data-state")).resolves.toBe("read");
    await context.close();
  }, 120_000);
});
