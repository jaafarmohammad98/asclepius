// Builds the static page, opens it in the locally installed Google Chrome and
// checks it against the command-line extractor. Skipped when the
// public sample has not been downloaded (npm run samples) or Chrome is absent.
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright-core";
import { build, preview, type PreviewServer } from "vite";
import { afterAll, describe, expect, it } from "vitest";
import { extractDutch } from "../src/index.ts";

const config = fileURLToPath(new URL("../web/vite.config.ts", import.meta.url));
const report = fileURLToPath(new URL("../../samples/male-2019/report.pdf", import.meta.url));

let server: PreviewServer | undefined;
let browser: Browser | undefined;
let outDir: string | undefined;

afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (outDir) await rm(outDir, { recursive: true, force: true });
});

describe("built static page", () => {
  it.skipIf(!existsSync(report))(
    "extracts the male sample like the command line, with no network requests after load",
    async (ctx) => {
      try {
        browser = await chromium.launch({ channel: "chrome" });
      } catch {
        return ctx.skip("Google Chrome is not installed");
      }

      // built into a temporary folder so the test never replaces extractor/dist
      outDir = await mkdtemp(join(tmpdir(), "extractor-web-"));
      const shared = { configFile: config, logLevel: "silent", build: { outDir, emptyOutDir: true } } as const;
      await build(shared);
      server = await preview({ ...shared, preview: { port: 0 } });
      const url = server.resolvedUrls!.local[0];

      const page = await browser.newPage();
      await page.goto(url, { waitUntil: "networkidle" });

      const requests: string[] = [];
      page.on("request", (r) => requests.push(r.url()));

      // the same report twice: a second report must work after the first is closed
      await page.setInputFiles("#files", [report, report]);
      await page.waitForSelector("#out[data-state=done]");
      const inBrowser = JSON.parse((await page.textContent("#out"))!);

      const onCommandLine = await extractDutch(new Uint8Array(await readFile(report)));
      expect(onCommandLine.markers.length).toBeGreaterThan(0);
      expect(inBrowser).toEqual([onCommandLine, onCommandLine].map((r) => JSON.parse(JSON.stringify(r))));
      expect(requests).toEqual([]);
    },
    120_000,
  );
});
