import { defineConfig } from "vite";

// Builds the browser page into extractor/dist as a static folder. Relative
// asset paths (base "./") let the folder be hosted under any path.
export default defineConfig({
  root: import.meta.dirname,
  base: "./",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    target: "es2022",
  },
  worker: { format: "es" },
});
