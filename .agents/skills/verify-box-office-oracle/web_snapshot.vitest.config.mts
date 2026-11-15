import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Same alias as web/vitest.config.mts, with no cache written through the node_modules symlink.
export default defineConfig({
  cacheDir: process.env.VERIFY_VITE_CACHE,
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    include: ["tests/verify-snapshot.test.ts"],
    fsModuleCache: false,
  },
});
