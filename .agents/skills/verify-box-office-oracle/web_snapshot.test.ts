import path from "node:path";
import { describe, expect, it, vi } from "vitest";

// Runs inside a checkout copy. The web data loaders parse each exported file with its Zod schema,
// so a schema mismatch throws here instead of rendering wrong figures.
describe("the exported web/data snapshot", () => {
  it("validates against the schemas the web app reads", async () => {
    const webRoot = process.env.VERIFY_WEB_ROOT;
    const expectedRows = Number(process.env.VERIFY_EXPECT_ROWS);
    expect(webRoot).toBeTruthy();
    expect(Number.isInteger(expectedRows)).toBe(true);
    vi.spyOn(process, "cwd").mockReturnValue(path.resolve(webRoot ?? ""));
    const { loadModelMeta, loadMovies, loadOraclePredictions } = await import("@/lib/data");

    const movies = await loadMovies();
    const predictions = await loadOraclePredictions();
    const meta = await loadModelMeta();

    expect(movies).not.toBeNull();
    expect(predictions).not.toBeNull();
    expect(meta).not.toBeNull();
    expect(movies).toHaveLength(expectedRows);
    expect(Object.keys(predictions ?? {})).toHaveLength(expectedRows);
    for (const movie of movies ?? []) {
      expect(predictions?.[String(movie.tmdb_id)]).toBeDefined();
    }
  });
});
