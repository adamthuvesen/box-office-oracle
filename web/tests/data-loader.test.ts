import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("data loaders without a web/data snapshot", () => {
  beforeEach(() => {
    vi.resetModules();
    // The loaders resolve data/ from the working directory at import time.
    vi.spyOn(process, "cwd").mockReturnValue(mkdtempSync(path.join(tmpdir(), "web-data-missing-")));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns null so the pages render the empty state", async () => {
    const { loadModelMeta, loadMovie, loadMovies, loadOraclePredictions } =
      await import("@/lib/data");

    expect(await loadMovies()).toBeNull();
    expect(await loadOraclePredictions()).toBeNull();
    expect(await loadModelMeta()).toBeNull();
    expect(await loadMovie(1)).toBeNull();
  });
});
