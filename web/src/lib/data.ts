import { promises as fs } from "node:fs";
import path from "node:path";
import { z } from "zod";
import {
  movieSchema,
  oraclePredictionsSchema,
  modelMetaSchema,
  type Movie,
  type OraclePredictions,
  type ModelMeta,
} from "@/lib/types";

const DATA_DIR = path.join(process.cwd(), "data");

async function readJson<T>(
  file: string,
  schema: z.ZodType<T>,
): Promise<T | null> {
  let raw: string;
  try {
    raw = await fs.readFile(path.join(DATA_DIR, file), "utf-8");
  } catch {
    return null; // Snapshot not exported yet — pages render the empty state.
  }
  // Schema drift must fail loud, not render wrong figures.
  return schema.parse(JSON.parse(raw));
}

export async function loadMovies(): Promise<Movie[] | null> {
  "use cache";
  return readJson("movies.json", z.array(movieSchema));
}

export async function loadOraclePredictions(): Promise<OraclePredictions | null> {
  "use cache";
  return readJson("predictions.json", oraclePredictionsSchema);
}

export async function loadModelMeta(): Promise<ModelMeta | null> {
  "use cache";
  return readJson("model_meta.json", modelMetaSchema);
}

export async function loadMovie(tmdbId: number): Promise<Movie | null> {
  const movies = await loadMovies();
  return movies?.find((m) => m.tmdb_id === tmdbId) ?? null;
}
