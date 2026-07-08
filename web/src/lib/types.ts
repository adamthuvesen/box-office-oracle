import { z } from "zod";

export const movieSchema = z.object({
  tmdb_id: z.number(),
  imdb_id: z.string().nullable(),
  title: z.string(),
  release_date: z.string().nullable(),
  release_year: z.number(),
  genres: z.array(z.string()),
  director: z.string().nullable(),
  actors: z.array(z.string()),
  mpaa: z.string().nullable(),
  runtime: z.number().nullable(),
  production_budget: z.number().nullable(),
  production_budget_source: z.string().nullable(),
  production_company: z.string().nullable(),
  overview: z.string().nullable(),
  tagline: z.string().nullable(),
  keywords: z.array(z.string()),
  /** Null when no reliable actual exists yet (future releases, gross artifacts). */
  worldwide_gross: z.number().nullable(),
  poster_path: z.string().nullable(),
  backdrop_path: z.string().nullable(),
});

// scripts/score_all_movies.py output: one entry per movie, keyed by tmdb_id.
// out_of_sample = fold-clean CV prediction (the model never saw the movie's
// year); in_sample = final all-data model; no_actuals = no real gross exists.
export const oraclePredictionSchema = z.object({
  predicted_gross: z.number(),
  prediction_kind: z.enum(["out_of_sample", "in_sample", "no_actuals"]),
  actual_gross: z.number().nullable(),
  ape: z.number().nullable(),
});

export const oraclePredictionsSchema = z.record(
  z.string(),
  oraclePredictionSchema,
);

export const backtestYearSchema = z.object({
  year: z.number(),
  n_movies: z.number(),
  baseline_r2_log: z.number(),
  model_r2_log: z.number(),
  gain_r2_log: z.number(),
  baseline_spearman: z.number(),
  model_spearman: z.number(),
  baseline_r2: z.number(),
  model_r2: z.number(),
  gain_r2: z.number(),
  model_rmsle: z.number(),
  model_median_ape: z.number(),
});

export const modelInfoSchema = z
  .object({
    model_id: z.string().nullish(),
    version: z.union([z.string(), z.number()]).nullish(),
    status: z.string().nullish(),
    created_at: z.string().nullish(),
    metrics: z.record(z.string(), z.unknown()).nullish(),
    loaded: z.boolean().nullish(),
    timestamp: z.string().nullish(),
  })
  .nullable();

export const modelMetaSchema = z.object({
  generated_at: z.string(),
  feature_schema_version: z.string(),
  per_year: z.array(backtestYearSchema),
  model_info: modelInfoSchema,
  prediction_api_url: z.string().nullable(),
});

export type Movie = z.infer<typeof movieSchema>;
export type OraclePrediction = z.infer<typeof oraclePredictionSchema>;
export type OraclePredictions = z.infer<typeof oraclePredictionsSchema>;
export type BacktestYear = z.infer<typeof backtestYearSchema>;
export type ModelMeta = z.infer<typeof modelMetaSchema>;

export interface PredictionResult {
  prediction: number;
  model_id: string;
  model_version: number;
  prediction_interval_heuristic: [number, number] | null;
  timestamp: string;
  processing_time_ms: number;
  mock?: boolean;
}

export function posterUrl(
  path: string | null,
  size: "w185" | "w342" | "w500" | "original" = "w342",
): string | null {
  return path ? `https://image.tmdb.org/t/p/${size}${path}` : null;
}

export function backdropUrl(
  path: string | null,
  size: "w780" | "w1280" | "original" = "w1280",
): string | null {
  return path ? `https://image.tmdb.org/t/p/${size}${path}` : null;
}
