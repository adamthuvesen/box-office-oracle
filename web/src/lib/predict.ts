import { z } from "zod";

import contract from "@/generated/predict-contract.json";

// Bounds, defaults, genres, and IP tiers come from the Python owners through
// scripts/generate_web_contract.py (`make fix` regenerates the file).
const request = contract.predict_request.properties;

/** Mirrors the inference API contract (box_office/inference/app/predictor.py). */
export const predictRequestSchema = z.strictObject({
  budget: z.number().min(request.budget.minimum),
  runtime: z.number().min(request.runtime.minimum).max(request.runtime.maximum),
  genre: z.union([z.string().min(1), z.array(z.string().min(1)).min(1)]),
  release_month: z
    .number()
    .int()
    .min(request.release_month.minimum)
    .max(request.release_month.maximum),
  release_year: z
    .number()
    .int()
    .min(request.release_year.minimum)
    .max(request.release_year.maximum),
  mpaa: z.string().default(request.mpaa.default),
  director: z.string().optional(),
  actors: z.array(z.string()).optional(),
  production_company: z.string().optional(),
  // Pre-release IP/franchise strength; defaults describe an original movie.
  ip_tier: z
    .number()
    .int()
    .min(request.ip_tier.minimum)
    .max(request.ip_tier.maximum)
    .default(request.ip_tier.default),
  prior_franchise_gross: z
    .number()
    .min(request.prior_franchise_gross.minimum)
    .default(request.prior_franchise_gross.default),
  is_franchise_followup: z
    .boolean()
    .default(request.is_franchise_followup.default),
  return_confidence: z.boolean().default(request.return_confidence.default),
});

export type PredictRequest = z.infer<typeof predictRequestSchema>;
/** What the client sends — fields with server-side defaults stay optional. */
export type PredictRequestInput = z.input<typeof predictRequestSchema>;

/** The genres the model's genre vocabulary knows. */
export const GENRES: readonly string[] = contract.genres;

/** Ordinal IP tiers (1 = strongest, 5 = none). */
export const IP_TIERS: readonly { value: number; label: string }[] =
  contract.ip_tiers;

export const MPAA_RATINGS = [
  "G",
  "PG",
  "PG-13",
  "R",
  "NC-17",
  "Not Rated",
] as const;

export const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
] as const;

/** The slice of a catalog movie the comparables strip needs — keeps the RSC→client payload small. */
export interface ComparableMovie {
  tmdb_id: number;
  title: string;
  release_year: number;
  poster_path: string | null;
  worldwide_gross: number;
}
